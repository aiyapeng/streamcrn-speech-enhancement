"""
把"单帧流式步"导出为 ONNX，用于 C++/移动端/嵌入式部署。

ONNX 图的输入/输出全部是固定形状张量（无 Python 状态对象）：
  输入 : 当前帧复数谱 + 各编码层时间缓存 + GRU 隐状态
  输出 : 增强帧复数谱 + 更新后的各缓存 + 更新后的 GRU 隐状态
部署端在外部维护这些缓存，循环调用即可实现流式降噪。

用法：
    python export_onnx.py --ckpt ckpt/best.pt --out streamcrn_step.onnx
    python export_onnx.py --out streamcrn_step.onnx   # 无权重也可导出+校验结构
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from model import StreamCRN, StreamState


class ONNXStreamStep(nn.Module):
    """把 streaming_step 包装成"扁平张量进、扁平张量出"的可导出模块。"""

    def __init__(self, model: StreamCRN):
        super().__init__()
        self.model = model.eval()
        self.n_layers = len(model.mcfg.enc_channels)
        self.use_df = model.df_order >= 2

    def forward(self, spec_frame, *rest):
        caches = list(rest[:self.n_layers])
        gru_h = rest[self.n_layers]
        df_buf = rest[self.n_layers + 1] if self.use_df else None
        state = StreamState(enc_caches=caches, gru_h=gru_h, df_buf=df_buf)
        enh, ns = self.model.streaming_step(spec_frame, state)
        outs = (enh, *ns.enc_caches, ns.gru_h)
        if self.use_df:
            outs = outs + (ns.df_buf,)
        return outs


def make_dummy_inputs(model: StreamCRN, device="cpu"):
    st = model.init_state(batch=1, device=device)
    spec_frame = torch.randn(1, 2, model.n_freq, 1, device=device)
    return spec_frame, st.enc_caches, st.gru_h, st.df_buf


def export(model, out_path, device="cpu"):
    wrapper = ONNXStreamStep(model).to(device).eval()
    spec_frame, caches, gru_h, df_buf = make_dummy_inputs(model, device)
    use_df = model.df_order >= 2
    inputs = (spec_frame, *caches, gru_h) + ((df_buf,) if use_df else ())

    in_names = ["spec_frame"] + [f"enc_cache_{i}_in" for i in range(len(caches))] + ["gru_h_in"]
    out_names = ["enh_frame"] + [f"enc_cache_{i}_out" for i in range(len(caches))] + ["gru_h_out"]
    if use_df:
        in_names += ["df_buf_in"]; out_names += ["df_buf_out"]

    torch.onnx.export(
        wrapper, inputs, out_path,
        input_names=in_names, output_names=out_names,
        opset_version=17, do_constant_folding=True,
        dynamo=False,   # 用稳定的 TorchScript 导出器，避免 Pad 算子的版本转换问题
    )
    print(f"已导出 ONNX: {out_path}")
    return inputs, in_names, out_names


def verify(model, out_path, inputs, in_names, out_names, device="cpu"):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(out_path))
    print("ONNX 结构检查通过 [PASS]")

    # PyTorch 参考输出
    wrapper = ONNXStreamStep(model).to(device).eval()
    with torch.no_grad():
        torch_out = wrapper(*inputs)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feed = {n: t.cpu().numpy() for n, t in zip(in_names, inputs)}
    ort_out = sess.run(out_names, feed)

    max_err = max(np.abs(t.cpu().numpy() - o).max() for t, o in zip(torch_out, ort_out))
    print(f"onnxruntime vs PyTorch 最大误差: {max_err:.2e}  ({'一致 [PASS]' if max_err<1e-4 else '偏差偏大 [WARN]'})")

    # 用 ONNX 连续跑几帧，验证状态可循环维护
    feed = {n: t.cpu().numpy() for n, t in zip(in_names, inputs)}
    n_enc = len(in_names) - (3 if "df_buf_in" in in_names else 2)  # spec_frame + enc_caches + gru_h(+df_buf)
    for _ in range(5):
        outs = sess.run(out_names, feed)
        feed["spec_frame"] = np.random.randn(*feed["spec_frame"].shape).astype(np.float32)
        for i in range(n_enc):
            feed[f"enc_cache_{i}_in"] = outs[i + 1]
        feed["gru_h_in"] = outs[1 + n_enc]
        if "df_buf_in" in feed:
            feed["df_buf_in"] = outs[2 + n_enc]
    print("ONNX 逐帧状态循环 5 帧 OK [PASS]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="streamcrn_step.onnx")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.ckpt:
        from model import load_streamcrn
        model, _ = load_streamcrn(args.ckpt, args.device)
        print(f"已加载权重: {args.ckpt}")
    else:
        model = StreamCRN()
        print("未提供权重，导出随机初始化模型（用于验证部署结构）")
    model.eval()

    inputs, in_names, out_names = export(model, args.out, args.device)
    verify(model, args.out, inputs, in_names, out_names, args.device)


if __name__ == "__main__":
    main()
