"""逐帧语音增强与实时率测试入口。"""
import argparse
import time

import numpy as np
import torch

from config import CFG
import stft as S
from model import StreamCRN, load_streamcrn


class StreamingDenoiser:
    """把 StreamingSTFT -> model.streaming_step -> StreamingISTFT 串起来的实时降噪器。"""

    def __init__(self, model: StreamCRN, cfg=CFG, device="cpu"):
        self.model = model.eval().to(device)
        self.cfg = cfg
        self.device = device
        self.n_fft = cfg.stft.n_fft
        self.hop = cfg.stft.hop_length
        self.win = cfg.stft.win_length
        self.reset()

    def reset(self):
        self.sstft = S.StreamingSTFT(self.n_fft, self.hop, self.win, batch=1, device=self.device)
        self.sistft = S.StreamingISTFT(self.n_fft, self.hop, self.win, batch=1, device=self.device)
        self.state = self.model.init_state(batch=1, device=self.device)
        self._primed = 0
        self._n_prime = self.win // self.hop   # 填满第一个窗所需的 hop 数
        # 预热阶段输出静音块，文件评测时按声明时延对齐
        self.output_alignment_samples = (self._n_prime - 1) * self.hop
        self.delay = self.output_alignment_samples

    @torch.no_grad()
    def process_hop(self, hop_samples: torch.Tensor):
        """输入一个 hop 长度的样本块 [hop] -> 输出一个 hop 的增强样本 [hop]（预热期返回静音）。"""
        x = hop_samples.view(1, -1).to(self.device)
        sp = self.sstft.push(x)                                   # [1, F]
        self._primed += 1
        if self._primed < self._n_prime:
            return torch.zeros_like(hop_samples)                  # 首窗未填满
        frame_ri = torch.stack([sp.real, sp.imag], dim=1).unsqueeze(-1)  # [1,2,F,1]
        enh_f, self.state = self.model.streaming_step(frame_ri, self.state)
        enh_sp = torch.complex(enh_f[:, 0, :, 0], enh_f[:, 1, :, 0])     # [1, F]
        out = self.sistft.push(enh_sp)                            # [1, hop]
        return out.view(-1).cpu()

    @torch.no_grad()
    def process_wav(self, wav: torch.Tensor, align: bool = False):
        """整段逐帧增强；align=True 仅用于文件保存/侵入式质量评测。"""
        self.reset()
        L = wav.shape[0]
        n_hops = L // self.hop
        outs = []
        for k in range(n_hops):
            seg = wav[k * self.hop:(k + 1) * self.hop]
            outs.append(self.process_hop(seg))
        output = torch.cat(outs) if outs else torch.zeros(0)
        return output[self.delay:] if align else output


def load_model(ckpt_path=None, device="cpu"):
    if ckpt_path:
        model, _ = load_streamcrn(ckpt_path, device)
        print(f"已加载权重: {ckpt_path}")
    else:
        model = StreamCRN()
        print("未提供权重，使用随机初始化（仅用于测速/流程验证）")
    return model


def measure_rtf(model, cfg=CFG, seconds=20.0, device="cpu", warmup=1.0):
    """测量单核 CPU 上的 RTF。"""
    torch.set_num_threads(1)
    sr = cfg.stft.sample_rate
    hop = cfg.stft.hop_length
    dn = StreamingDenoiser(model, cfg, device)

    # 预热
    wu = torch.randn(int(sr * warmup))
    dn.reset()
    for k in range(wu.shape[0] // hop):
        dn.process_hop(wu[k * hop:(k + 1) * hop])

    x = torch.randn(int(sr * seconds))
    n_hops = x.shape[0] // hop
    dn.reset()
    per_hop = []
    t0 = time.perf_counter()
    for k in range(n_hops):
        h0 = time.perf_counter()
        dn.process_hop(x[k * hop:(k + 1) * hop])
        per_hop.append(time.perf_counter() - h0)
    elapsed = time.perf_counter() - t0

    audio_dur = n_hops * hop / sr
    rtf = elapsed / audio_dur
    per_hop = np.array(per_hop) * 1000  # ms
    hop_budget_ms = hop / sr * 1000
    return {
        "RTF": rtf,
        "audio_sec": audio_dur,
        "compute_sec": elapsed,
        "hop_budget_ms": hop_budget_ms,
        "per_hop_ms_mean": float(per_hop.mean()),
        "per_hop_ms_p95": float(np.percentile(per_hop, 95)),
        "per_hop_ms_max": float(per_hop.max()),
        "algorithmic_latency_ms": cfg.stft.n_fft / sr * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="out", default="enhanced.wav")
    ap.add_argument("--rtf", action="store_true", help="测量实时率")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    print(f"参数量: {model.num_params():,}")

    if args.rtf or args.inp is None:
        r = measure_rtf(model, CFG, seconds=args.seconds, device=args.device)
        print("\n=== 实时率测量（单核 CPU）===")
        print(f"音频时长      : {r['audio_sec']:.1f} s")
        print(f"计算耗时      : {r['compute_sec']:.2f} s")
        print(f"RTF           : {r['RTF']:.4f}   ({'实时 [PASS]' if r['RTF']<1 else '不达标 [FAIL]'})")
        print(f"每帧预算      : {r['hop_budget_ms']:.2f} ms (帧移 {CFG.stft.hop_length} @ {CFG.stft.sample_rate}Hz)")
        print(f"每帧耗时 mean : {r['per_hop_ms_mean']:.3f} ms")
        print(f"每帧耗时 p95  : {r['per_hop_ms_p95']:.3f} ms")
        print(f"每帧耗时 max  : {r['per_hop_ms_max']:.3f} ms")
        print(f"算法时延      : {r['algorithmic_latency_ms']:.1f} ms (= 一个窗长)")

    if args.inp is not None:
        import soundfile as sf
        data, sr = sf.read(args.inp, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != CFG.stft.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, CFG.stft.sample_rate)
        wav = wav.squeeze(0)
        dn = StreamingDenoiser(model, CFG, args.device)
        enh = dn.process_wav(wav, align=True)
        sf.write(args.out, enh.numpy(), CFG.stft.sample_rate)
        print(f"\n已输出降噪结果: {args.out}  ({enh.shape[0]/CFG.stft.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
