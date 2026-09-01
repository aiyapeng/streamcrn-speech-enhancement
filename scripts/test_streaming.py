"""STFT 重构、流式一致性和因果性测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import CFG
import stft as S
from model import StreamCRN


def test_stft_roundtrip():
    n_fft, hop, win = CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length
    x = torch.randn(1, hop * 40 + win)
    spec = S.stft(x, n_fft, hop, win)
    rec = S.istft(spec, n_fft, hop, win)
    Lr = rec.shape[1]
    err = (rec[:, win:Lr - win] - x[:, win:Lr - win]).abs().max().item()
    assert err < 1e-4, err
    print(f"[1] STFT<->ISTFT 完美重构 OK (err={err:.2e})")


def test_model_streaming_equivalence():
    torch.manual_seed(0)
    m = StreamCRN().eval()          # 必须 eval：BN 用 running stats，逐帧一致
    F, T = m.n_freq, 37
    spec = torch.randn(1, 2, F, T)

    # 离线
    with torch.no_grad():
        y_off = m(spec)

    # 流式逐帧
    st = m.init_state(batch=1)
    outs = []
    with torch.no_grad():
        for t in range(T):
            frame = spec[..., t:t + 1]
            enh, st = m.streaming_step(frame, st)
            outs.append(enh)
    y_str = torch.cat(outs, dim=-1)

    err = (y_off - y_str).abs().max().item()
    assert err < 1e-4, f"流式与离线不一致: {err}"
    print(f"[2] 模型 离线 vs 流式 等价 OK (err={err:.2e})")


def test_causality():
    """验证未来帧扰动不改变历史帧输出。"""
    torch.manual_seed(1)
    m = StreamCRN()
    F, T, k = m.n_freq, 30, 15
    spec = torch.randn(1, 2, F, T)
    spec2 = spec.clone()
    spec2[..., k:] += 3.0 * torch.randn_like(spec2[..., k:])  # 扰动未来
    for mode in ["train", "eval"]:
        getattr(m, mode)()
        with torch.no_grad():
            y1 = m(spec); y2 = m(spec2)
        past_diff = (y1[..., :k] - y2[..., :k]).abs().max().item()
        future_diff = (y1[..., k:] - y2[..., k:]).abs().max().item()
        assert past_diff < 1e-5, f"[{mode}] 违反因果性! 过去输出被未来影响: {past_diff}"
        assert future_diff > 1e-3, f"[{mode}] 未来输出应随扰动改变（健全性检查）"
    print(f"[3] 严格因果性 OK (train & eval 两模式过去帧变化均 < 1e-5)")


def test_end_to_end_stream():
    """时域 -> 流式STFT -> 模型流式 -> 流式ISTFT，与离线整段一致。"""
    torch.manual_seed(2)
    n_fft, hop, win = CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length
    m = StreamCRN().eval()
    L = hop * 50 + win
    x = torch.randn(1, L)

    # 离线
    with torch.no_grad():
        spec = S.stft(x, n_fft, hop, win)
        enh = m(S.complex_to_ri(spec).__mul__(1.0))  # [1,2,F,T]
        enh_c = S.ri_to_complex(enh)
        y_off = S.istft(enh_c, n_fft, hop, win)

    # 流式
    sstft = S.StreamingSTFT(n_fft, hop, win, batch=1)
    sistft = S.StreamingISTFT(n_fft, hop, win, batch=1)
    st = m.init_state(batch=1)
    T = spec.shape[-1]
    n_prime = win // hop
    ys = []
    with torch.no_grad():
        # 先喂满第一个窗
        for k in range(T + n_prime - 1):
            seg = x[:, k * hop:(k + 1) * hop]
            sp = sstft.push(seg)                       # [1,F]
            if k < n_prime - 1:
                continue
            frame_ri = torch.stack([sp.real, sp.imag], dim=1).unsqueeze(-1)  # [1,2,F,1]
            enh_f, st = m.streaming_step(frame_ri, st)
            enh_sp = torch.complex(enh_f[:, 0, :, 0], enh_f[:, 1, :, 0])     # [1,F]
            ys.append(sistft.push(enh_sp))
    y_str = torch.cat(ys, dim=1)
    Ls = min(y_str.shape[1], y_off.shape[1])
    err = (y_str[:, win:Ls - win] - y_off[:, win:Ls - win]).abs().max().item()
    assert err < 1e-3, f"端到端流式与离线不一致: {err}"
    print(f"[4] 端到端(时域->时域) 离线 vs 流式 等价 OK (err={err:.2e})")


def test_denoiser_delay_and_alignment():
    """验证文件级流式路径的时延补偿及离线一致性。"""
    from scripts import infer_stream as I
    torch.manual_seed(3)
    n_fft, hop, win = CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length
    m = StreamCRN().eval()
    dn = I.StreamingDenoiser(m, CFG, "cpu")

    # (1) 脉冲法测延迟：过一个单位冲激，输出能量重心相对输入的位移应等于 dn.delay（在一个 hop 容差内）
    L = hop * 40
    imp = torch.zeros(L); imp[hop * 5] = 1.0
    y_imp = dn.process_wav(imp, align=False)
    in_peak = hop * 5
    out_peak = int(torch.argmax(y_imp.abs()).item())
    measured_delay = out_peak - in_peak
    assert abs(measured_delay - dn.delay) <= hop, \
        f"脉冲实测延迟 {measured_delay} 与声明 self.delay {dn.delay} 不符"

    # (2) 对齐后 process_wav 应与离线 istft 在稳态一致
    x = torch.randn(L)
    with torch.no_grad():
        ri = S.complex_to_ri(S.stft(x.unsqueeze(0), n_fft, hop, win))
        y_off = S.istft(S.ri_to_complex(m(ri)), n_fft, hop, win)[0]
    y_str = dn.process_wav(x, align=True)      # 已补偿延迟
    Ls = min(y_str.numel(), y_off.numel())
    err = (y_str[win:Ls - win] - y_off[win:Ls - win]).abs().max().item()
    assert err < 1e-3, f"对齐后 process_wav 与离线不一致: {err}"
    print(f"[5] process_wav 真实路径: 延迟={dn.delay} 样本, 脉冲实测={measured_delay}, 对齐后与离线一致 (err={err:.2e})")


def test_deep_filtering():
    """Deep Filtering(K>=2) 模式：严格因果 + 逐帧流式与离线数值一致。"""
    from config import CFG
    old = CFG.model.df_order
    CFG.model.df_order = 5
    try:
        torch.manual_seed(4)
        m = StreamCRN().eval()
        F, T, k = m.n_freq, 30, 15
        spec = torch.randn(1, 2, F, T)
        # 因果性
        sp2 = spec.clone(); sp2[..., k:] += 3.0 * torch.randn_like(sp2[..., k:])
        with torch.no_grad():
            y1 = m(spec); y2 = m(sp2)
        past = (y1[..., :k] - y2[..., :k]).abs().max().item()
        assert past < 1e-5, f"DF 违反因果性: {past}"
        # 离线 vs 流式
        st = m.init_state(batch=1); ys = []
        with torch.no_grad():
            for t in range(T):
                ef, st = m.streaming_step(spec[..., t:t + 1], st); ys.append(ef)
        y_str = torch.cat(ys, dim=-1)
        err = (y1 - y_str).abs().max().item()
        assert err < 1e-3, f"DF 离线 vs 流式 不一致: {err}"
        print(f"[6] Deep Filtering(K=5): 严格因果(过去帧{past:.1e}) + 离线=流式(err={err:.1e}) OK")
    finally:
        CFG.model.df_order = old


if __name__ == "__main__":
    test_stft_roundtrip()
    test_model_streaming_equivalence()
    test_causality()
    test_end_to_end_stream()
    test_denoiser_delay_and_alignment()
    test_deep_filtering()
    print("\n全部通过 [PASS]")
