"""支持离线与逐帧处理的因果 STFT/ISTFT 前端。"""
import torch
import torch.nn as nn


def make_window(win_length: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # 分析和合成共用 sqrt-Hann 窗；50% 重叠时满足 COLA
    hann = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    return torch.sqrt(hann.clamp_min(0.0))


def stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """x: [B, L] 时域 -> 复数谱 [B, F, T]（center=False，因果）。"""
    window = make_window(win_length, device=x.device, dtype=x.dtype)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, normalized=False, return_complex=True,
    )


def istft(spec: torch.Tensor, n_fft: int, hop_length: int, win_length: int,
          length: int = None) -> torch.Tensor:
    """复数谱 [B, F, T] -> 时域 [B, L]，sqrt-Hann WOLA 重构（center=False）。

    分析窗与合成窗均为 sqrt-Hann，50% 重叠时直接执行加权叠加。
    """
    assert n_fft == win_length
    assert hop_length * 2 == win_length, "本实现依赖 50% overlap(COLA) 免除包络归一化"
    B, F, T = spec.shape
    win = make_window(win_length, device=spec.device, dtype=spec.real.dtype)
    frames = torch.fft.irfft(spec, n=n_fft, dim=1)          # [B, n_fft, T]
    frames = frames * win.view(1, -1, 1)                    # 合成窗
    out_len = (T - 1) * hop_length + n_fft
    wav = torch.nn.functional.fold(
        frames, output_size=(out_len, 1),
        kernel_size=(n_fft, 1), stride=(hop_length, 1),
    ).reshape(B, out_len)                                    # OLA 求和（无除法）
    if length is not None:
        wav = wav[:, :length]
    return wav


def complex_to_ri(spec: torch.Tensor) -> torch.Tensor:
    """将复数谱转换为实部/虚部双通道张量 [B, 2, F, T]。"""
    return torch.stack([spec.real, spec.imag], dim=1)


def ri_to_complex(ri: torch.Tensor) -> torch.Tensor:
    """[B, 2, F, T] -> 复数谱 [B, F, T]。"""
    return torch.complex(ri[:, 0], ri[:, 1])


class StreamingSTFT:
    """逐帧 STFT。每 push 一个 hop 长度的样本块，产出一帧复数谱 [B, F]。"""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, batch: int = 1, device="cpu"):
        assert n_fft == win_length, "本实现假定 n_fft == win_length"
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = make_window(win_length, device=device)
        self.buf = torch.zeros(batch, win_length, device=device)  # 滑动样本缓存
        self.filled = False

    def reset(self):
        self.buf.zero_()
        self.filled = False

    @torch.no_grad()
    def push(self, hop_samples: torch.Tensor):
        """hop_samples: [B, hop] -> 返回该帧复数谱 [B, F]。"""
        # 左移 hop，填入新样本（滑窗）
        self.buf = torch.roll(self.buf, shifts=-self.hop, dims=1)
        self.buf[:, -self.hop:] = hop_samples
        frame = self.buf * self.win
        spec = torch.fft.rfft(frame, n=self.n_fft, dim=1)  # [B, F]
        return spec


class StreamingISTFT:
    """逐帧 overlap-add ISTFT（sqrt-Hann WOLA，无除法）。每 push 一帧复数谱，产出 hop 样本。

    因分析/合成同为 sqrt-Hann，50% overlap 下 OLA 恒为 1.0，直接求和即可，
    与离线 istft() 数学一致，且不会在频谱被修改后产生边界尖峰。
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, batch: int = 1, device="cpu"):
        assert n_fft == win_length
        assert hop_length * 2 == win_length
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = make_window(win_length, device=device)
        self.ola = torch.zeros(batch, win_length, device=device)      # 信号累加

    def reset(self):
        self.ola.zero_()

    @torch.no_grad()
    def push(self, spec_frame: torch.Tensor):
        """spec_frame: [B, F] -> 返回 [B, hop] 时域样本。"""
        frame = torch.fft.irfft(spec_frame, n=self.n_fft, dim=1)      # [B, win]
        self.ola += frame * self.win
        out = self.ola[:, :self.hop].clone()                         # 无除法（COLA）
        self.ola = torch.roll(self.ola, shifts=-self.hop, dims=1)
        self.ola[:, -self.hop:] = 0.0
        return out
