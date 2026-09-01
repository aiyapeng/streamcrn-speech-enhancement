"""时域 SI-SDR 与压缩复数谱联合损失。"""
import torch
import torch.nn as nn

import stft as S
from config import CFG


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8,
                polarity_weight: float = 0.0) -> torch.Tensor:
    """返回批量负 SI-SDR；可选极性项用于惩罚整体反相。"""
    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    alpha = (torch.sum(est * ref, dim=1, keepdim=True)) / (torch.sum(ref ** 2, dim=1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    ratio = (torch.sum(target ** 2, dim=1) + eps) / (torch.sum(noise ** 2, dim=1) + eps)
    sisdr = 10 * torch.log10(ratio + eps)
    loss = -sisdr.mean()
    if polarity_weight > 0.0:
        cos = torch.sum(est * ref, dim=1) / (
            torch.sqrt(torch.sum(est ** 2, dim=1) + eps) * torch.sqrt(torch.sum(ref ** 2, dim=1) + eps))
        loss = loss + polarity_weight * torch.relu(-cos).mean()   # 反相(cos<0)才罚
    return loss


def _compress(spec: torch.Tensor, c: float, eps: float = 1e-8):
    """复数谱功率律压缩：返回 (压缩后复数 re, im, 压缩后幅度)。spec: [B,F,T] 复数。"""
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + eps)
    comp_mag = mag ** c
    unit_r = spec.real / mag
    unit_i = spec.imag / mag
    return comp_mag * unit_r, comp_mag * unit_i, comp_mag


def compressed_spec_loss(est_spec: torch.Tensor, ref_spec: torch.Tensor,
                         c: float = 0.3) -> torch.Tensor:
    """est_spec/ref_spec: [B,F,T] 复数。幅度 MSE + 复数 MSE。"""
    er, ei, emag = _compress(est_spec, c)
    rr, ri, rmag = _compress(ref_spec, c)
    mag_loss = torch.mean((emag - rmag) ** 2)
    cplx_loss = torch.mean((er - rr) ** 2 + (ei - ri) ** 2)
    return mag_loss + cplx_loss


class SEobjective(nn.Module):
    """组合损失。输入：增强复数谱 + 干净时域参考。"""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.n_fft = cfg.stft.n_fft
        self.hop = cfg.stft.hop_length
        self.win = cfg.stft.win_length
        self.alpha = cfg.train.alpha_sisdr
        self.beta = cfg.train.beta_spec
        self.c = cfg.train.spec_compress

    def forward(self, enh_spec_ri: torch.Tensor, clean_wav: torch.Tensor):
        """enh_spec_ri: [B,2,F,T]；clean_wav: [B,L]。"""
        enh_spec = torch.complex(enh_spec_ri[:, 0], enh_spec_ri[:, 1])   # [B,F,T]
        enh_wav = S.istft(enh_spec, self.n_fft, self.hop, self.win)
        # 干净谱（作为频域目标）
        clean_spec = S.stft(clean_wav, self.n_fft, self.hop, self.win)
        # 对齐长度
        L = min(enh_wav.shape[1], clean_wav.shape[1])
        Tc = min(enh_spec.shape[-1], clean_spec.shape[-1])
        l_sisdr = si_sdr_loss(enh_wav[:, :L], clean_wav[:, :L], polarity_weight=1.0)
        l_spec = compressed_spec_loss(enh_spec[..., :Tc], clean_spec[..., :Tc], self.c)
        loss = self.alpha * l_sisdr + self.beta * l_spec
        return loss, {"sisdr": (-l_sisdr).item(), "spec": l_spec.item()}
