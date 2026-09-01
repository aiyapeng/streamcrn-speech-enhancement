"""复数域因果 StreamCRN 与 Deep Filtering 实现。"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, STFTConfig


class CausalNorm2d(nn.Module):
    """按帧在通道和频率维归一化，并施加逐通道仿射变换。"""

    def __init__(self, num_ch: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_ch))
        self.beta = nn.Parameter(torch.zeros(num_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, F, T]
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


def load_streamcrn(ckpt_path, device="cpu"):
    """按 checkpoint 内保存的模型配置(mcfg)重建 StreamCRN 并加载权重。

    这样 lite/quality(--big) 等不同尺寸的权重都能被评测/推理/导出脚本正确加载，
    无需手工指定结构。返回 (eval 模式的 model, checkpoint dict)。
    """
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = ck.get("mcfg")
    if mcfg:
        from config import CFG
        CFG.model.enc_channels = list(mcfg["enc_channels"])
        CFG.model.gru_hidden = int(mcfg["gru_hidden"])
        CFG.model.gru_layers = int(mcfg["gru_layers"])
        if "df_order" in mcfg: CFG.model.df_order = int(mcfg["df_order"])
    model = StreamCRN().to(device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    return model, ck


def _conv_freq_out(f_in: int, kernel: int, stride: int, pad: int) -> int:
    return (f_in + 2 * pad - kernel) // stride + 1


def F_pad_time(x: torch.Tensor, n: int) -> torch.Tensor:
    """在最后一维(时间)前面补 n 帧零：[...,T] -> [...,T+n]（用于 Deep Filtering 取过去帧）。"""
    if n <= 0:
        return x
    return F.pad(x, (n, 0))


class CausalEncBlock(nn.Module):
    """因果编码块：频率维下采样卷积 + BN + PReLU。时间维因果（只看过去 kt-1 帧）。"""

    def __init__(self, in_ch, out_ch, freq_kernel, freq_stride, freq_pad, time_kernel):
        super().__init__()
        self.kt = time_kernel
        self.freq_pad = freq_pad
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(freq_kernel, time_kernel),
            stride=(freq_stride, 1),
            padding=0,
        )
        self.bn = CausalNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        # x: [B, C, F, T]。时间维左 pad (kt-1)，频率维对称 pad。
        x = F.pad(x, (self.kt - 1, 0, self.freq_pad, self.freq_pad))
        x = self.conv(x)
        x = self.act(self.bn(x))
        return x

    @torch.no_grad()
    def stream(self, x_frame, cache):
        """x_frame: [B, C, F, 1]；cache: [B, C, F, kt-1] -> (y[B,C',F',1], new_cache)。"""
        xin = torch.cat([cache, x_frame], dim=-1)          # [B,C,F,kt]
        new_cache = xin[..., -(self.kt - 1):] if self.kt > 1 else cache
        xin = F.pad(xin, (0, 0, self.freq_pad, self.freq_pad))  # 仅频率维 pad
        y = self.conv(xin)                                  # 时间维恰好卷成 1 帧
        y = self.act(self.bn(y))
        return y, new_cache


class FreqUpBlock(nn.Module):
    """解码块：仅在频率维上采样的转置卷积（时间核=1，逐帧）+ BN + PReLU。"""

    def __init__(self, in_ch, out_ch, freq_kernel, freq_stride, freq_pad,
                 f_in, f_out, last=False):
        super().__init__()
        out_pad = f_out - ((f_in - 1) * freq_stride - 2 * freq_pad + freq_kernel)
        assert 0 <= out_pad < freq_stride, f"频率维尺寸无法精确还原: {f_in}->{f_out}, out_pad={out_pad}"
        self.last = last
        self.convt = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=(freq_kernel, 1),
            stride=(freq_stride, 1),
            padding=(freq_pad, 0),
            output_padding=(out_pad, 0),
        )
        if not last:
            self.bn = CausalNorm2d(out_ch)
            self.act = nn.PReLU(out_ch)

    def forward(self, x):
        y = self.convt(x)
        if not self.last:
            y = self.act(self.bn(y))
        return y

    @torch.no_grad()
    def stream(self, x_frame):
        # 时间核=1 => 逐帧，无需缓存
        return self.forward(x_frame)


@dataclass
class StreamState:
    enc_caches: List[torch.Tensor] = field(default_factory=list)
    gru_h: Optional[torch.Tensor] = None
    df_buf: Optional[torch.Tensor] = None      # Deep Filtering: 过去 K-1 帧噪声谱 [B,2,F,K-1]


class StreamCRN(nn.Module):
    def __init__(self, mcfg: ModelConfig = None, scfg: STFTConfig = None):
        super().__init__()
        from config import CFG                     # 调用时读取全局配置，使 --big / load_streamcrn 的覆盖生效
        mcfg = mcfg if mcfg is not None else CFG.model
        scfg = scfg if scfg is not None else CFG.stft
        self.mcfg = mcfg
        self.n_freq = scfg.n_freq
        self.kt = mcfg.time_kernel
        self.bounded = mcfg.bounded_mask
        self.df_order = int(getattr(mcfg, "df_order", 0))
        self.out_ch = 2 if self.df_order < 2 else 2 * self.df_order  # DF: 每点 K 个复数抽头

        chs = mcfg.enc_channels
        fk, fs, fp = mcfg.freq_kernel, mcfg.freq_stride, mcfg.freq_pad

        # 编码器 + 记录各层频率尺寸
        self.enc = nn.ModuleList()
        freqs = [self.n_freq]
        in_ch = 2
        for oc in chs:
            self.enc.append(CausalEncBlock(in_ch, oc, fk, fs, fp, self.kt))
            freqs.append(_conv_freq_out(freqs[-1], fk, fs, fp))
            in_ch = oc
        self.enc_freqs = freqs            # [257,129,65,33,17,9]
        self.bottleneck_freq = freqs[-1]
        feat_dim = chs[-1] * self.bottleneck_freq

        # 瓶颈 GRU（时间建模）
        self.gru = nn.GRU(feat_dim, mcfg.gru_hidden, mcfg.gru_layers, batch_first=True)
        self.gru_proj = nn.Linear(mcfg.gru_hidden, feat_dim)

        # 解码器（频率维上采样 + 跳连）
        self.dec = nn.ModuleList()
        rev_ch = chs[::-1]                 # [64,64,48,32,16]
        rev_f = freqs[::-1]               # [9,17,33,65,129,257]
        dec_out = rev_ch[1:] + [self.out_ch]   # 末层输出 2(掩码) 或 2K(DF)
        for i in range(len(chs)):
            in_c = rev_ch[i] * 2           # 跳连拼接 -> 通道翻倍
            out_c = dec_out[i]
            last = (i == len(chs) - 1)
            self.dec.append(FreqUpBlock(in_c, out_c, fk, fs, fp,
                                        f_in=rev_f[i], f_out=rev_f[i + 1], last=last))

        # 以近似恒等变换初始化输出层
        last_convt = self.dec[-1].convt
        with torch.no_grad():
            last_convt.weight.mul_(0.1)
            if last_convt.bias is not None:
                last_convt.bias.zero_()
                if self.df_order < 2:
                    last_convt.bias[0] = 3.0        # 掩码：实部通道 tanh(3)=0.995≈1
                else:
                    last_convt.bias[0] = 1.0        # DF：第0个抽头(当前帧)实部=1、其余=0 => 恒等

    # ---------- 掩码 ----------
    def _apply_mask(self, mask, spec_ri):
        mr, mi = mask[:, 0], mask[:, 1]
        if self.bounded:
            mag_raw = torch.sqrt(mr ** 2 + mi ** 2 + 1e-12)
            mag = torch.tanh(mag_raw)
            mr = mag * mr / (mag_raw + 1e-12)
            mi = mag * mi / (mag_raw + 1e-12)
        xr, xi = spec_ri[:, 0], spec_ri[:, 1]
        er = mr * xr - mi * xi
        ei = mr * xi + mi * xr
        return torch.stack([er, ei], dim=1)   # [B,2,F,T]

    # ---------- Deep Filtering（离线）----------
    def _apply_df(self, coef, spec_ri):
        """coef: [B, 2K, F, T]（前 K 通道=实抽头，后 K=虚抽头）；spec_ri: [B,2,F,T]。

        enh[f,t] = Σ_{k=0}^{K-1} (cr_k+i·ci_k)[f,t] · noisy[f, t-k]   —— 严格因果的复数 FIR。
        """
        K = self.df_order
        B, _, F, T = spec_ri.shape
        cr = coef[:, :K]                          # [B,K,F,T]
        ci = coef[:, K:]                          # [B,K,F,T]
        xr, xi = spec_ri[:, 0], spec_ri[:, 1]     # [B,F,T]
        # 时间前向 pad K-1 帧零，构造 t-k 的取值
        xr_p = F_pad_time(xr, K - 1)              # [B,F,T+K-1]
        xi_p = F_pad_time(xi, K - 1)
        er = xr.new_zeros(B, F, T)
        ei = xr.new_zeros(B, F, T)
        for k in range(K):
            # t-k 对应 xr_p[..., (K-1-k) : (K-1-k)+T]
            s = K - 1 - k
            nr = xr_p[..., s:s + T]
            ni = xi_p[..., s:s + T]
            er = er + cr[:, k] * nr - ci[:, k] * ni
            ei = ei + cr[:, k] * ni + ci[:, k] * nr
        return torch.stack([er, ei], dim=1)       # [B,2,F,T]

    def _finalize(self, out, spec_ri):
        return self._apply_df(out, spec_ri) if self.df_order >= 2 else self._apply_mask(out, spec_ri)

    # ---------- 离线 forward ----------
    def forward(self, spec_ri):
        """spec_ri: [B, 2, F, T] -> 增强后的 [B, 2, F, T]。"""
        B = spec_ri.shape[0]
        skips = []
        x = spec_ri
        for blk in self.enc:
            x = blk(x)
            skips.append(x)                     # [B,C,F_l,T]
        # 瓶颈
        Bc, C, Fb, T = x.shape
        g_in = x.permute(0, 3, 1, 2).reshape(B, T, C * Fb)   # [B,T,C*Fb]
        g_out, _ = self.gru(g_in)
        g_out = self.gru_proj(g_out)                          # [B,T,C*Fb]
        x = g_out.reshape(B, T, C, Fb).permute(0, 2, 3, 1)    # [B,C,Fb,T]
        # 解码 + 跳连
        for i, blk in enumerate(self.dec):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = blk(x)
        out = x                                               # [B,2或2K,F,T]
        return self._finalize(out, spec_ri)

    # ---------- 流式 ----------
    def init_state(self, batch=1, device="cpu") -> StreamState:
        caches = []
        in_ch = 2
        for i, oc in enumerate(self.mcfg.enc_channels):
            f = self.enc_freqs[i]
            caches.append(torch.zeros(batch, in_ch, f, self.kt - 1, device=device))
            in_ch = oc
        h = torch.zeros(self.mcfg.gru_layers, batch, self.mcfg.gru_hidden, device=device)
        df_buf = None
        if self.df_order >= 2:
            df_buf = torch.zeros(batch, 2, self.n_freq, self.df_order - 1, device=device)
        return StreamState(enc_caches=caches, gru_h=h, df_buf=df_buf)

    @torch.no_grad()
    def streaming_step(self, spec_frame_ri, state: StreamState):
        """spec_frame_ri: [B, 2, F, 1] -> (enh_frame[B,2,F,1], new_state)。"""
        B = spec_frame_ri.shape[0]
        x = spec_frame_ri
        skips = []
        new_caches = []
        for i, blk in enumerate(self.enc):
            x, nc = blk.stream(x, state.enc_caches[i])
            skips.append(x)
            new_caches.append(nc)
        # 瓶颈（单帧）
        C, Fb = x.shape[1], x.shape[2]
        g_in = x.permute(0, 3, 1, 2).reshape(B, 1, C * Fb)
        g_out, h_new = self.gru(g_in, state.gru_h)
        g_out = self.gru_proj(g_out)
        x = g_out.reshape(B, 1, C, Fb).permute(0, 2, 3, 1)
        for i, blk in enumerate(self.dec):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = blk.stream(x)
        # 末端：掩码 或 Deep Filtering
        if self.df_order < 2:
            enh = self._apply_mask(x, spec_frame_ri)
            return enh, StreamState(enc_caches=new_caches, gru_h=h_new)
        # Deep Filtering 逐帧：用 [过去K-1帧 | 当前帧] 做复数 FIR
        K = self.df_order
        window = torch.cat([state.df_buf, spec_frame_ri], dim=-1)   # [B,2,F,K]（时间从旧到新）
        coef = x                                                    # [B,2K,F,1]
        cr = coef[:, :K, :, 0]                                      # [B,K,F]
        ci = coef[:, K:, :, 0]
        wr = window[:, 0]                                           # [B,F,K]
        wi = window[:, 1]
        er = spec_frame_ri.new_zeros(B, self.n_freq)
        ei = spec_frame_ri.new_zeros(B, self.n_freq)
        for k in range(K):
            # 抽头 k 乘 frame t-k：窗口里 t-k 的下标是 (K-1-k)
            idx = K - 1 - k
            nr = wr[..., idx]; ni = wi[..., idx]
            er = er + cr[:, k] * nr - ci[:, k] * ni
            ei = ei + cr[:, k] * ni + ci[:, k] * nr
        enh = torch.stack([er, ei], dim=1).unsqueeze(-1)           # [B,2,F,1]
        new_df_buf = window[..., -(K - 1):]                        # 保留最近 K-1 帧
        return enh, StreamState(enc_caches=new_caches, gru_h=h_new, df_buf=new_df_buf)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = StreamCRN()
    print("参数量: {:,}".format(m.num_params()))
    print("编码器频率尺寸:", m.enc_freqs)
    x = torch.randn(2, 2, m.n_freq, 50)
    y = m(x)
    print("forward:", tuple(x.shape), "->", tuple(y.shape))
