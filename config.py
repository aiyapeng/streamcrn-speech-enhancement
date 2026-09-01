"""StreamCRN 训练与流式推理配置。"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class STFTConfig:
    sample_rate: int = 16000
    n_fft: int = 512          # 32 ms 窗长 -> 算法时延 32 ms
    hop_length: int = 256     # 16 ms 帧移，50% 重叠，Hann 满足 COLA
    win_length: int = 512
    # 不使用中心填充，避免引入未来样本
    center: bool = False

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1   # 257


@dataclass
class ModelConfig:
    # 编码器每层输出通道（输入是复数谱的实/虚两个通道）
    enc_channels: List[int] = field(default_factory=lambda: [16, 32, 48, 64, 64])
    # 频率维卷积核 / 步长 / padding（只在频率维下采样，时间维 stride 恒为 1）
    freq_kernel: int = 5
    freq_stride: int = 2
    freq_pad: int = 2
    # 时间维卷积核：只看"当前帧 + 过去 (time_kernel-1) 帧"，因果
    time_kernel: int = 2
    # 瓶颈 GRU
    gru_hidden: int = 128
    gru_layers: int = 1
    # 是否使用 tanh 限制复数掩码幅度
    bounded_mask: bool = True
    # K<2 使用复数掩码；K>=2 对当前帧及 K-1 帧历史谱执行因果复数 FIR
    df_order: int = 0


@dataclass
class TrainConfig:
    seg_seconds: float = 3.0          # 训练片段长度
    batch_size: int = 16
    lr: float = 8e-4
    weight_decay: float = 0.0
    epochs: int = 120
    grad_clip: float = 5.0
    # 损失权重：L = alpha*(-SI-SDR) + beta*压缩复数谱 + gamma*多分辨率STFT + delta*非对称过抑制惩罚
    alpha_sisdr: float = 1.0
    beta_spec: float = 15.0
    spec_compress: float = 0.3        # 功率律压缩指数（DNS/DeepFilterNet 常用 0.3）
    # 可选正则项，默认关闭
    gamma_mrstft: float = 0.0         # 多分辨率 STFT 幅度损失
    delta_asym: float = 0.0           # 非对称过抑制惩罚
    # 在线混合的 SNR 范围（dB）
    snr_min: float = -5.0
    snr_max: float = 20.0
    num_workers: int = 4
    seed: int = 1234


@dataclass
class Config:
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()
