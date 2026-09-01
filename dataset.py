"""语音增强的在线混合、成对语音和合成测试数据接口。"""
import os
import glob
import random
import math

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from config import CFG


def _load_wav(path, target_sr):
    # 优先使用 soundfile，读取失败时改用 torchaudio
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [L, C]
        wav = torch.from_numpy(data.T)                              # [C, L]
    except Exception:
        wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)     # 转单声道
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)                    # [L]


def _rand_segment(wav, seg_len):
    L = wav.shape[0]
    if L < seg_len:
        reps = math.ceil(seg_len / max(L, 1))
        wav = wav.repeat(reps)[:seg_len]
    else:
        s = random.randint(0, L - seg_len)
        wav = wav[s:s + seg_len]
    return wav


def _mix_at_snr(clean, noise, snr_db, eps=1e-8):
    cp = clean.pow(2).mean()
    npow = noise.pow(2).mean()
    scale = torch.sqrt(cp / (npow + eps) / (10 ** (snr_db / 10)))
    noisy = clean + scale * noise
    # 防削波归一化
    peak = noisy.abs().max()
    if peak > 0.99:
        g = 0.99 / peak
        noisy, clean = noisy * g, clean * g
    return noisy, clean


class OnTheFlyMixDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, cfg=CFG, length=2000, train=True):
        self.sr = cfg.stft.sample_rate
        self.seg = int(cfg.train.seg_seconds * self.sr)
        self.snr_min, self.snr_max = cfg.train.snr_min, cfg.train.snr_max
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True))
        assert self.clean_files, f"未找到干净语音: {clean_dir}"
        assert self.noise_files, f"未找到噪声: {noise_dir}"
        self.length = length
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean = _load_wav(random.choice(self.clean_files), self.sr)
        noise = _load_wav(random.choice(self.noise_files), self.sr)
        clean = _rand_segment(clean, self.seg)
        noise = _rand_segment(noise, self.seg)
        snr = random.uniform(self.snr_min, self.snr_max)
        noisy, clean = _mix_at_snr(clean, noise, snr)
        return noisy.float(), clean.float()


class PairedDataset(Dataset):
    """VoiceBank-DEMAND 式：clean_dir 与 noisy_dir 下同名 wav。"""

    def __init__(self, clean_dir, noisy_dir, cfg=CFG, seg_seconds=None,
                 random_crop=True, manifest=None):
        self.sr = cfg.stft.sample_rate
        self.seg = int((seg_seconds or cfg.train.seg_seconds) * self.sr)
        self.random_crop = random_crop
        self.clean_dir, self.noisy_dir = clean_dir, noisy_dir
        if manifest:
            with open(manifest, "r", encoding="utf-8") as handle:
                self.names = [line.strip() for line in handle if line.strip()]
        else:
            self.names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.wav")))
        assert self.names, f"未找到成对数据: {clean_dir}"
        missing = [name for name in self.names
                   if not os.path.isfile(os.path.join(clean_dir, name))
                   or not os.path.isfile(os.path.join(noisy_dir, name))]
        if missing:
            raise FileNotFoundError(f"成对数据缺失 {len(missing)} 条，示例: {missing[:3]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        clean = _load_wav(os.path.join(self.clean_dir, name), self.sr)
        noisy = _load_wav(os.path.join(self.noisy_dir, name), self.sr)
        L = min(clean.shape[0], noisy.shape[0])
        clean, noisy = clean[:L], noisy[:L]
        if L > self.seg:
            s = random.randint(0, L - self.seg) if self.random_crop else (L - self.seg) // 2
            clean, noisy = clean[s:s + self.seg], noisy[s:s + self.seg]
        elif L < self.seg:
            pad = self.seg - L
            clean = torch.nn.functional.pad(clean, (0, pad))
            noisy = torch.nn.functional.pad(noisy, (0, pad))
        return noisy.float(), clean.float()


class SyntheticSEDataset(Dataset):
    """生成类语音信号与有色噪声，用于流水线自测。"""

    def __init__(self, cfg=CFG, length=256, train=True, seed=0):
        self.sr = cfg.stft.sample_rate
        self.seg = int(cfg.train.seg_seconds * self.sr)
        self.snr_min, self.snr_max = cfg.train.snr_min, cfg.train.snr_max
        self.length = length
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.length

    def _speech_like(self):
        t = np.arange(self.seg) / self.sr
        f0 = self.rng.uniform(90, 200)                      # 基频
        # 缓慢音高漂移
        f0_t = f0 * (1 + 0.03 * np.sin(2 * np.pi * self.rng.uniform(0.3, 1.5) * t))
        phase = 2 * np.pi * np.cumsum(f0_t) / self.sr
        sig = np.zeros_like(t)
        formants = self.rng.uniform(400, 3000, size=3)
        for k in range(1, 25):
            amp = 1.0 / k
            # 共振峰包络
            fk = k * f0
            env = sum(np.exp(-((fk - fm) ** 2) / (2 * 300 ** 2)) for fm in formants)
            sig += amp * env * np.sin(k * phase)
        # 音节级调幅 + 停顿
        am = 0.5 + 0.5 * np.sin(2 * np.pi * self.rng.uniform(2, 5) * t + self.rng.uniform(0, 6))
        am = np.clip(am - 0.15, 0, None)
        sig = sig * am
        sig = sig / (np.abs(sig).max() + 1e-8) * self.rng.uniform(0.3, 0.8)
        return sig.astype(np.float32)

    def _colored_noise(self):
        white = self.rng.randn(self.seg).astype(np.float32)
        # 简单一阶低通得到有色噪声
        a = self.rng.uniform(0.0, 0.95)
        out = np.zeros_like(white)
        prev = 0.0
        for i in range(len(white)):
            prev = a * prev + (1 - a) * white[i]
            out[i] = prev
        out = out / (np.abs(out).max() + 1e-8)
        return out.astype(np.float32)

    def __getitem__(self, idx):
        clean = torch.from_numpy(self._speech_like())
        noise = torch.from_numpy(self._colored_noise())
        snr = self.rng.uniform(self.snr_min, self.snr_max)
        noisy, clean = _mix_at_snr(clean, noise, float(snr))
        return noisy.float(), clean.float()


def build_dataset(kind, **kw):
    kind = kind.lower()
    if kind == "mix":
        return OnTheFlyMixDataset(**kw)
    if kind == "paired":
        return PairedDataset(**kw)
    if kind == "synthetic":
        return SyntheticSEDataset(**kw)
    raise ValueError(kind)
