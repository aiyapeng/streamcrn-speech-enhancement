"""Create a deterministic noisy sample and run the bundled StreamCRN model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy import signal
import torch


DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

from config import CFG
from metrics import si_sdr
from model import load_streamcrn
from scripts.infer_stream import StreamingDenoiser


def main() -> None:
    clean, sample_rate = sf.read(
        DEMO_DIR / "clean_reference.wav", dtype="float32", always_2d=True
    )
    clean = clean.mean(axis=1)
    if sample_rate != CFG.stft.sample_rate:
        raise ValueError(f"expected {CFG.stft.sample_rate} Hz, got {sample_rate}")
    hop = CFG.stft.hop_length
    clean = clean[: len(clean) - len(clean) % hop]

    rng = np.random.default_rng(20260901)
    white = rng.standard_normal(len(clean)).astype(np.float32)
    colored = signal.lfilter([1.0, 0.65, 0.25], [1.0], white).astype(np.float32)
    time_axis = np.arange(len(clean), dtype=np.float32) / sample_rate
    colored += 0.35 * np.sin(2 * np.pi * 120 * time_axis)
    colored -= colored.mean()
    target_snr_db = 3.0
    noise_scale = np.sqrt(
        np.mean(clean**2) / (np.mean(colored**2) * 10 ** (target_snr_db / 10))
    )
    noisy = np.clip(clean + colored * noise_scale, -0.98, 0.98).astype(np.float32)

    torch.set_num_threads(4)
    model, _ = load_streamcrn(ROOT / "checkpoints" / "streamcrn_df_k5.pt", "cpu")
    denoiser = StreamingDenoiser(model, CFG, "cpu")
    enhanced = denoiser.process_wav(torch.from_numpy(noisy), align=True).numpy()
    aligned_clean = clean[: len(enhanced)]
    aligned_noisy = noisy[: len(enhanced)]

    sf.write(DEMO_DIR / "noisy_input.wav", noisy, sample_rate, subtype="PCM_16")
    sf.write(DEMO_DIR / "enhanced_output.wav", enhanced, sample_rate, subtype="PCM_16")

    input_score = si_sdr(aligned_noisy, aligned_clean)
    output_score = si_sdr(enhanced, aligned_clean)
    metrics = {
        "sample_rate": sample_rate,
        "target_input_snr_db": target_snr_db,
        "input_si_sdr_db": input_score,
        "output_si_sdr_db": output_score,
        "si_sdr_improvement_db": output_score - input_score,
        "algorithmic_latency_ms": CFG.stft.n_fft / sample_rate * 1000.0,
        "finite_output": bool(np.isfinite(enhanced).all()),
    }
    (DEMO_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
