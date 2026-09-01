"""Evaluate speech distortion when already-clean development speech is processed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import CFG
from dataset import _load_wav
from model import load_streamcrn
import stft as S


def sisdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate.astype(np.float64) - float(np.mean(estimate))
    reference = reference.astype(np.float64) - float(np.mean(reference))
    scale = float(np.dot(estimate, reference)) / (float(np.dot(reference, reference)) + 1e-12)
    target = scale * reference
    error = estimate - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(error, error) + 1e-12)))


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "p10": float(np.percentile(array, 10)),
            "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
            "minimum": float(array.min()), "maximum": float(array.max())}


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_streamcrn(args.ckpt, device)
    names = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for index, name in enumerate(names):
        clean = _load_wav(str(args.clean_dir / name), CFG.stft.sample_rate).to(device)
        spectrum = S.stft(clean[None], CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length)
        enhanced_ri = model(S.complex_to_ri(spectrum))
        enhanced = S.istft(S.ri_to_complex(enhanced_ri), CFG.stft.n_fft,
                           CFG.stft.hop_length, CFG.stft.win_length)[0]
        length = min(clean.numel(), enhanced.numel())
        reference = clean[:length].cpu().numpy()
        estimate = enhanced[:length].cpu().numpy()
        relative_l1 = float(np.sum(np.abs(estimate - reference)) / (np.sum(np.abs(reference)) + 1e-12))
        rows.append({"file": name, "relative_l1": relative_l1, "sisdr": sisdr(estimate, reference)})
        if (index + 1) % 100 == 0:
            print(f"evaluated {index + 1}/{len(names)}", flush=True)
    report = {"files": len(rows),
              "relative_l1": aggregate([row["relative_l1"] for row in rows]),
              "sisdr": aggregate([row["sisdr"] for row in rows]), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
