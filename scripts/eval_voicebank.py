"""Evaluate a locked VoiceBank manifest with offline or true streaming inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from config import CFG
from dataset import _load_wav
from infer_stream import StreamingDenoiser
from metrics import evaluate_pair
from model import StreamCRN, load_streamcrn
import stft as S


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def enhance_offline(model, noisy):
    spectrum = S.stft(noisy.unsqueeze(0), CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length)
    enhanced_ri = model(S.complex_to_ri(spectrum))
    return S.istft(torch.complex(enhanced_ri[:, 0], enhanced_ri[:, 1]),
                   CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length)[0]


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.nanmean(array)), "p10": float(np.nanpercentile(array, 10)),
            "median": float(np.nanmedian(array)), "p90": float(np.nanpercentile(array, 90)),
            "minimum": float(np.nanmin(array)), "maximum": float(np.nanmax(array))}


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--consume_locked_test", action="store_true",
                        help="required acknowledgement when the manifest is the official test set")
    parser.add_argument("--allow_reused_official_test", action="store_true",
                        help="explicitly label and permit a previously consumed benchmark; never blind")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    locked_test = "test_locked" in manifest.name
    consumption_marker = manifest.with_name("test_consumed.json")
    prior_consumption = None
    if locked_test and consumption_marker.exists() and not args.allow_reused_official_test:
        raise RuntimeError(f"official test was already consumed: {consumption_marker}")
    if locked_test and consumption_marker.exists():
        prior_consumption = json.loads(consumption_marker.read_text(encoding="utf-8"))
    if locked_test and not args.consume_locked_test:
        raise RuntimeError("official test is locked; pass --consume_locked_test only after development selection is frozen")

    names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    clean_dir, noisy_dir = Path(args.clean_dir), Path(args.noisy_dir)
    missing = [name for name in names if not (clean_dir / name).is_file() or not (noisy_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} pairs, examples={missing[:3]}")
    device = torch.device(args.device)
    model, checkpoint = load_streamcrn(args.ckpt, device)
    denoiser = StreamingDenoiser(model, CFG, device) if args.streaming else None

    rows = []
    for index, name in enumerate(names):
        clean = _load_wav(str(clean_dir / name), CFG.stft.sample_rate)
        noisy = _load_wav(str(noisy_dir / name), CFG.stft.sample_rate)
        original_length = min(clean.numel(), noisy.numel())
        clean, noisy = clean[:original_length], noisy[:original_length]
        enhanced = denoiser.process_wav(noisy) if args.streaming else enhance_offline(model, noisy.to(device)).cpu()
        if args.streaming:
            # process_wav preserves the callback timeline and emits silence while the
            # first analysis window fills. Remove only that timestamp offset for
            # intrusive quality metrics; the deployment latency remains unchanged.
            enhanced = enhanced[denoiser.output_alignment_samples:]
        length = min(enhanced.numel(), original_length)
        reference, mixture, estimate = clean[:length].numpy(), noisy[:length].numpy(), enhanced[:length].numpy()
        before, after = evaluate_pair(mixture, reference, CFG.stft.sample_rate), evaluate_pair(estimate, reference, CFG.stft.sample_rate)
        rows.append({"file": name, "noisy": before, "enhanced": after,
                     "improvement": {key: after[key] - before[key] for key in ("pesq", "stoi", "sisdr")}})
        if (index + 1) % 100 == 0:
            print(f"evaluated {index + 1}/{len(names)}")

    aggregate = {}
    for metric in ("pesq", "stoi", "sisdr"):
        aggregate[f"noisy_{metric}"] = stats([row["noisy"][metric] for row in rows])
        aggregate[f"enhanced_{metric}"] = stats([row["enhanced"][metric] for row in rows])
        aggregate[f"improvement_{metric}"] = stats([row["improvement"][metric] for row in rows])
    report = {
        "mode": "streaming" if args.streaming else "offline",
        "files": len(rows),
        "checkpoint": {"path": str(Path(args.ckpt).resolve()), "sha256": sha256(Path(args.ckpt))},
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "official_test_acknowledged": bool(args.consume_locked_test),
        "official_benchmark_reuse": prior_consumption is not None,
        "prior_official_consumption": prior_consumption,
        "streaming_timing": {
            "algorithmic_latency_samples": CFG.stft.win_length if args.streaming else 0,
            "algorithmic_latency_ms": 1000.0 * CFG.stft.win_length / CFG.stft.sample_rate if args.streaming else 0.0,
            "metric_alignment_samples": denoiser.output_alignment_samples if args.streaming else 0,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if locked_test and prior_consumption is None:
        consumption = {
            "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "manifest_sha256": report["manifest"]["sha256"],
            "result_path": str(out.resolve()),
            "result_sha256": sha256(out),
            "mode": report["mode"],
            "files": len(rows),
        }
        consumption_marker.write_text(
            json.dumps(consumption, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif locked_test:
        reuse_audit = {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "reused official benchmark; not blind and not valid for model selection",
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "manifest_sha256": report["manifest"]["sha256"],
            "result_path": str(out.resolve()),
            "result_sha256": sha256(out),
            "prior_consumption": prior_consumption,
        }
        audit_path = out.with_name(out.stem + "_REUSE_AUDIT.json")
        audit_path.write_text(json.dumps(reuse_audit, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"mode": report["mode"], "files": len(rows), "aggregate": aggregate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
