"""Evaluate one self-describing DF checkpoint in an isolated process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import build_dataset
from model import load_streamcrn
from train_strict_df import validate, validate_clean


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--noisy_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev_seg", type=float, default=6.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_streamcrn(args.ckpt, device)
    development = build_dataset("paired", clean_dir=args.clean_dir, noisy_dir=args.noisy_dir,
                                manifest=args.manifest, cfg=CFG, seg_seconds=args.dev_seg,
                                random_crop=False)
    clean_development = build_dataset("paired", clean_dir=args.clean_dir, noisy_dir=args.clean_dir,
                                      manifest=args.manifest, cfg=CFG, seg_seconds=args.dev_seg,
                                      random_crop=False)
    development_loader = DataLoader(development, batch_size=args.batch, shuffle=False, num_workers=0)
    clean_loader = DataLoader(clean_development, batch_size=args.batch, shuffle=False, num_workers=0)
    report = {
        "checkpoint": {"path": str(args.ckpt.resolve()), "sha256": sha256(args.ckpt),
                       "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
                       "mcfg": checkpoint.get("mcfg")},
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest),
                     "items": len(development)},
        "development": validate(model, development_loader, device),
        "clean": validate_clean(model, clean_loader, device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"checkpoint": report["checkpoint"], "manifest": report["manifest"],
               "development": report["development"], "clean": report["clean"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
