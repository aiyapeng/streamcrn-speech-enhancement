"""支持断点续训和说话人隔离验证的 StreamCRN 训练入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import build_dataset
from losses import SEobjective
from metrics import evaluate_pair
from model import StreamCRN
import stft as S


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_ri(waveform: torch.Tensor) -> torch.Tensor:
    return S.complex_to_ri(
        S.stft(waveform, CFG.stft.n_fft, CFG.stft.hop_length, CFG.stft.win_length)
    )


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "p10": float(np.nanpercentile(array, 10)),
        "median": float(np.nanmedian(array)),
        "p90": float(np.nanpercentile(array, 90)),
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
    }


def manifest_metadata(path: str) -> dict:
    resolved = os.path.abspath(path)
    digest = hashlib.sha256()
    rows = 0
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            rows += block.count(b"\n")
    return {"path": resolved, "rows": rows, "sha256": digest.hexdigest()}


@torch.no_grad()
def validate(model: StreamCRN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    rows = []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        enhanced_ri = model(to_ri(noisy))
        enhanced = S.istft(S.ri_to_complex(enhanced_ri), CFG.stft.n_fft,
                           CFG.stft.hop_length, CFG.stft.win_length)
        length = min(enhanced.shape[1], clean.shape[1], noisy.shape[1])
        for index in range(enhanced.shape[0]):
            estimate = enhanced[index, :length].float().cpu().numpy()
            reference = clean[index, :length].float().cpu().numpy()
            mixture = noisy[index, :length].float().cpu().numpy()
            after = evaluate_pair(estimate, reference, CFG.stft.sample_rate)
            before = evaluate_pair(mixture, reference, CFG.stft.sample_rate)
            rows.append({key: after[key] - before[key] for key in ("pesq", "stoi", "sisdr")}
                        | {f"enhanced_{key}": after[key] for key in ("pesq", "stoi", "sisdr")}
                        | {f"noisy_{key}": before[key] for key in ("pesq", "stoi", "sisdr")})
    return {key: distribution([row[key] for row in rows]) for key in rows[0]}


@torch.no_grad()
def validate_clean(model: StreamCRN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    relative_l1 = []
    sisdr_values = []
    for clean_input, clean_ref in loader:
        clean_input = clean_input.to(device, non_blocking=True)
        clean_ref = clean_ref.to(device, non_blocking=True)
        enhanced_ri = model(to_ri(clean_input))
        enhanced = S.istft(S.ri_to_complex(enhanced_ri), CFG.stft.n_fft,
                           CFG.stft.hop_length, CFG.stft.win_length)
        length = min(enhanced.shape[1], clean_ref.shape[1])
        estimate, reference = enhanced[:, :length], clean_ref[:, :length]
        relative_l1.extend(
            (torch.sum(torch.abs(estimate - reference), dim=1) /
             torch.sum(torch.abs(reference), dim=1).clamp_min(1e-12)).cpu().tolist()
        )
        estimate_zero = estimate - estimate.mean(dim=1, keepdim=True)
        reference_zero = reference - reference.mean(dim=1, keepdim=True)
        scale = (
            torch.sum(estimate_zero * reference_zero, dim=1, keepdim=True) /
            torch.sum(reference_zero.square(), dim=1, keepdim=True).clamp_min(1e-12)
        )
        target = scale * reference_zero
        ratio = (
            torch.sum(target.square(), dim=1) /
            torch.sum((estimate_zero - target).square(), dim=1).clamp_min(1e-12)
        )
        sisdr_values.extend((10.0 * torch.log10(ratio.clamp_min(1e-12))).cpu().tolist())
    return {"relative_l1": distribution(relative_l1), "sisdr": distribution(sisdr_values)}


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = build_dataset("paired", clean_dir=args.clean_dir, noisy_dir=args.noisy_dir,
                          manifest=args.train_manifest, cfg=CFG, random_crop=True)
    development = build_dataset("paired", clean_dir=args.val_clean, noisy_dir=args.val_noisy,
                                manifest=args.val_manifest, cfg=CFG, seg_seconds=args.dev_seg,
                                random_crop=False)
    clean_development = build_dataset("paired", clean_dir=args.val_clean, noisy_dir=args.val_clean,
                                      manifest=args.val_manifest, cfg=CFG, seg_seconds=args.dev_seg,
                                      random_crop=False)
    common = {"num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    if args.workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True, **common)
    development_loader = DataLoader(development, batch_size=args.batch, shuffle=False, **common)
    clean_loader = DataLoader(clean_development, batch_size=args.batch, shuffle=False, **common)
    return train_loader, development_loader, clean_loader


def atomic_save(payload: dict, path: str) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_config() -> dict:
    return {
        "enc_channels": list(CFG.model.enc_channels),
        "gru_hidden": int(CFG.model.gru_hidden),
        "gru_layers": int(CFG.model.gru_layers),
        "df_order": int(CFG.model.df_order),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--val_clean", required=True)
    parser.add_argument("--val_noisy", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--until_epoch", type=int)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seg", type=float, default=3.0)
    parser.add_argument("--dev_seg", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--skip_validation", action="store_true",
                        help="save train state without PESQ/clean evaluation; evaluate in a separate process")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--big", action="store_true")
    parser.add_argument("--df_order", type=int, default=5)
    parser.add_argument("--min_dev_pesqi", type=float, default=0.0,
                        help="development signal gate; the final PESQ target belongs to an isolated test")
    parser.add_argument("--max_clean_l1", type=float, default=0.08)
    args = parser.parse_args()

    if args.big:
        CFG.model.enc_channels = [24, 48, 64, 96, 96]
        CFG.model.gru_hidden = 256
        CFG.model.gru_layers = 2
    CFG.model.df_order = args.df_order
    CFG.train.seg_seconds = args.seg
    set_seed(CFG.train.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    train_loader, development_loader, clean_loader = build_loaders(args)
    total_steps = args.epochs * len(train_loader)
    if total_steps <= args.warmup_steps:
        raise ValueError("total effective steps must exceed warmup_steps")

    model = StreamCRN().to(device)
    criterion = SEobjective(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.train.weight_decay)

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(1, step + 1) / max(1, args.warmup_steps)
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, global_step, amp_skips = 1, 0, 0
    best_candidate_pesqi, best_accepted_pesqi = -1e9, -1e9

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("mcfg") != model_config():
            raise RuntimeError(f"resume model config mismatch: {checkpoint.get('mcfg')} != {model_config()}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        amp_skips = int(checkpoint.get("amp_skips", 0))
        best_candidate_pesqi = float(checkpoint.get("best_candidate_pesqi", best_candidate_pesqi))
        best_accepted_pesqi = float(checkpoint.get("best_accepted_pesqi", best_accepted_pesqi))
        set_seed(CFG.train.seed + start_epoch)
        print(f"resume epoch={start_epoch} global_step={global_step} lr={optimizer.param_groups[0]['lr']:.8f}", flush=True)

    run_config = {
        "args": vars(args), "config": asdict(CFG), "mcfg": model_config(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "parameters": model.num_params(), "train_batches": len(train_loader),
        "development_items": len(development_loader.dataset), "total_steps": total_steps,
        "manifests": {"fit": manifest_metadata(args.train_manifest),
                      "development_selection": manifest_metadata(args.val_manifest)},
        "acceptance": {"development_pesqi_mean_min": args.min_dev_pesqi,
                       "clean_relative_l1_mean_max": args.max_clean_l1,
                       "improvement_p10_min": 0.0,
                       "final_isolated_test_pesq_mean_min": 2.8},
    }
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)
    print(f"device={device} amp={use_amp} params={model.num_params():,} "
          f"df_order={model.df_order} train_batches={len(train_loader)} dev={len(development_loader.dataset)}",
          flush=True)

    end_epoch = min(args.epochs, args.until_epoch or args.epochs)
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        started, loss_sum = time.time(), 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                enhanced_ri = model(to_ri(noisy))
                loss, _ = criterion(enhanced_ri, clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss epoch={epoch} effective_step={global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = (not use_amp) or scaler.get_scale() >= previous_scale
            if optimizer_ran:
                scheduler.step()
                global_step += 1
            else:
                amp_skips += 1
            loss_sum += float(loss.item())

        should_validate = (not args.skip_validation) and (
            epoch % args.validate_every == 0 or epoch == end_epoch
        )
        development = validate(model, development_loader, device) if should_validate else None
        clean = validate_clean(model, clean_loader, device) if should_validate else None
        elapsed = time.time() - started
        record = {"epoch": epoch, "global_step": global_step, "loss": loss_sum / len(train_loader),
                  "development": development, "clean": clean, "lr": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": elapsed, "amp_skips_total": amp_skips,
                  "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
        with open(os.path.join(args.out, "train_log.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if development is None:
            print(f"[{epoch:03d}/{args.epochs}] step={global_step} loss={record['loss']:.4f} "
                  f"development=skipped lr={record['lr']:.7f} {elapsed:.1f}s", flush=True)
        else:
            pesq = development["enhanced_pesq"]["mean"]
            pesqi = development["pesq"]["mean"]
            print(f"[{epoch:03d}/{args.epochs}] step={global_step} loss={record['loss']:.4f} "
                  f"PESQ={pesq:.3f}({pesqi:+.3f}) STOIi={development['stoi']['mean']:+.4f} "
                  f"SI-SDRi={development['sisdr']['mean']:+.2f}dB "
                  f"cleanL1={clean['relative_l1']['mean']:.3f} lr={record['lr']:.7f} {elapsed:.1f}s", flush=True)
            snapshot = {"model": model.state_dict(), "mcfg": model_config(), "epoch": epoch,
                        "global_step": global_step, "development": development, "clean": clean}
            atomic_save(snapshot, os.path.join(args.out, f"development_epoch_{epoch:03d}.pt"))
            mean_gate = development["stoi"]["mean"] >= 0.0 and development["sisdr"]["mean"] > 0.0
            if mean_gate and pesqi > best_candidate_pesqi:
                best_candidate_pesqi = pesqi
                atomic_save({**snapshot, "selection": "max development PESQi with positive mean STOIi/SI-SDRi"},
                            os.path.join(args.out, "best.pt"))
                print(f"  saved candidate best PESQi={pesqi:+.3f}", flush=True)
            strict_gate = (
                pesqi >= args.min_dev_pesqi and clean["relative_l1"]["mean"] <= args.max_clean_l1 and
                all(development[key]["p10"] >= 0.0 for key in ("pesq", "stoi", "sisdr"))
            )
            if strict_gate and pesqi > best_accepted_pesqi:
                best_accepted_pesqi = pesqi
                atomic_save({**snapshot, "selection": "predeclared strict acceptance gates passed"},
                            os.path.join(args.out, "accepted.pt"))
                print(f"  saved accepted model PESQi={pesqi:+.3f}", flush=True)

        atomic_save({"model": model.state_dict(), "mcfg": model_config(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                     "amp_skips": amp_skips, "best_candidate_pesqi": best_candidate_pesqi,
                     "best_accepted_pesqi": best_accepted_pesqi}, os.path.join(args.out, "last.pt"))
    print(f"stage complete candidate={best_candidate_pesqi:+.3f} accepted={best_accepted_pesqi:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
