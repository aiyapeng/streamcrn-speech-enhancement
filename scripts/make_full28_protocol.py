"""Create deterministic speaker-disjoint fit/development manifests without test access."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def write_manifest(path: Path, names: list[str]) -> dict:
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    return {"path": str(path.resolve()), "rows": len(names), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev_speakers", default="p226,p287")
    parser.add_argument("--dev_select", type=int, default=256)
    parser.add_argument("--seed", default="streamcrn-vb-full28-df-v1-20260823")
    args = parser.parse_args()
    root, out = Path(args.data_root).resolve(), Path(args.out).resolve()
    clean_dir, noisy_dir = root / "clean_trainset_28spk_wav", root / "noisy_trainset_28spk_wav"
    clean = {path.name for path in clean_dir.glob("*.wav")}
    noisy = {path.name for path in noisy_dir.glob("*.wav")}
    if clean != noisy or len(clean) != 11_572:
        raise RuntimeError(f"unexpected paired cardinality: clean={len(clean)} noisy={len(noisy)}")
    speakers = sorted({name.split("_", 1)[0] for name in clean})
    dev_speakers = {value.strip() for value in args.dev_speakers.split(",") if value.strip()}
    if len(speakers) != 28 or len(dev_speakers) != 2 or not dev_speakers.issubset(speakers):
        raise RuntimeError(f"invalid 26/2 split: speakers={speakers}, dev={sorted(dev_speakers)}")
    fit_speakers = set(speakers) - dev_speakers
    fit = sorted(name for name in clean if name.split("_", 1)[0] in fit_speakers)
    development = sorted(name for name in clean if name.split("_", 1)[0] in dev_speakers)
    ranked = sorted(
        development,
        key=lambda name: hashlib.sha256(f"{args.seed}|dev-select|{name}".encode()).hexdigest(),
    )
    selection = ranked[: args.dev_select]
    out.mkdir(parents=True, exist_ok=True)
    manifests = {
        "fit": write_manifest(out / "fit.txt", fit),
        "development": write_manifest(out / "development.txt", development),
        "development_select": write_manifest(out / "development_select.txt", selection),
    }
    protocol = {
        "name": "streamcrn_voicebank_full28_df_speaker_disjoint_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "data_root": str(root),
        "all_speakers": speakers,
        "fit_speakers": sorted(fit_speakers),
        "development_speakers": sorted(dev_speakers),
        "speaker_disjoint": not bool(fit_speakers & dev_speakers),
        "official_test_read": False,
        "manifests": manifests,
    }
    path = out / "protocol.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(path), "sha256": sha256(path),
                      "counts": {key: value["rows"] for key, value in manifests.items()},
                      "fit_speakers": len(fit_speakers),
                      "development_speakers": sorted(dev_speakers)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
