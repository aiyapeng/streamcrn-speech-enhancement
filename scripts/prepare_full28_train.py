"""Verify and safely install only the official VoiceBank-DEMAND train archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ARCHIVES = {
    "clean_trainset_28spk_wav.zip": {
        "bytes": 2_486_057_279,
        "md5": "d2d5a45ec32f8fcbf201bde0447e20ba",
        "directory": "clean_trainset_28spk_wav",
    },
    "noisy_trainset_28spk_wav.zip": {
        "bytes": 2_830_205_201,
        "md5": "1fca9e8bafb8cd069f6653c6d92f9e9c",
        "directory": "noisy_trainset_28spk_wav",
    },
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_extract(archive: Path, root: Path) -> None:
    resolved_root = root.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (root / member.filename).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        zipped.extractall(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit: dict[str, object] = {"root": str(root), "archives": {}}
    for name, expected in ARCHIVES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = {"bytes": path.stat().st_size, "md5": digest(path, "md5")}
        if actual["bytes"] != expected["bytes"] or actual["md5"] != expected["md5"]:
            raise RuntimeError(f"archive verification failed: {name}: {actual} != {expected}")
        audit["archives"][name] = actual | {"sha256": digest(path, "sha256")}
        if args.extract:
            safe_extract(path, root)

    clean_dir = root / "clean_trainset_28spk_wav"
    noisy_dir = root / "noisy_trainset_28spk_wav"
    clean = {path.name for path in clean_dir.glob("*.wav")}
    noisy = {path.name for path in noisy_dir.glob("*.wav")}
    if clean != noisy or len(clean) != 11_572:
        raise RuntimeError(
            f"pair/count mismatch: clean={len(clean)} noisy={len(noisy)} "
            f"clean_only={len(clean-noisy)} noisy_only={len(noisy-clean)}"
        )
    speakers = sorted({name.split("_", 1)[0] for name in clean})
    if len(speakers) != 28:
        raise RuntimeError(f"expected 28 speakers, got {len(speakers)}: {speakers}")
    audit.update({"pairs": len(clean), "speakers": speakers, "paired": True,
                  "official_test_read": False})
    destination = root / "train_install_audit.json"
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
