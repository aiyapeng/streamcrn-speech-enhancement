"""Write a machine-readable RTF and ONNX consistency audit for a DF checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export_onnx import ONNXStreamStep, make_dummy_inputs
from infer_stream import measure_rtf
from model import load_streamcrn


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    model, checkpoint = load_streamcrn(args.ckpt, "cpu")
    onnx.checker.check_model(onnx.load(str(args.onnx)))
    spec, caches, gru_h, df_buf = make_dummy_inputs(model, "cpu")
    inputs = (spec, *caches, gru_h) + ((df_buf,) if model.df_order >= 2 else ())
    input_names = ["spec_frame"] + [f"enc_cache_{i}_in" for i in range(len(caches))] + ["gru_h_in"]
    output_names = ["enh_frame"] + [f"enc_cache_{i}_out" for i in range(len(caches))] + ["gru_h_out"]
    if model.df_order >= 2:
        input_names.append("df_buf_in")
        output_names.append("df_buf_out")
    wrapper = ONNXStreamStep(model).eval()
    with torch.no_grad():
        torch_outputs = wrapper(*inputs)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feed = {name: tensor.numpy() for name, tensor in zip(input_names, inputs)}
    ort_outputs = session.run(output_names, feed)
    max_error = max(float(np.max(np.abs(a.numpy() - b))) for a, b in zip(torch_outputs, ort_outputs))
    for _ in range(5):
        outputs = session.run(output_names, feed)
        feed["spec_frame"] = np.random.randn(*feed["spec_frame"].shape).astype(np.float32)
        for index in range(len(caches)):
            feed[f"enc_cache_{index}_in"] = outputs[index + 1]
        feed["gru_h_in"] = outputs[len(caches) + 1]
        if "df_buf_in" in feed:
            feed["df_buf_in"] = outputs[len(caches) + 2]
    report = {
        "checkpoint": {"sha256": sha256(args.ckpt), "bytes": args.ckpt.stat().st_size,
                       "epoch": checkpoint.get("epoch"), "mcfg": checkpoint.get("mcfg")},
        "onnx": {"sha256": sha256(args.onnx), "bytes": args.onnx.stat().st_size,
                 "checker": "pass", "max_abs_error": max_error, "state_loop_frames": 5},
        "parameters": model.num_params(), "df_order": model.df_order,
        "rtf_cpu_single_thread": measure_rtf(model, seconds=args.seconds, device="cpu"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
