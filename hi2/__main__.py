# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""``python -m hi2 quantize|verify`` (run from tools/hi2 or with it on PYTHONPATH)."""

import argparse
import json
from pathlib import Path

from .calibrate import SELECTABLE, Config
from .checkpoint import DEFAULT_EXCLUDE, export_dense, verify_checkpoint
from .encode import EFFORT
from .evalq import evaluate
from .pipeline import RunConfig, run


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hi2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="HF checkpoint -> HI2 flex checkpoint")
    q.add_argument("src", type=Path)
    q.add_argument("dst", type=Path)
    q.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="tables, statistics and resume state (default: <dst>.work)",
    )
    q.add_argument(
        "--devices",
        default="cpu",
        help="cpu, cuda:0, or a list such as cuda:0,cuda:1 (HIP on ROCm). "
        "Several devices split calibration sequences and shard each "
        "tensor's rows",
    )
    q.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=".txt or .jsonl calibration text; omit for data-free",
    )
    q.add_argument(
        "--calib-samples",
        type=int,
        default=128,
        help="sequences of --seq-len tokens; 128 x 2048 = 262k tokens",
    )
    q.add_argument("--seq-len", type=int, default=2048)
    q.add_argument(
        "--target-bpw",
        type=float,
        default=3.0,
        help="model-wide bits per weight (global allocation)",
    )
    q.add_argument(
        "--rotation",
        choices=("on", "auto", "off"),
        default="on",
        help="Hadamard flags (default on; pairs with --smooth when calibrated)",
    )
    q.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="smoothing exponent alpha (0 = off); needs --calib",
    )
    q.add_argument(
        "--rounding",
        choices=("gptq", "nearest"),
        default="gptq",
        help="gptq (default) = Hessian-aware error-compensated rounding (spec 8.3), "
        "falls back to nearest without --calib",
    )
    q.add_argument("--damp", type=float, default=0.01, help="Hessian damping for gptq")
    q.add_argument(
        "--sensitivity",
        choices=("none", "probe"),
        default="none",
        help="probe = weight the allocation by each tensor's measured KL cost (spec 8.6)",
    )
    q.add_argument("--sens-seqs", type=int, default=8)
    q.add_argument("--sens-len", type=int, default=512)
    q.add_argument(
        "--sens-floor",
        type=float,
        default=0.05,
        help="lower bound of a tensor weight relative to the median",
    )
    q.add_argument(
        "--sequential",
        action="store_true",
        help="gptq on the quantized prefix (spec 8.4): a second layerwise model pass",
    )
    q.add_argument("--effort", choices=list(EFFORT), default="normal")
    q.add_argument(
        "--bits",
        type=_ints,
        default=[2, 4],
        help="allowed code widths (2,4 default; add 8 for INT8 rows)",
    )
    q.add_argument(
        "--blocks", type=_ints, default=[16, 32, 64], help="allowed block sizes"
    )
    q.add_argument("--include", default=None, help="regex: only these weights")
    q.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex: keep dense")
    q.add_argument("--limit-layers", type=int, default=None)
    q.add_argument(
        "--compute-dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    v = sub.add_parser("verify", help="reload an HI2 checkpoint and compare")
    v.add_argument("src", type=Path)
    v.add_argument("dst", type=Path)
    v.add_argument("--max-tensors", type=int, default=8)
    x = sub.add_parser(
        "export-dense", help="write the dequantised model as a plain fp16 HF dir"
    )
    x.add_argument("src", type=Path)
    x.add_argument("dst", type=Path)
    x.add_argument("out", type=Path)
    e = sub.add_parser(
        "eval", help="perplexity / KLD of an HI2 checkpoint vs the original"
    )
    e.add_argument("src", type=Path)
    e.add_argument("dst", type=Path)
    e.add_argument("--text", type=Path, required=True)
    e.add_argument("--samples", type=int, default=8)
    e.add_argument("--seq-len", type=int, default=2048)
    e.add_argument(
        "--skip-tokens",
        type=int,
        default=0,
        help="skip this many leading tokens (the calibration region)",
    )
    e.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.cmd == "export-dense":
        export_dense(a.src, a.dst, a.out)
        out = {"exported": str(a.out)}
    elif a.cmd == "eval":
        out = evaluate(
            a.src, a.dst, a.text, a.samples, a.seq_len, a.device, a.skip_tokens
        )
    elif a.cmd == "quantize":
        cands = [(nb, b) for nb, b in SELECTABLE if nb in a.bits and b in a.blocks]
        if not cands:
            raise SystemExit("--bits/--blocks leave no candidate configuration")
        rc = RunConfig(
            src=a.src,
            dst=a.dst,
            workdir=a.workdir or Path(str(a.dst) + ".work"),
            devices=[d for d in a.devices.split(",") if d],
            calib=a.calib,
            calib_samples=a.calib_samples,
            seq_len=a.seq_len,
            target_bpw=a.target_bpw,
            include=a.include,
            exclude=a.exclude,
            limit_layers=a.limit_layers,
            compute_dtype=a.compute_dtype,
            sensitivity=a.sensitivity,
            sens_seqs=a.sens_seqs,
            sens_len=a.sens_len,
            sens_floor=a.sens_floor,
            cfg=Config(
                candidates=cands,
                rotation=a.rotation,
                smooth=a.smooth,
                rounding=a.rounding,
                sequential=a.sequential,
                damp=a.damp,
                effort=a.effort,
            ),
        )
        out = run(rc)
    else:
        out = verify_checkpoint(a.src, a.dst, a.max_tensors)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
