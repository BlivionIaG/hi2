# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""``python -m hi2 quantize|verify|eval|export-dense`` (run from tools/hi2 or with it on
PYTHONPATH). ``quantize`` runs layer-wise quantization-aware distillation after PTQ by
default when calibration text is given (``--no-qad`` for PTQ only)."""

import argparse
import json
import sys
from pathlib import Path

from .calibrate import SELECTABLE, Config
from .checkpoint import DEFAULT_EXCLUDE, export_dense, verify_checkpoint
from .encode import EFFORT
from .evalq import evaluate
from .pipeline import RunConfig, plan, run
from .qad import PRESETS as QAD_PRESETS
from .qad import make_config


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def resolve_devices(spec: str) -> list[str]:
    """``auto`` = every visible GPU (narrow it with HIP_VISIBLE_DEVICES /
    CUDA_VISIBLE_DEVICES), else the CPU; otherwise a comma-separated list."""
    if spec != "auto":
        return [d for d in spec.split(",") if d]
    import torch

    n = torch.accelerator.device_count() if torch.accelerator.is_available() else 0
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]


def _qad_overrides(a) -> dict:
    over = {}
    if a.qad_config is not None:
        over.update(json.loads(Path(a.qad_config).read_text()))
    for key, val in (
        ("steps", a.qad_steps),
        ("lr_w", a.qad_lr),
        ("lr_p", a.qad_lr_params),
        ("batch", a.qad_batch),
        ("samples", a.qad_samples),
        ("budget", a.qad_budget),
        ("init", a.qad_init),
        ("layers", a.qad_layers),
    ):
        if val is not None:
            over[key] = val
    return over


def main(argv=None):
    if hasattr(
        sys.stdout, "reconfigure"
    ):  # progress lines show up promptly in log files
        sys.stdout.reconfigure(line_buffering=True)
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
        default="auto",
        help="auto (default: every visible GPU, else CPU), cpu, cuda:0, or a list such as "
        "cuda:0,cuda:1 (HIP on ROCm). Several devices split calibration sequences and "
        "shard each tensor's rows",
    )
    q.add_argument(
        "--calib",
        default=None,
        help=".txt or .jsonl calibration text (several: comma-separated, concatenated "
        "in order); omit for data-free",
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
        default=None,
        help="model-wide bits per weight (global allocation); default is the "
        "quality mode below",
    )
    q.add_argument(
        "--max-error-ratio",
        type=float,
        default=None,
        help="quality target: the fewest bits whose weighted error is at most this "
        "multiple of the best achievable (rows that matter less get INT2, the rest "
        "INT4); the bpw is a result",
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
        default=None,
        help="probe = weight the allocation by each tensor's measured KL cost (spec 8.6); "
        "default probe when calibrated with QAD on, else none",
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
        "--sens-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="dtype of the whole-model sensitivity probe (float16 when fp32 does not fit; "
        "bfloat16 is emulated, very slowly, on gfx1030)",
    )
    q.add_argument(
        "--expert-min-tokens",
        type=float,
        default=2.0,
        help="MoE: an expert routed fewer than this x its input width tokens tops its "
        "statistics up with the layer average (0 = never)",
    )
    q.add_argument(
        "--expert-share-floor",
        type=float,
        default=0.05,
        help="MoE: lower bound of an expert's routed-token share in its allocation weight",
    )
    q.add_argument(
        "--sequential",
        action="store_true",
        help="gptq on the quantized prefix (spec 8.4): a second layerwise model pass",
    )
    q.add_argument(
        "--act-order",
        action="store_true",
        help="round columns in descending Hessian-diagonal order (storage unchanged)",
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
    g = q.add_argument_group("quantization-aware distillation (spec 13)")
    g.add_argument(
        "--qad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="on by default: after PTQ, distill every layer against the unquantized model "
        "(needs --calib; data-free runs fall back to PTQ). --no-qad = PTQ only. "
        "See docs/rdna2/hi2-qad.md",
    )
    g.add_argument(
        "--qad-preset",
        choices=list(QAD_PRESETS),
        default="default",
        help="fast | default | thorough (steps and sequences per layer)",
    )
    g.add_argument(
        "--qad-budget",
        type=float,
        default=None,
        help="fraction of preset steps x layers to spend; layers that gain little get "
        "fewer steps and stop early (preset default)",
    )
    g.add_argument(
        "--qad-samples",
        type=int,
        default=None,
        help="training sequences of --seq-len tokens (preset default)",
    )
    g.add_argument(
        "--qad-layers", default=None, help='layers to train, e.g. "0-5,9" (default all)'
    )
    g.add_argument(
        "--qad-config",
        default=None,
        help="JSON file of advanced QAD options (fields of hi2.qad.QadConfig)",
    )
    for flag, typ in (
        ("--qad-steps", int),
        ("--qad-lr", float),
        ("--qad-lr-params", float),
        ("--qad-batch", int),
        ("--qad-init", str),
    ):
        g.add_argument(flag, type=typ, default=None, help=argparse.SUPPRESS)
    q.add_argument(
        "--dry-run",
        action="store_true",
        help="check the inputs (model, calibration tokens, devices, RAM) and print the "
        "plan without quantizing",
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
    e.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="dtype of both models (float16 when an fp32 copy does not fit the device)",
    )
    a = ap.parse_args(argv)
    if a.cmd == "export-dense":
        export_dense(a.src, a.dst, a.out)
        out = {"exported": str(a.out)}
    elif a.cmd == "eval":
        out = evaluate(
            a.src,
            a.dst,
            a.text,
            a.samples,
            a.seq_len,
            a.device,
            a.skip_tokens,
            dtype=a.dtype,
        )
    elif a.cmd == "quantize":
        cands = [(nb, b) for nb, b in SELECTABLE if nb in a.bits and b in a.blocks]
        if not cands:
            raise SystemExit("--bits/--blocks leave no candidate configuration")
        qc = make_config(a.qad_preset, _qad_overrides(a)) if a.qad else None
        sens = a.sensitivity or ("probe" if a.qad and a.calib else "none")
        rc = RunConfig(
            src=a.src,
            dst=a.dst,
            workdir=a.workdir or Path(str(a.dst) + ".work"),
            devices=resolve_devices(a.devices),
            calib=[Path(p) for p in a.calib.split(",") if p] if a.calib else None,
            qad=qc,
            calib_samples=a.calib_samples,
            seq_len=a.seq_len,
            target_bpw=a.target_bpw,
            max_error_ratio=a.max_error_ratio,
            include=a.include,
            exclude=a.exclude,
            limit_layers=a.limit_layers,
            compute_dtype=a.compute_dtype,
            sensitivity=sens,
            sens_seqs=a.sens_seqs,
            sens_len=a.sens_len,
            sens_floor=a.sens_floor,
            sens_dtype=a.sens_dtype,
            expert_share_floor=a.expert_share_floor,
            cfg=Config(
                candidates=cands,
                rotation=a.rotation,
                smooth=a.smooth,
                rounding=a.rounding,
                sequential=a.sequential,
                act_order=a.act_order,
                damp=a.damp,
                effort=a.effort,
                expert_min_tokens=a.expert_min_tokens,
            ),
        )
        out = plan(rc) if a.dry_run else run(rc)
    else:
        out = verify_checkpoint(a.src, a.dst, a.max_tensors)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
