# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 quantize|verify|eval|export-dense|restack|size|calib|cache`` (or ``python -m hi2``).

``quantize`` runs layer-wise quantization-aware distillation after PTQ by default when
calibration text is given (``--no-qad`` for PTQ only).
"""

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

# candidate presets: the (nbits, block) pairs each one removes
CANDIDATE_PRESETS = {"all": set(), "lean": {(4, 16)}}


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def cache_command(a) -> dict:
    from . import cache as C

    root = a.cache_dir or C.default_root()
    if a.action == "prune":
        gone = C.prune(root, a.older_than, a.keep_gb, a.digest, a.dry_run)
        return {
            "root": str(root),
            "dry_run": a.dry_run,
            "removed": [f"{e['kind']}/{e['digest']}" for e in gone],
            "freed_gib": round(sum(e["bytes"] for e in gone) / 2**30, 3),
        }
    es = C.entries(root)
    by_kind: dict[str, float] = {}
    for e in es:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + e["bytes"] / 2**30
    out = {
        "root": str(root),
        "entries": len(es),
        "gib": round(sum(e["bytes"] for e in es) / 2**30, 3),
        "gib_by_kind": {k: round(v, 3) for k, v in by_kind.items()},
    }
    if a.action == "list":
        import time

        out["list"] = [
            {
                "entry": f"{e['kind']}/{e['digest']}",
                "gib": round(e["bytes"] / 2**30, 3),
                "complete": e["complete"],
                "last_used": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(e["last_used"])
                ),
                "model": (e["key"] or {}).get("_src"),
                "calib": (e["key"] or {}).get("_calib"),
            }
            for e in es
        ]
    return out


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
        ("final_kl", a.qad_final_kl),
        ("kl_layers", a.qad_kl_layers),
        ("train_norms", a.qad_train_norms),
    ):
        if val is not None:
            over[key] = val
    return over


def _refine_passes(a, qc) -> int:
    """The sweeps of local code refinement: the flag, else 2 for a calibrated GPTQ run without
    QAD. With QAD it is off: QAD takes only the PTQ ladders, and starting it from refined codes
    (``--qad-init ptq``) is worse (docs/format-spec.md, 8.9)."""
    if a.refine_passes is not None:
        return a.refine_passes
    return (
        2 if a.calib and qc is None and a.rounding == "gptq" and not a.act_order else 0
    )


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
        "--cache-devices",
        type=int,
        default=0,
        help="reuse cached statistics and probe made on this many devices, on fewer devices "
        "(0 = the real count); the results of the cached phases are those of N devices",
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
        type=_floats,
        default=None,
        help="quality target: the fewest bits whose weighted error is at most this "
        "multiple of the best achievable (rows that matter less get INT2, the rest "
        "INT4); the bpw is a result. Default 6 with QAD, 4 for PTQ only. Several "
        "values (a,b,c) write one checkpoint per ratio to <dst>-r<ratio> from one "
        "phase 1, probe and encode pass",
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
        "--refine-passes",
        type=int,
        default=None,
        help="sweeps of local code refinement after GPTQ (spec 8.9). Two sweeps lower the "
        "KLD of a PTQ model by 3-13 percent for about 10 percent more encode time. Default 2 "
        "without QAD, off with QAD (it does not combine with it); 0 = off",
    )
    q.add_argument(
        "--smooth-search",
        type=_floats,
        default=None,
        help="per-tensor smoothing exponent (spec 8.5): try these values (a,b,c) and "
        "--smooth for every tensor or shared input and keep the lowest-error one",
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
        "--sens-batch",
        type=int,
        default=1,
        help="probe sequences per forward pass (1 = exact original probe; more is "
        "faster, logits round differently)",
    )
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
    q.add_argument(
        "--candidates",
        choices=sorted(CANDIDATE_PRESETS),
        default="all",
        help="candidate preset on top of --bits/--blocks: lean drops INT4/B16, which "
        "holds 0.1-0.8%% of the weights at 3-3.5 bpw (docs/speed.md)",
    )
    q.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="persistent cache of statistics/Hessians, tables and probe results "
        "(default $HI2_CACHE_DIR or ~/.cache/hi2); see `hi2 cache`",
    )
    q.add_argument(
        "--no-cache",
        action="store_true",
        help="keep statistics and Hessians in the workdir only (the old layout)",
    )
    q.add_argument(
        "--cache-fp16-above",
        type=int,
        default=0,
        help="store cached Sigma in fp16 for inputs wider than this (halves the disk; "
        "lossy, see docs/speed.md); 0 = fp32",
    )
    q.add_argument(
        "--moe-layout",
        choices=("per-expert", "stacked"),
        default="stacked",
        help="routed experts on disk: one stack per layer and gate_up/down group "
        "(default; loads ~9x faster in the fork) or one HI2 tensor per expert "
        "projection (docs/moe.md); the weights are identical",
    )
    q.add_argument("--int8-force", type=int, default=0, help=argparse.SUPPRESS)
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
        "See docs/qad.md",
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
    g.add_argument(
        "--qad-final-kl",
        type=int,
        default=None,
        metavar="STEPS",
        help="after layer-wise QAD, STEPS optimizer steps of whole-network logit KL "
        "training the ladder bytes and norm weights only (default 200 with the default "
        "and thorough presets, 0 with fast; split over the devices when it does not fit one GPU; "
        "skipped if it still does not fit; 0 turns it off; docs/lowbit.md)",
    )
    for flag, typ in (
        ("--qad-kl-layers", int),
        ("--qad-train-norms", str),
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
    r = sub.add_parser(
        "restack",
        help="rewrite an HI2 checkpoint's routed experts per-expert <-> stacked "
        "(data movement only, no re-encode)",
    )
    r.add_argument("dst", type=Path, help="HI2 checkpoint")
    r.add_argument("out", type=Path, help="new directory")
    r.add_argument("--layout", choices=("per-expert", "stacked"), default="stacked")
    z = sub.add_parser(
        "size", help="where the bytes of an HI2 checkpoint go (headers only)"
    )
    z.add_argument("dst", type=Path, help="HI2 checkpoint")
    z.add_argument("--src", type=Path, default=None, help="source model, for the ratio")
    z.add_argument("--by", choices=("kind", "layer", "tensor"), default="kind")
    z.add_argument("--top", type=int, default=20, help="rows of the breakdown table")
    z.add_argument("--json", action="store_true", help="print the report as JSON")
    k = sub.add_parser(
        "calib", help="build a mixed-domain calibration text (and a held-out text)"
    )
    k.add_argument("out", type=Path, help="calibration text to write")
    k.add_argument(
        "sources",
        nargs="+",
        help="SRC or SRC:WEIGHT; .txt, or .json/.jsonl with text or chat records",
    )
    k.add_argument(
        "--tokens",
        type=int,
        default=540_672,
        help="size of the text; the default covers QAD (264 x 2048)",
    )
    k.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="model directory whose tokenizer counts tokens (default: 4 characters each)",
    )
    k.add_argument("--holdout", type=Path, default=None, help="held-out text to write")
    k.add_argument(
        "--holdout-tokens", type=int, default=200_000, help="size of the held-out text"
    )
    k.add_argument(
        "--holdout-share",
        type=float,
        default=0.2,
        help="share of each source's documents kept out of the calibration text",
    )
    c = sub.add_parser("cache", help="list, size or prune the persistent cache")
    c.add_argument("action", choices=("list", "size", "prune"))
    c.add_argument("--cache-dir", type=Path, default=None)
    c.add_argument("--older-than", type=float, default=None, help="days unused")
    c.add_argument("--keep-gb", type=float, default=None, help="prune LRU down to this")
    c.add_argument("--digest", action="append", default=None, help="remove this entry")
    c.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "calib":
        from .calib import build_mix, parse_source

        count = None
        if a.tokenizer is not None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
        share = a.holdout_share if a.holdout is not None else 0.0
        text, held, rep = build_mix(
            [parse_source(x) for x in a.sources],
            a.tokens,
            count,
            share,
            a.holdout_tokens,
        )
        a.out.write_text(text)
        if a.holdout is not None:
            a.holdout.write_text(held)
        print(json.dumps(rep, indent=2))
        return
    if a.cmd == "size":
        from .size import format_report, size_report

        rep = size_report(a.dst, a.src, a.by)
        print(json.dumps(rep, indent=2) if a.json else format_report(rep, a.top))
        return
    if a.cmd == "cache":
        out = cache_command(a)
    elif a.cmd == "restack":
        from .stack import restack

        out = restack(a.dst, a.out, a.layout, log=lambda m: print(m, file=sys.stderr))
        for k in ("before", "after"):
            out[k].pop("per_shard")
    elif a.cmd == "export-dense":
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
        from .cache import default_root

        ratios = a.max_error_ratio
        cands = [
            (nb, b)
            for nb, b in SELECTABLE
            if nb in a.bits
            and b in a.blocks
            and (nb, b) not in CANDIDATE_PRESETS[a.candidates]
        ]
        if not cands:
            raise SystemExit("--bits/--blocks leave no candidate configuration")
        qc = make_config(a.qad_preset, _qad_overrides(a)) if a.qad else None
        sens = a.sensitivity or ("probe" if a.qad and a.calib else "none")
        rc = RunConfig(
            src=a.src,
            dst=a.dst,
            workdir=a.workdir or Path(str(a.dst) + ".work"),
            devices=resolve_devices(a.devices),
            cache_devices=a.cache_devices,
            calib=[Path(p) for p in a.calib.split(",") if p] if a.calib else None,
            qad=qc,
            calib_samples=a.calib_samples,
            seq_len=a.seq_len,
            target_bpw=a.target_bpw,
            max_error_ratio=ratios[0] if ratios and len(ratios) == 1 else None,
            max_error_ratios=ratios if ratios and len(ratios) > 1 else None,
            sigma_fp16_above=a.cache_fp16_above,
            cache_dir=None if a.no_cache else (a.cache_dir or default_root()),
            include=a.include,
            exclude=a.exclude,
            limit_layers=a.limit_layers,
            compute_dtype=a.compute_dtype,
            sensitivity=sens,
            sens_seqs=a.sens_seqs,
            sens_len=a.sens_len,
            sens_floor=a.sens_floor,
            sens_batch=a.sens_batch,
            sens_dtype=a.sens_dtype,
            expert_share_floor=a.expert_share_floor,
            moe_layout=a.moe_layout,
            int8_force=a.int8_force,
            cfg=Config(
                candidates=cands,
                rotation=a.rotation,
                smooth=a.smooth,
                smooth_search=tuple(a.smooth_search or ()),
                rounding=a.rounding,
                sequential=a.sequential,
                act_order=a.act_order,
                refine_passes=_refine_passes(a, qc),
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
