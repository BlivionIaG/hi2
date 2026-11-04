# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 quantize|verify|eval|export-dense|restack|size|calib|cache`` (or ``python -m hi2``).

``quantize`` runs layer-wise quantization-aware distillation after PTQ by default when
calibration text is given (``--no-qad`` for PTQ only).
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

# candidate presets: the (nbits, block) pairs each one removes
CANDIDATE_PRESETS = {"all": set(), "lean": {(4, 16)}}

# Names only: compute configuration and preset values stay in their owning modules.
EFFORT_NAMES = ("fast", "normal", "max")
QAD_PRESET_NAMES = ("fast", "default", "thorough")
RESULT_SCHEMA_VERSION = 1


class _UsageError(ValueError):
    def __init__(self, message, command=None):
        super().__init__(message)
        self.command = command


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message, self.prog.partition(" ")[2] or None)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def _version() -> str:
    try:
        return f"hi2 {version('hi2')}"
    except PackageNotFoundError:
        return f"hi2 (uninstalled source checkout: {Path(__file__).parent.parent})"


@contextmanager
def _stderr_logs():
    """Keep Python, subprocess and native stdout chatter off the result channel.

    This is a process-wide CLI boundary, not a redirection for concurrent service
    requests. Services should pass their own log callbacks to the core APIs.
    """
    import ctypes

    # Native libraries may buffer C stdout even after their Python call returns.
    fflush = ctypes.CDLL("msvcrt" if os.name == "nt" else None).fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    sys.stdout.flush()
    sys.stderr.flush()
    fflush(None)
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            try:
                yield
            finally:
                sys.stdout.flush()
                fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)


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
        ("dtype", a.qad_dtype),
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


def _tune_rc(a, rc):
    """Measure PTQ trials in an exclusively owned, disposable workspace."""
    import dataclasses

    from .evalq import evaluate
    from .paths import check_output_paths
    from .pipeline import calib_paths, run
    from .runtime import atomic_json
    from .service import plan_tuning
    from .tune import START_RATIOS, tune

    tuning_plan = plan_tuning(
        rc,
        target_kld=a.target_kld,
        text=a.eval_text,
        ratios=a.tune_ratios or START_RATIOS,
        device=a.eval_device,
        dtype=a.eval_dtype,
        skip_tokens=a.eval_skip_tokens,
    )
    if a.dry_run:
        return tuning_plan

    report_path = rc.dst.parent / (rc.dst.name + ".tune.json")
    check_output_paths(
        inputs=[rc.src, *calib_paths(rc.calib), a.eval_text],
        outputs=[rc.dst, report_path],
    )
    rc.dst.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    points = []
    # Neither the final workdir nor a predictable <dst>-tune-r* path is disposable.
    # Each round has its own markers; the persistent cache can still share phase 1.
    with TemporaryDirectory(prefix=f".{rc.dst.name}-tune-", dir=rc.dst.parent) as tmp:
        workspace = Path(tmp).resolve()
        print(f"tuning workspace: {workspace}", file=sys.stderr)

        def measure(ratios):
            ratios = list(ratios)
            if len(ratios) == 1:
                ratios.append(ratios[0] * 1.5)
            with TemporaryDirectory(prefix="round-", dir=workspace) as trial_tmp:
                trial = Path(trial_tmp)
                probe = dataclasses.replace(
                    rc,
                    dst=trial / "checkpoint",
                    workdir=trial / "work",
                    qad=None,
                    target_bpw=None,
                    max_error_ratio=None,
                    max_error_ratios=ratios,
                )
                record = {"ratios": ratios, "path": str(trial), "status": "running"}
                trials.append(record)
                out = run(probe)["outputs"]
                pts = []
                for ratio in ratios:
                    dst = Path(f"{probe.dst}-r{ratio:g}")
                    res = evaluate(
                        rc.src,
                        dst,
                        a.eval_text,
                        device=a.eval_device,
                        skip_tokens=a.eval_skip_tokens,
                        dtype=a.eval_dtype,
                        log=lambda *_: None,
                    )
                    point = (
                        ratio,
                        res["kld_nats_per_token"],
                        out[str(dst)]["allocation_bpw"],
                    )
                    pts.append(point)
                    points.append(point)
                record["status"] = "completed"
                return pts

        try:
            fit = tune(measure, a.target_kld, a.tune_ratios or START_RATIOS)
            atomic_json(report_path, fit)
            a.tune_report = str(report_path)
        except BaseException as exc:
            # Preserve small, unique diagnostics outside the disposable workspace,
            # including partial measurements. Cleanup still belongs to TemporaryDirectory.
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            if trials and trials[-1]["status"] == "running":
                trials[-1]["status"] = status
            diagnostics = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": status,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "workspace": str(workspace),
                "target_kld": a.target_kld,
                "trials": trials,
                "points": points,
            }
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=f"{rc.dst.name}.tune-error-",
                    suffix=".json",
                    dir=rc.dst.parent,
                    delete=False,
                ) as diagnostic_file:
                    json.dump(diagnostics, diagnostic_file, indent=2)
                    a.tune_diagnostics = diagnostic_file.name
                print(f"tuning diagnostics: {a.tune_diagnostics}", file=sys.stderr)
            except OSError as diagnostic_error:
                print(
                    f"could not save tuning diagnostics: {diagnostic_error}",
                    file=sys.stderr,
                )
            raise

    if fit["status"] != "ok":
        print(
            f"warning: target KLD {a.target_kld} is {fit['status']} the measured range; "
            f"using ratio {fit['ratio']:.2f} (about {fit['bpw']:.2f} bpw)",
            file=sys.stderr,
        )
    print(
        f"target KLD {a.target_kld}: error ratio {fit['ratio']:.2f}, about "
        f"{fit['bpw']:.2f} bpw (PTQ-measured; independently validate final QAD quality)",
        file=sys.stderr,
    )
    return dataclasses.replace(rc, max_error_ratio=fit["ratio"], max_error_ratios=None)


def build_parser() -> argparse.ArgumentParser:
    """Build help and parse arguments without importing any compute modules."""
    ap = _Parser(
        prog="hi2",
        formatter_class=_HelpFormatter,
        description="Prepare calibration data, plan and quantize HF checkpoints, then verify or evaluate.",
        epilog=(
            "Start here:\n"
            "  hi2 calib calib.txt corpus.jsonl --holdout heldout.txt\n"
            "  hi2 quantize MODEL OUT --calib calib.txt --dry-run\n"
            "  hi2 quantize MODEL OUT --calib calib.txt --no-qad --devices cuda:0\n"
            "  hi2 quantize MODEL OUT --devices cpu  # data-free PTQ\n\n"
            "Calibrated quantization enables QAD and preset-dependent final-KL training;\n"
            "--no-qad selects PTQ only. Auto devices means every visible GPU, else CPU.\n"
            "Recovery: rerun the same command with the same workdir; do not change the\n"
            "recipe when reusing resume state. See docs/usage.md and COMMAND --help.\n"
            "--json emits one versioned result/error object; logs always go to stderr.\n"
            "Help and version remain plain text, including with --json."
        ),
    )
    ap.add_argument("--version", action="version", version=_version())
    ap.add_argument(
        "--json", action="store_true", help="versioned JSON result or error"
    )
    ap.add_argument(
        "--quiet", action="store_true", help="suppress progress diagnostics"
    )
    ap.add_argument(
        "--debug", action="store_true", help="include failure tracebacks on stderr"
    )
    ap.add_argument("--log-format", choices=("text", "jsonl"), default="text")
    sub = ap.add_subparsers(dest="cmd", required=True, title="workflows")

    def command(name, help):
        parser = sub.add_parser(
            name,
            help=help,
            description=help,
            formatter_class=_HelpFormatter,
            epilog=f"Examples and input formats: docs/usage.md ({name}).",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="versioned JSON result or error; diagnostics go to stderr",
        )
        parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument(
            "--log-format", choices=("text", "jsonl"), default=argparse.SUPPRESS
        )
        return parser

    q = command("quantize", "HF checkpoint -> HI2 flex checkpoint")
    q.add_argument("src", type=Path, help="source HF model directory")
    q.add_argument("dst", type=Path, help="output checkpoint directory")
    common = q.add_argument_group("common workflow and resource controls")
    common.add_argument(
        "--calib",
        default=None,
        help=".txt or .jsonl with text fields; several paths may be comma-separated. "
        "Omit for data-free PTQ. Prepare chat records with `hi2 calib` first",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="check inputs and print the plan without quantizing",
    )
    common.add_argument(
        "--devices",
        default="auto",
        help="auto uses every visible GPU, else CPU; cpu, cuda:0, or cuda:0,cuda:1 "
        "(HIP on ROCm). Narrow auto with HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES. "
        "Several devices split calibration sequences and shard tensor rows",
    )
    common.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="tables, statistics and resume state (when omitted: <dst>.work). "
        "Resume by rerunning the same command",
    )
    common.add_argument(
        "--qad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="distill every layer after PTQ when calibrated; --no-qad = PTQ only. "
        "Data-free runs fall back to PTQ. Adds training time and memory; docs/qad.md",
    )
    common.add_argument(
        "--qad-preset",
        choices=QAD_PRESET_NAMES,
        default="default",
        help="layer-training budget; default/thorough also request whole-network "
        "final-KL training, fast disables that phase",
    )
    common.add_argument(
        "--target-bpw",
        type=float,
        default=None,
        help="model-wide bits per weight; replaces the automatic quality target",
    )
    common.add_argument(
        "--max-error-ratio",
        type=_floats,
        default=None,
        help="quality target: fewest bits within this multiple of the best weighted "
        "error; bpw is a result. Omit for the core recipe's automatic target "
        "(shown by --dry-run). Several values a,b,c write <dst>-r<ratio> from one "
        "phase 1/probe/encode pass, but repeat QAD for each output",
    )

    data = q.add_argument_group("calibration and persistent storage")
    data.add_argument(
        "--calib-samples",
        type=int,
        default=128,
        help="calibration sequences of --seq-len tokens",
    )
    data.add_argument("--seq-len", type=int, default=2048, help="tokens per sequence")
    data.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="persistent statistics/Hessians, tables and probe cache "
        "(when omitted: $HI2_CACHE_DIR or ~/.cache/hi2); see `hi2 cache`",
    )
    data.add_argument(
        "--no-cache",
        action="store_true",
        help="keep statistics and Hessians in the workdir only",
    )
    data.add_argument(
        "--moe-layout",
        choices=("per-expert", "stacked"),
        default="stacked",
        help="expert layout on disk: one stack per layer/group or one tensor per "
        "expert projection; identical weights, different loader costs (docs/moe.md)",
    )

    tuning = q.add_argument_group(
        "expert: measured KLD target (additional PTQ/evaluation passes)"
    )
    tuning.add_argument(
        "--target-kld",
        type=float,
        default=None,
        metavar="KLD",
        help="measure KL(original||quantized) on --eval-text for several PTQ ratios, "
        "then run with the largest ratio within KLD. Replaces --target-bpw and "
        "--max-error-ratio; needs --calib and --eval-text (docs/usage.md)",
    )
    tuning.add_argument(
        "--eval-text", type=Path, default=None, help="held-out text for tuning"
    )
    tuning.add_argument(
        "--eval-skip-tokens", type=int, default=0, help="leading eval tokens to skip"
    )
    tuning.add_argument(
        "--eval-device",
        default="cuda:0",
        help="tuning evaluation device; independent of --devices",
    )
    tuning.add_argument(
        "--eval-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="tuning evaluation math dtype",
    )
    tuning.add_argument(
        "--tune-ratios",
        type=_floats,
        default=None,
        help="first ratios a,b,c measured by --target-kld; omitted uses hi2.tune.START_RATIOS",
    )

    encoding = q.add_argument_group("expert: encoding and candidate search")
    encoding.add_argument(
        "--rotation",
        choices=("on", "auto", "off"),
        default="on",
        help="Hadamard flags; pairs with --smooth when calibrated",
    )
    encoding.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="smoothing exponent alpha (0 = off); needs --calib",
    )
    encoding.add_argument(
        "--refine-passes",
        type=int,
        default=None,
        help="local code refinement sweeps after GPTQ (spec 8.9); omitted uses two "
        "for calibrated GPTQ without QAD or act-order, otherwise zero",
    )
    encoding.add_argument(
        "--smooth-search",
        type=_floats,
        default=None,
        help="try these per-tensor smoothing exponents a,b,c plus --smooth (spec 8.5)",
    )
    encoding.add_argument(
        "--rounding",
        choices=("gptq", "nearest"),
        default="gptq",
        help="Hessian-aware error-compensated rounding (spec 8.3); data-free falls back to nearest",
    )
    encoding.add_argument(
        "--damp", type=float, default=0.01, help="Hessian damping for GPTQ"
    )
    encoding.add_argument(
        "--sequential",
        action="store_true",
        help="GPTQ on the quantized prefix: a second layerwise pass (spec 8.4); not with QAD",
    )
    encoding.add_argument(
        "--act-order",
        action="store_true",
        help="round columns in descending Hessian-diagonal order; storage unchanged",
    )
    encoding.add_argument(
        "--effort", choices=EFFORT_NAMES, default="normal", help="ladder search effort"
    )
    encoding.add_argument(
        "--bits",
        type=_ints,
        default=[2, 4],
        help="allowed code widths; 8 opts into INT8 rows",
    )
    encoding.add_argument(
        "--blocks", type=_ints, default=[16, 32, 64], help="allowed block sizes"
    )
    encoding.add_argument(
        "--candidates",
        choices=sorted(CANDIDATE_PRESETS),
        default="all",
        help="preset on top of --bits/--blocks: lean drops INT4/B16 (docs/speed.md)",
    )
    encoding.add_argument("--include", default=None, help="regex: only these weights")
    encoding.add_argument(
        "--exclude",
        default=None,
        help="regex: keep dense; omitted uses the core exclusions for embeddings, "
        "output head, norms, routers, indexers, convolutions and biases",
    )
    encoding.add_argument(
        "--limit-layers", type=int, default=None, help="limit decoder layers"
    )
    encoding.add_argument(
        "--compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="calibration compute dtype",
    )
    encoding.add_argument("--int8-force", type=int, default=0, help=argparse.SUPPRESS)

    probe = q.add_argument_group(
        "expert: sensitivity, MoE statistics and cache precision"
    )
    probe.add_argument(
        "--sensitivity",
        choices=("none", "probe"),
        default=None,
        help="weight allocation by measured per-tensor KL cost (spec 8.6); "
        "omitted uses probe when calibrated with QAD, else none",
    )
    probe.add_argument("--sens-seqs", type=int, default=8, help="probe sequences")
    probe.add_argument(
        "--sens-len", type=int, default=512, help="probe sequence length"
    )
    probe.add_argument(
        "--sens-batch",
        type=int,
        default=1,
        help="sequences per probe forward pass; larger batches are faster but round differently",
    )
    probe.add_argument(
        "--sens-floor",
        type=float,
        default=0.05,
        help="lower bound of a tensor weight relative to the median",
    )
    probe.add_argument(
        "--sens-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="whole-model probe dtype; auto uses float32 when it fits, otherwise float16. "
        "bfloat16 is emulated very slowly on gfx1030",
    )
    probe.add_argument(
        "--expert-min-tokens",
        type=float,
        default=2.0,
        help="experts routed fewer than this x input-width tokens use layer-average "
        "statistics to top up (0 = never)",
    )
    probe.add_argument(
        "--expert-share-floor",
        type=float,
        default=0.05,
        help="lower bound of an expert's routed-token share in its allocation weight",
    )
    probe.add_argument(
        "--cache-devices",
        type=int,
        default=0,
        help="reuse statistics/probe made with this many devices on fewer devices "
        "(0 = real count); cached phases retain the original device count's results",
    )
    probe.add_argument(
        "--cache-fp16-above",
        type=int,
        default=0,
        help="store cached Sigma in fp16 above this input width; lossy, halves disk (0 = fp32)",
    )

    g = q.add_argument_group("expert: quantization-aware distillation (spec 13)")
    g.add_argument(
        "--qad-budget",
        type=float,
        default=None,
        help="fraction of preset steps x layers; low-gain layers stop early (preset default)",
    )
    g.add_argument(
        "--qad-samples",
        type=int,
        default=None,
        help="training sequences of --seq-len tokens (preset default)",
    )
    g.add_argument(
        "--qad-layers",
        default=None,
        help='layers to train, e.g. "0-5,9" (omitted: all)',
    )
    g.add_argument(
        "--qad-config",
        default=None,
        help="path to a JSON file of advanced hi2.qad.QadConfig options",
    )
    g.add_argument(
        "--qad-final-kl",
        type=int,
        default=None,
        metavar="STEPS",
        help="whole-network logit-KL optimizer steps after layer-wise QAD; omitted "
        "uses the preset, 0 disables. Splits over devices if needed; may be skipped "
        "if it still does not fit (docs/lowbit.md)",
    )
    g.add_argument(
        "--qad-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="layer forward/backward math dtype (preset default); latents, ladder "
        "bytes and loss stay float32. Not bit-comparable across dtypes (docs/qad.md)",
    )
    for flag, typ, description in (
        ("--qad-kl-layers", int, "last layers trained with logit KL"),
        ("--qad-train-norms", str, "train norm weights (QadConfig boolean)"),
        ("--qad-steps", int, "optimizer steps per layer"),
        ("--qad-lr", float, "weight learning rate"),
        ("--qad-lr-params", float, "ladder-parameter learning rate"),
        ("--qad-batch", int, "training sequences per batch"),
        ("--qad-init", str, "latent initialization (QadConfig init)"),
    ):
        g.add_argument(
            flag, type=typ, default=None, help=f"{description} (preset default)"
        )
    samples, length = q.get_default("calib_samples"), q.get_default("seq_len")
    q.epilog = (
        f"Default cost: --devices {q.get_default('devices')} selects every visible GPU,\n"
        f"else CPU. Calibrated runs request QAD={q.get_default('qad')} with preset\n"
        f"{q.get_default('qad_preset')!r}, including layer-wise and final-KL training;\n"
        f"PTQ calibration alone uses {samples} x {length} = {samples * length:,} tokens.\n"
        "QAD needs additional training/held-out tokens and memory (see --dry-run).\n"
        "Data-free runs use nearest-rounding PTQ; --no-qad keeps calibrated runs PTQ-only.\n\n"
        "Plan:      hi2 quantize MODEL OUT --calib calib.txt --dry-run\n"
        "PTQ only:  hi2 quantize MODEL OUT --calib calib.txt --no-qad --devices cuda:0\n"
        "Data-free: hi2 quantize MODEL OUT --devices cpu\n"
        "Recovery: rerun the same command and workdir; cached/completed work is reused.\n"
        "Examples: docs/usage.md. Training cost and recovery boundaries: docs/qad.md."
    )

    v = command(
        "verify",
        "reload an HI2 checkpoint and compare (sampled numerical verification)",
    )
    v.add_argument("src", type=Path)
    v.add_argument("dst", type=Path)
    v.add_argument(
        "--max-tensors",
        type=int,
        default=8,
        help="maximum tensors to compare numerically",
    )
    x = command(
        "export-dense", "write the dequantised model as a plain fp16 HF directory"
    )
    x.add_argument("src", type=Path)
    x.add_argument("dst", type=Path)
    x.add_argument("out", type=Path)
    e = command(
        "eval",
        "perplexity / KLD vs the original (dense numerical check, not native kernels)",
    )
    e.add_argument("src", type=Path)
    e.add_argument("dst", type=Path)
    e.add_argument(
        "--text", type=Path, required=True, help="held-out .txt or text-field .jsonl"
    )
    e.add_argument("--samples", type=int, default=8, help="evaluation sequences")
    e.add_argument("--seq-len", type=int, default=2048, help="tokens per sequence")
    e.add_argument(
        "--skip-tokens",
        type=int,
        default=0,
        help="skip leading tokens (the calibration region)",
    )
    e.add_argument(
        "--device",
        default="cpu",
        help="device or comma-separated list to split the model layer-wise across GPUs",
    )
    e.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="math dtype of both models; float16 when an fp32 copy does not fit",
    )
    r = command(
        "restack", "rewrite MoE expert layout (data movement only, no re-encode)"
    )
    r.add_argument("dst", type=Path, help="HI2 checkpoint")
    r.add_argument("out", type=Path, help="new directory")
    r.add_argument("--layout", choices=("per-expert", "stacked"), default="stacked")
    z = command(
        "size", "where checkpoint bytes go (headers only; human table unless --json)"
    )
    z.add_argument("dst", type=Path, help="HI2 checkpoint")
    z.add_argument("--src", type=Path, default=None, help="source model, for the ratio")
    z.add_argument("--by", choices=("kind", "layer", "tensor"), default="kind")
    z.add_argument(
        "--top", type=int, default=20, help="rows of the human breakdown table"
    )

    k = command("calib", "build mixed-domain calibration and optional held-out text")
    k.add_argument("out", type=Path, help="calibration text to write")
    k.add_argument(
        "sources",
        nargs="+",
        help="SRC or SRC:WEIGHT; .txt, or .json/.jsonl with text or chat records. "
        "Direct quantize accepts only text-field JSONL, not chat records",
    )
    k.add_argument(
        "--tokens", type=int, default=540_672, help="target calibration tokens"
    )
    k.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="local model/tokenizer directory for actual token counts (omitted: characters/4 estimate)",
    )
    templates = k.add_mutually_exclusive_group()
    templates.add_argument(
        "--chat-template",
        action="store_true",
        help="explicitly render chat records with --tokenizer's template; plain text is unchanged",
    )
    templates.add_argument(
        "--chat-template-file",
        type=Path,
        default=None,
        help="explicit Jinja chat template override; requires --tokenizer",
    )
    k.add_argument(
        "--holdout", type=Path, default=None, help="distinct held-out text output"
    )
    k.add_argument(
        "--holdout-tokens", type=int, default=200_000, help="target held-out tokens"
    )
    k.add_argument(
        "--min-holdout-tokens",
        type=int,
        default=1,
        help="minimum held-out tokens; for eval use samples x seq-len + skip-tokens. "
        "Counts are estimates unless --tokenizer is supplied",
    )
    k.add_argument(
        "--holdout-share",
        type=float,
        default=0.2,
        help="share of each source's documents reserved for held-out text",
    )
    k.epilog = (
        "Plain mixer: hi2 calib calib.txt corpus.jsonl --holdout heldout.txt\n"
        "Chat template: hi2 calib calib.txt chats.jsonl --tokenizer MODEL --chat-template\n"
        "Rendering is opt-in, deterministic and recorded with tokenizer/template identity.\n"
        "Text records and .txt are not reformatted. Feed the prepared .txt to quantize.\n"
        "Examples and split/token budgets: docs/usage.md."
    )
    c = command("cache", "list, size or prune the persistent cache")
    c.add_argument("action", choices=("list", "size", "prune"))
    c.add_argument("--cache-dir", type=Path, default=None)
    c.add_argument("--older-than", type=float, default=None, help="days unused")
    c.add_argument("--keep-gb", type=float, default=None, help="prune LRU down to this")
    c.add_argument("--digest", action="append", default=None, help="remove this entry")
    c.add_argument(
        "--dry-run", action="store_true", help="describe pruning without deleting"
    )
    command("doctor", "inspect the installed backend without running a model")
    inspect = command(
        "inspect", "inspect model/artifact metadata and compatibility evidence"
    )
    inspect.add_argument("path", type=Path)
    status = command(
        "status", "read run ownership, phase outcomes and safe recovery guidance"
    )
    status.add_argument("workdir", type=Path)
    validate = command("validate", "fully validate every tensor in an HI2 artifact")
    validate.add_argument("path", type=Path)
    portable = command(
        "export-portable",
        "materialize a portable HI2 artifact without dense conversion",
    )
    portable.add_argument("src", type=Path)
    portable.add_argument("out", type=Path)
    portable.add_argument("--assets-from", type=Path)
    quality = command(
        "quality", "measure final dense numerical quality and enforce acceptance gates"
    )
    quality.add_argument("src", type=Path)
    quality.add_argument("dst", type=Path)
    quality.add_argument("--text", type=Path, required=True)
    quality.add_argument("--exclude-data", type=Path, action="append", default=[])
    quality.add_argument("--max-kld", type=float)
    quality.add_argument("--max-ppl-ratio", type=float)
    quality.add_argument("--samples", type=int, default=8)
    quality.add_argument("--seq-len", type=int, default=2048)
    quality.add_argument("--device", default="cpu")
    quality.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    recipe = command("recipe", "save or replay a versioned resolved run recipe")
    recipe.add_argument("action", choices=("save", "run"))
    recipe.add_argument("path", nargs="?", type=Path)
    recipe.add_argument("--from-plan", type=Path)
    recipe.add_argument("--out", type=Path)
    recipe.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="explicit replay override; dotted fields such as cfg.effort are supported",
    )
    recipe.add_argument("--dry-run", action="store_true")
    jobs = command("jobs", "submit, inspect or cancel a durable local job")
    jobs.add_argument("action", choices=("submit", "status", "cancel"))
    jobs.add_argument("target", help="request JSON file for submit, otherwise a job ID")
    jobs.add_argument("--state-root", type=Path, required=True)
    jobs.add_argument(
        "--operation",
        choices=("quantize", "validate", "quality", "portable"),
        default="quantize",
    )
    jobs.add_argument("--idempotency-key")
    jobs.add_argument("--cursor", type=int, default=0)
    jobs.add_argument("--limit", type=int, default=100)
    return ap


def dispatch_command(a) -> dict:
    """Run a parsed command; keep transport/output handling in main()."""
    from .service import Progress

    progress = Progress(
        quiet=getattr(a, "quiet", False),
        jsonl=getattr(a, "log_format", "text") == "jsonl",
    )
    if a.cmd == "doctor":
        from .service import doctor

        return doctor()
    if a.cmd == "inspect":
        from .service import inspect_artifact

        return inspect_artifact(a.path)
    if a.cmd == "status":
        from .service import status

        return status(a.workdir)
    if a.cmd == "validate":
        from .checkpoint import validate_checkpoint

        return validate_checkpoint(a.path, log=progress)
    if a.cmd == "export-portable":
        from .checkpoint import export_portable

        return export_portable(a.src, a.out, assets_from=a.assets_from)
    if a.cmd == "quality":
        from .quality import evaluate_quality

        return evaluate_quality(
            a.src,
            a.dst,
            a.text,
            max_kld=a.max_kld,
            max_ppl_ratio=a.max_ppl_ratio,
            excluded=a.exclude_data,
            samples=a.samples,
            seq_len=a.seq_len,
            device=a.device,
            dtype=a.dtype,
            log=progress,
        )
    if a.cmd == "recipe":
        from .service import config_from_dict, execute_run, plan_run, save_recipe

        if a.action == "save":
            if a.from_plan is None or a.out is None or a.path is not None or a.set:
                raise ValueError(
                    "recipe save requires --from-plan PLAN --out RECIPE, without overrides"
                )
            document = json.loads(a.from_plan.read_text())
            rc = config_from_dict(
                {
                    "schema_version": document.get("schema_version"),
                    "config": document["config"],
                }
            )
            save_recipe(a.out, rc)
            return {"recipe": str(a.out.resolve())}
        if a.path is None or a.from_plan is not None or a.out is not None:
            raise ValueError(
                "recipe run requires a recipe path; use --set FIELD=JSON for overrides"
            )
        document = json.loads(a.path.read_text())
        for override in a.set:
            name, separator, value = override.partition("=")
            if not separator or not name or any(not part for part in name.split(".")):
                raise ValueError("recipe overrides use FIELD=JSON")
            target = document["config"]
            parts = name.split(".")
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(f"unknown recipe object: {name}")
                target = target[part]
            target[parts[-1]] = json.loads(value)
        rc = config_from_dict(document)
        return plan_run(rc, progress) if a.dry_run else execute_run(rc, progress)
    if a.cmd == "jobs":
        from .jobs import JobStore

        store = JobStore(a.state_root)
        if a.action == "submit":
            if not a.idempotency_key:
                raise ValueError("jobs submit requires --idempotency-key")
            return store.submit(
                a.operation, json.loads(Path(a.target).read_text()), a.idempotency_key
            )
        if a.action == "cancel":
            return store.cancel(a.target)
        return store.status(a.target, cursor=a.cursor, limit=a.limit)
    if a.cmd == "calib":
        from .calib import build_mix, parse_source
        from .paths import check_output_paths

        sources = [parse_source(x) for x in a.sources]
        inputs = [path for path, _ in sources]
        inputs.extend(p for p in (a.tokenizer, a.chat_template_file) if p is not None)
        outputs = [a.out] + ([a.holdout] if a.holdout is not None else [])
        check_output_paths(inputs=inputs, outputs=outputs)
        template_requested = a.chat_template or a.chat_template_file is not None
        if template_requested and a.tokenizer is None:
            raise ValueError(
                "--chat-template / --chat-template-file requires --tokenizer"
            )
        if a.holdout is not None and a.holdout_share <= 0:
            raise ValueError("--holdout requires --holdout-share greater than zero")
        count = render_chat = tokenizer_id = template_id = None
        if a.tokenizer is not None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
            tokenizer_id = (
                str(a.tokenizer.resolve()) if a.tokenizer.exists() else tok.name_or_path
            )

            def count(text):
                return len(tok(text, add_special_tokens=False)["input_ids"])

            if template_requested:
                import hashlib

                template = tok.get_chat_template(
                    a.chat_template_file.read_text(encoding="utf-8")
                    if a.chat_template_file is not None
                    else None
                )
                if not template:
                    raise ValueError("the selected chat template is empty")
                template_id = (
                    "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
                )

                def render_chat(messages):
                    return tok.apply_chat_template(
                        messages,
                        chat_template=template,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

        share = a.holdout_share if a.holdout is not None else 0.0
        text, held, rep = build_mix(
            sources,
            a.tokens,
            count,
            share,
            a.holdout_tokens,
            render_chat=render_chat,
            tokenizer_id=tokenizer_id,
            chat_template_id=template_id,
            min_holdout_tokens=a.min_holdout_tokens,
        )
        a.out.write_text(text, encoding="utf-8")
        if a.holdout is not None:
            a.holdout.write_text(held, encoding="utf-8")
        return rep
    if a.cmd == "size":
        from .size import size_report

        return size_report(a.dst, a.src, a.by)
    if a.cmd == "cache":
        out = cache_command(a)
    elif a.cmd == "restack":
        from .stack import restack

        out = restack(a.dst, a.out, a.layout, log=lambda m: print(m, file=sys.stderr))
        for k in ("before", "after"):
            out[k].pop("per_shard")
    elif a.cmd == "export-dense":
        from .checkpoint import export_dense
        from .paths import check_output_paths

        check_output_paths(inputs=[a.src, a.dst], outputs=[a.out])
        export_dense(a.src, a.dst, a.out)
        out = {"exported": str(a.out)}
    elif a.cmd == "eval":
        from .evalq import evaluate

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
        from .calibrate import SELECTABLE, Config
        from .checkpoint import DEFAULT_EXCLUDE
        from .pipeline import RunConfig, plan, run
        from .qad import make_config

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
            exclude=DEFAULT_EXCLUDE if a.exclude is None else a.exclude,
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
        if a.target_kld is not None:
            rc = _tune_rc(a, rc)
            if a.dry_run:
                return rc
        out = plan(rc, log=progress) if a.dry_run else run(rc, log=progress)
    elif a.cmd == "verify":
        from .checkpoint import verify_checkpoint

        out = verify_checkpoint(a.src, a.dst, a.max_tensors)
    else:
        raise ValueError(f"unknown command: {a.cmd}")
    return out


def _artifacts(a, out=None) -> list[dict]:
    """Reference produced artifacts, never infer successful internal compute phases."""
    artifacts = []
    if out is not None:
        if a.cmd == "calib":
            artifacts.append({"kind": "calibration", "path": str(a.out)})
            if a.holdout is not None:
                artifacts.append({"kind": "holdout", "path": str(a.holdout)})
        elif a.cmd in ("export-dense", "export-portable", "restack"):
            artifacts.append({"kind": "checkpoint", "path": str(a.out)})
        elif a.cmd == "quantize" and not a.dry_run:
            artifacts.extend(
                {"kind": "checkpoint", "path": path}
                for path in out.get("outputs", {str(a.dst): out})
            )
    for name in ("tune_report", "tune_diagnostics"):
        path = getattr(a, name, None)
        if path is not None:
            artifacts.append({"kind": name, "path": path})
    return artifacts


def _success_result(a, out: dict) -> dict:
    """Schema 1 adds metadata without moving existing result fields."""
    return {
        **out,
        "schema_version": RESULT_SCHEMA_VERSION,
        "command": a.cmd,
        "status": out.get("status", "ok"),
        "phases": out.get("phases", [{"phase": a.cmd, "status": "completed"}]),
        "artifacts": out.get("artifacts", []) + _artifacts(a, out),
    }


def _report_error(command, exc, *, json_mode, exit_code, args=None) -> int:
    message = str(exc)
    kind = "UsageError" if isinstance(exc, _UsageError) else type(exc).__name__
    from .errors import describe_error

    category, classified_code, remedy = describe_error(exc)
    exit_code = classified_code
    if isinstance(exc, _UsageError):
        category, exit_code = "invalid_input", 2
    if isinstance(exc, ModuleNotFoundError):
        message = (
            f"{exc}. The {command or 'requested'} command needs dependency "
            f"{exc.name!r}. Install hi2 and its declared dependencies in the Python "
            f"environment {sys.executable}; see docs/usage.md. Help/version do not "
            "require compute dependencies."
        )
    elif isinstance(exc, KeyboardInterrupt):
        message = "interrupted by user"
    elif isinstance(exc, SystemExit) and not isinstance(exc.code, str):
        message = f"command exited with status {exc.code}"
    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
    print(f"hi2: {message}", file=sys.stderr)
    print(f"Remedy: {remedy}", file=sys.stderr)
    if json_mode:
        print(
            json.dumps(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "command": command,
                    "status": status,
                    "error": {
                        "type": kind,
                        "category": category,
                        "message": message,
                        "exit_code": exit_code,
                        "remedy": remedy,
                    },
                    **({"quality": exc.report} if hasattr(exc, "report") else {}),
                    "phases": [{"phase": command, "status": status}] if command else [],
                    "artifacts": _artifacts(args) if args is not None else [],
                    "workdir": str(args.workdir)
                    if args is not None and getattr(args, "workdir", None)
                    else None,
                },
                indent=2,
                allow_nan=False,
            )
        )
    return exit_code


def main(argv=None) -> int:
    """CLI transport: one JSON object with --json; all execution logs on stderr.

    Successful JSON results keep their original top-level fields and add schema 1
    metadata. Failures carry error.type/message/exit_code; usage errors exit 2,
    command failures exit 1 (or the core's explicit nonzero code), interrupts 130.
    Human mode retains command summaries and the size table. Help/version are text.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    options = argv[: argv.index("--")] if "--" in argv else argv
    json_mode = "--json" in options
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except _UsageError as exc:
        print(ap.format_usage(), file=sys.stderr, end="")
        return _report_error(exc.command, exc, json_mode=json_mode, exit_code=2)
    json_mode = a.json
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    try:
        with _stderr_logs():
            out = dispatch_command(a)
            if a.cmd == "size" and not json_mode:
                from .size import format_report

                rendered = format_report(out, a.top)
            else:
                rendered = json.dumps(
                    _success_result(a, out) if json_mode else out,
                    indent=2,
                    allow_nan=False,
                )
    except KeyboardInterrupt as exc:
        return _report_error(a.cmd, exc, json_mode=json_mode, exit_code=130, args=a)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) and exc.code != 0 else 1
        return _report_error(a.cmd, exc, json_mode=json_mode, exit_code=code, args=a)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        if a.debug:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return _report_error(a.cmd, exc, json_mode=json_mode, exit_code=1, args=a)
    except Exception as exc:
        if a.debug:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return _report_error(a.cmd, exc, json_mode=json_mode, exit_code=1, args=a)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
