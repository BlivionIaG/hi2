# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The three phases: tabulate (calibration), allocate (global), encode (write)."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch

from .allocate import allocate_by_quality, allocate_global
from .calibrate import Config, TensorTable, build_table
from .checkpoint import (
    DEFAULT_EXCLUDE,
    is_eligible,
    make_resolver,
    weight_map,
    write_checkpoint,
)
from .executor import Executor
from .moe import expert_parts, expert_weights, group_key, probe_units

DEFAULT_ERROR_RATIO = 4.0  # placeholder until measured (see docs)


@dataclass
class RunConfig:
    src: Path
    dst: Path
    workdir: Path
    devices: list[str] = field(default_factory=lambda: ["cpu"])
    calib: Path | None = None  # None = data-free (unless calib_ids is given)
    calib_ids: torch.Tensor | None = None  # [samples, seq_len], skips tokenising
    calib_samples: int = 128
    seq_len: int = 2048
    target_bpw: float | None = None  # model-wide bits per weight, or
    max_error_ratio: float | None = None  # quality target (INT2 where it matters less)
    include: str | None = None
    exclude: str = DEFAULT_EXCLUDE
    limit_layers: int | None = None
    compute_dtype: str = "float32"
    cfg: Config = field(default_factory=Config)
    # spec 8.6: none | probe (measure each tensor's KL cost and weight the allocation)
    sensitivity: str = "none"
    sens_seqs: int = 8
    sens_len: int = 512
    sens_floor: float = 0.05
    sens_dtype: str = (
        "float32"  # dtype of the whole-model probe (float16 if fp32 does not fit)
    )
    # MoE: lower bound of an expert's routed-token share (relative to uniform routing)
    # in its allocation weight
    expert_share_floor: float = 0.05


def load_calibration_ids(
    path: Path, tokenizer, samples: int, seq_len: int
) -> torch.Tensor:
    """``[samples, seq_len]`` token ids from a .txt file or a .jsonl with ``text``."""
    if path.suffix == ".jsonl":
        text = "\n\n".join(
            json.loads(ln)["text"] for ln in path.read_text().splitlines() if ln
        )
    else:
        text = path.read_text()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = samples * seq_len
    if ids.numel() < need:
        raise SystemExit(
            f"calibration file has {ids.numel()} tokens, need {need} "
            f"({samples} x {seq_len}); lower --calib-samples or --seq-len"
        )
    return ids[:need].reshape(samples, seq_len)


def _phase1_key(rc: RunConfig) -> dict:
    """Settings that determine the phase-1 tables; a workdir whose ``phase1.json``
    matches them is reused (e.g. one calibration for several allocation targets)."""
    c = rc.cfg
    return {
        "src": str(rc.src),
        "calib": str(rc.calib),
        "calib_samples": rc.calib_samples,
        "seq_len": rc.seq_len,
        "include": rc.include,
        "exclude": rc.exclude,
        "limit_layers": rc.limit_layers,
        "compute_dtype": rc.compute_dtype,
        "candidates": [list(x) for x in c.candidates],
        "rotation": c.rotation,
        "smooth": c.smooth,
        "rounding": c.rounding,
        "sequential": c.sequential,
        "effort": c.effort,
        "expert_min_tokens": c.expert_min_tokens,
    }


def _load_model(rc: RunConfig, log=print):
    """The HF model in RAM, with fused 3-D experts replaced by per-expert Linears."""
    from transformers import AutoModelForCausalLM

    from .moe import check_unfused, unfuse_experts

    model = AutoModelForCausalLM.from_pretrained(
        rc.src, dtype=getattr(torch, rc.compute_dtype), low_cpu_mem_usage=True
    ).eval()
    if unfuse_experts(model):
        check_unfused(model, rc.src, make_resolver(weight_map(rc.src)), log)
    return model


def phase1_tables(rc: RunConfig, ex: Executor, log=print) -> dict[str, TensorTable]:
    inc = re.compile(rc.include) if rc.include else None
    exc = re.compile(rc.exclude)
    tdir = rc.workdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    if rc.calib is None and rc.calib_ids is None:
        return _tabulate_streaming(rc, ex, inc, exc, tdir, log)
    marker = rc.workdir / "phase1.json"
    if rc.calib_ids is None and marker.exists():
        done = json.loads(marker.read_text())
        hess_ok = rc.cfg.rounding != "gptq" or (rc.workdir / "hessian").is_dir()
        if done.get("key") == _phase1_key(rc) and hess_ok:
            log(f"phase 1: reusing {len(done['tensors'])} tables from {rc.workdir}")
            return {
                n: torch.load(tdir / (n.replace("/", "_") + ".pt"), weights_only=False)
                for n in done["tensors"]
            }
    from transformers import AutoTokenizer

    from .layerwise import calibrate_layerwise

    if rc.calib_ids is not None:
        ids = rc.calib_ids
    else:
        tok = AutoTokenizer.from_pretrained(rc.src)
        ids = load_calibration_ids(rc.calib, tok, rc.calib_samples, rc.seq_len)
    model = _load_model(rc, log)
    wmap = weight_map(rc.src)
    tables = calibrate_layerwise(
        model,
        ids,
        rc.cfg,
        ex,
        rc.workdir,
        inc,
        exc,
        rc.limit_layers,
        log,
        make_resolver(wmap),
    )
    missing = [n for n in tables if n not in wmap]
    if missing:
        raise SystemExit(f"module names not found in the checkpoint: {missing[:3]}")
    for name, t in tables.items():
        torch.save(t, tdir / (name.replace("/", "_") + ".pt"))
    marker.write_text(json.dumps({"key": _phase1_key(rc), "tensors": list(tables)}))
    return tables


def _tabulate_streaming(rc, ex, inc, exc, tdir, log) -> dict[str, TensorTable]:
    """Data-free: stream tensors from the checkpoint (resumable per tensor)."""
    from safetensors import safe_open

    wmap = weight_map(rc.src)
    tables: dict[str, TensorTable] = {}
    for shard in sorted(set(wmap.values())):
        with safe_open(rc.src / shard, "pt") as f:
            for name in [n for n, s in wmap.items() if s == shard]:
                sl = f.get_slice(name)
                if not is_eligible(
                    name, sl.get_shape(), _dtype(sl.get_dtype()), inc, exc
                ):
                    continue
                path = tdir / (name.replace("/", "_") + ".pt")
                if path.exists():
                    tables[name] = torch.load(path, weights_only=False)
                    continue
                t = build_table(name, f.get_tensor(name), None, rc.cfg, ex)
                torch.save(t, path)
                tables[name] = t
                log(f"tabulated {name} {t.shape}")
    return tables


def _probe_ids(rc) -> torch.Tensor:
    """First ``sens_seqs`` sequences of the calibration data, cut to ``sens_len``."""
    from transformers import AutoTokenizer

    if rc.calib_ids is not None:
        ids = rc.calib_ids
    else:
        ids = load_calibration_ids(
            rc.calib, AutoTokenizer.from_pretrained(rc.src), rc.sens_seqs, rc.seq_len
        )
    return ids[: rc.sens_seqs, : rc.sens_len]


def _sequential_encode(rc, ex, tables, picks, log) -> dict:
    from transformers import AutoTokenizer

    from .layerwise import encode_layerwise

    if rc.calib_ids is not None:
        ids = rc.calib_ids
    else:
        ids = load_calibration_ids(
            rc.calib,
            AutoTokenizer.from_pretrained(rc.src),
            rc.calib_samples,
            rc.seq_len,
        )
    model = _load_model(rc, log)
    inc = re.compile(rc.include) if rc.include else None
    return encode_layerwise(
        model,
        ids,
        rc.cfg,
        ex,
        tables,
        picks,
        inc,
        re.compile(rc.exclude),
        rc.limit_layers,
        log,
        make_resolver(weight_map(rc.src)),
    )


def _dtype(s: str) -> torch.dtype:
    return {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}.get(
        s, torch.int8
    )


def summarize(report: dict[str, dict]) -> dict:
    params = sum(r["shape"][0] * r["shape"][1] for r in report.values())
    bpw = sum(r["bpw"] * r["shape"][0] * r["shape"][1] for r in report.values()) / max(
        params, 1
    )
    rows = sum(r["shape"][0] for r in report.values())
    int4 = sum(r["frac_int4"] * r["shape"][0] for r in report.values()) / max(rows, 1)
    by_kind: dict[str, list[float]] = {}
    for name, r in report.items():
        kind = name.rsplit(".", 2)[-2]
        by_kind.setdefault(kind, []).append(r["bpw"])
    return {
        "tensors": len(report),
        "params": params,
        "avg_bpw": bpw,
        "frac_int4_rows": int4,
        "bpw_by_projection": {k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
    }


def _peak_memory(ex: Executor) -> dict:
    """Peak resident memory of this process and peak allocated memory per GPU (GiB)."""
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    out = {"host_rss": round(rss / 2**30, 2)}
    for d in ex.devices:
        if d.type == "cuda":
            out[str(d)] = round(torch.accelerator.max_memory_allocated(d) / 2**30, 2)
    return out


def _expert_summary(report: dict[str, dict]) -> dict:
    """Bpw and INT4 row share of the expert tensors against the rest."""

    def agg(items):
        params = sum(r["shape"][0] * r["shape"][1] for r in items)
        rows = sum(r["shape"][0] for r in items)
        return {
            "tensors": len(items),
            "params": params,
            "avg_bpw": sum(r["bpw"] * r["shape"][0] * r["shape"][1] for r in items)
            / max(params, 1),
            "frac_int4_rows": sum(r["frac_int4"] * r["shape"][0] for r in items)
            / max(rows, 1),
        }

    ex = [r for n, r in report.items() if expert_parts(n) is not None]
    dense = [r for n, r in report.items() if expert_parts(n) is None]
    return {"expert": agg(ex), "non_expert": agg(dense)}


def run(rc: RunConfig, log=print) -> dict:
    t_start = time.time()
    rc.workdir.mkdir(parents=True, exist_ok=True)
    ex = Executor(rc.devices)
    calibrated = rc.calib is not None or rc.calib_ids is not None
    if rc.cfg.rounding == "gptq" and not calibrated:
        log("no calibration data: using nearest rounding")
        rc.cfg.rounding = "nearest"
        rc.cfg.sequential = False
    log(
        f"phase 1: tabulating on {[str(d) for d in ex.devices]}"
        f"{'' if calibrated else ' (data-free)'}"
    )
    tables = phase1_tables(rc, ex, log)
    t_phase1 = time.time()
    # MoE: all experts of a layer and projection weigh as one dense tensor, split by
    # routed-token share (empty for dense models)
    weights = expert_weights(tables, rc.expert_share_floor)
    if weights:
        log(
            f"MoE: {len(weights)} expert tensors in "
            f"{len({group_key(n) for n in weights})} layer/projection groups"
        )
    sens = dict(weights) or None
    if rc.sensitivity == "probe":
        if not calibrated:
            raise SystemExit("--sensitivity probe needs calibration data (--calib)")
        from .sensitivity import probe_sensitivity

        skey = {
            "phase1": _phase1_key(rc),
            "seqs": rc.sens_seqs,
            "len": rc.sens_len,
            "floor": rc.sens_floor,
            "dtype": rc.sens_dtype,
            "share_floor": rc.expert_share_floor,
        }
        spath, kpath = (
            rc.workdir / "sensitivity.json",
            rc.workdir / "sensitivity.key.json",
        )
        if spath.exists() and kpath.exists() and json.loads(kpath.read_text()) == skey:
            log("phase 2a: reusing the sensitivity probe of this workdir")
            sens = json.loads(spath.read_text())
        else:
            log("phase 2a: probing per-tensor sensitivity")
            sens = probe_sensitivity(
                rc.src,
                tables,
                rc.workdir,
                _probe_ids(rc),
                rc.cfg,
                ex,
                floor=rc.sens_floor,
                log=log,
                units=probe_units(tables, weights),
                dtype=getattr(torch, rc.sens_dtype),
            )
            spath.write_text(json.dumps(sens, indent=1))
            kpath.write_text(json.dumps(skey))
    if rc.target_bpw is not None and rc.max_error_ratio is not None:
        raise ValueError("give either target_bpw or max_error_ratio, not both")
    if rc.target_bpw is None:
        ratio = (
            DEFAULT_ERROR_RATIO if rc.max_error_ratio is None else rc.max_error_ratio
        )
        log(
            f"phase 2: quality allocation over {len(tables)} tensors, "
            f"weighted error <= {ratio:g} x the best achievable"
        )
        alloc = allocate_by_quality(list(tables.values()), ratio, sens)
        if not alloc.feasible:
            log("  warning: ratio below 1 is unreachable; using the most accurate")
    else:
        log(
            f"phase 2: global allocation over {len(tables)} tensors, "
            f"target {rc.target_bpw} bpw"
        )
        alloc = allocate_global(list(tables.values()), rc.target_bpw, sens)
        if not alloc.feasible:
            log(
                "  warning: target is below the cheapest configuration; using the cheapest"
            )
    mix = ", ".join(
        f"INT{nb} {100 * f:.1f}%"
        for nb, f in sorted((alloc.frac_by_bits or {}).items())
    )
    log(
        f"  allocation bpw {alloc.bpw:.4f} error ratio {alloc.error_ratio:.3f} "
        f"rows: {mix}"
    )
    pre = None
    if rc.cfg.rounding == "gptq" and rc.cfg.sequential:
        log("phase 3a: sequential encode pass (spec 8.4)")
        pre = _sequential_encode(rc, ex, tables, alloc.picks, log)
    log("phase 3: encoding and writing")
    meta = {
        "mode": "flex",
        "target_bpw": rc.target_bpw,
        "max_error_ratio": rc.max_error_ratio,
        "allocation_error_ratio": alloc.error_ratio,
        "calibrated": calibrated,
        "rounding": rc.cfg.rounding,
        "sequential": rc.cfg.sequential,
        "smooth": rc.cfg.smooth,
        "rotation": rc.cfg.rotation,
        "calib_tokens": rc.calib_samples * rc.seq_len if calibrated else 0,
    }
    report = write_checkpoint(
        rc.src, rc.dst, rc.workdir, tables, alloc.picks, rc.cfg, ex, meta, log, pre
    )
    summary = summarize(report) | {
        "allocation_bpw": alloc.bpw,
        "feasible": alloc.feasible,
        "allocation_error_ratio": alloc.error_ratio,
        "frac_rows_by_bits": alloc.frac_by_bits,
        "runtime_s": {
            "phase1": round(t_phase1 - t_start, 1),
            "total": round(time.time() - t_start, 1),
        },
        "peak_memory_gib": _peak_memory(ex),
    }
    tok = rc.workdir / "expert_tokens.json"
    if weights:
        summary["experts"] = _expert_summary(report)
        if tok.exists():
            cov = json.loads(tok.read_text())
            summary["experts"]["fallback_tensors"] = len(cov["fallback"])
            summary["experts"]["coverage"] = cov["coverage"]
    (rc.dst / "hi2_report.json").write_text(
        json.dumps({"summary": summary, "tensors": report}, indent=2)
    )
    return summary
