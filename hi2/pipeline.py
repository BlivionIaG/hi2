# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The three phases: tabulate (calibration), allocate (global), encode (write)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch

from .allocate import allocate_global
from .calibrate import Config, TensorTable, build_table
from .checkpoint import (
    DEFAULT_EXCLUDE,
    is_eligible,
    make_resolver,
    weight_map,
    write_checkpoint,
)
from .executor import Executor


@dataclass
class RunConfig:
    src: Path
    dst: Path
    workdir: Path
    devices: list[str] = field(default_factory=lambda: ["cpu"])
    calib: Path | None = None  # None = data-free (unless calib_ids is given)
    calib_ids: torch.Tensor | None = None  # [samples, seq_len], skips tokenising
    calib_samples: int = 64
    seq_len: int = 2048
    target_bpw: float = 3.0
    include: str | None = None
    exclude: str = DEFAULT_EXCLUDE
    limit_layers: int | None = None
    compute_dtype: str = "float32"
    cfg: Config = field(default_factory=Config)


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


def phase1_tables(rc: RunConfig, ex: Executor, log=print) -> dict[str, TensorTable]:
    inc = re.compile(rc.include) if rc.include else None
    exc = re.compile(rc.exclude)
    tdir = rc.workdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    if rc.calib is None and rc.calib_ids is None:
        return _tabulate_streaming(rc, ex, inc, exc, tdir, log)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .layerwise import calibrate_layerwise

    if rc.calib_ids is not None:
        ids = rc.calib_ids
    else:
        tok = AutoTokenizer.from_pretrained(rc.src)
        ids = load_calibration_ids(rc.calib, tok, rc.calib_samples, rc.seq_len)
    model = AutoModelForCausalLM.from_pretrained(
        rc.src, dtype=getattr(torch, rc.compute_dtype), low_cpu_mem_usage=True
    ).eval()
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


def _sequential_encode(rc, ex, tables, picks, log) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model = AutoModelForCausalLM.from_pretrained(
        rc.src, dtype=getattr(torch, rc.compute_dtype), low_cpu_mem_usage=True
    ).eval()
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


def run(rc: RunConfig, log=print) -> dict:
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
    log(
        f"phase 2: global allocation over {len(tables)} tensors, target {rc.target_bpw} bpw"
    )
    alloc = allocate_global(list(tables.values()), rc.target_bpw)
    log(f"  allocation bpw {alloc.bpw:.4f} feasible={alloc.feasible}")
    if not alloc.feasible:
        log("  warning: target is below the cheapest configuration; using the cheapest")
    pre = None
    if rc.cfg.rounding == "gptq" and rc.cfg.sequential:
        log("phase 3a: sequential encode pass (spec 8.4)")
        pre = _sequential_encode(rc, ex, tables, alloc.picks, log)
    log("phase 3: encoding and writing")
    meta = {
        "mode": "flex",
        "target_bpw": rc.target_bpw,
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
    }
    (rc.dst / "hi2_report.json").write_text(
        json.dumps({"summary": summary, "tensors": report}, indent=2)
    )
    return summary
