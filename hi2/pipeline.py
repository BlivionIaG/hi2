# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The three phases: tabulate (calibration), allocate (global), encode (write)."""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch

from .allocate import allocate_by_quality, allocate_global
from .cache import StatsView
from .calibrate import Config, TensorTable, build_table
from .checkpoint import (
    DEFAULT_EXCLUDE,
    Output,
    is_eligible,
    make_resolver,
    weight_map,
    write_checkpoints,
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
    # several quality targets from one phase 1 / probe, written to <dst>-r<ratio>
    max_error_ratios: list[float] | None = None
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
    sens_batch: int = 1  # probe sequences per forward (1 = the original probe, exact)
    sens_dtype: str = (
        "float32"  # dtype of the whole-model probe (float16 if fp32 does not fit)
    )
    # MoE: lower bound of an expert's routed-token share (relative to uniform routing)
    # in its allocation weight
    expert_share_floor: float = 0.05
    # persistent cache of statistics, tables and probe results (hi2.cache); None keeps
    # everything in the workdir as before
    cache_dir: Path | None = None
    sigma_fp16_above: int = 0  # cache Sigma in fp16 for K above this (0 = always fp32)
    # layer-wise quantization-aware distillation after the allocation (spec 13); None = off
    qad: object | None = None  # hi2.qad.QadConfig
    # routed experts on disk: per-expert (one HI2 tensor per expert projection) or
    # stacked (one per layer and projection group, hi2.stack); same weights either way
    moe_layout: str = "stacked"
    # experiment (docs/lowbit.md): the N most sensitive tensors all-INT8, the
    # others INT2/INT4 (needs INT8 candidates); 0 = off
    int8_force: int = 0


def calib_paths(path) -> list[Path]:
    """One path, a list of paths, or a comma-separated string of paths."""
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    return [Path(p) for p in str(path).split(",") if p]


def read_calibration_text(path) -> str:
    """Text of one or several .txt / .jsonl (``text`` field) files, concatenated."""
    parts = []
    for p in calib_paths(path):
        if p.suffix == ".jsonl":
            parts.append(
                "\n\n".join(
                    json.loads(ln)["text"] for ln in p.read_text().splitlines() if ln
                )
            )
        else:
            parts.append(p.read_text())
    return "\n\n".join(parts)


def load_calibration_ids(path, tokenizer, samples: int, seq_len: int) -> torch.Tensor:
    """``[samples, seq_len]`` token ids from .txt files or .jsonl files with ``text``
    (several files are concatenated in the order given)."""
    text = read_calibration_text(path)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = samples * seq_len
    if ids.numel() < need:
        raise SystemExit(
            f"calibration file has {ids.numel()} tokens, need {need} "
            f"({samples} x {seq_len}); lower --calib-samples or --seq-len"
        )
    return ids[:need].reshape(samples, seq_len)


def _windows(text, tokenizer, samples, seq_len, gen, what) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_win = ids.numel() // seq_len
    if n_win < samples:
        raise SystemExit(
            f"{what} has {ids.numel()} tokens = {n_win} windows of {seq_len}, need "
            f"{samples}; add text or lower --qad-samples / --seq-len"
        )
    pick = torch.randperm(n_win, generator=gen)[:samples]
    return ids[: n_win * seq_len].reshape(n_win, seq_len)[pick.sort().values]


def sample_calibration_windows(
    path, tokenizer, samples: int, seq_len: int, seed: int = 0, balanced: bool = False
) -> torch.Tensor:
    """``[samples, seq_len]`` token ids: non-overlapping windows drawn uniformly (seeded)
    from the whole calibration text, so every domain of a concatenated file is
    represented, not only its head. ``balanced`` draws the same number of windows from
    each calibration file instead (domain balance regardless of file sizes)."""
    gen = torch.Generator().manual_seed(seed)
    paths = calib_paths(path)
    if not balanced or len(paths) == 1:
        text = read_calibration_text(path)
        return _windows(text, tokenizer, samples, seq_len, gen, "calibration text")
    parts = []
    for i, p in enumerate(paths):
        n = samples // len(paths) + (1 if i < samples % len(paths) else 0)
        text = read_calibration_text(p)
        parts.append(_windows(text, tokenizer, n, seq_len, gen, str(p)))
    return torch.cat(parts)


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


def _calib_ids(rc: RunConfig) -> torch.Tensor:
    if rc.calib_ids is not None:
        return rc.calib_ids
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(rc.src)
    return load_calibration_ids(rc.calib, tok, rc.calib_samples, rc.seq_len)


def _save_tables(tables: dict, tdir: Path) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        torch.save(t, tdir / (name.replace("/", "_") + ".pt"))


def _load_tables(tdir: Path, names: list[str]) -> dict[str, TensorTable]:
    return {
        n: torch.load(tdir / (n.replace("/", "_") + ".pt"), weights_only=False)
        for n in names
    }


def phase1_tables(rc: RunConfig, ex: Executor, log=print) -> dict[str, TensorTable]:
    inc = re.compile(rc.include) if rc.include else None
    exc = re.compile(rc.exclude)
    tdir = rc.workdir / "tables"
    if rc.calib is None and rc.calib_ids is None:
        tdir.mkdir(parents=True, exist_ok=True)
        return _tabulate_streaming(rc, ex, inc, exc, tdir, log)
    if rc.cache_dir is not None:
        return _phase1_cached(rc, ex, inc, exc, log)
    marker = rc.workdir / "phase1.json"
    if rc.calib_ids is None and marker.exists():
        done = json.loads(marker.read_text())
        hess_ok = rc.cfg.rounding != "gptq" or (rc.workdir / "hessian").is_dir()
        if done.get("key") == _phase1_key(rc) and hess_ok:
            log(f"phase 1: reusing {len(done['tensors'])} tables from {rc.workdir}")
            return _load_tables(tdir, done["tensors"])
    from .layerwise import calibrate_layerwise

    ids = _calib_ids(rc)
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
    _save_tables(tables, tdir)
    marker.write_text(json.dumps({"key": _phase1_key(rc), "tensors": list(tables)}))
    return tables


def _phase1_cached(
    rc: RunConfig, ex: Executor, inc, exc, log
) -> dict[str, TensorTable]:
    """Phase 1 through the persistent cache: statistics (forward passes) are reused for
    any run with the same model, calibration tokens and placement, tables for the same
    phase-1 settings on top. The workdir only records which entries it used."""
    from . import cache as C
    from .layerwise import CacheSink, calibrate_layerwise, tables_from_stats

    root = Path(rc.cache_dir)
    ids = _calib_ids(rc)
    want_h = rc.cfg.rounding == "gptq" and not rc.cfg.sequential
    need = "hessian" if want_h else None
    sent = C.Entry(root, C.stats_key(rc, ids, ex.devices))
    tent = C.Entry(root, C.tables_key(sent.key, rc.cfg))
    tables = None
    if not sent.complete(need):
        with sent.building():
            if not sent.complete(need):
                log(
                    f"cache: statistics miss ({sent.path}), running the calibration pass"
                )
                C.check_space(root, C.estimate_stats_bytes(rc.src, inc, exc), log)
                sent.begin()
                try:
                    model = _load_model(rc, log)
                    wmap = weight_map(rc.src)
                    sink = CacheSink(sent.work, rc.sigma_fp16_above)
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
                        sink,
                    )
                    del model
                    missing = [n for n in tables if n not in wmap]
                    if missing:
                        raise SystemExit(
                            f"module names not found in the checkpoint: {missing[:3]}"
                        )
                    sink.flush(wait=True)
                    sent.finish({"hessian": want_h, "tensors": len(tables)})
                except BaseException:
                    sent.abort()
                    raise
                with tent.building():  # fresh tables: store them under the stats lock
                    tent.begin()
                    _save_tables(tables, tent.work / "tables")
                    tent.finish({"tensors": list(tables)})
    else:
        log(f"cache: statistics hit ({sent.path})")
        sent.touch()
    if tables is not None:
        pass
    elif tent.complete():
        log(f"cache: tables hit ({tent.path})")
        tent.touch()
        tables = _load_tables(tent.path / "tables", tent.info()["tensors"])
    else:
        with tent.building():
            if tent.complete():
                tables = _load_tables(tent.path / "tables", tent.info()["tensors"])
            else:
                log(f"cache: tables miss ({tent.path}), tabulating from the statistics")
                tent.begin()
                try:
                    tables = tables_from_stats(
                        rc.src, sent.path, rc.cfg, ex, rc.compute_dtype, log
                    )
                    _save_tables(tables, tent.work / "tables")
                    tent.finish({"tensors": list(tables)})
                except BaseException:
                    tent.abort()
                    raise
    C.write_ref(rc.workdir, stats=sent.path, tables=tent.path)
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


def _qad_encode(
    rc, ex, tables, picks, sens, log, target_bpw=None, ratio=None, tag=""
) -> dict:
    """Phase 3q: layer-wise QAD (spec 13); returns ``{name: HI2Tensor}``. ``tag`` names
    the output in multi-output mode (own per-layer artifacts ``workdir/qad<tag>`` and
    report ``workdir/qad_report<tag>.json``)."""
    from transformers import AutoTokenizer

    from .qad import fingerprint, qad_fingerprint_fields, qad_layerwise

    qc = rc.qad
    n_seq = qc.samples + qc.val
    if rc.calib_ids is not None:
        ids = rc.calib_ids
        if ids.shape[0] < n_seq:
            raise SystemExit(
                f"QAD needs {n_seq} calibration sequences (samples + val), got {ids.shape[0]}"
            )
        ids = ids[:n_seq]
    elif qc.sampling == "head":
        ids = load_calibration_ids(
            rc.calib, AutoTokenizer.from_pretrained(rc.src), n_seq, rc.seq_len
        )
    else:
        bal = qc.sampling == "balanced"
        ids = sample_calibration_windows(
            rc.calib,
            AutoTokenizer.from_pretrained(rc.src),
            n_seq,
            rc.seq_len,
            qc.seed,
            bal,
        )
        how = "in equal shares per calibration file" if bal else "from the whole text"
        log(
            f"QAD data: {n_seq} random non-overlapping {rc.seq_len}-token windows {how}"
        )
    model = _load_model(rc, log)
    fp = fingerprint(
        {
            "phase1": _phase1_key(rc),
            "target_bpw": target_bpw,
            "max_error_ratio": ratio,
            "sens": sens,
            "picks": fingerprint({k: v.tolist() for k, v in sorted(picks.items())}),
            "qad": qad_fingerprint_fields(qc),
            "n_seq": n_seq,
            **({"int8_force": rc.int8_force} if rc.int8_force else {}),
        }
    )
    inc = re.compile(rc.include) if rc.include else None
    out = qad_layerwise(
        model,
        ids,
        tables,
        picks,
        rc.cfg,
        qc,
        ex,
        rc.workdir,
        inc,
        re.compile(rc.exclude),
        make_resolver(weight_map(rc.src)),
        fp,
        rc.limit_layers,
        log,
        qdir=rc.workdir / f"qad{tag}",
    )
    (rc.workdir / f"qad_report{tag}.json").write_text(json.dumps(out.layers, indent=1))
    tensors, dense = out.tensors, out.dense
    if qc.final_kl > 0:
        tensors, dense = _final_kl(rc, ex, model, ids, tensors, dense, fp, log, tag)
    return tensors, dense


def _final_kl(rc, ex, model, ids, tensors, dense, fp, log, tag=""):
    """Phase 3k: the final logit-KL phase (:mod:`hi2.finalkl`) on top of the layer-wise
    result, resumable from ``workdir/qad<tag>/final_kl.pt``."""
    from dataclasses import asdict

    from .finalkl import final_kl
    from .qad import FINAL_FIELDS, fingerprint, split_indices

    qc = rc.qad
    ffp = fingerprint(
        {
            "layerwise": fp,
            "final": {k: v for k, v in asdict(qc).items() if k in FINAL_FIELDS},
            "warmup": qc.warmup,
            "seed": qc.seed,
        }
    )
    art = rc.workdir / f"qad{tag}" / "final_kl.pt"
    if art.exists():
        rec = torch.load(art, weights_only=False)
        if rec.get("fingerprint") == ffp:
            log(f"final KL: reusing {art.name}")
            return rec["tensors"], rec["dense"]
    log(f"phase 3k: final logit-KL phase, {qc.final_kl} steps")
    train_idx, val_idx = split_indices(ids.shape[0], qc)
    t, d, info = final_kl(
        model,
        ids,
        train_idx,
        val_idx,
        tensors,
        dense,
        qc,
        ex.primary,
        make_resolver(weight_map(rc.src)),
        log,
    )
    torch.save({"fingerprint": ffp, "tensors": t, "dense": d, "info": info}, art)
    (rc.workdir / f"final_kl{tag}.json").write_text(
        json.dumps({k: v for k, v in info.items()}, indent=1)
    )
    return t, d


def _text_config(src: Path) -> dict:
    cfg = json.loads((Path(src) / "config.json").read_text())
    return cfg.get("text_config", cfg)


def check_inputs(rc: RunConfig, log=print) -> dict:
    """Fail early (before hours of work) on a missing model, too little calibration
    text, or an impossible QAD setting; returns the facts the plan prints."""
    if not (Path(rc.src) / "config.json").exists():
        raise SystemExit(
            f"{rc.src} is not a Hugging Face model directory (no config.json)"
        )
    calibrated = rc.calib is not None or rc.calib_ids is not None
    if rc.qad is not None and not calibrated:
        log(
            "warning: QAD needs calibration text (--calib FILE[,FILE...]); this data-free "
            "run falls back to PTQ only (experiments/build_calib_mix.py builds a "
            "mixed-domain calibration file)"
        )
        rc.qad = None
    if rc.qad is not None and rc.cfg.sequential:
        raise SystemExit("QAD already runs on the quantized prefix; drop --sequential")
    facts: dict = {"calibrated": calibrated}
    need = rc.calib_samples * rc.seq_len if calibrated else 0
    if rc.qad is not None:
        need = max(need, (rc.qad.samples + rc.qad.val) * rc.seq_len)
    if rc.calib is not None:
        for p in calib_paths(rc.calib):
            if not p.exists():
                raise SystemExit(f"calibration file {p} does not exist")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(rc.src)
        have = tok(
            read_calibration_text(rc.calib),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.shape[1]
        facts["calib_tokens_available"] = have
        if have < need:
            what = (
                "--qad-samples (+ held-out)"
                if rc.qad is not None
                else "--calib-samples"
            )
            raise SystemExit(
                f"calibration text has {have} tokens, need {need}: add text (several files "
                f"can be given, comma-separated) or lower {what} / --seq-len"
            )
    elif rc.calib_ids is not None:
        have = rc.calib_ids.shape[0]
        n = rc.qad.samples + rc.qad.val if rc.qad is not None else rc.calib_samples
        if have < n:
            raise SystemExit(f"{have} calibration sequences given, need {n}")
    facts["calib_tokens_needed"] = need
    return facts


def _log_qad_plan(rc: RunConfig, ex: Executor, log) -> None:
    from .qad import estimate_seconds

    try:
        sec, _ = estimate_seconds(
            rc.src, rc.qad, rc.seq_len, rc.include, rc.exclude, ex.primary.type != "cpu"
        )
        est = (
            f"roughly {sec / 3600:.1f} h"
            if sec >= 5400
            else f"roughly {sec / 60:.0f} min"
        )
    except Exception as e:  # the estimate is informative only
        est = f"unknown ({e})"
    log(
        f"QAD is on (default): after PTQ every decoder layer is distilled against the "
        f"unquantized model, {est} on {ex.primary} on top of PTQ; pass --no-qad for PTQ only"
    )


def plan(rc: RunConfig, log=print) -> dict:
    """``--dry-run``: validate and describe the run without doing it."""
    from dataclasses import asdict

    facts = check_inputs(rc, log)
    tc = _text_config(rc.src)
    hidden, n_layers = tc.get("hidden_size"), tc.get("num_hidden_layers")
    if rc.limit_layers is not None and n_layers:
        n_layers = min(n_layers, rc.limit_layers)
    out = {
        "src": str(rc.src),
        "dst": str(rc.dst),
        "workdir": str(rc.workdir),
        "devices": rc.devices,
        "target_bpw": rc.target_bpw,
        "max_error_ratio": rc.max_error_ratio,
        "sensitivity": rc.sensitivity,
        "rounding": rc.cfg.rounding,
        "layers": n_layers,
        "moe_layout": rc.moe_layout,
        **facts,
    }
    if rc.qad is not None and hidden:
        from .qad import parse_layers

        qc = rc.qad
        n_seq = qc.samples + qc.val
        out["qad"] = asdict(qc)
        out["qad_trained_layers"] = len(
            [
                i
                for i in parse_layers(qc.layers, tc["num_hidden_layers"])
                if i < n_layers
            ]
        )
        out["qad_hidden_state_ram_gib_fp32"] = round(
            4 * n_seq * rc.seq_len * hidden * 4 / 2**30, 2
        )
        out["resume"] = "per layer under <workdir>/qad (re-run the same command)"
    log("dry run: nothing written")
    return out


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
    int8 = sum(r.get("frac_int8", 0.0) * r["shape"][0] for r in report.values())
    by_kind: dict[str, list[float]] = {}
    for name, r in report.items():
        kind = name.rsplit(".", 2)[-2]
        by_kind.setdefault(kind, []).append(r["bpw"])
    return {
        "tensors": len(report),
        "params": params,
        "avg_bpw": bpw,
        "frac_int4_rows": int4,
        **({"frac_int8_rows": int8 / max(rows, 1)} if int8 else {}),
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


def _probe(rc: RunConfig, ex: Executor, tables, weights, log) -> dict | None:
    """Phase 2a: the sensitivity probe, reused from the cache (``probe`` entry keyed by
    the tables and the probe settings and tokens) or from the workdir."""
    from .sensitivity import REF_CANDIDATE, probe_sensitivity

    pids = _probe_ids(rc)
    skey = {
        "phase1": _phase1_key(rc),
        "seqs": rc.sens_seqs,
        "len": rc.sens_len,
        "floor": rc.sens_floor,
        "dtype": rc.sens_dtype,
        "share_floor": rc.expert_share_floor,
        **({"batch": rc.sens_batch} if rc.sens_batch != 1 else {}),
    }
    spath, kpath = rc.workdir / "sensitivity.json", rc.workdir / "sensitivity.key.json"

    def compute():
        return probe_sensitivity(
            rc.src,
            tables,
            rc.workdir,
            pids,
            rc.cfg,
            ex,
            floor=rc.sens_floor,
            log=log,
            units=probe_units(tables, weights),
            dtype=getattr(torch, rc.sens_dtype),
            batch=rc.sens_batch,
        )

    ref = rc.workdir / "cache_ref.json"
    if rc.cache_dir is not None and ref.exists():
        from . import cache as C

        tkey = json.loads(
            (Path(json.loads(ref.read_text())["tables"]) / "key.json").read_text()
        )
        pent = C.Entry(Path(rc.cache_dir), C.probe_key(tkey, rc, pids, REF_CANDIDATE))
        with pent.building():
            if pent.complete():
                log(f"cache: sensitivity probe hit ({pent.path})")
                pent.touch()
                sens = json.loads((pent.path / "sensitivity.json").read_text())
            else:
                log(f"cache: sensitivity probe miss ({pent.path})")
                log("phase 2a: probing per-tensor sensitivity")
                sens = compute()
                pent.begin()
                (pent.work / "sensitivity.json").write_text(json.dumps(sens, indent=1))
                pent.finish({})
        C.write_ref(rc.workdir, probe=pent.path)
    elif spath.exists() and kpath.exists() and json.loads(kpath.read_text()) == skey:
        log("phase 2a: reusing the sensitivity probe of this workdir")
        return json.loads(spath.read_text())
    else:
        log("phase 2a: probing per-tensor sensitivity")
        sens = compute()
    spath.write_text(json.dumps(sens, indent=1))
    kpath.write_text(json.dumps(skey))
    return sens


def _allocate(rc: RunConfig, tables, sens, target_bpw, ratio, log):
    if target_bpw is None:
        ratio = DEFAULT_ERROR_RATIO if ratio is None else ratio
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
            f"target {target_bpw} bpw"
        )
        alloc = allocate_global(list(tables.values()), target_bpw, sens)
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
    return alloc


def output_dir(dst: Path, ratio: float) -> Path:
    """Output of one ratio in multi-output mode: ``<dst>-r<ratio>``."""
    return Path(f"{dst}-r{ratio:g}")


def staging_dir(rc: RunConfig, dst: Path) -> Path:
    """Per-expert staging copy of ``dst`` for ``moe_layout="stacked"``."""
    return rc.workdir / "per-expert" / dst.name


def run(rc: RunConfig, log=print) -> dict:
    """Quantize. With ``rc.max_error_ratios`` (several values) phase 1 and the probe
    run once and every ratio is allocated and written to ``<dst>-r<ratio>`` in one
    encode pass; the return value is then ``{"outputs": {dir: summary}}``."""
    t_start = time.time()
    if rc.moe_layout not in ("per-expert", "stacked"):
        raise ValueError(f"moe_layout {rc.moe_layout!r}: per-expert or stacked")
    rc.workdir.mkdir(parents=True, exist_ok=True)
    ex = Executor(rc.devices)
    calibrated = rc.calib is not None or rc.calib_ids is not None
    if rc.qad is not None:
        check_inputs(rc, log)  # fails early, or drops QAD for a data-free run
    if rc.qad is not None:
        _log_qad_plan(rc, ex, log)
        if rc.max_error_ratios is not None and len(rc.max_error_ratios) > 1:
            log(
                f"warning: QAD runs once per output: {len(rc.max_error_ratios)} ratios = "
                f"{len(rc.max_error_ratios)} QAD passes in sequence (the estimate above "
                "is per pass); pass --no-qad for a quick PTQ sweep"
            )
    if rc.cfg.rounding == "gptq" and not calibrated:
        log("no calibration data: using nearest rounding")
        rc.cfg.rounding = "nearest"
        rc.cfg.sequential = False
    multi = rc.max_error_ratios is not None and len(rc.max_error_ratios) > 1
    if rc.target_bpw is not None and (rc.max_error_ratio is not None or multi):
        raise ValueError("give either target_bpw or max_error_ratio, not both")
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
        sens = _probe(rc, ex, tables, weights, log)
    t_probe = time.time()
    if rc.int8_force > 0:
        from .allocate import force_int8

        tables, chosen = force_int8(tables, sens, rc.int8_force)
        log(
            f"INT8 rule: {len(chosen)} most sensitive tensors all-INT8, the rest INT2/INT4: "
            + ", ".join(chosen)
        )
    if multi:
        targets = [(None, r, output_dir(rc.dst, r)) for r in rc.max_error_ratios]
    else:
        ratio = rc.max_error_ratio
        if ratio is None and rc.max_error_ratios:
            ratio = rc.max_error_ratios[0]
        targets = [(rc.target_bpw, ratio, rc.dst)]
    outs, allocs = [], []
    t_qad = 0.0  # QAD passes (spec 13), reported apart from the allocation
    for target_bpw, ratio, dst in targets:
        alloc = _allocate(rc, tables, sens, target_bpw, ratio, log)
        pre, dense = None, None
        if rc.qad is not None:
            what = (
                f" for ratio {ratio:g} ({len(targets)} outputs, one QAD each)"
                if multi
                else ""
            )
            log(f"phase 3q: layer-wise quantization-aware distillation (spec 13){what}")
            tag = f"-r{ratio:g}" if multi else ""
            tq = time.time()
            pre, dense = _qad_encode(
                rc, ex, tables, alloc.picks, sens, log, target_bpw, ratio, tag
            )
            t_qad += time.time() - tq
        elif rc.cfg.rounding == "gptq" and rc.cfg.sequential:
            log("phase 3a: sequential encode pass (spec 8.4)")
            pre = _sequential_encode(rc, ex, tables, alloc.picks, log)
        meta = {
            "mode": "flex",
            "target_bpw": target_bpw,
            "max_error_ratio": ratio
            if (multi or rc.max_error_ratio is not None)
            else None,
            "allocation_error_ratio": alloc.error_ratio,
            "calibrated": calibrated,
            "rounding": rc.cfg.rounding,
            "sequential": rc.cfg.sequential,
            "smooth": rc.cfg.smooth,
            "rotation": rc.cfg.rotation,
            "calib_tokens": rc.calib_samples * rc.seq_len if calibrated else 0,
        }
        if rc.qad is not None:
            from .qad import qad_meta

            meta["qad"] = qad_meta(rc.qad) | {
                "tokens": (rc.qad.samples + rc.qad.val) * rc.seq_len
            }
            if dense:
                meta["qad"]["dense_trained"] = len(dense)
        done = rc.workdir / ("shards_done" if not multi else f"shards_done-r{ratio:g}")
        # stacked: write per expert to a staging dir in the workdir, then restack
        # (byte-level, hi2.stack.restack) into dst
        write_to = dst if rc.moe_layout == "per-expert" else staging_dir(rc, dst)
        outs.append(Output(write_to, alloc.picks, meta, done, pre, dense=dense or None))
        allocs.append(alloc)
    dst_of = {id(o): t[2] for o, t in zip(outs, targets)}
    t_alloc = time.time()
    log(f"phase 3: encoding and writing {len(outs)} checkpoint(s)")
    write_checkpoints(rc.src, outs, rc.workdir, tables, rc.cfg, ex, log)
    t_end = time.time()
    tok = StatsView(rc.workdir).expert_tokens_path()
    summaries = {}
    for o, alloc in zip(outs, allocs):
        report = o.report
        summary = summarize(report) | {
            "allocation_bpw": alloc.bpw,
            "feasible": alloc.feasible,
            "allocation_error_ratio": alloc.error_ratio,
            "frac_rows_by_bits": alloc.frac_by_bits,
            "runtime_s": {
                "phase1": round(t_phase1 - t_start, 1),
                "probe": round(t_probe - t_phase1, 1),
                "allocate": round(t_alloc - t_probe - t_qad, 1),
                **({"qad": round(t_qad, 1)} if rc.qad is not None else {}),
                "encode": round(t_end - t_alloc, 1),
                "total": round(t_end - t_start, 1),
                "outputs_in_this_run": len(outs),
            },
            "peak_memory_gib": _peak_memory(ex),
        }
        if weights:
            summary["experts"] = _expert_summary(report)
            if tok is not None:
                cov = json.loads(tok.read_text())
                summary["experts"]["fallback_tensors"] = len(cov["fallback"])
                summary["experts"]["coverage"] = cov["coverage"]
        (o.dst / "hi2_report.json").write_text(
            json.dumps({"summary": summary, "tensors": report}, indent=2)
        )
        final = dst_of[id(o)]
        if final != o.dst:
            from .stack import restack

            log(f"restacking {o.dst} -> {final}")
            restack(o.dst, final, rc.moe_layout, log, overwrite=True)
            shutil.rmtree(o.dst)
            summary["moe_layout"] = rc.moe_layout
        summaries[str(final)] = summary
    if multi:
        return {"outputs": summaries}
    return summaries[str(targets[0][2])]
