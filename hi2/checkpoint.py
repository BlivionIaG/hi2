# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Checkpoint I/O: shard enumeration, eligibility, the streaming writer (phase 3)
and verification. Tensor names follow spec 2.4 (``<name>.hi2_<field>``).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch

from .calibrate import Config, EncodeRequest, TensorTable, encode_many
from .executor import Executor
from .format import FIELDS, HI2Tensor, dequantize, tensor_bpw
from .stats import ColumnStats

# Embeddings, the output head, norms, routers and indexer projections stay dense.
DEFAULT_EXCLUDE = (
    r"(embed|lm_head|norm|\.mlp\.gate(\.weight)?$|router|indexer|conv1d|\.bias$)"
)
SPEC_VERSION = "3.2"


def weight_map(d: Path) -> dict[str, str]:
    """Tensor name -> shard file; a single-file checkpoint has no index."""
    from safetensors import safe_open

    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    with safe_open(d / "model.safetensors", "pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")  # noqa: SIM118


_LAYER_KEY = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(.*)$")


def make_resolver(wmap: dict[str, str]):
    """Map a module's weight name to the checkpoint's tensor name.

    Multimodal checkpoints often nest the text model (``model.language_model.layers.N``)
    while the text-only class names it ``model.layers.N``. Tries the name itself, the
    usual ``language_model`` rewrites, then a unique match on ``layers.N.<rest>``.
    Returns ``None`` when nothing matches unambiguously.
    """
    index: dict[tuple[int, str], list[str]] = {}
    for key in wmap:
        m = _LAYER_KEY.match(key)
        if m:
            index.setdefault((int(m.group(1)), m.group(2)), []).append(key)

    def resolve(name: str) -> str | None:
        for cand in (
            name,
            name.replace("model.", "model.language_model.", 1),
            "language_model." + name,
        ):
            if cand in wmap:
                return cand
        m = _LAYER_KEY.match(name)
        hits = index.get((int(m.group(1)), m.group(2)), []) if m else []
        return hits[0] if len(hits) == 1 else None

    return resolve


def is_eligible(name: str, shape, dtype, include, exclude) -> bool:
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and dtype.is_floating_point
        and shape[1] % 64 == 0
        and (include is None or include.search(name) is not None)
        and exclude.search(name) is None
    )


def to_tensors(t: HI2Tensor, base: str) -> dict[str, torch.Tensor]:
    out = {f"{base}.hi2_{f}": getattr(t, f) for f in FIELDS if f != "had"}
    out[f"{base}.hi2_had"] = t.had
    if t.col_scale is not None:
        out[f"{base}.hi2_col_scale"] = t.col_scale
    return out


def from_tensors(d: dict[str, torch.Tensor], base: str) -> HI2Tensor:
    g = {f: d[f"{base}.hi2_{f}"] for f in FIELDS}
    return HI2Tensor(
        **g,
        shape=(g["row_nbits"].numel(), g["had"].numel() * 16),
        col_scale=d.get(f"{base}.hi2_col_scale"),
    )


def read_tensors(f, base: str) -> dict[str, torch.Tensor]:
    """The ``<base>.hi2_*`` tensors (including the optional smoothing vector) from an
    open safetensors file."""
    names = [f"{base}.hi2_{fld}" for fld in FIELDS]
    cs = f"{base}.hi2_col_scale"
    if cs in f.keys():  # noqa: SIM118
        names.append(cs)
    return {n: f.get_tensor(n) for n in names}


def table_hessian_path(workdir: Path, name: str) -> Path:
    return workdir / "hessian" / (name.replace("/", "_") + ".pt")


def table_stats_path(workdir: Path, name: str) -> Path:
    return workdir / "stats" / (name.replace("/", "_") + ".pt")


# tensors encoded together (u factors, weights and temporaries on the device)
ENCODE_BATCH_BYTES = int(float(os.environ.get("HI2_ENCODE_BATCH_GB", "6")) * 2**30)


@dataclass
class Output:
    """One checkpoint written by :func:`write_checkpoints`."""

    dst: Path
    picks: dict[str, torch.Tensor]
    meta: dict
    done_dir: Path  # per-shard "finished" markers (resume)
    pre_encoded: dict | None = None
    report: dict = field(default_factory=dict)
    new_map: dict = field(default_factory=dict)
    # trained dense tensors (QAD norms) written instead of the source's
    dense: dict | None = None


def write_checkpoint(
    src: Path,
    dst: Path,
    workdir: Path,
    tables: dict[str, TensorTable],
    picks: dict[str, torch.Tensor],
    cfg: Config,
    ex: Executor,
    meta: dict,
    log=print,
    pre_encoded: dict | None = None,
) -> dict:
    """Phase 3: stream shards, encode the tables' tensors with their picks (or take
    them from ``pre_encoded``), copy the rest. Existing output shards are kept."""
    out = Output(dst, picks, meta, workdir / "shards_done", pre_encoded)
    write_checkpoints(src, [out], workdir, tables, cfg, ex, log)
    return out.report


def _requests_for(names, path, view, tables, outs, cfg):
    """Load the source tensors, statistics and Hessians of ``names`` (one shard file)
    and build an :class:`EncodeRequest` per tensor and output that still needs it."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return _requests_from(names, f, view, tables, outs, cfg)


def _requests_from(names, f, view, tables, outs, cfg):
    reqs, slots = [], []
    loaded: dict[str, torch.Tensor] = {}
    for name in names:
        need = [
            o for o in outs if not (o.pre_encoded is not None and name in o.pre_encoded)
        ]
        if not need:
            continue
        w = f.get_tensor(name)
        st = view.stats_state(name)
        stats = None
        if st is not None and st["n"] > 0:
            stats = ColumnStats.from_state(st, "cpu")
        sigma = sid = None
        if cfg.rounding == "gptq":
            sid = view.sigma_id(name)
            if sid is not None:  # one load per shared input (gate/up, q/k/v)
                if sid not in loaded:
                    loaded[sid] = view.sigma(name)
                sigma = loaded[sid]
        for o in need:
            reqs.append(
                EncodeRequest(w, stats, tables[name], o.picks[name], sigma, sid)
            )
            slots.append((name, o))
    return reqs, slots


def _batches(names, tables, outs, budget):
    """Consecutive tensors of a shard grouped so that the factors (K^2) and weights of
    a group stay under ``budget`` bytes (fp32) per device."""
    cur, size = [], 0
    for name in names:
        r, k = tables[name].shape
        b = 4 * (k * k + 3 * r * k * len(outs))
        if cur and size + b > budget:
            yield cur
            cur, size = [], 0
        cur.append(name)
        size += b
    if cur:
        yield cur


def write_checkpoints(
    src: Path,
    outs: list[Output],
    workdir: Path,
    tables: dict[str, TensorTable],
    cfg: Config,
    ex: Executor,
    log=print,
) -> None:
    """Phase 3 for one or several outputs (allocations) in one pass over the source:
    each shard is read once, every tensor's Hessian is loaded and factorised once for
    all outputs and all tensors sharing its input, and the tensors of a shard are
    encoded in batches (:func:`hi2.calibrate.encode_many`) while the next batch is read
    from disk. Existing output shards are kept."""
    from concurrent.futures import ThreadPoolExecutor

    from safetensors import safe_open
    from safetensors.torch import save_file

    from .cache import StatsView

    view = StatsView(workdir)
    wmap = weight_map(src)
    shards = sorted(set(wmap.values()))
    for o in outs:
        o.dst.mkdir(parents=True, exist_ok=True)
    warned = False
    io = ThreadPoolExecutor(1)
    for shard in shards:
        names = [n for n, s in wmap.items() if s == shard]
        todo = [
            o
            for o in outs
            if not ((o.dst / shard).exists() and (o.done_dir / shard).exists())
        ]
        data: dict[int, dict[str, torch.Tensor | None]] = {id(o): {} for o in outs}
        with safe_open(src / shard, "pt") as f:
            for name in names:
                if name in tables:
                    continue
                t = f.get_tensor(name) if todo else None
                for o in outs:
                    v = t
                    if o.dense is not None and name in o.dense and o in todo:
                        v = o.dense[name].contiguous()
                    data[id(o)][name] = v if o in todo else None
            enc_names = [n for n in names if n in tables]
            for o in outs:
                if o in todo:
                    continue
                for name in enc_names:  # finished shard: only the index entries
                    base = name.removesuffix(".weight")
                    for fld in FIELDS:
                        data[id(o)][f"{base}.hi2_{fld}"] = None
                    if tables[name].col_scale is not None:
                        data[id(o)][f"{base}.hi2_col_scale"] = None
            if todo:
                if cfg.rounding == "gptq" and not warned:
                    missing = [n for n in enc_names if view.sigma_path(n) is None]
                    if missing:
                        log(
                            f"warning: {len(missing)} tensors have no Hessian; nearest rounding"
                        )
                        warned = True
                groups = list(_batches(enc_names, tables, todo, ENCODE_BATCH_BYTES))
                sp = src / shard
                nxt = (
                    io.submit(_requests_for, groups[0], sp, view, tables, todo, cfg)
                    if groups
                    else None
                )
                for gi, group in enumerate(groups):
                    reqs, slots = nxt.result()
                    if gi + 1 < len(groups):
                        nxt = io.submit(
                            _requests_for, groups[gi + 1], sp, view, tables, todo, cfg
                        )
                    encoded = encode_many(reqs, cfg, ex) if reqs else []
                    by = {(n, id(o)): t for (n, o), t in zip(slots, encoded)}
                    for name in group:
                        base = name.removesuffix(".weight")
                        for o in todo:
                            if o.pre_encoded is not None and name in o.pre_encoded:
                                t = o.pre_encoded[name]
                            else:
                                t = by[(name, id(o))]
                            data[id(o)].update(to_tensors(t, base))
                            o.report[name] = _tensor_report(
                                tables[name], o.picks[name], t
                            )
                            rep = o.report[name]
                            tag = "" if len(outs) == 1 else f"[{o.dst.name}] "
                            log(
                                f"{tag}{name} {tuple(tables[name].shape)} bpw={rep['bpw']:.3f} "
                                f"rel_err={rep['pred_rel_err']:.4f} "
                                f"int4={rep['frac_int4']:.2f}"
                                + (
                                    f" int8={rep['frac_int8']:.2f}"
                                    if rep["frac_int8"]
                                    else ""
                                )
                            )
                    del reqs, encoded, by
        for o in outs:
            out = data[id(o)]
            if o in todo:
                ordered = {}
                for name in names:  # keep the source order of tensors in the shard
                    if name in tables:
                        base = name.removesuffix(".weight") + ".hi2_"
                        ordered.update(
                            {k: v for k, v in out.items() if k.startswith(base)}
                        )
                    else:
                        ordered[name] = out[name]
                save_file(ordered, str(o.dst / shard), metadata={"format": "pt"})
                o.done_dir.mkdir(parents=True, exist_ok=True)
                (o.done_dir / shard).write_text("ok")
                out = ordered
            for k in out:
                o.new_map[k] = shard
    io.shutdown()
    for o in outs:
        _finish_output(src, o, shards)


def _finish_output(src: Path, o: Output, shards) -> None:
    index = {"metadata": {}, "weight_map": o.new_map}
    (o.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    cfgj = json.loads((src / "config.json").read_text())
    cfgj["quantization_config"] = {
        "quant_method": "rdna_hint_flex",
        "version": SPEC_VERSION,
        **o.meta,
    }
    (o.dst / "config.json").write_text(json.dumps(cfgj, indent=2))
    skip = {"config.json", "model.safetensors.index.json", *shards}
    for entry in src.iterdir():
        if entry.name in skip or entry.name.startswith("."):
            continue
        link = o.dst / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)


def _tensor_report(table: TensorTable, picks: torch.Tensor, t: HI2Tensor) -> dict:
    r = table.shape[0]
    chosen = table.sse[picks, torch.arange(r)].sum().item()
    nb = t.row_nbits.to(torch.int64)
    bl = t.row_block.to(torch.int64)
    return {
        "shape": list(table.shape),
        "bpw": tensor_bpw(t),
        "pred_rel_err": chosen / max(table.energy, 1e-30),
        "frac_int4": float((nb == 4).float().mean()),
        "frac_int8": float((nb == 8).float().mean()),
        "block_hist": {int(b): int((bl == b).sum()) for b in (16, 32, 64)},
        "rotated_groups": int(t.had.sum()),
        "groups": int(t.had.numel()),
        "n_tokens": table.n_tokens,
    }


def verify_checkpoint(src: Path, dst: Path, max_tensors: int = 8, log=print) -> dict:
    """Reload quantised tensors from ``dst`` (either MoE layout) and compare against
    ``src`` weights."""
    from safetensors import safe_open

    from .stack import Hi2Checkpoint

    report = json.loads((dst / "hi2_report.json").read_text())["tensors"]
    wsrc = weight_map(src)
    ck = Hi2Checkpoint(dst)
    worst, checked = 0.0, 0
    for name, info in list(report.items())[:max_tensors]:
        t = ck.get(name.removesuffix(".weight"))
        with safe_open(src / wsrc[name], "pt") as f:
            w = f.get_tensor(name).double()
        rel = ((dequantize(t) - w) ** 2).sum().item() / (w**2).sum().item()
        if abs(tensor_bpw(t) - info["bpw"]) > 1e-6:
            raise AssertionError(f"{name}: bpw changed on reload")
        worst = max(worst, rel)
        checked += 1
        log(f"verify {name}: weight energy loss {rel:.4f} bpw {tensor_bpw(t):.3f}")
    return {"checked": checked, "worst_unweighted_energy_loss": worst}


def load_dequantized(dst: Path) -> dict[str, torch.Tensor]:
    """Dense fp32 state dict of an HI2 checkpoint (either MoE layout): quantised
    tensors are dequantised back to ``<name>.weight`` (original basis); the rest are
    copied."""
    from safetensors import safe_open

    from .stack import Hi2Checkpoint

    ck = Hi2Checkpoint(dst)
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(ck.wmap.values())):
        with safe_open(dst / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                if ".hi2_" in name:
                    continue
                out[name] = f.get_tensor(name)
            for base, t in ck.iter_shard(shard, f):
                out[f"{base}.weight"] = dequantize(t).float()
    return out


def export_dense(src: Path, dst: Path, out: Path) -> None:
    """Write the dequantised model as a plain fp16 Hugging Face checkpoint in ``out``
    (``model.safetensors`` + ``config.json`` without ``quantization_config`` + the
    tokenizer/other files of ``src`` symlinked), e.g. to score it in another engine."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.to(torch.float16).contiguous() for k, v in load_dequantized(dst).items()
    }
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    skip = {"config.json", "model.safetensors", "model.safetensors.index.json"}
    for entry in src.iterdir():
        if (
            entry.name in skip
            or entry.name.startswith(".")
            or entry.suffix == ".safetensors"
        ):
            continue
        link = out / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)
