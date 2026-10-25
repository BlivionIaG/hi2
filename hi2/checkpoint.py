# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Checkpoint I/O: shard enumeration, eligibility, the streaming writer (phase 3)
and verification. Tensor names follow spec 2.4 (``<name>.hi2_<field>``).
"""

import json
import os
from pathlib import Path

import regex as re
import torch

from .calibrate import Config, TensorTable, encode_with_picks
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
        return {k: "model.safetensors" for k in f.keys()}  # noqa: SIM118


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
    from safetensors import safe_open
    from safetensors.torch import save_file

    wmap = weight_map(src)
    shards = sorted(set(wmap.values()))
    dst.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    report: dict[str, dict] = {}
    for shard in shards:
        names = [n for n, s in wmap.items() if s == shard]
        done = (dst / shard).exists() and (workdir / "shards_done" / shard).exists()
        out: dict[str, torch.Tensor] = {}
        with safe_open(src / shard, "pt") as f:
            for name in names:
                if name not in tables:
                    if not done:
                        out[name] = f.get_tensor(name)
                    else:
                        out[name] = None
                    continue
                table, p = tables[name], picks[name]
                base = name.removesuffix(".weight")
                if not done:
                    if pre_encoded is not None and name in pre_encoded:
                        t = pre_encoded[name]
                    else:
                        w = f.get_tensor(name)
                        sp = table_stats_path(workdir, name)
                        stats = None
                        if sp.exists():
                            stats = ColumnStats.from_state(torch.load(sp), "cpu")
                        hp = table_hessian_path(workdir, name)
                        sigma = torch.load(hp) if hp.exists() else None
                        if cfg.rounding == "gptq" and sigma is None:
                            log(f"warning: {name} has no Hessian; nearest rounding")
                        t = encode_with_picks(w, stats, table, p, cfg, ex, sigma)
                    out.update(to_tensors(t, base))
                    report[name] = _tensor_report(table, p, t)
                    log(
                        f"{name} {tuple(table.shape)} bpw={report[name]['bpw']:.3f} "
                        f"rel_err={report[name]['pred_rel_err']:.4f} "
                        f"int4={report[name]['frac_int4']:.2f}"
                    )
                else:
                    for fld in FIELDS:
                        out[f"{base}.hi2_{fld}"] = None
                    if table.col_scale is not None:
                        out[f"{base}.hi2_col_scale"] = None
        if not done:
            save_file(out, str(dst / shard), metadata={"format": "pt"})
            (workdir / "shards_done").mkdir(parents=True, exist_ok=True)
            (workdir / "shards_done" / shard).write_text("ok")
        for k in out:
            new_map[k] = shard
    index = {"metadata": {}, "weight_map": new_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    cfgj = json.loads((src / "config.json").read_text())
    cfgj["quantization_config"] = {
        "quant_method": "hadamard_int2",
        "version": SPEC_VERSION,
        **meta,
    }
    (dst / "config.json").write_text(json.dumps(cfgj, indent=2))
    skip = {"config.json", "model.safetensors.index.json", *shards}
    for entry in src.iterdir():
        if entry.name in skip or entry.name.startswith("."):
            continue
        link = dst / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)
    return report


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
        "block_hist": {int(b): int((bl == b).sum()) for b in (16, 32, 64)},
        "rotated_groups": int(t.had.sum()),
        "groups": int(t.had.numel()),
        "n_tokens": table.n_tokens,
    }


def verify_checkpoint(src: Path, dst: Path, max_tensors: int = 8, log=print) -> dict:
    """Reload quantised tensors from ``dst`` and compare against ``src`` weights."""
    from safetensors import safe_open

    report = json.loads((dst / "hi2_report.json").read_text())["tensors"]
    wsrc, wdst = weight_map(src), weight_map(dst)
    worst, checked = 0.0, 0
    for name, info in list(report.items())[:max_tensors]:
        base = name.removesuffix(".weight")
        with safe_open(dst / wdst[f"{base}.hi2_codes"], "pt") as f:
            d = read_tensors(f, base)
        with safe_open(src / wsrc[name], "pt") as f:
            w = f.get_tensor(name).double()
        t = from_tensors(d, base)
        rel = ((dequantize(t) - w) ** 2).sum().item() / (w**2).sum().item()
        if abs(tensor_bpw(t) - info["bpw"]) > 1e-6:
            raise AssertionError(f"{name}: bpw changed on reload")
        worst = max(worst, rel)
        checked += 1
        log(f"verify {name}: weight energy loss {rel:.4f} bpw {tensor_bpw(t):.3f}")
    return {"checked": checked, "worst_unweighted_energy_loss": worst}


def load_dequantized(dst: Path) -> dict[str, torch.Tensor]:
    """Dense fp32 state dict of an HI2 checkpoint: quantised tensors are
    dequantised back to ``<name>.weight`` (original basis); the rest are copied."""
    from safetensors import safe_open

    wmap = weight_map(dst)
    out: dict[str, torch.Tensor] = {}
    bases = {k.removesuffix(".hi2_codes") for k in wmap if k.endswith(".hi2_codes")}
    for shard in sorted(set(wmap.values())):
        with safe_open(dst / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                if ".hi2_" in name:
                    continue
                out[name] = f.get_tensor(name)
            for base in sorted(b for b in bases if wmap[f"{b}.hi2_codes"] == shard):
                d = read_tensors(f, base)
                out[f"{base}.weight"] = dequantize(from_tensors(d, base)).float()
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
