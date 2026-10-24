# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Checkpoint I/O: shard enumeration, eligibility, the streaming writer (phase 3)
and verification. Tensor names follow spec 2.4 (``<name>.hi2_<field>``).
"""

import json
import os
from pathlib import Path

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
    return out


def from_tensors(d: dict[str, torch.Tensor], base: str) -> HI2Tensor:
    g = {f: d[f"{base}.hi2_{f}"] for f in FIELDS}
    return HI2Tensor(**g, shape=(g["row_nbits"].numel(), g["had"].numel() * 16))


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
) -> dict:
    """Phase 3: stream shards, encode the tables' tensors with their picks, copy the
    rest. Existing output shards are kept (resume)."""
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
                    w = f.get_tensor(name)
                    sp = table_stats_path(workdir, name)
                    stats = None
                    if sp.exists():
                        stats = ColumnStats.from_state(torch.load(sp), "cpu")
                    t = encode_with_picks(w, stats, table, p, cfg, ex)
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
            d = {
                f"{base}.hi2_{fld}": f.get_tensor(f"{base}.hi2_{fld}") for fld in FIELDS
            }
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
                d = {
                    f"{base}.hi2_{fld}": f.get_tensor(f"{base}.hi2_{fld}")
                    for fld in FIELDS
                }
                out[f"{base}.weight"] = dequantize(from_tensors(d, base)).float()
    return out
