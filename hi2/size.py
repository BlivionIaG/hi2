# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 size``: where the bytes of an HI2 checkpoint go.

Sizes come from the safetensors headers, so no tensor data is read, except the small
per-row precision array (``hi2_row_nbits``) that gives the INT2/INT4/INT8 shares. The
same byte accounting works for both expert layouts. It counts everything a loader has
to read: codes, block parameters, row constants, the smoothing vector, and the row
index (offsets, precisions, block sizes, rotation flags).
"""

import re
from collections import defaultdict
from pathlib import Path

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}

# HI2 field -> the group it is reported under
FIELD_GROUP = {
    "codes": "codes",
    "blk_params": "block parameters",
    "row_consts": "row constants",
    "col_scale": "smoothing vector",
    "code_off": "row index",
    "param_off": "row index",
    "row_nbits": "row index",
    "row_block": "row index",
    "had": "row index",
    "part_off": "row index",
}
GROUPS = ("codes", "block parameters", "row constants", "smoothing vector", "row index")

_HI2 = re.compile(r"^(?P<base>.+)\.hi2_(?P<field>[a-z0-9_]+)$")
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def _tensor_info(d: Path) -> dict[str, tuple[tuple[int, ...], int, Path]]:
    """Tensor name -> (shape, bytes, shard file), from the headers only."""
    from safetensors import safe_open

    from .checkpoint import weight_map

    out = {}
    for shard in sorted(set(weight_map(d).values())):
        with safe_open(d / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                sl = f.get_slice(name)
                shape = tuple(sl.get_shape())
                n = 1
                for s in shape:
                    n *= s
                out[name] = (shape, n * DTYPE_BYTES[sl.get_dtype()], d / shard)
    return out


def kind(base: str) -> str:
    """A tensor family name: the layer index and the expert index dropped."""
    s = re.sub(r"^.*?\.layers\.\d+\.", "", base)
    return re.sub(r"\.experts\.\d+\.", ".experts.*.", s)


def layer(base: str) -> str:
    m = _LAYER.search(base)
    return m.group(1) if m else "other"


def dense_kind(name: str) -> str:
    low = name.lower()
    for key, label in (
        ("embed", "embeddings"),
        ("lm_head", "lm_head"),
        ("norm", "norms"),
    ):
        if key in low:
            return label
    return "other dense (routers, convolutions, biases)"


def _precision_weights(path: Path, name: str, k: int) -> dict[int, int]:
    """Weights per row precision of one HI2 base (reads its ``row_nbits`` array)."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        nb = f.get_tensor(name)
    return {int(b): int((nb == b).sum()) * k for b in nb.unique().tolist()}


def size_report(ckpt: Path, src: Path | None = None, by: str = "kind") -> dict:
    """Byte accounting of the checkpoint ``ckpt``; with ``src``, the compression
    against the source checkpoint's tensors."""
    info = _tensor_info(ckpt)
    bases: dict[str, dict] = {}
    dense: dict[str, int] = defaultdict(int)
    for name, (shape, nbytes, path) in info.items():
        m = _HI2.match(name)
        if not m:
            dense[dense_kind(name)] += nbytes
            continue
        b = bases.setdefault(
            m["base"],
            {
                "bytes": defaultdict(int),
                "rows": 0,
                "k": 0,
                "prec": {},
                "stacked": False,
            },
        )
        field = m["field"]
        b["bytes"][FIELD_GROUP.get(field, "row index")] += nbytes
        if field == "row_nbits":
            b["rows"] = shape[0]
            b["nbits_at"] = (path, name)
        elif field == "had":
            b["k"] = shape[-1] * 16
        elif field == "part_off":
            b["stacked"] = True
    for b in bases.values():
        b["prec"] = _precision_weights(*b["nbits_at"], b["k"])
        b["weights"] = b["rows"] * b["k"]

    key = {"kind": kind, "layer": layer, "tensor": lambda base: base}[by]
    rows: dict[str, dict] = {}
    totals: dict[str, int] = defaultdict(int)
    prec: dict[int, int] = defaultdict(int)
    for base, b in bases.items():
        r = rows.setdefault(key(base), {"bases": 0, "weights": 0, "bytes": 0})
        r["bases"] += 1
        r["weights"] += b["weights"]
        r["bytes"] += sum(b["bytes"].values())
        for g, n in b["bytes"].items():
            totals[g] += n
        for nb, w in b["prec"].items():
            prec[nb] += w
    weights = sum(b["weights"] for b in bases.values())
    linear_bytes = sum(totals.values())
    for r in rows.values():
        r["bpw"] = 8 * r["bytes"] / r["weights"] if r["weights"] else 0.0

    files = {p for _, _, p in info.values()}
    is_moe = any(".experts." in base for base in bases)
    layout = (
        "stacked"
        if any(b["stacked"] for b in bases.values())
        else "per-expert"
        if is_moe
        else "dense"
    )
    out = {
        "path": str(ckpt),
        "layout": layout,
        "shards": len(files),
        "file_bytes": sum(p.stat().st_size for p in files),
        "linears": {
            "bases": len(bases),
            "weights": weights,
            "bytes": linear_bytes,
            "bpw": 8 * linear_bytes / weights if weights else 0.0,
            "by_field": {g: totals.get(g, 0) for g in GROUPS},
            "bpw_by_field": {
                g: 8 * totals.get(g, 0) / weights if weights else 0.0 for g in GROUPS
            },
            "precision_share": {
                f"INT{nb}": w / weights for nb, w in sorted(prec.items())
            },
        },
        "dense": dict(sorted(dense.items(), key=lambda kv: -kv[1])),
        "dense_bytes": sum(dense.values()),
        "tensor_bytes": linear_bytes + sum(dense.values()),
        "by": by,
        "rows": dict(sorted(rows.items(), key=lambda kv: -kv[1]["bytes"])),
    }
    if src is not None:
        sinfo = _tensor_info(src)
        sbytes = sum(n for _, n, _ in sinfo.values())
        out["source"] = {
            "path": str(src),
            "tensor_bytes": sbytes,
            "ratio": sbytes / out["tensor_bytes"] if out["tensor_bytes"] else 0.0,
        }
    return out


def _mib(n: float) -> str:
    return f"{n / 2**20:,.1f}"


def format_report(rep: dict, top: int = 20) -> str:
    """The report as aligned text."""
    lin = rep["linears"]
    lines = [
        f"{rep['path']}  ({rep['layout']} layout, {rep['shards']} shard(s), "
        f"{_mib(rep['file_bytes'])} MiB on disk)",
        "",
        f"Quantized linears: {lin['bases']} tensor(s), {lin['weights'] / 1e6:,.1f} M "
        f"weights, {_mib(lin['bytes'])} MiB, {lin['bpw']:.3f} bpw",
    ]
    for g in GROUPS:
        if lin["by_field"][g]:
            lines.append(
                f"  {g:<18}{_mib(lin['by_field'][g]):>12} MiB{lin['bpw_by_field'][g]:>9.3f} bpw"
            )
    share = ", ".join(f"{k} {v:.1%}" for k, v in lin["precision_share"].items())
    lines += [f"  weights by precision: {share}", ""]
    lines.append(f"Dense tensors kept: {_mib(rep['dense_bytes'])} MiB")
    for k, v in rep["dense"].items():
        lines.append(f"  {k:<46}{_mib(v):>12} MiB")
    lines += ["", f"Tensor bytes in total: {_mib(rep['tensor_bytes'])} MiB"]
    if "source" in rep:
        s = rep["source"]
        lines.append(
            f"Source tensors: {_mib(s['tensor_bytes'])} MiB, "
            f"compression {s['ratio']:.2f}x"
        )
    lines += ["", f"By {rep['by']} (largest first):"]
    lines.append(f"  {'name':<44}{'tensors':>8}{'M weights':>11}{'MiB':>10}{'bpw':>8}")
    for i, (name, r) in enumerate(rep["rows"].items()):
        if i == top:
            lines.append(f"  ... {len(rep['rows']) - top} more (use --top)")
            break
        lines.append(
            f"  {name:<44}{r['bases']:>8}{r['weights'] / 1e6:>11.2f}"
            f"{_mib(r['bytes']):>10}{r['bpw']:>8.3f}"
        )
    return "\n".join(lines)
