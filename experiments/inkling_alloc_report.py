# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Where the bits went: the allocation of a quantized Inkling-2 run by layer group, projection and layer.

Reads the ``hi2_report.json`` of a run (or the run directory) and tabulates, for the routed
experts, the share of INT2 / INT4 / INT8 by weights, the block mix (16 / 32 / 64) by rows,
the bits per weight (everything the format stores: codes, block parameters, row constants)
and the implied size next to the stock quantized experts (NVFP4 at 4.5 bits in layers 0-19,
MXFP8 at 8.25 bits in layers 20-23). Nothing is measured here: the numbers are the
allocator's decisions, as the encoder reported them tensor by tensor.

    python experiments/inkling_alloc_report.py RUN_OR_REPORT [--json OUT.json] [--markdown]
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXPERT = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$"
)
LAYER_GROUPS = (
    ("L0-5", 0, 5),
    ("L6-17", 6, 17),
    ("L18-19", 18, 19),
    ("L20-23", 20, 23),
)
STOCK_BPW = {"nvfp4": 4.5, "mxfp8": 8.25}
MXFP8_FROM_LAYER = 20


def allocation_by_group(report: dict) -> dict[str, dict]:
    """Shares and bits per weight of the routed experts by scope: ``all``, the layer groups,
    the three projections and every layer (``layer 00``..)."""
    acc: dict[str, dict] = {}
    for name, t in report["tensors"].items():
        m = EXPERT.search(name)
        if m is None or "shape" not in t:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        rows, cols = t["shape"]
        weights = rows * cols
        f4, f8 = t.get("frac_int4", 0.0), t.get("frac_int8", 0.0)
        hist = t.get("block_hist", {})
        group = next(g for g, lo, hi in LAYER_GROUPS if lo <= layer <= hi)
        stock = STOCK_BPW["mxfp8" if layer >= MXFP8_FROM_LAYER else "nvfp4"]
        for scope in ("all", group, proj, f"layer {layer:02d}"):
            a = acc.setdefault(
                scope,
                dict.fromkeys(
                    (
                        "tensors",
                        "weights",
                        "w2",
                        "w4",
                        "w8",
                        "bits",
                        "stock_bits",
                        "rows",
                        "b16",
                        "b32",
                        "b64",
                    ),
                    0,
                ),
            )
            a["tensors"] += 1
            a["weights"] += weights
            a["w4"] += f4 * weights
            a["w8"] += f8 * weights
            a["w2"] += max(0.0, 1.0 - f4 - f8) * weights
            a["bits"] += t["bpw"] * weights
            a["stock_bits"] += stock * weights
            a["rows"] += sum(hist.values()) or rows
            for b in ("16", "32", "64"):
                a["b" + b] += hist.get(b, 0)
    out = {}
    for scope, a in acc.items():
        w, rows = a["weights"], max(a["rows"], 1)
        out[scope] = {
            "tensors": a["tensors"],
            "weights_g": round(w / 1e9, 4),
            "int2": round(a["w2"] / w, 4),
            "int4": round(a["w4"] / w, 4),
            "int8": round(a["w8"] / w, 4),
            "b16": round(a["b16"] / rows, 4),
            "b32": round(a["b32"] / rows, 4),
            "b64": round(a["b64"] / rows, 4),
            "bpw": round(a["bits"] / w, 4),
            "gib": round(a["bits"] / 8 / 2**30, 3),
            "stock_gib": round(a["stock_bits"] / 8 / 2**30, 3),
        }
    return out


def render(alloc: dict[str, dict], title: str = "") -> str:
    head = (
        "| scope | tensors | weights G | INT2 % | INT4 % | INT8 % | B16 % | B32 % | B64 % "
        "| bpw | GiB | stock GiB |"
    )
    sep = "| " + " | ".join(["---"] * 12) + " |"

    def row(scope: str) -> str:
        r = alloc[scope]
        return (
            f"| {scope} | {r['tensors']} | {r['weights_g']:.3f} | {100 * r['int2']:.1f} "
            f"| {100 * r['int4']:.1f} | {100 * r['int8']:.1f} | {100 * r['b16']:.1f} "
            f"| {100 * r['b32']:.1f} | {100 * r['b64']:.1f} | {r['bpw']:.3f} "
            f"| {r['gib']:.3f} | {r['stock_gib']:.3f} |"
        )

    scopes = [
        s
        for s in (
            "all",
            *(g for g, _, _ in LAYER_GROUPS),
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        if s in alloc
    ]
    layers = sorted(s for s in alloc if s.startswith("layer "))
    lines = [title] if title else []
    lines += [
        head,
        sep,
        *map(row, scopes),
        "",
        "per layer:",
        "",
        head,
        sep,
        *map(row, layers),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "report", type=Path, help="hi2_report.json or the directory holding it"
    )
    ap.add_argument(
        "--json", type=Path, default=None, help="also write the numbers here"
    )
    ap.add_argument(
        "--markdown", action="store_true", help="print the tables as Markdown"
    )
    a = ap.parse_args(argv)
    path = a.report / "hi2_report.json" if a.report.is_dir() else a.report
    report = json.loads(path.read_text())
    alloc = allocation_by_group(report)
    if not alloc:
        print(f"{path}: no routed expert tensor in the report", file=sys.stderr)
        return 1
    if a.json is not None:
        a.json.write_text(json.dumps(alloc, indent=1) + "\n")
    if a.markdown:
        print(render(alloc, f"Allocation of {path}"))
    else:
        r = alloc["all"]
        print(
            f"{path}: {r['weights_g']:.3f} G expert weights, INT2 {100 * r['int2']:.1f} % "
            f"INT4 {100 * r['int4']:.1f} % INT8 {100 * r['int8']:.1f} %, {r['bpw']:.3f} bits "
            f"per weight = {r['gib']:.2f} GiB (stock {r['stock_gib']:.2f} GiB)"
        )
        for scope in (g for g, _, _ in LAYER_GROUPS if g in alloc):
            r = alloc[scope]
            print(
                f"  {scope:7s} INT2 {100 * r['int2']:5.1f} % INT4 {100 * r['int4']:5.1f} % "
                f"INT8 {100 * r['int8']:5.1f} %  {r['bpw']:.3f} bpw  {r['gib']:.3f} GiB"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
