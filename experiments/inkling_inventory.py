# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Inventory of the stock Inkling-2-Mini checkpoint, from the safetensors headers only.

For every tensor class (``hi2.inkling_source.tensor_role``, the experts split by their
format, the MTP file by its own rules) the script counts tensors, logical elements, bytes
and the stock bits per element, writes everything to a JSON file and checks it against
the numbers of the design review (NVFP4 experts 11,466,178,560 B at 4.50 bpw, MXFP8
experts 4,204,265,472 B at 8.25 bpw, 128 experts x 7,962,624 weights per layer,
embeddings 463 M each, attention about 645 M, non-experts about 3.86 GB, about 26.4 B
parameters). Every discrepancy is printed and recorded under ``discrepancies``; no tensor
data is read.

Two of the review's figures are definitions, not counts, and the checks say which:
"26.4 B parameters" is the main file (text and vision, without the MTP file), and the
totals "18.68 GiB stock / 14.05 GiB at 3.5 bpw" are the main file plus the MTP file
without ``markov_head.markov_w1`` (the 201,024 x 256 table of the Markov head).

    python experiments/inkling_inventory.py ~/hfcache/inkling2-mini --out inventory.json
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import regex as re

from hi2.inkling_source import tensor_role

GIB = 2**30
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1, "F8_E4M3": 1, "I64": 8}
INDEX = re.compile(r"\.\d+(?=\.|$)")
EXTRAS = ("_amax", "_outer_scale_denominator")  # per-expert NVFP4 constants


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def mtp_role(name: str) -> str:
    for rx, role in (
        (r"^markov_head\.", "mtp_markov_head"),
        (r"\.self_attn\.(q|k|v|o)_proj(_ctx)?\.weight$", "mtp_attn_linear"),
        (r"\.mlp\.(gate|up|down)_proj\.weight$", "mtp_mlp_linear"),
        (r"norm", "mtp_norm"),
    ):
        if re.search(rx, name):
            return role
    return "mtp_other"


def classify(files: dict[str, dict]) -> dict[str, dict]:
    """``{name: {file, dtype, shape, bytes, elements, cls, weights}}`` of every tensor.

    ``elements`` is the stored element count; ``weights`` is the logical count of the
    tensors that carry weights (an NVFP4 byte holds two; scales and amax are overhead)
    and the stored count for every other class.
    """
    fmt: dict[tuple[str, int], str] = {}  # (file, layer) -> expert format
    for fn, hdr in files.items():
        for name, h in hdr.items():
            m = re.match(r"^.*\.layers\.(\d+)\.mlp\.experts\.w13_weight$", name)
            if m:
                fmt[(fn, int(m.group(1)))] = "nvfp4" if h["dtype"] == "U8" else "mxfp8"
    out: dict[str, dict] = {}
    for fn, hdr in files.items():
        for name, h in hdr.items():
            n = math.prod(h["shape"])
            nbytes = h["data_offsets"][1] - h["data_offsets"][0]
            if nbytes != n * DTYPE_BYTES[h["dtype"]]:
                raise ValueError(f"{name}: {nbytes} B for {h['dtype']} {h['shape']}")
            cls = mtp_role(name) if fn.startswith("mtp") else tensor_role(name)
            weights = n
            if cls == "expert":
                layer = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
                cls = f"expert_{fmt[(fn, layer)]}"
                carries = name.endswith(("w13_weight", "w2_weight"))
                weights = n * (2 if cls == "expert_nvfp4" else 1) if carries else 0
            out[name] = {
                "file": fn,
                "dtype": h["dtype"],
                "shape": h["shape"],
                "bytes": nbytes,
                "elements": n,
                "cls": cls,
                "weights": weights,
            }
    return out


def summarize(tensors: dict[str, dict]) -> list[dict]:
    """One row per class: counts, bytes, stock bits per weight, name patterns."""
    rows: dict[str, dict] = {}
    for name, t in tensors.items():
        r = rows.setdefault(
            t["cls"],
            {
                "class": t["cls"],
                "tensors": 0,
                "elements": 0,
                "weights": 0,
                "bytes": 0,
                "patterns": {},
            },
        )
        r["tensors"] += 1
        r["elements"] += t["elements"]
        r["weights"] += t["weights"]
        r["bytes"] += t["bytes"]
        key = f"{INDEX.sub('.N', name)} {t['dtype']} {t['shape']}"
        r["patterns"][key] = r["patterns"].get(key, 0) + 1
    out = []
    for r in sorted(rows.values(), key=lambda r: -r["bytes"]):
        r["stock_bpw"] = 8 * r["bytes"] / r["weights"] if r["weights"] else None
        r["patterns"] = [{"pattern": k, "count": v} for k, v in r["patterns"].items()]
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=Path("inkling_inventory.json"))
    a = ap.parse_args()

    cfg = json.loads((a.ckpt / "config.json").read_text())["text_config"]
    index = json.loads((a.ckpt / "model.safetensors.index.json").read_text())
    files = {}
    for p in sorted(a.ckpt.glob("*.safetensors")):
        hdr = read_header(p)
        hdr.pop("__metadata__", None)
        files[p.name] = hdr
    tensors = classify(files)
    rows = summarize(tensors)
    by = {r["class"]: r for r in rows}
    main_t = {n: t for n, t in tensors.items() if not t["file"].startswith("mtp")}
    mtp_t = {n: t for n, t in tensors.items() if t["file"].startswith("mtp")}

    def total(sel, key="bytes"):
        return sum(t[key] for t in sel.values())

    nvfp4_ws = total(
        {
            n: t
            for n, t in tensors.items()
            if t["cls"] == "expert_nvfp4" and not n.endswith(EXTRAS)
        }
    )  # weights + scales
    mxfp8_ws = by["expert_mxfp8"]["bytes"]
    experts_w = by["expert_nvfp4"]["weights"] + by["expert_mxfp8"]["weights"]
    expert_bytes = by["expert_nvfp4"]["bytes"] + by["expert_mxfp8"]["bytes"]
    non_expert_main = total(
        {n: t for n, t in main_t.items() if not t["cls"].startswith("expert_")}
    )
    markov_w1 = tensors["markov_head.markov_w1.weight"]["bytes"]
    mtp_bytes = total(mtp_t)
    params_main = sum(
        t["weights"] if t["cls"].startswith("expert_") else t["elements"]
        for t in main_t.values()
    )
    params_mtp = total(mtp_t, "elements")
    fmt_by_layer = {}
    for n, t in tensors.items():
        m = re.match(r"^.*\.layers\.(\d+)\.mlp\.experts\.", n)
        if m:
            fmt_by_layer[int(m.group(1))] = t["cls"].removeprefix("expert_")
    per_expert = 3 * cfg["hidden_size"] * cfg["intermediate_size"]

    print(f"{a.ckpt}: {len(tensors)} tensors in {list(files)}")
    for r in rows:
        bpw = "-" if r["stock_bpw"] is None else f"{r['stock_bpw']:.3f}"
        print(
            f"  {r['class']:<18} {r['tensors']:>5} tensors {r['weights']:>16,} weights "
            f"{r['bytes']:>16,} B  stock bpw {bpw}"
        )

    sizes = {}
    for bpw in (3.0, 3.5, 4.0):
        e = experts_w * bpw / 8
        sizes[str(bpw)] = {
            "experts_gib": e / GIB,
            "total_main_file_gib": (e + non_expert_main) / GIB,
            "total_text_only_gib": (e + non_expert_main - by["vision"]["bytes"]) / GIB,
            "total_review_definition_gib": (e + non_expert_main + mtp_bytes - markov_w1)
            / GIB,
        }
    review_stock = (expert_bytes + non_expert_main + mtp_bytes - markov_w1) / GIB
    attn = by["attn_linear"]["weights"]

    checks = []

    def check(label, got, expected, rel=0.0, note=None):
        if isinstance(expected, tuple):  # a range
            ok = expected[0] <= got <= expected[1]
        else:
            ok = abs(got - expected) <= rel * abs(expected)
        checks.append(
            {
                "check": label,
                "expected": expected,
                "got": got,
                "ok": bool(ok),
                "note": note,
            }
        )
        print(
            f"  {'ok  ' if ok else 'DIFF'} {label}: got {got:,.4g} expected {expected}"
        )

    print("checks against the design review:")
    check("experts NVFP4 bytes (weights + scales)", nvfp4_ws, 11_466_178_560)
    check("experts MXFP8 bytes (weights + scales)", mxfp8_ws, 4_204_265_472)
    check("NVFP4 stock bpw", 8 * nvfp4_ws / by["expert_nvfp4"]["weights"], 4.5)
    check("MXFP8 stock bpw", 8 * mxfp8_ws / by["expert_mxfp8"]["weights"], 8.25)
    check("NVFP4 layers", sum(v == "nvfp4" for v in fmt_by_layer.values()), 20)
    check("MXFP8 layers", sum(v == "mxfp8" for v in fmt_by_layer.values()), 4)
    check("routed experts per layer", cfg["n_routed_experts"], 128)
    check("weights per expert", per_expert, 7_962_624)
    check("embedding elements", by["embed"]["elements"], 463_159_296)
    check("unembedding elements", by["unembed"]["elements"], 463_159_296)
    check("attention linear elements (q k v o gate wr)", attn, (640e6, 650e6))
    check(
        "non-expert bytes of the main file (incl. vision)",
        non_expert_main,
        3.86e9,
        rel=0.005,
    )
    check(
        "index metadata.total_size = main file bytes",
        total(main_t),
        index["metadata"]["total_size"],
    )
    check(
        "parameters of the main file (text + vision)",
        params_main,
        26.4e9,
        rel=0.005,
        note="the review's 26.4 B excludes the MTP file; MTP adds "
        f"{params_mtp / 1e9:.3f} B ({(params_main + params_mtp) / 1e9:.3f} B in all)",
    )
    check(
        "stock experts GiB (weights, scales, amax)",
        expert_bytes / GIB,
        14.59,
        rel=0.001,
    )
    check(
        "experts at 3.5 bpw GiB",
        sizes["3.5"]["experts_gib"],
        9.96,
        rel=0.001,
        note="9.967 GiB: the review truncated the third decimal",
    )
    check(
        "stock total GiB",
        review_stock,
        18.68,
        rel=0.001,
        note="definition: main file + MTP file - markov_head.markov_w1 "
        f"(main file alone is {(expert_bytes + non_expert_main) / GIB:.2f} GiB)",
    )
    check(
        "HI2 total GiB at 3.5 bpw",
        sizes["3.5"]["total_review_definition_gib"],
        14.05,
        rel=0.001,
        note="same definition; the main file alone is "
        f"{sizes['3.5']['total_main_file_gib']:.2f} GiB",
    )
    discrepancies = [c for c in checks if not c["ok"]]
    result = {
        "checkpoint": str(a.ckpt),
        "config": {
            k: cfg[k]
            for k in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "n_routed_experts",
                "num_experts_per_tok",
                "vocab_size",
                "unpadded_vocab_size",
            )
        },
        "files": {
            f: {
                "tensors": len(h),
                "bytes": total({n: t for n, t in tensors.items() if t["file"] == f}),
            }
            for f, h in files.items()
        },
        "expert_format_by_layer": dict(sorted(fmt_by_layer.items())),
        "classes": rows,
        "totals": {
            "tensors": len(tensors),
            "bytes_main_file": total(main_t),
            "bytes_mtp_file": mtp_bytes,
            "bytes_vision": by["vision"]["bytes"],
            "bytes_non_expert_main_file": non_expert_main,
            "bytes_markov_w1": markov_w1,
            "expert_weights": experts_w,
            "params_main_file": params_main,
            "params_mtp_file": params_mtp,
            "stock_total_gib_review_definition": review_stock,
        },
        "sizes_by_expert_bpw": sizes,
        "checks": checks,
        "discrepancies": discrepancies,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.out}; {len(discrepancies)} discrepancies")
    for d in discrepancies:
        print(
            f"  DISCREPANCY {d['check']}: expected {d['expected']}, got {d['got']} {d['note'] or ''}"
        )
    for c in checks:
        if c["note"]:
            print(f"  note on '{c['check']}': {c['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
