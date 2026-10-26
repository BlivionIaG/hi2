# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Bit-identity of two HI2 checkpoints that differ only in MoE layout.

For every logical HI2 tensor (per-expert name) the nine stored arrays must be
byte-identical (dtype, shape, values) and the float64 dequantization (original basis)
torch.equal; every dense tensor must be identical too. Prints a JSON summary with the
tensor counts, safetensors header and file sizes of both checkpoints.

usage: python experiments/compare_layouts.py A B
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from hi2.checkpoint import weight_map
from hi2.format import FIELDS, dequantize
from hi2.stack import Hi2Checkpoint, checkpoint_stats


def dense_tensors(d: Path) -> dict[str, tuple[str, str]]:
    wmap = weight_map(d)
    return {k: (v, k) for k, v in wmap.items() if ".hi2_" not in k}


def main(a: Path, b: Path) -> dict:
    t0 = time.time()
    ca, cb = Hi2Checkpoint(a), Hi2Checkpoint(b)
    la, lb = ca.logical(), cb.logical()
    if set(la) != set(lb):
        raise SystemExit(f"logical tensors differ: {len(set(la) ^ set(lb))}")
    arrays = deq = 0
    worst = []
    for i, base in enumerate(sorted(la)):
        ta, tb = ca.get(base), cb.get(base)
        for f in (*FIELDS, "col_scale"):
            x, y = getattr(ta, f), getattr(tb, f)
            if (x is None) != (y is None):
                raise SystemExit(f"{base}.{f}: present in one layout only")
            if x is None:
                continue
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                raise SystemExit(f"{base}.{f}: arrays differ")
            arrays += 1
        if ta.shape != tb.shape:
            raise SystemExit(f"{base}: shape differs")
        wa, wb = dequantize(ta), dequantize(tb)
        if not torch.equal(wa, wb):
            worst.append(base)
        deq += 1
        if i % 500 == 0:
            print(f"{i}/{len(la)} {base}", file=sys.stderr, flush=True)
    if worst:
        raise SystemExit(f"dequantized weights differ: {worst[:5]}")
    da, db = dense_tensors(a), dense_tensors(b)
    if da.keys() != db.keys():
        raise SystemExit("dense tensor sets differ")
    for k in da:
        with safe_open(a / da[k][0], "pt") as f, safe_open(b / db[k][0], "pt") as g:
            x, y = f.get_tensor(k), g.get_tensor(k)
            if x.dtype != y.dtype or not torch.equal(x, y):
                raise SystemExit(f"dense {k} differs")
    sa, sb = checkpoint_stats(a), checkpoint_stats(b)
    return {
        "a": str(a),
        "b": str(b),
        "logical_hi2_tensors": len(la),
        "stacked_in_a": len(ca.stacked),
        "stacked_in_b": len(cb.stacked),
        "arrays_identical": arrays,
        "dequantized_identical": deq,
        "dense_identical": len(da),
        "a_stats": sa,
        "b_stats": sb,
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
