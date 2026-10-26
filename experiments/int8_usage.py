# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Which tensors the quality allocation gives INT8 rows, at given ratios (from a finished phase 1
with --bits 2,4,8), plus each tensor's sensitivity and the projection/layer totals.
usage: int8_usage.py <workdir> <ratio> [ratio ...]"""

import json
import sys
from pathlib import Path

import torch

from hi2.allocate import allocate_by_quality

torch.set_num_threads(4)
wd = Path(sys.argv[1])
tdir = wd / "tables"
ref = wd / "cache_ref.json"
if not tdir.is_dir() and ref.exists():
    tdir = Path(json.loads(ref.read_text())["tables"]) / "tables"
tables = [torch.load(p, weights_only=False) for p in sorted(tdir.glob("*.pt"))]
sp = wd / "sensitivity.json"
sens = json.loads(sp.read_text()) if sp.exists() else {}
rank = {n: i for i, n in enumerate(sorted(sens, key=sens.get, reverse=True))}
for r in map(float, sys.argv[2:]):
    a = allocate_by_quality(tables, r, sens)
    rows = w8 = wt = 0
    hits = []
    for t in tables:
        ci = [i for i, (nb, _) in enumerate(t.candidates) if nb == 8]
        p = a.picks[t.name]
        n8 = int(sum((p == i).sum() for i in ci)) if ci else 0
        rows += t.shape[0]
        wt += t.shape[0] * t.shape[1]
        w8 += n8 * t.shape[1]
        if n8:
            hits.append((t.name, n8, t.shape[0], sens.get(t.name, 1.0)))
    print(
        f"ratio {r:g}: bpw {a.bpw:.4f}, INT8 rows {sum(h[1] for h in hits)} of {rows} "
        f"({100 * w8 / wt:.3f} % of weights) in {len(hits)} tensors"
    )
    for name, n8, rr, s in sorted(hits, key=lambda h: -h[1])[:25]:
        print(
            f"  {name.replace('model.language_model.', '')}: {n8}/{rr} rows, sens {s:.2f} (rank {rank.get(name, -1)})"
        )
