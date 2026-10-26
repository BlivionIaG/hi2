# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Map --max-error-ratio to the resulting bpw and INT2/INT4/INT8 row shares, from a finished phase 1.
usage: ratio_table.py <workdir> [ratio ...]   (uses <workdir>/tables, or the cache entry named in
       <workdir>/cache_ref.json, and sensitivity.json if present)
       ratio_table.py <workdir> --match BPW [BPW ...]   (the ratio whose allocation lands on each bpw)"""

import json
import sys
from pathlib import Path

import torch

from hi2.allocate import allocate_by_quality

torch.set_num_threads(4)  # tiny ops: more threads only contend
wd = Path(sys.argv[1])
args = sys.argv[2:]
match = args[:1] == ["--match"]
if match:
    args = args[1:]
ratios = [float(r) for r in args] or [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16]
tdir = wd / "tables"
ref = wd / "cache_ref.json"
if not tdir.is_dir() and ref.exists():
    tdir = Path(json.loads(ref.read_text())["tables"]) / "tables"
tables = [torch.load(p, weights_only=False) for p in sorted(tdir.glob("*.pt"))]
sp = wd / "sensitivity.json"
sens = json.loads(sp.read_text()) if sp.exists() else None
print(f"{len(tables)} tensors, sensitivity {'yes' if sens else 'no'}")
print("ratio  bpw     INT2%  INT4%  INT8%")
# --match: bisect the ratio (log scale) for each target bpw; bpw falls as the ratio grows
if match:
    found = []
    for target in ratios:
        lo, hi = 1.0, 1e6
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            if allocate_by_quality(tables, mid, sens).bpw > target:
                lo = mid
            else:
                hi = mid
        found.append(float(f"{hi:.4g}"))
    ratios = found
for r in ratios:
    a = allocate_by_quality(tables, r, sens)
    f = a.frac_by_bits or {}
    print(
        f"{r:<6g} {a.bpw:.4f}  {100 * f.get(2, 0):5.1f}  {100 * f.get(4, 0):5.1f}  {100 * f.get(8, 0):5.1f}"
    )
