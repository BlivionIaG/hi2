# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Map --max-error-ratio to the resulting bpw and INT2/INT4 row shares, from a finished phase 1.
usage: ratio_table.py <workdir> [ratio ...]   (uses <workdir>/tables and sensitivity.json if present)"""

import json
import sys
from pathlib import Path

import torch
from hi2.allocate import allocate_by_quality

torch.set_num_threads(4)  # tiny ops: more threads only contend
wd = Path(sys.argv[1])
ratios = [float(r) for r in sys.argv[2:]] or [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16]
tables = [
    torch.load(p, weights_only=False) for p in sorted((wd / "tables").glob("*.pt"))
]
sp = wd / "sensitivity.json"
sens = json.loads(sp.read_text()) if sp.exists() else None
print(f"{len(tables)} tensors, sensitivity {'yes' if sens else 'no'}")
print("ratio  bpw     INT2%  INT4%  INT8%")
for r in ratios:
    a = allocate_by_quality(tables, r, sens)
    f = a.frac_by_bits or {}
    print(
        f"{r:<6g} {a.bpw:.4f}  {100 * f.get(2, 0):5.1f}  {100 * f.get(4, 0):5.1f}  {100 * f.get(8, 0):5.1f}"
    )
