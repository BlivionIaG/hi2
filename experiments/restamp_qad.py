# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Reuse the per-layer QAD artifacts of an older run whose fingerprint fields have since changed.

<dst qad dir> holds the first layers recomputed by the current code (current fingerprint) and
the remaining layers copied from <src qad dir> (old fingerprint). Every recomputed layer must
equal the source's layer bit for bit (codes, block parameters, row constants); then the copied
layers get the current fingerprint, so the next run with the same command reuses them.
usage: restamp_qad.py <src qad dir> <dst qad dir>"""

import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
recs = {p.name: torch.load(p, weights_only=False) for p in sorted(dst.glob("layer*.pt"))}
old = {n: torch.load(src / n, weights_only=False) for n in recs}
fps = {r["fingerprint"] for r in recs.values()}
old_fp = {r["fingerprint"] for r in old.values()}
new = [n for n, r in recs.items() if r["fingerprint"] not in old_fp]
if not new or len({recs[n]["fingerprint"] for n in new}) != 1:
    raise SystemExit("need at least one layer recomputed by the current code")
cur = recs[new[0]]["fingerprint"]
for n in new:
    a, b = old[n]["tensors"], recs[n]["tensors"]
    for k in a:
        for f in ("codes", "blk_params", "row_consts", "row_nbits", "row_block"):
            if not torch.equal(getattr(a[k], f), getattr(b[k], f)):
                raise SystemExit(f"{n} {k}.{f} differs: not the same computation")
print(f"{len(new)} recomputed layers equal the source bit for bit ({', '.join(new)})")
done = 0
for n, r in recs.items():
    if r["fingerprint"] != cur:
        r["fingerprint"] = cur
        torch.save(r, dst / n)
        done += 1
print(f"{done} copied layers re-stamped with {cur} ({len(fps)} fingerprints before)")
