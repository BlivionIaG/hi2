# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fidelity table (ppl ratio, KLD, top-1 on both held-out texts, bpw, INT2 share, runtime) of runs.
usage: summarize_qad.py <runs dir> <tag> [tag ...]   (reads TAG.eval.log, TAG.evalmix.log,
TAG/hi2_report.json, TAG.quant.log and TAG.t0/t1)"""

import json
import sys
from pathlib import Path

import regex as re


def last_json(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    i = txt.rfind("{\n")
    try:
        return json.loads(txt[i : txt.rfind("}") + 1])
    except ValueError:
        return None


runs = Path(sys.argv[1])
print(
    "| run | bpw | INT2 rows | INT8 rows | docs ppl ratio | docs KLD | docs top-1 | mix ppl ratio | mix KLD | mix top-1 | quantize min |"
)
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for tag in sys.argv[2:]:
    cells = []
    rep = runs / tag / "hi2_report.json"
    s = json.loads(rep.read_text())["summary"] if rep.exists() else {}
    log = (
        (runs / f"{tag}.quant.log").read_text(errors="ignore")
        if (runs / f"{tag}.quant.log").exists()
        else ""
    )
    m = re.search(r"INT2 ([0-9.]+)%", log)
    cells += [f"{s.get('avg_bpw', float('nan')):.3f}", f"{m.group(1)}%" if m else "n/a"]
    cells.append(f"{100 * s.get('frac_int8_rows', 0.0):.3f}%")
    for kind in ("eval", "evalmix"):
        r = last_json(runs / f"{tag}.{kind}.log")
        if r is None:
            cells += ["n/a"] * 3
            continue
        cells += [
            f"{r['ppl_quantized'] / r['ppl_original']:.4f}",
            f"{r['kld_nats_per_token']:.4f}",
            f"{100 * r['top1_agreement']:.1f}%",
        ]
    try:
        t = int((runs / f"{tag}.t1").read_text()) - int(
            (runs / f"{tag}.t0").read_text()
        )
        cells.append(f"{t / 60:.0f}")
    except (OSError, ValueError):
        cells.append("n/a")
    print(f"| {tag} | " + " | ".join(cells) + " |")
