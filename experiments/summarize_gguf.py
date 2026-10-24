# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Summarise GGUF baseline logs: llama.cpp perplexity, whole-file bpw and
linear-layers-only bpw (the attention and MLP weights, comparable to hi2's bpw).

    python summarize_gguf.py <gguf_dir> [--json out.json]
"""

import json
import sys
from pathlib import Path

import regex as re

LINE = re.compile(
    r"\[\s*\d+/\s*\d+\]\s+(\S+)\s+-\s+\[\s*(\d+),\s*(\d+),.*?type =\s+(\S+)(?:, converting to\s+(\S+))?.*?size =\s+([\d.]+) MiB(?: ->\s+([\d.]+) MiB)?"
)
PPL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def linears_bpw(log: Path):
    bits = params = 0.0
    kinds: dict[str, int] = {}
    for ln in log.read_text(errors="ignore").splitlines():
        m = LINE.search(ln)
        if not m:
            continue
        name, ncols, nrows, _src, dst, _size_in, size_out = m.groups()
        if not name.startswith("blk.") or not name.endswith(".weight"):
            continue
        if "norm" in name or "bias" in name or int(nrows) == 1:
            continue
        size = float(size_out if size_out else m.group(6))
        bits += size * 2**20 * 8
        params += int(ncols) * int(nrows)
        kinds[dst or _src] = kinds.get(dst or _src, 0) + int(ncols) * int(nrows)
    return (bits / params if params else None), kinds


def main(d: Path):
    rows = {}
    for ppl_log in sorted(d.glob("*.ppl.log")):
        tag = ppl_log.name[: -len(".ppl.log")]
        m = PPL.search(ppl_log.read_text(errors="ignore"))
        if not m:
            continue
        row = {"ppl": float(m.group(1)), "ppl_err": float(m.group(2))}
        q = d / f"{tag}.quant.log"
        if q.exists():
            txt = q.read_text(errors="ignore")
            mm = re.search(r"quant size\s+=\s+[\d.]+ MiB \(([\d.]+) BPW\)", txt)
            row["file_bpw"] = float(mm.group(1)) if mm else None
            row["linears_bpw"], row["types"] = linears_bpw(q)
        rows[tag] = row
    return rows


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    base = out.get("f16", {}).get("ppl")
    print(f"f16 ppl {base}")
    print(
        f"{'type':22s} {'lin_bpw':>8s} {'file_bpw':>9s} {'ppl':>8s} {'ppl/f16':>8s}  fallbacks"
    )
    for tag, r in sorted(out.items(), key=lambda kv: kv[1].get("linears_bpw") or 99):
        if tag == "f16":
            continue
        lb, fb = r.get("linears_bpw"), r.get("file_bpw")
        print(
            f"{tag:22s} {lb if lb is None else round(lb, 3)!s:>8} {fb!s:>9} {r['ppl']:8.3f} {r['ppl'] / base:8.3f}  {','.join(sorted(r.get('types', {})))}"
        )
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps(out, indent=1)
        )
