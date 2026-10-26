# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Re-tabulate a cached run from its cached statistics (``tables_from_stats``, no model
forward) and compare every table with the cached one built during the calibration
pass: sse, energy, flags, smoothing vector, bit for bit. Prints the time.

usage: python experiments/check_tables.py MODEL WORKDIR [--device cuda:0]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from hi2.calibrate import Config
from hi2.executor import Executor
from hi2.layerwise import tables_from_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ref = json.loads((a.workdir / "cache_ref.json").read_text())
    tkey = json.loads((Path(ref["tables"]) / "key.json").read_text())
    cfg = Config(
        candidates=[tuple(c) for c in tkey["candidates"]],
        rotation=tkey["rotation"],
        smooth=tkey["smooth"],
        effort=tkey["effort"],
    )
    skey = json.loads((Path(ref["stats"]) / "key.json").read_text())
    t0 = time.perf_counter()
    new = tables_from_stats(
        a.model,
        Path(ref["stats"]),
        cfg,
        Executor([a.device]),
        skey["compute_dtype"],
        log=lambda *_: None,
    )
    dt = time.perf_counter() - t0
    bad = []
    for name, t in new.items():
        old = torch.load(
            Path(ref["tables"]) / "tables" / (name.replace("/", "_") + ".pt"),
            weights_only=False,
        )
        same = (
            torch.equal(t.sse, old.sse)
            and t.energy == old.energy
            and torch.equal(t.had, old.had)
            and t.n_tokens == old.n_tokens
            and (
                (t.col_scale is None and old.col_scale is None)
                or torch.equal(t.col_scale, old.col_scale)
            )
        )
        if not same:
            bad.append(name)
    print(
        json.dumps(
            {
                "tables": len(new),
                "differ": len(bad),
                "first": bad[:5],
                "seconds": round(dt, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
