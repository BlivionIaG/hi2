# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""torch.profiler table of the new encode of a few real tensors (where the remaining
time goes). usage: python experiments/profile_new.py MODEL WORKDIR --match REGEX"""

import argparse
import json
import time
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

from hi2.allocate import allocate_global
from hi2.cache import StatsView
from hi2.calibrate import Config, EncodeRequest, encode_many
from hi2.checkpoint import weight_map
from hi2.executor import Executor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--match", default=r"layers\.3\.mlp\.(gate|down)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ref = json.loads((a.workdir / "cache_ref.json").read_text())
    tdir = Path(ref["tables"]) / "tables"
    tables = {
        p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
        for p in tdir.iterdir()
    }
    alloc = allocate_global(list(tables.values()), 3.0, None)
    view, wmap, ex, cfg = (
        StatsView(a.workdir),
        weight_map(a.model),
        Executor([a.device]),
        Config(),
    )
    reqs = []
    for n in sorted(x for x in tables if re.search(a.match, x)):
        with safe_open(a.model / wmap[n], "pt") as f:
            w = f.get_tensor(n)
        reqs.append(
            EncodeRequest(
                w, None, tables[n], alloc.picks[n], view.sigma(n), view.sigma_id(n)
            )
        )
    encode_many(reqs[:1], cfg, ex)  # warm up (kernel load)
    torch.accelerator.synchronize(dev)
    for r in reqs:
        t0 = time.perf_counter()
        encode_many([r], cfg, ex)
        torch.accelerator.synchronize(dev)
        print(r.table.name, "wall", round(time.perf_counter() - t0, 3), flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        encode_many(reqs, cfg, ex)
        torch.accelerator.synchronize(dev)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))


if __name__ == "__main__":
    main()
