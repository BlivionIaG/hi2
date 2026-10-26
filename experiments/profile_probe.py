# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""cProfile of the sensitivity probe on the first units of a finished cached run, and
of the cache write of one layer's Sigma.

usage: python experiments/profile_probe.py MODEL WORKDIR CALIB --units 24
"""

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import torch
from hi2.cache import sigma_payload
from hi2.calibrate import Config
from hi2.executor import Executor
from hi2.pipeline import load_calibration_ids
from hi2.sensitivity import probe_sensitivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("calib", type=Path)
    ap.add_argument("--units", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sink-only", action="store_true")
    a = ap.parse_args()
    if a.sink_only:
        a.units = 0
    from transformers import AutoTokenizer

    ref = json.loads((a.workdir / "cache_ref.json").read_text())
    tdir = Path(ref["tables"]) / "tables"
    tables = {
        p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
        for p in tdir.iterdir()
    }
    names = sorted(tables)[: a.units]
    ids = load_calibration_ids(
        a.calib, AutoTokenizer.from_pretrained(a.model), 8, 2048
    )[:, :512]
    ex = Executor([a.device])
    if a.units == 0:
        names = []
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    if names:
        probe_sensitivity(
            a.model,
            tables,
            a.workdir,
            ids,
            Config(),
            ex,
            units=[[(n, 1.0)] for n in names],
            log=lambda *_: None,
        )
    prof.disable()
    print(f"probe of {len(names)} units: {time.perf_counter() - t0:.1f} s")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
    dev = torch.device(a.device)
    for k in (1024, 3584, 9216):
        x = torch.randn(4096, k, device=dev)
        s = x.T @ x
        torch.accelerator.synchronize(dev)
        t0 = time.perf_counter()
        sigma_payload(s, 4096)
        torch.accelerator.synchronize(dev)
        print(f"sigma_payload K={k}: {time.perf_counter() - t0:.3f} s")
    import tempfile

    from hi2.hessian import HessianStats
    from hi2.layerwise import CacheSink

    hess = {}
    for name, k in (("a", 2560), ("b", 2560), ("c", 9216), ("d", 4096)):
        h = HessianStats(k, dev)
        h.update(torch.randn(2048, k, device=dev))
        hess[name] = h
    lin = {n: f"layer.{n}.weight" for n in hess}
    with tempfile.TemporaryDirectory(dir=a.workdir) as td:
        sink = CacheSink(Path(td))
        for rep in range(3):
            t0 = time.perf_counter()
            sink.hessians({n: n for n in hess}, hess, lin)
            t1 = time.perf_counter()
            sink.flush(wait=True)
            print(
                f"sink layer {rep}: call {t1 - t0:.3f} s, with writes {time.perf_counter() - t0:.3f} s"
            )


if __name__ == "__main__":
    main()
