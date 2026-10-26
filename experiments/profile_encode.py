# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Profile the phase-3 encode of real tensors: where the old per-tensor GPTQ encode
spends its time, and old against new (batched, fused kernel) with a bit-exactness check.

Needs a workdir of a finished (e.g. ``--limit-layers 4``) run whose statistics are in
the cache (``cache_ref.json``) or in the workdir itself.

usage: python experiments/profile_encode.py MODEL WORKDIR --device cuda:0 \
           [--match 'layers\\.3\\.'] [--target-bpw 3.0] [--out result.json]
"""

import argparse
import json
import time
from pathlib import Path

import regex as re
import torch
from hi2.allocate import allocate_global
from hi2.cache import StatsView
from hi2.calibrate import (
    Config,
    EncodeRequest,
    _smoothed,
    encode_many,
    encode_with_picks_reference,
)
from hi2.checkpoint import weight_map
from hi2.encode import _safe, fit_ladder, quantize_params
from hi2.executor import Executor
from hi2.format import decode_params, fwht_groups, ladder_dequantize
from hi2.hessian import BLOCK_COLUMNS, inverse_factor, transform_sigma
from hi2.stats import ColumnStats


class Timer:
    def __init__(self, dev):
        self.dev = dev
        self.t: dict[str, float] = {}

    def sync(self):
        if self.dev.type == "cuda":
            torch.accelerator.synchronize(self.dev)

    def add(self, key, t0):
        self.sync()
        self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0
        return time.perf_counter()


def timed_reference(w, table, picks, sigma, cfg, dev, tm: Timer):
    """The reference encode of one tensor on one device, split into its stages."""
    t = time.perf_counter()
    k = w.shape[1]
    had = table.had
    w, _ = _smoothed(w, table.col_scale)
    wt = w.to(dev, torch.float32)
    wt = fwht_groups(wt, had) if had.any() else wt
    t = tm.add("transform_w", t)
    hs = transform_sigma(sigma.to(dev), table.col_scale, had)
    d = hs.diagonal().clone()
    t = tm.add("transform_sigma", t)
    u = inverse_factor(hs, cfg.damp)
    t = tm.add("factorise", t)
    for ci, (nbits, block) in enumerate(table.candidates):
        rows = (picks == ci).nonzero().flatten()
        if rows.numel() == 0:
            continue
        n = 2**nbits
        wc = wt[rows.to(dev)].clone()
        db = d.reshape(1, k // block, block)
        wb = wc.reshape(-1, k // block, block)
        lo, step = fit_ladder(wb, db, n, cfg.effort)
        lo_u8, step_u8, consts = quantize_params(lo, step)
        lo_d, step_d = decode_params(
            lo_u8, step_u8, consts[:, 0:1], consts[:, 1:2], consts[:, 2:3]
        )
        lo_d, step_d = lo_d.float(), step_d.float()
        t = tm.add("fit_ladder", t)
        codes = torch.zeros_like(wc)
        for i1 in range(0, k, BLOCK_COLUMNS):
            i2 = min(i1 + BLOCK_COLUMNS, k)
            w1 = wc[:, i1:i2].clone()
            err1 = torch.zeros_like(w1)
            u1 = u[i1:i2, i1:i2]
            for i in range(i2 - i1):
                j = i1 + i
                b = j // block
                lo_j, st_j = lo_d[:, b], step_d[:, b]
                col = w1[:, i]
                q = torch.round((col - lo_j) / _safe(st_j)).clamp(0, n - 1)
                q = torch.where(st_j == 0, torch.zeros_like(q), q)
                wq = (lo_j + q * st_j).to(torch.float16).float()
                codes[:, j] = q
                e = (col - wq) / u1[i, i]
                w1[:, i:] -= e[:, None] * u1[i, i:][None, :]
                err1[:, i] = e
            t = tm.add("column_loop", t)
            wc[:, i2:] -= err1 @ u[i1:i2, i2:]
            t = tm.add("block_gemm", t)
        rec = ladder_dequantize(
            codes.reshape(-1, k // block, block), lo_d, step_d, torch.float32
        ).float()
        _ = (d * (rec.reshape(-1, k) - wt[rows.to(dev)]) ** 2).sum(-1)
        t = tm.add("finish", t)


def count_launches(fn, dev):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize(dev)
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--match", default=r"layers\.3\.")
    ap.add_argument("--target-bpw", type=float, default=3.0)
    ap.add_argument("--effort", default="normal")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument(
        "--launches", action="store_true", help="count kernel launches (slow)"
    )
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    ex = Executor([a.device])
    cfg = Config(effort=a.effort)
    ref = json.loads((a.workdir / "cache_ref.json").read_text())
    tdir = Path(ref["tables"]) / "tables"
    tables = {
        p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
        for p in tdir.iterdir()
    }
    alloc = allocate_global(list(tables.values()), a.target_bpw, None)
    view = StatsView(a.workdir)
    wmap = weight_map(a.model)
    from safetensors import safe_open

    names = sorted(n for n in tables if re.search(a.match, n))
    result = {"device": a.device, "target_bpw": a.target_bpw, "tensors": {}}
    reqs = []
    for name in names:
        with safe_open(a.model / wmap[name], "pt") as f:
            w = f.get_tensor(name)
        t0 = time.perf_counter()
        sigma = view.sigma(name)
        t_load = time.perf_counter() - t0
        st = view.stats_state(name)
        stats = ColumnStats.from_state(st) if st and st["n"] > 0 else None
        table, picks = tables[name], alloc.picks[name]
        groups = {
            f"{nb}/{b}": int((picks == ci).sum())
            for ci, (nb, b) in enumerate(table.candidates)
        }
        row = {
            "shape": list(table.shape),
            "rows_per_candidate": groups,
            "sigma_load_s": round(t_load, 3),
        }
        reqs.append(EncodeRequest(w, stats, table, picks, sigma, view.sigma_id(name)))
        if not a.no_reference:
            tm = Timer(dev)
            timed_reference(w, table, picks, sigma, cfg, dev, tm)
            row["reference_breakdown_s"] = {k: round(v, 3) for k, v in tm.t.items()}
            row["reference_total_s"] = round(sum(tm.t.values()), 3)
            t0 = time.perf_counter()
            old = encode_with_picks_reference(w, stats, table, picks, cfg, ex, sigma)
            row["reference_untimed_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        new = encode_many([reqs[-1]], cfg, ex)[0]
        row["new_s"] = round(time.perf_counter() - t0, 3)
        if not a.no_reference:
            row["bit_identical"] = all(
                torch.equal(getattr(old, f), getattr(new, f))
                for f in (
                    "codes",
                    "code_off",
                    "blk_params",
                    "param_off",
                    "row_consts",
                    "row_nbits",
                    "row_block",
                    "had",
                )
            )
        print(name, json.dumps(row), flush=True)
        result["tensors"][name] = row
    t0 = time.perf_counter()
    encode_many(reqs, cfg, ex)
    result["new_batched_all_s"] = round(time.perf_counter() - t0, 3)
    if a.launches and names:
        r0 = reqs[0]
        if not a.no_reference:
            result["launches_reference_first"] = count_launches(
                lambda: encode_with_picks_reference(
                    r0.w, r0.stats, r0.table, r0.picks, cfg, ex, r0.sigma
                ),
                dev,
            )
        result["launches_new_first"] = count_launches(
            lambda: encode_many([r0], cfg, ex), dev
        )
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
