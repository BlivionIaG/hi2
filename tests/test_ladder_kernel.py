# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The fused ladder search of :mod:`hi2.ladder_kernel` against the torch reference.

The kernel evaluates the same 6x6 grid with the same formulas in one pass over the
block; only the order the weighted error is summed in differs, so the ladder it picks
matches the reference up to float32 rounding."""

import os

import pytest
import torch

from hi2 import ladder_kernel
from hi2.encode import EFFORT, _recon, fit_ladder


def _device():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("HIP device required")
    dev = torch.device("cuda:0")
    if not ladder_kernel.supported(dev):
        pytest.skip("gfx10/gfx11 required")
    return dev


def _reference_grid(wb, db, n, grid):
    mn, mx = wb.amin(-1), wb.amax(-1)
    span = mx - mn
    best_lo = best_step = best_err = None
    for a in grid:
        for b in grid:
            lo = mn + a * span
            step = (mx - b * span - lo) / (n - 1)
            rec, _ = _recon(wb, lo, step, n)
            err = (db * (rec - wb) ** 2).sum(-1)
            if best_err is None:
                best_lo, best_step, best_err = lo, step, err
            else:
                m = err < best_err
                best_lo = torch.where(m, lo, best_lo)
                best_step = torch.where(m, step, best_step)
                best_err = torch.where(m, err, best_err)
    return best_lo, best_step


def test_grid_matches_the_reference():
    dev = _device()
    torch.manual_seed(0)
    # 101 rows x 640 blocks is not a multiple of the 256-thread block, so the tail
    # threads must return instead of reading past the tile.
    wb = torch.randn(101, 640, 16, dtype=torch.float32, device=dev)
    db = torch.rand(640, 16, dtype=torch.float32, device=dev) + 0.5
    grid = EFFORT["normal"]["grid"]

    lo_k, st_k, _ = ladder_kernel.ladder_grid(wb, db, grid, 4)
    lo_t, st_t = _reference_grid(wb, db, 4, grid)

    assert torch.equal(lo_k, lo_t)  # lo depends on `a` only, and every block agreed
    assert (st_k - st_t).abs().max() < 1e-6


def test_fit_ladder_agrees_with_the_torch_path():
    dev = _device()
    torch.manual_seed(1)
    wb = torch.randn(256, 640, 16, dtype=torch.float32, device=dev)
    db = torch.rand(1, 640, 16, dtype=torch.float32, device=dev) + 0.5

    lo_k, st_k = fit_ladder(wb, db, 4, "normal")
    os.environ["HI2_LADDER_BACKEND"] = "torch"
    try:
        ladder_kernel._mod = None
        ladder_kernel._failed = None
        lo_t, st_t = fit_ladder(wb, db, 4, "normal")
    finally:
        os.environ.pop("HI2_LADDER_BACKEND", None)
        ladder_kernel._mod = None
        ladder_kernel._failed = None

    assert (lo_k - lo_t).abs().max() < 1e-6
    assert (st_k - st_t).abs().max() < 1e-6
