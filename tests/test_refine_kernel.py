# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused refinement sweep (gfx10/gfx11 GPU): codes, and so decoded values, are bit-identical to
the torch loop for single candidates, mixed candidates in one sweep, row counts that are not a
multiple of 64 and widths that are not a multiple of 128 columns."""

import pytest
import torch

pytest.importorskip("safetensors")

from hi2.hessian import encode_rows_gptq_reference, inverse_factor  # noqa: E402
from hi2.refine import refine_groups  # noqa: E402
from hi2.refine_kernel import available, supported  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not supported(torch.device("cuda", 0)),
    reason="needs a gfx10/gfx11 HIP device",
)


def _rows(nbits, block, n, k, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(500, k, generator=g) * torch.exp(0.5 * torch.randn(k, generator=g))
    hs = ((x.T @ x) / 500).to(dev)
    w = (torch.randn(n, k, generator=g) * 0.3).to(dev)
    enc = encode_rows_gptq_reference(
        w, hs.diagonal().clone(), inverse_factor(hs), nbits, block
    )
    return w, enc, hs


def _both(groups, hs, passes, monkeypatch):
    out = {}
    for backend in ("torch", "kernel"):
        monkeypatch.setenv("HI2_REFINE_BACKEND", backend)
        out[backend] = refine_groups(groups, hs, passes)
    return out["torch"], out["kernel"]


@pytest.mark.parametrize(
    "nbits,block,n,k",
    [(2, 32, 64, 256), (4, 64, 100, 192), (2, 16, 37, 160), (4, 16, 1, 128)],
)
def test_kernel_matches_the_torch_loop_bit_for_bit(nbits, block, n, k, monkeypatch):
    dev = torch.device("cuda", 0)
    assert available(dev)
    w, enc, hs = _rows(nbits, block, n, k, seed=n, dev=dev)
    ref, got = _both([(w, enc, nbits, block)], hs, 3, monkeypatch)
    assert torch.equal(ref[0], got[0])
    assert not torch.equal(got[0], enc.codes)  # the sweep did move codes


def test_mixed_candidates_share_one_sweep(monkeypatch):
    dev = torch.device("cuda", 0)
    k = 192
    w2, e2, hs = _rows(2, 32, 70, k, seed=1, dev=dev)
    w4, e4, _ = _rows(4, 64, 45, k, seed=1, dev=dev)
    groups = [(w2, e2, 2, 32), (w4, e4, 4, 64)]
    ref, got = _both(groups, hs, 2, monkeypatch)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert int(got[0].max()) <= 3 and int(got[1].max()) <= 15


def test_zero_passes_and_converged_sweeps_return_the_input(monkeypatch):
    dev = torch.device("cuda", 0)
    w, enc, hs = _rows(2, 32, 33, 128, seed=2, dev=dev)
    monkeypatch.setenv("HI2_REFINE_BACKEND", "kernel")
    assert torch.equal(refine_groups([(w, enc, 2, 32)], hs, 0)[0], enc.codes)
    once = refine_groups([(w, enc, 2, 32)], hs, 40)[0]
    again = refine_groups(
        [(w, type(enc)(once, enc.params, enc.consts, enc.sse, enc.comp), 2, 32)], hs, 1
    )[0]
    assert torch.equal(again, once)
