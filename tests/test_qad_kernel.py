# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Fused QAD decode (gfx10/gfx11 GPU): the stored-basis values equal the torch path and the
exported tensor bit for bit, and the gradients of the latent weights and of both ladder bytes
agree with autograd of the torch graph (up to float32 summation order), including blocks
outside the clamp range, zero-step blocks, and both latent units."""

import pytest
import torch

pytest.importorskip("safetensors")

from test_qad import _perturb, _ptq, _w, _x  # noqa: E402

from hi2.format import dequantize_stored  # noqa: E402
from hi2.qad import FakeQuantLinear  # noqa: E402
from hi2.qad_kernel import available, supported  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not supported(torch.device("cuda", 0)),
    reason="needs a gfx10/gfx11 HIP device",
)


def _fq(units, seed=0, rows=40, k=128, zero_blocks=False):
    w, x = _w(rows, k, seed + 1), _x(k=k, seed=seed)
    t0, comp = _ptq(w, x)
    fq = FakeQuantLinear(w, t0, comp, "weights", units)
    _perturb(fq, seed=seed + 3)
    if zero_blocks:
        with torch.no_grad():
            for g in fq.groups:
                g.step_u[::3, ::2] = 0.0
                g.lo_u[1::4] += 40.0
    return fq.to("cuda")


def _both(fq, monkeypatch):
    out = {}
    g = torch.Generator().manual_seed(7)
    grad = torch.randn(fq.shape, generator=g).cuda()
    for backend in ("torch", "kernel"):
        monkeypatch.setenv("HI2_QAD_BACKEND", backend)
        s = fq.stored(True)
        params = [fq.latent] + [p for gr in fq.groups for p in (gr.lo_u, gr.step_u)]
        grads = torch.autograd.grad(s, params, grad)
        out[backend] = (fq.stored(False).detach(), s.detach(), grads)
    return out


@pytest.mark.parametrize("units", ["abs", "step"])
@pytest.mark.parametrize("zero_blocks", [False, True])
def test_forward_is_bit_exact_and_gradients_match_autograd(
    units, zero_blocks, monkeypatch
):
    fq = _fq(units, zero_blocks=zero_blocks)
    assert available(fq.latent.device)
    out = _both(fq, monkeypatch)
    ref, got = out["torch"], out["kernel"]
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert torch.equal(got[0].double().cpu(), dequantize_stored(fq.cpu().export()))
    for a, b in zip(ref[2], got[2]):
        scale = float(a.abs().max()) + 1e-12
        assert float((a - b).abs().max()) <= 2e-4 * scale, (
            float((a - b).abs().max()),
            scale,
        )
    assert sum(float(b.abs().sum()) for b in got[2]) > 0


def test_clamped_values_get_no_latent_gradient_and_the_bytes_do(monkeypatch):
    """A latent far outside the ladder is clamped: no gradient to it, a gradient to lo/step."""
    fq = _fq("abs", seed=2)
    with torch.no_grad():
        fq.latent[:, :16] += 50.0 * fq.latent.abs().max()
    out = _both(fq, monkeypatch)
    g_ref, g_got = out["torch"][2], out["kernel"][2]
    assert torch.equal(g_got[0][:, :16], torch.zeros_like(g_got[0][:, :16]))
    assert torch.equal(g_ref[0][:, :16], torch.zeros_like(g_ref[0][:, :16]))
    assert any(float(x.abs().sum()) > 0 for x in g_got[1:])


@pytest.mark.parametrize("units", ["abs", "step"])
@pytest.mark.parametrize(
    "smooth,rotation", [(0.5, "on"), (0.0, "off"), (0.5, "auto"), (0.0, "on")]
)
def test_original_basis_forward_and_gradients_match_the_torch_path(
    units, smooth, rotation, monkeypatch
):
    w, x = _w(40, 128, 5), _x(k=128, seed=4)
    t0, comp = _ptq(w, x, smooth=smooth, rotation=rotation)
    fq = FakeQuantLinear(w, t0, comp, "weights", units)
    _perturb(fq, seed=8)
    fq = fq.to("cuda")
    grad = torch.randn(fq.shape, generator=torch.Generator().manual_seed(9)).cuda()
    out = {}
    for backend in ("torch", "kernel"):
        monkeypatch.setenv("HI2_QAD_BACKEND", backend)
        wt = fq()
        params = [fq.latent] + [p for g in fq.groups for p in (g.lo_u, g.step_u)]
        out[backend] = (wt.detach(), torch.autograd.grad(wt, params, grad))
    ref, got = out["torch"], out["kernel"]
    scale = float(ref[0].abs().max())
    assert (
        float((ref[0] - got[0]).abs().max()) <= 1e-6 * scale
    )  # same values, summation order
    for a, b in zip(ref[1], got[1]):
        s = float(a.abs().max()) + 1e-12
        assert float((a - b).abs().max()) <= 2e-4 * s


def test_has_had_matches_the_flags():
    w, x = _w(32, 128, 6), _x(k=128, seed=5)
    on = FakeQuantLinear(w, _ptq(w, x, rotation="on")[0], None, "weights")
    off = FakeQuantLinear(w, _ptq(w, x, rotation="off")[0], None, "weights")
    assert on.has_had and not off.has_had
