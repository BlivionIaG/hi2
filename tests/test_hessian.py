# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Hessian-aware rounding (spec 8.3): transform, factor, compensated encode."""

import pytest
import torch
from hi2.encode import encode_rows
from hi2.format import decode_params, fwht_groups, ladder_dequantize
from hi2.hessian import (
    HessianStats,
    encode_rows_gptq,
    inverse_factor,
    transform_sigma,
)


def _correlated(t=4000, k=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(k, k, generator=g) * 0.15 + torch.eye(k)
    x = torch.randn(t, k, generator=g) @ mix
    x *= torch.exp(0.5 * torch.randn(1, k, generator=g))
    return x


def _recon(enc, nbits_block):
    nb, b = nbits_block
    n, k = enc.codes.shape
    lo, step = decode_params(
        enc.params[..., 0],
        enc.params[..., 1],
        enc.consts[:, 0:1],
        enc.consts[:, 1:2],
        enc.consts[:, 2:3],
    )
    rec = ladder_dequantize(
        enc.codes.reshape(n, k // b, b).float(), lo.float(), step.float(), torch.float32
    )
    return rec.float().reshape(n, k)


def test_hessian_stats_match_direct_second_moment():
    x = _correlated(1000, 64)
    st = HessianStats(64)
    st.update(x[:300])
    st.update(x[300:])
    assert st.n == 1000
    assert torch.allclose(st.mean(), x.T @ x / 1000, rtol=1e-4, atol=1e-4)
    other = HessianStats(64)
    other.update(x[:500])
    rest = HessianStats(64)
    rest.update(x[500:])
    rest.merge(other)
    assert torch.allclose(rest.mean(), x.T @ x / 1000, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("flags", [[0, 0, 0, 0], [1, 0, 1, 1]])
def test_transform_sigma_equals_covariance_of_transformed_activation(smooth, flags):
    x = _correlated(2000, 64)
    fl = torch.tensor(flags, dtype=torch.uint8).repeat_interleave(1)
    s = (torch.rand(64) * 2 + 0.5).to(torch.float16) if smooth else None
    xt = x / s.float() if smooth else x
    xt = fwht_groups(xt, fl)
    want = xt.T @ xt / x.shape[0]
    got = transform_sigma(x.T @ x / x.shape[0], s, fl)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3 * want.abs().max())


def test_inverse_factor_inverts_and_survives_a_singular_matrix():
    x = _correlated(500, 32)
    hs = x.T @ x / 500
    u = inverse_factor(hs, 0.01)
    hinv = u.T @ u
    damped = hs + 0.01 * hs.diagonal().mean() * torch.eye(32)
    assert torch.allclose(hinv @ damped, torch.eye(32), atol=1e-3)
    singular = torch.zeros(16, 16)
    singular[0, 0] = 1.0
    u2 = inverse_factor(singular, 0.01)  # heavily damped, must not raise
    assert torch.isfinite(u2).all()


@pytest.mark.parametrize("nbits,block", [(2, 32), (2, 16), (4, 64)])
def test_compensated_rounding_lowers_the_hessian_error(nbits, block):
    torch.manual_seed(1)
    k, rows = 128, 48
    x = _correlated(6000, k)
    hs = x.T @ x / x.shape[0]
    w = torch.randn(rows, k) * torch.exp(0.3 * torch.randn(1, k))
    d = hs.diagonal().clone()
    u = inverse_factor(hs, 0.01)
    near = encode_rows(w, d, nbits, block, "fast")
    comp = encode_rows_gptq(w, d, u, nbits, block, "fast")

    def herr(enc):
        e = _recon(enc, (nbits, block)) - w
        return ((e @ hs) * e).sum().item()

    # same ladders (fitted before rounding), different codes
    assert torch.equal(near.params, comp.params) and torch.equal(
        near.consts, comp.consts
    )
    assert herr(comp) < herr(near) * 0.97
    assert comp.codes.dtype == torch.uint8 and int(comp.codes.max()) < 2**nbits


def test_diagonal_hessian_gives_nearest_rounding():
    torch.manual_seed(2)
    k, rows = 64, 24
    w = torch.randn(rows, k)
    d = torch.rand(k) + 0.5
    u = inverse_factor(torch.diag(d), 0.0)
    near = encode_rows(w, d, 2, 32, "fast")
    comp = encode_rows_gptq(w, d, u, 2, 32, "fast")
    assert torch.equal(near.codes, comp.codes)


def test_row_chunking_is_consistent(monkeypatch):
    import hi2.hessian as hm

    torch.manual_seed(3)
    k = 64
    x = _correlated(3000, k)
    hs = x.T @ x / 3000
    w = torch.randn(40, k)
    d = hs.diagonal().clone()
    u = inverse_factor(hs, 0.01)
    whole = encode_rows_gptq(w, d, u, 2, 32, "fast")
    monkeypatch.setattr(hm, "ROW_ELEMS", 64 * 7)  # 7 rows per chunk
    chunked = encode_rows_gptq(w, d, u, 2, 32, "fast")
    assert torch.equal(whole.codes, chunked.codes)
