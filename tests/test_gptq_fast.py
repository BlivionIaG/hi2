# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The batched / fused GPTQ encoder is bit-identical to the reference column loop."""

import pytest
import torch

from hi2.hessian import (
    GptqJob,
    encode_jobs_gptq,
    encode_rows_gptq,
    encode_rows_gptq_reference,
    inverse_factor,
)

CASES = [(2, 16), (2, 32), (2, 64), (4, 16), (4, 32), (4, 64), (8, 32), (8, 64)]


def _layer(rows, k, seed, act_order=False, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(k, k, generator=g) * 0.1 + torch.eye(k)
    x = torch.randn(4 * k, k, generator=g) @ mix
    x *= torch.exp(0.5 * torch.randn(1, k, generator=g))
    hs = x.T @ x / x.shape[0]
    w = torch.randn(rows, k, generator=g) * torch.exp(
        0.4 * torch.randn(rows, 1, generator=g)
    )
    w[:, :: max(1, k // 5)] *= 8  # a few outlier columns
    d = hs.diagonal().clone()
    perm = None
    if act_order:
        perm = torch.argsort(d, descending=True)
        hs = hs[perm][:, perm]
    u = inverse_factor(hs, 0.01)
    dev = torch.device(device)
    return w.to(dev), d.to(dev), u.to(dev), None if perm is None else perm.to(dev)


def _same(a, b):
    """Every field, including ``comp`` (the QAD initialisation) when both carry it."""
    fields = ["codes", "params", "consts", "sse"]
    if a.comp is not None or b.comp is not None:
        fields.append("comp")
    for f in fields:
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and x.shape == y.shape, f
        assert torch.equal(x.cpu(), y.cpu()), f


@pytest.mark.parametrize("nbits,block", CASES)
@pytest.mark.parametrize("act_order", [False, True])
def test_torch_engine_matches_reference(nbits, block, act_order):
    w, d, u, perm = _layer(24, 192, nbits * 100 + block, act_order)
    ref = encode_rows_gptq_reference(w, d, u, nbits, block, "fast", perm)
    _same(encode_rows_gptq(w, d, u, nbits, block, "fast", perm, backend="torch"), ref)


def test_batched_jobs_match_one_call_each(monkeypatch):
    import hi2.hessian as hm

    monkeypatch.setattr(hm, "ROW_ELEMS", 256 * 5)  # several row chunks per job
    monkeypatch.setattr(hm, "BATCH_ELEMS", 256 * 23)  # several batches
    jobs, refs = [], []
    for i, (nb, b) in enumerate(CASES):
        k = 256 if i % 3 else 128  # two widths in one call
        w, d, u, perm = _layer(13 + i, k, i, act_order=i % 2 == 1)
        jobs.append(GptqJob(w, d, u, nb, b, perm))
        refs.append(encode_rows_gptq_reference(w, d, u, nb, b, "fast", perm))
    for got, ref in zip(
        encode_jobs_gptq(jobs, "fast", backend="torch", comp=True), refs
    ):
        _same(got, ref)


def _hip():
    if not torch.accelerator.is_available() or not torch.version.hip:
        return None
    from hi2 import gptq_kernel

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    if not gptq_kernel.supported(dev) or gptq_kernel.load() is None:
        return None
    return dev


@pytest.mark.parametrize("nbits,block", CASES)
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("k", [192, 448])  # last block shorter than 128
def test_kernel_matches_reference(nbits, block, act_order, k):
    dev = _hip()
    if dev is None:
        pytest.skip("needs a gfx10/gfx11 HIP device and the compiled kernel")
    w, d, u, perm = _layer(200, k, nbits * 7 + block + k, act_order, dev)
    ref = encode_rows_gptq_reference(w, d, u, nbits, block, "fast", perm)
    _same(encode_rows_gptq(w, d, u, nbits, block, "fast", perm, backend="kernel"), ref)


def test_kernel_batches_jobs_bit_identically(monkeypatch):
    dev = _hip()
    if dev is None:
        pytest.skip("needs a gfx10/gfx11 HIP device and the compiled kernel")
    import hi2.hessian as hm

    monkeypatch.setattr(hm, "ROW_ELEMS", 512 * 70)
    jobs, refs = [], []
    for i in range(12):
        nb, b = CASES[i % len(CASES)]
        w, d, u, perm = _layer(50 + 31 * i, 512, 1000 + i, i % 3 == 0, dev)
        jobs.append(GptqJob(w, d, u, nb, b, perm))
        refs.append(encode_rows_gptq_reference(w, d, u, nb, b, "normal", perm))
    for got, ref in zip(
        encode_jobs_gptq(jobs, "normal", backend="kernel", comp=True), refs
    ):
        _same(got, ref)


def test_dequantize_on_the_gpu_is_identical():
    dev = _hip()
    if dev is None:
        pytest.skip("needs a gfx10/gfx11 HIP device")
    from hi2.calibrate import Config, build_table, encode_with_picks
    from hi2.executor import Executor
    from hi2.format import dequantize

    g = torch.Generator().manual_seed(9)
    w = torch.randn(96, 256, generator=g) * torch.exp(torch.randn(1, 256, generator=g))
    cfg, ex = Config(effort="fast"), Executor("cpu")
    table = build_table("t", w, None, cfg, ex)
    picks = torch.randint(0, len(table.candidates), (96,), generator=g)
    table.col_scale = (torch.rand(256, generator=g) + 0.5).half()
    t = encode_with_picks(w, None, table, picks, cfg, ex)
    assert torch.equal(dequantize(t), dequantize(t, dev).cpu())
