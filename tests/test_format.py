# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""v3.2 format: decode, packing, layout, reference semantics, accounting."""

import pytest
import torch

from hi2.encode import encode_rows, quantize_params
from hi2.format import (
    assemble,
    block_bits_per_weight,
    decode_params,
    dequantize,
    dequantize_stored,
    fwht_groups,
    hadamard16,
    ladder_dequantize,
    metadata_bits,
    pack_codes,
    reference_linear,
    tensor_bpw,
    unpack_codes,
)


def _quantize(w, nbits, block, had):
    """Uniform-config tensor through the real encoder."""
    r, k = w.shape
    wt = fwht_groups(w.float(), had) if had.any() else w.float()
    enc = encode_rows(wt, torch.ones(k), nbits, block, "fast")
    rows = torch.arange(r)
    return assemble(
        [(rows, enc.codes, enc.params, enc.consts)],
        torch.full((r,), nbits),
        torch.full((r,), block),
        had,
        k,
    )


@pytest.mark.parametrize("nbits", [2, 4])
def test_pack_roundtrip_and_lsb_first(nbits):
    codes = torch.randint(0, 2**nbits, (5, 64))
    packed = pack_codes(codes, nbits)
    assert packed.dtype == torch.int32 and packed.shape == (5, 64 * nbits // 32)
    assert torch.equal(unpack_codes(packed, nbits), codes)
    one = torch.zeros(1, 32, dtype=torch.int64)
    one[0, 1] = 3 if nbits == 2 else 9
    assert int(pack_codes(one, nbits)[0, 0]) == (int(one[0, 1]) << nbits)


def test_pack_rejects_code_outside_ladder():
    with pytest.raises(ValueError):
        pack_codes(torch.full((1, 16), 4), 2)


def test_hadamard16_orthonormal_involution_and_flags():
    h = hadamard16()
    assert torch.allclose(h @ h, torch.eye(16, dtype=torch.float64))
    x = torch.randn(3, 64, dtype=torch.float64)
    flags = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
    y = fwht_groups(x, flags)
    assert torch.equal(y[:, 16:32], x[:, 16:32])
    assert torch.allclose(fwht_groups(y, flags), x, atol=1e-12)
    with pytest.raises(ValueError):
        fwht_groups(x, flags[:3])


def test_decode_params_is_fp32_fma_then_fp16_and_fp32_mul():
    lo_u8 = torch.arange(256)
    lo_min, lo_scale = torch.tensor(-0.37).half(), torch.tensor(0.00391).half()
    step_scale = torch.tensor(0.0123).half()
    lo, step = decode_params(lo_u8, lo_u8, lo_min, lo_scale, step_scale)
    ref_lo = (
        (lo_u8.double() * lo_scale.double() + lo_min.double()).float().to(torch.float16)
    )
    assert torch.equal(lo, ref_lo)
    assert torch.equal(step, (lo_u8.float() * step_scale.float()).to(torch.float16))


def test_reconstruction_rounds_once_in_fp16():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (4096,))
    lo = (torch.randn(4096) * 50).half()
    step = (torch.rand(4096) * 16).half()
    got = ladder_dequantize(codes.unsqueeze(-1), lo, step).squeeze(-1)
    assert torch.equal(got, (lo.double() + codes.double() * step.double()).half())


def test_param_quantisation_keeps_constant_and_zero_blocks_exact():
    lo = torch.tensor([[0.5, -0.2, 0.1]])
    step = torch.tensor([[0.0, 0.03, 0.0]])  # blocks 0 and 2 are constant
    lo_u8, step_u8, consts = quantize_params(lo, step)
    dlo, dstep = decode_params(
        lo_u8, step_u8, consts[:, 0:1], consts[:, 1:2], consts[:, 2:3]
    )
    assert dstep[0, 0] == 0 and dstep[0, 2] == 0
    assert torch.allclose(dlo.float(), lo, atol=2e-3)


@pytest.mark.parametrize("had_val", [0, 1])
@pytest.mark.parametrize("nbits,block", [(2, 16), (2, 64), (4, 32)])
def test_reference_linear_matches_original_basis_matmul(nbits, block, had_val):
    torch.manual_seed(1)
    r, k = 8, 128
    t = _quantize(torch.randn(r, k), nbits, block, torch.full((k // 16,), had_val))
    x = torch.randn(5, k, dtype=torch.float64)
    assert torch.allclose(
        reference_linear(x, t), x @ dequantize(t).T, atol=1e-8, rtol=1e-8
    )


def test_mixed_rows_layout_offsets_and_roundtrip():
    torch.manual_seed(2)
    r, k = 4, 128
    w = torch.randn(r, k)
    nbits = torch.tensor([2, 4, 2, 4])
    block = torch.tensor([32, 64, 64, 32])
    pieces = []
    for nb in (2, 4):
        for b in (32, 64):
            rows = ((nbits == nb) & (block == b)).nonzero().flatten()
            if rows.numel():
                enc = encode_rows(w[rows], torch.ones(k), nb, b, "fast")
                pieces.append((rows, enc.codes, enc.params, enc.consts))
    t = assemble(pieces, nbits, block, torch.zeros(k // 16), k)
    assert t.code_off.tolist() == [0, 8, 24, 32, 48]
    assert t.param_off.tolist() == [0, 4, 6, 8, 12]
    assert t.codes.numel() == 48 and t.blk_params.numel() == 24
    # every row reconstructs close to the source at its own precision
    rec = dequantize_stored(t)
    err = ((rec - w.double()) ** 2).sum(-1) / (w.double() ** 2).sum(-1)
    assert (err[nbits == 4] < err[nbits == 2].max()).all() or err.max() < 0.2


def test_int4_and_small_blocks_are_more_accurate():
    torch.manual_seed(3)
    w = torch.randn(32, 256)
    z = torch.zeros(16)

    def err(nbits, block):
        t = _quantize(w, nbits, block, z)
        return ((dequantize(t) - w.double()).norm() / w.double().norm()).item()

    assert err(4, 64) < err(2, 64)
    assert err(2, 16) < err(2, 64)


def test_bit_accounting():
    assert block_bits_per_weight(2, 16) == 3.0
    assert block_bits_per_weight(4, 64) == 4.25
    r, k = 8, 128
    t = _quantize(torch.randn(r, k), 2, 32, torch.zeros(k // 16))
    bits = (2 * k + 16 * k / 32) * r + 48 * r + 1
    assert tensor_bpw(t) == pytest.approx(bits / (r * k))
    mixed = torch.tensor([2, 4, 2, 2])
    assert (
        metadata_bits(mixed, torch.tensor([32, 64, 32, 32]), torch.ones(8))
        == 1 + 4 + 8  # 1 hadamard, 1 width bit x 4 rows, 2 block bits x 4 rows
    )


def test_col_scale_reference_linear_and_bpw():
    torch.manual_seed(5)
    r, k = 8, 128
    s = (torch.rand(k) * 3 + 0.2).to(torch.float16)
    w = torch.randn(r, k)
    ws = w.float() * s.float()
    t = _quantize(ws, 2, 32, torch.ones(k // 16, dtype=torch.uint8))
    t.col_scale = s
    x = torch.randn(5, k, dtype=torch.float64)
    # kernel-style path (x / s, rotate, contract) equals x @ dequantize(t).T
    assert torch.allclose(
        reference_linear(x, t), x @ dequantize(t).T, atol=1e-8, rtol=1e-8
    )
    # the product is close to the unquantised one, and s costs 16 bits per column
    ref = x @ w.double().T
    assert ((reference_linear(x, t) - ref).norm() / ref.norm()) < 0.9
    t2 = _quantize(ws, 2, 32, torch.ones(k // 16, dtype=torch.uint8))
    assert tensor_bpw(t) - tensor_bpw(t2) == pytest.approx(16 / r)
