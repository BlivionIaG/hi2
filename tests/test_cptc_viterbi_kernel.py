# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Numeric contracts for the fused L16 CPTC Viterbi kernel (HIP/Triton).

The kernel is opt-in on a HIP/CUDA device; the tests that launch it are skipped
when no accelerator is visible. They compare the closed code stream with
:func:`hi2.cptc.viterbi_codes` bit for bit, and re-measure the returned
closed-stream SSE independently from the emitted codes, so a wrong pin order, a
wrong walk or a greedy shortcut fails numerically rather than by inspection.
"""

import pytest
import torch

from hi2 import cptc

triton = pytest.importorskip("triton")
from hi2.cptc_viterbi_kernel import (  # noqa: E402
    viterbi_codes_cuda,
    viterbi_scaled_codes_cuda,
)

_DEPTH = 16
_SHAPES = ((2, 16), (2, 32), (2, 64), (4, 16), (4, 32), (4, 64))
_GPU = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(
    not _GPU, reason="CPTC Viterbi kernel tests need a HIP/CUDA device"
)


def _fp16_values(seed, rows=None, scale=4.0):
    gen = torch.Generator().manual_seed(seed)
    shape = (rows, 1 << _DEPTH) if rows is not None else (1 << _DEPTH,)
    return (torch.randn(shape, generator=gen) * scale).half().float()


def _weights(seed, n, block):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, block, generator=gen) * 3.0


def _column_weights(seed, block):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(block, generator=gen) + 0.1


def _reference(w, d, values, nbits, block):
    return cptc.viterbi_codes(w, d, values, nbits, _DEPTH, block)


def _decoded_closed_sse(w, d, values, codes, nbits, block):
    """Independent Algorithm-2 loss of ``codes`` (the reference's measurement)."""
    q = cptc.window_indices(codes, nbits, _DEPTH, block)
    v = cptc._gather_state(values, q).float().to(torch.float16).float()
    diff = v - w
    return (d * diff * diff).sum(-1)


def _assert_matches_reference(got, ref):
    got_codes, got_sse = got
    ref_codes, ref_sse = ref
    assert got_codes.dtype is torch.uint8
    assert got_codes.shape == ref_codes.shape
    assert got_sse.dtype is torch.float32
    assert got_sse.shape == ref_sse.shape
    assert torch.equal(
        got_codes.cpu(), ref_codes
    ), "closed code stream differs from hi2.cptc.viterbi_codes"
    assert torch.allclose(got_sse.cpu(), ref_sse, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("nbits,block", _SHAPES)
@needs_gpu
def test_shared_values_match_reference(nbits, block):
    seed = 100 * nbits + block
    w = _weights(seed, 3, block)
    d = _column_weights(seed + 1, block)
    values = _fp16_values(seed + 2)
    ref = _reference(w.clone(), d.clone(), values.clone(), nbits, block)

    w0, d0, values0 = w.clone(), d.clone(), values.clone()
    got = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    _assert_matches_reference(got, ref)

    # The returned SSE is the actual closed stream's fp16 decode error, not
    # the DP cost: recompute it from the emitted codes.
    manual = _decoded_closed_sse(w, d, values, got[0].cpu(), nbits, block)
    assert torch.allclose(got[1].cpu(), manual, rtol=1e-5, atol=1e-4)
    # No input may be mutated (w is the strided-capable input).
    assert torch.equal(w, w0)
    assert torch.equal(d, d0)
    assert torch.equal(values, values0)


@pytest.mark.parametrize("nbits,block", ((2, 16), (2, 64), (4, 32)))
@needs_gpu
def test_per_row_values_match_reference(nbits, block):
    seed = 500 + 10 * nbits + block
    n = 4
    base = _fp16_values(seed)
    factors = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    values = (base[None, :] * factors[:, None]).half().float()
    w = _weights(seed + 1, n, block)
    d = _column_weights(seed + 2, block)

    ref = _reference(w.clone(), d.clone(), values.clone(), nbits, block)
    got = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    _assert_matches_reference(got, ref)

    # The shared table must reproduce exactly the factor-1.0 row, and the
    # per-row scaling must actually change the row-0 stream.
    shared_codes, _ = _reference(w, d, base, nbits, block)
    assert torch.equal(got[0][1].cpu(), shared_codes[1])
    row0_scaled, _ = _reference(w[0:1], d, values[0], nbits, block)
    row0_shared, _ = _reference(w[0:1], d, base, nbits, block)
    assert not torch.equal(row0_scaled, row0_shared)
    assert torch.equal(got[0][0:1].cpu(), row0_scaled)


@needs_gpu
@pytest.mark.parametrize("nbits,block", ((2, 32), (4, 64)))
def test_strided_rows_and_no_input_mutation(nbits, block):
    n = 3
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(900 + block)
    big = torch.randn(n, 2 * block, generator=gen)
    off = block // 4
    w_cpu = big[:, off : off + block]
    assert w_cpu.stride(0) == 2 * block and w_cpu.stride(1) == 1

    wide = torch.zeros(n, (1 << _DEPTH) + 32)
    wide[:, : 1 << _DEPTH] = _fp16_values(901 + block, rows=n)
    values_cpu = wide[:, : 1 << _DEPTH]
    assert values_cpu.stride(1) == 1
    assert values_cpu.stride(0) == (1 << _DEPTH) + 32

    d = _column_weights(902 + block, block)
    ref = _reference(w_cpu.contiguous(), d.clone(), values_cpu.clone(), nbits, block)

    w_gpu = big.cuda()[:, off : off + block]
    values_gpu = wide.cuda()[:, : 1 << _DEPTH]
    assert w_gpu.stride(0) == 2 * block and w_gpu.stride(1) == 1
    assert values_gpu.stride(0) == (1 << _DEPTH) + 32
    w0, d0, v0 = w_gpu.clone(), d.clone(), values_gpu.clone()
    got = viterbi_codes_cuda(
        w_gpu, d.cuda(), values_gpu, nbits, _DEPTH, block
    )
    _assert_matches_reference(got, ref)
    assert torch.equal(w_gpu, w0)
    assert torch.equal(d, d0)
    assert torch.equal(values_gpu, v0)


@pytest.mark.parametrize("nbits,block", ((2, 16), (4, 32)))
@needs_gpu
def test_cyclic_seam_planted_periodic_stream(nbits, block):
    """A period-B/2 stream needs the cyclic window at the block seam.

    Both half-block rotations fit ``w`` exactly, so this exercises the
    rotated walk, the closed pins and the earliest-candidate tie rule on a
    landscape with more than one valid optimum.
    """
    gen = torch.Generator().manual_seed(700 + block + nbits)
    half = block // 2
    pattern = torch.randint(
        0, 1 << nbits, (half,), generator=gen, dtype=torch.uint8
    )
    stream = pattern.repeat(2).reshape(1, block)
    values = _fp16_values(701 + block)
    q = cptc.window_indices(stream, nbits, _DEPTH, block)
    w = values[q].reshape(1, block).repeat(2, 1)
    d = _column_weights(702 + block, block)

    ref_codes, ref_sse = _reference(w, d, values, nbits, block)
    got = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    _assert_matches_reference(got, (ref_codes, ref_sse))

    manual = _decoded_closed_sse(w, d, values, got[0].cpu(), nbits, block)
    assert torch.allclose(got[1].cpu(), manual, rtol=1e-5, atol=1e-4)
    # The planted closed stream is exact; the candidate pins must reach it.
    assert torch.equal(ref_sse, torch.zeros_like(ref_sse))
    assert torch.equal(got[1].cpu(), torch.zeros_like(got[1].cpu()))


@pytest.mark.parametrize("nbits,block", ((2, 16), (4, 64)))
@needs_gpu
def test_tie_policy_prefers_lowest_symbol_and_prefix(nbits, block):
    """All streams decode identically, so every DP tie is resolved by policy.

    The reference picks the lowest high symbol, the lowest free end prefix and
    the earliest pin; a ``<=`` implementation emits a different stream.
    """
    n = 2
    values = torch.full((1 << _DEPTH,), 2.0)
    w = torch.full((n, block), 0.5)
    d = _column_weights(1100 + block, block)

    ref = _reference(w, d, values, nbits, block)
    got = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    _assert_matches_reference(got, ref)
    zeros = torch.zeros(n, block, dtype=torch.uint8)
    assert torch.equal(ref[0], zeros)
    assert torch.equal(got[0].cpu(), zeros)

    manual = _decoded_closed_sse(w, d, values, got[0].cpu(), nbits, block)
    assert torch.allclose(got[1].cpu(), manual, rtol=1e-5, atol=1e-4)


@needs_gpu
def test_zero_and_subnormal_state_values():
    block = 32
    w = _weights(1300, 3, block) * 0.125
    d = _column_weights(1301, block)

    # Multiples of the smallest fp16 subnormal are exact half-rounded states.
    tiny = torch.tensor(2.0**-24, dtype=torch.float32)
    subnormal = (torch.arange(1 << _DEPTH) & 7).float() * tiny
    assert torch.equal(subnormal.half().float(), subnormal)
    for values in (subnormal, torch.zeros(1 << _DEPTH)):
        ref = _reference(w.clone(), d.clone(), values.clone(), 2, block)
        got = viterbi_codes_cuda(
            w.cuda(), d.cuda(), values.cuda(), 2, _DEPTH, block
        )
        _assert_matches_reference(got, ref)


@needs_gpu
def test_fp16_values_agree_with_fp32():
    nbits, block = 2, 32
    w = _weights(1400, 3, block)
    d = _column_weights(1401, block)
    values = _fp16_values(1402)
    ref = _reference(w, d, values, nbits, block)
    got32 = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    got16 = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.half().cuda(), nbits, _DEPTH, block
    )
    _assert_matches_reference(got32, ref)
    _assert_matches_reference(got16, ref)
    assert torch.equal(got16[0], got32[0])


@needs_gpu
def test_empty_rows_return_empty_tensors():
    dev = torch.device("cuda")
    block = 16
    w = torch.empty(0, block, device=dev)
    d = torch.ones(block, device=dev)
    for values in (
        torch.empty(1 << _DEPTH, device=dev),
        torch.empty(0, 1 << _DEPTH, device=dev),
    ):
        codes, sse = viterbi_codes_cuda(w, d, values, 2, _DEPTH, block)
        assert codes.shape == (0, block) and codes.dtype is torch.uint8
        assert sse.shape == (0,) and sse.dtype is torch.float32
        assert codes.device == w.device and sse.device == w.device


def test_rejects_unsupported_and_malformed_inputs_before_device():
    w = torch.zeros(1, 16)
    d = torch.ones(16)
    values = torch.zeros(1 << _DEPTH)
    with pytest.raises(NotImplementedError):
        viterbi_codes_cuda(w, d, values, 2, 12, 16)
    with pytest.raises(ValueError):
        viterbi_codes_cuda(w, d, values, 3, _DEPTH, 16)
    with pytest.raises(ValueError):
        viterbi_codes_cuda(w, d, values, 2, _DEPTH, 8)
    with pytest.raises(ValueError, match="stride"):
        viterbi_codes_cuda(torch.zeros(1, 32)[:, ::2], d, values, 2, _DEPTH, 16)
    with pytest.raises(ValueError, match="entries"):
        viterbi_codes_cuda(w, d, values[:-1].clone(), 2, _DEPTH, 16)
    with pytest.raises(TypeError):
        viterbi_codes_cuda(
            torch.zeros(1, 16, dtype=torch.float64), d, values, 2, _DEPTH, 16
        )
    with pytest.raises(ValueError, match="CUDA|HIP"):
        viterbi_codes_cuda(w, d, values, 2, _DEPTH, 16)


_LUT_ENTRIES = 127


def _serialized_lut(codec):
    """The [127] mirror-expanded fp16 table the computed entry consumes."""
    lut = cptc.index_table_value(
        torch.arange(_LUT_ENTRIES), codec.table, float(codec.table_scale)
    )
    assert lut.dtype is torch.float16 and lut.shape == (_LUT_ENTRIES,)
    return lut


def _oracle_values(codec, scale):
    """Expanded fp16 decoder table for one scale, built through the reference.

    This is an independent construction path (host ``to(fp16)``) for the same
    ``fp16(scale * lut[index(q)])`` the kernel reconstructs, so the computed
    entry is checked against the generic entry, not against itself.
    """
    lut = _serialized_lut(codec).float()
    idx = cptc.decode_index(
        torch.arange(1 << _DEPTH, dtype=torch.int64), codec.a0, codec.a1
    )
    return (lut * float(scale)).to(torch.float16).float()[idx]


def _oracle_rows(codec, scales):
    return torch.stack([_oracle_values(codec, s) for s in scales.tolist()])


@pytest.mark.parametrize("nbits,block", _SHAPES)
@needs_gpu
def test_scaled_entry_matches_expanded_oracle(nbits, block):
    codec = cptc.get_codec(_DEPTH, nbits)
    seed = 2100 + 10 * nbits + block
    n = 3
    w = _weights(seed, n, block)
    d = _column_weights(seed + 1, block)
    scales = torch.tensor([0.03, 0.25, 1.5], dtype=torch.float16)
    values = _oracle_rows(codec, scales)
    ref = _reference(w.clone(), d.clone(), values.clone(), nbits, block)

    lut = _serialized_lut(codec)
    w0, d0, lut0, scales0 = w.clone(), d.clone(), lut.clone(), scales.clone()
    got = viterbi_scaled_codes_cuda(
        w.cuda(),
        d.cuda(),
        lut.cuda(),
        scales.cuda(),
        codec.a0,
        codec.a1,
        nbits,
        _DEPTH,
        block,
    )
    _assert_matches_reference(got, ref)

    manual = _decoded_closed_sse(w, d, values, got[0].cpu(), nbits, block)
    assert torch.allclose(got[1].cpu(), manual, rtol=1e-5, atol=1e-4)
    assert torch.equal(w, w0)
    assert torch.equal(d, d0)
    assert torch.equal(lut, lut0)
    assert torch.equal(scales, scales0)


@pytest.mark.parametrize("nbits,block", ((2, 16), (2, 64), (4, 32), (4, 64)))
@needs_gpu
def test_scaled_entry_equals_generic_entry_on_gpu(nbits, block):
    """Both entries must decode identically for exactly equal half-rounded values."""
    codec = cptc.get_codec(_DEPTH, nbits)
    seed = 2300 + 10 * nbits + block
    n = 3
    w = _weights(seed, n, block)
    d = _column_weights(seed + 1, block)
    scales = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float16)
    values = _oracle_rows(codec, scales)

    got_generic = viterbi_codes_cuda(
        w.cuda(), d.cuda(), values.cuda(), nbits, _DEPTH, block
    )
    got_scaled = viterbi_scaled_codes_cuda(
        w.cuda(),
        d.cuda(),
        _serialized_lut(codec).cuda(),
        scales.cuda(),
        codec.a0,
        codec.a1,
        nbits,
        _DEPTH,
        block,
    )
    ref = _reference(w, d, values, nbits, block)
    _assert_matches_reference(got_scaled, ref)
    assert torch.equal(got_scaled[0], got_generic[0])
    assert torch.equal(got_scaled[1], got_generic[1])


@pytest.mark.parametrize("nbits", (2, 4))
@needs_gpu
def test_scaled_entry_zero_and_subnormal_scales(nbits):
    block = 16
    codec = cptc.get_codec(_DEPTH, nbits)
    scales = torch.tensor([0.0, 2.0**-20, 2.0**-24], dtype=torch.float16)
    w = _weights(2600 + nbits, 3, block) * scales.float().unsqueeze(1)
    d = _column_weights(2601 + nbits, block)
    values = _oracle_rows(codec, scales)
    ref = _reference(w, d, values, nbits, block)
    w_gpu, d_gpu = w.cuda(), d.cuda()
    generic = viterbi_codes_cuda(
        w_gpu, d_gpu, values.cuda(), nbits, _DEPTH, block
    )
    scaled = viterbi_scaled_codes_cuda(
        w_gpu,
        d_gpu,
        _serialized_lut(codec).cuda(),
        scales.cuda(),
        codec.a0,
        codec.a1,
        nbits,
        _DEPTH,
        block,
    )
    assert torch.equal(scaled[0], generic[0])
    assert torch.equal(scaled[1], generic[1])
    assert torch.equal(scaled[0].cpu(), ref[0])
    assert torch.allclose(scaled[1].cpu(), ref[1], rtol=2e-5, atol=1e-30)
    assert scaled[1][0].item() == 0.0


@needs_gpu
def test_scaled_entry_strided_rows_and_no_mutation():
    nbits, block = 4, 64
    codec = cptc.get_codec(_DEPTH, nbits)
    n = 3
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(2500)
    big = torch.randn(n, 2 * block, generator=gen)
    off = block // 8
    w_cpu = big[:, off : off + block]
    assert w_cpu.stride(0) == 2 * block and w_cpu.stride(1) == 1
    d = _column_weights(2501, block)
    scales = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float16)
    values = _oracle_rows(codec, scales)
    ref = _reference(w_cpu.contiguous(), d.clone(), values.clone(), nbits, block)

    w_gpu = big.cuda()[:, off : off + block]
    lut = _serialized_lut(codec).cuda()
    scales_gpu = scales.cuda()
    w0, d0, lut0, scales0 = w_gpu.clone(), d.clone(), lut.clone(), scales_gpu.clone()
    got = viterbi_scaled_codes_cuda(
        w_gpu, d.cuda(), lut, scales_gpu, codec.a0, codec.a1, nbits, _DEPTH, block
    )
    _assert_matches_reference(got, ref)
    assert torch.equal(w_gpu, w0)
    assert torch.equal(d, d0)
    assert torch.equal(lut, lut0)
    assert torch.equal(scales_gpu, scales0)


@needs_gpu
def test_scaled_entry_shared_scale_stride_zero():
    """A broadcast fp16 scale (stride 0) must decode like the same shared table."""
    nbits, block = 2, 32
    codec = cptc.get_codec(_DEPTH, nbits)
    n = 3
    w = _weights(2550, n, block)
    d = _column_weights(2551, block)
    scale = torch.tensor(0.5, dtype=torch.float16)
    row = _oracle_values(codec, 0.5)
    values = row.unsqueeze(0).repeat(n, 1)
    ref = _reference(w, d, values, nbits, block)

    dev = torch.device("cuda")
    scales = torch.tensor(0.5, dtype=torch.float16, device=dev).expand(n)
    assert scales.stride(0) == 0
    got = viterbi_scaled_codes_cuda(
        w.cuda(),
        d.cuda(),
        _serialized_lut(codec).cuda(),
        scales,
        codec.a0,
        codec.a1,
        nbits,
        _DEPTH,
        block,
    )
    _assert_matches_reference(got, ref)
    assert torch.equal(scales, scale.to(dev).expand(n))


@needs_gpu
def test_scaled_entry_empty_rows_return_empty_tensors():
    dev = torch.device("cuda")
    block = 16
    codec = cptc.get_codec(_DEPTH, 2)
    w = torch.empty(0, block, device=dev)
    d = torch.ones(block, device=dev)
    lut = _serialized_lut(codec).cuda()
    scales = torch.empty(0, dtype=torch.float16, device=dev)
    codes, sse = viterbi_scaled_codes_cuda(
        w, d, lut, scales, codec.a0, codec.a1, 2, _DEPTH, block
    )
    assert codes.shape == (0, block) and codes.dtype is torch.uint8
    assert sse.shape == (0,) and sse.dtype is torch.float32
    assert codes.device == w.device and sse.device == w.device


def test_scaled_rejects_malformed_inputs_before_device():
    codec = cptc.get_codec(_DEPTH, 2)
    w = torch.zeros(1, 16)
    d = torch.ones(16)
    lut = _serialized_lut(codec)
    scales = torch.ones(1, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        viterbi_scaled_codes_cuda(w, d, lut, scales, codec.a0, codec.a1, 2, 12, 16)
    with pytest.raises(ValueError):
        viterbi_scaled_codes_cuda(w, d, lut, scales, codec.a0, codec.a1, 3, _DEPTH, 16)
    with pytest.raises(ValueError, match="127"):
        viterbi_scaled_codes_cuda(
            w, d, lut[:-1].clone(), scales, codec.a0, codec.a1, 2, _DEPTH, 16
        )
    with pytest.raises(TypeError):
        viterbi_scaled_codes_cuda(
            w, d, lut.float(), scales, codec.a0, codec.a1, 2, _DEPTH, 16
        )
    with pytest.raises(TypeError):
        viterbi_scaled_codes_cuda(
            w, d, lut, scales.float(), codec.a0, codec.a1, 2, _DEPTH, 16
        )
    with pytest.raises(ValueError, match="depth"):
        viterbi_scaled_codes_cuda(w, d, lut, scales, 0, 0, 2, _DEPTH, 16)
    with pytest.raises(ValueError, match="CUDA|HIP"):
        viterbi_scaled_codes_cuda(
            w, d, lut, scales, codec.a0, codec.a1, 2, _DEPTH, 16
        )
