# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Contracts for the fused CPTC projection kernel (:mod:`hi2.cptc_projection_kernel`).

The GPU kernel cannot run in this repository's CPU gate, so the numeric CPU tests
exercise ``_fused_projection``: a torch transcription of the kernel's fixed
phase/sweep/candidate schedule (cyclic window rebuild from the current code stream,
per-candidate left-to-right affected-window accumulation matching the reference's
fp32 reduction order, fp16 decoded value, strict lowest-symbol ties). Every case is
checked against :func:`hi2.cptc.project_codes`, the integration reference, so a
wrong phase order, bit field, window wrap, scale rounding, reduction order, or tie
rule fails here. The authored GPU tests compare the real Triton kernel with the same
reference; they include the exact row 849 HIP0 counterexample and skip unless a HIP
GPU and Triton are present; they are not part of this CPU proof.
"""

import pytest
import torch

from hi2.cptc import (
    get_codec,
    index_table_value,
    project_codes,
    window_indices,
)
from hi2.cptc_projection_kernel import project_codes_cuda

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _gpu_available() -> bool:
    return _HAS_TRITON and torch.version.hip is not None and torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _gpu_available(), reason="needs a HIP GPU with Triton"
)
_GPU_CASES = [(2, 16), (2, 32), (2, 64), (4, 16), (4, 32), (4, 64)]

# Actual HIP0 counterexample from the 2048-row K2/B64 microbench (row 849):
# the old kernel's tree reduction rounded sym1 and sym2 to one error and kept the
# lower symbol, while the reference's offset-ordered sum kept sym2. This row is
# embedded so the GPU gate fails on that old behavior without any remote artifact.
_COUNTEREXAMPLE_W = (
    0.02602997049689293, 2.024040699005127, -0.2656393349170685, 0.2236558049917221,
    -0.49779054522514343, 0.05886099860072136, -1.3394581079483032, 0.09729765355587006,
    -0.5106471180915833, 0.31738850474357605, 0.22765140235424042, -0.3033588230609894,
    -0.780559241771698, -0.08247673511505127, 1.0363590717315674, -1.2639334201812744,
    -0.9838951230049133, 0.5208390355110168, 0.2606208920478821, 1.73879075050354,
    -0.41038835048675537, -0.7356225848197937, -0.4422789216041565, 0.6822887659072876,
    1.5045868158340454, 0.7538005113601685, 0.4418857991695404, 0.38613229990005493,
    -0.22763590514659882, -1.7321158647537231, -0.22281047701835632, 0.02466457709670067,
    2.735213279724121, 1.116473913192749, 0.3715311884880066, 0.59215247631073,
    -0.2861318290233612, 0.6921614408493042, -0.9327385425567627, -0.21350078284740448,
    0.2025289237499237, -0.9904570579528809, -0.17438098788261414, -1.4591225385665894,
    -0.6583974957466125, -0.6224234104156494, -2.0109589099884033, 0.44984060525894165,
    0.050116900354623795, -1.9391974210739136, 0.6071464419364929, 0.16357068717479706,
    -0.6666773557662964, 0.9351477026939392, 0.26669853925704956, -0.5346954464912415,
    -0.927227258682251, -0.8574003577232361, 1.2464489936828613, -1.7570081949234009,
    1.454068660736084, -1.0208791494369507, 0.7220560312271118, 0.6737179160118103,
)
_COUNTEREXAMPLE_INIT = (
    2, 0, 2, 1, 3, 3, 3, 3, 0, 2, 2, 3, 2, 2, 2, 1,
    1, 1, 0, 1, 0, 3, 3, 0, 3, 0, 0, 0, 1, 0, 2, 0,
    1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2,
    3, 0, 3, 0, 3, 1, 3, 0, 1, 3, 1, 3, 0, 1, 3, 2,
)
_COUNTEREXAMPLE_REF = (
    1, 0, 2, 1, 3, 2, 3, 3, 3, 2, 2, 3, 0, 0, 2, 1,
    1, 3, 0, 1, 0, 3, 3, 0, 2, 2, 0, 3, 0, 0, 2, 0,
    1, 2, 2, 2, 2, 1, 1, 0, 0, 2, 2, 2, 0, 3, 1, 2,
    0, 0, 3, 1, 0, 1, 3, 0, 0, 3, 1, 3, 0, 3, 2, 2,
)
_COUNTEREXAMPLE_COL = 33


# --------------------------------------------------------------------- reference


def _tap_width(depth: int, nbits: int) -> int:
    return (depth + nbits - 1) // nbits


def _phase_period(depth: int, nbits: int) -> int:
    span = _tap_width(depth, nbits)
    return 1 << (span - 1).bit_length()


def _decode_index(q: torch.Tensor, a0: int, a1: int) -> torch.Tensor:
    """The Q6 mirror index, written independently from the reference module."""
    return (((a0 * q) & 0xFFFF) >> 10) + (((a1 * q) & 0xFFFF) >> 10)


def _windows(codes: torch.Tensor, nbits: int, depth: int) -> torch.Tensor:
    """Little-endian tail-biting windows, computed without hi2's helpers."""
    span = _tap_width(depth, nbits)
    c = codes.to(torch.int64)
    q = torch.zeros_like(c)
    for u in range(span):
        q |= torch.roll(c, -u, dims=1) << (u * nbits)
    return q & ((1 << depth) - 1)


def _decoded_values(
    codes: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor,
    a0: int,
    a1: int,
    nbits: int,
    depth: int,
) -> torch.Tensor:
    """Exact stored-basis values: fp16(scale_fp32 * lut_fp32), one rounding."""
    q = _windows(codes, nbits, depth)
    idx = _decode_index(q, a0, a1)
    table = lut.to(torch.float16).to(torch.float32)
    s = scales.to(torch.float16).to(torch.float32).reshape(-1, 1)
    return (s * table[idx]).to(torch.float16).to(torch.float32)


def _weighted_sse(
    w: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor,
    a0: int,
    a1: int,
    nbits: int,
    depth: int,
    d: torch.Tensor | None,
) -> torch.Tensor:
    diff = _decoded_values(codes, scales, lut, a0, a1, nbits, depth) - w.to(
        torch.float32
    )
    weight = (
        torch.ones(w.shape[1], dtype=torch.float32, device=w.device)
        if d is None
        else d.to(torch.float32)
    )
    return (weight * (diff * diff)).sum(-1)


def _fused_projection(
    w: torch.Tensor,
    init_codes: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor,
    a0: int,
    a1: int,
    nbits: int,
    depth: int,
    block: int,
    sweeps: int = 1,
    d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch transcription of the fused kernel's arithmetic and fixed schedule.

    Mirrors :func:`_project_kernel`: positions are updated in ``t mod period``
    phases, the ``ceil(L/K)`` affected windows are rebuilt from the current code
    stream (cyclic), the decoded value is rounded to fp16, and each candidate's
    loss is accumulated in the reference's affected-window order (offset
    ``0 .. span-1``) rather than reduced as one tree. Ties keep the lowest symbol.
    """
    n = w.shape[0]
    span = _tap_width(depth, nbits)
    period = _phase_period(depth, nbits)
    if block % period:
        raise ValueError("test case period does not divide block")
    npos = block // period
    dev = w.device
    codes = init_codes.to(dev, torch.int64).clone()
    wf = w.to(dev, torch.float32)
    s = scales.to(dev, torch.float16).to(torch.float32)
    table = lut.to(dev, torch.float16).to(torch.float32)
    weight = (
        torch.ones(block, dtype=torch.float32, device=dev)
        if d is None
        else d.to(dev, torch.float32)
    )
    depth_mask = (1 << depth) - 1
    syms = torch.arange(1 << nbits, device=dev)
    for _ in range(max(1, sweeps)):
        for phase in range(period):
            pos = phase + torch.arange(npos, device=dev) * period
            # Accumulate each candidate's loss in the reference's window order
            # (offset 0 .. span-1) instead of reducing the window axis as one tree.
            cand = torch.zeros(
                (1 << nbits, n, npos), dtype=torch.float32, device=dev
            )
            for j in range(span):
                affected = (pos + block - j) % block
                q = _windows(codes, nbits, depth)[:, affected]
                keep_j = ~(((1 << nbits) - 1) << (j * nbits))
                qc = (
                    (q[None, :, :] & keep_j) | (syms[:, None, None] << (j * nbits))
                ) & depth_mask
                idx = _decode_index(qc, a0, a1)
                val = (s[None, :, None] * table[idx]).to(torch.float16).to(torch.float32)
                diff = val - wf[:, affected][None, :, :]
                term = diff * diff
                if d is not None:
                    term = weight[affected][None, None, :] * term
                cand += term
            best_err = cand.min(0).values
            best_sym = torch.where(
                cand == best_err[None, :, :], syms[:, None, None], 1 << nbits
            ).min(0).values
            codes[:, pos] = best_sym
    return codes.to(torch.uint8)


def _case(
    nbits: int,
    block: int,
    depth: int = 16,
    n: int = 4,
    seed: int = 0,
    sweeps: int = 1,
    with_d: bool = False,
):
    """Deterministic tensors plus the codec and mirror-expanded fp16 lut."""
    torch.manual_seed(seed)
    codec = get_codec(depth, nbits, rounds=0)
    w = torch.randn(n, block) * 0.7
    scales = (torch.rand(n) * 0.06 + 0.005).to(torch.float16)
    init = torch.randint(0, 1 << nbits, (n, block), dtype=torch.uint8)
    lut = index_table_value(
        torch.arange(127), codec.table, codec.table_scale
    )  # fp16, mirror-expanded
    d = torch.rand(block) * 2.0 if with_d else None
    return codec, w, scales, init, lut, d


def _counterexample_row():
    """The exact HIP0 K2/B64 row 849 that exposed the reduction-order bug."""
    codec = get_codec(16, 2, rounds=0)
    w = torch.tensor([_COUNTEREXAMPLE_W], dtype=torch.float32)
    init = torch.tensor([_COUNTEREXAMPLE_INIT], dtype=torch.uint8)
    scales = torch.tensor([0.02], dtype=torch.float16)
    lut = index_table_value(torch.arange(127), codec.table, codec.table_scale)
    expected = torch.tensor([_COUNTEREXAMPLE_REF], dtype=torch.uint8)
    return codec, w, init, scales, lut, expected


# ------------------------------------------------------------------- CPU numerics


@pytest.mark.parametrize("nbits,block", _GPU_CASES)
@pytest.mark.parametrize("sweeps", [1, 2])
def test_fused_model_matches_reference_default_depth(nbits, block, sweeps):
    codec, w, scales, init, lut, _ = _case(nbits, block, sweeps=sweeps, seed=11)
    got = _fused_projection(
        w, init, scales, lut, codec.a0, codec.a1, nbits, 16, block, sweeps=sweeps
    )
    ref = project_codes(
        w, codec, block, init, scales, sweeps=sweeps, lut=lut.float()
    )
    assert torch.equal(got, ref)
    assert torch.equal(
        _weighted_sse(w, got, scales, lut, codec.a0, codec.a1, nbits, 16, None),
        _weighted_sse(w, ref, scales, lut, codec.a0, codec.a1, nbits, 16, None),
    )


def test_fused_model_actual_counterexample_row_order():
    """Row 849 keeps the reference's sym2; the old tree sum kept sym1."""
    codec, w, init, scales, lut, expected = _counterexample_row()
    got = _fused_projection(
        w, init, scales, lut, codec.a0, codec.a1, 2, 16, 64, sweeps=1
    )
    ref = project_codes(w, codec, 64, init, scales, sweeps=1, lut=lut.float())
    assert torch.equal(ref, expected)
    assert torch.equal(got, expected)
    assert int(got[0, _COUNTEREXAMPLE_COL]) == 2


@pytest.mark.parametrize(
    "depth,nbits,block",
    [
        (12, 2, 16),
        (12, 2, 32),
        (13, 2, 32),
        (9, 2, 16),
        (5, 2, 16),
        (12, 4, 32),
        (15, 4, 64),
    ],
)
def test_fused_model_matches_reference_off_default_depths(depth, nbits, block):
    """Covers the padded window axis for spans that are not powers of two."""
    codec, w, scales, init, lut, _ = _case(nbits, block, depth=depth, seed=23)
    got = _fused_projection(
        w, init, scales, lut, codec.a0, codec.a1, nbits, depth, block, sweeps=2
    )
    ref = project_codes(
        w, codec, block, init, scales, sweeps=2, lut=lut.float()
    )
    assert torch.equal(got, ref)


def test_fused_model_multiple_sweeps_follow_the_reference_and_do_not_regress():
    codec, w, scales, init, lut, _ = _case(2, 32, sweeps=1, seed=31)
    errors = []
    previous = None
    for sweeps in (1, 2, 3):
        got = _fused_projection(
            w, init, scales, lut, codec.a0, codec.a1, 2, 16, 32, sweeps=sweeps
        )
        ref = project_codes(
            w, codec, 32, init, scales, sweeps=sweeps, lut=lut.float()
        )
        assert torch.equal(got, ref)
        err = _weighted_sse(w, got, scales, lut, codec.a0, codec.a1, 2, 16, None)
        if previous is not None:
            assert (err <= previous).all()
        previous = err
        errors.append(err)
    assert not torch.equal(
        _fused_projection(
            w, init, scales, lut, codec.a0, codec.a1, 2, 16, 32, sweeps=1
        ),
        _fused_projection(
            w, init, scales, lut, codec.a0, codec.a1, 2, 16, 32, sweeps=2
        ),
    )


@pytest.mark.parametrize("nbits,block", [(2, 16), (2, 64), (4, 32)])
def test_fused_model_weights_d_match_the_reference(nbits, block):
    codec, w, scales, init, lut, d = _case(
        nbits, block, sweeps=2, seed=37, with_d=True
    )
    d = d.clone()
    d[::3] = 0.0  # zero weights are valid and must not perturb the candidate order
    got = _fused_projection(
        w, init, scales, lut, codec.a0, codec.a1, nbits, 16, block, sweeps=2, d=d
    )
    ref = project_codes(
        w, codec, block, init, scales, sweeps=2, d=d, lut=lut.float()
    )
    assert torch.equal(got, ref)
    left = _weighted_sse(w, got, scales, lut, codec.a0, codec.a1, nbits, 16, d)
    right = _weighted_sse(w, ref, scales, lut, codec.a0, codec.a1, nbits, 16, d)
    assert torch.equal(left, right)


def test_fused_model_zero_subnormal_scales_and_ties_keep_lowest_symbol():
    codec = get_codec(16, 2, rounds=0)
    lut = index_table_value(torch.arange(127), codec.table, codec.table_scale)
    torch.manual_seed(41)
    n, block = 3, 32
    w = torch.randn(n, block)
    init = torch.randint(0, 4, (n, block), dtype=torch.uint8)

    zero = torch.zeros(n, dtype=torch.float16)
    got = _fused_projection(w, init, zero, lut, codec.a0, codec.a1, 2, 16, block)
    assert torch.equal(got, torch.zeros_like(got))  # every candidate ties at 0
    ref = project_codes(w, codec, block, init, zero, lut=lut.float())
    assert torch.equal(got, ref)

    subnormal = torch.tensor(
        [2.0**-24, 2.0**-20, 2.0**-14], dtype=torch.float16
    )
    got = _fused_projection(w, init, subnormal, lut, codec.a0, codec.a1, 2, 16, block)
    ref = project_codes(w, codec, block, init, subnormal, lut=lut.float())
    assert torch.equal(got, ref)
    vals = _decoded_values(got, subnormal, lut, codec.a0, codec.a1, 2, 16)
    assert torch.isfinite(vals).all()
    assert torch.equal(
        _weighted_sse(w, got, subnormal, lut, codec.a0, codec.a1, 2, 16, None),
        _weighted_sse(w, ref, subnormal, lut, codec.a0, codec.a1, 2, 16, None),
    )

    # All-zero d makes every candidate tie exactly, so the lowest symbol wins.
    got = _fused_projection(
        w,
        init,
        torch.full((n,), 0.05, dtype=torch.float16),
        lut,
        codec.a0,
        codec.a1,
        2,
        16,
        block,
        d=torch.zeros(block),
    )
    assert torch.equal(got, torch.zeros_like(got))


def test_fused_model_row_strides_and_no_input_mutation():
    codec, w, scales, init, lut, d = _case(2, 32, sweeps=2, seed=43, with_d=True)
    wbuf = torch.randn(w.shape[0], 32 + 9)
    wbuf[:, :32] = w
    ibuf = torch.randint(0, 4, (init.shape[0], 32 + 5), dtype=torch.uint8)
    ibuf[:, :32] = init
    dbuf = torch.randn(32 + 2)
    dbuf[:32] = d
    w_view = wbuf[:, :32]
    init_view = ibuf[:, :32]
    d_view = dbuf[:32]
    assert w_view.stride(0) != w_view.shape[1]
    assert init_view.stride(0) != init_view.shape[1]

    snapshots = (
        w_view.clone(),
        init_view.clone(),
        scales.clone(),
        lut.clone(),
        d_view.clone(),
    )
    got = _fused_projection(
        w_view,
        init_view,
        scales,
        lut,
        codec.a0,
        codec.a1,
        2,
        16,
        32,
        sweeps=2,
        d=d_view,
    )
    ref = project_codes(
        w_view, codec, 32, init_view, scales, sweeps=2, d=d_view, lut=lut.float()
    )
    assert torch.equal(got, ref)
    for value, snapshot in zip(
        (w_view, init_view, scales, lut, d_view), snapshots
    ):
        assert torch.equal(value, snapshot)
    assert torch.equal(
        got,
        _fused_projection(
            w_view.contiguous(),
            init_view.contiguous(),
            scales,
            lut,
            codec.a0,
            codec.a1,
            2,
            16,
            32,
            sweeps=2,
            d=d_view.contiguous(),
        ),
    )


def test_fused_model_cyclic_seam_windows():
    codec, w, scales, init, lut, _ = _case(2, 16, seed=47)
    got = _fused_projection(
        w, init, scales, lut, codec.a0, codec.a1, 2, 16, 16, sweeps=2
    )
    assert torch.equal(
        got, project_codes(w, codec, 16, init, scales, sweeps=2, lut=lut.float())
    )
    for nbits in (2, 4):
        codec = get_codec(16, nbits, rounds=0)
        torch.manual_seed(53)
        codes = torch.randint(0, 1 << nbits, (2, 64), dtype=torch.uint8)
        cyclic = _windows(codes, nbits, 16)
        assert torch.equal(cyclic, window_indices(codes, nbits, 16, 64))
        span = _tap_width(16, nbits)
        padded = torch.cat(
            [codes.to(torch.int64), torch.zeros(2, span - 1, dtype=torch.int64)], 1
        )
        tail = torch.zeros_like(cyclic)
        for u in range(span):
            tail |= padded[:, u : u + 64] << (u * nbits)
        # The seam window must read codes across the block boundary; a non-cyclic
        # implementation falls back on zeros there and disagrees by construction.
        assert not torch.equal(
            cyclic[:, -1:], tail[:, -1:] & ((1 << 16) - 1)
        )


def test_projection_rejects_unsupported_shape_and_metadata():
    block = 64
    w = torch.zeros(0, block)
    init = torch.zeros(0, block, dtype=torch.uint8)
    scales = torch.zeros(0, dtype=torch.float16)
    lut = torch.zeros(127, dtype=torch.float16)
    args = (w, init, scales, lut, 1, 2)
    with pytest.raises(ValueError, match="2 or 4 bits"):
        project_codes_cuda(*args, nbits=3, depth=16, block=block)
    with pytest.raises(ValueError, match="depth"):
        project_codes_cuda(*args, nbits=2, depth=17, block=block)
    with pytest.raises(ValueError, match="depth"):
        project_codes_cuda(*args, nbits=2, depth=3, block=block)
    with pytest.raises(ValueError, match="block"):
        project_codes_cuda(
            torch.zeros(0, 8),
            torch.zeros(0, 8, dtype=torch.uint8),
            scales,
            lut,
            1,
            2,
            2,
            16,
            8,
        )
    with pytest.raises(ValueError, match="127 lut entries"):
        project_codes_cuda(w, init, scales, lut[:126], 1, 2, 2, 16, block)
    with pytest.raises(ValueError, match="fp16 scales"):
        project_codes_cuda(w, init, scales.float(), lut, 1, 2, 2, 16, block)
    with pytest.raises(ValueError, match="fp32 weights"):
        project_codes_cuda(w.half(), init, scales, lut, 1, 2, 2, 16, block)
    with pytest.raises(ValueError, match="uint8 init"):
        project_codes_cuda(w, init.int(), scales, lut, 1, 2, 2, 16, block)
    with pytest.raises(ValueError, match="a0/a1"):
        project_codes_cuda(w, init, scales, lut, 0x1_0000, 2, 2, 16, block)
    with pytest.raises(ValueError, match="d as"):
        project_codes_cuda(
            w, init, scales, lut, 1, 2, 2, 16, block, d=torch.zeros(63)
        )


def test_projection_geometry_matches_reference_schedule():
    from hi2.cptc import projection_period
    from hi2.cptc_projection_kernel import _geometry

    for nbits in (2, 4):
        for depth in range(4, 17):
            span = (depth + nbits - 1) // nbits
            reference_period = projection_period(depth, nbits)
            assert reference_period >= span
            for block in (16, 32, 64):
                pad_span, period, npos = _geometry(nbits, depth, block)
                assert (pad_span, period) == (span, reference_period)
                assert block % period == 0
                assert period * npos == block


def test_projection_n0_returns_empty_without_a_device():
    block = 64
    w = torch.zeros(0, block)
    init = torch.zeros(0, block, dtype=torch.uint8)
    scales = torch.zeros(0, dtype=torch.float16)
    lut = torch.zeros(127, dtype=torch.float16)
    before = (w.clone(), init.clone(), scales.clone(), lut.clone())
    out = project_codes_cuda(
        w, init, scales, lut, 0x1234, 0x5678, 2, 16, block, sweeps=2
    )
    assert out.shape == (0, block) and out.dtype == torch.uint8
    for value, snapshot in zip((w, init, scales, lut), before):
        assert torch.equal(value, snapshot)


# -------------------------------------------------------------------- GPU parity
# Authored, not run in this CPU proof: skip only when no HIP GPU or no Triton.


def _gpu_case(nbits, block, depth=16, n=8, seed=101, sweeps=1, with_d=False):
    """Deterministic CPU-generated inputs, moved to the GPU (portable seeds)."""
    codec, w, scales, init, lut, d = _case(
        nbits, block, depth=depth, n=n, seed=seed, sweeps=sweeps, with_d=with_d
    )
    return codec, w.cuda(), scales.cuda(), init.cuda(), lut.cuda(), (
        None if d is None else d.cuda()
    )


def _gpu_reference(w, codec, block, init, scales, sweeps, d, lut):
    return project_codes(
        w, codec, block, init, scales, sweeps=sweeps, d=d, lut=lut.float()
    )


def _check_gpu_parity(nbits, block, depth=16, sweeps=1, with_d=False, seed=101):
    codec, w, scales, init, lut, d = _gpu_case(
        nbits, block, depth=depth, sweeps=sweeps, with_d=with_d, seed=seed
    )
    snapshots = (
        w.clone(),
        init.clone(),
        scales.clone(),
        lut.clone(),
        None if d is None else d.clone(),
    )
    got = project_codes_cuda(
        w, init, scales, lut, codec.a0, codec.a1, nbits, depth, block,
        sweeps=sweeps, d=d,
    )
    ref = _gpu_reference(w, codec, block, init, scales, sweeps, d, lut)
    assert got.shape == (w.shape[0], block) and got.dtype == torch.uint8
    for value, snapshot in zip((w, init, scales, lut, d), snapshots):
        assert (value is None and snapshot is None) or torch.equal(value, snapshot)
    left = _weighted_sse(w, got, scales, lut, codec.a0, codec.a1, nbits, depth, d)
    right = _weighted_sse(w, ref, scales, lut, codec.a0, codec.a1, nbits, depth, d)
    # The reference runs the same greedy sweep with torch reductions; a different
    # summation order on another backend can only move an exact tie by an ULP, so
    # the GPU contract is the decoded objective ("match reference/objective"), not
    # bitwise equal codes. The CPU tests above pin the codes exactly.
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    assert (left <= right * (1 + 1e-5) + 1e-6).all()
    return got, ref, left, right


@requires_gpu
@pytest.mark.parametrize("nbits,block", _GPU_CASES)
def test_gpu_projection_matches_reference(nbits, block):
    _check_gpu_parity(nbits, block, sweeps=1)


@requires_gpu
def test_gpu_projection_actual_counterexample_row():
    """The exact row 849 that failed on the old tree reduction must match now."""
    codec, w, init, scales, lut, expected = _counterexample_row()
    dev = torch.device("cuda", 0)
    wg, initg, sg, lg = w.to(dev), init.to(dev), scales.to(dev), lut.to(dev)
    ref = project_codes(wg, codec, 64, initg, sg, sweeps=1, lut=lg.float())
    got = project_codes_cuda(
        wg, initg, sg, lg, codec.a0, codec.a1, 2, 16, 64, sweeps=1
    )
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    assert torch.equal(got.cpu(), expected)
    assert int(got[0, _COUNTEREXAMPLE_COL]) == 2


@requires_gpu
@pytest.mark.parametrize("nbits,block", [(2, 32), (4, 64)])
def test_gpu_projection_multiple_sweeps_and_weights(nbits, block):
    _, _, left1, right1 = _check_gpu_parity(nbits, block, sweeps=1, seed=111)
    _, _, left2, right2 = _check_gpu_parity(nbits, block, sweeps=2, seed=111)
    # A second sweep measurably helps this seeded case in the reference; an ignored
    # sweeps argument would leave the kernel's objective unchanged.
    assert float((right1 - right2).mean()) > 1e-3
    assert float((left1 - left2).mean()) > 1e-3
    assert (left2 <= left1 * (1 + 1e-6) + 1e-6).all()
    _check_gpu_parity(nbits, block, sweeps=2, with_d=True, seed=103)


@requires_gpu
@pytest.mark.parametrize("depth,nbits,block", [(12, 2, 32), (13, 2, 32), (12, 4, 32)])
def test_gpu_projection_off_default_depth_matches_reference(depth, nbits, block):
    _check_gpu_parity(nbits, block, depth=depth, sweeps=2, seed=107)


@requires_gpu
def test_gpu_projection_row_strides_and_no_mutation():
    codec, w, scales, init, lut, d = _gpu_case(2, 32, sweeps=2, seed=109, with_d=True)
    wbuf = torch.randn(w.shape[0], 32 + 7, device=w.device)
    wbuf[:, :32] = w
    ibuf = torch.randint(0, 4, (init.shape[0], 32 + 3), dtype=torch.uint8).cuda()
    ibuf[:, :32] = init
    dbuf = d.new_empty(32 + 5)
    dbuf[:32] = d
    w_view, init_view, d_view = wbuf[:, :32], ibuf[:, :32], dbuf[:32]
    snapshots = (
        w_view.clone(),
        init_view.clone(),
        scales.clone(),
        lut.clone(),
        d_view.clone(),
    )
    got = project_codes_cuda(
        w_view, init_view, scales, lut, codec.a0, codec.a1, 2, 16, 32,
        sweeps=2, d=d_view,
    )
    ref = _gpu_reference(w_view, codec, 32, init_view, scales, 2, d_view, lut)
    assert torch.equal(got, ref)
    for value, snapshot in zip(
        (w_view, init_view, scales, lut, d_view), snapshots
    ):
        assert torch.equal(value, snapshot)


@requires_gpu
def test_gpu_projection_zero_subnormal_scales_and_ties():
    codec, w, scales, init, lut, _ = _gpu_case(2, 32, seed=113)
    block = 32
    zeros = torch.zeros(w.shape[0], dtype=torch.float16, device=w.device)
    got = project_codes_cuda(
        w, init, zeros, lut, codec.a0, codec.a1, 2, 16, block
    )
    assert torch.equal(got, torch.zeros_like(got))
    assert torch.equal(
        got, _gpu_reference(w, codec, block, init, zeros, 1, None, lut)
    )
    subnormal = torch.tensor(
        [2.0**-24, 2.0**-20, 2.0**-14], dtype=torch.float16, device=w.device
    )
    subnormal = subnormal.repeat((w.shape[0] + 2) // 3)[: w.shape[0]]
    got = project_codes_cuda(
        w, init, subnormal, lut, codec.a0, codec.a1, 2, 16, block
    )
    # torch's GPU sum order is not bitwise stable across backends for these
    # near-tie subnormal candidates, so the canonical exact reference is CPU.
    # Pin the kernel exactly to that reference and to the GPU objective.
    ref_cpu = project_codes(
        w.cpu(), codec, block, init.cpu(), subnormal.cpu(), sweeps=1,
        lut=lut.cpu().float(),
    )
    assert torch.equal(got.cpu(), ref_cpu)
    ref_gpu = _gpu_reference(w, codec, block, init, subnormal, 1, None, lut)
    left = _weighted_sse(
        w, got, subnormal, lut, codec.a0, codec.a1, 2, 16, None
    )
    right = _weighted_sse(
        w, ref_gpu, subnormal, lut, codec.a0, codec.a1, 2, 16, None
    )
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    d0 = torch.zeros(block, device=w.device)
    got = project_codes_cuda(
        w, init, scales, lut, codec.a0, codec.a1, 2, 16, block, d=d0
    )
    assert torch.equal(got, torch.zeros_like(got))


@requires_gpu
def test_gpu_projection_n0_returns_empty():
    block = 64
    dev = torch.device("cuda")
    w = torch.zeros(0, block, device=dev)
    init = torch.zeros(0, block, dtype=torch.uint8, device=dev)
    scales = torch.zeros(0, dtype=torch.float16, device=dev)
    lut = torch.zeros(127, dtype=torch.float16, device=dev)
    out = project_codes_cuda(
        w, init, scales, lut, 0x1234, 0x5678, 2, 16, block, sweeps=2
    )
    assert out.shape == (0, block) and out.dtype == torch.uint8


@requires_gpu
@pytest.mark.parametrize("nbits,block", [(2, 16), (2, 32), (4, 32)])
def test_gpu_module_projection_flattens_blocks_without_crossing_seams(nbits, block):
    """``CptcFakeQuantLinear`` over several stored blocks equals the per-block CPU
    reference, for the projected codes and for the consumer-visible decode.

    The module issues one kernel call for the whole ``[rows x blocks, B]`` flatten.
    Crossing a stored block's seam, pairing a row with another block's scale or init
    codes, or reordering the flatten would all decode to different values than the
    independent per-block reference; every block has its own scale, and latent
    inputs differ across rows and blocks.
    """
    from hi2.cptc import assemble_cptc
    from hi2.qad import CptcFakeQuantLinear

    torch.manual_seed(211)
    rows, nblk = 5, 4
    k = nblk * block
    codec = get_codec(16, nbits, rounds=0)
    w0 = torch.randn(rows, k)
    codes = torch.randint(0, 1 << nbits, (rows, k), dtype=torch.uint8)
    scales = torch.linspace(0.125, 0.5, rows * nblk).reshape(rows, nblk).half()
    t = assemble_cptc(
        [(torch.arange(rows), codes, scales)],
        torch.full((rows,), nbits, dtype=torch.uint8),
        torch.full((rows,), block, dtype=torch.uint8),
        torch.zeros(k // 16, dtype=torch.uint8),
        k,
        {nbits: codec},
    )
    w = 2.5 * w0
    m = CptcFakeQuantLinear(w, t, None, "weights", "abs", sweeps=1).cuda()
    dev = m.latent.device
    row_ids = torch.arange(rows, device=dev)
    init = m._unpack(row_ids, nbits)
    pidx = (
        m.param_off[row_ids].long().unsqueeze(1) + torch.arange(nblk, device=dev)
    ).reshape(rows, nblk)
    s_row = m.scales()[pidx].float()
    got = m._project_codes(m.latent, codec, block, init, s_row)
    lut = index_table_value(torch.arange(127), codec.table, codec.table_scale)
    init_cpu, scales_cpu = init.cpu(), s_row.cpu()
    ref = torch.empty(rows, k, dtype=torch.uint8)
    for b in range(nblk):
        lo, hi = b * block, (b + 1) * block
        ref[:, lo:hi] = project_codes(
            w[:, lo:hi],
            codec,
            block,
            init_cpu[:, lo:hi],
            scales_cpu[:, b].half(),
            sweeps=1,
            lut=lut.float(),
        )
    assert got.shape == (rows, k) and got.dtype == torch.uint8
    assert torch.equal(got.cpu(), ref)
    # the consumer sees exactly the independent per-block decode
    stored = m.stored(False).cpu()
    expected = torch.empty(rows, k, dtype=torch.float32)
    for b in range(nblk):
        lo, hi = b * block, (b + 1) * block
        expected[:, lo:hi] = _decoded_values(
            ref[:, lo:hi],
            scales_cpu[:, b],
            lut,
            codec.a0,
            codec.a1,
            nbits,
            16,
        )
    assert torch.equal(stored, expected)


@requires_gpu
def test_gpu_frozen_export_is_portable_after_device_buffers_are_reused(tmp_path):
    from hi2.cptc import CptcTensor
    from hi2.pinned import PinnedWindow
    from hi2.qad import CptcFrozenCodeLinear

    fields = {
        "codes": torch.arange(12, dtype=torch.int32) + 1,
        "code_off": torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        "scale": torch.linspace(0.125, 0.5, 6).half(),
        "param_off": torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        "row_nbits": torch.full((3,), 2, dtype=torch.uint8),
        "row_block": torch.full((3,), 32, dtype=torch.uint8),
        "had": torch.tensor([1, 0, 1, 0], dtype=torch.uint8),
        "col_scale": torch.linspace(0.5, 1.5, 64).half(),
    }
    tensor = CptcTensor(**fields, shape=(3, 64), codecs={2: get_codec(16, 2, rounds=0)})
    module = CptcFrozenCodeLinear(tensor).cuda()
    artifact = module.export(PinnedWindow(nbytes=4096))
    payload = {name: getattr(artifact, name) for name in fields}
    assert all(value.device.type == "cpu" for value in payload.values())
    module.codes.zero_()
    path = tmp_path / "frozen.pt"
    torch.save(payload, path)
    restored = torch.load(path, weights_only=True)
    for name, original in fields.items():
        assert restored[name].device.type == "cpu", name
        assert torch.equal(restored[name], original), name
