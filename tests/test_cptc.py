# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU contracts for the opt-in CPTC format (arXiv 2611.22019).

These tests use the paper's tiny brute-force cases and the revision-1 runtime
metadata contract; they do not run a model or a native kernel.
"""

import base64
import copy

import pytest
import torch

from hi2 import cptc
from hi2.cptc import (
    CPTC_FIELDS,
    CptcTensor,
    assemble_cptc,
    checkpoint_config,
    codecs_from_checkpoint_config,
    encode_rows_cptc,
    get_codec,
)


def _roundtrip_config():
    codecs = {k: get_codec(16, k, rounds=0) for k in (2, 4)}
    return codecs, checkpoint_config(codecs)


def test_pack_unpack_and_windows_match_the_independent_bitstream():
    gen = torch.Generator().manual_seed(7)
    for nbits, block in ((2, 16), (2, 64), (4, 16), (4, 32), (4, 64)):
        codes = torch.randint(
            0, 1 << nbits, (3, block), generator=gen, dtype=torch.uint8
        )
        words = cptc.pack_codes(codes, nbits)
        assert torch.equal(cptc.unpack_codes(words, nbits), codes)
        # Independent Algorithm-2 stream: value t occupies bits t*K..t*K+K-1,
        # and the last L-K bits wrap to the front of the block.
        depth = 16
        q = cptc.window_indices(codes, nbits, depth, block)
        for row in range(codes.shape[0]):
            stream = sum(int(codes[row, t]) << (t * nbits) for t in range(block))
            wrapped = stream | (stream << (block * nbits))
            for t in range(block):
                want = (wrapped >> (t * nbits)) & ((1 << depth) - 1)
                assert int(q[row, t]) == want


def test_closed_viterbi_matches_bruteforce_tiny_block():
    """L=6,K=2,T=8: enumerate all 2^16 streams, then prove the closed DP matches.

    This is the objective-agreement invariant: DP cost == loss of independent
    Algorithm-2 decode, with the pinned start statistic equal to the end statistic.
    """
    torch.manual_seed(0)
    K, L, B = 2, 6, 8
    codec = cptc.generate_table(L, K, rounds=0)
    w = torch.randn(1, B)
    d = torch.ones(B)
    scale = torch.tensor([1.0])
    state = codec.state_values()

    best = (float("inf"), None)
    for bits in range(1 << (B * K)):
        codes = torch.tensor(
            [(bits >> (t * K)) & ((1 << K) - 1) for t in range(B)],
            dtype=torch.uint8,
        ).reshape(1, B)
        loss = float(
            cptc._block_sse(w, d, state, codes, cptc._CodecView(K, L), B, scale)[0][0]
        )
        if loss < best[0]:
            best = (loss, codes)

    # Full closed DP over every start statistic (S=2^(L-K)=16).
    s_len, step = 1 << (L - K), 1 << K
    r = torch.arange(s_len)
    high = torch.arange(step)
    q_all = (high[:, None] * s_len + r[None, :]).reshape(-1)
    values = (scale.reshape(-1, 1) * state[None, :]).to(torch.float16).float()
    v_all = values[0][q_all].reshape(step, s_len)
    a_idx = (high[:, None] * (s_len // step) + (r[None, :] >> K)).reshape(step, s_len)
    walk = list(range(B - 1, -1, -1))

    def forward(v0):
        a, bps = v0, []
        for t in walk:
            base = a[:, a_idx].reshape(1, step, s_len)
            cost = base + (v_all[None, :, :] - w[:, t, None, None]) ** 2 * d[t]
            a, bp = cost.min(dim=1)
            bps.append(bp)
        return a, bps

    dp_best = (float("inf"), None)
    for pin in range(s_len):
        v0 = torch.full((1, s_len), float("inf"))
        v0[0, pin] = 0.0
        af, bps = forward(v0)
        cur = torch.tensor([pin])
        codes = torch.zeros(1, B, dtype=torch.uint8)
        for i in range(B - 1, -1, -1):
            t = walk[i]
            codes[:, t] = (cur & (step - 1)).to(torch.uint8)
            if i:
                cur = bps[i][torch.arange(1), cur] * (s_len // step) + (cur >> K)
        loss = float(
            cptc._block_sse(w, d, state, codes, cptc._CodecView(K, L), B, scale)[0][0]
        )
        assert abs(loss - float(af[0, pin])) < 1e-5
        if loss < dp_best[0]:
            dp_best = (loss, codes)

    assert abs(dp_best[0] - best[0]) < 1e-5
    assert torch.equal(dp_best[1], best[1])


def test_config_is_the_runtime_revision_1_contract():
    codecs, cfg = _roundtrip_config()
    assert set(cfg) == {
        "name",
        "revision",
        "depth",
        "tap_q",
        "layout",
        "shift",
        "block",
        "tables",
    }
    assert cfg["revision"] == 1
    assert cfg["layout"] == "mirror_int8"
    assert cfg["block"] == "row_block"
    assert set(cfg["tables"]) == {"2", "4"}
    for k in (2, 4):
        entry = cfg["tables"][str(k)]
        assert set(entry) == {"a0", "a1", "table_scale", "tbl_b64"}
        raw = base64.b64decode(entry["tbl_b64"], validate=True)
        assert len(raw) == 64
        assert raw[-1] == 0
    back = codecs_from_checkpoint_config(cfg)
    for k in (2, 4):
        assert torch.equal(back[k].table, codecs[k].table)
        assert float(back[k].table_scale) == float(codecs[k].table_scale)
    hist = cptc.index_histogram(16, 4)
    assert torch.equal(hist, hist.flip(0))


def test_checkpoint_rejects_fp16_overflow_without_mixing_row_codebooks():
    from hi2.checkpoint import _validate_cptc_fields

    table = torch.arange(-63, 1, dtype=torch.int8)
    codecs = {
        k: cptc.CptcCodec(
            nbits=k, depth=16, a0=a0, a1=a1, table=table,
            table_scale=torch.tensor(scale, dtype=torch.float16),
        )
        for k, a0, a1, scale in (
            (2, 0x7AE6, 0xC8B3, 1 / 128),
            (4, 0x6A78, 0x85CF, 1 / 32),
        )
    }
    fields = {
        "codes": torch.zeros(12, dtype=torch.int32),
        "code_off": torch.tensor([0, 4, 12], dtype=torch.int32),
        "scale": torch.tensor([65504, 0.25], dtype=torch.float16),
        "param_off": torch.tensor([0, 1, 2], dtype=torch.int32),
        "row_nbits": torch.tensor([2, 4], dtype=torch.uint8),
        "row_block": torch.tensor([64, 64], dtype=torch.uint8),
        "had": torch.zeros(4, dtype=torch.uint8),
    }
    tensor = CptcTensor(**fields, shape=(2, 64), codecs=codecs)
    decoded = cptc.dequantize_stored_cptc(tensor)
    assert bool(torch.isfinite(decoded).all())
    _validate_cptc_fields("valid mixed rows", fields, codecs)
    bad = copy.deepcopy(fields)
    bad["scale"][1] = 65504
    with pytest.raises(ValueError):
        _validate_cptc_fields("overflow", bad, codecs)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(revision=2),
        lambda c: c.update(depth=15),
        lambda c: c.update(layout="float"),
        lambda c: c.update(unknown=1),
        lambda c: c["tables"]["2"].update(unknown=1),
        lambda c: c["tables"]["2"].update(a0=1, a1=65535),
        lambda c: c["tables"]["2"].update(table_scale=60000),
        lambda c: c["tables"]["2"].update(
            tbl_b64=base64.b64encode(b"\x80" + bytes(63)).decode()
        ),
        lambda c: c["tables"].pop("2"),
        lambda c: c["tables"]["2"].update(tbl_b64="not-base64"),
        lambda c: c["tables"]["2"].update(table_scale=float("nan")),
        lambda c: c["tables"]["4"].update(a0=65536),
        lambda c: c["tables"]["4"].update(tbl_b64=base64.b64encode(b"\0" * 63).decode()),
    ],
)
def test_invalid_metadata_fails_closed(change):
    _, cfg = _roundtrip_config()
    bad = copy.deepcopy(cfg)
    change(bad)
    with pytest.raises(ValueError):
        codecs_from_checkpoint_config(bad)


def test_qad_weight_latent_projection_changes_codes_and_matches_export():
    from hi2.qad import CptcFakeQuantLinear

    torch.manual_seed(3)
    K, L, B = 2, 16, 16
    codec = get_codec(L, K, rounds=0)
    w = torch.randn(3, 32)
    d = torch.ones(32)
    codes, scales, _ = encode_rows_cptc(w, d, K, B, codec, effort="fast")
    t = assemble_cptc(
        [(torch.arange(3), codes, scales)],
        torch.full((3,), K, dtype=torch.uint8),
        torch.full((3,), B, dtype=torch.uint8),
        torch.zeros(2, dtype=torch.uint8),
        32,
        {2: codec},
    )
    lat = CptcFakeQuantLinear(w, t, sweeps=2)
    with torch.no_grad():
        assert torch.equal(lat.export().codes, t.codes)
    stored = lat.stored(True)
    loss = (stored - 0.25).pow(2).sum()
    loss.backward()
    assert lat.delta.grad is not None and torch.isfinite(lat.delta.grad).all()
    assert lat.latent.grad is not None and torch.isfinite(lat.latent.grad).all()
    with torch.no_grad():
        lat.latent[0].neg_()  # a real weight move, not only a scale move
        lat.delta.add_(0.05)
    out = lat.export()
    assert isinstance(out, CptcTensor)
    assert torch.equal(out.codes, t.codes) is False
    assert torch.equal(out.code_off, t.code_off)
    assert not torch.equal(out.scale, t.scale)
    assert torch.equal(
        cptc.dequantize_stored_cptc(out).float(), lat.stored(False)
    )


def test_projection_covers_off_default_depth_and_blocks():
    from hi2.cptc import projection_period, project_codes

    torch.manual_seed(7)
    K, L = 2, 12
    codec = get_codec(L, K, rounds=0)
    w = torch.randn(2, 64)
    d = torch.ones(64)
    assert projection_period(L, K) == 8
    for B in (16, 32, 64):
        init, scales, _ = encode_rows_cptc(
            w[:, :B], d[:B], K, B, codec, effort="fast"
        )
        p = project_codes(w[:, :B], codec, B, init, scales[:, 0], sweeps=2)
        dec = cptc.block_values(p, codec, scales[:, 0], B).float()
        assert torch.isfinite(dec).all()
        exact = encode_rows_cptc(w[:, :B], d[:B], K, B, codec, effort="exact")[2]
        proj = (d[:B] * (dec - w[:, :B]) ** 2).sum(-1)
        assert (proj >= exact - 1e-4).all()
    odd = get_codec(13, K, rounds=0)
    init, scales, _ = encode_rows_cptc(w[:, :16], d[:16], K, 16, odd, effort="fast")
    p = project_codes(w[:, :16], odd, 16, init, scales[:, 0], sweeps=1)
    assert torch.isfinite(cptc.block_values(p, odd, scales[:, 0], 16).float()).all()
    cfg = cptc.checkpoint_config({2: get_codec(12, 2), 4: get_codec(12, 4)})
    parsed = cptc.codecs_from_checkpoint_config(cfg)
    assert parsed[2].depth == 12 and parsed[4].depth == 12


def test_cptc_qad_keeps_default_300_200_schedule():
    from hi2.qad import FINAL_KL_STEPS, PRESETS, QadConfig

    assert QadConfig().steps == 300
    assert FINAL_KL_STEPS == 200
    assert PRESETS["default"].get("final_kl") == 200
    assert "steps" not in PRESETS["default"]


def test_qad_training_changes_codes_and_improves_heldout_loss():
    from torch import nn

    from hi2.qad import CptcFakeQuantLinear, QadConfig, train_layer

    torch.manual_seed(11)
    K, B, n = 2, 32, 32
    codec = get_codec(16, K, rounds=0)
    w0 = torch.randn(n, n)
    codes, scales, _ = encode_rows_cptc(w0, torch.ones(n), K, B, codec, effort="fast")
    t = assemble_cptc(
        [(torch.arange(n), codes, scales)],
        torch.full((n,), K, dtype=torch.uint8),
        torch.full((n,), B, dtype=torch.uint8),
        torch.zeros(n // 16, dtype=torch.uint8),
        n,
        {K: codec},
    )
    layer = nn.Sequential(nn.Linear(n, n, bias=False))
    target = torch.randn(n, n)
    xs = [torch.randn(1, 4, n) for _ in range(4)]
    ys = [x @ target.T for x in xs]
    m = CptcFakeQuantLinear(w0, t, sweeps=1, units="step")
    qc = QadConfig(
        steps=24,
        batch=2,
        lr_w=0.03,
        lr_p=0.02,
        evals=2,
        keep_best=False,
        data_parallel=False,
        seed=5,
    )
    info = train_layer(
        layer, {"0": m}, xs, ys, [0, 1], [2, 3], {}, qc, torch.device("cpu")
    )
    out = m.export()
    assert info["val_after"] < info["val_before"]
    assert not torch.equal(out.codes, t.codes)
    assert torch.equal(
        cptc.dequantize_stored_cptc(out).float(), m.stored(False)
    )


def test_cptc_gptq_nondiagonal_matches_sequential_frozen_q_reference():
    from hi2.cptc import encode_block, encode_rows_cptc_gptq

    torch.manual_seed(21)
    N, K, bit, block = 3, 32, 2, 8
    codec = get_codec(16, bit, rounds=0)
    w = torch.randn(N, K)
    d = torch.rand(K) + 0.5
    u = torch.triu(torch.randn(K, K) * 0.15)
    u.diagonal().copy_(torch.rand(K) + 0.5)
    got, _, _, _, wc, states = encode_rows_cptc_gptq(
        w, d, u, bit, block, codec, effort="fast", return_state=True
    )
    ref = torch.zeros(N, K, dtype=torch.uint8)
    wc_ref = w.clone()
    ref_states = []
    for j0 in range(0, K, block):
        j1 = j0 + block
        wb = wc_ref[:, j0:j1].clone()
        cb, sb, _ = encode_block(
            wb, d[j0:j1], codec, block, method="viterbi", effort="fast"
        )
        ref[:, j0:j1] = cb
        q = cptc.block_values(cb, codec, sb, block).float()
        for i in range(block):
            j = j0 + i
            e = (wc_ref[:, j] - q[:, i]) / u[j, j]
            if j + 1 < K:
                wc_ref[:, j + 1 :] -= e[:, None] * u[j, j + 1 :][None, :]
        ref_states.append(wc_ref.clone())
    assert torch.equal(got, ref)
    for j0, state, ref_state in zip(range(0, K, block), states, ref_states):
        j1 = j0 + block
        assert torch.allclose(state[:, j1:], ref_state[:, j1:], atol=1e-6, rtol=1e-5)
    # The within-block errors and the pushed future block are the claimed
    # recurrence E U_BB = W_B - Q_B, E @ U_Btail; the old pre-compensation loop
    # differs here for non-diagonal U even when the codes happen to agree.
    j0, j1 = 0, block
    wb = w[:, j0:j1].clone()
    cb, sb, _ = encode_block(
        wb, d[j0:j1], codec, block, method="viterbi", effort="fast"
    )
    q = cptc.block_values(cb, codec, sb, block).float()
    ub = torch.triu(u[j0:j1, j0:j1])
    e_fast = torch.linalg.solve_triangular(
        ub.T.contiguous(), (wb - q).T.contiguous(), upper=False
    ).T
    e_ref, cur = [], w.clone()
    for i in range(block):
        j = j0 + i
        e = (cur[:, j] - q[:, i]) / u[j, j]
        e_ref.append(e)
        cur[:, j + 1 :] -= e[:, None] * u[j, j + 1 :][None, :]
    e_ref = torch.stack(e_ref, 1)
    assert torch.allclose(e_fast, e_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        e_fast @ u[j0:j1, j1:], e_ref @ u[j0:j1, j1:], atol=1e-5, rtol=1e-5
    )


def _short_stream_values(streams, table, scale, nbits, depth):
    # Manual Algorithm 2 cyclic windows, independent of window_indices.
    streams = torch.as_tensor(streams, dtype=torch.int64)
    t = streams.shape[-1]
    q = torch.zeros_like(streams)
    for j in range((depth + nbits - 1) // nbits):
        q |= torch.roll(streams, -j, dims=-1) << (j * nbits)
    q &= (1 << depth) - 1
    a0, a1 = cptc.coefficients(depth, nbits)
    return cptc.index_table_value(cptc.decode_index(q, a0, a1), table, scale)


def test_viterbi_short_depth_wrap_matches_exhaustive():
    from itertools import product
    from hi2.cptc import viterbi_codes, window_indices

    torch.manual_seed(71)
    table = torch.randint(-127, 127, (64,), dtype=torch.int8)
    scale = torch.tensor(0.017, dtype=torch.float16)
    for nbits, depth, block in ((4, 6, 3), (3, 5, 3)):
        a0, a1 = cptc.coefficients(depth, nbits)
        states = torch.arange(1 << depth)
        values = cptc.index_table_value(
            cptc.decode_index(states, a0, a1), table, scale
        ).float()
        w = torch.randn(1, block)
        d = torch.rand(block) + 0.1
        codes, sse = viterbi_codes(w, d, values, nbits, depth, block)
        streams = torch.tensor(
            list(product(range(1 << nbits), repeat=block)), dtype=torch.int64
        )
        vals = _short_stream_values(streams, table, scale, nbits, depth)
        exhaustive = (d * (vals - w) ** 2).sum(-1)
        best = exhaustive.min()
        assert abs(float(sse) - float(best)) < 1e-5
        manual = _short_stream_values(codes, table, scale, nbits, depth)
        q = window_indices(codes, nbits, depth, block)
        seam = values[q]
        assert torch.equal(manual[0].half(), seam[0].half())
        # The emitted stream is a closed path: its last-to-first window decodes too.
        last = (q[:, -1] >> nbits)
        assert int(last[0]) < (1 << (depth - nbits))


def test_project_codes_blocks_matches_single_block_reference():
    from hi2.cptc import project_codes, project_codes_blocks

    gen = torch.Generator().manual_seed(19)
    codec = get_codec(16, 4, rounds=0)
    n, k, block = 3, 64, 16
    w = torch.randn(n, k, generator=gen)
    init = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    scales = torch.rand(n, k // block, generator=gen).half() + 0.02
    d = torch.rand(k, generator=gen) + 0.1
    got = project_codes_blocks(
        w, codec, block, init, scales, sweeps=2, d=d
    )
    ref = torch.empty_like(init)
    for b in range(k // block):
        lo, hi = b * block, (b + 1) * block
        ref[:, lo:hi] = project_codes(
            w[:, lo:hi],
            codec,
            block,
            init[:, lo:hi],
            scales[:, b],
            sweeps=2,
            d=d[lo:hi],
        )
    assert torch.equal(got, ref)


def test_zero_and_subnormal_scales_survive_zero_step_exports():
    from hi2.qad import CptcFakeQuantLinear, CptcFrozenCodeLinear

    torch.manual_seed(31)
    K, L, B = 2, 16, 16
    codec = get_codec(L, K, rounds=0)
    w = torch.randn(2, 32)
    codes, scales, _ = encode_rows_cptc(
        w, torch.ones(32), K, B, codec, effort="fast"
    )
    subnormal = torch.tensor(5.960464477539063e-08, dtype=torch.float16)
    scales = scales.clone()
    scales[0, 0] = 0.0
    scales[1, 1] = subnormal
    t = assemble_cptc(
        [(torch.arange(2), codes, scales)],
        torch.full((2,), K, dtype=torch.uint8),
        torch.full((2,), B, dtype=torch.uint8),
        torch.zeros(2, dtype=torch.uint8),
        32,
        {K: codec},
    )
    frozen = CptcFrozenCodeLinear(t)
    out = frozen.export()
    assert torch.equal(out.codes, t.codes)
    assert torch.equal(out.scale, t.scale)
    assert bool((out.scale == 0).any()) and bool((out.scale > 0).any())
    assert bool((out.scale == subnormal).any())

    full = CptcFakeQuantLinear(w, t, None, "ptq", "abs", sweeps=1)
    full_out = full.export()
    assert torch.equal(full_out.scale, t.scale)
    assert full_out.scale.min().item() == 0.0


def test_zero_and_tiny_weights_decode_through_consumer_at_default_depth():
    """A zero block must decode exactly to zero; a tiny block must retain a
    positive subnormal fp16 scale and a bounded absolute decode error."""
    from hi2.qad import CptcFrozenCodeLinear

    torch.manual_seed(41)
    K, L, B, rows, k = 2, 16, 16, 2, 32
    codec = get_codec(L, K, rounds=0)

    def encoded(w):
        d = torch.ones(k)
        codes, scales, _ = encode_rows_cptc(
            w, d, K, B, codec, effort="normal"
        )
        t = assemble_cptc(
            [(torch.arange(rows), codes, scales)],
            torch.full((rows,), K, dtype=torch.uint8),
            torch.full((rows,), B, dtype=torch.uint8),
            torch.zeros(2, dtype=torch.uint8),
            k,
            {K: codec},
        )
        return t, scales

    zero = torch.zeros(rows, k)
    t_zero, scales_zero = encoded(zero)
    assert torch.equal(scales_zero, torch.zeros_like(scales_zero))
    decoded_zero = cptc.dequantize_stored_cptc(t_zero).float()
    assert decoded_zero.shape == zero.shape
    assert torch.equal(decoded_zero, zero)
    consumer_zero = CptcFrozenCodeLinear(t_zero).stored(False)
    assert torch.equal(consumer_zero, zero)
    ref_zero = cptc.reference_linear_cptc(torch.randn(3, k), t_zero)
    assert torch.equal(ref_zero, torch.zeros_like(ref_zero))

    tiny_value = torch.tensor(2.0**-20, dtype=torch.float32)
    tiny = torch.full((rows, k), tiny_value)
    t_tiny, scales_tiny = encoded(tiny)
    assert torch.isfinite(scales_tiny).all()
    assert bool((scales_tiny > 0).all())
    assert float(scales_tiny.max()) < 2.0**-14
    decoded_tiny = cptc.dequantize_stored_cptc(t_tiny).float()
    consumer_tiny = CptcFrozenCodeLinear(t_tiny).stored(False)
    assert torch.isfinite(decoded_tiny).all()
    assert torch.isfinite(consumer_tiny).all()
    # The fp16 lattice is coarser than this weight; the point of the regression
    # is that a zero/subnormal scale is not replaced by the old 2^-14 floor.
    assert (decoded_tiny - tiny).abs().max() <= 2.0**-12
    assert (consumer_tiny - tiny).abs().max() <= 2.0**-12
    assert not torch.equal(decoded_tiny, torch.zeros_like(decoded_tiny))


def test_final_kl_cptc_student_keeps_codes_frozen_and_trains_scales():
    from hi2.qad import CptcFrozenCodeLinear

    torch.manual_seed(9)
    K, L, B = 4, 16, 32
    codec = get_codec(L, K, rounds=0)
    w = torch.randn(2, 32)
    d = torch.ones(32)
    codes, scales, _ = encode_rows_cptc(w, d, K, B, codec, effort="fast")
    t = assemble_cptc(
        [(torch.arange(2), codes, scales)],
        torch.full((2,), K, dtype=torch.uint8),
        torch.full((2,), B, dtype=torch.uint8),
        torch.zeros(2, dtype=torch.uint8),
        32,
        {K: codec},
    )
    m = CptcFrozenCodeLinear(t)
    assert not hasattr(m, "latent")
    loss = (m.weight() - 0.1).pow(2).sum()
    loss.backward()
    assert m.delta.grad is not None and torch.isfinite(m.delta.grad).all()
    with torch.no_grad():
        m.delta.add_(0.05)
    out = m.export()
    assert torch.equal(out.codes, t.codes)
    assert torch.equal(out.code_off, t.code_off)
    assert not torch.equal(out.scale, t.scale)
    assert torch.equal(
        cptc.dequantize_stored_cptc(out).float(), m.stored(False)
    )


def test_checkpoint_tensor_io_and_consumer_numerics(tmp_path):
    from safetensors.torch import save_file
    from safetensors import safe_open

    from hi2.checkpoint import from_tensors, read_tensors, to_tensors

    torch.manual_seed(5)
    K, L, B = 4, 16, 32
    codec = get_codec(L, K, rounds=0)
    w = torch.randn(2, 64)
    codes, scales, _ = encode_rows_cptc(w, torch.ones(64), K, B, codec, effort="fast")
    t = assemble_cptc(
        [(torch.arange(2), codes, scales)],
        torch.full((2,), K, dtype=torch.uint8),
        torch.full((2,), B, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
        64,
        {K: codec},
    )
    path = tmp_path / "c.safetensors"
    save_file(to_tensors(t, "layer"), str(path))
    with safe_open(path, "pt") as f:
        d = read_tensors(f, "layer")
    back = from_tensors(d, "layer", {K: codec})
    assert torch.equal(back.codes, t.codes)
    assert torch.equal(back.scale, t.scale)
    assert torch.equal(
        cptc.dequantize_stored_cptc(back), cptc.dequantize_stored_cptc(t)
    )
    x = torch.randn(5, 64)
    ref = cptc.reference_linear_cptc(x, t)
    dense = cptc.dequantize_stored_cptc(t).float()
    assert torch.allclose(ref.float(), x @ dense.T, atol=1e-5)
    assert set(to_tensors(t, "layer")) == {
        f"layer.hi2_{f}" for f in CPTC_FIELDS
    }
