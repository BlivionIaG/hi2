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


def _scale_fixture(rows=4, k=32, nbits=2, block=16):
    """A CPTC tensor whose scale rows cover zero, subnormal and normal fp16 values."""
    codec = get_codec(16, nbits, rounds=0)
    w = torch.randn(rows, k, generator=torch.Generator().manual_seed(83))
    codes, scales, _ = encode_rows_cptc(
        w, torch.ones(k), nbits, block, codec, effort="fast"
    )
    scales = scales.clone()
    scales[0, :] = 0.0  # a completely zero row
    scales[1, 0] = 0.0
    scales[1, 1] = torch.tensor(5.960464477539063e-08, dtype=torch.float16)
    scales[3, 0] = 0.0
    t = assemble_cptc(
        [(torch.arange(rows), codes, scales)],
        torch.full((rows,), nbits, dtype=torch.uint8),
        torch.full((rows,), block, dtype=torch.uint8),
        torch.zeros(k // 16, dtype=torch.uint8),
        k,
        {nbits: codec},
    )
    return w, t


def test_cptc_row_u8_scale_units_keep_initialization_and_trust_region():
    import math

    from hi2.qad import CptcFakeQuantLinear, CptcFrozenCodeLinear

    w, t = _scale_fixture()
    frozen = CptcFrozenCodeLinear(t, scale_units="row_u8")
    off = frozen.param_off.to(torch.int64)
    scale0 = frozen.scale0
    # The unit is exactly the row's initial maximum block scale / 255, repeated
    # over the row's block positions; a zero row gets unit 0.
    for row in range(off.numel() - 1):
        lo, hi = int(off[row]), int(off[row + 1])
        want = scale0[lo:hi].max() / 255.0
        assert torch.equal(
            frozen.scale_unit[lo:hi], want.expand(hi - lo)
        ), row
    assert float(frozen.scale_unit[off[1] : off[2]].max()) > 0
    assert torch.equal(frozen.scales(), scale0)  # exact at delta = 0
    out = frozen.export()
    assert torch.equal(out.scale, t.scale)
    assert bool((out.scale == 0).any())
    assert bool((out.scale == torch.tensor(5.960464477539063e-08, dtype=torch.float16)).any())
    assert torch.equal(cptc.dequantize_stored_cptc(out)[0], torch.zeros(32, dtype=torch.float64))

    with torch.no_grad():
        frozen.delta[int(off[1])] = 1.0
        frozen.delta[int(off[1]) + 1] = -2.0
        frozen.delta[int(off[2])] = 50.0
    want = (scale0 + frozen.delta * frozen.scale_unit).clamp(0.0, 65504.0)
    assert torch.equal(frozen.scales(), want)

    with torch.no_grad():
        frozen.delta.fill_(1e9)
        frozen.project()
    scales = frozen.scales()
    positive = scale0 > 0
    assert bool((scales >= 0).all())
    assert bool(
        (scales[positive] <= scale0[positive] * (math.exp(0.5) + 1e-6)).all()
    )
    assert bool(
        (scales[positive] >= scale0[positive] * (math.exp(-0.5) - 1e-6)).all()
    )
    assert bool((scales[scale0 == 0] == 0).all())
    assert torch.equal(frozen.export().scale, scales.half())

    log = CptcFrozenCodeLinear(t)  # default log path is unchanged
    with torch.no_grad():
        log.delta.fill_(1e9)
        log.project()
    assert torch.equal(log.delta, torch.full_like(log.delta, 0.5))
    assert torch.allclose(
        log.scales(),
        (scale0 * math.exp(0.5)).clamp(0.0, 65504.0),
        rtol=1e-6,
        atol=0.0,
    )

    lat = CptcFakeQuantLinear(
        w, t, None, "ptq", "abs", sweeps=1, scale_units="row_u8"
    )
    lat_out = lat.export()
    assert torch.equal(lat_out.scale, t.scale)
    assert torch.equal(
        cptc.dequantize_stored_cptc(lat_out).float(), lat.stored(False)
    )
    assert torch.equal(lat.stored(False)[0], torch.zeros(32))
    normal = int(lat.param_off[2])
    with torch.no_grad():
        lat.delta[normal].add_(0.5)
        lat.latent[2].add_(0.01)
    moved = lat.export()
    expected = (t.scale[normal].float() + 0.5 * lat.scale_unit[normal]).half()
    assert expected > t.scale[normal]
    assert moved.scale[normal] == expected
    assert torch.equal(cptc.dequantize_stored_cptc(moved).float(), lat.stored(False))


def _two_width_tensors():
    codecs = {k: get_codec(16, k, rounds=0) for k in (2, 4)}
    out = {}
    for nbits, block in ((2, 16), (4, 32)):
        w = torch.randn(3, 64, generator=torch.Generator().manual_seed(90 + nbits))
        codes, scales, _ = encode_rows_cptc(
            w, torch.ones(64), nbits, block, codecs[nbits], effort="fast"
        )
        out[nbits] = assemble_cptc(
            [(torch.arange(3), codes, scales)],
            torch.full((3,), nbits, dtype=torch.uint8),
            torch.full((3,), block, dtype=torch.uint8),
            torch.zeros(4, dtype=torch.uint8),
            64,
            {nbits: codecs[nbits]},
        )
    return codecs, out


def test_cptc_shared_tables_are_one_global_pair_and_exports_agree():
    import copy

    from hi2.finalkl import CptcLinear
    from hi2.qad import CptcSharedTables, cptc_codecs_from_tensors

    codecs, tensors = _two_width_tensors()
    initial = {k: c.table.clone() for k, c in codecs.items()}
    book = CptcSharedTables(codecs)
    m2 = CptcLinear(tensors[2], tables=book)
    m4 = CptcLinear(tensors[4], tables=book)
    assert m2.tables is book and m4.tables is book
    # Model/device staging must not duplicate or replace the one master copy.
    params = list(book.parameters())
    m2.to(torch.device("cpu"))
    assert m2.tables is book and m4.tables is book
    for before, after in zip(params, book.parameters()):
        assert before is after

    loss = m2.weight().square().sum() + m4.weight().square().sum()
    loss.backward()
    for p in book.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert sum(float(p.grad.abs().sum()) for p in book.parameters()) > 0
    # Fixed metadata is frozen and the input/PTQ codecs are never mutated.
    for k, codec in codecs.items():
        assert torch.equal(codec.table, initial[k])
        assert torch.equal(book._codecs[k].table_scale, codec.table_scale)
        assert book._codecs[k].a0 == codec.a0 and book._codecs[k].a1 == codec.a1

    with torch.no_grad():
        book.table_u[2][0] += 0.6
        book.table_u[4][1] -= 0.6
        book.project()
    out2, out4 = m2.export(), m4.export()
    selected = {k: book.selected_int8(k) for k in (2, 4)}
    assert set(out2.codecs) == {2} and set(out4.codecs) == {4}
    for out, used in ((out2, (2,)), (out4, (4,))):
        for k in used:
            assert torch.equal(out.codecs[k].table, selected[k])
            assert out.codecs[k].table_scale.item() == codecs[k].table_scale.item()
            assert out.codecs[k].a0 == codecs[k].a0
            assert int(out.codecs[k].table[-1]) == 0
            assert out.codecs[k].table.dtype == torch.int8
            assert bool((out.codecs[k].table <= 127).all())
            assert bool((out.codecs[k].table >= -127).all())
    assert out2.codecs[2] is not tensors[2].codecs[2]
    assert not torch.equal(out2.codecs[2].table, initial[2])

    # Every exported tensor's K tables and the ONE global config agree exactly.
    from hi2.cptc import checkpoint_config, codecs_from_checkpoint_config

    seen = cptc_codecs_from_tensors({"a": out2, "b": out4})
    cfg = checkpoint_config(seen)
    back = codecs_from_checkpoint_config(copy.deepcopy(cfg))
    import base64

    for k in (2, 4):
        raw = base64.b64decode(cfg["tables"][str(k)]["tbl_b64"])
        assert raw == selected[k].to(torch.int8).numpy().tobytes()
        assert torch.equal(back[k].table, selected[k])
        exported = out2.codecs[2] if k == 2 else out4.codecs[4]
        assert torch.equal(back[k].table, exported.table)


def test_cptc_shared_table_lut_preserves_tiny_fp32_gradients():
    """The LUT forward is the exact runtime fp16 value, but the lookup carries an
    fp32 surrogate; a tiny per-weight gradient must not underflow through half."""
    from hi2.qad import CptcSharedTables

    codec = get_codec(16, 2)
    book = CptcSharedTables({2: codec})
    lut = book.lut(2, "cpu")
    assert lut.dtype == torch.float32
    # Hard-forward values are exactly the fp16 runtime table.
    want_lut = cptc.index_table_value(
        torch.arange(127), book.selected_int8(2), codec.table_scale
    ).float()
    assert torch.equal(lut, want_lut)
    assert int(codec.table[10]) == -74  # not at the clamp boundary

    idx = torch.full((8192,), 10, dtype=torch.int64)
    (lut[idx].float().sum() * 1e-10).backward()
    grad = book.table_u[2].grad
    assert grad is not None
    # Compare the same FP32 accumulation, not a real-arithmetic sum: repeated
    # float32 additions legitimately round before the table-scale derivative.
    reference = codec.table.float().clone().requires_grad_()
    ((reference * float(codec.table_scale))[idx].sum() * 1e-10).backward()
    assert reference.grad is not None
    assert torch.equal(grad, reference.grad)
    assert grad[10] > 0


def test_cptc_fixed_and_shared_tables_keep_hard_forward_and_scale_grads():
    """The default fixed-table path and the live-table path must agree exactly on
    forward values and on the block-scale gradients."""
    from hi2.qad import CptcFrozenCodeLinear, CptcSharedTables

    codecs, tensors = _two_width_tensors()
    t = tensors[2]
    fixed = CptcFrozenCodeLinear(t)
    book = CptcSharedTables({2: codecs[2]})
    shared = CptcFrozenCodeLinear(t, tables=book)
    assert torch.equal(fixed.stored(False), shared.stored(False))

    delta = torch.linspace(-0.2, 0.2, fixed.delta.numel()).reshape_as(fixed.delta)
    with torch.no_grad():
        fixed.delta.copy_(delta)
        shared.delta.copy_(delta)
    assert torch.equal(fixed.stored(False), shared.stored(False))

    fixed_loss = fixed.stored(True).square().sum()
    shared_loss = shared.stored(True).square().sum()
    fixed_loss.backward()
    shared_loss.backward()
    assert torch.equal(fixed.delta.grad, shared.delta.grad)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs for the sharded table"
)
def test_cptc_shared_tables_train_across_two_devices_with_one_master():
    """The global table lives on device 0; a layer on device 1 reads a differentiable
    copy, so both shard gradients land on the one master and one AdamW update."""
    from hi2.finalkl import CptcLinear
    from hi2.qad import CptcIndexCache, CptcSharedTables

    dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
    codecs, tensors = _two_width_tensors()
    book = CptcSharedTables(codecs).to(dev0)
    ref = CptcSharedTables(codecs).to(dev0)
    cache = CptcIndexCache(1 << 20)
    ref_cache = CptcIndexCache(1 << 20)
    m2 = CptcLinear(tensors[2], tables=book, index_cache=cache).to(dev0)
    m4 = CptcLinear(tensors[4], tables=book, index_cache=cache).to(dev1)
    r2 = CptcLinear(tensors[2], tables=ref, index_cache=ref_cache).to(dev0)
    r4 = CptcLinear(tensors[4], tables=ref, index_cache=ref_cache).to(dev0)
    d2 = torch.linspace(-0.2, 0.2, m2.delta.numel()).reshape_as(m2.delta)
    d4 = torch.linspace(-0.2, 0.2, m4.delta.numel()).reshape_as(m4.delta)
    with torch.no_grad():
        m2.delta.copy_(d2)
        m4.delta.copy_(d4.to(dev1))
        r2.delta.copy_(d2)
        r4.delta.copy_(d4)

    loss = m2.weight().square().sum() + m4.weight().to(dev0).square().sum()
    ref_loss = r2.weight().square().sum() + r4.weight().square().sum()
    book.backward(loss)
    ref.backward(ref_loss)

    torch.testing.assert_close(m2.weight(), r2.weight(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        m4.weight().to(dev0), r4.weight(), rtol=1e-4, atol=1e-5
    )
    assert cache.used_bytes > 0 and cache.used_bytes == ref_cache.used_bytes
    assert cache.entries == ref_cache.entries
    assert sum(
        int(s.numel()) for m in (m2, m4) for s in m._index_slots if s is not None
    ) == cache.used_bytes
    assert m2._index_slots[0].device == dev0
    assert m4._index_slots[0].device == dev1
    assert m2.delta.grad is not None and m2.delta.grad.device == dev0
    assert m4.delta.grad is not None and m4.delta.grad.device == dev1
    torch.testing.assert_close(m2.delta.grad, r2.delta.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        m4.delta.grad.to(dev0), r4.delta.grad, rtol=1e-4, atol=1e-5
    )
    for p, q in zip(book.parameters(), ref.parameters()):
        assert p.device == dev0
        assert p.grad is not None and p.grad.device == dev0
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5)

    opt = torch.optim.AdamW(book.parameters(), lr=0.05)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=0.05)
    opt.step()
    ref_opt.step()
    for p, q in zip(book.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5)
    # Use one exact table state for the live-read check; the update was compared above.
    with torch.no_grad():
        for p, q in zip(book.parameters(), ref.parameters()):
            q.copy_(p)

    # Negate the whole live table: one arbitrary entry may not occur in this fixture.
    # Cached indices stay immutable, but every nonzero selected value must change.
    before2, before4 = m2.weight().detach().clone(), m4.weight().detach().clone()
    assert torch.count_nonzero(before2) and torch.count_nonzero(before4)
    with torch.no_grad():
        for k in book.widths():
            book.table_u[k].neg_()
            ref.table_u[k].neg_()
    torch.testing.assert_close(
        m2.weight(), r2.weight(), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        m4.weight().to(dev0), r4.weight(), rtol=1e-4, atol=1e-5
    )
    assert not torch.equal(before2, m2.weight().detach())
    assert not torch.equal(before4, m4.weight().detach())
    assert cache.used_bytes == ref_cache.used_bytes


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs a CUDA device")
def test_cptc_index_cache_slot_move_replaces_without_double_counting():
    """A device move replaces the one cached index slot; the budget does not grow and
    the retained slot bytes remain exactly the accounted bytes."""
    from hi2.qad import CptcFrozenCodeLinear, CptcIndexCache

    _, tensors = _two_width_tensors()
    t = tensors[2]
    cache = CptcIndexCache(1 << 20)
    m = CptcFrozenCodeLinear(t, index_cache=cache)
    m.stored(False)
    used = cache.used_bytes
    assert used == t.shape[0] * t.shape[1]
    assert cache.entries == 1
    assert m._index_slots[0] is not None and m._index_slots[0].device.type == "cpu"

    dev = torch.device("cuda:0")
    m.to(dev)
    m.stored(False)
    assert cache.used_bytes == used and cache.entries == 1
    assert m._index_slots[0].device == dev
    assert m._index_slots[0].dtype == torch.uint8
    assert sum(s.numel() for s in m._index_slots if s is not None) == used


def test_cptc_final_table_projection_pins_last_entry():
    from hi2.qad import CptcSharedTables

    codecs, _ = _two_width_tensors()
    book = CptcSharedTables(codecs)
    with torch.no_grad():
        for p in book.parameters():
            p.fill_(1000.0)
        book.project()
    for p in book.parameters():
        assert float(p.max()) <= 127.0 and float(p.min()) >= -127.0
        assert float(p[-1]) == 0.0
    for k in (2, 4):
        table = book.selected_int8(k)
        assert int(table[-1]) == 0 and table.dtype == torch.int8


def test_cptc_index_cache_budget_is_bounded_and_matches_both_paths():
    from hi2.finalkl import CptcLinear
    from hi2.qad import (
        CptcFrozenCodeLinear,
        CptcIndexCache,
        CptcSharedTables,
    )

    codecs, tensors = _two_width_tensors()
    t = tensors[2]
    rows, k = t.shape
    per_group = rows * k  # one uint8 index per stored weight

    exact = CptcIndexCache(per_group)
    cached = CptcFrozenCodeLinear(t, index_cache=exact)
    plain = CptcFrozenCodeLinear(t)
    with torch.no_grad():
        cached.delta.copy_(torch.linspace(-0.2, 0.3, cached.delta.numel()))
        plain.delta.copy_(cached.delta)
    assert torch.equal(cached.stored(True), plain.stored(True))
    cached.stored(True).square().sum().backward()
    plain.stored(True).square().sum().backward()
    assert torch.equal(cached.delta.grad, plain.delta.grad)
    assert exact.used_bytes == per_group and exact.entries == 1
    assert exact.used_bytes <= exact.budget_bytes
    assert cached._index_slots[0].dtype == torch.uint8

    too_small = CptcIndexCache(per_group - 1)
    refused = CptcFrozenCodeLinear(t, index_cache=too_small)
    with torch.no_grad():
        refused.delta.copy_(cached.delta)
    refused.stored(False)
    assert too_small.used_bytes == 0 and too_small.entries == 0
    for m in (cached, plain, refused):
        assert torch.equal(m.stored(False), cached.stored(False))

    # Trainable tables: cached indices must not hide a changed table, and the
    # weights, table/scale gradients and one optimizer step must match exactly.
    book_a = CptcSharedTables(codecs)
    book_b = CptcSharedTables(codecs)
    cache = CptcIndexCache(8 * per_group)
    a = CptcLinear(t, tables=book_a, index_cache=cache)
    b = CptcLinear(t, tables=book_b)
    with torch.no_grad():
        a.delta.copy_(torch.linspace(-0.4, 0.4, a.delta.numel()))
        b.delta.copy_(a.delta)
        for kk in (2, 4):
            book_a.table_u[kk].copy_(torch.linspace(-20, 20, 64))
            book_b.table_u[kk].copy_(book_a.table_u[kk])
    assert torch.equal(a.weight(), b.weight())
    a.weight().square().sum().backward()
    b.weight().square().sum().backward()
    assert torch.equal(a.delta.grad, b.delta.grad)
    for pa, pb in zip(book_a.parameters(), book_b.parameters()):
        if pa.grad is None or pb.grad is None:
            assert pa.grad is None and pb.grad is None
        else:
            assert torch.equal(pa.grad, pb.grad)
    opt_a = torch.optim.AdamW([a.delta, *book_a.parameters()], lr=0.1)
    opt_b = torch.optim.AdamW([b.delta, *book_b.parameters()], lr=0.1)
    opt_a.step()
    opt_b.step()
    assert torch.equal(a.delta, b.delta)
    for pa, pb in zip(book_a.parameters(), book_b.parameters()):
        assert torch.equal(pa, pb)
    assert cache.used_bytes == per_group and cache.used_bytes <= cache.budget_bytes

    # Move the table after the indices were cached: the cached module must decode
    # with the new table exactly like the uncached one.
    with torch.no_grad():
        for kk in (2, 4):
            book_a.table_u[kk].add_(3.0)
            book_b.table_u[kk].add_(3.0)
    assert torch.equal(a.weight(), b.weight())


@pytest.mark.parametrize("magnitude", [1e-4, 1e-10])
def test_cptc_stored_table_gradient_matches_fp32_reference(magnitude):
    from hi2.qad import CptcFrozenCodeLinear, CptcSharedTables

    codecs, tensors = _two_width_tensors()
    book = CptcSharedTables(codecs)
    module = CptcFrozenCodeLinear(tensors[2], tables=book)
    upstream = torch.randn(
        module.shape, generator=torch.Generator().manual_seed(104)
    ) * magnitude
    (module.stored(True) * upstream).sum().backward()

    master = book.table_u[2].detach().clone().requires_grad_()
    low = torch.cat([master.clamp(-127, 127)[:-1], master.new_zeros(1)])
    lut = torch.cat([low, -low[:-1].flip(0)]) * codecs[2].table_scale.float()
    rows = module.rows[0]
    nbits, block = module.groups[0]
    idx = module._build_index(rows, nbits, block, codecs[2]).long()
    nblk = module.shape[1] // block
    pidx = module.param_off[rows].long().unsqueeze(1) + torch.arange(nblk)
    scales = module.scales().detach()[pidx].unsqueeze(-1)
    ref = (scales * lut[idx]).reshape(module.shape)
    (ref * upstream).sum().backward()
    torch.testing.assert_close(book.table_u[2].grad, master.grad, rtol=2e-6, atol=1e-14)


def test_cptc_table_gather_saves_compact_indices_and_accumulates_chunks():
    from hi2.qad import _CptcTableGather

    indices = (torch.arange((2 << 20) // 8 + 19) % 127).to(torch.uint8)
    lut = torch.linspace(-1, 1, 127).requires_grad_()
    reference = lut.detach().clone().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tensor.numel() * tensor.element_size()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        values = _CptcTableGather.apply(lut, indices)
    assert saved == [(torch.uint8, indices.numel())]
    # Binary-exact contributions make differing parallel reduction orders irrelevant.
    upstream = ((torch.arange(indices.numel()) % 17) - 8).float() * (2.0 ** -20)
    values.backward(upstream)
    expected = reference[indices.long()]
    expected.backward(upstream)
    assert torch.equal(values, expected)
    assert torch.equal(lut.grad, reference.grad)


def test_cptc_pipeline_metadata_uses_changed_exported_tables_and_rejects_disagreement():
    from hi2.calibrate import Config
    from hi2.cptc import codecs_from_checkpoint_config
    from hi2.pipeline import _cptc_checkpoint_metadata
    from hi2.qad import CptcFrozenCodeLinear, CptcSharedTables

    codecs, tensors = _two_width_tensors()
    book = CptcSharedTables(codecs)
    module = CptcFrozenCodeLinear(tensors[2], tables=book)
    with torch.no_grad():
        book.table_u[2][10].add_(4)
    exported = module.export()
    assert not torch.equal(exported.codecs[2].table, codecs[2].table)
    cfg = Config(cptc=True, cptc_depth=16, cptc_table_rounds=0)
    parsed = codecs_from_checkpoint_config(
        _cptc_checkpoint_metadata(cfg, {"selected": exported})
    )
    assert torch.equal(parsed[2].table, exported.codecs[2].table)
    with pytest.raises(ValueError, match="disagree"):
        _cptc_checkpoint_metadata(
            cfg, {"selected": exported, "stale": tensors[2]}
        )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs a CUDA device")
def test_cptc_table_gather_gpu_gradients_repeat_exactly_with_rebuilt_indices():
    from hi2.qad import _CptcTableGather

    gen = torch.Generator().manual_seed(118)
    count = (2 << 20) // 8 + 19
    indices = torch.randint(0, 127, (count,), generator=gen, dtype=torch.uint8).cuda()
    upstream = (torch.randn(count, generator=gen) * 1e-4).cuda()
    initial = torch.linspace(-1, 1, 127).cuda()
    expected = initial[indices.long()]
    gradients = []
    updates = []
    for rebuilt in (False, True, False, True):
        lut = initial.clone().requires_grad_()
        optimizer = torch.optim.AdamW([lut], lr=0.02)
        values = _CptcTableGather.apply(lut, indices.clone() if rebuilt else indices)
        assert torch.equal(values, expected)
        values.backward(upstream)
        gradients.append(lut.grad.detach().clone())
        optimizer.step()
        updates.append(lut.detach().clone())
    for gradient, update in zip(gradients[1:], updates[1:]):
        assert torch.equal(gradient, gradients[0])
        assert torch.equal(update, updates[0])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_cptc_shared_table_gradients_and_updates_match_across_cache_repeats():
    """Multiple groups on both shards must not reorder the global FP32 gradient."""
    from hi2.cptc import assemble_cptc
    from hi2.qad import CptcFrozenCodeLinear, CptcIndexCache, CptcSharedTables

    gen = torch.Generator().manual_seed(20261120)
    rows, width = 64, 4096
    widths = torch.tensor(([2] * 16 + [4] * 16) * 2, dtype=torch.uint8)
    blocks = torch.tensor([32] * 32 + [64] * 32, dtype=torch.uint8)
    codecs = {bits: get_codec(16, bits) for bits in (2, 4)}
    pieces = []
    for bits, block in ((2, 32), (4, 32), (2, 64), (4, 64)):
        rr = ((widths == bits) & (blocks == block)).nonzero().flatten()
        codes = torch.randint(0, 2**bits, (len(rr), width), generator=gen)
        scales = torch.linspace(0.1, 1.3, width // block).half().repeat(len(rr), 1)
        pieces.append((rr, codes, scales))
    tensor = assemble_cptc(
        pieces, widths, blocks,
        torch.tensor([0, 1] * (width // 32), dtype=torch.uint8),
        width, codecs, col_scale=torch.linspace(0.5, 1.5, width).half(),
    )
    devices = ("cuda:0", "cuda:1")
    inputs = [torch.randn(16, width, generator=gen).to(d) for d in devices]
    targets = [torch.randn(16, rows, generator=gen).to(d) for d in devices]
    reference = None
    for budget in (0, 0, 1 << 20, 1 << 20):
        book = CptcSharedTables(codecs).to(devices[0])
        cache = CptcIndexCache(budget)
        modules = [
            CptcFrozenCodeLinear(tensor, tables=book, index_cache=cache).to(d)
            for d in devices
        ]
        weights = [m.weight() for m in modules]
        outputs = [torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)]
        losses = [(y - target).square().mean() for y, target in zip(outputs, targets)]
        loss = losses[0] + losses[1].to(devices[0])
        assert torch.isfinite(loss)
        book.backward(loss)
        gradients = [p.grad.detach().cpu().clone() for p in book.parameters()]
        optimizer = torch.optim.AdamW(book.parameters(), lr=0.02)
        optimizer.step()
        record = (
            [w.detach().cpu().clone() for w in weights],
            [y.detach().cpu().clone() for y in outputs],
            gradients,
            [p.detach().cpu().clone() for p in book.parameters()],
        )
        if reference is None:
            reference = record
        else:
            for expected, actual in zip(reference, record):
                for a, b in zip(expected, actual):
                    assert torch.equal(a, b)
