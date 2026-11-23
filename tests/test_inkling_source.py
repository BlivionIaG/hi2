# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The Inkling-2 NVFP4 / MXFP8 source reader (``hi2.inkling_source``), CPU only.

The decoders are compared bit for bit with an independent numpy implementation of the
two formats (formulas, no lookup tables) and, when the vLLM fork's reference tools are
at hand, with ``tiny_reference.py`` / ``ref_checkpoint.py`` themselves (a tiny checkpoint
made by ``make_tiny_checkpoint.py``; the real checkpoint's layers 0 and 20 when
``INKLING2_CKPT`` names it). Set ``INKLING2_TOOLS`` to the fork's
``tools/rdna/inkling2`` directory; the sibling worktree is found by itself."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from hi2.inkling_source import (  # noqa: E402
    FP4_VALUES,
    InklingSource,
    dequant_mxfp8,
    dequant_nvfp4,
    e4m3_to_float32,
    expert_name,
    name_map,
    tensor_role,
    ue8m0_to_float32,
)
from hi2.moe import (  # noqa: E402
    ExpertMLP,
    UnfusedExperts,
    expert_parts,
    group_key,
    sibling_leader,
)

E, H, I_, VOCAB, UNPADDED = (
    4,
    64,
    32,
    24,
    20,
)  # tiny geometry: K is 64 / 32 (blocks 16, 32)
DEN = 2688.0


# ---------------------------------------------------------------- independent decoders


def _np_fp4(code: np.ndarray) -> np.ndarray:
    """e2m1 from its bit fields: sign 3, exponent 2-1, mantissa 0 (code 8 is -0.0)."""
    s, e, m = (code >> 3) & 1, (code >> 1) & 3, code & 1
    mag = np.where(e == 0, m * 0.5, (1 + m / 2) * 2.0 ** (e.astype(np.float64) - 1))
    return np.where(s == 1, -mag, mag).astype(np.float32)


def _np_e4m3(code: np.ndarray) -> np.ndarray:
    """e4m3fn from its bit fields (bias 7, no infinity; NaN is exponent 15 mantissa 7)."""
    s, e, m = code >> 7, (code >> 3) & 15, code & 7
    e = e.astype(np.float64)
    mag = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * 2.0 ** (e - 7))
    mag = np.where((e == 15) & (m == 7), np.nan, mag)
    return np.where(s == 1, -mag, mag).astype(np.float32)


def _np_nvfp4(packed, scale, amax, den) -> np.ndarray:
    e, n, kh = packed.shape
    vals = np.stack((_np_fp4(packed & 0xF), _np_fp4(packed >> 4)), -1).reshape(e, n, -1)
    glob = (amax.astype(np.float64) / np.float64(den)).astype(np.float32)
    block = _np_e4m3(scale) * glob[:, None, None]
    return vals * np.repeat(block, 16, axis=-1)


def _np_mxfp8(weight, scale) -> np.ndarray:
    block = (2.0 ** (scale.astype(np.float64) - 127)).astype(np.float32)
    return _np_e4m3(weight) * np.repeat(block, 32, axis=-1)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit patterns (so -0.0 and NaN payloads count), as integers."""
    return t.contiguous().view({4: torch.int32, 2: torch.int16}[t.element_size()])


# ---------------------------------------------------------------- independent encoders


def _quantize_nvfp4(w: torch.Tensor):
    """fp32 ``[E, N, K]`` -> (U8 ``[E, N, K/2]``, E4M3 ``[E, N, K/16]``, amax ``[E]``)."""
    e, n, k = w.shape
    amax = w.abs().amax(dim=(1, 2))
    glob = (amax / DEN)[:, None, None]
    blocks = w.reshape(e, n, k // 16, 16)
    q8 = (blocks.abs().amax(-1) / 6.0 / glob).clamp(2.0**-9, 448.0)
    scale_q = q8.to(torch.float8_e4m3fn)
    q = blocks / (scale_q.float() * glob)[..., None]
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    idx = (q.abs()[..., None] - mags).abs().argmin(-1)
    nib = (idx | (((q < 0) & (idx != 0)).long() << 3)).reshape(e, n, k)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scale_q.contiguous(), amax.contiguous()


def _quantize_mxfp8(w: torch.Tensor):
    """fp32 ``[E, N, K]`` -> (E4M3 ``[E, N, K]``, UE8M0 U8 ``[E, N, K/32]``)."""
    e, n, k = w.shape
    blocks = w.reshape(e, n, k // 32, 32)
    bmax = blocks.abs().amax(-1).clamp(min=2.0**-100)
    exp = torch.ceil(torch.log2(bmax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(exp)[..., None]).clamp(-448.0, 448.0)
    return q.to(torch.float8_e4m3fn).reshape(e, n, k).contiguous(), (exp + 127).to(
        torch.uint8
    )


def _interleave(w: torch.Tensor) -> torch.Tensor:
    """``[E, 2I, ...]`` rows ``[gate; up]`` -> ``[g0, u0, g1, u1, ...]``."""
    half = w.shape[1] // 2
    out = torch.empty_like(w)
    out[:, 0::2], out[:, 1::2] = w[:, :half], w[:, half:]
    return out


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).sum().sqrt() / (b**2).sum().sqrt())


# ---------------------------------------------------------------- a tiny serving checkpoint


def _write_tiny(path: Path, fmts, seed=0, index=True):
    """A checkpoint directory in the serving layout with the given expert formats per
    layer; returns the original fp32 ``(gate, up, down)`` of every layer."""
    g = torch.Generator().manual_seed(seed)
    t: dict[str, torch.Tensor] = {}
    truth = []
    for layer, fmt in enumerate(fmts):
        gate = torch.randn(E, I_, H, generator=g) * (0.02 + 0.01 * layer)
        up = torch.randn(E, I_, H, generator=g) * 0.03
        down = torch.randn(E, H, I_, generator=g) * 0.025
        truth.append((gate, up, down))
        w13 = _interleave(torch.cat([gate, up], dim=1))
        base = f"model.llm.layers.{layer}.mlp.experts."
        for nm, w in (("w13", w13), ("w2", down)):
            if fmt == "nvfp4":
                p, s, a = _quantize_nvfp4(w)
                t[f"{base}{nm}_weight"], t[f"{base}{nm}_weight_scale"] = p, s
                t[f"{base}{nm}_weight_amax"] = a
                t[f"{base}{nm}_weight_outer_scale_denominator"] = torch.tensor([DEN])
            else:
                q, s = _quantize_mxfp8(w)
                t[f"{base}{nm}_weight"], t[f"{base}{nm}_weight_scale"] = q, s
        t[f"model.llm.layers.{layer}.attn.wq_du.weight"] = torch.randn(
            48, H, generator=g
        ).bfloat16()
        t[f"model.llm.layers.{layer}.mlp.gate.weight"] = torch.randn(
            E, H, generator=g
        ).bfloat16()
    t["model.llm.embed.weight"] = torch.randn(VOCAB, H, generator=g).bfloat16()
    t["model.llm.unembed.weight"] = torch.randn(VOCAB, H, generator=g).bfloat16()
    path.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors" if index else "model.safetensors"
    save_file(t, str(path / shard), metadata={"format": "pt"})
    if index:
        wm = dict.fromkeys(sorted(t), shard)
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": wm})
        )
    n = len(fmts)
    (path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": n,
                    "n_routed_experts": E,
                    "hidden_size": H,
                    "intermediate_size": I_,
                    "unpadded_vocab_size": UNPADDED,
                    "local_layer_ids": [i for i in range(n) if i % 2 == 0],
                }
            }
        )
    )
    return truth


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("inkling_tiny")
    fmts = ["nvfp4", "nvfp4", "mxfp8", "mxfp8"]
    return path, fmts, _write_tiny(path, fmts)


# ---------------------------------------------------------------- the code tables


def test_fp4_table_from_the_bit_fields():
    codes = np.arange(16, dtype=np.uint8)
    assert np.array_equal(np.array(FP4_VALUES, dtype=np.float32), _np_fp4(codes))
    assert str(FP4_VALUES[8]) == "-0.0"


def test_e4m3_lookup_matches_the_fp8_cast_and_the_formula_for_every_code():
    codes = torch.arange(256, dtype=torch.uint8)
    got = e4m3_to_float32(codes)
    cast = codes.view(torch.float8_e4m3fn).float()
    nan = torch.isnan(cast)
    assert nan.sum() == 2 and torch.equal(torch.isnan(got), nan)
    assert torch.equal(_bits(got[~nan]), _bits(cast[~nan]))
    assert np.array_equal(got.numpy(), _np_e4m3(codes.numpy()), equal_nan=True)
    # float8 tensors work as well as their bytes
    assert torch.equal(
        _bits(e4m3_to_float32(codes.view(torch.float8_e4m3fn))[~nan]), _bits(cast[~nan])
    )


def test_ue8m0_scale_is_exp2_for_every_byte():
    u = torch.arange(256, dtype=torch.uint8)
    got, ref = ue8m0_to_float32(u), torch.exp2(u.float() - 127.0)
    assert torch.equal(_bits(got), _bits(ref))  # includes 2^-127 (subnormal) and inf
    assert got[0].item() == 2.0**-127 and got[255].item() == float("inf")


# ---------------------------------------------------------------- the decoders


def _random_nvfp4(seed, e=3, n=8, k=64):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (e, n, k // 2), dtype=np.uint8)
    scale = rng.integers(0, 256, (e, n, k // 16), dtype=np.uint8)
    scale[(scale & 0x7F) == 0x7F] = 0x38  # no NaN codes
    amax = rng.uniform(0.05, 6.0, e).astype(np.float32)
    return packed, scale, amax


def test_nvfp4_decode_is_bit_identical_to_the_numpy_formula():
    packed, scale, amax = _random_nvfp4(0)
    got = dequant_nvfp4(
        torch.from_numpy(packed),
        torch.from_numpy(scale).view(torch.float8_e4m3fn),
        torch.from_numpy(amax),
        torch.tensor([DEN]),
    )
    ref = _np_nvfp4(packed, scale, amax, DEN)
    assert got.dtype == torch.float32 and tuple(got.shape) == ref.shape
    assert torch.equal(_bits(got), _bits(torch.from_numpy(ref)))


def test_mxfp8_decode_is_bit_identical_to_the_numpy_formula():
    rng = np.random.default_rng(1)
    w = rng.integers(0, 256, (3, 8, 64), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x30
    s = rng.integers(90, 140, (3, 8, 2), dtype=np.uint8)
    got = dequant_mxfp8(
        torch.from_numpy(w).view(torch.float8_e4m3fn), torch.from_numpy(s)
    )
    assert torch.equal(_bits(got), _bits(torch.from_numpy(_np_mxfp8(w, s))))


def test_decoders_round_trip_the_quantised_weights():
    """A wrong nibble order, scale layout or row order would land at ~1.4, not 0.2."""
    w = torch.randn(3, 16, 64, generator=torch.Generator().manual_seed(2)) * 0.03
    p, s, a = _quantize_nvfp4(w)
    assert _rel_rms(dequant_nvfp4(p, s, a, torch.tensor([DEN])), w) < 0.2
    q, u = _quantize_mxfp8(w)
    assert _rel_rms(dequant_mxfp8(q, u), w) < 0.06


def test_mxfp8_rejects_the_nan_scale_and_bad_shapes():
    w = torch.zeros(1, 2, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="0xFF"):
        dequant_mxfp8(
            w.view(torch.float8_e4m3fn), torch.full((1, 2, 1), 255, dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="MXFP8 shapes"):
        dequant_mxfp8(
            w.view(torch.float8_e4m3fn), torch.zeros(1, 2, 2, dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="NVFP4 shapes"):
        dequant_nvfp4(
            torch.zeros(1, 2, 16, dtype=torch.uint8),
            torch.zeros(1, 2, 3, dtype=torch.uint8),
            torch.ones(1),
            torch.tensor([DEN]),
        )


# ---------------------------------------------------------------- the reader


def test_layer_formats_and_geometry(tiny):
    path, fmts, _ = tiny
    with InklingSource(path) as src:
        assert src.formats() == fmts
        assert (src.num_layers, src.n_experts, src.hidden, src.inter) == (4, E, H, I_)
        assert src.full_layers == {1, 3}
        assert src.sample_experts(2) == [0, E - 1]
        assert src.sample_experts(E) == list(range(E))


@pytest.mark.parametrize("index", [True, False])
def test_gate_up_are_the_de_interleaved_rows_and_match_the_originals(tmp_path, index):
    truth = _write_tiny(tmp_path, ["nvfp4", "mxfp8"], seed=3, index=index)
    with InklingSource(tmp_path) as src:
        for layer, tol in ((0, 0.2), (1, 0.06)):
            gate, up = src.gate_up(layer)
            down = src.down(layer)
            g0, u0, d0 = truth[layer]
            assert tuple(gate.shape) == (E, I_, H) and tuple(down.shape) == (E, H, I_)
            assert gate.is_contiguous() and up.is_contiguous()
            assert _rel_rms(gate, g0) < tol and _rel_rms(up, u0) < tol
            assert _rel_rms(down, d0) < tol
            # gate/up are not swapped and not the plain halves of w13
            assert _rel_rms(gate, u0) > 0.5 and _rel_rms(up, g0) > 0.5


def test_gate_up_equal_the_even_and_odd_rows_of_the_reference_decode(tiny):
    path, fmts, _ = tiny
    with InklingSource(path) as src:
        for layer, fmt in enumerate(fmts):
            base = f"model.llm.layers.{layer}.mlp.experts.w13_weight"
            w, s = src.tensor(base), src.tensor(base + "_scale")
            if fmt == "nvfp4":
                ref = _np_nvfp4(
                    w.numpy(),
                    s.view(torch.uint8).numpy(),
                    src.tensor(base + "_amax").numpy(),
                    DEN,
                )
            else:
                ref = _np_mxfp8(w.view(torch.uint8).numpy(), s.numpy())
            gate, up = src.gate_up(layer)
            assert torch.equal(_bits(gate), _bits(torch.from_numpy(ref[:, 0::2])))
            assert torch.equal(_bits(up), _bits(torch.from_numpy(ref[:, 1::2])))


def test_chunks_and_expert_subsets_do_not_change_a_value(tiny):
    path, _, _ = tiny
    with InklingSource(path) as src:
        for layer in (0, 3):
            gate, up = src.gate_up(layer)
            down = src.down(layer)
            for chunk in (1, 3, E):
                got = list(src.iter_experts(layer, chunk=chunk))
                assert [e for e, *_ in got] == list(range(E))
                for e, g, u, d in got:
                    assert torch.equal(g, gate[e]) and torch.equal(u, up[e])
                    assert torch.equal(d, down[e])
            sub = [3, 0, 1]  # runs [3] and [0, 1], in the order asked
            g3, u3 = src.gate_up(layer, sub)
            assert torch.equal(g3, gate[sub]) and torch.equal(u3, up[sub])
            assert torch.equal(src.down(layer, slice(1, 3)), down[1:3])
            ge, ue, de = src.expert(layer, 2)
            assert torch.equal(ge, gate[2]) and torch.equal(de, down[2])
    with InklingSource(path) as src, pytest.raises(IndexError):
        src.gate_up(0, [E])


def test_raw_decode_and_stack_are_the_halves_a_profile_times(tiny):
    path, fmts, _ = tiny
    with InklingSource(path) as src:
        for layer, fmt in enumerate(fmts):
            raw = src.raw(layer, "w13", [1, 3])
            w, s, amax, den = raw
            assert w.dtype == torch.uint8 and s.dtype == torch.uint8
            assert (amax is None) == (fmt == "mxfp8") == (den is None)
            stacked = src.stack(layer, "w13", [1, 3])
            assert torch.equal(src.decode(raw), stacked)
            gate, up = src.gate_up(layer, [1, 3])
            assert torch.equal(stacked[:, 0::2], gate)
            assert torch.equal(stacked[:, 1::2], up)
            assert torch.equal(src.decode(raw, slice(1, None, 2)), up)
            assert torch.equal(src.stack(layer, "w2", [1, 3]), src.down(layer, [1, 3]))
        with pytest.raises(ValueError):
            src.raw(0, "w3")
        with pytest.raises(ValueError):
            src.stack(0, "gate")


def test_fused_layer_is_what_unfuse_experts_takes(tiny):
    path, _, _ = tiny
    with InklingSource(path) as src:
        gate_up, down = src.fused_layer(1, chunk=3)
        assert tuple(gate_up.shape) == (E, 2 * I_, H) and gate_up.dtype == torch.float32
        gate, up = src.gate_up(1)
        assert torch.equal(gate_up[:, :I_], gate) and torch.equal(gate_up[:, I_:], up)
        assert torch.equal(down, src.down(1))
        bf = src.fused_layer(1, dtype=torch.bfloat16)
        assert torch.equal(bf[0], gate_up.bfloat16())

        class Fused(nn.Module):  # the transformers-5 layout moe.is_fused_experts checks
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
                self.down_proj = nn.Parameter(down, requires_grad=False)
                self.act_fn = nn.SiLU()

        # (the module list also holds the activation, as in the repo's own UnfusedExperts)
        mlps = [m for m in UnfusedExperts(Fused()) if isinstance(m, ExpertMLP)]
        assert len(mlps) == E
        for e, mlp in enumerate(mlps):
            assert torch.equal(mlp.gate_proj.weight, gate[e])
            assert torch.equal(mlp.up_proj.weight, up[e])
            assert torch.equal(mlp.down_proj.weight, down[e].contiguous())


def test_expert_names_are_the_names_moe_py_works_with(tiny):
    path, _, _ = tiny
    gate, up, down = (
        expert_name(5, 17, p) for p in ("gate_proj", "up_proj", "down_proj")
    )
    pre = "model.language_model.layers.5.mlp.experts"
    assert gate == f"{pre}.17.gate_proj.weight"
    assert expert_parts(gate) == (pre, 17, "gate_proj")
    assert group_key(down) == f"{pre}.*.down_proj"
    assert sibling_leader(up) == f"{pre}.17.gate_proj"
    with pytest.raises(ValueError):
        expert_name(0, 0, "w13")
    with InklingSource(path) as src:
        names = list(src.hf_expert_names(2, [1, 3]))
        assert names[:3] == [
            expert_name(2, 1, p) for p in ("gate_proj", "up_proj", "down_proj")
        ]
        assert len(names) == 6


def test_load_by_hugging_face_name(tiny):
    path, _, _ = tiny
    with InklingSource(path) as src:
        gate, up = src.gate_up(0)
        down = src.down(0)
        assert torch.equal(src.load(expert_name(0, 2, "gate_proj")), gate[2])
        assert torch.equal(src.load(expert_name(0, 2, "up_proj")), up[2])
        assert torch.equal(src.load(expert_name(0, 2, "down_proj")), down[2])
        w = src.load("model.language_model.layers.1.self_attn.q_proj.linear.weight")
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (48, H)
        head = src.load("lm_head.weight", dtype=torch.float32)
        assert tuple(head.shape) == (UNPADDED, H)  # the padded rows are dropped
        assert tuple(src.tensor("model.llm.unembed.weight").shape) == (VOCAB, H)
        with pytest.raises(KeyError):
            src.load("model.language_model.layers.0.nope.weight")
        with pytest.raises(KeyError):
            src.load("model.language_model.layers.0.mlp.experts.1.w1.weight")
        assert "model.llm.layers.0.attn.wq_du.weight" in src.hf_names()


def test_name_map_and_roles():
    m = name_map(24, {5, 11, 17, 23})
    assert m["model.llm.layers.7.attn.wq_du.weight"] == (
        "model.language_model.layers.7.self_attn.q_proj.linear.weight"
    )
    assert m["model.llm.layers.5.attn.k_sconv.static_w"].endswith(
        "k_proj.sconv.static_weight"
    )
    assert m["model.llm.layers.6.attn.k_sconv.weight"].endswith(
        "k_proj.sconv.conv1d.weight"
    )
    assert "model.llm.layers.5.attn.k_sconv.weight" not in m
    assert m["model.llm.attn_res.attn_queries.22"].startswith(
        "model.language_model.layers.23.attention_input"
    )
    assert "model.llm.attn_res.attn_queries.23" not in m
    assert m["model.llm.unembed.weight"] == "lm_head.weight"
    assert not any(".mlp.experts." in k for k in m)
    for name, role in {
        "model.llm.layers.3.mlp.experts.w13_weight_scale": "expert",
        "model.llm.layers.3.attn.wq_du.weight": "attn_linear",
        "model.llm.layers.3.attn.gate_attn.weight": "attn_linear",
        "model.llm.layers.3.attn.attn_sink_proj.weight": "attn_sink",
        "model.llm.layers.3.attn.k_sconv.weight_proj": "kv_sconv",
        "model.llm.layers.3.controllers.k.down_weight": "controller",
        "model.llm.layers.3.mlp.gate.weight": "router",
        "model.llm.layers.3.mlp.gate.bias": "router_bias",
        "model.llm.layers.3.attn_norm.weight": "norm",
        "model.llm.attn_res.mlp_queries.3": "attn_res_query",
        "model.llm.middle_cycle_cross_pass_queries": "cross_pass_query",
        "model.llm.embed.weight": "embed",
        "model.llm.unembed.weight": "unembed",
        "model.visual.backbone.layers.0.mlp.c_fc.weight": "vision",
        "something.else": "other",
    }.items():
        assert tensor_role(name) == role, name


# ---------------------------------------------------------------- the fork's reference tools


def _tools_dir() -> Path | None:
    env = os.environ.get("INKLING2_TOOLS")
    cands = [Path(env).expanduser()] if env else []
    cands.append(
        Path(__file__).resolve().parents[2] / "vllm-inkling2-merge/tools/rdna/inkling2"
    )
    return next((c for c in cands if (c / "tiny_reference.py").exists()), None)


def _import(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tools():
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    pytest.importorskip("numpy")
    return (
        _import("hi2_test_tiny_reference", d / "tiny_reference.py"),
        _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py"),
    )


def _equal_bits(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(_bits(a.contiguous()), _bits(b.contiguous())), "bits differ"
    return float((a - b).abs().max())


@pytest.mark.parametrize("chunk", [3, 8])
def test_matches_the_reference_decoders_on_the_forks_tiny_checkpoint(
    tools, tmp_path, chunk
):
    ref, make = tools
    spec = make.TinySpec(num_hidden_layers=6)
    make.write_checkpoint(tmp_path, spec, seed=11)
    raw = ref.read_checkpoint(tmp_path)
    rc = ref.load_ref_checkpoint()
    with InklingSource(tmp_path) as src:
        assert src.formats() == ["nvfp4"] * 5 + ["mxfp8"]
        assert src.full_layers == set(spec.full_layer_ids)
        for layer in range(spec.num_hidden_layers):
            prefix = f"model.llm.layers.{layer}.mlp.experts"
            gate_up_ref, down_ref = ref.dequant_experts(raw, prefix)
            gate_up, down = src.fused_layer(layer, chunk=chunk)
            assert _equal_bits(gate_up, gate_up_ref) == 0.0
            assert _equal_bits(down, down_ref) == 0.0
            # the bf16 reference (ref_checkpoint.expert_pair) is this value rounded once
            bf_up, bf_down = rc.expert_pair(raw, layer, prefix, None)
            assert _equal_bits(gate_up.bfloat16(), bf_up) == 0.0
            assert _equal_bits(down.bfloat16(), bf_down) == 0.0
        # the name map is the reference's (the head is added by the reference loaders)
        n, full = spec.num_hidden_layers, set(spec.full_layer_ids)
        ref_map = ref.adapt_name_map(rc.build_name_map(n), n, full)
        ref_map["model.llm.unembed.weight"] = "lm_head.weight"
        assert name_map(n, full) == ref_map


def test_matches_the_reference_decoders_on_the_real_checkpoint(tools):
    ckpt = os.environ.get("INKLING2_CKPT")
    if not ckpt:
        pytest.skip("INKLING2_CKPT (the real Inkling-2-Mini directory) is not set")
    ref, _ = tools
    rc = ref.load_ref_checkpoint()
    from safetensors import safe_open

    sel = [0, 17, 64, 127]
    report = {}
    shard = Path(ckpt) / "model-00001-of-00001.safetensors"

    def rows(f, name):  # only the experts asked for are read
        sl = f.get_slice(name)
        return torch.cat([sl[e : e + 1] for e in sel])

    with InklingSource(ckpt) as src, safe_open(shard, "pt") as f:
        assert src.formats() == ["nvfp4"] * 20 + ["mxfp8"] * 4
        for layer in (0, 20):
            base = f"model.llm.layers.{layer}.mlp.experts"
            gate, up = src.gate_up(layer, sel)
            down = src.down(layer, sel)
            for nm in ("w13", "w2"):
                w, s = (
                    rows(f, f"{base}.{nm}_weight"),
                    rows(f, f"{base}.{nm}_weight_scale"),
                )
                if layer < 20:
                    ref_full = ref.dequant_nvfp4(
                        w,
                        s,
                        rows(f, f"{base}.{nm}_weight_amax"),
                        f.get_tensor(f"{base}.{nm}_weight_outer_scale_denominator"),
                    )
                else:
                    ref_full = ref.dequant_mxfp8(w, s)
                if nm == "w13":
                    ref_gate_up = rc.deinterleave_gate_up(ref_full)
                    i = ref_gate_up.shape[1] // 2
                    report[(layer, "gate")] = _equal_bits(
                        gate, ref_gate_up[:, :i].contiguous()
                    )
                    report[(layer, "up")] = _equal_bits(
                        up, ref_gate_up[:, i:].contiguous()
                    )
                else:
                    report[(layer, "down")] = _equal_bits(down, ref_full)
    print("max abs diff vs tiny_reference:", report)
    assert set(report.values()) == {0.0}
