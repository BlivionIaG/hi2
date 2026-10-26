# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""MoE support (CPU): routed-token statistics, the token floor, shared gate/up (and
q/k/v) transforms, checkpoint name layout and the export round trip."""

import json

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

from hi2.calibrate import Config  # noqa: E402
from hi2.checkpoint import table_hessian_path, table_stats_path  # noqa: E402
from hi2.executor import Executor  # noqa: E402
from hi2.layerwise import calibrate_layerwise  # noqa: E402
from hi2.moe import (  # noqa: E402
    UnfusedExperts,
    expert_parts,
    expert_weights,
    unfuse_experts,
)
from hi2.stats import ColumnStats  # noqa: E402

H, INTER, E, TOPK, V = 128, 64, 4, 2, 64
RARE = E - 1  # only token V-1 is routed to it
RES = 1e-3  # residual scale: keeps the routing channels of the embedding intact


class FusedExperts(nn.Module):
    """transformers-5 style experts: 3-D parameters, ``(x, top_k_index, top_k_weights)``."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * INTER, H) / H**0.5)
        self.down_proj = nn.Parameter(torch.randn(E, H, INTER) / INTER**0.5)
        self.act_fn = nn.SiLU()

    def forward(self, x, idx, wts):
        out = torch.zeros_like(x)
        for e in range(E):
            tok, pos = torch.where(idx == e)
            g, u = nn.functional.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = nn.functional.linear(self.act_fn(g) * u, self.down_proj[e])
            out.index_add_(0, tok, y * wts[tok, pos, None])
        return out


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = FusedExperts()

    def forward(self, h):
        flat = h.reshape(-1, H)
        w, idx = torch.topk(torch.softmax(flat[:, :E], -1), TOPK, dim=-1)
        return self.experts(flat, idx, w / w.sum(-1, keepdim=True)).reshape(h.shape)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.mlp = Mlp()

    def attn_out(self, x):
        return x + RES * self.o_proj(
            self.q_proj(x) * torch.sigmoid(self.k_proj(x)) + self.v_proj(x)
        )

    def forward(self, x, **_):
        h = self.attn_out(x)
        return h + RES * self.mlp(h)


class Inner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embed = nn.Embedding(V, H)
        self.layers = nn.ModuleList(Block() for _ in range(layers))


class Toy(nn.Module):
    """Routing is set by the embedding: channel e of token t is 10 for its two experts
    (t % 3 and (t + 1) % 3; token V-1 goes to RARE and 0), 0 otherwise."""

    def __init__(self, layers=2):
        super().__init__()
        self.model = Inner(layers)
        with torch.no_grad():
            emb = self.model.embed.weight
            emb[:, :E] = 0
            for t in range(V):
                pair = (RARE, 0) if t == V - 1 else (t % 3, (t + 1) % 3)
                emb[t, list(pair)] = 10.0

    def forward(self, ids, use_cache=False):
        h = self.model.embed(ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


def _ids():
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, V - 1, (8, 64), generator=g)
    ids[0, 5] = ids[3, 9] = V - 1  # two tokens for the rare expert
    return ids


def _calibrate(tmp_path, **cfg_kw):
    torch.manual_seed(0)
    model = Toy().eval()
    assert unfuse_experts(model) == 2
    cfg = Config(effort="fast", **cfg_kw)
    tables = calibrate_layerwise(
        model, _ids(), cfg, Executor("cpu"), tmp_path, None, _exclude()
    )
    return model, tables


def _exclude():
    import regex as re
    from hi2.checkpoint import DEFAULT_EXCLUDE

    return re.compile(DEFAULT_EXCLUDE)


def _routed(ids):
    """Routed token count per expert (layer-independent by construction)."""
    counts = torch.zeros(E, dtype=torch.long)
    for t in ids.flatten().tolist():
        pair = (RARE, 0) if t == V - 1 else (t % 3, (t + 1) % 3)
        for e in pair:
            counts[e] += 1
    return counts


def test_unfused_view_matches_fused_forward():
    torch.manual_seed(0)
    fused = FusedExperts()
    x = torch.randn(10, H)
    idx = torch.randint(0, E, (10, TOPK))
    w = torch.rand(10, TOPK)
    ref = fused(x, idx, w)
    un = UnfusedExperts(fused)
    assert torch.allclose(un(x, idx, w), ref, atol=1e-6)
    assert torch.equal(un[2].up_proj.weight, fused.gate_up_proj[2, INTER:])
    assert un[2].gate_proj.weight.data_ptr() == fused.gate_up_proj[2].data_ptr()


def test_expert_statistics_come_from_routed_tokens_only(tmp_path):
    model, tables = _calibrate(tmp_path, expert_min_tokens=0)
    ids = _ids()
    counts = _routed(ids)
    with torch.no_grad():
        x = model.model.layers[0].attn_out(model.model.embed(ids)).reshape(-1, H)
        idx = torch.topk(torch.softmax(x[:, :E], -1), TOPK, dim=-1).indices
    for e in range(E):
        name = f"model.layers.0.mlp.experts.{e}.gate_proj.weight"
        st = ColumnStats.from_state(torch.load(table_stats_path(tmp_path, name)))
        assert st.n == counts[e] == tables[name].n_tokens
        xe = x[(idx == e).any(-1)].double()
        assert xe.shape[0] == counts[e]
        ref = (xe**2).sum(0) / xe.shape[0]
        assert torch.allclose(st.diag().double(), ref, rtol=1e-4)
        # gate and up share one Hessian file, E[x x^T] over the routed tokens
        hg = table_hessian_path(tmp_path, name)
        hu = table_hessian_path(tmp_path, name.replace("gate_proj", "up_proj"))
        assert hu.is_symlink() and hu.resolve() == hg.resolve()
        sigma = torch.load(hg).double()
        assert torch.allclose(sigma, xe.T @ xe / xe.shape[0], rtol=1e-4, atol=1e-6)
    # down_proj saw the same tokens
    down = "model.layers.1.mlp.experts.1.down_proj.weight"
    assert tables[down].n_tokens == counts[1]


def test_rare_expert_borrows_layer_average(tmp_path):
    _, tables = _calibrate(tmp_path, expert_min_tokens=2.0)
    counts = _routed(_ids())
    need = 2 * H
    assert counts[RARE] < need <= counts[:RARE].min()
    info = json.loads((tmp_path / "expert_tokens.json").read_text())
    fb = set(info["fallback"])
    for li in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{li}.mlp.experts.{RARE}.{proj}.weight"
            assert name in fb
            assert info["counts"][name] == counts[RARE] == tables[name].n_tokens
            st = ColumnStats.from_state(torch.load(table_stats_path(tmp_path, name)))
            assert st.n == need
        assert f"model.layers.{li}.mlp.experts.0.gate_proj.weight" not in fb
        cov = info["coverage"][f"model.layers.{li}.mlp.experts.*.gate_proj"]
        assert cov["min"] == counts[RARE] and cov["max"] == counts.max()
        assert cov["fallback"] == [RARE]
    # topped-up statistics = own tokens + (need - n) x layer average
    raw_dir = tmp_path / "raw"
    _calibrate(raw_dir, expert_min_tokens=0)
    name = f"model.layers.0.mlp.experts.{RARE}.gate_proj.weight"
    own = ColumnStats.from_state(torch.load(table_stats_path(raw_dir, name)))
    tot = [
        ColumnStats.from_state(
            torch.load(table_stats_path(raw_dir, name.replace(f".{RARE}.", f".{e}.")))
        )
        for e in range(E)
    ]
    tot_s, tot_n = sum(t.S for t in tot), sum(t.n for t in tot)
    want = own.S + (need - own.n) * tot_s / tot_n
    got = ColumnStats.from_state(torch.load(table_stats_path(tmp_path, name)))
    assert torch.allclose(got.S, want, rtol=1e-6)


@pytest.mark.parametrize("rotation", ["on", "auto"])
def test_shared_inputs_share_smoothing_and_rotation(tmp_path, rotation):
    _, tables = _calibrate(tmp_path, rotation=rotation)
    for li in range(2):
        p = f"model.layers.{li}."
        groups = [[p + f"{x}_proj.weight" for x in "qkv"]]
        groups += [
            [p + f"mlp.experts.{e}.{x}_proj.weight" for x in ("gate", "up")]
            for e in range(E)
        ]
        for names in groups:
            first = tables[names[0]]
            assert first.col_scale is not None
            for n in names[1:]:
                assert torch.equal(tables[n].col_scale, first.col_scale), n
                assert torch.equal(tables[n].had, first.had), n


def test_expert_weights_split_one_dense_weight_by_share(tmp_path):
    _, tables = _calibrate(tmp_path)
    w = expert_weights(tables)
    counts = _routed(_ids()).double()
    names = [f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(E)]
    share = counts / counts.mean()
    for e, n in enumerate(names):
        assert w[n] == pytest.approx(max(share[e].item(), 0.05) / E)
    assert sum(w[n] for n in names) == pytest.approx(1.0, abs=0.06)
    assert "model.layers.0.q_proj.weight" not in w


# ------------------------------------------------- end to end on a tiny OLMoE


def _save_per_expert(model, d):
    """Write ``model`` as the OLMoE checkpoint stores it: one 2-D tensor per expert
    projection (transformers may hold them fused in 3-D)."""
    from safetensors.torch import save_file

    out = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):
            pre = k.removesuffix("gate_up_proj")
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                out[f"{pre}{e}.gate_proj.weight"] = g.clone()
                out[f"{pre}{e}.up_proj.weight"] = u.clone()
        elif k.endswith("experts.down_proj"):
            pre = k.removesuffix("down_proj")
            for e in range(v.shape[0]):
                out[f"{pre}{e}.down_proj.weight"] = v[e].clone()
        else:
            out[k] = v.clone().contiguous()
    d.mkdir(parents=True, exist_ok=True)
    save_file(out, str(d / "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(d)
    return out


@pytest.fixture(scope="module")
def olmoe(tmp_path_factory):
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "OlmoeForCausalLM"):
        pytest.skip("transformers without OLMoE")
    cfg = tr.OlmoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = tr.OlmoeForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" not in n and "embed" not in n:
                p.normal_(0, 0.05)
    d = tmp_path_factory.mktemp("olmoe")
    state = _save_per_expert(model, d)
    ids = torch.randint(0, 256, (8, 64), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(ids[:1]).logits
        again = tr.AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        assert torch.allclose(again(ids[:1]).logits, ref, atol=1e-5)
    return d, state, ids, ref


def test_olmoe_end_to_end_layout_sharing_and_export(olmoe, tmp_path):
    from hi2.checkpoint import export_dense, load_dequantized, weight_map
    from hi2.pipeline import RunConfig, run
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    d, state, ids, ref = olmoe
    dst = tmp_path / "q"
    summary = run(
        RunConfig(
            src=d,
            dst=dst,
            workdir=tmp_path / "w",
            calib_ids=ids,
            calib_samples=8,
            seq_len=64,
            target_bpw=3.5,
            cfg=Config(effort="fast", rotation="auto"),
            moe_layout="per-expert",
        ),
        log=lambda *_: None,
    )
    assert summary["experts"]["expert"]["tensors"] == 2 * 8 * 3
    assert summary["experts"]["coverage"]
    wmap = weight_map(dst)
    expert_src = [k for k in state if expert_parts(k) is not None]
    assert len(expert_src) == 2 * 8 * 3
    for k in expert_src:
        base = k.removesuffix(".weight")
        assert k not in wmap
        for f in ("codes", "code_off", "param_off", "blk_params", "row_consts"):
            assert f"{base}.hi2_{f}" in wmap
    for li in range(2):
        assert f"model.layers.{li}.mlp.gate.weight" in wmap  # router stays dense
        assert f"model.layers.{li}.post_attention_layernorm.weight" in wmap

    def tens(name):
        with safe_open(dst / wmap[name], "pt") as f:
            return f.get_tensor(name)

    for li in range(2):
        p = f"model.layers.{li}."
        groups = [[p + f"self_attn.{x}_proj" for x in "qkv"]]
        groups += [
            [p + f"mlp.experts.{e}.{x}_proj" for x in ("gate", "up")] for e in range(8)
        ]
        for names in groups:
            for fld in ("hi2_col_scale", "hi2_had"):
                a = tens(f"{names[0]}.{fld}")
                for n in names[1:]:
                    assert torch.equal(tens(f"{n}.{fld}"), a), (n, fld)

    deq = load_dequantized(dst)
    g = "model.layers.1.mlp.experts.3.gate_proj.weight"
    assert deq[g].shape == state[g].shape
    rel = ((deq[g] - state[g]) ** 2).sum() / (state[g] ** 2).sum()
    assert rel < 0.2
    out = tmp_path / "dense"
    export_dense(d, dst, out)
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    sd = m.state_dict()
    fused = [k for k in sd if k.endswith("experts.gate_up_proj")]
    if fused:  # transformers fuses the per-expert tensors again on load
        gu = sd["model.layers.1.mlp.experts.gate_up_proj"][3]
        assert torch.equal(gu[:64], deq[g].half().float())
    else:
        assert torch.equal(sd[g], deq[g].half().float())
    with torch.no_grad():
        lg = m(ids[:1]).logits
    assert torch.isfinite(lg).all()
    assert ((lg - ref) ** 2).mean() < (ref**2).mean()


def test_olmoe_stacked_layout_is_bit_identical(olmoe, tmp_path):
    """``--moe-layout stacked`` and ``hi2 restack`` give the same weights as the
    per-expert layout: same arrays, same dequantized state dict, same fp16 export."""
    import hashlib

    from hi2.checkpoint import (
        export_dense,
        load_dequantized,
        verify_checkpoint,
        weight_map,
    )
    from hi2.pipeline import RunConfig, run
    from hi2.stack import Hi2Checkpoint, restack
    from safetensors import safe_open

    d, state, ids, _ = olmoe

    def quantize(dst, layout):
        return run(
            RunConfig(
                src=d,
                dst=dst,
                workdir=tmp_path / "w",  # the second run reuses phase 1
                calib_ids=ids,
                calib_samples=8,
                seq_len=64,
                target_bpw=3.5,
                cfg=Config(effort="fast", rotation="auto"),
                moe_layout=layout,
            ),
            log=lambda *_: None,
        )

    pe, st = tmp_path / "pe", tmp_path / "st"
    quantize(pe, "per-expert")
    summary = quantize(st, "stacked")
    assert summary["moe_layout"] == "stacked"
    assert not (tmp_path / "w" / "per-expert" / "st").exists()  # staging removed
    wpe, wst = weight_map(pe), weight_map(st)
    stacks = [k for k in wst if k.endswith(".hi2_part_off")]
    assert len(stacks) == 2 * 2  # gate_up and down per layer
    assert len(wpe) - len(wst) == 2 * 8 * 3 * 9 - len(stacks) * 10
    assert not any(".experts.0." in k for k in wst)
    for li in range(2):
        pre = f"model.layers.{li}.mlp.experts"
        with safe_open(st / wst[f"{pre}.gate_up.hi2_codes"], "pt") as f:
            assert f.get_tensor(f"{pre}.gate_up.hi2_part_off").tolist() == list(
                range(0, 2 * 64 * 8 + 1, 64)
            )
            assert tuple(f.get_slice(f"{pre}.gate_up.hi2_had").get_shape()) == (
                8,
                128 // 16,
            )
            assert tuple(f.get_slice(f"{pre}.down.hi2_col_scale").get_shape()) == (
                8,
                64,
            )
    a, b = Hi2Checkpoint(pe), Hi2Checkpoint(st)
    assert set(a.logical()) == set(b.logical())
    for base in a.logical():
        ta, tb = a.get(base), b.get(base)
        for fld in ("codes", "code_off", "blk_params", "param_off", "row_consts",
                    "row_nbits", "row_block", "had", "col_scale"):  # fmt: skip
            x, y = getattr(ta, fld), getattr(tb, fld)
            assert x.dtype == y.dtype and torch.equal(x, y), (base, fld)
    da, db = load_dequantized(pe), load_dequantized(st)
    assert da.keys() == db.keys() and all(torch.equal(da[k], db[k]) for k in da)
    assert verify_checkpoint(d, st, max_tensors=200, log=lambda *_: None)["checked"] > 0

    def export_hash(q):
        out = tmp_path / f"{q.name}.dense"
        export_dense(d, q, out)
        return hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()

    assert export_hash(pe) == export_hash(st)
    # restack converts without re-encoding, both ways
    st2, pe2 = tmp_path / "st2", tmp_path / "pe2"
    restack(pe, st2, "stacked", log=lambda *_: None)
    restack(st2, pe2, "per-expert", log=lambda *_: None)
    for x, y in ((st, st2), (pe, pe2)):
        wx, wy = weight_map(x), weight_map(y)
        assert wx.keys() == wy.keys()
        for k in wx:
            with safe_open(x / wx[k], "pt") as f, safe_open(y / wy[k], "pt") as g:
                assert torch.equal(f.get_tensor(k), g.get_tensor(k)), k
