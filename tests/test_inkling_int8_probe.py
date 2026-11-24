# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The INT8 class probe of Inkling-2 (``experiments/inkling_int8_probe.py``), pure torch on the CPU.

A toy model carries the names of the real one (``layers.N.self_attn.q_proj.linear`` ...)
and the decoys that must stay (``sink_proj``, the router); the report is checked on eval
results written by hand. The probe inside ``inkling_kl_eval.py`` is tested on the tiny Inkling
checkpoint in ``tests/test_inkling_kl_eval.py``."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from hi2.calibrate import Config, build_table, encode_with_picks
from hi2.executor import Executor
from hi2.format import tensor_bpw

K = 128


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "experiments" / "inkling_int8_probe.py"
    name = "hi2_test_inkling_int8_probe"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Module(nn.Module):
    pass


def _linear(i, o, g):
    m = nn.Linear(i, o, bias=False)
    m.weight.data = torch.randn(o, i, generator=g) * 0.05
    return m


class Toy(nn.Module):
    """Two decoder layers with the attention, router and sink of the real names."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.model = _Module()
        text = self.model.language_model = _Module()
        text.embed_tokens = nn.Embedding(40, K)
        text.embed_tokens.weight.data = torch.randn(40, K, generator=g) * 0.05
        text.layers = nn.ModuleList()
        for _ in range(2):
            layer, a = _Module(), _Module()
            layer.self_attn = a
            a.q_proj, a.k_proj, a.v_proj = _Module(), _Module(), _Module()
            a.q_proj.linear = _linear(K, 2 * K, g)
            a.k_proj.linear = _linear(K, K, g)
            a.v_proj.linear = _linear(K, K, g)
            a.o_proj = _Module()
            a.o_proj.linear = _linear(2 * K, K, g)
            a.o_proj.gate_proj = _linear(K, 2 * K, g)
            a.relative_position_bias = _Module()
            a.relative_position_bias.linear = _linear(K, 16, g)
            a.sink_proj = _linear(K, 4, g)  # stays as it is
            layer.mlp = _Module()
            layer.mlp.gate = _linear(K, 8, g)  # the router stays as it is
            text.layers.append(layer)
        self.lm_head = _linear(K, 50, g)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ------------------------------------------------------------------------ the classes


def test_the_attention_class_is_the_six_projections_of_every_layer_and_nothing_else(
    probe,
):
    names = sorted(probe.class_weights(Toy(), "attn"))
    want = sorted(
        f"model.language_model.layers.{i}.self_attn.{p}"
        for i in range(2)
        for p in (
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "o_proj.gate_proj",
            "relative_position_bias.linear",
        )
    )
    assert names == want


def test_the_embedding_and_the_head_are_one_tensor_each_and_other_names_are_refused(
    probe,
):
    toy = Toy()
    assert list(probe.class_weights(toy, "embed")) == ["embed_tokens"]
    assert (
        probe.class_weights(toy, "embed")["embed_tokens"]
        is toy.get_input_embeddings().weight
    )
    assert probe.class_weights(toy, "lm_head")["lm_head"] is toy.lm_head.weight
    with pytest.raises(SystemExit, match="unknown class"):
        probe.class_weights(toy, "experts")
    with pytest.raises(SystemExit, match="no tensor of the class"):
        probe.class_weights(nn.Linear(4, 4), "attn")


# ---------------------------------------------------------------------- the round trip


def _encoded_bits(w, block):
    cfg = Config(candidates=[(8, block)], rounding="nearest", smooth=0.0)
    ex = Executor("cpu")
    table = build_table("t", w, None, cfg, ex)
    t = encode_with_picks(
        w, None, table, torch.zeros(w.shape[0], dtype=torch.int64), cfg, ex
    )
    return tensor_bpw(t) * w.numel()


@pytest.mark.parametrize("block", [32, 64])
def test_the_bits_are_those_of_the_format(probe, block):
    w = torch.randn(24, K, generator=torch.Generator().manual_seed(1)) * 0.05
    assert probe.int8_bits(24, K, block) == pytest.approx(_encoded_bits(w, block))
    # a block of 32 costs 16 bits per 32 weights, a block of 64 per 64
    assert probe.int8_bits(1, 64, 32) - probe.int8_bits(1, 64, 64) == 16 * (2 - 1)


def test_the_round_trip_is_a_small_error_and_a_smaller_block_makes_it_smaller(probe):
    g = torch.Generator().manual_seed(2)
    w = torch.randn(64, 256, generator=g) * 0.05
    ex = Executor("cpu")
    err = {}
    for block in (32, 64):
        dq = probe.int8_roundtrip(w, block, ex)
        assert (
            dq.dtype == torch.float32 and dq.shape == w.shape and not torch.equal(dq, w)
        )
        err[block] = (
            ((dq - w).double().pow(2).sum() / w.double().pow(2).sum()).sqrt().item()
        )
        assert 1e-4 < err[block] < 1e-2  # 256 levels per block of a normal weight
    assert err[32] < err[64]


def test_encoding_in_chunks_of_rows_changes_nothing(probe, monkeypatch):
    w = torch.randn(23, 128, generator=torch.Generator().manual_seed(3)) * 0.05
    ex = Executor("cpu")
    whole = probe.int8_roundtrip(w, 64, ex)
    monkeypatch.setattr(probe, "CHUNK_ROWS", 5)  # 5 chunks, the last of 3 rows
    assert torch.equal(probe.int8_roundtrip(w, 64, ex), whole)


# ------------------------------------------------------------------------ apply


@pytest.mark.parametrize("cls", ["attn", "embed"])
def test_storing_a_class_changes_that_class_alone_and_counts_its_bytes(probe, cls):
    toy = Toy()
    before = _snapshot(toy)
    chosen = probe.class_weights(toy, cls)
    info, lm_student = probe.apply(toy, cls, 64, Executor("cpu"), lambda *_: None)
    assert lm_student is None
    names = {
        n for n, p in toy.named_parameters() if any(p is q for q in chosen.values())
    }
    assert names and len(names) == info["tensors"]
    for n, p in toy.named_parameters():
        assert torch.equal(p, before[n]) == (n not in names), n
    elements = sum(p.numel() for p in chosen.values())
    assert info["elements"] == elements and info["bytes_bf16"] == 2 * elements
    want = sum(probe.int8_bits(*p.shape, 64) for p in chosen.values()) / 8
    assert info["bytes_int8"] == pytest.approx(want)
    assert info["bytes_saved"] == pytest.approx(2 * elements - want)
    assert info["bits_per_weight"] == pytest.approx(8 * want / elements)
    assert 0 < info["weight_rms_rel_error"] < 1e-2
    assert info["weight_max_abs_error"] > 0
    assert "(8, 64)" in info["scheme"]


def test_the_head_is_stored_in_a_copy_and_the_model_keeps_its_own(probe):
    toy = Toy()
    before = _snapshot(toy)
    lm = toy.lm_head
    info, lm_student = probe.apply(
        toy, "lm_head", 32, Executor("cpu"), lambda *_: None, lm
    )
    assert all(torch.equal(p, before[n]) for n, p in toy.named_parameters())
    assert lm_student is not lm and not torch.equal(lm_student.weight, lm.weight)
    rel = (lm_student.weight - lm.weight).norm() / lm.weight.norm()
    assert 0 < rel < 1e-2  # the decode is close to the weights, not a copy of them
    assert info["tensors"] == 1 and info["elements"] == 50 * K
    with pytest.raises(SystemExit, match="needs the logits function"):
        probe.apply(toy, "lm_head", 32, Executor("cpu"), lambda *_: None)


def test_a_head_behind_a_soft_cap_is_replaced_inside_its_wrapper(probe):
    from hi2.finalkl import _Logits

    toy = Toy()
    cap = nn.Tanh()
    lm = _Logits(toy.lm_head, cap)
    _, lm_student = probe.apply(
        toy, "lm_head", 64, Executor("cpu"), lambda *_: None, lm
    )
    assert lm_student.head is not toy.lm_head and lm.head is toy.lm_head
    assert torch.equal(lm.head.weight, toy.lm_head.weight)
    assert not torch.equal(lm_student.head.weight, toy.lm_head.weight)


# ----------------------------------------------------------------------- the report


def _result(kls, top1=0.93, ppl=1.01, sem=None):
    n = len(kls)
    mean = sum(kls) / n
    if sem is None and n > 1:
        sem = (sum((x - mean) ** 2 for x in kls) / (n - 1) / n) ** 0.5
    return {
        "student": {
            "windows": n,
            "kl_mean": mean,
            "kl_sem": sem,
            "kl_per_window": kls,
            "top1_agreement": top1,
            "ppl_ratio": ppl,
        }
    }


BASE = [0.05, 0.10, 0.02, 0.30, 0.08, 0.04]


def test_a_class_that_changes_nothing_within_the_noise_passes(probe):
    var = _result([x + d for x, d in zip(BASE, (1e-5, -2e-5, 1e-5, 0, 2e-5, -1e-5))])
    c = probe.compare(_result(BASE), var)
    assert c["windows"] == 6 and c["kl_ok"] and c["top1_ok"] and c["pass"]
    assert c["increase"] == pytest.approx(sum((1e-5, -2e-5, 1e-5, 0, 2e-5, -1e-5)) / 6)
    assert c["increase_sem"] < 1e-4 and c["windows_worse"] == 3


def test_an_increase_beyond_the_paired_noise_fails_the_class(probe):
    var = _result([x + 1e-3 * (1 + 0.1 * i) for i, x in enumerate(BASE)])
    c = probe.compare(_result(BASE), var)
    assert c["z"] > 10 and not c["kl_ok"] and not c["pass"] and c["windows_worse"] == 6
    # the same increase passes when the pass line is far enough out
    assert probe.compare(_result(BASE), var, z_pass=1e6)["kl_ok"]


def test_a_decrease_never_fails_and_the_noise_is_the_paired_one(probe):
    # windows differ by 0.3 between each other (an unpaired sem would be ~0.04), the shift is
    # 2e-4 on every window: paired sem is 0, so 'within the noise' means no increase at all
    var = _result([x + 2e-4 for x in BASE])
    c = probe.compare(_result(BASE), var)
    assert c["increase_sem"] == pytest.approx(0, abs=1e-12) and not c["kl_ok"]
    down = probe.compare(_result(BASE), _result([x - 2e-4 for x in BASE]))
    assert down["kl_ok"] and down["increase"] < 0


def test_the_top_one_agreement_must_not_move_beyond_the_tolerance(probe):
    same = probe.compare(_result(BASE), _result(BASE, top1=0.93005))
    assert same["top1_ok"] and same["top1_change"] == pytest.approx(5e-5)
    for moved in (0.9298, 0.9302):  # lower or higher: both are a change
        c = probe.compare(_result(BASE), _result(BASE, top1=moved))
        assert not c["top1_ok"] and not c["pass"] and c["kl_ok"]
    assert probe.compare(_result(BASE), _result(BASE, top1=0.9302), top1_tol=1e-3)[
        "top1_ok"
    ]


def test_one_window_has_no_spread_and_only_an_unchanged_kl_passes(probe):
    assert probe.compare(_result([0.1]), _result([0.1]))["kl_ok"]
    assert not probe.compare(_result([0.1]), _result([0.1001]))["kl_ok"]


def test_different_windows_are_refused(probe):
    with pytest.raises(SystemExit, match="different numbers of windows"):
        probe.compare(_result(BASE), _result(BASE[:-1]))


def _write_evals(root: Path, classes=("attn", "embed", "lm_head"), worse=()):
    sets = {"mixed": BASE, "wiki": BASE[:4], "pydocs": BASE[2:]}
    for s, kls in sets.items():
        (root / f"bpw3.5_{s}.json").write_text(json.dumps(_result(kls)))
        for i, cls in enumerate(classes):
            noise = (
                1e-5,
                -2e-5,
                1e-5,
                0.0,
                2e-5,
                -1e-5,
            )  # zero-mean, within the paired noise
            more = 1e-2 if cls in worse else 0.0
            var = _result(
                [x + noise[j] + more * (1 + j / 10) for j, x in enumerate(kls)]
            )
            var["int8"] = {
                "class": cls,
                "tensors": 3 + i,
                "elements": 1000 * (i + 1),
                "bytes_bf16": 2**31,
                "bytes_int8": 2**30 * 1.03,
                "bytes_saved": 2**31 - 2**30 * 1.03,
            }
            (root / f"bpw3.5-int8{cls}64_{s}.json").write_text(json.dumps(var))
            (root / f"stock-int8{cls}64_{s}.json").write_text(
                json.dumps(_result([1e-4 * (1 + j) for j in range(len(kls))]))
            )


def test_the_report_gives_a_verdict_per_class_and_the_bytes_it_saves(
    probe, tmp_path, capsys
):
    _write_evals(tmp_path, worse=("embed",))
    out = tmp_path / "r.json"
    code = probe.main(
        ["report", "--eval", str(tmp_path), "--block", "64", "--json", str(out)]
    )
    assert code == 0  # a finished report is not an error whatever it says
    rep = json.loads(out.read_text())
    assert rep["attn"]["pass"] and rep["lm_head"]["pass"] and not rep["embed"]["pass"]
    assert set(rep["attn"]["sets"]) == {"mixed", "wiki", "pydocs"}
    assert rep["attn"]["info"]["tensors"] == 3
    alone = rep["attn"]["sets"]["mixed"]["alone"]  # the class on the stock experts
    assert alone["kl"] == pytest.approx(sum(1e-4 * (1 + j) for j in range(6)) / 6)
    assert alone["share_of_ptq_kl"] == pytest.approx(alone["kl"] / (sum(BASE) / 6))
    text = capsys.readouterr().out
    assert "| **attn** |" in text and "**PASS**" in text and "**stays bf16**" in text
    assert "1.000 -> 1.030" not in text  # the columns are bf16 -> INT8 (saved)
    assert "2.000 -> 1.030 (0.970)" in text


def test_a_failure_on_one_set_keeps_the_class_in_bf16(probe, tmp_path):
    _write_evals(tmp_path)
    path = tmp_path / "bpw3.5-int8attn64_wiki.json"
    var = json.loads(path.read_text())
    var["student"]["top1_agreement"] -= 5e-3
    path.write_text(json.dumps(var))
    rep = probe.build_report(tmp_path, "bpw3.5", ["attn", "embed"], 64)
    assert not rep["attn"]["pass"] and rep["embed"]["pass"]
    sets = rep["attn"]["sets"]
    assert sets["mixed"]["pass"] and sets["pydocs"]["pass"] and not sets["wiki"]["pass"]


def test_the_stock_student_evals_are_optional(probe, tmp_path):
    _write_evals(tmp_path)
    rep = probe.build_report(tmp_path, "bpw3.5", ["attn"], 64, stock=False)
    assert "alone" not in rep["attn"]["sets"]["mixed"]
    for p in tmp_path.glob("stock-*"):
        p.unlink()
    rep = probe.build_report(tmp_path, "bpw3.5", ["attn"], 64)
    assert "alone" not in rep["attn"]["sets"]["mixed"] and rep["attn"]["pass"]
