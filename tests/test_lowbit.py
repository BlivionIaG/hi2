# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Low-bit options (docs/lowbit.md): INT8 rows end to end, the final logit-KL
phase, norm training and the KL term of the layer-wise phase (CPU, tiny Qwen2)."""

import json
from dataclasses import asdict

import pytest
import torch

from hi2.allocate import ROW_OVERHEAD_BITS, row_overhead_bits
from hi2.calibrate import ALL_CANDIDATES, INT8_CANDIDATES, Config, build_table
from hi2.executor import Executor
from hi2.format import dequantize, dequantize_stored, tensor_bpw
from hi2.qad import QadConfig, make_config, qad_fingerprint_fields, qad_meta
from hi2.stats import ColumnStats

pytest.importorskip("transformers")
pytest.importorskip("safetensors")


def _table(cands):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(32, 128, generator=g) * 0.05
    st = ColumnStats(128)
    st.update(torch.randn(256, 128, generator=g))
    return build_table(
        "t", w, st, Config(candidates=cands, effort="fast"), Executor("cpu")
    )


def test_row_overhead_is_unchanged_without_int8():
    assert row_overhead_bits(_table(list(ALL_CANDIDATES))) == ROW_OVERHEAD_BITS
    assert row_overhead_bits(_table([(4, 32)])) == ROW_OVERHEAD_BITS
    both = list(ALL_CANDIDATES) + list(INT8_CANDIDATES)
    assert row_overhead_bits(_table(both)) == ROW_OVERHEAD_BITS + 1


def test_new_qad_fields_leave_default_config_and_fingerprint_unchanged():
    qc = QadConfig()
    old = {
        k
        for k in asdict(qc)
        if not k.startswith("final_")
        and k
        not in ("train_norms", "lr_norm", "kl_layers", "kl_weight", "kl_tokens", "data_parallel")
    }
    assert set(qad_meta(qc)) == old
    assert set(qad_fingerprint_fields(qc)) <= old
    on = make_config("default", {"final_kl": 5, "train_norms": "true"})
    assert on.train_norms is True and on.final_kl == 5
    assert "final_kl" not in qad_fingerprint_fields(on)  # layer artifacts stay reusable
    assert qad_fingerprint_fields(on)["train_norms"] is True
    assert qad_meta(on)["final_kl"] == 5


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tinylowbit")
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    with torch.no_grad():  # non-trivial norm weights, so a trained copy can differ
        for n, p in model.named_parameters():
            if "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    model.save_pretrained(d, safe_serialization=True)
    ids = torch.randint(0, 256, (10, 64), generator=torch.Generator().manual_seed(1))
    return d, model, ids


def _rc(tiny, tmp_path, tag, qad=None, **kw):
    from hi2.pipeline import RunConfig

    d, _, ids = tiny
    base = dict(
        src=d,
        dst=tmp_path / f"out-{tag}",
        workdir=tmp_path / f"work-{tag}",
        calib_ids=ids,
        target_bpw=3.0,
        calib_samples=8,
        seq_len=64,
        cfg=Config(effort="fast"),
        qad=make_config(
            "fast", {"steps": 6, "samples": 8, "val": 2, "batch": 2, **(qad or {})}
        ),
    )
    base.update(kw)
    return RunConfig(**base)


def _hi2(dst):
    from safetensors import safe_open

    from hi2.checkpoint import from_tensors, read_tensors, weight_map

    wmap = weight_map(dst)
    out = {}
    for k, shard in wmap.items():
        if k.endswith(".hi2_codes"):
            base = k.removesuffix(".hi2_codes")
            with safe_open(dst / shard, "pt") as f:
                out[base] = from_tensors(read_tensors(f, base), base)
    return out


def _dense(dst, name):
    from safetensors import safe_open

    from hi2.checkpoint import weight_map

    with safe_open(dst / weight_map(dst)[name], "pt") as f:
        return f.get_tensor(name)


def test_int8_rows_end_to_end_with_probe_cache_and_qad(tiny, tmp_path):
    """--bits 2,4,8: quality allocation, sensitivity probe, cache, batched GPTQ encode
    and QAD all handle mixed INT2/INT4/INT8 rows; verify, export-dense and the reported
    bpw agree."""
    from hi2.checkpoint import export_dense, verify_checkpoint
    from hi2.pipeline import run

    cands = list(ALL_CANDIDATES) + list(INT8_CANDIDATES)
    rc = _rc(
        tiny,
        tmp_path,
        "i8",
        target_bpw=None,
        max_error_ratio=40.0,
        cfg=Config(candidates=cands, effort="fast"),
        sensitivity="probe",
        sens_seqs=2,
        sens_len=32,
        cache_dir=tmp_path / "cache",
    )
    s = run(rc, log=lambda *_: None)
    assert s["frac_rows_by_bits"].get(8, 0) > 0 and s["frac_int8_rows"] > 0
    assert abs(s["allocation_bpw"] - s["avg_bpw"]) < 0.02
    widths = set()
    for t in _hi2(rc.dst).values():
        widths |= set(t.row_nbits.tolist())
        assert 0 < tensor_bpw(t) < 9
    assert 8 in widths and len(widths) >= 2
    assert verify_checkpoint(rc.src, rc.dst, 4, log=lambda *_: None)["checked"] == 4
    export_dense(rc.src, rc.dst, tmp_path / "dense-i8")
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tmp_path / "dense-i8")
    assert torch.isfinite(m(tiny[2][:1]).logits).all()


def _ptq_mixed(r=24, k=128):
    """GPTQ encode with rows on INT2, INT4 and INT8, smoothing and rotation."""
    from hi2.calibrate import encode_with_picks
    from hi2.hessian import HessianStats

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2048, k, generator=g) * torch.exp(
        0.5 * torch.randn(1, k, generator=g)
    )
    w = torch.randn(r, k, generator=g) * 0.05
    st, hs = ColumnStats(k), HessianStats(k)
    st.update(x)
    hs.update(x)
    cands = [(2, 32), (4, 64), (8, 32), (2, 16)]
    cfg = Config(candidates=cands, effort="fast")
    ex = Executor("cpu")
    table = build_table("t", w, st, cfg, ex)
    picks = torch.arange(r) % len(cands)
    return encode_with_picks(w, st, table, picks, cfg, ex, hs.mean())


def test_ladder_linear_is_the_exact_decode():
    from hi2.finalkl import LadderLinear

    t0 = _ptq_mixed()
    assert set(t0.row_nbits.tolist()) == {2, 4, 8}
    ll = LadderLinear(t0)
    assert torch.equal(ll.stored().double(), dequantize_stored(t0))
    assert torch.allclose(ll.weight().double(), dequantize(t0), atol=1e-6)
    t1 = ll.export()
    assert torch.equal(dequantize_stored(t1), dequantize_stored(t0))
    with torch.no_grad():  # moved bytes: export still equals the forward, codes kept
        for p in (*ll.lo_u, *ll.step_u):
            p.add_(2.0)
        ll.project()
    t2 = ll.export()
    assert torch.equal(ll.stored().double(), dequantize_stored(t2))
    assert torch.equal(t2.codes, t0.codes)


def test_final_kl_phase_writes_a_normal_checkpoint_and_resumes(tiny, tmp_path):
    from hi2.checkpoint import verify_checkpoint
    from hi2.pipeline import run

    base = _rc(tiny, tmp_path, "base")
    run(base, log=lambda *_: None)
    rc = _rc(tiny, tmp_path, "fkl", qad={"final_kl": 4, "final_lr_p": 0.5})
    # share the layer-wise artifacts: the final phase must not invalidate them
    import shutil

    rc.workdir.mkdir(parents=True)
    shutil.copytree(base.workdir / "qad", rc.workdir / "qad")
    logs: list[str] = []
    run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert sum("QAD layer" in ln and "reusing" in ln for ln in logs) == 2
    info = json.loads((rc.workdir / "final_kl.json").read_text())
    assert info["val_after"] <= info["val_before"]
    a, b = _hi2(base.dst), _hi2(rc.dst)
    assert a.keys() == b.keys()
    for k in a:  # codes frozen, only bytes may move
        assert torch.equal(a[k].codes, b[k].codes)
    qcfg = json.loads((rc.dst / "config.json").read_text())["quantization_config"]
    assert qcfg["qad"]["final_kl"] == 4
    if info["kept"] == "best":
        assert qcfg["qad"]["dense_trained"] > 0
        assert _dense(rc.dst, "model.norm.weight").dtype == torch.float16
    assert verify_checkpoint(rc.src, rc.dst, 3, log=lambda *_: None)["checked"] == 3
    logs.clear()
    shutil.rmtree(rc.dst)
    run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert any("final KL: reusing" in ln for ln in logs)


def test_layerwise_norms_and_kl_term(tiny, tmp_path):
    from hi2.pipeline import run

    rc = _rc(
        tiny,
        tmp_path,
        "nrm",
        qad={"train_norms": True, "lr_norm": 1e-2, "kl_layers": 1, "kl_tokens": 16},
    )
    run(rc, log=lambda *_: None)
    rep = json.loads((rc.workdir / "qad_report.json").read_text())
    assert all("norms" not in r for r in rep)
    trained = [r for r in rep if r["kept"] != "ptq"]
    src = tiny[1].state_dict()
    qcfg = json.loads((rc.dst / "config.json").read_text())["quantization_config"]
    assert qcfg["qad"]["train_norms"] is True and qcfg["qad"]["kl_layers"] == 1
    if trained:
        li = trained[0]["layer"]
        name = f"model.layers.{li}.input_layernorm.weight"
        got = _dense(rc.dst, name)
        assert got.dtype == torch.float16
        assert not torch.equal(got.float(), src[name].half().float())


def test_int8_force_rule_puts_the_most_sensitive_tensors_on_int8(tiny, tmp_path):
    from hi2.pipeline import run

    cands = list(ALL_CANDIDATES) + list(INT8_CANDIDATES)
    rc = _rc(
        tiny,
        tmp_path,
        "force",
        qad=None,
        target_bpw=4.0,
        cfg=Config(candidates=cands, effort="fast"),
        sensitivity="probe",
        sens_seqs=2,
        sens_len=32,
        int8_force=2,
    )
    rc.qad = None
    run(rc, log=lambda *_: None)
    sens = json.loads((rc.workdir / "sensitivity.json").read_text())
    top = sorted(sens, key=lambda n: -sens[n])[:2]
    for name, t in _hi2(rc.dst).items():
        nb = set(t.row_nbits.tolist())
        if f"{name}.weight" in top:
            assert nb == {8}
        else:
            assert 8 not in nb


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs for the layer-wise split"
)
def test_final_kl_splits_the_network_over_two_devices(tiny, tmp_path):
    from hi2.pipeline import run

    kw = dict(qad={"final_kl": 4, "final_lr_p": 0.5})
    one = _rc(tiny, tmp_path, "one", devices=["cuda:0"], **kw)
    two = _rc(tiny, tmp_path, "two", devices=["cuda:0", "cuda:1"], **kw)
    for rc in (one, two):
        run(rc, log=lambda *_: None)
    i1 = json.loads((one.workdir / "final_kl.json").read_text())
    i2 = json.loads((two.workdir / "final_kl.json").read_text())
    assert i2["val_after"] <= i2["val_before"]
    assert i2["val_before"] == pytest.approx(i1["val_before"], rel=1e-2)
    a, b = _hi2(one.dst), _hi2(two.dst)
    for k in a:
        assert torch.equal(a[k].codes, b[k].codes)
