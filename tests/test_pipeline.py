# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""End-to-end pipeline on a tiny random Qwen2 model (CPU)."""

import json

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from hi2.calibrate import Config  # noqa: E402
from hi2.checkpoint import load_dequantized, verify_checkpoint  # noqa: E402
from hi2.pipeline import RunConfig, run  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    # row/column scale spread so allocation has something to find
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 2 and "proj" in n:
                p.mul_(torch.exp(0.5 * torch.randn(p.shape[0], 1)))
                p.mul_(torch.exp(0.4 * torch.randn(1, p.shape[1])))
    model.save_pretrained(d, safe_serialization=True)
    ids = torch.randint(0, 256, (8, 64), generator=torch.Generator().manual_seed(1))
    return d, model, ids


def _rc(tiny, tmp_path, tag, **kw):
    d, _, ids = tiny
    base = dict(
        src=d,
        dst=tmp_path / f"out-{tag}",
        workdir=tmp_path / f"work-{tag}",
        calib_ids=ids,
        target_bpw=3.5,
        calib_samples=8,
        seq_len=64,
        cfg=Config(effort="fast"),
    )
    base.update(kw)
    return RunConfig(**base)


def _logits_err(model, state, ids):
    from transformers import Qwen2ForCausalLM

    q = Qwen2ForCausalLM(model.config).float().eval()
    q.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    with torch.no_grad():
        a, b = model(ids).logits, q(ids).logits
    return ((a - b).norm() / a.norm()).item()


def test_calibrated_layerwise_run_end_to_end(tiny, tmp_path):
    rc = _rc(tiny, tmp_path, "cal")
    summary = run(rc, log=lambda *_: None)
    assert summary["feasible"] and summary["avg_bpw"] <= 3.5 + 0.05
    cfgj = json.loads((rc.dst / "config.json").read_text())
    assert cfgj["quantization_config"]["quant_method"] == "rdna_hint_flex"
    assert cfgj["quantization_config"]["version"] == "3.2"
    rep = json.loads((rc.dst / "hi2_report.json").read_text())["tensors"]
    # 3 layers x (q,k,v,o,gate,up,down); lm_head / embeddings stay dense
    assert len(rep) == 21 and all(r["n_tokens"] > 0 for r in rep.values())
    assert verify_checkpoint(rc.src, rc.dst, 5, log=lambda *_: None)["checked"] == 5
    err = _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2])
    assert err < 0.5


def test_more_bits_means_less_logit_error(tiny, tmp_path):
    errs = {}
    for bpw in (2.6, 4.5):
        rc = _rc(tiny, tmp_path, f"b{bpw}", target_bpw=bpw)
        run(rc, log=lambda *_: None)
        errs[bpw] = _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2])
    assert errs[4.5] < errs[2.6]


def test_data_free_run(tiny, tmp_path):
    rc = _rc(tiny, tmp_path, "df", calib_ids=None)
    summary = run(rc, log=lambda *_: None)
    assert summary["feasible"]
    assert (
        json.loads((rc.dst / "config.json").read_text())["quantization_config"][
            "calibrated"
        ]
        is False
    )


def test_two_devices_match_one(tiny, tmp_path):
    one = _rc(tiny, tmp_path, "d1", devices=["cpu"])
    two = _rc(tiny, tmp_path, "d2", devices=["cpu", "cpu"])
    s1, s2 = run(one, log=lambda *_: None), run(two, log=lambda *_: None)
    assert s1["avg_bpw"] == pytest.approx(s2["avg_bpw"], abs=2e-3)
    r1 = json.loads((one.dst / "hi2_report.json").read_text())["tensors"]
    r2 = json.loads((two.dst / "hi2_report.json").read_text())["tensors"]
    for name in r1:
        assert r1[name]["pred_rel_err"] == pytest.approx(
            r2[name]["pred_rel_err"], rel=0.02
        )


def test_allocation_varies_across_tensors(tiny, tmp_path):
    rc = _rc(tiny, tmp_path, "var", target_bpw=3.0)
    run(rc, log=lambda *_: None)
    rep = json.loads((rc.dst / "hi2_report.json").read_text())["tensors"]
    bpws = [r["bpw"] for r in rep.values()]
    assert max(bpws) - min(bpws) > 0.3  # the global allocator is not uniform


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_rotation_modes_round_trip(tiny, tmp_path, mode):
    rc = _rc(tiny, tmp_path, f"rot-{mode}", cfg=Config(effort="fast", rotation=mode))
    run(rc, log=lambda *_: None)
    rep = json.loads((rc.dst / "hi2_report.json").read_text())["tensors"]
    assert any(r["rotated_groups"] > 0 for r in rep.values()) or mode == "auto"
    err = _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2])
    assert err < 0.6


def test_smoothing_and_rotation_round_trip(tiny, tmp_path):
    rc = _rc(
        tiny,
        tmp_path,
        "smooth",
        cfg=Config(effort="fast", rotation="on", smooth=0.5),
    )
    run(rc, log=lambda *_: None)
    rep = json.loads((rc.dst / "hi2_report.json").read_text())["tensors"]
    assert all(r["rotated_groups"] == r["groups"] for r in rep.values())
    cfgj = json.loads((rc.dst / "config.json").read_text())
    assert verify_checkpoint(rc.src, rc.dst, 4, log=lambda *_: None)["checked"] == 4
    err = _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2])
    assert err < 0.5
    assert cfgj["quantization_config"]["version"] == "3.2"


def test_resolver_maps_text_model_names_to_multimodal_checkpoint():
    from hi2.checkpoint import make_resolver

    wmap = {
        "model.language_model.layers.0.mlp.down_proj.weight": "a",
        "mtp.layers.0.mlp.down_proj.weight": "a",
        "model.visual.blocks.0.attn.qkv.weight": "a",
        "model.layers.2.self_attn.q_proj.weight": "a",
    }
    r = make_resolver(wmap)
    assert r("model.layers.0.mlp.down_proj.weight") == (
        "model.language_model.layers.0.mlp.down_proj.weight"
    )
    assert r("model.layers.2.self_attn.q_proj.weight") == (
        "model.layers.2.self_attn.q_proj.weight"
    )
    assert r("model.layers.7.mlp.down_proj.weight") is None


def test_gptq_rounding_round_trip_and_beats_nearest(tiny, tmp_path):
    base = dict(effort="fast", rotation="on", smooth=0.5)
    out = {}
    for rounding in ("nearest", "gptq"):
        rc = _rc(
            tiny,
            tmp_path,
            f"r-{rounding}",
            target_bpw=2.8,
            calib_ids=torch.randint(
                0, 256, (24, 64), generator=torch.Generator().manual_seed(7)
            ),
            calib_samples=24,
            cfg=Config(rounding=rounding, **base),
        )
        run(rc, log=lambda *_: None)
        assert verify_checkpoint(rc.src, rc.dst, 3, log=lambda *_: None)["checked"] == 3
        out[rounding] = _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2])
    assert (
        out["gptq"] < out["nearest"] * 1.05
    )  # at least not worse (tiny, random model)


def test_sequential_gptq_round_trip(tiny, tmp_path):
    ids = torch.randint(0, 256, (24, 64), generator=torch.Generator().manual_seed(7))
    cfg = Config(effort="fast", rounding="gptq", sequential=True)
    rc = _rc(
        tiny, tmp_path, "seq", target_bpw=2.8, calib_ids=ids, calib_samples=24, cfg=cfg
    )
    summary = run(rc, log=lambda *_: None)
    assert summary["feasible"]
    assert verify_checkpoint(rc.src, rc.dst, 3, log=lambda *_: None)["checked"] == 3
    assert _logits_err(tiny[1], load_dequantized(rc.dst), tiny[2]) < 0.6


def test_sensitivity_probe_weights_and_allocation(tiny, tmp_path):
    ids = torch.randint(0, 256, (16, 64), generator=torch.Generator().manual_seed(7))
    rc = _rc(
        tiny,
        tmp_path,
        "sens",
        target_bpw=3.0,
        calib_ids=ids,
        calib_samples=16,
        sensitivity="probe",
        sens_seqs=4,
        sens_len=64,
        cfg=Config(effort="fast", rounding="nearest"),
    )
    summary = run(rc, log=lambda *_: None)
    sens = json.loads((rc.workdir / "sensitivity.json").read_text())
    assert len(sens) == 21 and all(v > 0 for v in sens.values())
    assert max(sens.values()) > 2 * min(sens.values())  # tensors differ in sensitivity
    assert summary["feasible"] and summary["avg_bpw"] <= 3.0 + 0.05


def test_multi_device_executor_serializes_the_triton_autotuner():
    triton = pytest.importorskip("triton.runtime.autotuner")
    from hi2.executor import Executor, serialize_triton_autotune

    Executor(["cpu"])  # one device: no patch needed
    Executor(["cpu", "cpu"])  # several: thread-safe autotuning
    assert triton.Autotuner.run._hi2_locked
    run = triton.Autotuner.run
    serialize_triton_autotune()
    assert triton.Autotuner.run is run  # idempotent
