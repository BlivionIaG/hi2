# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Layer-wise QAD (spec 13): exact fake-quant forward, STE gradients, training, export (CPU)."""

import json

import pytest
import torch
from torch import nn

from hi2.calibrate import Config, build_table, encode_with_picks
from hi2.executor import Executor
from hi2.format import dequantize, dequantize_stored
from hi2.hessian import HessianStats
from hi2.qad import FakeQuantLinear, make_config, parse_layers, train_layer
from hi2.stats import ColumnStats

CANDS = [(2, 32), (2, 16), (4, 32), (4, 64), (8, 32)]


def _x(t=2048, k=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(k, k, generator=g) * 0.1 + torch.eye(k)
    return (torch.randn(t, k, generator=g) @ mix) * torch.exp(
        0.5 * torch.randn(1, k, generator=g)
    )


def _w(r, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(r, k, generator=g) * 0.05
    return w * torch.exp(0.4 * torch.randn(r, 1, generator=g))


def _ptq(w, x, rounding="gptq", act_order=False, rotation="on", smooth=0.5):
    """PTQ encode of ``w`` with mixed per-row picks; returns (table, t0, comp)."""
    k = w.shape[1]
    st = ColumnStats(k)
    st.update(x)
    hs = HessianStats(k)
    hs.update(x)
    cfg = Config(
        candidates=CANDS,
        rotation=rotation,
        smooth=smooth,
        rounding=rounding,
        act_order=act_order,
        effort="fast",
    )
    ex = Executor("cpu")
    table = build_table("t", w, st, cfg, ex)
    picks = torch.arange(w.shape[0]) % len(CANDS)
    co: dict = {}
    t0 = encode_with_picks(w, st, table, picks, cfg, ex, hs.mean(), co)
    return t0, co["comp"]


def _perturb(fq, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fq.latent.add_(0.01 * torch.randn(fq.latent.shape, generator=g))
        for grp in fq.groups:
            grp.lo_u.add_(3 * torch.randn(grp.lo_u.shape, generator=g))
            grp.step_u.add_(3 * torch.randn(grp.step_u.shape, generator=g))
    fq.project()


@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("rounding", ["gptq", "nearest"])
def test_ptq_init_reproduces_the_ptq_tensor_exactly(rounding, act_order):
    w, x = _w(40, 128), _x()
    t0, comp = _ptq(w, x, rounding, act_order)
    fq = FakeQuantLinear(w, t0, comp, "ptq")
    t1 = fq.export()
    assert torch.equal(t1.codes, t0.codes) and torch.equal(t1.blk_params, t0.blk_params)
    assert torch.equal(dequantize_stored(t1), dequantize_stored(t0))


@pytest.mark.parametrize("units", ["abs", "step"])
@pytest.mark.parametrize("smooth,rotation", [(0.5, "on"), (0.0, "off"), (0.5, "auto")])
def test_fake_quant_forward_equals_dequantized_export(smooth, rotation, units):
    w, x = _w(40, 128), _x()
    t0, comp = _ptq(w, x, smooth=smooth, rotation=rotation)
    for init in ("ptq", "weights"):
        fq = FakeQuantLinear(w, t0, comp, init, units)
        _perturb(fq)
        t = fq.export()
        # stored basis: bit-exact
        assert torch.equal(dequantize_stored(t), fq.stored(False).double())
        assert torch.equal(dequantize_stored(t), fq.stored(True).detach().double())
        # original basis: the forward runs in fp32, the reference dequantize in fp64
        ref = dequantize(t).float()
        got = fq().detach()
        assert (got - ref).abs().max() <= 1e-5 * ref.abs().max()
        assert t.row_nbits.tolist() == t0.row_nbits.tolist()
        assert t.row_block.tolist() == t0.row_block.tolist()
        assert torch.equal(t.row_consts, t0.row_consts)


@pytest.mark.parametrize("units", ["abs", "step"])
def test_ste_gradients_are_finite_and_nonzero(units):
    w, x = _w(40, 128), _x()
    t0, comp = _ptq(w, x)
    fq = FakeQuantLinear(w, t0, comp, "weights", units)
    xb = x[:256]
    loss = ((xb @ fq().T - xb @ w.T) ** 2).mean()
    loss.backward()
    assert torch.isfinite(fq.latent.grad).all() and fq.latent.grad.abs().sum() > 0
    for g in fq.groups:
        for p in (g.lo_u, g.step_u):
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert sum(g.step_u.grad.abs().sum() for g in fq.groups) > 0


class _MLP(nn.Module):
    def __init__(self, k=128, h=256, seed=5):
        super().__init__()
        self.up = nn.Linear(k, h, bias=False)
        self.down = nn.Linear(h, k, bias=False)
        with torch.no_grad():
            self.up.weight.copy_(_w(h, k, seed))
            self.down.weight.copy_(_w(k, h, seed + 1))

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.up(x)))


def test_qad_steps_reduce_layer_output_error_vs_ptq():
    torch.manual_seed(0)
    layer = _MLP()
    xs = [_x(256, 128, seed=10 + i).unsqueeze(0) for i in range(12)]
    with torch.no_grad():
        ys = [layer(x) for x in xs]
    # student input: a perturbed copy (as if produced by a quantized prefix)
    xs_s = [x + 0.02 * x.std() * torch.randn_like(x) for x in xs]
    fq = {}
    for name, inp in (("up", torch.cat(xs).reshape(-1, 128)), ("down", None)):
        mod = layer.get_submodule(name)
        if inp is None:
            with torch.no_grad():
                inp = torch.nn.functional.silu(layer.up(torch.cat(xs))).reshape(-1, 256)
        w = mod.weight.detach().float()
        t0, comp = _ptq(w, inp)
        fq[name] = FakeQuantLinear(w, t0, comp, "ptq")
    qc = make_config(
        "default", {"steps": 80, "batch": 2, "lr_w": 1e-3, "lr_p": 0.1, "val": 2}
    )
    info = train_layer(
        layer,
        fq,
        xs_s,
        ys,
        list(range(10)),
        [10, 11],
        {},
        qc,
        torch.device("cpu"),
        lambda *_: None,
    )
    assert info["val_after"] < 0.95 * info["val_before"], info
    # the exported tensors reproduce the trained forward
    for m in fq.values():
        assert torch.equal(dequantize_stored(m.export()), m.stored(False).double())


def test_step_budget_spreads_steps_and_reuses_early_stop_savings():
    from hi2.qad import StepBudget

    qc = make_config("default", {"steps": 100, "budget": 0.5, "min_steps": 10})
    b = StepBudget(qc, 4)
    assert b.total == 200
    first = b.allot()
    assert first == 100  # first layer: 2x the per-layer share, capped at steps
    b.record(
        {"steps": 60, "val_before": 1.0, "val_after": 0.6, "seconds": 6.0}
    )  # early stop
    second = b.allot()  # gain equal to the mean so far: the remaining share
    assert second == round((200 - 60) / 3)
    b.record({"steps": second, "val_before": 1.0, "val_after": 0.95, "seconds": 4.0})
    third = b.allot()  # small gain relative to the mean: fewer steps
    assert 10 <= third < (200 - 60 - second) / 2
    assert "saved" in b.summary()
    full = StepBudget(make_config("default", {"steps": 100, "budget": 1.0}), 4)
    assert full.allot() == 100


def test_early_stop_and_allotted_steps():
    layer = _MLP()
    xs = [_x(128, 128, seed=40 + i).unsqueeze(0) for i in range(6)]
    with torch.no_grad():
        ys = [layer(x) for x in xs]
    fq = {}
    acts = {"up": torch.cat(xs).reshape(-1, 128)}
    with torch.no_grad():
        acts["down"] = torch.nn.functional.silu(layer.up(torch.cat(xs))).reshape(
            -1, 256
        )
    for name in ("up", "down"):
        w = layer.get_submodule(name).weight.detach().float()
        t0, comp = _ptq(w, acts[name])
        fq[name] = FakeQuantLinear(w, t0, comp, "weights", "step")
    # a huge min_gain stops at the first evaluation after min_steps
    qc = make_config(
        "default",
        {
            "steps": 200,
            "batch": 1,
            "val": 2,
            "min_gain": 10.0,
            "min_steps": 10,
            "eval_every": 5,
        },
    )
    info = train_layer(
        layer,
        fq,
        xs,
        ys,
        [0, 1, 2, 3],
        [4, 5],
        {},
        qc,
        torch.device("cpu"),
        lambda *_: None,
        "",
        None,
        40,
    )
    assert info["allotted"] == 40 and info["early_stop"] and info["steps"] == 10


def test_config_presets_and_layer_spec():
    assert make_config("fast").budget < 1 and make_config("default").budget == 1
    assert make_config("default").steps < make_config("thorough").steps
    assert make_config("default").final_kl > 0 and make_config("thorough").final_kl > 0
    assert make_config("fast").final_kl == 0
    assert make_config("default", {"final_kl": 0}).final_kl == 0
    assert make_config("default", {"steps": "7", "keep_best": "false"}).steps == 7
    assert make_config("default", {"keep_best": "false"}).keep_best is False
    with pytest.raises(SystemExit):
        make_config("default", {"nope": 1})
    with pytest.raises(SystemExit):
        make_config("bogus")
    assert parse_layers("0-2,5", 8) == {0, 1, 2, 5}
    assert parse_layers(None, 3) == {0, 1, 2}
    with pytest.raises(SystemExit):
        parse_layers("9", 4)


# ---- end to end on a tiny random Qwen2 (writer, resume, errors) ----------------

pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tinyqad")
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
    model.save_pretrained(d, safe_serialization=True)
    ids = torch.randint(0, 256, (10, 64), generator=torch.Generator().manual_seed(1))
    return d, model, ids


def _rc(tiny, tmp_path, tag, **kw):
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
        qad=make_config("fast", {"steps": 6, "samples": 8, "val": 2, "batch": 2}),
    )
    base.update(kw)
    return RunConfig(**base)


def test_qad_pipeline_writes_a_normal_checkpoint_and_resumes(tiny, tmp_path):
    from safetensors import safe_open

    from hi2.checkpoint import from_tensors, read_tensors, verify_checkpoint, weight_map
    from hi2.pipeline import run

    rc = _rc(tiny, tmp_path, "q")
    logs: list[str] = []
    summary = run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert summary["feasible"]
    qcfg = json.loads((rc.dst / "config.json").read_text())["quantization_config"]
    assert qcfg["quant_method"] == "rdna_hint_flex" and qcfg["version"] == "3.2"
    assert qcfg["qad"]["steps"] == 6
    assert verify_checkpoint(rc.src, rc.dst, 4, log=lambda *_: None)["checked"] == 4
    rep = json.loads((rc.workdir / "qad_report.json").read_text())
    assert len(rep) == 2 and all("val_before" in r for r in rep)
    # the written tensors are exactly the per-layer QAD artifacts
    art = torch.load(rc.workdir / "qad" / "layer001.pt", weights_only=False)
    wmap = weight_map(rc.dst)
    for name, t in art["tensors"].items():
        base = name.removesuffix(".weight")
        with safe_open(rc.dst / wmap[f"{base}.hi2_codes"], "pt") as f:
            got = from_tensors(read_tensors(f, base), base)
        assert torch.equal(dequantize_stored(got), dequantize_stored(t))
    # resume: the finished layers are reused (phase 1 is not cached for calib_ids runs)
    logs.clear()
    import shutil

    shutil.rmtree(rc.dst)
    run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert sum("QAD layer" in ln and "reusing" in ln for ln in logs) == 2


def test_qad_layer_selection_keeps_ptq_elsewhere(tiny, tmp_path):
    from hi2.pipeline import run

    qc = make_config(
        "fast", {"steps": 3, "samples": 8, "val": 2, "batch": 2, "layers": "1"}
    )
    rc = _rc(tiny, tmp_path, "sel", qad=qc)
    run(rc, log=lambda *_: None)
    rep = json.loads((rc.workdir / "qad_report.json").read_text())
    assert rep[0]["kept"] == "ptq" and "val_before" in rep[1]


def test_qad_without_calibration_falls_back_to_ptq(tiny, tmp_path):
    from hi2.pipeline import run

    rc = _rc(tiny, tmp_path, "nocal", calib_ids=None)
    logs: list[str] = []
    summary = run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert summary["feasible"] and rc.qad is None
    assert any("falls back to PTQ" in ln for ln in logs)
    qcfg = json.loads((rc.dst / "config.json").read_text())["quantization_config"]
    assert "qad" not in qcfg and qcfg["calibrated"] is False


def test_cli_runs_qad_by_default_and_no_qad_disables(tiny, tmp_path, monkeypatch):
    import hi2.__main__ as cli

    seen = {}
    monkeypatch.setattr(cli, "run", lambda rc: seen.setdefault("rc", rc) and {})
    monkeypatch.setattr(cli, "resolve_devices", lambda spec: ["cpu"])
    calib = tmp_path / "c.txt"
    calib.write_text("x")
    cli.main(["quantize", str(tiny[0]), str(tmp_path / "o"), "--calib", str(calib)])
    rc = seen.pop("rc")
    assert rc.qad is not None and rc.sensitivity == "probe"
    cli.main(
        [
            "quantize",
            str(tiny[0]),
            str(tmp_path / "o"),
            "--calib",
            str(calib),
            "--no-qad",
        ]
    )
    rc = seen.pop("rc")
    assert rc.qad is None and rc.sensitivity == "none"
    cli.main(["quantize", str(tiny[0]), str(tmp_path / "o"), "--qad-budget", "0.4"])
    assert seen.pop("rc").qad.budget == 0.4


def test_qad_with_too_few_sequences_fails_before_work(tiny, tmp_path):
    from hi2.pipeline import run

    qc = make_config("fast", {"steps": 3, "samples": 20, "val": 2})
    rc = _rc(tiny, tmp_path, "few", qad=qc)
    with pytest.raises(SystemExit, match="need 22"):
        run(rc, log=lambda *_: None)
    assert not (rc.workdir / "phase1.done").exists()


class _CharTok:
    """One token per character (code point), enough for the window sampler."""

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        class _Out:
            pass

        o = _Out()
        o.input_ids = torch.tensor([[ord(c) for c in text]])
        return o


def test_window_sampling_random_and_balanced(tmp_path):
    from hi2.pipeline import sample_calibration_windows

    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("a" * 4000)
    b.write_text("b" * 400)
    tok = _CharTok()
    w = sample_calibration_windows([a, b], tok, 8, 40, seed=1)
    assert w.shape == (8, 40)
    wb = sample_calibration_windows([a, b], tok, 8, 40, seed=1, balanced=True)
    assert (wb[:, 0] == ord("b")).sum() == 4 and (wb[:, 0] == ord("a")).sum() == 4
    with pytest.raises(SystemExit, match="windows"):
        sample_calibration_windows([a, b], tok, 40, 40, balanced=True)


def test_ptq_fallback_is_kept_when_training_loses():
    layer = _MLP()
    xs = [_x(128, 128, seed=20 + i).unsqueeze(0) for i in range(6)]
    with torch.no_grad():
        ys = [layer(x) for x in xs]
    fq, fb, t0s = {}, {}, {}
    acts = {"up": torch.cat(xs).reshape(-1, 128)}
    with torch.no_grad():
        acts["down"] = torch.nn.functional.silu(layer.up(torch.cat(xs))).reshape(
            -1, 256
        )
    for name in ("up", "down"):
        w = layer.get_submodule(name).weight.detach().float()
        t0, comp = _ptq(w, acts[name])
        t0s[name] = t0
        fq[name] = FakeQuantLinear(w, t0, comp, "weights")
        fb[name] = FakeQuantLinear(w, t0, comp, "ptq")
    # a destructive learning rate: training cannot beat the PTQ encoding
    qc = make_config(
        "default", {"steps": 3, "batch": 1, "lr_w": 10.0, "lr_p": 50.0, "val": 2}
    )
    info = train_layer(
        layer,
        fq,
        xs,
        ys,
        [0, 1, 2, 3],
        [4, 5],
        {},
        qc,
        torch.device("cpu"),
        lambda *_: None,
        "",
        fb,
    )
    assert info["kept"] == "ptq" and info["val_after"] == info["val_before"]
    for name, m in fq.items():
        assert torch.equal(dequantize_stored(m.export()), dequantize_stored(t0s[name]))


def test_step_units_train_and_keep_exactness():
    torch.manual_seed(0)
    layer = _MLP()
    xs = [_x(256, 128, seed=30 + i).unsqueeze(0) for i in range(8)]
    with torch.no_grad():
        ys = [layer(x) for x in xs]
    fq = {}
    acts = {"up": torch.cat(xs).reshape(-1, 128)}
    with torch.no_grad():
        acts["down"] = torch.nn.functional.silu(layer.up(torch.cat(xs))).reshape(
            -1, 256
        )
    for name in ("up", "down"):
        w = layer.get_submodule(name).weight.detach().float()
        t0, comp = _ptq(w, acts[name])
        fq[name] = FakeQuantLinear(w, t0, comp, "weights", "step")
        # step units: the stored-basis values are the latent times the unit
        assert torch.allclose(
            fq[name].latent * fq[name].unit, fq[name].latent.detach() * fq[name].unit
        )
    qc = make_config(
        "default",
        {
            "steps": 40,
            "batch": 2,
            "lr_w": 0.05,
            "lr_p": 0.1,
            "val": 2,
            "latent_units": "step",
        },
    )
    info = train_layer(
        layer,
        fq,
        xs,
        ys,
        list(range(6)),
        [6, 7],
        {},
        qc,
        torch.device("cpu"),
        lambda *_: None,
    )
    assert info["val_after"] < info["val_before"]
    for m in fq.values():
        assert torch.equal(dequantize_stored(m.export()), m.stored(False).double())


def _ckpt_bytes(dst):
    return {
        p.name: p.read_bytes()
        for p in sorted(dst.iterdir())
        if p.suffix == ".safetensors"
    }


def test_qad_with_cache_equals_qad_without(tiny, tmp_path):
    """The cache only moves where the statistics live: a cached QAD run (statistics,
    Hessians and tables in the cache, batched PTQ-init encode) writes the same
    checkpoint as the workdir layout."""
    from hi2.pipeline import run

    plain = _rc(tiny, tmp_path, "plain")
    cached = _rc(tiny, tmp_path, "cached", cache_dir=tmp_path / "cache")
    run(plain, log=lambda *_: None)
    run(cached, log=lambda *_: None)
    assert not (cached.workdir / "hessian").exists()
    assert _ckpt_bytes(plain.dst) == _ckpt_bytes(cached.dst)


def test_qad_multi_ratio_runs_one_qad_per_output(tiny, tmp_path):
    from hi2.pipeline import output_dir, run

    logs: list[str] = []
    rc = _rc(
        tiny,
        tmp_path,
        "m",
        target_bpw=None,
        max_error_ratios=[2.0, 6.0],
        cache_dir=tmp_path / "cache",
    )
    out = run(rc, log=logs.append)
    assert len(out["outputs"]) == 2
    assert any("QAD runs once per output" in m for m in logs)
    for r in (2.0, 6.0):
        assert (rc.workdir / f"qad-r{r:g}").is_dir()
        single = _rc(
            tiny,
            tmp_path,
            f"s{r:g}",
            target_bpw=None,
            max_error_ratio=r,
            cache_dir=tmp_path / "cache",
        )
        run(single, log=lambda *_: None)
        assert _ckpt_bytes(single.dst) == _ckpt_bytes(output_dir(rc.dst, r))


def test_propagate_feeds_bf16_buffers_to_an_fp32_layer():
    from torch import nn

    from hi2.executor import Executor
    from hi2.qad import _propagate

    torch.manual_seed(0)
    layer = nn.Linear(16, 16).float()
    src = [torch.randn(1, 4, 16).bfloat16() for _ in range(3)]
    dst = [torch.zeros(1, 4, 16, dtype=torch.bfloat16) for _ in range(3)]
    _propagate(layer, Executor(["cpu"]), src, {}, dst)
    for s, d in zip(src, dst):
        assert torch.allclose(d.float(), layer(s.float()).bfloat16().float(), atol=2e-2)


def test_buffer_dtype_follows_the_available_ram(monkeypatch):
    import hi2.qad as q

    qc = make_config("default")
    need_fp32 = 4 * 264 * 2048 * 5120 * 4  # the 27B: three buffers and a transient set
    say = lambda free: monkeypatch.setattr(q, "_avail_ram", lambda: free)  # noqa: E731
    say(4 * need_fp32)
    assert q.buffer_dtype(qc, 264, 2048, 5120, lambda *_: None) == torch.float32
    say(int(1.5 * need_fp32))  # fp32 would take more than half of it, bf16 fits
    assert q.buffer_dtype(qc, 264, 2048, 5120, lambda *_: None) == torch.bfloat16
    say(need_fp32 // 4)
    with pytest.raises(SystemExit, match="RAM"):
        q.buffer_dtype(qc, 264, 2048, 5120, lambda *_: None)
    qc = make_config("default", {"buffer_dtype": "float32"})
    assert q.buffer_dtype(qc, 264, 2048, 5120, lambda *_: None) == torch.float32


def test_avail_ram_reads_memavailable():
    from hi2.qad import _avail_ram

    assert _avail_ram() > 0


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs for the data-parallel step"
)
def test_data_parallel_step_matches_the_single_device_run():
    def run(devices):
        torch.manual_seed(0)
        layer = _MLP()
        xs = [_x(256, 128, seed=10 + i).unsqueeze(0) for i in range(12)]
        with torch.no_grad():
            ys = [layer(x) for x in xs]
        xs_s = [x + 0.02 * x.std() * torch.randn_like(x) for x in xs]
        fq = {}
        for name in ("up", "down"):
            w = layer.get_submodule(name).weight.detach().float()
            inp = torch.cat(xs).reshape(-1, 128)
            if name == "down":
                with torch.no_grad():
                    inp = torch.nn.functional.silu(layer.up(torch.cat(xs))).reshape(-1, 256)
            t0, comp = _ptq(w, inp)
            fq[name] = FakeQuantLinear(w, t0, comp, "ptq")
        qc = make_config(
            "default",
            {"steps": 40, "batch": 4, "lr_w": 1e-3, "lr_p": 0.1, "val": 2, "final_kl": 0},
        )
        info = train_layer(
            layer, fq, xs_s, ys, list(range(10)), [10, 11], {}, qc, devices[0],
            lambda *_: None, replicas=devices,
        )
        return info, fq

    one, _ = run([torch.device("cuda:0")])
    two, fq2 = run([torch.device("cuda:0"), torch.device("cuda:1")])
    assert two["val_after"] < 0.95 * two["val_before"]
    assert two["val_after"] == pytest.approx(one["val_after"], rel=0.05)
    for m in fq2.values():
        assert torch.equal(dequantize_stored(m.export()), m.stored(False).double().cpu())
