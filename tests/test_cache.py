# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Statistics/tables/probe cache, batched multi-tensor encode, multi-output mode (CPU).

Everything here must be bit-identical to the uncached, per-tensor path."""

import json

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from hi2 import cache as C  # noqa: E402
from hi2.calibrate import (  # noqa: E402
    Config,
    EncodeRequest,
    encode_many,
    encode_with_picks_reference,
)
from hi2.executor import Executor  # noqa: E402
from hi2.pipeline import RunConfig, output_dir, run  # noqa: E402


def _save_sigma(d, name, sigma, n, fp16_above=0):
    d.mkdir(parents=True, exist_ok=True)
    torch.save(C.sigma_payload(sigma, n, fp16_above), d / C._fname(name))


QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
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
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 2 and "proj" in n:
                p.mul_(torch.exp(0.5 * torch.randn(p.shape[0], 1)))
                p.mul_(torch.exp(0.4 * torch.randn(1, p.shape[1])))
    model.save_pretrained(d, safe_serialization=True)
    ids = torch.randint(0, 256, (8, 64), generator=torch.Generator().manual_seed(1))
    return d, ids


def _rc(tiny, tmp_path, tag, cache=True, **kw):
    d, ids = tiny
    cfg = kw.pop("cfg", Config(effort="fast"))
    base = dict(
        src=d,
        dst=tmp_path / f"out-{tag}",
        workdir=tmp_path / f"work-{tag}",
        calib_ids=ids,
        target_bpw=3.0,
        calib_samples=8,
        seq_len=64,
        cfg=cfg,
        cache_dir=tmp_path / "cache" if cache else None,
    )
    base.update(kw)
    return RunConfig(**base)


def _files(dst):
    return {
        p.name: p.read_bytes()
        for p in sorted(dst.iterdir())
        if p.suffix in (".safetensors", ".json") and p.name != "hi2_report.json"
    }


def _same_checkpoint(a, b):
    fa, fb = _files(a), _files(b)
    assert fa.keys() == fb.keys()
    for k in fa:
        assert fa[k] == fb[k], k


def _no_forward(monkeypatch):
    import hi2.layerwise as lw
    import hi2.pipeline as pl

    def boom(*a, **k):
        raise AssertionError("the model must not run on a cache hit")

    monkeypatch.setattr(lw, "calibrate_layerwise", boom)
    monkeypatch.setattr(pl, "_load_model", boom)


def test_sigma_triangle_roundtrip_is_exact(tmp_path):
    x = torch.randn(300, 48)
    s = x.T @ x / 300
    s = 0.5 * (s + s.T)
    _save_sigma(tmp_path, "a.weight", s, 300)
    assert torch.equal(C.load_sigma(tmp_path / "a.weight.pt"), s)
    obj = torch.load(tmp_path / "a.weight.pt")
    assert "tri" in obj and obj["tri"].numel() == 48 * 49 // 2
    asym = s.clone()
    asym[0, 1] += 1e-3  # not symmetric: stored in full
    _save_sigma(tmp_path, "b.weight", asym, 300)
    assert torch.equal(C.load_sigma(tmp_path / "b.weight.pt"), asym)


@pytest.mark.parametrize("rounding", ["gptq", "nearest"])
def test_cached_run_equals_uncached_and_hits_on_rerun(
    tiny, tmp_path, monkeypatch, rounding
):
    cfg = Config(effort="fast", rounding=rounding)
    plain = _rc(tiny, tmp_path, "plain", cache=False, cfg=cfg)
    cached = _rc(tiny, tmp_path, "cached", cfg=cfg)
    run(plain, **QUIET)
    run(cached, **QUIET)
    _same_checkpoint(plain.dst, cached.dst)
    assert not (cached.workdir / "hessian").exists()  # Sigma lives in the cache
    ref = json.loads((cached.workdir / "cache_ref.json").read_text())
    assert (C.Path(ref["stats"]) / "complete.json").exists()
    # a second run (other workdir, other target) reuses statistics and tables
    logs = []
    _no_forward(monkeypatch)
    again = _rc(tiny, tmp_path, "again", cfg=cfg, target_bpw=3.6)
    run(again, log=logs.append)
    assert any("statistics hit" in m for m in logs) and any(
        "tables hit" in m for m in logs
    )
    plain2 = _rc(tiny, tmp_path, "plain2", cache=False, cfg=cfg, target_bpw=3.6)
    monkeypatch.undo()
    run(plain2, **QUIET)
    _same_checkpoint(plain2.dst, again.dst)


def test_new_phase1_settings_reuse_statistics_without_forward(
    tiny, tmp_path, monkeypatch
):
    run(_rc(tiny, tmp_path, "first"), **QUIET)
    cfg = Config(
        effort="fast",
        smooth=0.25,
        rotation="auto",
        candidates=[(2, 32), (4, 64), (2, 16)],
    )
    logs = []
    _no_forward(monkeypatch)
    other = _rc(tiny, tmp_path, "other", cfg=cfg)
    run(other, log=logs.append)
    assert any("statistics hit" in m for m in logs) and any(
        "tables miss" in m for m in logs
    )
    monkeypatch.undo()
    plain = _rc(tiny, tmp_path, "plain", cache=False, cfg=cfg)
    run(plain, **QUIET)
    _same_checkpoint(plain.dst, other.dst)


def test_other_calibration_tokens_miss(tiny, tmp_path):
    run(_rc(tiny, tmp_path, "a"), **QUIET)
    ids = tiny[1].clone()
    ids[0, 0] = (ids[0, 0] + 1) % 256
    logs = []
    run(_rc(tiny, tmp_path, "b", calib_ids=ids), log=logs.append)
    assert any("statistics miss" in m for m in logs)
    assert len([e for e in C.entries(tmp_path / "cache") if e["kind"] == "stats"]) == 2


def test_probe_is_cached(tiny, tmp_path, monkeypatch):
    kw = dict(sensitivity="probe", sens_seqs=2, sens_len=32)
    first = _rc(tiny, tmp_path, "p1", **kw)
    run(first, **QUIET)
    import hi2.sensitivity as sm

    monkeypatch.setattr(
        sm,
        "probe_sensitivity",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe re-ran")),
    )
    logs = []
    second = _rc(tiny, tmp_path, "p2", target_bpw=3.6, **kw)
    run(second, log=logs.append)
    assert any("sensitivity probe hit" in m for m in logs)
    assert json.loads((first.workdir / "sensitivity.json").read_text()) == json.loads(
        (second.workdir / "sensitivity.json").read_text()
    )


def test_multi_ratio_outputs_equal_single_runs(tiny, tmp_path):
    ratios = [2.0, 5.0]
    multi = _rc(tiny, tmp_path, "m", target_bpw=None, max_error_ratios=ratios)
    out = run(multi, **QUIET)
    assert set(out["outputs"]) == {str(output_dir(multi.dst, r)) for r in ratios}
    for r in ratios:
        single = _rc(
            tiny, tmp_path, f"s{r:g}", cache=False, target_bpw=None, max_error_ratio=r
        )
        run(single, **QUIET)
        _same_checkpoint(single.dst, output_dir(multi.dst, r))


@pytest.mark.parametrize("devices", [["cpu"], ["cpu", "cpu"]])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("rounding", ["gptq", "nearest"])
def test_encode_many_equals_per_tensor_reference(
    tiny, tmp_path, devices, act_order, rounding
):
    """Several tensors (two sharing an input), several devices: identical buffers and
    identical QAD initialisation values."""
    cfg = Config(effort="fast", act_order=act_order, rounding=rounding)
    rc = _rc(tiny, tmp_path, "t", cache=False, cfg=cfg)
    run(rc, **QUIET)
    from safetensors import safe_open

    from hi2.cache import StatsView
    from hi2.checkpoint import weight_map
    from hi2.stats import ColumnStats

    tables = {
        n: torch.load(p, weights_only=False)
        for n, p in ((p.stem, p) for p in (rc.workdir / "tables").iterdir())
    }
    view = StatsView(rc.workdir)
    wmap = weight_map(rc.src)
    names = [n for n in tables if "layers.1." in n]
    cfg, ex = rc.cfg, Executor(devices)
    g = torch.Generator().manual_seed(5)
    reqs, refs = [], []
    for n in names:
        with safe_open(rc.src / wmap[n], "pt") as f:
            w = f.get_tensor(n)
        t = tables[n]
        picks = torch.randint(0, len(t.candidates), (t.shape[0],), generator=g)
        st = view.stats_state(n)
        stats = ColumnStats.from_state(st) if st else None
        sigma = view.sigma(n)
        reqs.append(
            EncodeRequest(w, stats, t, picks, sigma, view.sigma_id(n), comp_out={})
        )
        co: dict = {}
        ref = encode_with_picks_reference(w, stats, t, picks, cfg, ex, sigma, co)
        refs.append((ref, co))
    assert len({r.sigma_id for r in reqs}) < len(reqs)  # q/k/v or gate/up share
    for got, (ref, co), r in zip(encode_many(reqs, cfg, ex), refs, reqs):
        assert torch.equal(r.comp_out["comp"], co["comp"])  # the QAD initialisation
        for f in (
            "codes",
            "code_off",
            "blk_params",
            "param_off",
            "row_consts",
            "row_nbits",
            "row_block",
            "had",
        ):
            assert torch.equal(getattr(got, f), getattr(ref, f)), f


def test_cache_list_and_prune(tiny, tmp_path):
    run(_rc(tiny, tmp_path, "a"), **QUIET)
    root = tmp_path / "cache"
    es = C.entries(root)
    assert {e["kind"] for e in es} == {"stats", "tables"} and all(
        e["complete"] for e in es
    )
    (root / "stats" / "deadbeef.tmp-999999999").mkdir()  # an interrupted build
    import os

    (root / "stats" / f"cafe.tmp-{os.getpid()}").mkdir()  # a build in progress
    gone = C.prune(root)
    assert [e["digest"] for e in gone] == ["deadbeef.tmp-999999999"]
    assert (root / "stats" / f"cafe.tmp-{os.getpid()}").exists()
    (root / "stats" / f"cafe.tmp-{os.getpid()}").rmdir()
    gone = C.prune(root, keep_gb=0)
    assert len(gone) == len(es) and C.entries(root) == []


def test_space_estimate_bounds_the_entry(tiny, tmp_path):
    rc = _rc(tiny, tmp_path, "a")
    run(rc, **QUIET)
    import regex as re

    from hi2.checkpoint import DEFAULT_EXCLUDE

    est = C.estimate_stats_bytes(rc.src, None, re.compile(DEFAULT_EXCLUDE))
    entry = C.Path(json.loads((rc.workdir / "cache_ref.json").read_text())["stats"])
    # triangle bytes (a CPU GEMM is not exactly symmetric, so the tiny test stores full)
    ks = [torch.load(p)["k"] for p in (entry / "sigma").iterdir()]
    assert 0 < sum(2 * k * (k + 1) for k in ks) <= est


def test_probe_prefix_replay_is_bit_identical(tiny, tmp_path):
    """Replaying the unchanged decoder layers before a unit gives the same KL as the
    full forward of the original probe."""
    from hi2.sensitivity import probe_sensitivity

    rc = _rc(tiny, tmp_path, "pr", cache=False)
    run(rc, **QUIET)
    tables = {
        p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
        for p in (rc.workdir / "tables").iterdir()
    }
    ids = tiny[1][:3, :32]
    kw = dict(log=lambda *_: None)
    full = probe_sensitivity(
        rc.src,
        tables,
        rc.workdir,
        ids,
        rc.cfg,
        Executor("cpu"),
        skip_prefix=False,
        **kw,
    )
    fast = probe_sensitivity(
        rc.src, tables, rc.workdir, ids, rc.cfg, Executor("cpu"), skip_prefix=True, **kw
    )
    assert full == fast
    batched = probe_sensitivity(
        rc.src, tables, rc.workdir, ids, rc.cfg, Executor("cpu"), batch=3, **kw
    )
    for n in full:
        assert batched[n] == pytest.approx(full[n], rel=1e-3)


def test_sigma_fp16_storage_is_opt_in(tmp_path):
    x = torch.randn(400, 80)
    s = x.T @ x / 400
    s = 0.5 * (s + s.T)
    _save_sigma(tmp_path, "a.weight", s, 400, fp16_above=64)
    got = C.load_sigma(tmp_path / "a.weight.pt")
    assert torch.equal(got, s.half().float())
    _save_sigma(tmp_path, "b.weight", s, 400, fp16_above=80)  # K not above: fp32
    assert torch.equal(C.load_sigma(tmp_path / "b.weight.pt"), s)


def test_cache_keys_can_record_another_device_count(monkeypatch):
    from types import SimpleNamespace

    import torch

    from hi2.cache import probe_key, stats_key

    cpu = torch.device("cpu")
    rc = SimpleNamespace(
        src=".",
        calib=".",
        compute_dtype="float32",
        include=None,
        exclude="x",
        limit_layers=None,
        cfg=SimpleNamespace(expert_min_tokens=2.0, effort="normal"),
        sigma_fp16_above=0,
        devices=["cpu"],
        cache_devices=0,
        sens_floor=0.05,
        sens_dtype="float32",
        expert_share_floor=0.05,
        sens_batch=1,
    )
    ids = torch.zeros(2, 4, dtype=torch.long)
    import hi2.cache as C

    monkeypatch.setattr(C, "model_fingerprint", lambda src: "m")
    one = stats_key(rc, ids, [cpu])
    four = stats_key(rc, ids, [cpu] * 4)
    assert one["n_devices"] == 1 and four["n_devices"] == 4
    rc.cache_devices = 4
    assert stats_key(rc, ids, [cpu])["n_devices"] == 4
    assert probe_key({"a": 1}, rc, ids, (2, 32))["devices"] == 4
