# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Phase 1 and the teacher pass of a looped model (``hi2.looped``), CPU only.

The fork's tiny Inkling-2 checkpoint (``make_tiny_checkpoint.py``: six layers, layers 2-3
run twice per forward, eight experts, top-2) is the model: its decoder layers need
``block_states`` and, in the second pass, ``messages``, so the generic layer loop cannot
feed them. Set ``INKLING2_TOOLS`` to the fork's ``tools/rdna/inkling2`` directory; the
sibling worktree is found by itself."""

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

from hi2.calibrate import Config  # noqa: E402
from hi2.checkpoint import make_resolver, source_map  # noqa: E402
from hi2.executor import Executor  # noqa: E402
from hi2.hessian import HessianStats  # noqa: E402
from hi2.layerwise import WorkdirSink, calibrate_layerwise  # noqa: E402
from hi2.looped import (  # noqa: E402
    Call,
    is_looped,
    is_looped_source,
    record_calls,
    schedule,
    teacher_head_inputs,
)
from hi2.pipeline import RunConfig, _load_model, run  # noqa: E402
from hi2.stats import ColumnStats  # noqa: E402

N_SEQ, T = 4, 12
EXPERTS = r"\.mlp\.experts\."


def _tools_dir() -> Path | None:
    cands = (
        [Path(os.environ["INKLING2_TOOLS"])] if os.environ.get("INKLING2_TOOLS") else []
    )
    cands.append(
        Path(__file__).resolve().parents[2] / "vllm-inkling2-merge/tools/rdna/inkling2"
    )
    return next((c for c in cands if (c / "make_tiny_checkpoint.py").exists()), None)


def _import(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    make = _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py")
    path = tmp_path_factory.mktemp("looped_tiny")
    make.write_checkpoint(path, make.TinySpec(num_hidden_layers=6), seed=11)
    return path


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 50, (N_SEQ, T), generator=torch.Generator().manual_seed(5))


def _rc(src, tmp_path, **kw):
    return RunConfig(
        src=src,
        dst=tmp_path / "out",
        workdir=tmp_path / "work",
        include=EXPERTS,
        devices=["cpu"],
        compute_dtype="float32",
        target_bpw=3.5,
        cfg=Config(effort="fast"),
        **kw,
    )


@pytest.fixture(scope="module")
def model(src, tmp_path_factory):
    """The tiny model in fp32 with its experts unfused, as phase 1 loads it."""
    return _load_model(
        _rc(src, tmp_path_factory.mktemp("looped_model")), lambda *_: None
    )


def test_only_the_inkling_model_type_takes_the_program(src):
    assert is_looped_source(src)
    assert not is_looped(SimpleNamespace(config=SimpleNamespace(model_type="llama")))
    assert not is_looped(nn.Linear(2, 2))


def test_the_schedule_is_what_the_model_calls(model, ids):
    from hi2.layerwise import find_layers

    path, layers = find_layers(model)
    text = model.get_submodule(path.rsplit(".", 1)[0])
    cfg = text.config
    assert (cfg.loop_layer_index_start, cfg.loop_layer_index_end) == (2, 4)
    sched = schedule(cfg)
    assert [(c.layer, c.messages) for c in sched] == [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (2, True),
        (3, True),
        (4, False),
        (5, False),
    ]
    calls = record_calls(model, layers, text, ids)
    assert [(c.layer, c.messages) for c in calls] == [
        (c.layer, c.messages) for c in sched
    ]
    assert all(isinstance(c, Call) for c in calls)
    # what the model gave each call: the pass offset of a loop layer's cache, the masks
    assert [("cache_offset" in c.kwargs) for c in calls] == [c.messages for c in calls]
    assert all(c.kwargs["position_ids"].shape == (1, T) for c in calls)
    assert all("attention_mask" in c.kwargs for c in calls)
    assert all(layers[i] is layer for i, layer in enumerate(layers))  # restored


@pytest.mark.parametrize(("seqs", "atol"), [("1", 1e-5), ("8", 2e-4)])
def test_the_program_reproduces_the_models_head_input(
    model, ids, monkeypatch, seqs, atol
):
    """The replay of ``Inkling2LoopTextModel.forward`` (block states, the messages of the
    second pass, the final attention residual and norm) gives the LM head's input of the
    model's own forward, sequence by sequence."""
    from hi2.finalkl import _head_input

    monkeypatch.setenv("HI2_LOOP_SEQS", seqs)
    with torch.no_grad():
        want = torch.stack([_head_input(model, ids[i : i + 1]) for i in range(N_SEQ)])
    got = teacher_head_inputs(model, ids, Executor("cpu"), log=lambda *_: None)
    assert got.shape == want.shape == (N_SEQ, T, model.config.text_config.hidden_size)
    assert torch.allclose(got, want.float(), rtol=1e-4, atol=atol), (
        (got - want).abs().max()
    )


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest difference, relative to the largest magnitude (sums of squares have
    entries near zero, where an element-wise relative tolerance means nothing)."""
    scale = max(float(b.double().abs().max()), 1e-30)
    return float((a.double() - b.double()).abs().max()) / scale


def _close(a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    return _gap(a, b) <= tol


class _Rec(WorkdirSink):
    """The sink of phase 1, keeping what it is handed."""

    def __init__(self, workdir):
        super().__init__(workdir)
        self.col: dict[str, tuple[list[str], ColumnStats | None]] = {}
        self.sigma: dict[str, tuple[torch.Tensor, int]] = {}

    def stats(self, leader, members, st):
        super().stats(leader, members, st)
        self.col[leader] = (members, st)

    def hessians(self, lead, hess, linears):
        for ld, h in hess.items():
            self.sigma[linears[ld]] = (h.S.clone(), h.n)


def _reference(model, ids, resolve):
    """Per-expert-Linear statistics of whole-model forwards: a hook on every expert
    Linear (the unfused experts the model runs), one sequence at a time. The loop
    layers' hooks fire in both passes, so both passes are in the same accumulator."""
    col: dict[str, ColumnStats] = {}
    hess: dict[str, HessianStats] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".mlp.experts." in name:
            ckpt = resolve(f"{name}.weight")
            col[ckpt] = ColumnStats(mod.in_features, torch.device("cpu"))
            hess[ckpt] = HessianStats(mod.in_features, torch.device("cpu"))

            def hook(_m, args, ckpt=ckpt):
                col[ckpt].update(args[0].detach())
                hess[ckpt].update(args[0].detach())

            hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i : i + 1], use_cache=False)
    for h in hooks:
        h.remove()
    return col, hess


@pytest.mark.parametrize(("seqs", "tol"), [("1", 1e-6), ("8", 1e-4)])
def test_calibration_statistics_equal_whole_model_forward_hooks(
    src, model, ids, tmp_path, monkeypatch, seqs, tol
):
    """The gate of the layer program: its per-Linear statistics (the column moments and
    the Hessians) are those of hooks on the unfused expert Linears in whole-model
    forwards, with the passes of the loop layers pooled into one accumulator per
    tensor."""
    monkeypatch.setenv("HI2_LOOP_SEQS", seqs)
    resolve = make_resolver(source_map(src))
    want_col, want_hess = _reference(model, ids, resolve)
    sink = _Rec(tmp_path / "work")
    import re

    cfg = Config(effort="fast", expert_min_tokens=0.0)  # raw sums: no top-up
    tables = calibrate_layerwise(
        model,
        ids,
        cfg,
        Executor("cpu"),
        tmp_path / "work",
        re.compile(EXPERTS),
        re.compile(r"$^"),
        log=lambda *_: None,
        resolve=resolve,
        sink=sink,
    )
    assert set(tables) == set(want_col) and len(tables) == 6 * 8 * 3
    seen = {m for members, _ in sink.col.values() for m in members}
    assert seen == set(want_col)
    for members, st in sink.col.values():
        for m in members:  # gate and up share the one accumulator of their input
            ref = want_col[m]
            if st is None:  # an expert no token reached: data-free
                assert ref.n == 0, m
                continue
            assert st.n == ref.n, m
            assert _close(st.S, ref.S, tol), (m, _gap(st.S, ref.S))
    assert set(sink.sigma) == set(sink.col)
    for ld, (members, _) in sink.col.items():
        s, n = sink.sigma[ld]
        for m in members:
            assert n == want_hess[m].n, m
            assert _close(s, want_hess[m].S, tol), (m, _gap(s, want_hess[m].S))
    # every token reaches top-k experts in every layer call; a loop layer is called twice
    total = dict.fromkeys(range(6), 0)
    for ckpt, ref in want_col.items():
        if ".gate_proj." in ckpt:
            total[int(ckpt.split(".layers.")[1].split(".")[0])] += ref.n
    assert total[0] == total[1] == total[4] == total[5] == N_SEQ * T * 2
    assert total[2] == total[3] == 2 * N_SEQ * T * 2


def test_a_stopped_program_tabulates_the_layers_it_was_asked_for(
    src, model, ids, tmp_path
):
    import re

    resolve = make_resolver(source_map(src))
    want_col, _ = _reference(model, ids, resolve)
    sink = _Rec(tmp_path / "work")
    tables = calibrate_layerwise(
        model,
        ids,
        Config(effort="fast", expert_min_tokens=0.0),
        Executor("cpu"),
        tmp_path / "work",
        re.compile(EXPERTS),
        re.compile(r"$^"),
        limit_layers=3,
        log=lambda *_: None,
        resolve=resolve,
        sink=sink,
    )
    assert {n.split(".layers.")[1].split(".")[0] for n in tables} == {"0", "1", "2"}
    # layer 2 is a loop layer: its statistics hold both of its calls
    for members, st in sink.col.values():
        for m in members:
            assert (0 if st is None else st.n) == want_col[m].n, m


def test_a_calibrated_run_completes_from_the_stock_layout(src, ids, tmp_path):
    """``hi2 quantize`` with calibration on the looped model: phase 1 through the
    program, allocation, encoding and the checkpoint (the run failed with a TypeError for
    the missing ``block_states`` before)."""
    from hi2.checkpoint import validate_checkpoint

    rc = _rc(src, tmp_path, calib_ids=ids, calib_samples=N_SEQ, seq_len=T)
    summary = run(rc, log=lambda *_: None)
    assert summary["tensors"] == 6 * 8 * 3
    assert (rc.workdir / "hessian").is_dir() or (rc.workdir / "stats").is_dir()
    validate_checkpoint(rc.dst)


# --------------------- the loop structure of the real model: three passes, blocks of layers


@pytest.fixture(scope="module")
def src3(tmp_path_factory):
    """Twelve layers in blocks of three, layers 3-8 looped three times (Inkling-2-Mini:
    24 layers in blocks of six, layers 6-17 three times): the third pass takes its
    messages from the second, which the six-layer model with two passes never does."""
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    make = _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py")
    path = tmp_path_factory.mktemp("looped_tiny3")
    spec = make.TinySpec(
        num_hidden_layers=12,
        n_layers_per_attn_res_block=3,
        loop_start=3,
        loop_end=9,
        loop_num_passes=3,
    )
    make.write_checkpoint(path, spec, seed=13)
    return path


@pytest.fixture(scope="module")
def model3(src3, tmp_path_factory):
    return _load_model(
        _rc(src3, tmp_path_factory.mktemp("looped_model3")), lambda *_: None
    )


def test_three_passes_are_what_the_model_calls(model3, ids):
    from hi2.layerwise import find_layers

    path, layers = find_layers(model3)
    text = model3.get_submodule(path.rsplit(".", 1)[0])
    sched = schedule(text.config)
    assert len(sched) == 9 + 2 * 6 + 3  # the first pass and the layers before the end
    assert [c.messages for c in sched] == [False] * 9 + [True] * 12 + [False] * 3
    calls = record_calls(model3, layers, text, ids)
    assert [(c.layer, c.messages) for c in calls] == [
        (c.layer, c.messages) for c in sched
    ]
    assert [("cache_offset" in c.kwargs) for c in calls] == [c.messages for c in calls]


def test_three_passes_reproduce_the_models_head_input(model3, ids, monkeypatch):
    from hi2.finalkl import _head_input

    monkeypatch.setenv("HI2_LOOP_SEQS", "1")
    with torch.no_grad():
        want = torch.stack([_head_input(model3, ids[i : i + 1]) for i in range(N_SEQ)])
    got = teacher_head_inputs(model3, ids, Executor("cpu"), log=lambda *_: None)
    assert torch.allclose(got, want.float(), rtol=1e-4, atol=1e-4), (
        (got - want).abs().max()
    )


def test_three_passes_statistics_equal_whole_model_forward_hooks(
    src3, model3, ids, tmp_path, monkeypatch
):
    import re

    monkeypatch.setenv("HI2_LOOP_SEQS", "1")
    resolve = make_resolver(source_map(src3))
    want_col, want_hess = _reference(model3, ids, resolve)
    sink = _Rec(tmp_path / "work")
    tables = calibrate_layerwise(
        model3,
        ids,
        Config(effort="fast", expert_min_tokens=0.0),
        Executor("cpu"),
        tmp_path / "work",
        re.compile(EXPERTS),
        re.compile(r"$^"),
        log=lambda *_: None,
        resolve=resolve,
        sink=sink,
    )
    assert set(tables) == set(want_col) and len(tables) == 12 * 8 * 3
    for members, st in sink.col.values():
        for m in members:
            ref = want_col[m]
            if st is None:
                assert ref.n == 0, m
                continue
            assert st.n == ref.n and _close(st.S, ref.S, 1e-6), (m, _gap(st.S, ref.S))
    for ld, (members, _) in sink.col.items():
        s, n = sink.sigma[ld]
        for m in members:
            assert n == want_hess[m].n and _close(s, want_hess[m].S, 1e-6), m
    total = dict.fromkeys(range(12), 0)
    for ckpt, ref in want_col.items():
        if ".gate_proj." in ckpt:
            total[int(ckpt.split(".layers.")[1].split(".")[0])] += ref.n
    for layer, n in total.items():  # a layer of the loop is called once per pass
        assert n == N_SEQ * T * 2 * (3 if 3 <= layer < 9 else 1), layer


# ---------------------------------------------------------------- the preset and its guards


def test_the_include_preset_is_the_routed_experts_of_a_looped_model(src, tmp_path):
    """``--include`` omitted: Inkling-2 quantizes its routed experts (attention, the
    controllers, sinks, convolutions and the vision tower stay dense); an explicit
    ``--include`` wins, and no other model gets a default."""
    from hi2.looped import DEFAULT_INCLUDE, default_include
    from hi2.service import resolve_config

    rc = _rc(src, tmp_path)
    rc.include = None
    resolved = resolve_config(rc, check_ownership=False)
    assert resolved.include == DEFAULT_INCLUDE == EXPERTS
    rc.include = r"\.self_attn\."
    assert resolve_config(rc, check_ownership=False).include == r"\.self_attn\."
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text('{"model_type": "qwen2"}')
    assert default_include(other) is None
    assert default_include(tmp_path / "nothing") is None


def test_a_looped_model_refuses_what_needs_a_layers_own_input(src, tmp_path):
    from hi2.pipeline import _check_looped
    from hi2.qad import make_config

    rc = _rc(src, tmp_path)
    rc.cfg = Config(effort="fast", sequential=True)
    with pytest.raises(SystemExit, match="sequential"):
        _check_looped(rc, lambda *_: None)
    rc = _rc(src, tmp_path, qad=make_config("fast", {"layers": "0-3"}))
    with pytest.raises(SystemExit, match="layer-wise QAD"):
        _check_looped(rc, lambda *_: None)
    rc = _rc(src, tmp_path, qad=make_config("fast"))
    notes: list[str] = []
    _check_looped(rc, notes.append)
    assert rc.qad.layers == "none" and notes  # PTQ per layer, then the final KL


# ---------------------------------------------------------------- the final logit-KL phase


def test_the_logit_softcap_is_part_of_the_kl_loss(model):
    """The model's logits are ``softcap(lm_head(h).float())``: a KL on the raw head's logits
    is another loss (the cap bounds them by ``final_logit_softcapping``)."""
    from hi2.finalkl import _kl_loss, _logits_fn

    cap = model.logit_softcap.softcap
    g = torch.Generator().manual_seed(2)
    hs = torch.randn(7, model.config.text_config.hidden_size, generator=g) * 14
    ht = torch.randn(7, model.config.text_config.hidden_size, generator=g) * 14
    lm = _logits_fn(model)
    with torch.no_grad():
        raw = model.lm_head(hs).float()
        assert raw.abs().max() > cap  # the cap bites on these inputs
        assert torch.equal(lm(hs), torch.tanh(raw / cap) * cap)
        want = torch.nn.functional.kl_div(
            torch.log_softmax(torch.tanh(model.lm_head(hs).float() / cap) * cap, -1),
            torch.log_softmax(torch.tanh(model.lm_head(ht).float() / cap) * cap, -1),
            log_target=True,
            reduction="batchmean",
        )
        got = _kl_loss(lm, hs, ht, grad=False)
        plain = _kl_loss(model.lm_head, hs, ht, grad=False)
    assert got == pytest.approx(float(want), rel=1e-5)
    assert abs(got - plain) > 1e-3 * plain  # the uncapped loss is a different number


def test_a_half_precision_head_takes_the_float32_loss_inputs(model):
    """The loss hands the head float32 hidden states (the teacher's head inputs are kept in
    float32): a float16 head takes them in its own dtype (the GPU smoke died with ``expected
    mat1 and mat2 to have the same dtype`` before), the gradient reaches the float32 input."""
    import math

    from hi2.finalkl import _kl_loss, _logits_fn

    cap = model.logit_softcap.softcap
    head = nn.Linear(16, 40).half()
    plain = nn.Module()
    plain.get_output_embeddings = lambda: head
    plain.logit_softcap = model.logit_softcap
    lm = _logits_fn(plain)
    g = torch.Generator().manual_seed(4)
    hs, ht = torch.randn(5, 16, generator=g), torch.randn(5, 16, generator=g)
    with torch.no_grad():
        assert torch.equal(lm(hs), torch.tanh(head(hs.half()).float() / cap) * cap)
        assert math.isfinite(_kl_loss(lm, hs, ht, grad=False))
    hs.requires_grad_(True)
    _kl_loss(lm, hs, ht)
    assert hs.grad is not None and hs.grad.dtype == torch.float32
    assert torch.isfinite(hs.grad).all() and hs.grad.abs().max() > 0


def test_a_model_without_a_soft_cap_keeps_its_head(model):
    from hi2.finalkl import _logits_fn

    plain = nn.Module()
    plain.get_output_embeddings = lambda: model.lm_head
    assert _logits_fn(plain) is model.lm_head
    plain.logit_softcap = 30.0  # a number is not the model's soft-cap op
    assert _logits_fn(plain) is model.lm_head


def test_shard_layers_places_what_reads_the_layers_outputs(src, tmp_path, monkeypatch):
    """Layers 0-2 on the first device, 3-5 on the last, with the final attention residual
    (it reads the last layers' states) beside the head and the final norm, and the
    messages of a pass beside the embeddings; both move their inputs to their device.
    The moves are recorded, not made: a CPU box has one device."""
    import hi2.sensitivity as sens

    placed: dict[int, torch.device] = {}
    monkeypatch.setattr(
        sens, "_move_own", lambda mod, dev: placed.__setitem__(id(mod), dev)
    )
    monkeypatch.setattr(
        sens,
        "module_to_device",
        lambda mod, dev, *a, **k: placed.__setitem__(id(mod), dev) or mod,
    )
    import hi2.layerwise as lw

    moved: list[
        torch.device
    ] = []  # shard_layers' hooks call to_device (bound when it runs)
    monkeypatch.setattr(lw, "to_device", lambda obj, dev: (moved.append(dev), obj)[1])
    m = _load_model(_rc(src, tmp_path), lambda *_: None)
    first, last = torch.device("cpu"), torch.device("meta")
    handles = sens.shard_layers(m, [first, last])
    text = m.model.language_model
    assert [placed[id(layer)] for layer in text.layers] == [first] * 3 + [last] * 3
    for tail in (text.final_input, text.norm, m.lm_head):
        assert placed[id(tail)] == last
    for head in (text.embed_tokens, text.message_input):
        assert placed[id(head)] == first
    assert len(handles) == 6 + 2
    # the hooks bring the inputs over: each module's pre-hook moves its inputs to the
    # device it is placed on (the call fails afterwards on the toy arguments: only the
    # move matters)
    h = text.config.hidden_size
    states = (torch.zeros(1, 2, 1, h), torch.zeros(1, 2, h))

    def moves_to(module, *args):
        moved.clear()
        with contextlib.suppress(Exception):
            module(*args)
        return set(moved)

    assert moves_to(text.final_input, *states) == {last}
    assert moves_to(text.message_input, *states, 0) == {first}
    assert moves_to(text.layers[0], states[1], states[0]) == {first}
    assert moves_to(text.layers[5], states[1], states[0]) == {last}
    for handle in handles:
        handle.remove()


def test_expert_checkpointing_changes_the_memory_not_the_values(model):
    """With ``checkpoint_experts`` the weight graph of every expert Linear is rebuilt in
    the backward instead of being kept: the same outputs and gradients, far fewer saved
    bytes."""
    from hi2.moe import UnfusedExperts

    experts = next(m for m in model.modules() if isinstance(m, UnfusedExperts))
    h = experts[0].gate_proj.in_features
    g = torch.Generator().manual_seed(3)
    x = torch.randn(10, h, generator=g)
    n_experts = sum(hasattr(m, "gate_proj") for m in experts)  # act_fn is a child too
    idx = torch.randint(0, n_experts, (10, 2), generator=g)
    w = torch.rand(10, 2, generator=g)

    def run(flag):
        experts.checkpoint_experts = flag
        saved = [0]

        def pack(t):
            saved[0] += t.numel() * t.element_size()
            return t

        params = [
            lin.weight
            for mlp in experts
            if hasattr(mlp, "gate_proj")  # the ModuleList also lists act_fn
            for lin in (mlp.gate_proj, mlp.up_proj)
        ]
        for p in params:
            p.requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = experts(x.clone().requires_grad_(True), idx, w)
        out.square().sum().backward()
        grads = [p.grad.clone() for p in params if p.grad is not None]
        for p in params:
            p.grad = None
            p.requires_grad_(False)
        experts.checkpoint_experts = False
        return out.detach(), grads, saved[0]

    out0, g0, kept0 = run(False)
    out1, g1, kept1 = run(True)
    assert torch.equal(out0, out1)
    assert len(g0) == len(g1) > 0 and all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert kept1 < kept0


def _spy_final_kl(monkeypatch):
    """Wrap ``hi2.finalkl.final_kl``: the student's footprint before, the measured bytes after."""
    import hi2.finalkl as fk

    real = fk.final_kl
    seen: dict = {}

    def spy(
        model, ids, train_idx, val_idx, tensors, dense, qc, devs, resolve, *a, **kw
    ):
        seen["foot"] = fk.student_footprint(
            model, tensors, resolve, qc, ids.shape[1], 1
        )
        seen["tensors"] = tensors
        calls: list = []
        hooks = [
            layer.register_forward_pre_hook(
                lambda _m, args, i=i: calls.append(
                    (
                        i,
                        sum(
                            t.numel() * t.element_size()
                            for t in args[:3]
                            if torch.is_tensor(t)
                        ),
                    )
                )
            )
            for i, layer in enumerate(model.model.language_model.layers)
        ]
        with torch.no_grad():
            fk._head_input(model, ids[:1])
        for h in hooks:
            h.remove()
        seen["calls"] = calls
        out = real(
            model, ids, train_idx, val_idx, tensors, dense, qc, devs, resolve, *a, **kw
        )
        seen["model"] = model
        return out

    monkeypatch.setattr(fk, "final_kl", spy)
    return seen


def _qad(**kw):
    from hi2.qad import make_config

    return make_config("fast", {"steps": 2, "samples": 3, "val": 1, "batch": 2, **kw})


def test_the_teachers_head_inputs_are_kept_for_the_same_token_ids(src, ids, tmp_path):
    """Every allocation of a run trains on the same windows: the head inputs of the layer
    program are kept in the workdir and reused for the same source, dtype and token ids,
    and recomputed when any of them changes."""
    from hi2.pipeline import _teacher_head_inputs

    rc = _rc(src, tmp_path, calib_ids=ids)
    rc.workdir.mkdir(parents=True)
    model = _load_model(rc, lambda *_: None)
    ex = Executor("cpu")
    notes: list[str] = []
    first = _teacher_head_inputs(rc, ex, model, ids, notes.append)
    assert (rc.workdir / "teacher_head_inputs.pt").exists()
    assert any("layer program" in n for n in notes) and first.shape[0] == N_SEQ
    notes.clear()
    again = _teacher_head_inputs(rc, ex, model, ids, notes.append)
    assert any("reusing" in n for n in notes) and torch.equal(first, again)
    notes.clear()
    other = ids.clone()
    other[0, 0] = (other[0, 0] + 1) % 50
    changed = _teacher_head_inputs(rc, ex, model, other, notes.append)
    assert any("layer program" in n for n in notes)  # not reused: another token id
    assert not torch.equal(first[0], changed[0])  # the changed window moved ...
    # ... and the others only by the float noise of the batched expert GEMMs (4e-5)
    assert torch.allclose(first[1:], changed[1:], atol=1e-3)


def test_a_cpu_final_kl_of_two_steps_completes_on_the_looped_model(
    src, model, ids, tmp_path, monkeypatch
):
    """The gate of the final-KL item: ``hi2 quantize`` with calibration and ``--qad`` on the
    looped model, on the CPU: phase 1 by the layer program, PTQ per layer, the teacher's
    head inputs by the layer program, two final-KL steps on the student, the checkpoint."""
    from hi2.checkpoint import validate_checkpoint
    from hi2.finalkl import _head_input

    seen = _spy_final_kl(monkeypatch)
    rc = _rc(
        src,
        tmp_path,
        calib_ids=ids,
        calib_samples=N_SEQ,
        seq_len=T,
        qad=_qad(final_kl=2, final_lr_p=0.5),
    )
    logs: list[str] = []
    summary = run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert summary["tensors"] == 6 * 8 * 3
    info = json.loads((rc.workdir / "final_kl.json").read_text())
    assert info["steps"] == 2 and info["teacher_seconds"] == 0  # no teacher placement
    assert info["val_after"] <= info["val_before"]
    report = json.loads((rc.workdir / "qad_report.json").read_text())
    assert len(report) == 6 and {r["kept"] for r in report} == {
        "ptq"
    }  # nothing trained
    assert any("layer program" in line for line in logs)
    # the head inputs of the program are the model's own
    rec = torch.load(rc.workdir / "teacher_head_inputs.pt", weights_only=False)
    want = torch.stack([_head_input(model, ids[i : i + 1]) for i in range(N_SEQ)])
    assert torch.allclose(rec["head_inputs"], want, rtol=1e-4, atol=2e-4)
    validate_checkpoint(rc.dst)
    assert seen["foot"]["static"]  # the footprint was asked before the phase ran
    from hi2.moe import UnfusedExperts

    experts = [m for m in seen["model"].modules() if isinstance(m, UnfusedExperts)]
    assert experts and all(
        m.checkpoint_experts for m in experts
    )  # per-expert recompute


def test_the_footprint_is_what_the_student_holds(src, ids, tmp_path, monkeypatch):
    """``student_footprint`` against the real student: the bytes of the ladder linears
    (exact), of everything the student holds besides the optimizer (within the norms of
    the vision tower, which it counts as trained), and the layer inputs it says the
    backward keeps (exact: read from the layers' own inputs)."""
    import hi2.finalkl as fk
    from hi2.looped import saved_input_bytes

    seen = _spy_final_kl(monkeypatch)
    rc = _rc(
        src,
        tmp_path,
        calib_ids=ids,
        calib_samples=N_SEQ,
        seq_len=T,
        qad=_qad(final_kl=1),
    )
    run(rc, log=lambda *_: None)
    model, foot = seen["model"], seen["foot"]
    ladders = [m for m in model.modules() if isinstance(m, fk.LadderLinear)]
    assert len(ladders) == 6 * 8 * 3
    held = sum(
        p.numel() * p.element_size() for m in ladders for p in m.parameters()
    ) + sum(b.numel() * b.element_size() for m in ladders for b in m.buffers())
    assert held == sum(
        fk.tensor_student_bytes(t, False) for t in seen["tensors"].values()
    )
    # the whole student: parameters and buffers, plus a gradient and two fp32 Adam moments
    # (12 bytes) for every trained value
    values = sum(p.numel() for m in ladders for p in (*m.lo_u, *m.step_u))
    norms = sum(
        p.numel() for n, p in model.named_parameters() if fk.is_norm_param(n, p)
    )
    total = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    predicted = sum(foot["static"])
    assert fk.ADAM_STATE_BYTES == 12
    assert predicted == pytest.approx(total + 12 * (values + norms), rel=2e-3), (
        predicted,
        total,
    )
    # the layer inputs one sequence's backward keeps
    text = model.model.language_model
    db = model.lm_head.weight.element_size()
    assert saved_input_bytes(text, T, db) == seen["calls"]


def test_the_looped_fit_leaves_room_for_the_allocators_reserve(src, model, monkeypatch):
    """The GPU smoke (two V620 of 29.98 GiB, every value with a gradient): a student the
    formula puts at 22.9 GiB reserved 27.1 GiB and the card showed 27.8 GiB used, so the
    check admits a formula need up to 80 % of the card, not the 92 % it had: a float32
    dense student (25.2 GiB by the formula) is skipped, the float16 one (22.9) runs."""
    from types import SimpleNamespace

    import hi2.finalkl as fk
    from hi2.pipeline import _looped_fits

    card = int(29.98 * 2**30)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(total_memory=card),
    )
    ex = SimpleNamespace(devices=[torch.device("cuda:0"), torch.device("cuda:1")])
    rc = SimpleNamespace(src=src, qad=None)
    notes = [0, 0]

    def fits(need_gib):
        foot = {
            "devices": [need_gib * 2**30, 1],
            "static": notes,
            "activations": notes,
            "transient": 0,
        }
        monkeypatch.setattr(fk, "student_footprint", lambda *a, **k: foot)
        return _looped_fits(rc, ex, model, {}, 2048, lambda *_: None)

    assert fits(22.9) and not fits(25.2)
    assert fits(0.79 * 29.98) and not fits(0.81 * 29.98)


# ------------------------------------------------ the GPU smoke script (experiments/)


def _smoke():
    path = (
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_looped_smoke.py"
    )
    return _import("hi2_test_inkling_looped_smoke", path)


def test_the_smoke_standin_is_a_valid_ladder_tensor():
    """The stand-in the GPU smoke trains in place of an expert has the bytes the formula
    counts for a real tensor and a decode that is finite."""
    from hi2.finalkl import LadderLinear, tensor_student_bytes

    t = _smoke().standin(128, 256, (2, 4), 32, torch.Generator().manual_seed(0))
    assert t.shape == (128, 256)
    assert 32 * t.codes.numel() == 3 * 128 * 256  # 2 and 4 bits alternate: 3.0 bits
    ll = LadderLinear(t)
    w = ll.weight()
    assert w.shape == (128, 256) and torch.isfinite(w).all() and w.abs().max() < 1.0
    held = sum(p.numel() * p.element_size() for p in ll.parameters()) + sum(
        b.numel() * b.element_size() for b in ll.buffers()
    )
    assert held == tensor_student_bytes(t, train_bytes=False)


def _smoke_argv(src, tmp_path, *extra):
    argv = ["smoke", "--src", str(src), "--transformers", str(tmp_path)]
    argv += ["--devices", "cpu", "cpu", "--compute-dtype", "float32"]
    argv += ["--seqs", "2", "--seq-len", "24", "--student-steps", "2"]
    argv += ["--workdir", str(tmp_path / "work"), "--out", str(tmp_path / "smoke.json")]
    return argv + list(extra)


def test_the_smoke_script_runs_the_student_stage_on_the_tiny_model(
    src, tmp_path, monkeypatch
):
    """``--student-steps`` on the tiny model (CPU): the stand-in student, the formula and
    two final-KL steps with the time of one; the program on the compared length equals
    the model's own forward; the monkeypatches of the stage are undone."""
    from hi2.finalkl import LadderLinear

    smoke = _smoke()
    monkeypatch.setattr(
        sys, "argv", _smoke_argv(src, tmp_path, "--compare-tokens", "12")
    )
    export, step = LadderLinear.export, torch.optim.AdamW.step
    zero = torch.optim.AdamW.zero_grad
    assert smoke.main() == 0
    res = json.loads((tmp_path / "smoke.json").read_text())
    assert res["ok"] and res["student_ok"]
    assert res["compare"]["positions"] == 12 and res["compare"]["relative"] < 1e-4
    assert res["compare_prefix_relative"] < 1e-4  # the program does not see the length
    detail = res["student_detail"]
    assert detail["standins"]["tensors"] == 6 * 8 * 3
    assert detail["standins"]["code_bits_per_weight"] == 3.0
    assert len(detail["formula"]["need_gib"]) == 2
    assert len(detail["final_kl"]["step_seconds"]) == 1  # the gap between two steps
    assert LadderLinear.export is export and torch.optim.AdamW.step is step
    assert torch.optim.AdamW.zero_grad is zero


def test_the_smoke_student_stage_alone_gives_every_value_a_gradient(
    src, tmp_path, monkeypatch
):
    """``--skip-compare`` runs the student stage alone; ``--student-full-grads`` gives every
    trained value a gradient (the memory worst case a random stand-in does not reach): all
    of them from the second step on, the first step has the ones the routing reached."""
    smoke = _smoke()
    extra = ("--skip-compare", "--student-full-grads")
    monkeypatch.setattr(sys, "argv", _smoke_argv(src, tmp_path, *extra))
    assert smoke.main() == 0
    res = json.loads((tmp_path / "smoke.json").read_text())
    assert "compare" not in res and res["ok"] and res["student_ok"]
    grads = res["student_detail"]["final_kl"]["gradient_values"]
    assert len(grads) == 2 and grads[0]["total"] > 0
    assert grads[0]["with_gradient"] <= grads[0]["total"]
    assert grads[1]["with_gradient"] == grads[1]["total"] == grads[0]["total"]
