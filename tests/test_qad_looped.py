# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Layer-wise QAD on a looped model (``hi2.qad_looped``), CPU only.

The fork's tiny Inkling-2 checkpoint (``make_tiny_checkpoint.py``: six layers, layers 2-3
run twice) is the model; a second fixture is twelve layers in blocks of three, layers 3-8
looped three times, so the third pass takes its messages from the second. Set
``INKLING2_TOOLS`` to the fork's ``tools/rdna/inkling2`` directory; the sibling worktree
is found by itself. The teacher states, the pass messages, the group split and a full
``hi2 quantize`` are exercised; the tiny random model's KL is not a quality signal.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

from hi2.calibrate import Config  # noqa: E402
from hi2.executor import Executor  # noqa: E402
from hi2.looped import _Program, run_program  # noqa: E402
from hi2.pipeline import RunConfig, _load_model, run  # noqa: E402
from hi2.qad import _GroupBatches, make_config  # noqa: E402
from hi2.qad_looped import (  # noqa: E402
    _Samples,
    _capture,
    _group_names,
    _layout,
    _needed,
    _samples_for,
    _state_count,
)

N_SEQ, T = 8, 12
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
    path = tmp_path_factory.mktemp("qad_looped_tiny")
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
    return _load_model(
        _rc(src, tmp_path_factory.mktemp("qad_looped_model")), lambda *_: None
    )


@pytest.fixture(scope="module")
def prog(model, ids):
    return _Program(model, ids, Executor("cpu"), lambda *_: None)


@pytest.fixture(scope="module")
def src3(tmp_path_factory):
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    make = _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py")
    path = tmp_path_factory.mktemp("qad_looped_tiny3")
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
def prog3(src3, tmp_path_factory):
    model = _load_model(
        _rc(src3, tmp_path_factory.mktemp("qad_looped_model3")), lambda *_: None
    )
    ids3 = torch.randint(0, 50, (3, 8), generator=torch.Generator().manual_seed(7))
    return _Program(model, ids3, Executor("cpu"), lambda *_: None)


# ---------------------------------------------------------------- the teacher states


def test_the_capture_keeps_one_state_per_unique_call(prog, tmp_path):
    qc = make_config("fast", {"final_kl": 0})
    teacher = _capture(
        prog, list(range(6)), qc, tmp_path / "qad", "fp", lambda *_: None
    )
    layout = _layout(prog.calls, prog.layers, prog.text.config)
    inputs, outputs = _needed(layout, prog.calls, list(range(6)))
    assert len(teacher.slots) == _state_count(
        prog.calls, inputs, outputs, prog.text.config.loop_layer_index_start
    ) == 9
    # the first call's input is the captured layer-0 input, bit for bit
    for w in range(prog.n_seq):
        assert teacher.slots[teacher.inputs[0]][w] is prog.hidden[w]
    # an output the program continues with is the next call's input, not a second copy
    for ci in range(len(prog.calls) - 1):
        if prog.calls[ci + 1].layer == prog.calls[ci].layer + 1:
            assert teacher.outputs[ci] == teacher.inputs[ci + 1]
    # the selected layers' calls all have a target, and the pass start reuses the pre state
    calls_of = layout.calls_of
    for li in range(6):
        for ci in calls_of[li]:
            assert ci in teacher.outputs and ci in teacher.inputs
    assert teacher.inputs[calls_of[2][1]] == teacher.inputs[calls_of[2][0]]
    # final_kl 0: the pass stops early and does not write the final-KL head inputs
    assert not (tmp_path / "qad" / "head_inputs.pt").exists()


def test_too_many_teacher_states_fail_before_the_pass(prog, tmp_path, monkeypatch):
    monkeypatch.setenv("HI2_QAD_STATES_GIB", "0")
    qc = make_config("fast", {"final_kl": 0})
    with pytest.raises(SystemExit, match="HI2_QAD_STATES_GIB"):
        _capture(prog, list(range(6)), qc, tmp_path / "qad", "fp", lambda *_: None)


def test_the_heads_are_saved_for_the_final_kl(prog, tmp_path):
    qc = make_config("fast", {"final_kl": 2})
    qdir = tmp_path / "qad"
    _capture(prog, list(range(6)), qc, qdir, "fp", lambda *_: None)
    rec = torch.load(qdir / "head_inputs.pt", weights_only=False)
    assert rec["fingerprint"] == "fp"
    hs = rec["head_inputs"]
    assert hs.shape == (prog.n_seq, T, prog.hidden[0].shape[-1])
    assert torch.isfinite(hs).all()


# ---------------------------------------------------------------- the pass messages


def test_the_later_passes_see_the_states_of_the_pass_before(prog3, tmp_path):
    prog = prog3
    cfg = prog.text.config
    a = cfg.loop_layer_index_start
    ref: dict[int, list] = {}
    ref_bs: dict[int, list] = {}

    def call(ci, c, h, bs, messages):
        if messages is not None:
            ref[ci] = [messages[i].detach().clone() for i in range(prog.n_seq)]
            ref_bs[ci] = [s.detach().clone() for s in bs]
        out, _ = prog.run(ci, h, bs, messages)
        return out

    run_program(prog.text, prog.hidden, call)
    selected = list(range(a, cfg.loop_layer_index_end))
    qc = make_config("fast", {"final_kl": 0})
    teacher = _capture(prog, selected, qc, tmp_path / "qad", "fp", lambda *_: None)
    layout = _layout(prog.calls, prog.layers, cfg)
    li = selected[2]
    samples, *_ = _samples_for(teacher, layout, li, qc, teacher.n_seq)
    calls = _Samples(prog, teacher, layout, samples, [torch.device("cpu")])
    seen = 0
    for i, (w, ci) in enumerate(samples):
        if not prog.calls[ci].messages:
            continue
        got = calls.kwargs(i, torch.device("cpu"), torch.float32)
        want = ref[ci][w]
        assert torch.allclose(got["messages"], want, atol=1e-6), (ci, w)
        assert torch.allclose(
            got["block_states"], ref_bs[ci][w], atol=0.0
        ), (ci, w)
        wide = calls.kwargs(i, torch.device("cpu"), torch.float64)
        assert wide["messages"].dtype == torch.float64
        assert wide["block_states"].dtype == torch.float64
        seen += 1
    assert seen == 2 * prog.n_seq  # two passes with messages per window


def test_a_loop_layer_has_one_sample_per_pass_and_whole_groups(prog, tmp_path):
    qc = make_config("fast", {"final_kl": 0, "samples": 6, "val": 2})
    teacher = _capture(prog, [2], qc, tmp_path / "qad", "fp", lambda *_: None)
    layout = _layout(prog.calls, prog.layers, prog.text.config)
    samples, xs, ys, train_idx, val_idx, groups = _samples_for(
        teacher, layout, 2, qc, teacher.n_seq
    )
    assert len(samples) == 2 * teacher.n_seq
    assert groups == [w for w in range(teacher.n_seq) for _ in range(2)]
    assert set(train_idx) | set(val_idx) == set(range(len(samples)))
    assert not set(train_idx) & set(val_idx)
    gb = _GroupBatches(groups, qc.batch, torch.Generator().manual_seed(0), train_idx)
    for _ in range(6):
        batch = gb.next()
        assert len(batch) >= qc.batch
        for w in {groups[i] for i in batch}:
            assert {i for i in batch if groups[i] == w} == {i for i in train_idx if groups[i] == w}


# ---------------------------------------------------------------- the expert groups


def _layer_with_experts(n: int = 8):
    layer = nn.Module()
    experts = nn.Module()
    for e in range(n):
        mod = nn.Module()
        for proj in ("gate_proj", "up_proj", "down_proj"):
            setattr(mod, proj, nn.Linear(4, 4, bias=False))
        setattr(experts, str(e), mod)
    mlp = nn.Module()
    mlp.experts = experts
    mlp.gate = nn.Linear(4, n, bias=False)
    layer.mlp = mlp
    return layer


def test_the_experts_are_partitioned_over_the_groups():
    layer = _layer_with_experts(8)
    linears = {
        f"mlp.experts.{e}.{p}": f"x.{e}.{p}"
        for e in range(8)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    linears["mlp.gate"] = "x.gate"
    groups, n = _group_names(layer, linears, 3)
    assert n == 3 and len(groups) == 3
    flat = [n for g in groups for n in g]
    assert sorted(flat) == sorted(linears) and len(flat) == len(set(flat))
    assert "mlp.gate" in groups[0]
    for e in range(8):
        assert sum(f"mlp.experts.{e}.gate_proj" in g for g in groups) == 1


def test_zero_groups_follow_the_weight_threshold(monkeypatch):
    import hi2.qad_looped as QL

    layer = _layer_with_experts(4)
    linears = {
        f"mlp.experts.{e}.{p}": f"x.{e}.{p}"
        for e in range(4)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    monkeypatch.setattr(QL, "GROUP_WEIGHTS", 48)
    groups, n = _group_names(layer, linears, 0)
    assert n == 4 and len(groups) == 4  # 4 experts x 3 x 16 weights / 48 = 4
    monkeypatch.setattr(QL, "GROUP_WEIGHTS", 10**9)
    groups, n = _group_names(layer, linears, 0)
    assert n == 1 and len(groups) == 1


def test_128_experts_split_into_four_groups(monkeypatch):
    import hi2.qad_looped as QL

    layer = _layer_with_experts(128)
    linears = {
        f"mlp.experts.{e}.{p}": f"x.{e}.{p}"
        for e in range(128)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    # the real layer is 1.019 G weights over 128 experts; this tiny stand-in is 6,144,
    # so 1,536 is the same 4-group split without allocating the real weights
    monkeypatch.setattr(QL, "GROUP_WEIGHTS", 1536)
    groups, n = _group_names(layer, linears, 0)
    assert n == 4 and [len(g) for g in groups] == [96, 96, 96, 96]
    experts = [
        {int(n.split(".")[2]) for n in g if ".experts." in n} for g in groups
    ]
    assert sorted(i for g in experts for i in g) == list(range(128))
    assert all(len(g) == 32 for g in experts)


def test_the_config_carries_the_group_count():
    from hi2.qad import fingerprint, qad_fingerprint_fields

    qc = make_config("fast", {"expert_groups": 2})
    assert qc.expert_groups == 2
    assert qad_fingerprint_fields(qc)["expert_groups"] == 2
    assert fingerprint(qad_fingerprint_fields(qc))
    with pytest.raises(SystemExit, match="expert_groups"):
        make_config("fast", {"expert_groups": -1})


def _expert_tensors(dst):
    from hi2.format import dequantize_stored
    from hi2.stack import Hi2Checkpoint

    ck = Hi2Checkpoint(dst)
    out = {}
    for base in sorted(ck.logical()):
        if ".mlp.experts." in base and ".layers." in base:
            out[base] = dequantize_stored(ck.get(base))
    return out


def _expert_raw(dst):
    from hi2.stack import Hi2Checkpoint

    ck = Hi2Checkpoint(dst)
    return {
        base: ck.get(base)
        for base in sorted(ck.logical())
        if ".mlp.experts." in base and ".layers." in base
    }


def test_steps_zero_qad_experts_are_bit_identical_to_ptq(src, ids, tmp_path):
    """The steps=0 control: with the PTQ initialisation and its fallback, a QAD run at
    0 optimizer steps must write exactly the PTQ-only checkpoint for its layers. It
    failed while init_layer dropped the PTQ run's two code-refinement sweeps."""
    from hi2.qad import make_config

    ptq_rc = _rc(src, tmp_path / "ptq", calib_ids=ids, calib_samples=N_SEQ, seq_len=T)
    run(ptq_rc, log=lambda *_: None)
    qc = make_config(
        "fast",
        {
            "steps": 0,
            "samples": 6,
            "val": 2,
            "batch": 2,
            "layers": "0-2",
            "expert_groups": 1,
            "init_from": str(ptq_rc.dst),
        },
    )
    qad_rc = _rc(
        src,
        tmp_path / "qad0",
        calib_ids=ids,
        calib_samples=N_SEQ,
        seq_len=T,
        qad=qc,
    )
    logs = []
    run(qad_rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    a, b = _expert_tensors(ptq_rc.dst), _expert_tensors(qad_rc.dst)
    assert a.keys() == b.keys() and a
    covered = [
        line for line in logs if "init-from " in line and " linears from " in line
    ]
    assert covered and all(
        int(line.split("init-from ")[1].split("/")[0])
        == int(line.split("/")[1].split()[0])
        for line in covered
    ), covered[:3]
    for name in a:
        assert torch.equal(a[name], b[name]), name
    report = json.loads((qad_rc.workdir / "qad_report.json").read_text())
    assert all(report[i]["steps"] == 0 for i in range(3))


def test_a_kept_ptq_group_exports_the_init_from_bytes(src, ids, tmp_path):
    """A group whose held-out loss never beats PTQ falls back to PTQ (`kept: ptq`).
    Training must still export the init-from HI2Tensor byte for byte: the fallback
    FakeQuantLinear holds the compensated weights and re-encoding them moves codes."""
    from hi2.format import FIELDS
    from hi2.qad import make_config

    ptq_rc = _rc(src, tmp_path / "ptq", calib_ids=ids, calib_samples=N_SEQ, seq_len=T)
    run(ptq_rc, log=lambda *_: None)
    qc = make_config(
        "fast",
        {
            "steps": 1,
            "samples": 6,
            "val": 2,
            "batch": 2,
            "layers": "0-2",
            "expert_groups": 1,
            "init_from": str(ptq_rc.dst),
            "lr_w": 0.0,
            "lr_p": 0.0,
        },
    )
    qad_rc = _rc(
        src,
        tmp_path / "qad-ptq-fallback",
        calib_ids=ids,
        calib_samples=N_SEQ,
        seq_len=T,
        qad=qc,
    )
    run(qad_rc, log=lambda *_: None)
    report = json.loads((qad_rc.workdir / "qad_report.json").read_text())
    assert report and all(r["kept"] == "ptq" for r in report), report
    trained = [r for r in report if r["steps"] > 0]
    assert len(trained) == 3
    for r in trained:
        recs = r["group_records"]
        assert recs and len(recs) == len(r["kept_per_group"])
        assert [g["kept"] for g in recs] == r["kept_per_group"]
        assert r["layer_val_before"] == recs[0]["val_before"]
        assert r["layer_val_after"] == recs[-1]["val_after"]
    a, b = _expert_raw(ptq_rc.dst), _expert_raw(qad_rc.dst)
    assert a.keys() == b.keys() and a
    for name in a:
        for field in (*FIELDS, "col_scale"):
            x, y = getattr(a[name], field), getattr(b[name], field)
            assert (x is None) == (y is None), (name, field)
            if x is not None:
                assert torch.equal(x, y), (name, field)


# ---------------------------------------------------------------- the CLI


def test_a_ptq_fallback_is_the_ptq_tensor_not_a_re_encoding():
    """The PTQ fallback must export the init-from tensor byte for byte even when the
    compensated weights would re-encode differently. Regression: the fallback was a
    FakeQuantLinear over the compensated weights and moved 48 % of the packed code
    words on Inkling-2-Mini while still reporting ``kept ptq``."""
    from hi2.format import FIELDS, assemble
    from hi2.qad import fake_quants, make_config

    g = torch.Generator().manual_seed(3)
    r, k = 8, 16
    t0 = assemble(
        [
            (
                torch.arange(r),
                torch.randint(0, 16, (r, k), generator=g, dtype=torch.uint8),
                torch.randint(0, 256, (r, 1, 2), generator=g, dtype=torch.uint8),
                torch.tensor([[0.0, 1.0, 1.0]]).repeat(r, 1).half(),
            )
        ],
        row_nbits=torch.full((r,), 4),
        row_block=torch.full((r,), k),
        had=torch.zeros(1),
        k=k,
    )
    layer = nn.Module()
    layer.lin = nn.Linear(k, r, bias=False)

    class Stager:
        to_host = staticmethod(lambda t: t)

    qc = make_config("fast", {"init": "weights"})
    _, fb = fake_quants(
        layer, {"lin": "lin"}, {"lin": t0}, {"lin": torch.zeros(r, k)}, qc, Stager()
    )
    out = fb["lin"].export()
    for field in (*FIELDS, "col_scale"):
        x, y = getattr(t0, field), getattr(out, field)
        assert (x is None) == (y is None), field
        if x is not None:
            assert torch.equal(x, y), field


def test_the_cli_carries_the_looped_qad_options():
    from hi2.__main__ import build_parser, _qad_overrides

    a = build_parser().parse_args(
        [
            "quantize",
            "src",
            "dst",
            "--qad",
            "--qad-layers",
            "0-5",
            "--qad-final-kl",
            "0",
            "--qad-expert-groups",
            "4",
            "--qad-accum",
            "2",
            "--qad-clip",
            "1",
            "--qad-val",
            "16",
        ]
    )
    assert a.qad_layers == "0-5" and a.qad_final_kl == 0
    over = _qad_overrides(a)
    assert over["layers"] == "0-5" and over["final_kl"] == 0
    assert over["expert_groups"] == 4
    assert over["accum"] == 2 and over["clip"] == 1.0 and over["val"] == 16


# ---------------------------------------------------------------- the pipeline


def test_a_looped_quantize_trains_the_single_pass_and_loop_layers(
    src, ids, tmp_path, monkeypatch
):
    from hi2.checkpoint import validate_checkpoint

    import hi2.qad_looped as QL

    seen: dict[str, list[int] | None] = {}
    orig = QL.train_layer

    def spy(*a, **kw):
        seen[a[10]] = kw.get("groups")
        return orig(*a, **kw)

    monkeypatch.setattr(QL, "train_layer", spy)

    qc = make_config(
        "fast",
        {
            "steps": 2,
            "samples": 6,
            "val": 2,
            "batch": 2,
            "layers": "0-2",
            "expert_groups": 1,
        },
    )
    rc = _rc(src, tmp_path, calib_ids=ids, calib_samples=N_SEQ, seq_len=T, qad=qc)
    run(rc, log=lambda *_: None)
    validate_checkpoint(rc.dst)
    rep = json.loads((rc.workdir / "qad_report.json").read_text())
    assert len(rep) == 6
    for li, passes in ((0, 1), (1, 1), (2, 2)):
        assert rep[li]["passes"] == passes and rep[li]["steps"] == 2
        assert rep[li]["val_before"] > 0 and rep[li]["val_after"] > 0
    # the loop span's passes of one window reach one step; a single-call layer has none
    assert seen["L0"] is None and seen["L1"] is None
    assert seen["L2"] == [w for w in range(N_SEQ) for _ in range(2)]
    assert (rc.workdir / "qad" / "layer002.pt").exists()
    art = torch.load(rc.workdir / "qad" / "layer002.pt", weights_only=False)
    trace = art["info"]["step_trace"]
    assert [row["step"] for row in trace] == [1, 2]
    assert all(row["grad_norm"] > 0 for row in trace)
    assert not (rc.workdir / "qad" / "layer003.pt").exists()  # PTQ only, no artifact
    assert not (rc.workdir / "qad" / "head_inputs.pt").exists()
    qcfg = json.loads((rc.dst / "config.json").read_text())["quantization_config"]
    assert qcfg["qad"]["expert_groups"] == 1


def test_the_trained_layers_are_reused_on_a_rerun(src, ids, tmp_path):
    import shutil

    qc = make_config(
        "fast",
        {"steps": 1, "samples": 6, "val": 2, "batch": 2, "layers": "1"},
    )
    rc = _rc(src, tmp_path, calib_ids=ids, calib_samples=N_SEQ, seq_len=T, qad=qc)
    run(rc, log=lambda *_: None)
    first = (rc.workdir / "qad" / "layer001.pt").read_bytes()
    shutil.rmtree(rc.dst)
    logs: list[str] = []
    run(rc, log=lambda *a: logs.append(" ".join(map(str, a))))
    assert sum("reusing layer001.pt" in ln for ln in logs) == 1
    assert any("teacher pass is skipped" in ln for ln in logs)
    assert (rc.workdir / "qad" / "layer001.pt").read_bytes() == first
