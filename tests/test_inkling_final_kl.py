# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_final_kl.py`` and the log of ``hi2.finalkl`` (CPU, the fork's tiny
Inkling-2 checkpoint): a finished output read back is the tensors the pipeline's own layer-wise
phase hands to the final-KL phase; the phase started from it begins at the KL the evaluation
measured; the flipped bytes it reports are the bytes that differ between the two outputs; the
windows and the teacher files of the phase are the ones it was asked for and nothing else."""

import argparse
import functools
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

import hi2.finalkl as finalkl  # noqa: E402
import hi2.pipeline as P  # noqa: E402
from hi2.calibrate import Config  # noqa: E402
from hi2.checkpoint import validate_checkpoint  # noqa: E402
from hi2.pipeline import RunConfig, run  # noqa: E402
from hi2.qad import make_config  # noqa: E402

EXPERTS = r"\.mlp\.experts\."
N_TRAIN, N_VAL, T = 3, 1, 12
PHASE = {
    "steps": 2,
    "samples": N_TRAIN,
    "val": N_VAL,
    "batch": 2,
    "final_kl": 4,
    "final_batch": 1,
    "final_lr_p": 0.5,
    "final_evals": 2,
}
FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
)


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
def fk():
    return _import(
        "hi2_test_inkling_final_kl",
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_final_kl.py",
    )


@pytest.fixture(scope="module")
def kl():
    return _import(
        "hi2_test_inkling_kl_eval",
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_kl_eval.py",
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    make = _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py")
    path = tmp_path_factory.mktemp("fkl_tiny")
    make.write_checkpoint(path, make.TinySpec(num_hidden_layers=6), seed=11)
    return path


@pytest.fixture(scope="module")
def ids():
    return torch.randint(
        0, 50, (N_TRAIN + N_VAL, T), generator=torch.Generator().manual_seed(5)
    )


@pytest.fixture(autouse=True)
def _training_windows_of_the_tiny_model(fk, ids, monkeypatch):
    """The tiny checkpoint has no tokenizer: its training windows are the first token ids."""
    monkeypatch.setattr(
        fk, "train_windows", lambda src, calib, seq_len, n, seed: ids[:n]
    )


def _rc(src, root: Path, name: str, ids, **kw):
    return RunConfig(
        src=src,
        dst=root / name,
        workdir=root / f"{name}_work",
        include=EXPERTS,
        devices=["cpu"],
        compute_dtype="float32",
        target_bpw=3.5,
        cfg=Config(effort="fast"),
        calib_ids=ids,
        calib_samples=ids.shape[0],
        seq_len=T,
        **kw,
    )


def _differing(a: dict, b: dict) -> tuple[int, int]:
    """``(differing ladder bytes, ladder bytes)`` between two readings of HI2 outputs."""
    assert sorted(a) == sorted(b)
    diff = sum(int((a[k].blk_params != b[k].blk_params).sum()) for k in a)
    return diff, sum(t.blk_params.numel() for t in a.values())


def _differing_by_kind(a: dict, b: dict) -> tuple[int, int]:
    """``(differing lo bytes, differing step bytes)``: ``blk_params`` holds ``lo, step`` per block."""
    lo = sum(int((a[k].blk_params[0::2] != b[k].blk_params[0::2]).sum()) for k in a)
    step = sum(int((a[k].blk_params[1::2] != b[k].blk_params[1::2]).sum()) for k in a)
    return lo, step


@pytest.fixture(scope="module")
def world(tiny, ids, fk, kl, tmp_path_factory):
    """A finished PTQ output, the held-out windows with the teacher cache of the KL evaluation,
    the teacher's head inputs of the training windows, and the final-KL run from the output."""
    root = tmp_path_factory.mktemp("fkl_world")
    ptq = _rc(tiny, root, "ptq", ids)
    run(ptq, log=lambda *_: None)
    held = np.random.default_rng(9).integers(0, 50, (4, T)).astype(np.int32)
    np.save(root / "held.npy", held)
    code = kl.main(
        ["--src", str(tiny), "--dst", str(ptq.dst), "--ids", str(root / "held.npy")]
        + ["--devices", "cpu", "--compute-dtype", "float32", "--groups", "none"]
        + ["--teacher-cache", str(root / "held_teacher.pt")]
        + ["--out", str(root / "held_eval.json")]
    )
    assert code == 0
    mp = pytest.MonkeyPatch()
    mp.setattr(fk, "train_windows", lambda src, calib, seq_len, n, seed: ids[:n])
    try:
        args = argparse.Namespace(
            src=tiny,
            calib=root / "unused.txt",
            out=root / "heads.pt",
            train_windows=N_TRAIN,
            seq_len=T,
            seed=0,
            devices=["cpu"],
        )
        assert fk.cmd_teacher(args) == 0
        ns = argparse.Namespace(
            init_from=ptq.dst,
            heads=root / "heads.pt",
            heldout_ids=root / "held.npy",
            heldout_teacher=root / "held_teacher.pt",
            heldout_windows=4,
            val_offset=0,
        )
        rc = _rc(tiny, root, "fkl", ids, qad=make_config("fast", PHASE))
        mp.setattr(P, "_qad_encode", functools.partial(fk.qad_from_output, ns))
        notes: list[str] = []
        run(rc, log=lambda *m: notes.append(" ".join(map(str, m))))
    finally:
        mp.undo()
    return {
        "root": root,
        "ptq": ptq,
        "rc": rc,
        "ns": ns,
        "args": args,
        "notes": notes,
        "eval": json.loads((root / "held_eval.json").read_text()),
        "info": json.loads((rc.workdir / "final_kl.json").read_text()),
    }


# ------------------------------------------------------------- the start of the phase


def test_the_tensors_of_an_output_are_what_the_layerwise_phase_hands_over(
    tiny, ids, fk, world, tmp_path, monkeypatch
):
    """The phase may start from the output of a plain encode because the layer-wise phase of
    a looped model trains nothing: the tensors the pipeline's own run gives ``final_kl`` are
    the tensors read back from the finished output. The allocation is the same to the byte
    (rows, bit widths, blocks, offsets); the values differ by the float noise of the two
    encode paths (measured on the tiny model: 3 ladder bytes of 4340 by one level, 28 code
    words of 7760 in two tensors, one fp16 constant), never by more than a fraction of a percent.
    The phase trains the tensors in place: they are copied when it is entered."""
    seen = {}
    real = finalkl.final_kl

    def spy(model, ids_, train_idx, val_idx, tensors, *a, **kw):
        seen["tensors"] = {
            k: {
                **{
                    f: getattr(t, f).clone()
                    for f in (*FIELDS, "col_scale")
                    if getattr(t, f) is not None
                },
                "shape": tuple(t.shape),
            }
            for k, t in tensors.items()
        }
        return real(model, ids_, train_idx, val_idx, tensors, *a, **kw)

    monkeypatch.setattr(finalkl, "final_kl", spy)
    qc = make_config("fast", {**PHASE, "final_kl": 1})
    run(_rc(tiny, tmp_path, "own", ids, qad=qc), log=lambda *_: None)
    mine = fk.read_output_tensors(world["ptq"].dst)
    assert sorted(seen["tensors"]) == sorted(mine) and len(mine) == 6 * 8 * 3
    differing: dict[str, list] = {}
    totals = {f: sum(getattr(t, f).numel() for t in mine.values()) for f in FIELDS}
    for name, t in mine.items():
        want = seen["tensors"][name]
        assert tuple(t.shape) == want["shape"], name
        for field in (*FIELDS, "col_scale"):
            got = getattr(t, field)
            assert (got is None) == (field not in want), (name, field)
            if got is not None and not torch.equal(got, want[field]):
                n = int((got != want[field]).sum())
                biggest = float((got.double() - want[field].double()).abs().max())
                differing.setdefault(field, []).append((name, n, biggest))
    summary = {  # field: tensors differing, values differing of the values, largest difference
        f: (len(v), sum(x[1] for x in v), totals.get(f), max(x[2] for x in v))
        for f, v in differing.items()
    }
    structure = {"code_off", "param_off", "row_nbits", "row_block", "had", "col_scale"}
    assert not structure & set(differing), summary  # the allocation and the layout
    for field, (tensors, values, total, _) in summary.items():
        assert tensors <= len(mine) // 20 and values <= total // 100, (field, summary)
    if "blk_params" in summary:
        assert summary["blk_params"][3] <= 1, summary  # a ladder byte moves one level


def test_the_phase_starts_from_the_kl_the_evaluation_measured(world):
    """The held-out window of the phase is window 0 of the held-out set: the KL at step 0 is
    the evaluation's own number for that window (the same student, the same teacher)."""
    per_window = world["eval"]["student"]["kl_per_window"]
    info = world["info"]
    assert info["val_before"] == pytest.approx(per_window[0], rel=1e-3)
    assert info["evals"][0]["step"] == 0
    assert info["evals"][0]["windows"] == [info["val_before"]]


def test_the_windows_of_the_phase_sit_where_the_pipeline_splits_them(fk, ids, world):
    assert fk.window_layout(N_TRAIN, N_VAL) == ([0, 1, 2], [3])
    train_idx, val_idx = fk.window_layout(48, 8)  # the stage-5 pilot
    assert len(val_idx) == 8 and len(train_idx) == 48
    assert sorted(train_idx + val_idx) == list(range(56))
    assert not set(train_idx) & set(val_idx)
    assert val_idx == sorted(val_idx) and val_idx[-1] == 55  # spread over the set
    got_ids, got_heads = fk.phase_inputs(world["ns"], world["rc"], world["rc"].qad)
    held = np.load(world["ns"].heldout_ids)
    assert torch.equal(got_ids[:3], ids[:3])
    assert torch.equal(got_ids[3], torch.from_numpy(held[0].astype(np.int64)))
    teacher = torch.load(world["ns"].heads, weights_only=True)["heads"]
    cached = torch.load(world["ns"].heldout_teacher, weights_only=True)["heads"]
    assert torch.equal(got_heads[:3], teacher) and torch.equal(got_heads[3], cached[0])
    shifted = argparse.Namespace(**{**vars(world["ns"]), "val_offset": 2})
    got_ids, got_heads = fk.phase_inputs(shifted, world["rc"], world["rc"].qad)
    assert torch.equal(got_ids[3], torch.from_numpy(held[2].astype(np.int64)))
    assert torch.equal(got_heads[3], cached[2])


# ------------------------------------------------------------------ what the phase reports


def test_the_log_has_a_line_per_step_and_the_peak_per_device(world):
    info = world["info"]
    steps = info["steps_log"]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    assert all(r["seconds"] >= 0 and r["train"] > 0 for r in steps)
    assert all(r["peak_gib"] == [] for r in steps)  # the CPU has no peak to report
    assert info["gpu_peak_gib"] is None and info["gpu_peak_gib_per_device"] is None
    lines = [n for n in world["notes"] if "final KL step" in n]
    assert (
        sum("held-out" in n for n in lines) == 2
    )  # every 2 steps of 4 (final_evals 2)
    assert sum("held-out" not in n for n in lines) == 4
    assert [e["step"] for e in info["evals"]] == [0, 2, 4]
    assert [t[0] for t in info["trace"]] == [0, 2, 4]  # the old trace is unchanged
    assert all(len(e["windows"]) == N_VAL for e in info["evals"])
    for e in info["evals"]:
        assert e["held_out"] == pytest.approx(sum(e["windows"]) / N_VAL)


def test_the_flipped_bytes_are_the_bytes_that_differ_between_the_outputs(fk, world):
    info = world["info"]
    before = fk.read_output_tensors(world["ptq"].dst)
    after = fk.read_output_tensors(world["rc"].dst)
    diff, total = _differing(before, after)
    fl = info["flipped"]
    assert fl["count"] == diff and fl["values"] == total
    assert fl["count"] == fl["lo"] + fl["step"]
    assert (fl["lo"], fl["step"]) == _differing_by_kind(before, after)
    assert fl["share"] == pytest.approx(diff / total)
    assert (
        info["evals"][0]["flipped"]["count"] == 0
    )  # nothing moved before the first step
    assert any(e["flipped"]["count"] > 0 for e in info["evals"][1:])
    if info["kept"] == "init":  # nothing better than the start: the output is the start
        assert diff == 0 and info["best_step"] == 0
    else:
        assert info["best_step"] in (2, 4)
    assert info["val_after"] <= info["val_before"]
    validate_checkpoint(world["rc"].dst)


# ------------------------------------------------------------------ what it refuses


def test_a_teacher_file_of_other_windows_is_refused(fk, world, tmp_path):
    blob = torch.load(world["ns"].heads, weights_only=True)
    torch.save({"fingerprint": "0" * 64, "heads": blob["heads"]}, tmp_path / "other.pt")
    ns = argparse.Namespace(**{**vars(world["ns"]), "heads": tmp_path / "other.pt"})
    with pytest.raises(SystemExit, match="made for other training windows"):
        fk.phase_inputs(ns, world["rc"], world["rc"].qad)


def test_a_held_out_teacher_of_other_windows_is_refused(fk, world):
    # the cache is the teacher of the first 4 windows, not of the first 3
    ns = argparse.Namespace(**{**vars(world["ns"]), "heldout_windows": 3})
    with pytest.raises(SystemExit, match="is not the teacher of the first 3"):
        fk.phase_inputs(ns, world["rc"], world["rc"].qad)


def test_the_held_out_windows_must_lie_in_the_cache(fk, world):
    ns = argparse.Namespace(**{**vars(world["ns"]), "val_offset": 4})
    with pytest.raises(SystemExit, match="do not fit the first 4"):
        fk.phase_inputs(ns, world["rc"], world["rc"].qad)


def test_the_teacher_file_is_kept_for_the_same_windows(fk, world, capsys):
    out = world["args"].out
    stamp = out.stat().st_mtime_ns
    assert fk.cmd_teacher(world["args"]) == 0  # nothing to do: the same windows
    assert out.stat().st_mtime_ns == stamp
    assert "head inputs of these windows are there" in capsys.readouterr().out


def test_a_final_kl_of_zero_steps_is_not_this_drivers_job(fk, tiny, ids, world):
    qc = make_config("fast", {**PHASE, "final_kl": 0})
    rc = _rc(tiny, world["root"], "zero", ids, qad=qc)
    with pytest.raises(SystemExit, match="N > 0"):
        fk.qad_from_output(world["ns"], rc, None, {}, {}, None, lambda *_: None)


def test_the_hi2_command_goes_after_the_separator(fk):
    with pytest.raises(SystemExit, match="after `--`"):
        fk.cmd_student(argparse.Namespace(), [])
