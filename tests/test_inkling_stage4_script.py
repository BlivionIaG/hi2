# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_stage4.sh``: the commands it builds and the gates in front of
them. The Python runners are replaced by a recorder (``STAGE4_ENV``), so nothing heavy
runs: what is checked is which command would run, in which order, and that every gate
stops the run before it. ``HI2_INKLING_STAGE4_E2E=1`` adds the dry run of the whole chain
on the fork's tiny checkpoint (CPU; needs the staged real calibration set)."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "inkling_stage4.sh"
SHA = {"mixed": "ab" * 32, "wiki": "cd" * 32, "pydocs": "ef" * 32}

RECORDER = """
cpu_run() { printf 'CPU %s\\n' "$*" >> "$FAKE_LOG"; }
gpu_run() { local label=$1; shift; printf 'GPU %s %s\\n' "$label" "$*" >> "$FAKE_LOG"; }
"""


def _manifest(**over) -> dict:
    m = {
        "ok": True,
        "seq_len": 2048,
        "thresholds": {"production": True},
        "calibration": {"tokens": 542_995, "windows": 265, "pipeline_windows": 256},
        "heldout": {k: {"ids_sha256_all_windows": v} for k, v in SHA.items()},
    }
    for k, v in over.items():
        m[k] = v
    return m


@pytest.fixture
def box(tmp_path):
    """A work directory with a production-looking manifest and the recorder."""
    out = tmp_path / "stage4"
    (out / "calib").mkdir(parents=True)
    (out / "calib" / "manifest.json").write_text(json.dumps(_manifest()))
    env_file = tmp_path / "stage4.env"
    env_file.write_text(RECORDER)
    log = tmp_path / "fake.log"
    env = {
        **{k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"},
        "OUT": str(out),
        "SRC": str(tmp_path / "model"),
        "STAGE4_ENV": str(env_file),
        "FAKE_LOG": str(log),
        "PY": sys.executable,
        "MIN_FREE_GB": "0",
        "GIB_CACHE": "0",
        "GIB_OUT": "0",
    }

    class Box:
        pass

    b = Box()
    b.out, b.log, b.env, b.tmp = out, log, env, tmp_path

    def drive(*args, **over):
        e = {**env, **{k: str(v) for k, v in over.items()}}
        p = subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=e, capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        lines = log.read_text().splitlines() if log.exists() else []
        return p.returncode, p.stdout + p.stderr, lines

    def write_manifest(**over):
        (out / "calib" / "manifest.json").write_text(json.dumps(_manifest(**over)))

    b.drive, b.write_manifest = drive, write_manifest
    return b


def _gpu(lines):
    return [ln for ln in lines if ln.startswith("GPU ")]


def _flag(line: str, name: str) -> str:
    words = shlex.split(line)
    return words[words.index(name) + 1]


# ------------------------------------------------------------------------- the encodes


def test_an_encode_carries_every_flag_of_the_contract(box):
    code, out, lines = box.drive("quantize", "3.5")
    assert code == 0, out
    (gpu,) = _gpu(lines)
    assert gpu.startswith("GPU quantize-3.5 -m hi2 quantize ")
    for want in (
        r"--include \.mlp\.experts\.",
        "--bits 2,4,8",
        "--candidates lean",
        "--no-qad",
        "--sensitivity none",
        "--target-bpw 3.5",
        "--calib-samples 256",
        "--seq-len 2048",
        "--compute-dtype float16",
        "--devices cuda:0,cuda:1",
    ):
        assert want in gpu, want
    assert _flag(gpu, "--calib") == str(box.out / "calib" / "calib.txt")
    # the set was verified, as a production set, before the encode
    verify = next(ln for ln in lines if "inkling_build_calib.py verify" in ln)
    assert "--production" in verify and lines.index(verify) < lines.index(gpu)


def test_the_three_targets_share_one_cache_and_own_their_directories(box):
    lines = []
    for t in ("3.0", "3.5", "4.0"):
        code, out, lines = box.drive("quantize", t)
        assert code == 0, out
    gpu = _gpu(lines)
    assert [_flag(ln, "--target-bpw") for ln in gpu] == ["3.0", "3.5", "4.0"]
    assert len({_flag(ln, "--cache-dir") for ln in gpu}) == 1
    assert len({_flag(ln, "--workdir") for ln in gpu}) == 3
    assert len({ln.split()[6] for ln in gpu}) == 3  # the three output directories
    assert all(_flag(ln, "--cache-dir").endswith("/cache") for ln in gpu)


def test_the_pilot_is_small_isolated_and_ends_with_a_kl(box):
    code, out, lines = box.drive("pilot")
    assert code == 0, out
    quant, ev = _gpu(lines)
    assert _flag(quant, "--calib-samples") == "4"
    assert _flag(quant, "--target-bpw") == "3.5"
    assert _flag(quant, "--cache-dir").endswith(
        "/cache_pilot"
    )  # not inside the real cache
    assert _flag(quant, "--workdir").endswith("/work/pilot")
    assert ev.startswith("GPU pilot-eval ") and "inkling_kl_eval.py" in ev
    assert _flag(ev, "--windows") == "4" and _flag(ev, "--seq-len") == "512"
    assert _flag(ev, "--groups") == "none"
    assert _flag(ev, "--ids-sha256") == SHA["mixed"]


def test_a_real_encode_may_not_use_the_pilots_size(box):
    code, out, lines = box.drive("quantize", "3.5", SAMPLES=4)
    assert code == 1 and "< 262144" in out
    assert _gpu(lines) == []


# ------------------------------------------------------------------------------ the KL


def test_eval_scores_the_held_out_set_of_the_manifest(box):
    (box.out / "runs" / "bpw3.5").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.5")
    assert code == 0, out
    (ev,) = _gpu(lines)
    assert ev.startswith("GPU eval-3.5-mixed ")
    assert _flag(ev, "--ids").endswith("/calib/heldout_mixed_ids.npy")
    assert _flag(ev, "--ids-sha256") == SHA["mixed"]
    assert _flag(ev, "--windows") == "32" and _flag(ev, "--groups") == "auto"
    assert _flag(ev, "--breakdown-windows") == "8"
    assert _flag(ev, "--devices") == "cuda:0"  # nargs: the rest are words after it
    # the encode runs in float16, the measurement in float32 (float16 rounding alone is
    # a KL of about 3e-3 against float32 on the same weights)
    assert "cuda:0 cuda:1" in ev and _flag(ev, "--compute-dtype") == "float32"
    assert _flag(ev, "--out").endswith("/eval/bpw3.5_mixed.json")
    assert _flag(ev, "--teacher-cache").endswith("/teacher/mixed_w32_float32.pt")
    assert ev.split()[2].endswith("inkling_kl_eval.py")
    assert _flag(ev, "--src") == str(box.tmp / "model")
    assert _flag(ev, "--dst").endswith("/runs/bpw3.5")


def test_an_out_of_distribution_set_has_no_breakdown_and_its_own_hash(box):
    (box.out / "runs" / "bpw3.0").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.0", "wiki", MAX_KL="0.5")
    assert code == 0, out
    (ev,) = _gpu(lines)
    assert _flag(ev, "--ids-sha256") == SHA["wiki"] and _flag(ev, "--groups") == "none"
    assert "--breakdown-windows" not in ev and _flag(ev, "--max-kl") == "0.5"


def test_the_fine_set_breaks_the_mixed_windows_down_and_shares_their_teacher(box):
    (box.out / "runs" / "bpw3.5").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.5", "fine")
    assert code == 0, out
    (ev,) = _gpu(lines)
    assert ev.startswith("GPU eval-3.5-fine ")
    # the same held-out windows and the same teacher pass as `mixed`: no second teacher run
    assert _flag(ev, "--ids").endswith("/calib/heldout_mixed_ids.npy")
    assert _flag(ev, "--ids-sha256") == SHA["mixed"] and _flag(ev, "--windows") == "32"
    assert _flag(ev, "--teacher-cache").endswith("/teacher/mixed_w32_float32.pt")
    assert _flag(ev, "--out").endswith("/eval/bpw3.5_fine.json")
    assert _flag(ev, "--groups") == "0-2,3-5,6-8,9-11,12-14,15-17,18-20,21-23"
    assert _flag(ev, "--breakdown-windows") == "8"
    # the groups are a setting
    code, out, lines = box.drive("eval", "3.5", "fine", FINE_GROUPS="0-11,12-23")
    assert code == 0 and _flag(_gpu(lines)[-1], "--groups") == "0-11,12-23"


def test_the_routing_set_scores_the_mixed_windows_and_logs_the_routing(box):
    (box.out / "runs" / "bpw3.5-p0.5").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.5-p0.5", "routing")
    assert code == 0, out
    (ev,) = _gpu(lines)
    assert ev.startswith("GPU eval-3.5-p0.5-routing ") and "--routing-log" in ev
    # the mixed windows and the mixed teacher pass, the scores only (no breakdown)
    assert _flag(ev, "--ids").endswith("/calib/heldout_mixed_ids.npy")
    assert _flag(ev, "--ids-sha256") == SHA["mixed"] and _flag(ev, "--windows") == "32"
    assert _flag(ev, "--teacher-cache").endswith("/teacher/mixed_w32_float32.pt")
    assert _flag(ev, "--groups") == "none" and "--breakdown-windows" not in ev
    assert _flag(ev, "--out").endswith("/eval/bpw3.5-p0.5_routing.json")
    assert _flag(ev, "--dst").endswith("/runs/bpw3.5-p0.5")
    # no other set takes the log
    for s in ("mixed", "wiki", "pydocs", "fine"):
        code, out, lines = box.drive("eval", "3.5-p0.5", s)
        assert code == 0 and "--routing-log" not in _gpu(lines)[-1], s


def test_eval_needs_an_output_to_score(box):
    code, out, lines = box.drive("eval", "3.5")
    assert code == 1 and "no output at" in out and _gpu(lines) == []


# ------------------------------------------------------------------------------- gates


@pytest.mark.parametrize(
    ("over", "env", "message", "pilot_stops"),
    [
        ({"ok": False}, {}, "failed gate", True),
        ({"thresholds": {"production": False}}, {}, "not a production manifest", True),
        ({"seq_len": 1024}, {}, "windows of 1024 tokens", True),
        ({}, {"SEQ": 4096}, "windows of 2048 tokens, not 4096", True),
        # these two depend on SAMPLES, which the pilot does not use
        ({}, {"SAMPLES": 300}, "the build compared 256", False),
        (
            {"calibration": {"tokens": 100_000, "windows": 48, "pipeline_windows": 48}},
            {"SAMPLES": 48},
            "tokens < 262144",
            False,
        ),
    ],
)
def test_a_calibration_set_that_fails_a_gate_stops_the_gpu_steps(
    box, over, env, message, pilot_stops
):
    box.write_manifest(**over)
    (box.out / "runs" / "bpw3.5").mkdir(parents=True)
    for args in (("quantize", "3.5"), ("eval", "3.5")):
        code, out, lines = box.drive(*args, **env)
        assert code == 1 and message in out, (args, out)
        assert _gpu(lines) == [], args
    code, out, lines = box.drive("pilot", **env)
    if pilot_stops:
        assert code == 1 and message in out and _gpu(lines) == []
    else:  # the pilot is small on purpose: only the pilot's own windows are checked
        assert code == 0 and len(_gpu(lines)) == 2


def test_the_disk_gate_stops_before_the_encode(box):
    code, out, lines = box.drive("quantize", "3.5", MIN_FREE_GB=10**9)
    assert code == 1 and "needs" in out and "GiB" in out
    assert _gpu(lines) == []


def test_the_first_encode_budgets_the_cache_and_the_next_ones_do_not(box):
    code, out, _ = box.drive("quantize", "3.5", GIB_CACHE=3, GIB_OUT=2)
    assert code == 0 and "writes about 5 GiB" in out
    # a cache that already holds its GiB: only the output is new
    env_file = box.tmp / "full_cache.env"
    env_file.write_text(RECORDER + "used_gib() { echo 3; }\n")
    code, out, _ = box.drive(
        "quantize", "4.0", GIB_CACHE=3, GIB_OUT=2, STAGE4_ENV=env_file
    )
    assert code == 0 and "writes about 2 GiB" in out


def test_a_target_is_a_number(box):
    code, out, lines = box.drive("quantize", "high")
    assert code == 1 and "target in bits per weight" in out and lines == []


def test_the_gpu_steps_exclude_each_other(box):
    (box.out / "state" / "gpu.running").mkdir(parents=True)
    code, out, lines = box.drive("quantize", "3.5")
    assert code == 1 and "is running" in out and _gpu(lines) == []


def test_the_driver_never_picks_a_gpu(box):
    """Without a guard in the environment file the GPU wrapper refuses."""
    (box.tmp / "cpu_only.env").write_text(RECORDER.split("\n")[1] + "\n")
    code, out, lines = box.drive("quantize", "3.5", STAGE4_ENV=box.tmp / "cpu_only.env")
    assert code == 1 and "HIP_VISIBLE_DEVICES is not set" in out, out
    assert _gpu(lines) == []


# ---------------------------------------------------------------------------- the rest


def test_a_dry_run_prints_the_commands_and_runs_none(box):
    code, out, lines = box.drive("quantize", "3.5", DRY=1)
    assert code == 0 and lines == []
    assert "+ devrun quantize-3.5 -m hi2 quantize" in out
    assert "--target-bpw 3.5" in out and "--no-qad" in out
    code, out, lines = box.drive("pilot", DRY=1)
    assert code == 0 and lines == [] and "pilot-eval" in out


def test_clean_pilot_removes_the_pilot_and_nothing_else(box):
    keep = [box.out / "runs" / "bpw3.5", box.out / "cache", box.out / "work" / "bpw3.5"]
    gone = [
        box.out / "runs" / "pilot",
        box.out / "work" / "pilot",
        box.out / "cache_pilot",
    ]
    for d in (*keep, *gone):
        d.mkdir(parents=True, exist_ok=True)
    (box.out / "teacher").mkdir()
    (box.out / "teacher" / "pilot_teacher.pt").write_text("x")
    (box.out / "teacher" / "mixed_w32_float16.pt").write_text("x")
    code, out, _ = box.drive("clean-pilot")
    assert code == 0, out
    assert not any(d.exists() for d in gone)
    assert all(d.exists() for d in keep)
    assert not (box.out / "teacher" / "pilot_teacher.pt").exists()
    assert (box.out / "teacher" / "mixed_w32_float16.pt").exists()


def test_clean_pilot_refuses_a_path_outside_the_work_directory(box):
    elsewhere = box.tmp / "elsewhere"
    (elsewhere / "pilot").mkdir(parents=True)
    (elsewhere / "pilot" / "keep.txt").write_text("x")
    code, out, _ = box.drive("clean-pilot", RUNS=elsewhere)
    assert code == 1 and "refusing to remove" in out
    assert (elsewhere / "pilot" / "keep.txt").exists()


def test_plan_asks_hi2_for_its_dry_run_on_the_cpu(box):
    code, out, lines = box.drive("plan", "3.0")
    assert code == 0, out
    (cpu,) = [ln for ln in lines if "--dry-run" in ln]
    assert "--json" in cpu and "--target-bpw 3.0" in cpu
    assert cpu.rstrip().endswith("--devices cpu")  # the last --devices wins
    assert (box.out / "logs" / "plan_3.0.json").exists()


def test_the_usage_is_printed_for_no_step_and_fails_for_an_unknown_one(box):
    code, out, _ = box.drive()
    assert code == 0 and "quantize T" in out and "eval T [SET]" in out
    code, _, _ = box.drive("nonsense")
    assert code == 2


def _load(name: str, relpath: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_command_the_driver_builds_parses_with_the_real_parsers(box):
    """The recorder accepts any flag; the tools do not: parse what the driver would run."""
    from hi2.__main__ import build_parser

    kl = _load("hi2_s4_kl_parser", "experiments/inkling_kl_eval.py")
    calib = _load("hi2_s4_calib_parser", "experiments/inkling_build_calib.py")
    for t in ("3.0", "3.5", "4.0"):
        (box.out / "runs" / f"bpw{t}").mkdir(parents=True)
    steps = [("pilot",), ("plan", "3.5")]
    steps += [("quantize", t) for t in ("3.0", "3.5", "4.0")]
    steps += [
        ("eval", "3.5", s) for s in ("mixed", "wiki", "pydocs", "fine", "routing")
    ]
    for args in steps:
        code, out, lines = box.drive(*args, MAX_KL="1.0")
        assert code == 0, (args, out)
    seen = {"quantize": 0, "eval": 0, "verify": 0}
    for line in lines:
        words = line.split()
        if "quantize" in words and "-m" in words:
            a = build_parser().parse_args(words[words.index("quantize") :])
            assert a.cmd == "quantize" and a.qad is False and a.sensitivity == "none"
            assert a.candidates == "lean" and a.include == r"\.mlp\.experts\."
            assert a.bits == [2, 4, 8] or a.bits == "2,4,8"
            assert a.cache_dir and a.workdir and a.calib and a.target_bpw > 0
            seen["quantize"] += 1
        elif "inkling_kl_eval.py" in line:
            a = kl.parser().parse_args(words[3:])
            assert a.ids_sha256 in SHA.values() and a.out.suffix == ".json"
            seen["eval"] += 1
        elif "inkling_build_calib.py" in line:
            a = calib.parser().parse_args(words[2:])
            assert a.command == "verify" and a.production
            seen["verify"] += 1
    # pilot: 1 encode + 1 KL; plan: 1 dry-run plan; 3 encodes; 5 KLs; a verify before each
    assert seen == {"quantize": 5, "eval": 6, "verify": 9}


# ------------------------------------------------------- the final-KL phase and the variants


def _fkl_inputs(box, tag="3.5"):
    """What `finalkl` needs on disk: the output it starts from, the teacher's head inputs of the
    training windows and the held-out windows' teacher (empty files: the recorder reads none)."""
    (box.out / "runs" / f"bpw{tag}").mkdir(parents=True, exist_ok=True)
    (box.out / "teacher").mkdir(parents=True, exist_ok=True)
    (box.out / "teacher" / "fkl_train_w48_float32.pt").write_bytes(b"")
    (box.out / "teacher" / "mixed_w32_float32.pt").write_bytes(b"")


def test_the_teacher_heads_are_the_float32_program_on_the_training_windows(box):
    code, out, lines = box.drive("teacher-heads")
    assert code == 0, out
    (gpu,) = _gpu(lines)
    assert gpu.startswith("GPU teacher-heads ") and "inkling_final_kl.py teacher" in gpu
    assert _flag(gpu, "--train-windows") == "48" and _flag(gpu, "--seq-len") == "2048"
    assert _flag(gpu, "--calib") == str(box.out / "calib" / "calib.txt")
    assert _flag(gpu, "--out").endswith("/teacher/fkl_train_w48_float32.pt")
    assert _flag(gpu, "--src") == str(box.tmp / "model") and "cuda:0 cuda:1" in gpu
    code, out, lines = box.drive("teacher-heads", FKL_TRAIN=64)
    assert code == 0 and _flag(_gpu(lines)[-1], "--train-windows") == "64"
    assert _flag(_gpu(lines)[-1], "--out").endswith("/teacher/fkl_train_w64_float32.pt")


def test_the_final_kl_phase_is_the_encode_command_with_the_options_of_the_phase(box):
    _fkl_inputs(box)
    code, out, lines = box.drive("finalkl", "3.5")
    assert code == 0, out
    (gpu,) = _gpu(lines)
    assert gpu.startswith("GPU finalkl-3.5 ") and "inkling_final_kl.py student" in gpu
    mine, _, theirs = gpu.partition(" -- ")
    assert _flag(mine, "--init-from") == str(box.out / "runs" / "bpw3.5")
    assert _flag(mine, "--heads").endswith("/teacher/fkl_train_w48_float32.pt")
    assert _flag(mine, "--heldout-ids").endswith("/calib/heldout_mixed_ids.npy")
    assert _flag(mine, "--heldout-teacher").endswith("/teacher/mixed_w32_float32.pt")
    assert (
        _flag(mine, "--heldout-windows") == "32" and _flag(mine, "--val-offset") == "24"
    )
    words = shlex.split(theirs)
    assert words[:2] == ["quantize", str(box.tmp / "model")]
    # the output and the workdir are the phase's own; the statistics are the encode's
    assert words[2] == str(box.out / "runs" / "bpw3.5-fkl40lr0.03")
    assert _flag(theirs, "--workdir") == str(box.out / "work" / "bpw3.5-fkl40lr0.03")
    assert _flag(theirs, "--cache-dir") == str(box.out / "cache")
    assert "--no-qad" not in words and "--qad" in words
    assert (
        _flag(theirs, "--qad-final-kl") == "40"
        and _flag(theirs, "--qad-samples") == "48"
    )
    for want in ("--target-bpw 3.5", "--calib-samples 256", "--sensitivity none"):
        assert want in theirs, want
    assert "--candidates lean" in theirs and "--compute-dtype float16" in theirs
    cfg = json.loads(Path(_flag(theirs, "--qad-config")).read_text())
    assert cfg == {
        "final_batch": 1,
        "final_lr_p": 0.03,
        "val": 8,
        "final_evals": 4,
        "seed": 0,
    }


def test_the_phase_is_a_setting_not_a_constant(box):
    _fkl_inputs(box)
    _, _, lines = box.drive(
        "finalkl", "3.5",
        FKL_STEPS=100, FKL_BATCH=2, FKL_LR=0.05, FKL_VAL=4, FKL_EVALS=10, FKL_VAL_OFFSET=28,
    )  # fmt: skip
    mine, _, theirs = _gpu(lines)[-1].partition(" -- ")
    assert (
        _flag(theirs, "--qad-final-kl") == "100" and _flag(mine, "--val-offset") == "28"
    )
    assert shlex.split(theirs)[2].endswith("/runs/bpw3.5-fkl100lr0.05")
    cfg = json.loads(Path(_flag(theirs, "--qad-config")).read_text())
    assert cfg == {
        "final_batch": 2,
        "final_lr_p": 0.05,
        "val": 4,
        "final_evals": 10,
        "seed": 0,
    }


def test_the_default_learning_rate_of_the_phase_is_the_presets(box):
    """``FKL_LR`` is the preset's ``final_lr_p`` unless a run asks for another one: the single sweep
    that exists (4B at 3.0 bpw) found it best, and nothing was measured on Inkling."""
    qad = pytest.importorskip("hi2.qad")
    _fkl_inputs(box)
    _, _, lines = box.drive("finalkl", "3.5")
    _, _, theirs = _gpu(lines)[-1].partition(" -- ")
    cfg = json.loads(Path(_flag(theirs, "--qad-config")).read_text())
    assert cfg["final_lr_p"] == qad.QadConfig().final_lr_p


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("runs/bpw3.5", "no output at"),
        ("teacher/fkl_train_w48_float32.pt", "run teacher-heads first"),
        ("teacher/mixed_w32_float32.pt", "the teacher of the held-out windows"),
    ],
)
def test_the_final_kl_phase_stops_before_the_gpu_without_its_inputs(
    box, missing, message
):
    _fkl_inputs(box)
    path = box.out / missing
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    code, out, lines = box.drive("finalkl", "3.5")
    assert code == 1 and message in out and _gpu(lines) == []


def test_the_final_kl_phase_budgets_the_output_and_its_resume_copy(box):
    _fkl_inputs(box)
    code, out, lines = box.drive("finalkl", "3.5", GIB_OUT=7)
    assert code == 0 and "writes about 14 GiB" in out
    code, out, lines = box.drive("finalkl", "3.5", GIB_OUT=7, MIN_FREE_GB=10**9)
    assert code == 1 and "needs" in out


def test_a_tag_after_the_bits_names_a_variant_of_the_encode(box):
    code, out, lines = box.drive(
        "quantize", "3.5-p0.5", EXTRA_QUANTIZE="--expert-load-power 0.5"
    )
    assert code == 0, out
    (gpu,) = _gpu(lines)
    assert gpu.startswith("GPU quantize-3.5-p0.5 ")
    assert _flag(gpu, "--target-bpw") == "3.5"  # the bits, not the tag
    assert _flag(gpu, "--expert-load-power") == "0.5"
    assert shlex.split(gpu)[6] == str(box.out / "runs" / "bpw3.5-p0.5")
    assert _flag(gpu, "--workdir") == str(box.out / "work" / "bpw3.5-p0.5")
    assert _flag(gpu, "--cache-dir") == str(
        box.out / "cache"
    )  # the statistics are shared
    code, out, lines = box.drive("plan", "3.5-p0.5")
    assert code == 0 and (box.out / "logs" / "plan_3.5-p0.5.json").exists()
    (cpu,) = [ln for ln in lines if "--dry-run" in ln]
    assert _flag(cpu, "--target-bpw") == "3.5"
    (box.out / "runs" / "bpw3.5-p0.5").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.5-p0.5", "wiki")
    (ev,) = _gpu(lines)[-1:]
    assert code == 0 and _flag(ev, "--dst").endswith("/runs/bpw3.5-p0.5")
    assert _flag(ev, "--out").endswith("/eval/bpw3.5-p0.5_wiki.json")
    # the final-KL phase of a variant starts from the variant and names its output after it
    _fkl_inputs(box, "3.5-p0.5")
    code, out, lines = box.drive("finalkl", "3.5-p0.5")
    assert code == 0, out
    gpu = _gpu(lines)[-1]
    assert gpu.startswith("GPU finalkl-3.5-p0.5 ")
    assert _flag(gpu, "--target-bpw") == "3.5"
    assert _flag(gpu, "--init-from") == str(box.out / "runs" / "bpw3.5-p0.5")
    assert str(box.out / "runs" / "bpw3.5-p0.5-fkl40lr0.03") in gpu
    for bad in ("3.5-", "3.5-p 0.5", "-p0.5", "3.5/p0.5", "3.5--x"):
        code, out, _ = box.drive("quantize", bad)
        assert code == 1 and "target in bits per weight" in out, bad


def test_the_mixed_breakdown_can_be_left_out(box):
    (box.out / "runs" / "bpw3.5").mkdir(parents=True)
    code, out, lines = box.drive("eval", "3.5", "mixed", EVAL_GROUPS="none")
    (ev,) = _gpu(lines)
    assert (
        code == 0
        and _flag(ev, "--groups") == "none"
        and "--breakdown-windows" not in ev
    )
    code, out, lines = box.drive("eval", "3.5", "mixed")  # the default is unchanged
    assert _flag(_gpu(lines)[-1], "--groups") == "auto"


def test_the_commands_of_the_final_kl_phase_parse_with_the_real_parsers(box):
    from hi2.__main__ import build_parser

    fkl = _load("hi2_s4_fkl_parser", "experiments/inkling_final_kl.py")
    _fkl_inputs(box)
    _, _, lines = box.drive("teacher-heads")
    _, _, lines = box.drive("finalkl", "3.5")
    teacher = next(ln for ln in lines if " teacher " in ln)
    a = fkl.parser().parse_args(teacher.split()[3:])
    assert (
        a.cmd == "teacher"
        and a.train_windows == 48
        and a.devices == ["cuda:0", "cuda:1"]
    )
    student = next(ln for ln in lines if " student " in ln)
    mine, _, theirs = student.partition(" -- ")
    a = fkl.parser().parse_args(mine.split()[3:])
    assert a.cmd == "student" and a.val_offset == 24 and a.heldout_windows == 32
    h = build_parser().parse_args(theirs.split())
    assert h.cmd == "quantize" and h.qad is True and h.qad_final_kl == 40
    assert h.qad_samples == 48 and str(h.qad_config).endswith("fkl_qad_40_lr0.03.json")
    assert h.sensitivity == "none" and h.cache_dir and h.workdir and h.target_bpw == 3.5


# ---------------------------------------------------- the dry run of the chain, tiny model


@pytest.mark.skipif(
    os.environ.get("HI2_INKLING_STAGE4_E2E") != "1",
    reason="the toy-model dry run takes minutes: HI2_INKLING_STAGE4_E2E=1",
)
def test_the_whole_chain_on_the_tiny_checkpoint_on_the_cpu(tmp_path):
    staged = Path(
        os.environ.get(
            "REAL_STAGED",
            Path.home() / "w4a8_runs/inkling-hi2/stage4/calib/src",
        )
    )
    if not staged.is_dir():
        pytest.skip(f"{staged}: the staged real calibration set")
    env = {
        **os.environ,
        "TINY": "1",
        "OUT": str(tmp_path / "tiny"),
        "REAL_STAGED": str(staged),
        "PY": sys.executable,
    }

    def drive(*args):
        p = subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=env, capture_output=True, text=True, timeout=1800,
        )  # fmt: skip
        assert p.returncode == 0, (args, p.stdout[-3000:], p.stderr[-3000:])
        return p.stdout

    drive("tiny")
    drive("check")
    manifest = json.loads((tmp_path / "tiny/calib/manifest.json").read_text())
    assert manifest["ok"] and manifest["thresholds"]["production"] is False
    drive("quantize", "3.5")
    drive("eval", "3.5")
    res = json.loads((tmp_path / "tiny/eval/bpw3.5_mixed.json").read_text())
    assert res["ok"] and res["student"]["kl_mean"] > 0 and res["groups"]
    # a toy set is no production set: the production gate of the driver refuses it
    p = subprocess.run(
        ["bash", str(SCRIPT), "check"],
        env={**env, "PRODUCTION": "1"}, capture_output=True, text=True,
    )  # fmt: skip
    assert p.returncode == 1
