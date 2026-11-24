# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_kl_eval.py`` (CPU, the fork's tiny Inkling-2 checkpoint): the metric
equals the textbook KL, a coarser student scores worse than a finer one, the layer groups
attribute the damage to the layers that hold it, a student that cannot be scored is
refused before the model is loaded, and every gate fails on the input it exists for."""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hi2.calibrate import Config
from hi2.pipeline import RunConfig, run

EXPERTS = r"\.mlp\.experts\."
ATTENTION = r"\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.linear\.weight$"


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
def kl():
    return _import(
        "hi2_test_inkling_kl_eval",
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_kl_eval.py",
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    make = _import("hi2_test_make_tiny", d / "make_tiny_checkpoint.py")
    path = tmp_path_factory.mktemp("kl_tiny")
    make.write_checkpoint(path, make.TinySpec(num_hidden_layers=6), seed=11)
    return path


def _quantize(src, root: Path, name: str, bpw: float, include: str = EXPERTS) -> Path:
    rc = RunConfig(
        src=src,
        dst=root / name,
        workdir=root / f"{name}_work",
        include=include,
        devices=["cpu"],
        compute_dtype="float32",
        target_bpw=bpw,
        cfg=Config(effort="fast"),
    )
    run(rc, log=lambda *_: None)
    return rc.dst


@pytest.fixture(scope="module")
def students(tiny, tmp_path_factory):
    root = tmp_path_factory.mktemp("kl_students")
    return {
        "coarse": _quantize(tiny, root, "coarse", 2.6),
        "fine": _quantize(tiny, root, "fine", 6.0),
        "attention": _quantize(
            tiny, root, "attention", 3.5, f"({EXPERTS}|{ATTENTION})"
        ),
    }


@pytest.fixture(scope="module")
def ids_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("kl_ids") / "ids.npy"
    g = np.random.default_rng(3)
    np.save(path, g.integers(0, 500, (6, 24)).astype(np.int32))
    return path


def _score(kl, tiny, dst, ids_file, tmp_path, *extra, name="kl.json"):
    out = tmp_path / name
    argv = ["--src", str(tiny), "--dst", str(dst), "--ids", str(ids_file)]
    argv += ["--devices", "cpu", "--compute-dtype", "float32", "--out", str(out)]
    argv += ["--breakdown-windows", "3", *extra]
    code = kl.main(argv)
    return code, json.loads(out.read_text())


# ------------------------------------------------------------------------ the metric


def test_the_metric_is_the_kl_of_the_two_distributions(kl):
    torch.manual_seed(0)
    lm = nn.Linear(4, 9, bias=False)
    ht, hs = torch.randn(3, 11, 4), torch.randn(3, 11, 4)
    ids = torch.randint(0, 9, (3, 11))
    got = kl.kl_stats(lm, hs, ht, ids, chunk=4)  # 11 positions: chunks of 4, 4, 3
    lt, ls = F.log_softmax(lm(ht), -1), F.log_softmax(lm(hs), -1)
    want_kl = F.kl_div(ls, lt, log_target=True, reduction="none").sum(-1).mean(-1)
    assert torch.allclose(got["kl"], want_kl.double(), atol=1e-6)
    want_top1 = (lt.argmax(-1) == ls.argmax(-1)).float().mean(-1)
    assert torch.allclose(got["top1"], want_top1.double())
    want_ear = torch.minimum(lt.exp(), ls.exp()).sum(-1).mean(-1)
    assert torch.allclose(got["ear"], want_ear.double(), atol=1e-6)
    tgt = ids[:, 1:]  # the next token of positions 0..T-2
    nll_t = -lt[:, :-1].gather(2, tgt[..., None])[..., 0].mean(-1)
    nll_s = -ls[:, :-1].gather(2, tgt[..., None])[..., 0].mean(-1)
    assert torch.allclose(got["nll_teacher"], nll_t.double(), atol=1e-6)
    assert torch.allclose(got["nll_student"], nll_s.double(), atol=1e-6)
    assert (got["kl"] > 0).all()
    same = kl.kl_stats(lm, ht, ht, ids, chunk=4)  # identical inputs: exactly nothing
    assert same["kl"].abs().max() == 0 and same["top1"].min() == 1
    assert torch.allclose(same["ear"], torch.ones(3, dtype=torch.float64))


def test_the_metric_on_a_distribution_worked_by_hand(kl):
    """Logits [ln 1.5, 0] against [0, ln 4]: p = (.6, .4), q = (.2, .8)."""
    ht = torch.tensor([[[float(np.log(1.5)), 0.0]]])
    hs = torch.tensor([[[0.0, float(np.log(4))]]])
    got = kl.kl_stats(nn.Identity(), hs, ht, torch.zeros(1, 1, dtype=torch.long))
    want = 0.6 * np.log(0.6 / 0.2) + 0.4 * np.log(0.4 / 0.8)
    assert want == pytest.approx(0.381908, abs=1e-6)
    assert float(got["kl"][0]) == pytest.approx(want, abs=1e-6)
    assert float(got["ear"][0]) == pytest.approx(
        0.2 + 0.4, abs=1e-6
    )  # sum of min(p, q)
    assert float(got["top1"][0]) == 0.0  # the teacher prefers class 0, the student 1


def test_summaries_carry_the_standard_error_over_windows(kl):
    stats = {k: torch.zeros(4, dtype=torch.float64) for k in kl.METRICS}
    stats["kl"] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    stats["nll_student"] = torch.full((4,), 0.5, dtype=torch.float64)
    s = kl.summarize(stats)
    assert s["kl_mean"] == pytest.approx(0.25)
    assert s["kl_sem"] == pytest.approx(float(np.std([0.1, 0.2, 0.3, 0.4], ddof=1)) / 2)
    assert s["ppl_ratio"] == pytest.approx(float(np.exp(0.5)))
    assert s["kl_per_window"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    one = kl.summarize({k: v[:1] for k, v in stats.items()})
    assert one["kl_sem"] is None  # one window has no spread (and the JSON no NaN)


def test_the_groups_follow_the_loop_and_the_stock_formats(kl):
    real = ["nvfp4"] * 20 + ["mxfp8"] * 4  # Inkling-2-Mini
    assert kl.auto_groups(24, (6, 18), real) == [
        list(range(6)),
        list(range(6, 18)),
        [18, 19],
        list(range(20, 24)),
    ]
    assert kl.auto_groups(6, (2, 4), ["nvfp4"] * 5 + ["mxfp8"]) == [
        [0, 1],
        [2, 3],
        [4],
        [5],
    ]
    assert kl.parse_groups("auto", 6, (2, 4), ["nvfp4"] * 6) == [[0, 1], [2, 3], [4, 5]]
    assert kl.parse_groups("0-1,3", 6, (2, 4), ["nvfp4"] * 6) == [[0, 1], [3]]
    assert [kl.group_name(g) for g in ([0, 1, 2], [3])] == ["0-2", "3"]
    with pytest.raises(SystemExit, match="outside the 6 layers"):
        kl.parse_groups("4-6", 6, (2, 4), ["nvfp4"] * 6)


def test_the_teacher_cache_key_follows_the_model_the_tokens_and_the_dtype(kl, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x" * 8)
    ids = np.arange(48, dtype=np.int32).reshape(2, 24)
    key = kl.fingerprint(tmp_path, ids, "float16")
    assert kl.fingerprint(tmp_path, ids.copy(), "float16") == key
    assert kl.fingerprint(tmp_path, ids, "float32") != key
    assert kl.fingerprint(tmp_path, ids[:1], "float16") != key
    assert kl.fingerprint(tmp_path, ids.reshape(1, 48), "float16") != key  # the shape
    other = ids.copy()
    other[1, 5] += 1
    assert kl.fingerprint(tmp_path, other, "float16") != key
    (tmp_path / "model.safetensors").write_bytes(b"x" * 9)
    assert kl.fingerprint(tmp_path, ids, "float16") != key


# --------------------------------------------------------------- a student against stock


def test_a_finer_student_scores_better_and_every_gate_holds(
    kl, tiny, students, ids_file, tmp_path
):
    code, coarse = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, name="c.json"
    )
    assert code == 0 and coarse["ok"], coarse["gates"]
    code, fine = _score(kl, tiny, students["fine"], ids_file, tmp_path, name="f.json")
    assert code == 0 and fine["ok"], fine["gates"]
    c, f = coarse["student"], fine["student"]
    assert c["windows"] == f["windows"] == 6 and len(c["kl_per_window"]) == 6
    assert c["kl_mean"] > f["kl_mean"] > 0
    assert c["ppl_ratio"] != 1.0
    assert all(0.0 <= r["top1_agreement"] <= 1.0 for r in (c, f))
    n = 6 * 8 * 3  # layers x experts x projections of the tiny model
    for res in (coarse, fine):
        assert res["tensors"] == {
            "installed": n, "changed": n, "missing": 0, "expected": n,
        }  # fmt: skip
        assert all(g["ok"] for g in res["gates"].values())
        assert res["gates"]["groups_restored_the_stock_weights"]["detail"].startswith(
            "max diff 0 "
        )
    # the weight-space error is larger for the coarser student in every layer
    for layer in range(6):
        ce = coarse["weight_error_per_layer"][str(layer)]["relative_rms"]
        fe = fine["weight_error_per_layer"][str(layer)]["relative_rms"]
        assert ce > fe > 0, layer
    assert list(coarse["groups"]) == ["0-1", "2-3", "4", "5"]
    assert coarse["groups"]["5"]["formats"] == ["mxfp8"]
    assert coarse["groups"]["0-1"]["formats"] == ["nvfp4"]
    assert all(
        g["windows"] == 3 and g["kl_mean"] >= 0 for g in coarse["groups"].values()
    )
    # the same windows: the whole student is the reference of the groups
    assert coarse["student_on_breakdown_windows"]["windows"] == 3


def test_the_installed_weights_are_the_decodes_they_claim_to_be(kl, tiny, students):
    """Every expert Linear of the model holds, bit for bit, the student's decode when it
    is the student (read by ``load_dequantized``, a walk over the shards, not the lookup by
    name that the evaluation uses) and the stock decode of the reader (gate rows first)
    when it is the teacher; the two differ."""
    from hi2.checkpoint import load_dequantized
    from hi2.inkling_source import InklingSource
    from hi2.pipeline import _load_model
    from hi2.stack import Hi2Checkpoint

    dst = students["coarse"]
    cpu = torch.device("cpu")
    rc = RunConfig(
        src=tiny, dst=dst.parent / "x", workdir=dst.parent / "wx",
        compute_dtype="float32", devices=["cpu"],
    )  # fmt: skip
    model = _load_model(rc, lambda *_: None)
    with InklingSource(tiny, device=cpu) as src:
        ex = kl.Experts(model, src, Hi2Checkpoint(dst), cpu, lambda *_: None)
        ex.set(ex.all("student"))
        want = load_dequantized(dst)
        n = 0
        for layer, mlps in ex.mods.items():
            for e, mlp in enumerate(mlps):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{ex.path}.{layer}.mlp.experts.{e}.{proj}.weight"
                    assert torch.equal(getattr(mlp, proj).weight, want[name]), name
                    n += 1
        assert n == 6 * 8 * 3
        ex.set(ex.all("teacher"))
        differs = 0
        for layer, mlps in ex.mods.items():
            gate_up, down = src.fused_layer(layer, device=cpu, dtype=torch.float32)
            for e, mlp in enumerate(mlps):
                i = mlp.gate_proj.weight.shape[0]
                assert torch.equal(mlp.gate_proj.weight, gate_up[e, :i])
                assert torch.equal(mlp.up_proj.weight, gate_up[e, i:])
                assert torch.equal(mlp.down_proj.weight, down[e])
                name = f"{ex.path}.{layer}.mlp.experts.{e}.gate_proj.weight"
                differs += not torch.equal(mlp.gate_proj.weight, want[name])
        assert differs > 0  # the student is not the stock model


def test_one_group_over_every_layer_is_the_whole_student(
    kl, tiny, students, ids_file, tmp_path
):
    """The layer states are written and restored correctly: a group of all six layers
    gives the whole-student KL of the same windows, bit for bit."""
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path,
        "--groups", "0-5", "--breakdown-windows", "6",
    )  # fmt: skip
    assert code == 0
    assert res["groups"]["0-5"]["kl_per_window"] == res["student"]["kl_per_window"]


def test_the_groups_attribute_the_damage_to_the_layers_that_hold_it(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    """A student with the experts of layers 2 and 3 only (``--partial``; the output of a
    run is never partial, so the others' tensors are hidden from the reader): the groups of
    the other layers hold the stock weights, so their KL is exactly 0; the group 2-3 holds
    all of it and equals the whole student."""
    _only_layers(monkeypatch, {2, 3})
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path,
        "--partial", "--breakdown-windows", "6",
    )  # fmt: skip
    assert code == 0, res["gates"]
    groups = res["groups"]
    for name in ("0-1", "4", "5"):
        assert groups[name]["kl_mean"] == 0.0, name
    assert groups["2-3"]["kl_mean"] > 0
    assert groups["2-3"]["kl_per_window"] == res["student"]["kl_per_window"]
    assert (
        res["tensors"]["missing"] == 4 * 8 * 3
        and res["tensors"]["installed"] == 2 * 8 * 3
    )


def test_the_teacher_pass_is_cached_and_the_cache_belongs_to_its_tokens(
    kl, tiny, students, ids_file, tmp_path, capsys
):
    cache = tmp_path / "teacher.pt"
    args = ("--teacher-cache", str(cache), "--groups", "none")
    _, first = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, *args, name="1.json"
    )
    assert cache.exists() and "teacher_program" in first["seconds"]
    capsys.readouterr()
    _, second = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, *args, name="2.json"
    )
    assert "teacher head inputs from" in capsys.readouterr().out
    assert "teacher_program" not in second["seconds"]
    assert second["student"]["kl_per_window"] == first["student"]["kl_per_window"]
    # other tokens: the cache is not used, and the numbers are those of a fresh run
    _, fewer = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        *args,
        "--windows",
        "4",
        name="3.json",
    )
    assert "teacher_program" in fewer["seconds"] and fewer["windows"] == 4
    _, fresh = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        "--windows",
        "4",
        name="4.json",
    )
    assert fewer["student"]["kl_per_window"] == fresh["student"]["kl_per_window"]


def test_a_crop_of_the_windows_changes_what_is_scored(
    kl, tiny, students, ids_file, tmp_path
):
    _, full = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        name="a.json",
    )
    _, cropped = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        "--seq-len",
        "12",
        name="b.json",
    )
    assert full["seq_len"] == 24 and cropped["seq_len"] == 12
    assert full["student"]["kl_mean"] != cropped["student"]["kl_mean"]


# --------------------------------------------------------------------- what is refused


def _only_layers(monkeypatch, keep: set[int]) -> None:
    """The HI2 reader as if the output held the experts of the layers in ``keep`` only (the
    pipeline refuses to write such an output: an expert left out would be copied at four
    bytes per weight)."""
    from hi2.stack import Hi2Checkpoint

    real_get, real_logical = Hi2Checkpoint.get, Hi2Checkpoint.logical

    def held(base: str) -> bool:
        return int(re.search(r"\.layers\.(\d+)\.", base).group(1)) in keep

    def get(self, base):
        if not held(base):
            raise KeyError(base)
        return real_get(self, base)

    def logical(self):
        return {b: v for b, v in real_logical(self).items() if held(b)}

    monkeypatch.setattr(Hi2Checkpoint, "get", get)
    monkeypatch.setattr(Hi2Checkpoint, "logical", logical)


def _never_load(monkeypatch):
    import hi2.pipeline

    def boom(*a, **k):
        raise AssertionError("the model was loaded for a student that cannot be scored")

    monkeypatch.setattr(hi2.pipeline, "_load_model", boom)


def test_the_stock_directory_is_not_a_student(
    kl, tiny, ids_file, tmp_path, monkeypatch
):
    _never_load(monkeypatch)
    code, res = _score(kl, tiny, tiny, ids_file, tmp_path)
    assert code == 1 and not res["ok"]
    assert not res["gates"]["student_has_expert_tensors"]["ok"]
    assert "student" not in res  # nothing was scored


def test_a_student_without_every_expert_is_refused_unless_partial_is_asked(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    _never_load(monkeypatch)
    _only_layers(monkeypatch, {2, 3})
    code, res = _score(kl, tiny, students["coarse"], ids_file, tmp_path)
    gate = res["gates"]["student_covers_every_expert"]
    assert (
        code == 1
        and not gate["ok"]
        and "48 HI2 expert tensors of 144" in gate["detail"]
    )


def test_another_quantized_tensor_is_refused_because_it_would_be_ignored(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    """Attention projections in the output: their error is not in the scored model, so
    the number would flatter the student."""
    _never_load(monkeypatch)
    code, res = _score(kl, tiny, students["attention"], ids_file, tmp_path)
    gate = res["gates"]["no_other_tensor_is_quantized"]
    assert code == 1 and not gate["ok"] and "other HI2 tensors" in gate["detail"]
    assert res["gates"]["student_covers_every_expert"]["ok"]


def test_held_out_tokens_that_are_not_the_manifests_are_refused(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    _never_load(monkeypatch)
    right = kl.sha256_array(np.load(ids_file))
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, "--ids-sha256", "0" * 64
    )
    assert code == 1 and not res["gates"]["ids_match_the_manifest"]["ok"]
    monkeypatch.undo()  # the right hash passes the gate (checked on the gate alone)
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path,
        "--ids-sha256", right, "--groups", "none",
    )  # fmt: skip
    assert code == 0 and res["gates"]["ids_match_the_manifest"]["ok"]


def test_the_kl_limit_is_a_gate(kl, tiny, students, ids_file, tmp_path):
    code, res = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        "--max-kl",
        "0.0",
    )
    assert code == 1 and not res["gates"]["kl_within_the_limit"]["ok"]
    assert res["student"]["kl_mean"] > 0  # the numbers are still written
    code, res = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        "--max-kl",
        "1e9",
    )
    assert code == 0 and res["gates"]["kl_within_the_limit"]["ok"]


def test_a_student_that_is_the_teacher_is_refused_and_scores_exactly_zero(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    """An install that changes nothing (the pipeline's output was read as the stock
    weights, say) would report a KL of 0: the gate refuses to call that a measurement."""

    def same(self, layer):
        n = 3 * len(self.mods[layer])
        self.installed += n
        self.weight_error[layer] = {
            "relative_rms": 0.0, "tensors": n, "tensors_changed": 0, "tensors_missing": 0,
        }  # fmt: skip

    monkeypatch.setattr(kl.Experts, "_student", same)
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, "--groups", "none"
    )
    assert code == 1 and not res["gates"]["student_differs_from_the_teacher"]["ok"]
    assert res["student"]["kl_mean"] == 0.0  # same weights, same program


def test_a_negative_kl_is_a_failure(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    real = kl.kl_stats

    def shifted(lm, hs, ht, ids, device=None, chunk=256):
        out = real(lm, hs, ht, ids, device, chunk)
        if not torch.equal(hs, ht):  # the student, not the check on identical inputs
            out["kl"] = out["kl"] - 100.0
        return out

    monkeypatch.setattr(kl, "kl_stats", shifted)
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, "--groups", "none"
    )
    assert code == 1 and not res["gates"]["no_negative_kl"]["ok"]
    assert res["gates"]["metric_is_zero_on_identical_inputs"]["ok"]


@pytest.mark.parametrize(
    ("call", "gate"), [(0, "teacher_finite"), (1, "student_finite")]
)
def test_non_finite_head_inputs_are_a_failure(
    kl, tiny, students, ids_file, tmp_path, monkeypatch, call, gate
):
    import hi2.looped as looped

    real, n = looped.teacher_head_inputs, {"calls": 0}

    def poisoned(*a, **k):
        out = real(*a, **k)
        if n["calls"] == call:  # call 0 is the teacher's pass, call 1 the student's
            out = out.clone()
            out[0, 0, 0] = float("nan")
        n["calls"] += 1
        return out

    monkeypatch.setattr(looped, "teacher_head_inputs", poisoned)
    code, res = _score(
        kl, tiny, students["coarse"], ids_file, tmp_path, "--groups", "none"
    )
    assert code == 1 and not res["gates"][gate]["ok"]
    assert not res["gates"]["every_value_finite"]["ok"]


def test_a_layer_that_stays_quantized_fails_the_restore_gate(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    """If the groups left a layer on the student weights, the stock model run again would
    not equal the probe: the gate sees it."""
    real = kl.Experts._teacher

    def forgetful(self, layer):
        if layer == 3 and self.state.get(3) == "student":
            return  # layer 3 is never restored
        real(self, layer)

    monkeypatch.setattr(kl.Experts, "_teacher", forgetful)
    code, res = _score(kl, tiny, students["coarse"], ids_file, tmp_path)
    gate = res["gates"]["groups_restored_the_stock_weights"]
    assert code == 1 and not gate["ok"] and not gate["detail"].startswith("max diff 0 ")


# ------------------------------------------------------------------ the routing log


def _routing(kl, tiny, dst, ids, tmp_path, name, *extra):
    """``--routing-log`` on the windows ``ids``: ``(exit code, result, the counts file)``."""
    path = tmp_path / f"{name}_ids.npy"
    np.save(path, ids)
    code, res = _score(
        kl,
        tiny,
        dst,
        path,
        tmp_path,
        "--routing-log",
        "--groups",
        "none",
        *extra,
        name=f"{name}.json",
    )
    return code, res, np.load(tmp_path / f"{name}_routing.npz")


def test_the_class_of_an_expert_follows_its_mean_code_width(kl):
    bits = torch.tensor([[2.0, 2.99], [3.0, 5.99], [6.0, 8.0]], dtype=torch.float64)
    m = kl.class_masks(bits)
    assert m["int2"].tolist() == [[True, True], [False, False], [False, False]]
    assert m["int4"].tolist() == [[False, False], [True, True], [False, False]]
    assert m["int8"].tolist() == [[False, False], [False, False], [True, True]]
    assert (sum(x.long() for x in m.values()) == 1).all()  # each expert in one class


@pytest.mark.parametrize("per_window_forward", [False, True])
def test_the_routing_log_sends_every_token_to_its_own_window_on_two_devices(
    kl, tiny, students, ids_file, tmp_path, monkeypatch, per_window_forward
):
    """Six windows dealt round-robin to two devices in batches of two (windows 0 and 2, then 4
    on the first; 1 and 3, then 5 on the second), or one forward per window
    (``HI2_LAYER_BATCH=0``): the counts of window w are those of window w run alone,
    whichever device and batch held it."""
    monkeypatch.setenv("HI2_LOOP_SEQS", "2")
    if per_window_forward:
        monkeypatch.setenv("HI2_LAYER_BATCH", "0")
    ids = np.load(ids_file)
    both = ("--devices", "cpu", "cpu")
    code, res, npz = _routing(
        kl, tiny, students["coarse"], ids, tmp_path, "both", *both
    )
    assert code == 0 and res["ok"] and res["gates"]["routing_log_complete"]["ok"]
    counts = npz["counts"]
    assert counts.shape == (6, 6, 8)  # windows, layers, experts of the tiny model
    for w in range(6):
        solo = _routing(kl, tiny, students["coarse"], ids[w : w + 1], tmp_path, f"w{w}")
        assert solo[0] == 0 and np.array_equal(solo[2]["counts"][0], counts[w]), w
    assert len({counts[w].tobytes() for w in range(6)}) == 6  # the windows differ


def test_the_routing_log_adds_up_and_reports_the_classes(
    kl, tiny, students, ids_file, tmp_path
):
    ids = np.load(ids_file)
    code, res, npz = _routing(kl, tiny, students["coarse"], ids, tmp_path, "coarse")
    assert code == 0, res["gates"]
    r = res["routing"]
    passes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1}  # the loop layers run twice
    assert r["passes"] == {str(k): v for k, v in passes.items()}
    counts, bits = npz["counts"].astype(np.int64), npz["expert_bits"]
    for (
        layer,
        n,
    ) in passes.items():  # every token of every window, k experts, every pass
        assert (counts[:, layer].sum(-1) == n * 24 * r["top_k"]).all(), layer
    assert r["complete"] and list(r["groups"]) == ["0-1", "2-3", "4", "5"]
    # the shares, recomputed from the counts and the widths of the output
    int2 = bits < 3.0
    for name, g in r["groups"].items():
        slots = g["layers"]
        c = counts[:, slots]
        want = (c * int2[slots]).sum((1, 2)) / c.sum((1, 2))
        assert g["per_window"]["int2"] == pytest.approx(want.tolist(), abs=1e-12), name
        assert sum(g["assignments"].values()) == pytest.approx(1.0)
        assert sum(g["experts"].values()) == pytest.approx(1.0)
        assert g["mean_bits"] == pytest.approx(float((c * bits[slots]).sum() / c.sum()))
        assert len(g["per_window"]["int2"]) == len(g["mean_bits_per_window"]) == 6
    whole = r["all_layers"]
    assert whole["layers"] == list(range(6))
    assert whole["assignments"]["int2"] == pytest.approx(
        float((counts * int2).sum() / counts.sum())
    )
    assert sum(r["expert_bits_histogram"]["experts"]) == 6 * 8  # every expert once
    # a coarser output has more INT2-class experts and a lower code width than a finer one
    code, res, _ = _routing(kl, tiny, students["fine"], ids, tmp_path, "fine")
    assert code == 0
    fine, coarse = res["routing"]["all_layers"], r["all_layers"]
    assert fine["experts"]["int2"] < coarse["experts"]["int2"]
    assert fine["mean_bits"] > coarse["mean_bits"]


def test_the_routing_log_changes_nothing_it_measures(
    kl, tiny, students, ids_file, tmp_path
):
    import hi2.layerwise as layerwise

    original = layerwise._device_pass
    code, plain = _score(
        kl,
        tiny,
        students["coarse"],
        ids_file,
        tmp_path,
        "--groups",
        "none",
        name="p.json",
    )
    assert code == 0 and "routing" not in plain
    assert not (tmp_path / "p_routing.npz").exists()  # off unless asked for
    code, logged, _ = _routing(
        kl, tiny, students["coarse"], np.load(ids_file), tmp_path, "l"
    )
    assert code == 0
    assert plain["student"]["kl_per_window"] == logged["student"]["kl_per_window"]
    assert layerwise._device_pass is original  # the wrapper is gone


def test_a_log_that_misses_a_batch_fails_its_gate(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    real = kl.Routing._windows
    monkeypatch.setattr(
        kl.Routing,
        "_windows",
        lambda self, items, batch: [*real(self, items, batch), [0]],
    )  # one batch of windows more than the layer runs
    code, res, _ = _routing(
        kl, tiny, students["coarse"], np.load(ids_file), tmp_path, "x"
    )
    assert code == 1 and not res["gates"]["routing_log_complete"]["ok"]
    assert "expert forwards for" in res["gates"]["routing_log_complete"]["detail"]


def test_a_forward_that_is_not_the_windows_it_is_said_to_hold_stops_the_run(
    kl, tiny, students, ids_file, tmp_path, monkeypatch
):
    monkeypatch.setenv("HI2_LOOP_SEQS", "2")
    monkeypatch.setattr(
        kl.Routing, "_windows", lambda self, items, batch: [[i] for i in items]
    )
    with pytest.raises(RuntimeError, match="holds 48 tokens, not 1 windows of 24"):
        _routing(kl, tiny, students["coarse"], np.load(ids_file), tmp_path, "y")


@pytest.fixture(scope="module")
def students_mixed(tiny, tmp_path_factory):
    """An output whose experts do not all have one code width, and lie on both sides of the INT2
    class edge (2.90-3.07 bits, 79 % of them below 3; the others are one class each)."""
    return _quantize(tiny, tmp_path_factory.mktemp("kl_mixed"), "mixed", 3.7)


@pytest.mark.parametrize("which", ["coarse", "fine", "mixed"])
def test_the_widths_of_the_experts_are_those_of_the_allocation(
    kl, tiny, students, students_mixed, which
):
    """The mean code width of an expert read from the rows of its tensors equals the one the
    allocation report gives (2 bits plus 2 per INT4 and 6 per INT8 share of the rows),
    averaged over its gate, up and down."""
    from hi2.stack import Hi2Checkpoint

    dst = students_mixed if which == "mixed" else students[which]
    report = json.loads((dst / "hi2_report.json").read_text())["tensors"]
    path = "model.language_model.layers"
    bits = kl.expert_bits(Hi2Checkpoint(dst), path, list(range(6)), 8)
    assert tuple(bits.shape) == (6, 8) and bits.min() >= 2.0 and bits.max() <= 8.0
    wanted = []
    for layer in range(6):
        for e in range(8):
            rows = [
                report[f"{path}.{layer}.mlp.experts.{e}.{p}_proj.weight"]
                for p in ("gate", "up", "down")
            ]
            want = np.mean([2 + 2 * r["frac_int4"] + 6 * r["frac_int8"] for r in rows])
            assert float(bits[layer, e]) == pytest.approx(want, abs=1e-9), (layer, e)
            wanted.append(round(float(want), 9))
    assert (len(set(wanted)) > 1) == (which == "mixed")  # the others are all one width


def test_an_output_of_mixed_widths_weighs_the_widths_by_the_tokens_each_expert_gets(
    kl, tiny, students_mixed, ids_file, tmp_path
):
    """With one width everywhere the mean over the experts and over the assignments are the
    same number; with several they differ and the log reports the second."""
    code, res, npz = _routing(
        kl, tiny, students_mixed, np.load(ids_file), tmp_path, "mixed"
    )
    assert code == 0, res["gates"]
    counts = npz["counts"].astype(np.int64)
    bits = npz["expert_bits"].astype(np.float64)
    whole = res["routing"]["all_layers"]
    by_tokens = float((counts * bits).sum() / counts.sum())
    assert whole["mean_bits"] == pytest.approx(by_tokens, rel=1e-6)
    assert abs(by_tokens - float(bits.mean())) > 1e-4  # the router prefers some experts
    int2 = bits < 3.0
    assert 0 < int2.sum() < int2.size  # both classes are there
    assert whole["experts"]["int2"] == pytest.approx(float(int2.mean()))
    assert whole["assignments"]["int2"] == pytest.approx(
        float((counts * int2).sum() / counts.sum())
    )
    assert whole["assignments"]["int2"] != pytest.approx(
        whole["experts"]["int2"], abs=1e-3
    )
    for name, g in res["routing"]["groups"].items():  # each window of each group
        slots = g["layers"]
        c = counts[:, slots]
        want = ((c * bits[slots]).sum((1, 2)) / c.sum((1, 2))).tolist()
        assert g["mean_bits_per_window"] == pytest.approx(want, rel=1e-6), name
