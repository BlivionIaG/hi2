# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``--expert-load-power`` (``hi2.moe.expert_weights``): an expert's allocation weight follows the
tokens it serves to the power p. p = 1 (the default) is the weight the allocator always had, to
the bit; p = 0 weighs every expert of a layer and projection alike; the group stays at the weight
of one dense projection for any p; the option reaches the allocation from the command line."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hi2.moe import expert_parts, expert_weights, group_key

PROJ = ("gate", "up", "down")


def _tables(counts: dict[int, list[int]]) -> dict:
    """The tables of E experts per layer and projection with the given routed-token counts,
    and a dense tensor beside them."""
    t = {"model.layers.0.self_attn.q_proj.weight": SimpleNamespace(n_tokens=7)}
    for li, ns in counts.items():
        for proj in PROJ:
            for e, n in enumerate(ns):
                name = f"model.layers.{li}.mlp.experts.{e}.{proj}_proj.weight"
                t[name] = SimpleNamespace(n_tokens=n)
    return t


def _before_the_power(tables: dict, share_floor: float) -> dict[str, float]:
    """``expert_weights`` as it was before the option (the oracle of the default)."""
    groups: dict[str, list[str]] = {}
    for name in tables:
        if expert_parts(name) is not None:
            groups.setdefault(group_key(name), []).append(name)
    out: dict[str, float] = {}
    for names in groups.values():
        ns = [tables[n].n_tokens for n in names]
        mean = sum(ns) / len(ns)
        for n, c in zip(names, ns):
            share = c / mean if mean > 0 else 1.0
            out[n] = max(share, share_floor) / len(names)
    return out


def _groups(w: dict) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for name, v in w.items():
        by.setdefault(group_key(name), []).append(v)
    return by


# ------------------------------------------------------------------ the weight itself


@pytest.mark.parametrize("seed", range(6))
def test_the_default_is_the_weight_before_the_option_to_the_bit(seed):
    g = np.random.default_rng(seed)
    counts = {li: [int(x) for x in g.integers(0, 400, 16)] for li in range(3)}
    counts[3] = [0] * 16  # nobody routed there
    counts[4] = [1000] + [
        0
    ] * 15  # one expert serves everything: the others sit at the floor
    counts[5] = [int(x) for x in g.integers(0, 3, 16)]  # a few tokens: zeros and ties
    tables = _tables(counts)
    for floor in (0.0, 0.05, 0.5, 2.0):
        want = _before_the_power(tables, floor)
        assert len(want) == 6 * 16 * 3
        assert expert_weights(tables, floor) == want
        assert expert_weights(tables, floor, 1.0) == want
        assert expert_weights(tables, floor, power=1) == want
        assert expert_weights(tables, share_floor=floor) == want


def test_a_power_of_zero_weighs_the_experts_of_a_group_alike():
    tables = _tables({0: [900, 100, 0, 3], 1: [5, 5, 5, 5]})
    for floor in (0.05, 0.0):  # with no floor: 0 ** 0 is 1 as well
        w = expert_weights(tables, floor, 0.0)
        assert set(w) == {n for n in tables if ".experts." in n}  # no dense tensor
        assert all(v == 0.25 for v in w.values())


def test_a_power_keeps_each_group_at_the_weight_of_one_dense_projection():
    g = np.random.default_rng(1)
    tables = _tables({li: [int(x) for x in g.integers(0, 300, 12)] for li in range(3)})
    for power in (0.1, 0.5, 0.9, 1.5, 2.0, 3.0):
        w = expert_weights(tables, 0.05, power)
        assert len(_groups(w)) == 3 * len(PROJ)
        for key, values in _groups(w).items():
            assert sum(values) == pytest.approx(1.0, abs=1e-12), (power, key)
            assert all(v > 0 for v in values)


def test_a_power_by_hand():
    """Loads 100, 25, 25, 50: shares 2, 1/2, 1/2, 1 (mean 50); the weights go as share ** p."""
    tables = _tables({0: [100, 25, 25, 50]})
    names = [f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(4)]
    root = [2**0.5, 0.5**0.5, 0.5**0.5, 1.0]
    w = expert_weights(tables, 0.05, 0.5)
    for n, s in zip(names, root):
        assert w[n] == pytest.approx(s / sum(root), abs=1e-15)
    assert w[names[0]] == pytest.approx(0.3694, abs=5e-5)  # 1.41421 / 3.82843
    square = [4.0, 0.25, 0.25, 1.0]
    w = expert_weights(tables, 0.05, 2.0)
    for n, s in zip(names, square):
        assert w[n] == pytest.approx(s / 5.5, abs=1e-15)
    # between the load-proportional weight and the flat one: the busy expert loses weight
    flat = expert_weights(tables, 0.05, 0.0)[names[0]]
    load = expert_weights(tables, 0.05, 1.0)[names[0]]
    assert flat < expert_weights(tables, 0.05, 0.5)[names[0]] < load


def test_the_ratio_of_two_experts_is_their_load_ratio_to_the_power():
    tables = _tables({0: [80, 40, 20, 10, 50]})
    names = [f"model.layers.0.mlp.experts.{e}.up_proj.weight" for e in range(5)]
    for power in (0.25, 0.5, 1.5):
        w = expert_weights(tables, 0.0, power)
        assert w[names[0]] / w[names[1]] == pytest.approx(2**power)
        assert w[names[1]] / w[names[3]] == pytest.approx(4**power)
        order = sorted(range(5), key=lambda e: w[names[e]])
        assert order == [3, 2, 1, 4, 0]  # the order of the loads, at any power > 0


def test_the_floor_acts_on_the_share_before_the_power():
    tables = _tables(
        {0: [0, 100, 100, 100]}
    )  # mean 75: shares 0 -> 0.05, 4/3, 4/3, 4/3
    names = [f"model.layers.0.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    w = expert_weights(tables, 0.05, 0.5)
    assert w[names[0]] / w[names[1]] == pytest.approx((0.05 / (4 / 3)) ** 0.5)
    w = expert_weights(
        tables, 0.05, 2.0
    )  # above one the floored share is not lifted back
    assert w[names[0]] / w[names[1]] == pytest.approx((0.05 / (4 / 3)) ** 2)
    w = expert_weights(tables, 0.0, 0.5)  # no floor: the unrouted expert weighs nothing
    assert w[names[0]] == 0.0 and w[names[1]] == pytest.approx(1 / 3)


def test_a_group_nobody_routed_to_stays_uniform_at_any_power():
    tables = _tables({0: [0, 0, 0, 0, 0]})
    for power in (0.0, 0.5, 1.0, 2.0):
        assert set(expert_weights(tables, 0.05, power).values()) == {0.2}


def test_a_dense_model_has_no_expert_weights_at_any_power():
    tables = {"model.layers.0.self_attn.q_proj.weight": SimpleNamespace(n_tokens=3)}
    assert expert_weights(tables, 0.05, 0.5) == {} == expert_weights(tables)


# ------------------------------------------------- the option from the command line down


def _tools_dir() -> Path | None:
    cands = (
        [Path(os.environ["INKLING2_TOOLS"])] if os.environ.get("INKLING2_TOOLS") else []
    )
    cands.append(
        Path(__file__).resolve().parents[2] / "vllm-inkling2-merge/tools/rdna/inkling2"
    )
    return next((c for c in cands if (c / "make_tiny_checkpoint.py").exists()), None)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
    d = _tools_dir()
    if d is None:
        pytest.skip("the fork's tools/rdna/inkling2 is not available (INKLING2_TOOLS)")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hi2_test_make_tiny_load_power", d / "make_tiny_checkpoint.py"
    )
    make = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = make
    spec.loader.exec_module(make)
    path = tmp_path_factory.mktemp("load_power_tiny")
    make.write_checkpoint(path, make.TinySpec(num_hidden_layers=6), seed=11)
    return path


def _plan(tiny, tmp_path, capsys, *flags):
    from hi2.__main__ import main

    argv = ["quantize", str(tiny), str(tmp_path / "out"), "--devices", "cpu"]
    argv += ["--no-qad", "--dry-run", "--json", *flags]
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_the_flag_defaults_to_the_old_weight_and_reaches_the_run(
    tiny, tmp_path, capsys
):
    plan = _plan(tiny, tmp_path, capsys)
    assert plan["config"]["expert_load_power"] == 1.0
    plan = _plan(tiny, tmp_path, capsys, "--expert-load-power", "0.5")
    assert plan["config"]["expert_load_power"] == 0.5
    assert plan["config"]["expert_share_floor"] == 0.05  # the neighbour is untouched


def test_a_negative_or_non_finite_power_is_refused(tiny, tmp_path, capsys):
    from hi2.__main__ import main

    for bad in ("-0.5", "nan", "inf"):
        argv = ["quantize", str(tiny), str(tmp_path / "out"), "--devices", "cpu"]
        argv += ["--no-qad", "--dry-run", "--json", f"--expert-load-power={bad}"]
        code = main(argv)
        captured = capsys.readouterr()
        assert code != 0, bad
        assert "expert_load_power" in captured.out + captured.err, bad


def test_the_power_is_part_of_the_probe_key_only_when_it_is_not_one():
    from hi2.cache import probe_key

    def rc(power):
        return SimpleNamespace(
            sens_floor=0.05,
            sens_dtype="float32",
            expert_share_floor=0.05,
            expert_load_power=power,
            cache_devices=0,
            devices=["cpu"],
            sens_batch=1,
            cfg=SimpleNamespace(effort="fast"),
        )

    import torch

    ids = torch.arange(8).reshape(2, 4)
    one = probe_key({"k": 1}, rc(1.0), ids, (2, 16))
    assert "load_power" not in one  # the keys of the runs made before the option
    half = probe_key({"k": 1}, rc(0.5), ids, (2, 16))
    assert half["load_power"] == 0.5 and {**half, "load_power": None} != one
    assert half != probe_key({"k": 1}, rc(0.0), ids, (2, 16))


def test_a_probe_of_another_power_is_not_reused(tmp_path, monkeypatch):
    """The sensitivity probe kept in a workdir answers for the settings it was made with: the
    power is one of them (and a default run keeps the key it always had)."""
    import torch

    import hi2.pipeline as P
    import hi2.sensitivity as S

    calls: list = []
    monkeypatch.setattr(P, "_probe_ids", lambda rc: torch.zeros(1, 4, dtype=torch.long))
    monkeypatch.setattr(P, "_phase1_key", lambda rc: {"k": 1})
    monkeypatch.setattr(
        S, "probe_sensitivity", lambda *a, **kw: calls.append(kw) or {"t": 1.0}
    )

    def probe(power):
        rc = P.RunConfig(
            src=tmp_path,
            dst=tmp_path / "out",
            workdir=tmp_path / "work",
            sens_dtype="float32",
            expert_load_power=power,
        )
        rc.workdir.mkdir(exist_ok=True)
        P._probe(rc, None, {}, {}, lambda *_: None)
        return json.loads((rc.workdir / "sensitivity.key.json").read_text())

    assert "load_power" not in probe(1.0) and len(calls) == 1
    assert "load_power" not in probe(1.0) and len(calls) == 1  # reused
    assert (
        probe(0.5)["load_power"] == 0.5 and len(calls) == 2
    )  # another power: probed again
    assert (
        probe(0.5)["load_power"] == 0.5 and len(calls) == 2
    )  # and reused for that one
    assert probe(0.0)["load_power"] == 0.0 and len(calls) == 3


def test_the_power_reaches_the_weights_the_allocator_gets(tiny, tmp_path, monkeypatch):
    import torch

    import hi2.pipeline as P
    from hi2.calibrate import Config

    seen: dict = {}
    real = P._allocate

    def spy(rc, tables, sens, *a, **kw):
        seen["sens"] = dict(sens)
        seen["tokens"] = {n: t.n_tokens for n, t in tables.items()}
        return real(rc, tables, sens, *a, **kw)

    monkeypatch.setattr(P, "_allocate", spy)
    ids = torch.randint(0, 50, (4, 12), generator=torch.Generator().manual_seed(5))
    for power in (1.0, 0.0):
        rc = P.RunConfig(
            src=tiny,
            dst=tmp_path / f"out{power}",
            workdir=tmp_path / f"work{power}",
            include=r"\.mlp\.experts\.",
            devices=["cpu"],
            compute_dtype="float32",
            target_bpw=3.5,
            cfg=Config(effort="fast"),
            calib_ids=ids,
            calib_samples=4,
            seq_len=12,
            expert_load_power=power,
        )
        P.run(rc, log=lambda *_: None)
        tables = {n: SimpleNamespace(n_tokens=c) for n, c in seen["tokens"].items()}
        assert seen["sens"] == expert_weights(tables, rc.expert_share_floor, power)
        assert len(seen["sens"]) == 6 * 8 * 3
        flat = set(seen["sens"].values())
        assert (flat == {1 / 8}) == (power == 0.0), power  # p = 0: every expert alike
