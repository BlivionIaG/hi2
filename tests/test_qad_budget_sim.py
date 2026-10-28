# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The step-budget simulator of experiments/qad_budget_sim.py mirrors the budget of
hi2/qad.py:StepBudget and the early stop of hi2/qad.py:train_layer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.qad_budget_sim import (  # noqa: E402
    PRESETS,
    Budget,
    parse,
    simulate,
    val_at,
)

LOG = """QAD: 2 layers (2 trained), 256 train + 8 held-out sequences of 2048 tokens, 300 steps x 4 seqs per layer, lr 0.01/0.05, init weights, buffers float32, training on cuda:0
  L0 step 75/300 train 1.0000e-04 val 9.0000e-05
  L0 step 150/300 train 9.0000e-05 val 8.5000e-05
  L0 step 225/300 train 8.5000e-05 val 8.4000e-05
  L0 step 300/300 train 8.0000e-05 val 8.3000e-05
QAD layer 0/1: 8 linears, held-out nMSE 1.0000e-04 -> 8.3000e-05 (-17.0%, kept trained), 300/300 steps, 300 s, elapsed 5.0 min, ETA 5.0 min
  L1 step 75/300 train 5.0000e-04 val 4.9900e-04
  L1 step 150/300 train 4.9900e-04 val 4.9800e-04
  L1 step 225/300 train 4.9800e-04 val 4.9750e-04
  L1 step 300/300 train 4.9700e-04 val 4.9700e-04
QAD layer 1/1: 7 linears, held-out nMSE 5.0000e-04 -> 4.9700e-04 (-0.6%, kept trained), 300/300 steps, 280 s, elapsed 9.7 min, ETA 0.0 min
"""


def _log(tmp_path):
    p = tmp_path / "qad.log"
    p.write_text(LOG)
    return str(p)


def test_parse_reads_layers_curves_and_times(tmp_path):
    n_layers, n_trained, layers = parse(_log(tmp_path))
    assert (n_layers, n_trained) == (2, 2)
    assert len(layers) == 2
    assert layers[0]["linears"] == 8
    assert layers[0]["before"] == pytest.approx(1.0e-4)
    assert layers[0]["after"] == pytest.approx(8.3e-5)
    assert layers[0]["seconds"] == pytest.approx(300.0)
    assert layers[0]["curve"][-1] == (300, pytest.approx(8.3e-5))


def test_val_at_interpolates_the_measured_curve():
    curve = [(75, 9.0e-5), (150, 8.5e-5), (225, 8.4e-5), (300, 8.3e-5)]
    assert val_at(curve, 1.0e-4, 0) == pytest.approx(1.0e-4)
    assert val_at(curve, 1.0e-4, 75) == pytest.approx(9.0e-5)
    assert val_at(curve, 1.0e-4, 112) == pytest.approx(8.75333e-5)
    assert val_at(curve, 1.0e-4, 400) == pytest.approx(8.3e-5)


def test_budget_gives_every_layer_the_cap_when_it_is_whole():
    b = Budget(steps=300, budget=1.0, min_steps=0, n_layers=4)
    assert [b.allot() for _ in range(4)] == [300, 300, 300, 300]


def test_budget_spreads_the_steps_it_has_and_weights_by_the_previous_gain():
    b = Budget(steps=300, budget=0.5, min_steps=75, n_layers=4)
    assert b.total == 600
    first = b.allot()
    assert first == 300  # the first layer is weighted 2x, the base is 150
    b.record(first, 0.2)
    # one layer done: the base is the budget left over the layers left (300/3 = 100) and
    # the last gain is the mean, so the weight is 1
    second = b.allot()
    assert second == 100
    b.record(second, 0.05)
    # the last gain is now well under the mean, so the weight is clamped to 0.5 and the
    # floor of 75 steps applies
    third = b.allot()
    assert third == 75
    assert b.spent == 400  # the third allotment is not spent until the layer is done
    assert b.left == 2


def test_early_stop_halts_a_layer_that_stops_improving(tmp_path):
    layers = parse(_log(tmp_path))[2]
    cfg = dict(steps=300, budget=1.0, min_gain=0.01, min_steps=75, eval_every=50, evals=4)
    _, rows = simulate(layers, cfg)
    # layer 0 gains 5 % a step at first and less than 1 % after 150 steps: it stops at 200
    assert rows[0]["spent"] == 200
    assert rows[0]["stopped"]
    # layer 1 gains less than 1 % a step from the start: it stops at 100 (the first
    # evaluation that is past min_steps)
    assert rows[1]["spent"] == 100
    assert rows[1]["stopped"]


def test_a_run_without_an_early_stop_reproduces_the_measured_run(tmp_path):
    layers = parse(_log(tmp_path))[2]
    cfg = dict(steps=300, budget=1.0, min_gain=0.0, min_steps=0, eval_every=0, evals=4)
    budget, rows = simulate(layers, cfg)
    assert budget.spent == 600
    assert all(r["spent"] == 300 and not r["stopped"] for r in rows)
    assert all(r["val_final"] == pytest.approx(r["val_full"]) for r in rows)


def test_presets_match_the_ones_of_hi2():
    assert PRESETS["fast"]["budget"] == 0.5
    assert PRESETS["fast"]["min_gain"] == 0.01
    assert PRESETS["fast"]["min_steps"] == 75
    assert PRESETS["fast"]["eval_every"] == 50


def test_the_budget_mirrors_the_one_of_hi2():
    """With torch, the simulator and hi2.qad must give the same allotments."""
    pytest.importorskip("torch")
    from hi2.qad import QadConfig, StepBudget

    for budget, min_steps in ((1.0, 0), (0.5, 75), (0.75, 50)):
        qc = QadConfig(steps=300, budget=budget, min_steps=min_steps)
        mine = Budget(qc.steps, qc.budget, qc.min_steps, 8)
        theirs = StepBudget(qc, 8)
        for i in range(8):
            a, b = mine.allot(), theirs.allot()
            assert a == b, f"layer {i}: {a} != {b}"
            mine.record(a, 0.1 + 0.01 * i)
            theirs.record({"steps": a, "val_before": 1.0, "val_after": 1.0 - 0.1 - 0.01 * i})
