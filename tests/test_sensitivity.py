# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU probes with real encoding, model forwards, and measured KL effects."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from hi2 import sensitivity
from hi2.calibrate import Config, build_table, encode_with_picks
from hi2.executor import Executor
from hi2.format import dequantize


class _ParallelClassifier(nn.Module):
    """Identical projections with independently controlled influence on logits.

    Centering each projection on its original output makes the reference uniform.
    A zero gain is an unused branch, not an exactly representable weight matrix.
    """

    def __init__(self, weight, features, gains):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=2)
        self.projections = nn.ModuleList(
            nn.Linear(weight.shape[1], weight.shape[0], bias=False) for _ in gains
        )
        with torch.no_grad():
            for projection in self.projections:
                projection.weight.copy_(weight)
        self.register_buffer("features", features)
        self.register_buffer("baseline", features @ weight.T)
        self.gains = gains

    def forward(self, ids):
        features = self.features[ids]
        baseline = self.baseline[ids]
        logits = torch.zeros_like(baseline)
        for projection, gain in zip(self.projections, self.gains):
            logits = logits + gain * (projection(features) - baseline)
        return CausalLMOutput(logits=logits)


@pytest.fixture
def numerical_probe(tmp_path, monkeypatch):
    cfg = Config(
        candidates=[(2, 32)],
        rotation="off",
        smooth=0,
        rounding="nearest",
        effort="fast",
    )
    ex = Executor("cpu")
    weight = torch.stack((torch.linspace(-1, 1, 64), torch.zeros(64)))
    table = build_table("projection.weight", weight, None, cfg, ex)
    encoded = encode_with_picks(
        weight, None, table, torch.zeros(2, dtype=torch.long), cfg, ex
    )
    residual = dequantize(encoded)[0] - weight[0]
    assert residual.square().sum() > 0
    assert table.sse.sum() > 0
    # A quantized branch shifts the two logits apart by its gain. This keeps the
    # KL signal well above fp16 reference-log-prob rounding without mocking it.
    feature = residual / residual.square().sum()
    features = torch.stack((feature, -feature)).to(weight.dtype)
    ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    created = 0

    def make(gains, shares=None):
        nonlocal created
        grouped = shares is not None
        if shares is None:
            shares = [(1.0,) for _ in gains]
        assert len(gains) == len(shares)
        assert all(sum(weights) == 1 for weights in shares)
        branch_gains = [
            gain * share for gain, weights in zip(gains, shares) for share in weights
        ]
        with torch.random.fork_rng(devices=[]):
            model = _ParallelClassifier(weight, features, branch_gains).eval()
        src = tmp_path / f"source-{created}"
        created += 1
        src.mkdir()
        save_file(model.state_dict(), str(src / "model.safetensors"))
        names = list(dict(model.named_parameters()))
        tables = {name: replace(table, name=name) for name in names}
        groups, offset = [], 0
        for weights in shares:
            groups.append(list(zip(names[offset : offset + len(weights)], weights)))
            offset += len(weights)

        def run(*, units=None, floor=0.05):
            # Substitute only the loader boundary. Parameter discovery, table
            # lookup, nearest encoding, dequantization, forwards and KL are real.
            with monkeypatch.context() as patch:
                patch.setattr(
                    sensitivity,
                    "load_probe_model",
                    lambda *_args, **_kwargs: (model, torch.device("cpu"), False),
                )
                result = sensitivity.probe_sensitivity(
                    src,
                    tables,
                    tmp_path / "work",
                    ids,
                    cfg,
                    ex,
                    units=units if units is not None else groups if grouped else None,
                    floor=floor,
                    log=lambda *_: None,
                )
            for parameter in model.parameters():
                torch.testing.assert_close(parameter, weight, rtol=0, atol=0)
            return result

        return run, groups

    return make


@pytest.mark.parametrize(
    "shares",
    [None, [(0.25, 0.75), (1.0,), (0.125, 0.375, 0.5)]],
    ids=["individual", "grouped"],
)
def test_all_zero_effects_keep_neutral_unit_weights(numerical_probe, shares):
    probe, units = numerical_probe((0, 0, 0), shares)
    result = probe(floor=0.2)
    assert result == {name: share for unit in units for name, share in unit}


def test_zero_median_uses_only_positive_effects_for_scale(numerical_probe):
    probe, units = numerical_probe((0, 0, 0, 1, 2))
    result = probe(floor=0.1)
    for unit in units[:3]:
        assert result[unit[0][0]] == 0.1
    lower, upper = (unit[0][0] for unit in units[-2:])
    assert 0.1 < result[lower] < result[upper]
    # With two positive effects, their upper median is the stronger effect.
    assert result[upper] == pytest.approx(1.0)
    assert probe(units=units[-2:], floor=0.1) == {
        lower: result[lower],
        upper: result[upper],
    }
    assert probe(units=list(reversed(units)), floor=0.1) == result


def test_positive_median_preserves_relative_effects(numerical_probe):
    probe, units = numerical_probe((1, 2, 4))
    result = probe()
    lower, middle, upper = (result[unit[0][0]] for unit in units)
    assert 0.05 < lower < middle < upper
    assert middle == pytest.approx(1.0)


@pytest.mark.parametrize("gains", [(1, 2, 4), (0, 0, 0, 1, 2)])
def test_grouped_effects_keep_routing_shares_after_unit_normalization(
    numerical_probe, gains
):
    single_probe, single_units = numerical_probe(gains)
    single = single_probe()
    shares = [(0.125, 0.375, 0.5)] + [(0.25, 0.75)] * (len(gains) - 1)
    grouped_probe, grouped_units = numerical_probe(gains, shares)
    grouped = grouped_probe()
    for single_unit, grouped_unit in zip(single_units, grouped_units):
        unit_weight = single[single_unit[0][0]]
        assert sum(grouped[name] for name, _ in grouped_unit) == pytest.approx(
            unit_weight, rel=1e-4
        )
        for name, share in grouped_unit:
            assert grouped[name] == pytest.approx(unit_weight * share, rel=1e-4)
