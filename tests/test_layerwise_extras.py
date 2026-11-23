# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The extra inputs, the sequences-per-forward bound and the dtype switch of the phase-1
layer pass (``hi2.layerwise._run``), on a toy layer. Pure torch, CPU."""

import math

import pytest
import torch
from torch import nn

from hi2.layerwise import _layer_pass, _run
from hi2.stats import ColumnStats

K, T, N = 16, 5, 5
CPU = torch.device("cpu")
CALLS: list[tuple[int, torch.dtype, bool]] = []  # (sequences, hidden dtype, has aux)


class Toy(nn.Module):
    """A layer whose first parameter is fp32 (like a looped layer's residual query), with
    one Linear and an optional extra input."""

    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(K))
        self.lin = nn.Linear(K, K, bias=False)

    def forward(self, x, aux=None, scale=1.0):
        CALLS.append((x.shape[0], x.dtype, aux is not None))
        y = self.lin(x.to(self.lin.weight.dtype)) * scale
        return y if aux is None else y + aux.sum(-2)


@pytest.fixture
def layer():
    torch.manual_seed(0)
    CALLS.clear()
    return Toy()


def _data(n, dtype=torch.float32):
    g = torch.Generator().manual_seed(1)
    hidden = [torch.randn(1, T, K, generator=g).to(dtype) for _ in range(n)]
    aux = [torch.randn(1, T, 3, K, generator=g).to(dtype) for _ in range(n)]
    return hidden, aux


def _pass(layer, hidden, extras=None, **kw):
    (outs, stats, _, _), *_ = _layer_pass(
        layer,
        [CPU],
        len(hidden),
        hidden,
        {"scale": 2.0},
        {"lin": "w"},
        {"lin": K},
        extras=extras,
        **kw,
    )
    return outs, stats["lin"]


@pytest.mark.parametrize("batch", [None, 1, 2, 5, 8])
def test_extra_inputs_reach_their_sequence_whatever_the_batch(layer, batch):
    hidden, aux = _data(N)
    outs, st = _pass(layer, hidden, {"aux": aux}, batch=batch)
    for i in range(N):
        want = layer.lin(hidden[i]) * 2.0 + aux[i].sum(-2)
        assert torch.allclose(outs[i], want, atol=1e-6), i
    # the statistics are the sums over all the tokens, however they were batched
    ref = ColumnStats(K, CPU)
    ref.update(torch.cat(hidden).reshape(-1, K))
    assert st.n == ref.n == N * T
    assert torch.allclose(st.S, ref.S, atol=1e-6)
    # the bound: one forward without it, ceil(N / batch) with it, every one with the extra
    bound = batch or N
    assert [c[0] for c in CALLS] == [
        min(bound, N - s) for s in range(0, N, bound)
    ] and all(c[2] for c in CALLS)
    assert len(CALLS) == math.ceil(N / bound)


@pytest.mark.parametrize("cast", [True, False])
def test_the_hidden_dtype_is_the_layers_first_parameters_unless_the_cast_is_off(
    layer, cast
):
    hidden, aux = _data(2, torch.float16)
    _layer_pass(layer, [CPU], 2, hidden, {}, {}, {}, cast=cast, extras={"aux": aux})
    assert {c[1] for c in CALLS} == {torch.float32 if cast else torch.float16}


def test_the_old_calls_are_unchanged():
    """No extras, no bound, the cast on: one batched forward of all the sequences in the
    layer's first parameter's dtype, and ``_run`` keeps its positional order for callers
    outside the module."""
    CALLS.clear()
    torch.manual_seed(0)
    layer = Toy()
    hidden, _ = _data(3, torch.float16)
    outs, stats, hess, lead = _run(
        layer, CPU, [0, 1, 2], hidden, {}, {"lin": "w"}, {"lin": K}, True, False
    )
    assert [(3, torch.float32, False)] == CALLS
    assert set(outs) == {0, 1, 2} and lead == {"lin": "lin"}
    assert stats["lin"].n == hess["lin"].n == 3 * T
