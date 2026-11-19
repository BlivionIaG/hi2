# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The batched layer pass: the batch-1 kwargs of ``capture_inputs`` broadcast to n."""

import torch
from torch import nn

from hi2.layerwise import _expand_batch, _run, to_device


def test_batch_one_tensors_expand_and_views_stay_shared():
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    cos = torch.zeros(1, 4, 8)
    sin = torch.ones(1, 4, 8)
    kw = {
        "attention_mask": mask,
        "position_embeddings": (cos, sin),
        "cache_position": torch.arange(4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "use_cache": False,
    }

    out = _expand_batch(kw, 3)

    assert out["attention_mask"].shape == (3, 1, 4, 4)
    assert out["attention_mask"].data_ptr() == mask.data_ptr()  # expand, not copy
    assert out["position_embeddings"][0].shape == (3, 4, 8)
    assert out["position_embeddings"][1].shape == (3, 4, 8)
    assert out["position_ids"].shape == (3, 4)
    assert out["cache_position"].shape == (4,)  # 1-D: shared by every sequence
    assert out["use_cache"] is False


def test_scalars_and_missing_batch_dim_pass_through():
    out = _expand_batch({"attention_mask": None, "flag": 1, "t": torch.zeros(1)}, 2)
    assert out["attention_mask"] is None
    assert out["flag"] == 1
    assert out["t"].shape == (1,)  # dim 1, not a batch


def test_kwargs_go_through_the_pinned_stager():
    """Every host tensor of the kwargs reaches the device through the pinned window.

    A plain ``.to(dev)`` copies from pageable host memory, which this driver faults
    on after a couple of hours (``pinned.py``); the QSA layers read ``attention_mask``
    hard enough to hit it first."""

    class Stager:
        def __init__(self):
            self.seen = []

        def to_device(self, src, dev, dtype=None):
            self.seen.append(tuple(src.shape))
            return src  # stands in for the staged device copy

    stager = Stager()
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    cos = torch.zeros(1, 4, 8)
    kw = {
        "attention_mask": mask,
        "position_embeddings": (cos, torch.ones(1, 4, 8)),
        "cache_position": torch.arange(4),
        "use_cache": False,
    }

    out = to_device(kw, torch.device("meta"), stager)

    assert stager.seen == [(1, 1, 4, 4), (1, 4, 8), (1, 4, 8), (4,)]
    assert out["attention_mask"] is mask
    assert out["use_cache"] is False


def test_streamed_microbatches_match_unbatched_stats_outputs_and_partial_tail(
    monkeypatch,
):
    """The bounded cohort forward must be numerically the sum of the full cohort.

    Five sequences with a two-sequence microbatch exercise a partial last chunk;
    statistics are sums, so no batch-mean weighting is allowed.
    """
    torch.manual_seed(23)
    layer = nn.Sequential(nn.Linear(16, 16))
    hidden = [torch.randn(1, 3, 16) for _ in range(5)]
    linears = {"0": "weight"}
    k_of = {"0": 16}

    def run_case(batch: str, micro: str):
        monkeypatch.setenv("HI2_LAYER_BATCH", batch)
        monkeypatch.setenv("HI2_CALIB_MICROBATCH", micro)
        return _run(
            layer,
            torch.device("cpu"),
            list(range(len(hidden))),
            hidden,
            {},
            linears,
            k_of,
            with_hessian=True,
            resident=False,
        )

    ref_outs, ref_stats, ref_hess, ref_leaders = run_case("0", "2")
    got_outs, got_stats, got_hess, got_leaders = run_case("1", "2")

    assert set(got_outs) == set(ref_outs)
    for i in ref_outs:
        torch.testing.assert_close(got_outs[i], ref_outs[i], rtol=1e-5, atol=1e-6)
    assert got_stats["0"].n == ref_stats["0"].n == 15
    torch.testing.assert_close(
        got_stats["0"].S, ref_stats["0"].S, rtol=1e-5, atol=1e-6
    )
    assert got_hess["0"].n == ref_hess["0"].n == 15
    torch.testing.assert_close(
        got_hess["0"].S, ref_hess["0"].S, rtol=1e-4, atol=1e-5
    )
    assert got_leaders == ref_leaders
