# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The pinned staging window must not be shared between threads.

The encoders shard a tensor across the devices and run the shards in parallel
threads. When they shared one window, the shards overwrote each other's staging
buffer, and the encoded weights came out as another shard's data - which showed
up as a PTQ initialisation that was 60 % worse than it should be, and as a
checkpoint that did not match the trained tensors.

The window keeps its buffers per thread now; this is the test for that.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from hi2.pinned import PinnedWindow

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to pin host memory"
)


@cuda
def test_window_is_thread_safe():
    """Two threads stage different data through one window; each gets its own."""
    dev = torch.device("cuda:0")
    stager = PinnedWindow()
    shape = (512, 512)

    def stage(i: int) -> torch.Tensor:
        src = torch.full(shape, float(i), dtype=torch.float32)
        for _ in range(8):  # repeat: the race needs the threads to interleave
            out = stager.to_device(src, dev)
            torch.cuda.synchronize()
        return out

    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(stage, range(4)))
    for i, out in enumerate(outs):
        assert out.shape == shape
        assert bool((out == float(i)).all()), f"thread {i} got another thread's data"


@cuda
def test_window_round_trip_is_exact():
    """The staged copies are bit-identical to the direct ones."""
    dev = torch.device("cuda:0")
    stager = PinnedWindow()
    src = torch.randn(64, 128, dtype=torch.float32)
    assert bool((stager.to_device(src, dev) == src.to(dev)).all())
    d = src.to(dev)
    assert bool((stager.to_host(d) == d.cpu()).all())
