# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Compute placement: one or several devices (``cpu`` or ``cuda:N``; HIP on ROCm).

Quantisation of a tensor is sharded by output rows across the devices (rows are
independent given the allocation), one thread per device.
"""

import functools
import threading
from concurrent.futures import ThreadPoolExecutor

import torch


def activate(dev: torch.device) -> None:
    """Make ``dev`` the current device of the calling thread (no-op on CPU)."""
    if dev.type != "cpu" and dev.index is not None:
        torch.accelerator.set_device_index(dev.index)


def serialize_triton_autotune() -> None:
    """Make Triton's autotuner safe for one thread per device.

    Two threads that tune the same kernel at once leave an empty cache entry and the second
    one fails (``TypeError: 'NoneType' object is not a mapping``). The gated-delta-rule
    kernels of Qwen3.5-family layers (``fla``) autotune on first use, so a multi-device
    layer-wise pass hit it. Launches are asynchronous, so one lock around ``Autotuner.run``
    costs no device time. Safe to call more than once.
    """
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    orig = autotuner.Autotuner.run
    if getattr(orig, "_hi2_locked", False):
        return
    lock = threading.RLock()

    @functools.wraps(orig)
    def run(self, *args, **kwargs):
        with lock:
            return orig(self, *args, **kwargs)

    run._hi2_locked = True
    autotuner.Autotuner.run = run


class Executor:
    def __init__(self, devices: list[str] | str = "cpu"):
        if isinstance(devices, str):
            devices = [d for d in devices.split(",") if d]
        self.devices = [torch.device(d) for d in devices]
        if not self.devices:
            raise ValueError("no devices")
        if len(self.devices) > 1:
            serialize_triton_autotune()

    @property
    def primary(self) -> torch.device:
        return self.devices[0]

    def shards(self, rows: int):
        """Even contiguous row ranges, one per device (fewer if rows < devices)."""
        n = min(len(self.devices), max(rows, 1))
        bounds = [rows * i // n for i in range(n + 1)]
        return [(self.devices[i], bounds[i], bounds[i + 1]) for i in range(n)]

    def map_rows(self, fn, rows: int) -> list:
        """Run ``fn(device, lo, hi)`` for each shard (threaded when >1 device)."""
        shards = self.shards(rows)

        def run(shard):
            dev, lo, hi = shard
            activate(dev)
            return fn(dev, lo, hi)

        if len(shards) == 1:
            return [run(shards[0])]
        with ThreadPoolExecutor(len(shards)) as pool:
            return list(pool.map(run, shards))
