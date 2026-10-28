# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
# SPDX-License-Identifier: Apache-2.0
"""How much does the pinned staging cost, on the weights?

The pinned window is the fix for the GPU page faults (the copy engine addressing
pageable host memory), and it routes every bulk host-device copy through a mapped
buffer. This measures what that costs on the *weights* - the largest traffic in the
pipeline - against a direct ``.to(device)``.

Two shapes are timed: one weight of the shape the 27B's largest linear has, and a
whole 27B decoder layer (the 7 linears of a Qwen3.8-27B layer, 710 MiB in fp16). The
direct copy is the baseline; the staged copy is what the pipeline does. Every
measurement is repeated and the median is reported, because a single timing on a
loaded box is worth little.

The third variant is the staged copy without the per-chunk device temporary: the
chunk goes straight into the destination slice, so the traffic is one H2D instead of
an H2D plus a device-to-device copy.

    python experiments/staging_rate_check.py
"""

from __future__ import annotations

import statistics
import time

import torch

from hi2.pinned import PinnedWindow, module_to_device

# Qwen3.8-27B: hidden 5120, intermediate 17408.
HIDDEN = 5120
INTERMEDIATE = 17408
LINEARS = {
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (HIDDEN, HIDDEN),
    "v_proj": (HIDDEN, HIDDEN),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}
REPEATS = 3


def _layer() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        *[
            torch.nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)
            for out_f, in_f in LINEARS.values()
        ]
    )


def _bytes(mod: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters())


def _median_time(fn, repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return statistics.median(times)


def _staged_direct_into_slice(src: torch.Tensor, dev) -> torch.Tensor:
    """The staged copy without the per-chunk device temporary."""
    stager = PinnedWindow()
    out = torch.empty(tuple(src.shape), dtype=src.dtype, device=dev)
    rows = stager._rows(src)
    for i in range(0, src.shape[0], rows):
        part = src[i : i + rows]
        b = stager._buf(tuple(part.shape), src.dtype)
        b.copy_(part)
        out[i : i + rows].copy_(b)
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this needs a GPU")
    dev = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}  (median of {REPEATS})")

    w = torch.zeros((INTERMEDIATE, HIDDEN), dtype=torch.float16)
    nb = w.numel() * w.element_size()
    print(f"\none weight {tuple(w.shape)} fp16 = {nb / 2**20:.0f} MiB")
    d = _median_time(lambda: w.to(dev))
    s = _median_time(lambda: PinnedWindow().to_device(w, dev))
    sl = _median_time(lambda: _staged_direct_into_slice(w, dev))
    print(f"  {'direct .to(device)':22s} {d * 1000:8.1f} ms ({nb / 2**30 / d:5.2f} GiB/s)")
    print(f"  {'pinned staging':22s} {s * 1000:8.1f} ms ({nb / 2**30 / s:5.2f} GiB/s)   {s / d:4.2f}x")
    print(f"  {'staged, no temp':22s} {sl * 1000:8.1f} ms ({nb / 2**30 / sl:5.2f} GiB/s)   {sl / d:4.2f}x")

    layer = _layer()
    nb = _bytes(layer)
    print(f"\none 27B layer ({len(LINEARS)} linears) = {nb / 2**20:.0f} MiB")

    # both paths move the parameters, so a fresh layer is built outside the timer
    def timed(fn) -> float:
        fresh = _layer()  # untimed: the build is not the copy
        t0 = time.time()
        fn(fresh)
        return time.time() - t0

    d = statistics.median([timed(lambda m: m.to(dev)) for _ in range(REPEATS)])
    s = statistics.median([timed(lambda m: module_to_device(m, dev)) for _ in range(REPEATS)])
    print(f"  {'direct .to(device)':22s} {d * 1000:8.1f} ms ({nb / 2**30 / d:5.2f} GiB/s)")
    print(f"  {'pinned staging':22s} {s * 1000:8.1f} ms ({nb / 2**30 / s:5.2f} GiB/s)   {s / d:4.2f}x")


if __name__ == "__main__":
    main()
