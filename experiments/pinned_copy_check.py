# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Do the pinned staging copies give the same numbers, and what do they cost?

The 27B QAD moves tens of gigabytes per layer between the host's hidden-state
buffers and the cards, all of it pageable. The GPU's copy engine addresses those host
pages directly, and on this driver that ends in a protection fault on a host page
after a couple of hours (three 27B runs died that way, in both directions).
``hi2.pinned.PinnedWindow`` routes the copies through a small pinned window.

This checks that the staged copies are bit-identical to the direct ones (both
directions, with and without a dtype cast, and for a tensor larger than the window so
the chunked path runs) and measures the overhead against the direct copy.

    python pinned_copy_check.py
"""

import argparse
import time

import torch

from hi2.pinned import PinnedWindow, module_to_device, module_to_host


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-identical."""
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


def timeit(fn, repeat=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--window", type=int, default=64 << 20, help="pinned window, bytes")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    stager = PinnedWindow(args.window)
    shape = (args.seq_len, args.hidden)
    nbytes = args.seq_len * args.hidden * 4

    src = torch.randn(*shape, dtype=torch.float32)
    dev_src = src.to(dev)

    ok = True

    # host -> device
    direct = src.to(dev)
    staged = stager.to_device(src, dev)
    good = same(direct, staged)
    ok &= good
    print(f"host->device  {nbytes / 2**20:.0f} MiB  identical: {good}")

    # device -> host
    direct = dev_src.cpu()
    staged = stager.to_host(dev_src)
    good = same(direct, staged)
    ok &= good
    print(f"device->host  {nbytes / 2**20:.0f} MiB  identical: {good}")

    # with a dtype cast (the half-precision math path)
    direct = src.to(dev, dtype=torch.float16)
    staged = stager.to_device(src, dev, dtype=torch.float16)
    good = same(direct, staged)
    ok &= good
    print(f"host->device float16 cast    identical: {good}")

    # a tensor larger than the window: the chunked path
    big = torch.randn(shape[0] * 4, shape[1], dtype=torch.float32)
    direct = big.to(dev)
    staged = stager.to_device(big, dev)
    good = same(direct, staged)
    ok &= good
    print(f"host->device  {big.numel() * 4 / 2**20:.0f} MiB (chunked)  identical: {good}")

    # a whole module, the way the pipeline moves a decoder layer onto a card
    mod = torch.nn.Sequential(
        torch.nn.Linear(shape[1], shape[1], bias=False),
        torch.nn.Linear(shape[1], shape[1], bias=False),
    ).to(dev)
    ref = [p.detach().clone() for p in mod.parameters()]
    stager2 = PinnedWindow(args.window)
    module_to_device(mod, dev, dtype=torch.float32, stager=stager2)
    good = all(
        same(a, b.detach()) for a, b in zip(ref, mod.parameters())
    )
    ok &= good
    print(f"module->device (staged)      identical: {good}")

    # and back to the host, the way the final logit-KL phase brings the model back
    module_to_host(mod, stager2)
    good = all(
        same(a.detach().cpu(), b.detach().cpu()) for a, b in zip(ref, mod.parameters())
    )
    ok &= good
    print(f"module->host  (staged)      identical: {good}")

    # the cost
    t_direct = timeit(lambda: src.to(dev), args.repeat)
    t_staged = timeit(lambda: stager.to_device(src, dev), args.repeat)
    print(f"\nhost->device  direct {t_direct * 1e3:7.2f} ms   staged {t_staged * 1e3:7.2f} ms")
    t_direct = timeit(lambda: dev_src.cpu(), args.repeat)
    t_staged = timeit(lambda: stager.to_host(dev_src), args.repeat)
    print(f"device->host  direct {t_direct * 1e3:7.2f} ms   staged {t_staged * 1e3:7.2f} ms")

    print("\nOK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
