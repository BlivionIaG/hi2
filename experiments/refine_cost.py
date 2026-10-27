# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""What local code refinement costs: seconds per sweep against the GPTQ encode of the same rows,
on tensor shapes of real models (random data, so only the time means anything).

    python refine_cost.py --device cuda:0 --shapes 9216x2560 2560x9216 17408x5120 5120x17408
"""

import argparse
import time

import torch

from hi2.hessian import encode_rows_gptq, inverse_factor
from hi2.refine import refine_encoded


def timed(fn, device):
    torch.cuda.synchronize(device)
    t = time.time()
    out = fn()
    torch.cuda.synchronize(device)
    return out, time.time() - t


def measure(dev, r: int, k: int, nb: int, blk: int, passes: int) -> tuple[float, float]:
    """``(GPTQ seconds, refinement seconds)`` for random rows of shape ``r x k``."""
    torch.manual_seed(0)
    x = torch.randn(min(4 * k, 4096), k, device=dev) * torch.exp(
        0.5 * torch.randn(k, device=dev)
    )
    hs = (x.T @ x) / x.shape[0]
    w = torch.randn(r, k, device=dev) * 0.3
    u = inverse_factor(hs)
    enc, t_gptq = timed(
        lambda: encode_rows_gptq(w, hs.diagonal().clone(), u, nb, blk, "normal"), dev
    )
    _, t_ref = timed(lambda: refine_encoded(enc, w, hs, nb, blk, passes), dev)
    return t_gptq, t_ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shapes", nargs="+", default=["2560x9216"])
    ap.add_argument("--config", default="2:32", help="bits:block")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device(a.device)
    nb, blk = (int(x) for x in a.config.split(":"))
    measure(dev, 256, 512, nb, blk, 1)  # warm-up: kernel build and allocator
    for shape in a.shapes:
        r, k = (int(x) for x in shape.split("x"))
        t_gptq, t_ref = measure(dev, r, k, nb, blk, a.passes)
        print(
            f"{r:>6} x {k:<6} INT{nb}/B{blk}: GPTQ {t_gptq:7.2f} s   refine {a.passes} sweeps {t_ref:7.2f} s"
            f"   ({t_ref / a.passes:6.2f} s per sweep, {t_ref / a.passes / t_gptq:4.1f}x GPTQ per sweep)",
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
