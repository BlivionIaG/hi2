# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Is the float16 matrix product and its backward any good on this GPU?

The QAD in float16 (docs/qad.md, "The half-precision math, measured") runs but does not
learn: the training loss rises above the held-out loss and stalls, which is what a
gradient that is mostly noise looks like. This measures the matrix products directly:
the same product and the same backward in float32 and in float16 (the same numbers, cast),
and the relative difference of the result and of the two gradients.

    python fp16_gemm_check.py --m 2048 --k 5120 --n 17408
"""

import argparse
import time

import torch


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """The relative difference of two tensors."""
    num = ((a.float() - b.float()) ** 2).sum().item()
    den = (b.float() ** 2).sum().item()
    return (num / den) ** 0.5 if den else float("nan")


def run(x32, w32, g32, dtype, reduced=True, repeat=3):
    """The product and its backward in ``dtype``; the same numbers as the float32 run."""
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    x = x32.to(dtype).detach().requires_grad_(True)
    w = w32.to(dtype).detach().requires_grad_(True)
    g = g32.to(dtype)
    for _ in range(repeat):  # warm-up
        y = x @ w
        (y * g).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        x.grad = None
        w.grad = None
        y = x @ w
        (y * g).sum().backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / repeat
    return y.detach(), x.grad.detach(), w.grad.detach(), dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--n", type=int, default=17408)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    print(f"torch {torch.__version__}, hip {torch.version.hip}")
    print(
        f"{torch.cuda.get_device_properties(0).gcnArchName}, "
        f"allow_fp16_reduced_precision_reduction="
        f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}"
    )
    print(f"product [{a.m}, {a.k}] x [{a.k}, {a.n}]")
    torch.manual_seed(0)
    x32 = torch.randn(a.m, a.k, device="cuda")
    w32 = torch.randn(a.k, a.n, device="cuda")
    g32 = torch.randn(a.m, a.n, device="cuda")

    y32, gx32, gw32, t32 = run(x32, w32, g32, torch.float32, repeat=a.repeat)
    for reduced in (True, False):
        y16, gx16, gw16, t16 = run(
            x32, w32, g32, torch.float16, reduced=reduced, repeat=a.repeat
        )
        print(
            f"float16 (reduced precision reduction {reduced}): "
            f"{1e3 * t16:.1f} ms against {1e3 * t32:.1f} ms in float32 "
            f"({t32 / t16:.2f}x)"
        )
        print(f"  result        relative difference {rel(y16, y32):.3e}")
        print(f"  grad input   relative difference {rel(gx16, gx32):.3e}")
        print(f"  grad weight  relative difference {rel(gw16, gw32):.3e}")
        print(
            f"  grad weight norms: float32 {gw32.norm().item():.4e}, "
            f"float16 {gw16.norm().item():.4e}"
        )


if __name__ == "__main__":
    main()
