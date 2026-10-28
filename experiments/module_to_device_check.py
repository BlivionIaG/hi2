# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Does ``module_to_device`` give the same module as ``Module.to``?

The QAD's propagation runs each decoder layer on a card. The weights used to go
there with ``layer.to(dev)``; they now go through ``hi2.pinned.module_to_device``,
which stages the copy through a pinned window. The layer weights are frozen, so
a difference there would not show up in the training - it would show up in the
propagated hidden states, and only in the layers that are trained *after* the
change. This compares the two, bit for bit, on a module with the awkward parts
(a bias, an int64 buffer, a tied parameter, a parameter that requires grad).

    python module_to_device_check.py
"""

import torch

from hi2.pinned import module_to_device


class Tied(torch.nn.Module):
    """A parameter used by two submodules (like a tied embedding)."""

    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        return x @ self.w


class Block(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.lin = torch.nn.Linear(d, d, bias=True)
        self.norm = torch.nn.LayerNorm(d)
        self.tied = Tied(torch.randn(d, d))
        self.register_buffer("ids", torch.arange(d, dtype=torch.int64))
        self.register_buffer("scale", torch.randn(d))

    def forward(self, x):
        x = self.norm(self.lin(x))
        x = self.tied(x)
        return x * self.scale[self.ids.clamp(max=x.shape[-1] - 1)]


def main() -> int:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    a = Block()
    b = Block()
    b.load_state_dict(a.state_dict())

    a.to(dev)
    module_to_device(b, dev)

    ok = True
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        good = pa.shape == pb.shape and pa.dtype == pb.dtype and bool((pa == pb).all())
        ok &= good
        print(f"param {na:<14} identical: {good}  ({tuple(pa.shape)} {pa.dtype})")
    for (na, ba), (nb, bb) in zip(a.named_buffers(), b.named_buffers()):
        good = ba.shape == bb.shape and ba.dtype == bb.dtype and bool((ba == bb).all())
        ok &= good
        print(f"buffer {na:<14} identical: {good}  ({tuple(ba.shape)} {ba.dtype})")

    x = torch.randn(8, 64, device=dev)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    good = bool((ya == yb).all())
    ok &= good
    print(f"forward output    identical: {good}")
    print(f"a[0] == b[0]: {bool((a.lin.weight == b.lin.weight).all())}")

    print("\nOK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
