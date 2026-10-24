# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Calibration statistics: per linear, a 16x16 second-moment block per column
group, ``S_g = E[x_g x_g^T]``. That is enough to get the exact diagonal of the
activation second moment in the *rotated* basis (``diag(H S_g H)``) without
keeping activations, and it is tiny (``K * 16`` floats).
"""

import torch

from .format import GROUP, hadamard16

TOKEN_CHUNK = 4096


class ColumnStats:
    def __init__(self, k: int, device="cpu"):
        if k % GROUP:
            raise ValueError(f"K={k} must be a multiple of {GROUP}")
        self.k = k
        self.S = torch.zeros(
            k // GROUP, GROUP, GROUP, dtype=torch.float64, device=device
        )
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, self.k)
        for s in range(0, x.shape[0], TOKEN_CHUNK):
            xg = x[s : s + TOKEN_CHUNK].to(self.S.device, torch.float32)
            xg = xg.reshape(xg.shape[0], self.k // GROUP, GROUP)
            self.S += torch.einsum("tgi,tgj->gij", xg, xg).double()
        self.n += x.shape[0]

    def merge(self, other: "ColumnStats") -> None:
        self.S += other.S.to(self.S.device)
        self.n += other.n

    def diag(self, flags: torch.Tensor | None = None) -> torch.Tensor:
        """``E[x_j^2]`` per column, in the rotated basis where ``flags`` is set."""
        s = self.S / max(self.n, 1)
        plain = torch.diagonal(s, dim1=1, dim2=2)
        if flags is None:
            return plain.reshape(-1).float()
        h = hadamard16(torch.float64, s.device)
        rot = torch.einsum("ij,gjk,ik->gi", h, s, h)
        f = flags.to(s.device).bool().reshape(-1, 1)
        return torch.where(f, rot, plain).reshape(-1).float()

    def state(self) -> dict:
        return {"S": self.S.cpu(), "n": self.n, "k": self.k}

    @classmethod
    def from_state(cls, st: dict, device="cpu") -> "ColumnStats":
        c = cls(st["k"], device)
        c.S = st["S"].to(device)
        c.n = st["n"]
        return c
