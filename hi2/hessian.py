# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Hessian-aware (error-compensated) rounding, spec 8.3.

The layer output error of a row is ``(w_hat - w)^T Hs (w_hat - w)`` with
``Hs = E[x' x'^T]`` and ``x' = R (x / s)`` the runtime activation (smoothing then
rotation). ``Sigma = E[x x^T]`` is collected once in the original basis and mapped
to the stored basis here. Rounding is GPTQ with the ladder of every block fixed
beforehand, so the codes remain consistent with the decoded fp16 parameters.
"""

import torch

from .encode import EFFORT, Encoded, _safe, fit_ladder, quantize_params
from .format import GROUP, decode_params, hadamard16, ladder_dequantize

BLOCK_COLUMNS = 128  # lazy-update block of GPTQ
ROW_ELEMS = 1 << 26  # rows per chunk are bounded so the working copies fit


class HessianStats:
    """Full second moment ``sum x x^T`` of a Linear's input (fp32, ``[K, K]``)."""

    def __init__(self, k: int, device="cpu"):
        self.k = k
        self.S = torch.zeros(k, k, dtype=torch.float32, device=device)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, chunk: int = 4096) -> None:
        x = x.reshape(-1, self.k)
        for s in range(0, x.shape[0], chunk):
            xc = x[s : s + chunk].to(self.S.device, torch.float32)
            self.S += xc.T @ xc
        self.n += x.shape[0]

    def merge(self, other: "HessianStats") -> None:
        self.S += other.S.to(self.S.device)
        self.n += other.n

    def mean(self) -> torch.Tensor:
        return self.S / max(self.n, 1)


def transform_sigma(
    sigma: torch.Tensor, col_scale: torch.Tensor | None, had: torch.Tensor
) -> torch.Tensor:
    """``Hs = T Sigma T^T`` with ``T = R diag(1/s)`` (R block-diagonal, flagged groups)."""
    k = sigma.shape[0]
    g = k // GROUP
    s = sigma.float()
    if col_scale is not None:
        inv = (1.0 / col_scale.float()).to(s.device)
        s = s * inv[:, None] * inv[None, :]
    if bool(had.any()):
        h = hadamard16(torch.float32, s.device)
        eye = torch.eye(GROUP, dtype=torch.float32, device=s.device)
        m = torch.where(had.to(s.device).bool().reshape(-1, 1, 1), h, eye)
        s4 = s.reshape(g, GROUP, g, GROUP)
        s4 = torch.einsum("gij,gjhk->gihk", m, s4)
        s4 = torch.einsum("gihk,hlk->gihl", s4, m)
        s = s4.reshape(k, k)
    return 0.5 * (s + s.T)


def inverse_factor(hs: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor ``U`` of ``(Hs + damp * mean(diag) I)^-1``; the damping
    is raised tenfold (up to five times) if the factorisation fails."""
    k = hs.shape[0]
    mean = hs.diagonal().mean().clamp_min(1e-12)
    eye = torch.eye(k, dtype=hs.dtype, device=hs.device)
    last = None
    for attempt in range(6):
        h = hs + (damp * mean * 10**attempt) * eye
        try:
            chol = torch.linalg.cholesky(h)
            hinv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(0.5 * (hinv + hinv.T), upper=True)
        except RuntimeError as e:  # not positive definite
            last = e
    raise RuntimeError(f"Hessian is not factorisable even with heavy damping: {last}")


@torch.no_grad()
def encode_rows_gptq(
    w: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    nbits: int,
    block: int,
    effort: str = "normal",
    perm: torch.Tensor | None = None,
) -> Encoded:
    """Quantise ``w[N, K]`` (stored basis) with error compensation.

    With ``perm`` the columns are rounded in that order (``u`` must then be the factor
    of the Hessian permuted the same way, e.g. descending diagonal "act-order"); the
    blocks and the stored codes keep their original column positions.

    ``d`` are the ladder-fit weights (``diag(Hs)``), ``u`` the factor from
    :func:`inverse_factor`. The ladders are fitted on the uncompensated weights and
    then fixed, so the returned parameters are exactly what a decoder sees.
    """
    n_rows, k = w.shape
    step_rows = max(1, ROW_ELEMS // k)
    outs = []
    n = 2**nbits
    db = d.to(w.device, torch.float32).reshape(1, k // block, block)
    for s in range(0, n_rows, step_rows):
        wc = w[s : s + step_rows].float().clone()
        wb = wc.reshape(-1, k // block, block)
        lo, step = fit_ladder(wb, db, n, effort)
        lo_u8, step_u8, consts = quantize_params(lo, step)
        lo_d, step_d = decode_params(
            lo_u8, step_u8, consts[:, 0:1], consts[:, 1:2], consts[:, 2:3]
        )
        lo_d, step_d = lo_d.float(), step_d.float()
        codes = torch.zeros_like(wc)
        comp = torch.zeros_like(wc)  # compensated value each code was rounded from
        if perm is not None:
            wc = wc[:, perm]
            blk_of = (perm // block).tolist()
            perm_l = perm.tolist()
            codes_p = torch.zeros_like(wc)
        for i1 in range(0, k, BLOCK_COLUMNS):
            i2 = min(i1 + BLOCK_COLUMNS, k)
            w1 = wc[:, i1:i2].clone()
            err1 = torch.zeros_like(w1)
            u1 = u[i1:i2, i1:i2]
            for i in range(i2 - i1):
                j = i1 + i
                b = blk_of[j] if perm is not None else j // block
                lo_j, st_j = lo_d[:, b], step_d[:, b]
                col = w1[:, i]
                q = torch.round((col - lo_j) / _safe(st_j)).clamp(0, n - 1)
                q = torch.where(st_j == 0, torch.zeros_like(q), q)
                wq = (lo_j + q * st_j).to(torch.float16).float()
                (codes_p if perm is not None else codes)[:, j] = q
                comp[:, perm_l[j] if perm is not None else j] = col
                e = (col - wq) / u1[i, i]
                w1[:, i:] -= e[:, None] * u1[i, i:][None, :]
                err1[:, i] = e
            wc[:, i2:] -= err1 @ u[i1:i2, i2:]
        if perm is not None:
            codes = torch.zeros_like(codes)
            codes[:, perm] = codes_p
        rec = ladder_dequantize(
            codes.reshape(-1, k // block, block), lo_d, step_d, torch.float32
        ).float()
        diff = rec.reshape(-1, k) - w[s : s + step_rows].float()
        sse = (d.to(diff.device) * diff**2).sum(-1)
        outs.append(
            Encoded(
                codes.to(torch.uint8),
                torch.stack([lo_u8, step_u8], dim=-1),
                consts,
                sse,
                comp,
            )
        )
    if len(outs) == 1:
        return outs[0]
    return Encoded(
        *(
            torch.cat([getattr(o, f) for o in outs])
            for f in ("codes", "params", "consts", "sse", "comp")
        )
    )


__all__ = [
    "EFFORT",
    "HessianStats",
    "encode_rows_gptq",
    "inverse_factor",
    "transform_sigma",
]
