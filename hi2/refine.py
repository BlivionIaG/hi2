# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Local code refinement after GPTQ rounding (spec 8.9).

GPTQ rounds the columns of a row one after another and pushes each rounding error onto the
columns not yet rounded. It never goes back. A refinement sweep goes back: for every column in
turn it tries to move each code one level up or down and keeps the move when the Hessian
error ``e Hs e^T`` of the row falls (``e`` is the row's error in the stored basis). The ladders
(``lo``, ``step``) stay as they are, so the format and the decoder do not change.

The change of one code costs O(rows) given ``g = e Hs``. To keep memory traffic low the sweep
works in blocks of columns, as GPTQ does: inside a block ``g`` is kept for the block's columns
only, and one matrix product brings the other columns up to date at the end of the block.
"""

import torch

from .format import decode_params

BLOCK_COLUMNS = 128


@torch.no_grad()
def objective(
    w: torch.Tensor,
    codes: torch.Tensor,
    lo: torch.Tensor,
    step: torch.Tensor,
    hs,
    block,
) -> float:
    """Sum over rows of ``e Hs e^T`` for the decoded codes."""
    idx = torch.arange(w.shape[1], device=w.device) // block
    e = (lo[:, idx] + codes.float() * step[:, idx]).half().float() - w
    return float((e @ hs * e).sum())


def _expand(lo: torch.Tensor, step: torch.Tensor, block: int):
    """Per-column ladders ``[N, K]`` from the per-block ones ``[N, K / block]``."""
    return lo.repeat_interleave(block, dim=1), step.repeat_interleave(block, dim=1)


@torch.no_grad()
def refine_expanded(
    w: torch.Tensor,
    codes: torch.Tensor,
    lo_x: torch.Tensor,
    st_x: torch.Tensor,
    top: torch.Tensor,
    hs: torch.Tensor,
    passes: int,
) -> torch.Tensor:
    """Refined codes (float) of rows ``w[N, K]`` (stored basis) with per-column ladders ``lo_x``
    and ``st_x`` and the highest code of each row ``top[N]``; rows of different (bits, block)
    candidates can share one sweep. ``hs`` is the Hessian in the stored basis. The loop stops
    early when a sweep moves no code."""
    n_rows, k = w.shape
    c = codes.float().clone()
    e = (lo_x + c * st_x).half().float() - w
    g = e @ hs
    hd = hs.diagonal()
    rows = torch.arange(n_rows, device=w.device)
    deltas = torch.tensor([[1.0], [-1.0]], device=w.device)
    for _ in range(passes):
        start = c.clone()
        for b0 in range(0, k, BLOCK_COLUMNS):
            b1 = min(b0 + BLOCK_COLUMNS, k)
            gb = g[:, b0:b1].clone()
            dw_block = torch.zeros(n_rows, b1 - b0, device=w.device)
            hb = hs[b0:b1, b0:b1]
            for jj in range(b1 - b0):
                j = b0 + jj
                lo_j, st_j = lo_x[:, j], st_x[:, j]
                cj = c[:, j]
                cur = (lo_j + cj * st_j).half().float()
                cn = cj[None] + deltas  # [2, N]
                dw = (lo_j[None] + cn * st_j[None]).half().float() - cur[None]
                ok = (cn >= 0) & (cn <= top[None]) & (st_j > 0)[None]
                gain = torch.where(
                    ok, -(2 * dw * gb[:, jj][None] + dw * dw * hd[j]), -torch.inf
                )
                best, which = gain.max(0)
                take = best > 0
                dwj = torch.where(take, dw[which, rows], 0.0)
                c[:, j] = cj + torch.where(take, deltas[which, 0], 0.0)
                dw_block[:, jj] = dwj
                gb += dwj[:, None] * hb[jj][None, :]
            g += dw_block @ hs[b0:b1]
        if bool((c == start).all()):
            break
    return c


def refine_codes(
    w: torch.Tensor,
    codes: torch.Tensor,
    lo: torch.Tensor,
    step: torch.Tensor,
    hs: torch.Tensor,
    nbits: int,
    block: int,
    passes: int,
) -> torch.Tensor:
    """:func:`refine_expanded` for rows rounded at one ``(nbits, block)`` with decoded ladders
    ``lo`` and ``step`` ``[N, K / block]``; returns codes of the dtype of ``codes``."""
    lo_x, st_x = _expand(lo, step, block)
    top = torch.full((w.shape[0],), float(2**nbits - 1), device=w.device)
    return refine_expanded(w, codes, lo_x, st_x, top, hs, passes).to(codes.dtype)


def refine_groups(groups, hs: torch.Tensor, passes: int) -> list[torch.Tensor]:
    """One sweep over the rows of every ``(w, enc, nbits, block)`` group of a tensor (the rows
    of different candidates share the column loop); returns the refined codes per group, uint8
    like ``enc.codes``."""
    parts = []
    for w, enc, nbits, block in groups:
        lo, step = decode_params(
            enc.params[..., 0],
            enc.params[..., 1],
            enc.consts[:, 0:1],
            enc.consts[:, 1:2],
            enc.consts[:, 2:3],
        )
        lo_x, st_x = _expand(lo.float(), step.float(), block)
        top = torch.full((w.shape[0],), float(2**nbits - 1), device=w.device)
        parts.append((w, enc.codes.float(), lo_x, st_x, top))
    cat = [torch.cat([p[i] for p in parts]) for i in range(5)]
    out = refine_expanded(*cat, hs, passes)
    sizes = [p[0].shape[0] for p in parts]
    return [c.to(groups[i][1].codes.dtype) for i, c in enumerate(out.split(sizes))]


def refine_encoded(enc, w, hs, nbits: int, block: int, passes: int):
    """``enc`` (a :class:`hi2.encode.Encoded`) with refined codes; the ladders are untouched.
    The ``sse`` field is recomputed for the new codes, with the ladder weights ``diag(hs)``."""
    (codes,) = refine_groups([(w, enc, nbits, block)], hs, passes)
    lo, step = decode_params(
        enc.params[..., 0],
        enc.params[..., 1],
        enc.consts[:, 0:1],
        enc.consts[:, 1:2],
        enc.consts[:, 2:3],
    )
    idx = torch.arange(w.shape[1], device=w.device) // block
    rec = (lo.float()[:, idx] + codes.float() * step.float()[:, idx]).half().float()
    sse = (hs.diagonal() * (rec - w) ** 2).sum(-1)
    return type(enc)(codes, enc.params, enc.consts, sse, enc.comp)
