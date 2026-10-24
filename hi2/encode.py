# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Block encoder for the v3.2 ladder (spec 2.2, 8): activation-weighted least
squares fit of ``lo``/``step``, 8-bit parameter quantisation, code assignment.

Everything is device-generic float32; the fp16/fp32 decode of the block
parameters goes through :func:`hi2.format.decode_params`, so the codes chosen
here are consistent with what a kernel reconstructs.
"""

from dataclasses import dataclass

import torch

from .format import decode_params, ladder_dequantize

EFFORT = {
    "fast": dict(grid=(0.0, 0.06, 0.12), ls=1),
    "normal": dict(grid=(0.0, 0.04, 0.08, 0.12, 0.16, 0.20), ls=2),
    "max": dict(grid=(0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20), ls=4),
}
CHUNK_ELEMS = 1 << 24  # rows per encode call are bounded so temporaries fit


def _safe(step: torch.Tensor) -> torch.Tensor:
    return torch.where(step == 0, torch.ones_like(step), step)


def _recon(wb, lo, step, n):
    c = torch.round((wb - lo.unsqueeze(-1)) / _safe(step).unsqueeze(-1)).clamp(0, n - 1)
    return lo.unsqueeze(-1) + c * step.unsqueeze(-1), c


def _lsq(c, w, d):
    """Weighted least-squares ``w ~ lo + c * step`` per block. c, w, d: [N, nb, B]."""
    sd = d.sum(-1)
    sc = (d * c).sum(-1)
    scc = (d * c * c).sum(-1)
    sw = (d * w).sum(-1)
    scw = (d * c * w).sum(-1)
    det = sd * scc - sc * sc
    ok = det.abs() > 1e-20
    step = torch.where(
        ok,
        (sd * scw - sc * sw) / torch.where(ok, det, torch.ones_like(det)),
        torch.zeros_like(det),
    )
    lo = torch.where(ok, (sw - step * sc) / sd.clamp_min(1e-20), w.mean(-1))
    return lo, step


def fit_ladder(wb: torch.Tensor, db: torch.Tensor, n: int, effort: str = "normal"):
    """Per-block ``(lo, step)`` minimising ``sum d * (w_hat - w)^2``.

    ``wb`` is ``[N, nb, B]`` and ``db`` broadcasts to it. Start from the best of a
    grid of range shrinks of the min-max ladder, then refine by weighted least
    squares given the current codes, keeping a refit only where it lowers the error.
    """
    cfg = EFFORT[effort]
    mn, mx = wb.amin(-1), wb.amax(-1)
    span = mx - mn
    best_lo = best_step = best_err = None
    for a in cfg["grid"]:
        for b in cfg["grid"]:
            lo = mn + a * span
            step = (mx - b * span - lo) / (n - 1)
            rec, _ = _recon(wb, lo, step, n)
            err = (db * (rec - wb) ** 2).sum(-1)
            if best_err is None:
                best_lo, best_step, best_err = lo, step, err
            else:
                m = err < best_err
                best_lo = torch.where(m, lo, best_lo)
                best_step = torch.where(m, step, best_step)
                best_err = torch.where(m, err, best_err)
    for _ in range(cfg["ls"]):
        _, c = _recon(wb, best_lo, best_step, n)
        lo, step = _lsq(c, wb, db.expand_as(wb))
        rec, _ = _recon(wb, lo, step, n)
        err = (db * (rec - wb) ** 2).sum(-1)
        m = (err < best_err) & (step > 0)
        best_lo = torch.where(m, lo, best_lo)
        best_step = torch.where(m, step, best_step)
        best_err = torch.where(m, err, best_err)
    return best_lo, best_step


def quantize_params(lo: torch.Tensor, step: torch.Tensor):
    """8-bit block parameters relative to three fp16 row constants (spec 2.2).

    ``lo``/``step`` are ``[N, nb]``. Returns ``(lo_u8, step_u8, consts[N, 3])`` with
    ``consts = (lo_min, lo_scale, step_scale)`` already rounded to fp16, so the
    byte values are computed against exactly what the decoder will see.
    """
    step = step.clamp_min(0)
    step_scale = (step.amax(-1, keepdim=True) / 255).to(torch.float16)
    ss = step_scale.float()
    step_u8 = torch.where(
        ss > 0, torch.round(step / ss.clamp_min(1e-30)), torch.zeros_like(step)
    ).clamp(0, 255)
    lo_min = lo.amin(-1, keepdim=True).to(torch.float16)
    lo_scale = ((lo.amax(-1, keepdim=True) - lo_min.float()).clamp_min(0) / 255).to(
        torch.float16
    )
    ls = lo_scale.float()
    lo_u8 = torch.where(
        ls > 0,
        torch.round((lo - lo_min.float()) / ls.clamp_min(1e-30)),
        torch.zeros_like(lo),
    ).clamp(0, 255)
    consts = torch.cat([lo_min, lo_scale, step_scale], dim=-1)
    return lo_u8.to(torch.uint8), step_u8.to(torch.uint8), consts


@dataclass
class Encoded:
    codes: torch.Tensor  # uint8 [N, K]
    params: torch.Tensor  # uint8 [N, nb, 2]
    consts: torch.Tensor  # fp16 [N, 3]
    sse: torch.Tensor  # float32 [N], activation-weighted squared error


def encode_rows(
    w: torch.Tensor, d: torch.Tensor, nbits: int, block: int, effort: str = "normal"
) -> Encoded:
    """Quantise ``w[N, K]`` (already in the stored basis) at ``(nbits, block)``;
    ``d[K]`` are the column weights of the error metric."""
    n_rows, k = w.shape
    step_rows = max(1, CHUNK_ELEMS // k)
    out = []
    n = 2**nbits
    db = d.to(w.device, torch.float32).reshape(1, k // block, block)
    for s in range(0, n_rows, step_rows):
        wb = w[s : s + step_rows].float().reshape(-1, k // block, block)
        lo, step = fit_ladder(wb, db, n, effort)
        lo_u8, step_u8, consts = quantize_params(lo, step)
        lo_d, step_d = decode_params(
            lo_u8, step_u8, consts[:, 0:1], consts[:, 1:2], consts[:, 2:3]
        )
        lo_d, step_d = lo_d.float(), step_d.float()
        codes = torch.where(
            step_d.unsqueeze(-1) == 0,
            torch.zeros_like(wb),
            torch.round((wb - lo_d.unsqueeze(-1)) / _safe(step_d).unsqueeze(-1)),
        ).clamp(0, n - 1)
        rec = ladder_dequantize(codes, lo_d, step_d, torch.float32).float()
        sse = (db * (rec - wb) ** 2).sum((-1, -2))
        out.append(
            Encoded(
                codes.reshape(-1, k).to(torch.uint8),
                torch.stack([lo_u8, step_u8], dim=-1),
                consts,
                sse,
            )
        )
    if len(out) == 1:
        return out[0]
    return Encoded(
        *(
            torch.cat([getattr(o, f) for o in out])
            for f in ("codes", "params", "consts", "sse")
        )
    )
