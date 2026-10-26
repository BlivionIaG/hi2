# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Hessian-aware (error-compensated) rounding, spec 8.3.

The layer output error of a row is ``(w_hat - w)^T Hs (w_hat - w)`` with
``Hs = E[x' x'^T]`` and ``x' = R (x / s)`` the runtime activation (smoothing then
rotation). ``Sigma = E[x x^T]`` is collected once in the original basis and mapped
to the stored basis here. Rounding is GPTQ with the ladder of every block fixed
beforehand, so the codes remain consistent with the decoded fp16 parameters.
"""

import os
from dataclasses import dataclass

import torch

from .encode import EFFORT, Encoded, _safe, fit_ladder, quantize_params
from .format import GROUP, decode_params, hadamard16, ladder_dequantize

BLOCK_COLUMNS = 128  # lazy-update block of GPTQ
ROW_ELEMS = 1 << 26  # rows per chunk are bounded so the working copies fit
# chunks of several jobs batched through one kernel launch sequence (weights per batch)
BATCH_ELEMS = int(os.environ.get("HI2_BATCH_ELEMS", 1 << 27))


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
def encode_rows_gptq_reference(
    w: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    nbits: int,
    block: int,
    effort: str = "normal",
    perm: torch.Tensor | None = None,
) -> Encoded:
    """The original column-by-column implementation, kept verbatim as the reference
    for :func:`encode_rows_gptq` (bit-exactness tests). Quantise ``w[N, K]`` (stored
    basis) with error compensation.

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


@dataclass
class GptqJob:
    """One call of :func:`encode_rows_gptq`: rows ``w[N, K]`` (stored basis), ladder
    weights ``d[K]``, factor ``u[K, K]`` (of the permuted Hessian when ``perm`` is set)."""

    w: torch.Tensor
    d: torch.Tensor
    u: torch.Tensor
    nbits: int
    block: int
    perm: torch.Tensor | None = None


@dataclass
class _Unit:
    """A row chunk of a job, exactly the unit the reference processes in one pass."""

    job: int
    s: int
    wc: torch.Tensor  # [n, K] fp32 working copy, columns in rounding order
    lo_u8: torch.Tensor
    step_u8: torch.Tensor
    consts: torch.Tensor
    lo_d: torch.Tensor  # [n, nb] fp32 decoded
    step_d: torch.Tensor
    blk_of: torch.Tensor  # [K] int64 block of each column in rounding order
    codes: torch.Tensor | None = None  # uint8 [n, K] in rounding order
    comp: torch.Tensor | None = None  # fp32 [n, K] value each code was rounded from


def _prepare(ji: int, job: GptqJob, s: int, step_rows: int, effort: str) -> _Unit:
    w, k, n = job.w, job.w.shape[1], 2**job.nbits
    db = job.d.to(w.device, torch.float32).reshape(1, k // job.block, job.block)
    wc = w[s : s + step_rows].float().clone()
    wb = wc.reshape(-1, k // job.block, job.block)
    lo, step = fit_ladder(wb, db, n, effort)
    lo_u8, step_u8, consts = quantize_params(lo, step)
    lo_d, step_d = decode_params(
        lo_u8, step_u8, consts[:, 0:1], consts[:, 1:2], consts[:, 2:3]
    )
    cols = torch.arange(k, device=w.device)
    if job.perm is not None:
        wc = wc[:, job.perm]
        cols = job.perm.to(w.device)
    return _Unit(
        ji,
        s,
        wc,
        lo_u8,
        step_u8,
        consts,
        lo_d.float(),
        step_d.float(),
        cols // job.block,
    )


def _round_reference(unit: _Unit, job: GptqJob, want_comp: bool = False) -> None:
    """The reference column loop on one unit (``unit.codes``, and with ``want_comp``
    ``unit.comp``, in rounding order)."""
    wc, u, k, n = unit.wc, job.u, unit.wc.shape[1], 2**job.nbits
    lo_d, step_d = unit.lo_d, unit.step_d
    blk_of = unit.blk_of.tolist()
    codes = torch.zeros_like(wc)
    comp = torch.zeros_like(wc) if want_comp else None
    for i1 in range(0, k, BLOCK_COLUMNS):
        i2 = min(i1 + BLOCK_COLUMNS, k)
        w1 = wc[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        u1 = u[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            b = blk_of[j]
            lo_j, st_j = lo_d[:, b], step_d[:, b]
            col = w1[:, i]
            q = torch.round((col - lo_j) / _safe(st_j)).clamp(0, n - 1)
            q = torch.where(st_j == 0, torch.zeros_like(q), q)
            wq = (lo_j + q * st_j).to(torch.float16).float()
            codes[:, j] = q
            if comp is not None:
                comp[:, j] = col
            e = (col - wq) / u1[i, i]
            w1[:, i:] -= e[:, None] * u1[i, i:][None, :]
            err1[:, i] = e
        wc[:, i2:] -= err1 @ u[i1:i2, i2:]
    unit.codes = codes.to(torch.uint8)
    unit.comp = comp


def _round_kernel(
    units: list[_Unit], jobs: list[GptqJob], want_comp: bool = False
) -> None:
    """All ``units`` (same K, same device) through the fused kernel: one launch per
    128-column block for every row; the cross-block GEMM per unit, with the reference's
    shapes and strides so it rounds the same way."""
    from . import gptq_kernel

    dev, k = units[0].wc.device, units[0].wc.shape[1]
    sizes = [x.wc.shape[0] for x in units]
    total = sum(sizes)
    wc = torch.empty(total, k, dtype=torch.float32, device=dev)
    ldp = k // 16
    lo = torch.zeros(total, ldp, dtype=torch.float32, device=dev)
    st = torch.zeros(total, ldp, dtype=torch.float32, device=dev)
    views, off = [], 0
    for x, n in zip(units, sizes):
        wc[off : off + n] = x.wc
        nb = x.lo_d.shape[1]
        lo[off : off + n, :nb] = x.lo_d
        st[off : off + n, :nb] = x.step_d
        views.append((off, n))
        off += n
    row_grp = torch.repeat_interleave(
        torch.arange(len(units), dtype=torch.int32), torch.tensor(sizes)
    ).to(dev)
    us = [
        jobs[x.job].u for x in units
    ]  # the GEMM takes the factor as the reference does
    uk = [u.to(dev, torch.float32).contiguous() for u in us]  # the kernel needs [K, K]
    u_ptrs = torch.tensor([u.data_ptr() for u in uk], dtype=torch.int64).to(dev)
    blk = torch.stack([x.blk_of for x in units]).to(dev, torch.int32).contiguous()
    nmax = torch.tensor(
        [2 ** jobs[x.job].nbits - 1 for x in units], dtype=torch.float32
    ).to(dev)
    codes = torch.zeros(total, k, dtype=torch.uint8, device=dev)
    comp = torch.zeros(total, k, dtype=torch.float32, device=dev) if want_comp else None
    for i1 in range(0, k, BLOCK_COLUMNS):
        i2 = min(i1 + BLOCK_COLUMNS, k)
        err1 = torch.empty(total, i2 - i1, dtype=torch.float32, device=dev)
        gptq_kernel.gptq_block(
            wc, i1, i2 - i1, err1, codes, lo, st, row_grp, u_ptrs, blk, nmax, comp
        )
        if i2 < k:
            for (o, n), u in zip(views, us):
                wc[o : o + n, i2:] -= err1[o : o + n] @ u[i1:i2, i2:]
    for x, (o, n) in zip(units, views):
        x.codes = codes[o : o + n]
        x.comp = None if comp is None else comp[o : o + n]


def _finish(unit: _Unit, job: GptqJob) -> Encoded:
    w, k = job.w, job.w.shape[1]
    codes, comp = unit.codes, unit.comp
    if job.perm is not None:
        p = job.perm.to(codes.device)
        out = torch.zeros_like(codes)
        out[:, p] = codes
        codes = out
        if comp is not None:
            c = torch.zeros_like(comp)
            c[:, p] = comp
            comp = c
    n_rows = codes.shape[0]
    rec = ladder_dequantize(
        codes.reshape(-1, k // job.block, job.block),
        unit.lo_d,
        unit.step_d,
        torch.float32,
    ).float()
    diff = rec.reshape(-1, k) - w[unit.s : unit.s + n_rows].float()
    sse = (job.d.to(diff.device) * diff**2).sum(-1)
    return Encoded(
        codes.to(torch.uint8),
        torch.stack([unit.lo_u8, unit.step_u8], dim=-1),
        unit.consts,
        sse,
        comp,
    )


def _use_kernel(dev: torch.device, backend: str) -> bool:
    if backend == "torch":
        return False
    from . import gptq_kernel

    ok = gptq_kernel.available(dev)
    if backend == "kernel" and not ok:
        raise RuntimeError(f"gptq kernel unavailable on {dev}: {gptq_kernel.failure()}")
    return ok


@torch.no_grad()
def encode_jobs_gptq(
    jobs: list[GptqJob],
    effort: str = "normal",
    backend: str | None = None,
    comp: bool = False,
) -> list[Encoded]:
    """:func:`encode_rows_gptq` for several jobs (tensors, experts, candidate groups)
    at once, bit-identical to calling it on each.

    Each job is cut into the reference's row chunks; chunks with the same K on the same
    device are batched (up to ``BATCH_ELEMS`` weights) through the fused kernel when
    the device supports it (``backend`` ``auto`` | ``kernel`` | ``torch``, default
    ``$HI2_GPTQ_BACKEND`` or auto), else through the reference loop. With ``comp``
    every result carries ``Encoded.comp`` (the value each code was rounded from, the
    QAD latent initialisation), as the reference does."""
    backend = backend or os.environ.get("HI2_GPTQ_BACKEND", "auto")
    units: list[_Unit] = []
    for ji, job in enumerate(jobs):
        n_rows, k = job.w.shape
        step_rows = max(1, ROW_ELEMS // k)
        for s in range(0, n_rows, step_rows):
            units.append(_prepare(ji, job, s, step_rows, effort))
    batches: dict[tuple, list[list[_Unit]]] = {}
    for x in units:
        key = (x.wc.device, x.wc.shape[1])
        bl = batches.setdefault(key, [[]])
        if bl[-1] and sum(y.wc.numel() for y in bl[-1]) + x.wc.numel() > BATCH_ELEMS:
            bl.append([])
        bl[-1].append(x)
    for (dev, _k), bl in batches.items():
        kernel = _use_kernel(dev, backend)
        for batch in bl:
            if kernel:
                _round_kernel(batch, jobs, comp)
            else:
                for x in batch:
                    _round_reference(x, jobs[x.job], comp)
            for x in batch:
                x.wc = None  # free the working copy early
    per_job: list[list[Encoded]] = [[] for _ in jobs]
    for x in units:
        per_job[x.job].append(_finish(x, jobs[x.job]))
    out = []
    for encs in per_job:
        if len(encs) == 1:
            out.append(encs[0])
        else:
            out.append(
                Encoded(
                    *(
                        torch.cat([getattr(o, f) for o in encs])
                        for f in ("codes", "params", "consts", "sse")
                    ),
                    torch.cat([o.comp for o in encs]) if comp else None,
                )
            )
    return out


def encode_rows_gptq(
    w: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    nbits: int,
    block: int,
    effort: str = "normal",
    perm: torch.Tensor | None = None,
    backend: str | None = None,
    comp: bool = True,
) -> Encoded:
    """Quantise ``w[N, K]`` (stored basis) with error compensation.

    With ``perm`` the columns are rounded in that order (``u`` must then be the factor
    of the Hessian permuted the same way, e.g. descending diagonal "act-order"); the
    blocks and the stored codes keep their original column positions.

    ``d`` are the ladder-fit weights (``diag(Hs)``), ``u`` the factor from
    :func:`inverse_factor`. The ladders are fitted on the uncompensated weights and
    then fixed, so the returned parameters are exactly what a decoder sees. Same result
    as :func:`encode_rows_gptq_reference`, with the column loop fused on HIP devices.
    """
    return encode_jobs_gptq(
        [GptqJob(w, d, u, nbits, block, perm)], effort, backend, comp
    )[0]


__all__ = [
    "EFFORT",
    "GptqJob",
    "HessianStats",
    "encode_jobs_gptq",
    "encode_rows_gptq",
    "encode_rows_gptq_reference",
    "inverse_factor",
    "transform_sigma",
]
