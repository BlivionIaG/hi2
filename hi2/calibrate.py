# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Per-tensor calibration for flex (spec 8).

Two steps bracket the global allocation (``allocate.py``):

1. :func:`build_table` picks the rotation flags and computes, for every row and
   every candidate ``(nbits, block)``, the activation-weighted squared error.
2. :func:`encode_with_picks` encodes the rows with the configurations the
   allocator chose.
"""

from dataclasses import dataclass, field

import torch

from .encode import encode_rows
from .executor import Executor, activate
from .pinned import PinnedWindow
from .format import (
    GROUP,
    HI2Tensor,
    assemble,
    fwht_groups,
)
from .stats import ColumnStats

ALL_CANDIDATES = [(nb, b) for nb in (2, 4) for b in (16, 32, 64)]
# INT8 rows (v_dot4_i32_i8 in the I8 mode) for very sensitive projections; opt-in via --bits
INT8_CANDIDATES = [(8, 32), (8, 64)]
SELECTABLE = ALL_CANDIDATES + INT8_CANDIDATES
ROTATION_ROWS = 1024  # rows sampled to decide rotation per 64-column chunk
CHUNK = 64


@dataclass
class Config:
    candidates: list[tuple[int, int]] = field(
        default_factory=lambda: list(ALL_CANDIDATES)
    )
    # on | auto | off. Rotation alone hurts once an activation-aware encoder is used
    # (it mixes the channel scales the encoder exploits), but rotation after smoothing
    # helps at every bit width, and rotation helps a data-free encoder a lot
    # (Qwen2.5-0.5B, see the v3.2 spec).
    rotation: str = "on"
    # Smoothing exponent alpha: s = (E[x^2] / geomean) ** (alpha / 2), W -> W * s,
    # x -> x / s before the rotation. 0 disables it; needs calibration statistics.
    smooth: float = 0.5
    # Per-tensor exponent (spec 8.5): when set, every tensor, or group of tensors that share
    # an input (q/k/v, gate/up), keeps the exponent of this set (and ``smooth``) with the
    # lowest weighted INT2/B32 error. Empty = the single exponent ``smooth``.
    smooth_search: tuple[float, ...] = ()
    # gptq | nearest. gptq (spec 8.3) is Hessian-aware error-compensated rounding:
    # on Qwen3.5-0.8B at 3.0 bpw it takes the perplexity ratio from 1.92 to 1.32. It
    # needs calibration data; data-free runs use nearest rounding.
    rounding: str = "gptq"
    # Sequential propagation (spec 8.4): encode layer by layer on the quantized prefix.
    sequential: bool = False
    damp: float = 0.01  # Hessian damping, relative to the mean diagonal
    # round columns in descending Hessian-diagonal order (GPTQ "act-order"); storage unchanged
    act_order: bool = False
    # Local code refinement after GPTQ (spec 8.9, hi2.refine): sweeps over the codes that move
    # one when the Hessian error falls. 0 = off. Skipped with act_order, and for the PTQ
    # encoding QAD starts from: refined codes are a worse start for QAD (docs/format-spec.md, 8.9).
    refine_passes: int = 0
    effort: str = "normal"  # fast | normal | max
    # MoE: an expert that saw fewer than this x its widest input routed tokens tops its
    # statistics (and Hessian) up with the layer average of its projection; 0 = never
    expert_min_tokens: float = 2.0


@dataclass
class TensorTable:
    name: str
    shape: tuple[int, int]
    had: torch.Tensor  # uint8 [K/16]
    sse: torch.Tensor  # float32 [C, R], activation-weighted squared error
    candidates: list[tuple[int, int]]
    energy: float  # weighted energy sum(d * w^2) in the stored basis
    n_tokens: int  # calibration tokens behind the weights (0 = data-free)
    col_scale: torch.Tensor | None = None  # fp16 [K] smoothing vector, or None


def candidate_row_bits(nbits: int, block: int, k: int) -> float:
    return nbits * k + 16 * k / block


def _weights(
    stats: ColumnStats | None, k: int, flags, device, inv_scale=None
) -> torch.Tensor:
    if stats is None:
        return torch.ones(k, device=device)
    return stats.diag(flags, inv_scale).to(device)


def smoothing_scale(stats: ColumnStats | None, alpha: float) -> torch.Tensor | None:
    """fp16 smoothing vector ``s`` (or None): ``s_j = (d_j / geomean(d)) ** (alpha/2)``
    with ``d_j = E[x_j^2]``; rounded to fp16 so offline and device agree."""
    if alpha <= 0 or stats is None or stats.n == 0:
        return None
    d = stats.diag().clamp_min(1e-12)
    s = (d / d.log().mean().exp()).pow(alpha / 2).clamp(0.05, 20.0)
    return s.to(torch.float16).cpu()


def _smoothed(w: torch.Tensor, s: torch.Tensor | None):
    """``(W * s, 1 / s)``; both unchanged/None without smoothing."""
    if s is None:
        return w, None
    sf = s.float()
    return w.float() * sf, 1.0 / sf


def choose_rotation(
    w: torch.Tensor, stats: ColumnStats | None, cfg: Config, device, inv_scale=None
) -> torch.Tensor:
    """Per-64-column-chunk flags ``[K/16]`` uint8 (uniform across rows)."""
    k = w.shape[1]
    ng = k // GROUP
    if cfg.rotation == "off":
        return torch.zeros(ng, dtype=torch.uint8)
    if cfg.rotation == "on":
        return torch.ones(ng, dtype=torch.uint8)
    r = w.shape[0]
    idx = torch.arange(0, r, max(1, r // ROTATION_ROWS))[:ROTATION_ROWS]
    ws = w[idx].to(device, torch.float32)
    err = []
    for rot in (0, 1):
        fl = torch.full((ng,), rot, dtype=torch.uint8)
        wt = fwht_groups(ws, fl) if rot else ws
        d = _weights(stats, k, fl, device, inv_scale)
        enc = encode_rows(wt, d, 2, 64, "fast")
        # encode_rows only returns the total per row; recompute per chunk
        rec_err = _chunk_errors(wt, d, enc, 2, 64)
        err.append(rec_err)
    flags = (err[1] < err[0]).to(torch.uint8).repeat_interleave(CHUNK // GROUP)
    return flags.cpu()


def _chunk_errors(wt, d, enc, nbits, block) -> torch.Tensor:
    """Weighted INT2/B64 error per 64-column chunk (one block per chunk per row)."""
    from .format import decode_params, ladder_dequantize

    r, k = wt.shape
    lo, step = decode_params(
        enc.params[..., 0],
        enc.params[..., 1],
        enc.consts[:, 0:1],
        enc.consts[:, 1:2],
        enc.consts[:, 2:3],
    )
    rec = ladder_dequantize(
        enc.codes.reshape(r, k // block, block), lo.float(), step.float(), torch.float32
    ).float()
    wb = wt.reshape(r, k // block, block)
    db = d.reshape(1, k // block, block)
    return (db * (rec - wb) ** 2).sum((0, 2))


def _fixed_transform(
    ws: list[torch.Tensor], stats: ColumnStats | None, cfg: Config, ex: Executor, alpha
) -> tuple[torch.Tensor | None, torch.Tensor]:
    s16 = smoothing_scale(stats, alpha)
    if cfg.rotation != "auto":
        return s16, choose_rotation(ws[0], stats, cfg, ex.primary)
    cat = torch.cat([_smoothed(w, s16)[0] for w in ws])
    inv = None if s16 is None else 1.0 / s16.float()
    return s16, choose_rotation(cat, stats, cfg, ex.primary, inv)


def transform_error(
    ws: list[torch.Tensor],
    stats: ColumnStats,
    s16: torch.Tensor | None,
    had: torch.Tensor,
    ex: Executor,
) -> float:
    """Weighted INT2/B32 error of ``ws`` under one transform, relative to their weighted
    energy (the reference candidate of the sensitivity probe). A sample of the rows."""
    dev = ex.primary
    inv = None if s16 is None else 1.0 / s16.float()
    sse = energy = 0.0
    for w in ws:
        idx = torch.arange(0, w.shape[0], max(1, w.shape[0] // ROTATION_ROWS))
        wt = _smoothed(w[idx[:ROTATION_ROWS]], s16)[0].to(dev, torch.float32)
        if had.any():
            wt = fwht_groups(wt, had)
        d = _weights(stats, w.shape[1], had, dev, inv)
        sse += float(encode_rows(wt, d, 2, 32, "fast").sse.sum())
        energy += float((d * wt**2).sum())
    return sse / max(energy, 1e-30)


def shared_transform(
    ws: list[torch.Tensor], stats: ColumnStats | None, cfg: Config, ex: Executor
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``(smoothing vector, rotation flags)`` for Linears that receive the same input
    (q, k, v; gate, up): one ``s`` from the shared statistics and one set of flags
    chosen on their rows together, so the runtime transform of the input is one. With
    ``cfg.smooth_search`` the exponent of ``s`` is chosen from that set (spec 8.5)."""
    if not cfg.smooth_search or stats is None or stats.n == 0:
        return _fixed_transform(ws, stats, cfg, ex, cfg.smooth)
    best = None
    for alpha in dict.fromkeys((cfg.smooth, *cfg.smooth_search)):
        s16, had = _fixed_transform(ws, stats, cfg, ex, alpha)
        err = transform_error(ws, stats, s16, had, ex)
        if best is None or err < best[0]:
            best = (err, s16, had)
    return best[1], best[2]


def build_table(
    name: str,
    w: torch.Tensor,
    stats: ColumnStats | None,
    cfg: Config,
    ex: Executor,
    transform: tuple[torch.Tensor | None, torch.Tensor] | None = None,
    n_tokens: int | None = None,
) -> TensorTable:
    """``transform`` = ``(s, had)`` imposed by :func:`shared_transform`; ``n_tokens``
    overrides the reported token count (a borrowed-statistics expert reports its own)."""
    stager = PinnedWindow()
    r, k = w.shape
    if transform is None and cfg.smooth_search:
        transform = shared_transform([w], stats, cfg, ex)
    if transform is None:
        s16 = smoothing_scale(stats, cfg.smooth)
        w, inv = _smoothed(w, s16)
        had = choose_rotation(w, stats, cfg, ex.primary, inv)
    else:
        s16, had = transform  # own copies: safetensors refuses shared storage
        s16 = None if s16 is None else s16.clone()
        had = had.clone()
        w, inv = _smoothed(w, s16)

    def shard(dev, lo, hi):
        wt = stager.to_device(w[lo:hi], dev, dtype=torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        d = _weights(stats, k, had, dev, inv)
        sse = torch.stack(
            [encode_rows(wt, d, nb, b, cfg.effort).sse for nb, b in cfg.candidates]
        )
        return sse.cpu(), float((d * wt**2).sum())

    parts = ex.map_rows(shard, r)
    sse = torch.cat([p[0] for p in parts], dim=1)
    energy = sum(p[1] for p in parts)
    return TensorTable(
        name,
        (r, k),
        had,
        sse,
        list(cfg.candidates),
        energy,
        n_tokens if n_tokens is not None else (stats.n if stats else 0),
        s16,
    )


@dataclass
class EncodeRequest:
    """One tensor for :func:`encode_many`. Requests with the same ``sigma_id`` (the same
    input: q/k/v, gate/up, or one tensor encoded for several allocations) and the same
    smoothing vector and flags share one transformed Hessian and one factor."""

    w: torch.Tensor  # [R, K] original basis
    stats: ColumnStats | None
    table: TensorTable
    picks: torch.Tensor  # [R] candidate index per row
    sigma: torch.Tensor | None = None
    sigma_id: str | None = None
    # a dict: receives ``"comp"``, the [R, K] stored-basis values the codes were rounded
    # from (QAD latent initialisation), see encode_with_picks
    comp_out: dict | None = None


def _factor_key(req: EncodeRequest, i: int):
    t = req.table
    cs = None if t.col_scale is None else t.col_scale.numpy().tobytes()
    sid = req.sigma_id if req.sigma_id is not None else f"#{i}"
    return sid, cs, t.had.numpy().tobytes()


def encode_many(
    reqs: list[EncodeRequest], cfg: Config, ex: Executor
) -> list[HI2Tensor]:
    """Encode several tensors with their allocator picks; the same result as calling
    :func:`encode_with_picks` on each. Every tensor's rows are sharded across the devices
    as there; on each device all GPTQ candidate groups of all tensors go through one
    batched :func:`hi2.hessian.encode_jobs_gptq` call, and the Hessian transform and
    factorisation are done once per distinct input."""
    from .hessian import GptqJob, encode_jobs_gptq, inverse_factor, transform_sigma

    use_gptq = [cfg.rounding == "gptq" and r.sigma is not None for r in reqs]
    prepared = []
    for r in reqs:
        w, inv = _smoothed(r.w, r.table.col_scale)
        prepared.append((w, inv))
    n_dev = max((len(ex.shards(r.table.shape[0])) for r in reqs), default=1)

    stager = PinnedWindow()

    def on_device(i: int):
        dev = ex.devices[i]
        activate(dev)
        factors: dict = {}
        pieces: dict[int, list] = {}
        comps: list = []  # (request, rows, [rows, K] values the codes were rounded from)
        jobs, where, refine_with = [], [], []
        for ri, (r, (w, inv)) in enumerate(zip(reqs, prepared)):
            shards = ex.shards(r.table.shape[0])
            if i >= len(shards):
                continue
            _, lo, hi = shards[i]
            had, k = r.table.had, r.table.shape[1]
            wt = stager.to_device(w[lo:hi], dev, dtype=torch.float32)
            wt = fwht_groups(wt, had) if had.any() else wt
            if use_gptq[ri]:
                fk = _factor_key(r, ri)
                if fk not in factors:
                    hs = transform_sigma(
                        stager.to_device(r.sigma, dev, dtype=torch.float32),
                        r.table.col_scale,
                        had,
                    )
                    d = hs.diagonal().clone()
                    perm = None
                    if cfg.act_order:
                        perm = torch.argsort(d, descending=True)
                        hs = hs[perm][:, perm]
                    u = inverse_factor(hs, cfg.damp)
                    keep = hs if cfg.refine_passes and perm is None else None
                    del hs
                    factors[fk] = (d, u, perm, keep)
                d, u, perm, hs_keep = factors[fk]
            else:
                d = _weights(r.stats, k, had, dev, inv)
            out = pieces.setdefault(ri, [])
            for ci, (nb, b) in enumerate(r.table.candidates):
                rows = (r.picks[lo:hi] == ci).nonzero().flatten()
                if rows.numel() == 0:
                    continue
                if use_gptq[ri]:
                    jobs.append(GptqJob(wt[rows.to(dev)], d, u, nb, b, perm))
                    where.append((ri, rows + lo))
                    refine_with.append(hs_keep if r.comp_out is None else None)
                    out.append(None)  # placeholder keeps the candidate order
                else:
                    enc = encode_rows(wt[rows.to(dev)], d, nb, b, cfg.effort)
                    out.append((rows + lo, enc.codes, enc.params, enc.consts))
                    if r.comp_out is not None:
                        comps.append((ri, rows + lo, stager.to_host(wt[rows.to(dev)].float())))
        want = any(reqs[ri].comp_out is not None for ri, _ in where)
        encs = encode_jobs_gptq(jobs, cfg.effort, comp=want) if jobs else []
        slot = {
            ri: iter([j for j, p in enumerate(pieces[ri]) if p is None])
            for ri in pieces
        }
        if any(h is not None for h in refine_with):
            from .refine import refine_groups

            by_tensor: dict[int, list[int]] = {}
            for x, (ri, _) in enumerate(where):
                if refine_with[x] is not None:
                    by_tensor.setdefault(ri, []).append(x)
            for (
                xs
            ) in by_tensor.values():  # one column loop per tensor, all its candidates
                new = refine_groups(
                    [(jobs[x].w, encs[x], jobs[x].nbits, jobs[x].block) for x in xs],
                    refine_with[xs[0]],
                    cfg.refine_passes,
                )
                for x, codes in zip(xs, new):
                    encs[x] = type(encs[x])(
                        codes, encs[x].params, encs[x].consts, encs[x].sse, encs[x].comp
                    )
        for (ri, rows), enc in zip(where, encs):
            pieces[ri][next(slot[ri])] = (rows, enc.codes, enc.params, enc.consts)
            if reqs[ri].comp_out is not None:
                comps.append((ri, rows, stager.to_host(enc.comp.float())))
        return pieces, comps

    if n_dev == 1:
        per_dev = [on_device(0)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(n_dev) as pool:
            per_dev = list(pool.map(on_device, range(n_dev)))
    out = []
    for ri, r in enumerate(reqs):
        if r.comp_out is not None:
            full = torch.zeros(r.table.shape, dtype=torch.float32)
            for _, cs in per_dev:
                for rj, rows, c in cs:
                    if rj == ri:
                        full[rows.cpu()] = c
            r.comp_out["comp"] = full
        pieces = [p for pd, _ in per_dev for p in pd.get(ri, [])]
        t = r.table
        nbits = torch.tensor([t.candidates[i][0] for i in r.picks.tolist()])
        block = torch.tensor([t.candidates[i][1] for i in r.picks.tolist()])
        out.append(assemble(pieces, nbits, block, t.had, t.shape[1], t.col_scale, stager=stager))
    return out


def encode_with_picks(
    w: torch.Tensor,
    stats: ColumnStats | None,
    table: TensorTable,
    picks: torch.Tensor,
    cfg: Config,
    ex: Executor,
    sigma: torch.Tensor | None = None,
    comp_out: dict | None = None,
) -> HI2Tensor:
    """Encode rows with the candidate index ``picks[r]`` chosen by the allocator.

    With ``comp_out`` (a dict), ``comp_out["comp"]`` receives the ``[R, K]`` stored-basis
    values the codes were rounded from (the compensated weights under gptq, the
    transformed weights under nearest rounding): the QAD latent initialisation."""
    req = EncodeRequest(w, stats, table, picks, sigma, comp_out=comp_out)
    return encode_many([req], cfg, ex)[0]


def encode_with_picks_reference(
    w: torch.Tensor,
    stats: ColumnStats | None,
    table: TensorTable,
    picks: torch.Tensor,
    cfg: Config,
    ex: Executor,
    sigma: torch.Tensor | None = None,
    comp_out: dict | None = None,
) -> HI2Tensor:
    """The per-tensor implementation before :func:`encode_many`, kept as the
    reference of the equality tests (``comp_out`` as in :func:`encode_with_picks`)."""
    r, k = w.shape
    had = table.had
    w, inv = _smoothed(w, table.col_scale)
    nbits = torch.tensor([table.candidates[i][0] for i in picks.tolist()])
    block = torch.tensor([table.candidates[i][1] for i in picks.tolist()])

    use_gptq = cfg.rounding == "gptq" and sigma is not None

    stager = PinnedWindow()

    def shard(dev, lo, hi):
        wt = stager.to_device(w[lo:hi], dev, dtype=torch.float32)
        wt = fwht_groups(wt, had) if had.any() else wt
        if use_gptq:
            from .hessian import (
                encode_rows_gptq_reference as encode_rows_gptq,
            )
            from .hessian import inverse_factor, transform_sigma

            hs = transform_sigma(
                stager.to_device(sigma, dev, dtype=torch.float32),
                table.col_scale,
                had,
            )
            d = hs.diagonal().clone()
            perm = None
            if cfg.act_order:
                perm = torch.argsort(d, descending=True)
                hs = hs[perm][:, perm]
            u = inverse_factor(hs, cfg.damp)
            del hs
        else:
            d = _weights(stats, k, had, dev, inv)
        pieces = []
        for ci, (nb, b) in enumerate(table.candidates):
            rows = (picks[lo:hi] == ci).nonzero().flatten()
            if rows.numel() == 0:
                continue
            if use_gptq:
                enc = encode_rows_gptq(wt[rows.to(dev)], d, u, nb, b, cfg.effort, perm)
            else:
                enc = encode_rows(wt[rows.to(dev)], d, nb, b, cfg.effort)
            pieces.append(
                (rows + lo, stager.to_host(enc.codes), stager.to_host(enc.params), stager.to_host(enc.consts))
            )
            if comp_out is not None:
                c = enc.comp if enc.comp is not None else wt[rows.to(dev)]
                comps.append((rows + lo, stager.to_host(c.float())))
        return pieces

    comps: list = []
    pieces = [p for part in ex.map_rows(shard, r) for p in part]
    if comp_out is not None:
        full = torch.zeros(r, k, dtype=torch.float32)
        for rows, c in comps:
            full[rows] = c
        comp_out["comp"] = full
    return assemble(pieces, nbits, block, had, k, table.col_scale, stager=stager)
