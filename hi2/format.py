# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""HadamardINT2-FP16 v3.2: storage format, decode and reference semantics.

Section numbers refer to ``docs/hadamard-int2-v3.2-spec.md``.

* 2.1 blocks of ``B`` in {16, 32, 64} along K, per-row INT2/INT4.
* 2.2 affine ladder ``w = fp16(lo + code * step)`` with 8-bit block parameters
  decoded through three fp16 per-row constants.
* 2.3 optional orthonormal order-16 Hadamard on flagged column groups.
* 2.4 flat per-row buffers, 2.5 bit accounting.
"""

import math
from dataclasses import dataclass

import torch

GROUP = 16
BLOCK_SIZES = (16, 32, 64)
NBITS = (2, 4)
K_MULTIPLE = 64


def hadamard16(dtype: torch.dtype = torch.float64, device="cpu") -> torch.Tensor:
    """Normalised order-16 Walsh-Hadamard matrix: symmetric, ``H @ H = I``."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(GROUP)).to(dtype=dtype, device=device)


def fwht_groups(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Transform the flagged 16-column groups of ``x[..., K]`` (an involution)."""
    k = x.shape[-1]
    if k % GROUP:
        raise ValueError(f"K={k} must be a multiple of {GROUP}")
    if flags.numel() != k // GROUP:
        raise ValueError(f"flags has {flags.numel()} entries, expected {k // GROUP}")
    work = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float64
    xg = x.to(work).reshape(*x.shape[:-1], k // GROUP, GROUP)
    rot = xg @ hadamard16(work, x.device)
    out = torch.where(flags.to(x.device).bool().reshape(-1, 1), rot, xg)
    return out.reshape(x.shape).to(x.dtype)


# ---- block parameters (2.2) --------------------------------------------------
def decode_params(
    lo_u8: torch.Tensor,
    step_u8: torch.Tensor,
    lo_min: torch.Tensor,
    lo_scale: torch.Tensor,
    step_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 ``lo``/``step`` of each block from its two bytes and the row constants
    (broadcast over the block dim): ``lo = fp16(fma_f32(u, lo_scale, lo_min))``,
    ``step = fp16(u * step_scale)``. float64 holds the fma exactly before the one
    rounding to fp32."""
    lo = (
        (lo_u8.double() * lo_scale.double() + lo_min.double()).float().to(torch.float16)
    )
    step = (step_u8.float() * step_scale.float()).to(torch.float16)
    return lo, step


def ladder_dequantize(
    codes: torch.Tensor, lo: torch.Tensor, step: torch.Tensor, dtype=torch.float64
) -> torch.Tensor:
    """``lo + code * step``, one rounding to fp16 (exact in float64; a float32
    ``dtype`` is for candidate search only)."""
    v = lo.to(dtype).unsqueeze(-1) + codes.to(dtype) * step.to(dtype).unsqueeze(-1)
    return v.to(torch.float16)


# ---- code packing (v3.1 6.2, unchanged) --------------------------------------
def pack_codes(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    """``[..., n]`` codes -> ``[..., n * nbits / 32]`` int32, LSB-first."""
    per_word = 32 // nbits
    n = codes.shape[-1]
    if n % per_word:
        raise ValueError(f"{n} codes is not a multiple of {per_word}")
    if int(codes.max()) >= 2**nbits or int(codes.min()) < 0:
        raise ValueError("code outside the ladder")
    c = codes.to(torch.int64).reshape(*codes.shape[:-1], n // per_word, per_word)
    word = torch.zeros(c.shape[:-1], dtype=torch.int64, device=codes.device)
    for i in range(per_word):
        word |= c[..., i] << (nbits * i)
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)


def unpack_codes(words: torch.Tensor, nbits: int) -> torch.Tensor:
    per_word = 32 // nbits
    w = words.to(torch.int64) & 0xFFFFFFFF
    mask = (1 << nbits) - 1
    c = torch.stack([(w >> (nbits * i)) & mask for i in range(per_word)], dim=-1)
    return c.reshape(*words.shape[:-1], -1)


# ---- tensor container (2.4) ---------------------------------------------------
@dataclass
class HI2Tensor:
    """Flat per-tensor buffers of spec 2.4 (CPU tensors)."""

    codes: torch.Tensor  # int32 [sum words_per_row]
    code_off: torch.Tensor  # int32 [R+1]
    blk_params: torch.Tensor  # uint8 [2 * sum K/B_r]: (lo_u8, step_u8) per block
    param_off: torch.Tensor  # int32 [R+1], offsets in blocks
    row_consts: torch.Tensor  # fp16 [R, 3]: lo_min, lo_scale, step_scale
    row_nbits: torch.Tensor  # uint8 [R]
    row_block: torch.Tensor  # uint8 [R]
    had: torch.Tensor  # uint8 [K/16]
    shape: tuple[int, int]
    # Optional smoothing vector s (fp16 [K]): the stored basis is H(W * s) and the
    # runtime activation is H(x / s). None means no smoothing.
    col_scale: torch.Tensor | None = None

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.codes,
                self.code_off,
                self.blk_params,
                self.param_off,
                self.row_consts,
                self.row_nbits,
                self.row_block,
                self.had,
            )
        )


FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
)


def assemble(
    pieces: list[tuple[torch.Tensor, ...]],
    row_nbits: torch.Tensor,
    row_block: torch.Tensor,
    had: torch.Tensor,
    k: int,
    col_scale: torch.Tensor | None = None,
) -> HI2Tensor:
    """Build flat buffers from per-config row groups.

    Each piece is ``(rows, codes[r, K] uint8, params[r, nb, 2] uint8,
    consts[r, 3] fp16)`` for rows sharing one ``(nbits, B)``. The buffers are built on
    the device of the first piece's codes (integer packing and scatters: the same
    result on any device) and returned on the CPU.
    """
    dev = pieces[0][1].device if pieces else torch.device("cpu")
    r = row_nbits.numel()
    nb = row_nbits.to(torch.int64)
    words = (k * nb) // 32
    blocks = k // row_block.to(torch.int64)
    code_off = torch.zeros(r + 1, dtype=torch.int64)
    code_off[1:] = torch.cumsum(words, 0)
    param_off = torch.zeros(r + 1, dtype=torch.int64)
    param_off[1:] = torch.cumsum(blocks, 0)
    code_off_d, param_off_d = code_off.to(dev), param_off.to(dev)
    codes = torch.zeros(int(code_off[-1]), dtype=torch.int32, device=dev)
    params = torch.zeros(int(param_off[-1]) * 2, dtype=torch.uint8, device=dev)
    consts = torch.zeros(r, 3, dtype=torch.float16, device=dev)
    for rows, c, p, rc in pieces:
        rows_c = rows.cpu()
        rows = rows.to(dev)
        nbits = int(nb[rows_c[0]])
        wr = k * nbits // 32
        widx = code_off_d[rows].unsqueeze(1) + torch.arange(wr, device=dev)
        codes[widx.reshape(-1)] = pack_codes(c.to(dev), nbits).reshape(-1)
        nblk = int(blocks[rows_c[0]])
        pidx = (
            param_off_d[rows].unsqueeze(1) + torch.arange(nblk, device=dev)
        ).unsqueeze(-1) * 2
        pidx = pidx + torch.arange(2, device=dev)
        params[pidx.reshape(-1)] = p.to(dev).reshape(-1)
        consts[rows] = rc.to(dev)
    return HI2Tensor(
        codes=codes.cpu(),
        code_off=code_off.to(torch.int32),
        blk_params=params.cpu(),
        param_off=param_off.to(torch.int32),
        row_consts=consts.cpu(),
        row_nbits=row_nbits.to(torch.uint8),
        row_block=row_block.to(torch.uint8),
        had=had.to(torch.uint8).reshape(-1),
        shape=(r, k),
        col_scale=col_scale,
    )


def _on(t: HI2Tensor, device) -> HI2Tensor:
    if device is None:
        return t
    fields = {f: getattr(t, f).to(device) for f in FIELDS}
    cs = None if t.col_scale is None else t.col_scale.to(device)
    return HI2Tensor(**fields, shape=t.shape, col_scale=cs)


def dequantize_stored(t: HI2Tensor, device=None) -> torch.Tensor:
    """fp16 reconstruction in the stored (rotated) basis as float64 ``[R, K]``
    (computed on ``device`` if given; every step is exact in float64, so the result
    does not depend on the device)."""
    t = _on(t, device)
    dev = t.codes.device
    r, k = t.shape
    out = torch.zeros(r, k, dtype=torch.float64, device=dev)
    nb = t.row_nbits.to(torch.int64)
    bl = t.row_block.to(torch.int64)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            wr = k * nbits // 32
            widx = t.code_off[rows].to(torch.int64).unsqueeze(1) + torch.arange(
                wr, device=dev
            )
            codes = unpack_codes(t.codes[widx], nbits).reshape(
                rows.numel(), k // bs, bs
            )
            nblk = k // bs
            pidx = (
                t.param_off[rows].to(torch.int64).unsqueeze(1)
                + torch.arange(nblk, device=dev)
            ).unsqueeze(-1) * 2 + torch.arange(2, device=dev)
            p = t.blk_params[pidx]  # [r, nblk, 2]
            rc = t.row_consts[rows].unsqueeze(1)  # [r, 1, 3]
            lo, step = decode_params(
                p[..., 0], p[..., 1], rc[..., 0], rc[..., 1], rc[..., 2]
            )
            out[rows] = ladder_dequantize(codes, lo, step).double().reshape(-1, k)
    return out


def dequantize(t: HI2Tensor, device=None) -> torch.Tensor:
    """Reconstruction in the original basis (the transform is its own inverse; the
    smoothing vector, if any, is divided back out). On ``device`` if given: the
    Hadamard products are by 1/4 and sums of 16 fp16 values, exact in float64, and the
    division is correctly rounded, so any device gives the same result."""
    w = fwht_groups(dequantize_stored(t, device), t.had)
    if t.col_scale is not None:
        w = w / t.col_scale.to(w.device).double()
    return w


def reference_linear(x: torch.Tensor, t: HI2Tensor) -> torch.Tensor:
    """``x[M, K] @ W.T`` as a kernel computes it: transform the activation with the
    same flags (after dividing by the smoothing vector, if any), then contract
    against the stored-basis reconstruction."""
    xd = x.double()
    if t.col_scale is not None:
        xd = xd / t.col_scale.double()
    return fwht_groups(xd, t.had) @ dequantize_stored(t).T


# ---- accounting (2.5) ---------------------------------------------------------
def metadata_bits(row_nbits, row_block, had) -> int:
    """1 Hadamard bit if all groups agree else K/16; ``ceil(log2(#widths))`` bits per
    row when the rows use more than one code width (1 bit for INT2/INT4 as in v3.1);
    2 bits per row when block sizes differ."""
    had = had.reshape(-1)
    uniform = bool((had == had[0]).all())
    bits = 1 if uniform else had.numel()
    widths = len(set(row_nbits.tolist()))
    if widths > 1:
        bits += row_nbits.numel() * math.ceil(math.log2(widths))
    if len(set(row_block.tolist())) > 1:
        bits += 2 * row_nbits.numel()
    return bits


def block_bits_per_weight(nbits: int, block: int) -> float:
    """Code bits plus the two 8-bit block parameters (row constants excluded)."""
    return nbits + 16 / block


def tensor_bpw(t: HI2Tensor) -> float:
    """Exact bits per weight: code and block-parameter bits, 48/K row constants,
    and the metadata of 2.5."""
    r, k = t.shape
    nb = t.row_nbits.to(torch.float64)
    bl = t.row_block.to(torch.float64)
    row_bits = (nb * k + 16 * k / bl).sum().item()
    meta = metadata_bits(t.row_nbits, t.row_block, t.had)
    extra = 16 * k if t.col_scale is not None else 0
    return (row_bits + 48 * r + meta + extra) / (r * k)
