# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Stacked layout of routed-expert HI2 tensors (``--moe-layout stacked``).

The per-expert layout writes every expert projection as its own HI2 tensor
(``<pre>.<E>.<proj>.hi2_<field>``, nine arrays each). The stacked layout writes the
experts of one layer and projection group as one set of arrays,
``<pre>.<group>.hi2_<field>`` (e.g. ``model.layers.3.mlp.experts.gate_up.hi2_codes``):

* row buffers (``codes``, ``code_off``, ``blk_params``, ``param_off``, ``row_consts``,
  ``row_nbits``, ``row_block``) are the per-expert buffers concatenated in
  (expert, part) order, with the offsets rebased to the concatenated buffers, so the
  stack is one valid flat container of ``R = sum`` rows;
* ``had`` is ``[E, K/16]`` and ``col_scale`` (if the experts are smoothed) ``[E, K]``:
  one transform per expert, shared by the parts of a group (gate and up);
* ``part_off`` (int32 ``[E*P + 1]``) gives the rows of each (expert, part) slice.

Only the storage changes: :func:`split_stack` returns the per-expert arrays byte for
byte, so both layouts dequantize identically. ``docs/hi2-moe.md`` is the
loader-facing description.
"""

import json
import os
import shutil
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch

from .format import FIELDS, HI2Tensor
from .moe import expert_parts

LAYOUTS = ("per-expert", "stacked")
# group name -> parts in row order (gate before up, as the fork's w13 expects)
GROUPS = {
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
    "w13": ("w1", "w3"),  # Mixtral naming
    "w2": ("w2",),
}
PART_GROUP = {p: (g, i) for g, parts in GROUPS.items() for i, p in enumerate(parts)}
ROW_FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
)
MARK = "part_off"  # the array that tells a stack from a single tensor
INT32_MAX = 2**31 - 1


class StackError(ValueError):
    """A group that cannot be stacked (it is then kept per expert)."""


# ------------------------------------------------------------------ tensors


def build_stack(experts: list[list[HI2Tensor]]) -> dict[str, torch.Tensor]:
    """``experts[e][p]`` (all experts, every part of the group) -> the stacked arrays
    (field name without the ``hi2_`` prefix -> tensor). Raises :class:`StackError` if
    the experts do not share K, a group's parts differ in transform, smoothing is on
    for some experts only, or an offset would not fit int32."""
    if not experts or not experts[0]:
        raise StackError("empty group")
    nparts = len(experts[0])
    k = experts[0][0].shape[1]
    first_cs = experts[0][0].col_scale is not None
    for e, parts in enumerate(experts):
        if len(parts) != nparts:
            raise StackError(f"expert {e} has {len(parts)} parts, expected {nparts}")
        for p in parts:
            if p.shape[1] != k:
                raise StackError(f"expert {e}: K={p.shape[1]}, expected {k}")
            if (p.col_scale is not None) != first_cs:
                raise StackError("smoothing on for some experts only")
            if not torch.equal(p.had, parts[0].had):
                raise StackError(f"expert {e}: parts have different rotation flags")
            if first_cs and not torch.equal(p.col_scale, parts[0].col_scale):
                raise StackError(f"expert {e}: parts have different smoothing")
    pieces = [p for parts in experts for p in parts]
    out: dict[str, torch.Tensor] = {}
    for buf, off in (("codes", "code_off"), ("blk_params", "param_off")):
        bases, total = [], 0
        for p in pieces:
            o = getattr(p, off).to(torch.int64)
            if int(o[0]) != 0:
                raise StackError(f"{off} does not start at 0")
            bases.append(o[:-1] + total)
            total += int(o[-1])
        if total > INT32_MAX:
            raise StackError(f"{off} total {total} exceeds int32")
        out[buf] = torch.cat([getattr(p, buf).reshape(-1) for p in pieces])
        out[off] = torch.cat(bases + [torch.tensor([total])]).to(torch.int32)
    for f in ("row_consts", "row_nbits", "row_block"):
        out[f] = torch.cat([getattr(p, f) for p in pieces])
    rows = torch.tensor([0] + [p.shape[0] for p in pieces], dtype=torch.int64)
    out[MARK] = torch.cumsum(rows, 0).to(torch.int32)
    out["had"] = torch.stack([parts[0].had.reshape(-1) for parts in experts])
    if first_cs:
        out["col_scale"] = torch.stack(
            [parts[0].col_scale.reshape(-1) for parts in experts]
        )
    return out


def stack_dims(d: dict[str, torch.Tensor], nparts: int) -> tuple[int, int]:
    """``(E, K)`` of a stack, checked against ``part_off``."""
    e = d["had"].shape[0]
    if d[MARK].numel() != e * nparts + 1:
        raise ValueError(
            f"part_off has {d[MARK].numel()} entries, expected {e * nparts + 1}"
        )
    return e, d["had"].shape[1] * 16


def split_stack(d: dict[str, torch.Tensor], nparts: int) -> list[list[HI2Tensor]]:
    """Inverse of :func:`build_stack`: the per-expert tensors, byte-identical to the
    ones that were stacked (fresh contiguous copies)."""
    e_count, k = stack_dims(d, nparts)
    po = d[MARK].to(torch.int64)
    co = d["code_off"].to(torch.int64)
    bo = d["param_off"].to(torch.int64)
    out = []
    for e in range(e_count):
        parts = []
        for p in range(nparts):
            r0, r1 = int(po[e * nparts + p]), int(po[e * nparts + p + 1])
            c0, c1, b0, b1 = int(co[r0]), int(co[r1]), int(bo[r0]), int(bo[r1])
            parts.append(
                HI2Tensor(
                    codes=d["codes"][c0:c1].clone(),
                    code_off=(co[r0 : r1 + 1] - c0).to(d["code_off"].dtype),
                    blk_params=d["blk_params"][2 * b0 : 2 * b1].clone(),
                    param_off=(bo[r0 : r1 + 1] - b0).to(d["param_off"].dtype),
                    row_consts=d["row_consts"][r0:r1].clone(),
                    row_nbits=d["row_nbits"][r0:r1].clone(),
                    row_block=d["row_block"][r0:r1].clone(),
                    had=d["had"][e].clone(),
                    shape=(r1 - r0, k),
                    col_scale=d["col_scale"][e].clone() if "col_scale" in d else None,
                )
            )
        out.append(parts)
    return out


# ------------------------------------------------------------- name helpers


def split_base(base: str) -> tuple[str, str] | None:
    """``(pre, group)`` of a stack base ``<pre>.<group>`` (``pre`` ends in ``experts``)."""
    pre, _, group = base.rpartition(".")
    if group in GROUPS and pre.endswith("experts"):
        return pre, group
    return None


def member_base(pre: str, e: int, part: str) -> str:
    return f"{pre}.{e}.{part}"


def hi2_bases(names) -> tuple[list[str], list[str]]:
    """``(single, stacked)`` HI2 base names among checkpoint tensor names."""
    names = set(names)
    stacked = sorted(
        n.removesuffix(f".hi2_{MARK}") for n in names if n.endswith(f".hi2_{MARK}")
    )
    st = set(stacked)
    single = sorted(
        b
        for b in (
            n.removesuffix(".hi2_codes") for n in names if n.endswith(".hi2_codes")
        )
        if b not in st
    )
    return single, stacked


def stack_groups(single: list[str]) -> dict[str, list[list[str]]]:
    """Per-expert bases that form complete stacks: ``{stack base: members[e][p]}``.
    A group stacks when every part of every expert 0..E-1 is present."""
    found: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for b in single:
        p = expert_parts(b)
        if p is None or p[2] not in PART_GROUP:
            continue
        g, _ = PART_GROUP[p[2]]
        found.setdefault((p[0], g), {}).setdefault(p[1], {})[p[2]] = b
    out = {}
    for (pre, g), by_e in found.items():
        parts = GROUPS[g]
        if sorted(by_e) != list(range(len(by_e))):
            continue
        if any(set(by_e[e]) != set(parts) for e in by_e):
            continue
        out[f"{pre}.{g}"] = [[by_e[e][p] for p in parts] for e in range(len(by_e))]
    return out


def field_names(base: str, with_scale: bool, stacked: bool) -> list[str]:
    names = [f"{base}.hi2_{f}" for f in FIELDS]
    if with_scale:
        names.append(f"{base}.hi2_col_scale")
    if stacked:
        names.append(f"{base}.hi2_{MARK}")
    return names


def read_fields(f, base: str, stacked: bool = False) -> dict[str, torch.Tensor]:
    """The arrays of one HI2 tensor or stack from an open safetensors file, keyed by
    field name (no prefix)."""
    keys = set(f.keys())
    names = field_names(base, f"{base}.hi2_col_scale" in keys, stacked)
    return {n.rsplit(".hi2_", 1)[1]: f.get_tensor(n) for n in names}


def to_fields(t: HI2Tensor) -> dict[str, torch.Tensor]:
    d = {f: getattr(t, f) for f in FIELDS}
    if t.col_scale is not None:
        d["col_scale"] = t.col_scale
    return d


def from_fields(d: dict[str, torch.Tensor]) -> HI2Tensor:
    return HI2Tensor(
        **{f: d[f] for f in FIELDS},
        shape=(d["row_nbits"].numel(), d["had"].numel() * 16),
        col_scale=d.get("col_scale"),
    )


# ------------------------------------------------------------- checkpoints


class Hi2Checkpoint:
    """Read access to the HI2 tensors of a checkpoint in either layout, by logical
    name (``<base>``, i.e. the source weight name without ``.weight``)."""

    def __init__(self, d: Path):
        from .checkpoint import weight_map

        self.dir = Path(d)
        self.wmap = weight_map(self.dir)
        self.single, self.stacked = hi2_bases(self.wmap)
        cfg = self.dir / "config.json"
        qc = (
            json.loads(cfg.read_text()).get("quantization_config", {})
            if cfg.exists()
            else {}
        )
        self.groups = {
            **GROUPS,
            **{g: tuple(p) for g, p in qc.get("moe_stacks", {}).items()},
        }
        self._cache: dict[str, list[list[HI2Tensor]]] = {}  # last few stacks read

    def parts(self, stack: str) -> tuple[str, ...]:
        return self.groups[stack.rpartition(".")[2]]

    def shard(self, base: str) -> str:
        return self.wmap[f"{base}.hi2_codes"]

    def members(self, stack: str, f=None) -> list[list[HI2Tensor]]:
        if stack in self._cache:
            return self._cache[stack]
        if f is None:
            from safetensors import safe_open

            with safe_open(self.dir / self.shard(stack), "pt") as g:
                d = read_fields(g, stack, stacked=True)
        else:
            d = read_fields(f, stack, stacked=True)
        out = split_stack(d, len(self.parts(stack)))
        if len(self._cache) >= 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[stack] = out
        return out

    def logical(self) -> dict[str, tuple[str, int, int] | None]:
        """Logical base -> ``(stack, expert, part)``, or None for a single tensor. The
        expert count of a stack is read from its ``part_off`` shape."""
        from safetensors import safe_open

        out: dict[str, tuple[str, int, int] | None] = {b: None for b in self.single}
        for s in self.stacked:
            parts = self.parts(s)
            with safe_open(self.dir / self.shard(s), "pt") as f:
                n = f.get_slice(f"{s}.hi2_{MARK}").get_shape()[0]
            pre = s.rpartition(".")[0]
            for e in range((n - 1) // len(parts)):
                for i, p in enumerate(parts):
                    out[member_base(pre, e, p)] = (s, e, i)
        return out

    def get(self, base: str) -> HI2Tensor:
        """One logical HI2 tensor (single, or one expert part of a stack)."""
        if base in self.single:
            from safetensors import safe_open

            with safe_open(self.dir / self.shard(base), "pt") as f:
                return from_fields(read_fields(f, base))
        p = expert_parts(base)
        if p is not None:
            g, i = PART_GROUP.get(p[2], (None, 0))
            s = f"{p[0]}.{g}"
            if s in self.stacked:
                return self.members(s)[p[1]][i]
        raise KeyError(base)

    def iter_shard(self, shard: str, f):
        """``(logical base, HI2Tensor)`` of every HI2 tensor stored in ``shard``."""
        for b in self.single:
            if self.shard(b) == shard:
                yield b, from_fields(read_fields(f, b))
        for s in self.stacked:
            if self.shard(s) != shard:
                continue
            pre = s.rpartition(".")[0]
            for e, parts in enumerate(
                split_stack(read_fields(f, s, stacked=True), len(self.parts(s)))
            ):
                for name, t in zip(self.parts(s), parts):
                    yield member_base(pre, e, name), t


def header_bytes(path: Path) -> int:
    """Length of a safetensors file's JSON header."""
    with open(path, "rb") as fh:
        return int.from_bytes(fh.read(8), "little")


def checkpoint_stats(d: Path) -> dict:
    """Tensor count, header and file sizes of a checkpoint's shards."""
    from .checkpoint import weight_map

    wmap = weight_map(d)
    shards = sorted(set(wmap.values()))
    return {
        "tensors": len(wmap),
        "shards": len(shards),
        "header_bytes": sum(header_bytes(d / s) for s in shards),
        "file_bytes": sum((d / s).stat().st_size for s in shards),
        "per_shard": {
            s: {
                "tensors": sum(1 for v in wmap.values() if v == s),
                "header_bytes": header_bytes(d / s),
                "file_bytes": (d / s).stat().st_size,
            }
            for s in shards
        },
    }


def _plan_stacked(ck: Hi2Checkpoint, files: dict, log) -> dict[str, list[list[str]]]:
    """Complete groups whose experts can be stacked (checked on the small arrays)."""
    plan = {}
    for s, members in stack_groups(ck.single).items():
        try:
            small = []
            for parts in members:
                row = []
                for b in parts:
                    f = files[ck.shard(b)]
                    names = f.keys()
                    cs = f"{b}.hi2_col_scale"
                    row.append(
                        HI2Tensor(
                            codes=torch.empty(0, dtype=torch.int32),
                            code_off=f.get_tensor(f"{b}.hi2_code_off"),
                            blk_params=torch.empty(0, dtype=torch.uint8),
                            param_off=f.get_tensor(f"{b}.hi2_param_off"),
                            row_consts=torch.empty(0, 3, dtype=torch.float16),
                            row_nbits=torch.empty(0, dtype=torch.uint8),
                            row_block=torch.empty(0, dtype=torch.uint8),
                            had=f.get_tensor(f"{b}.hi2_had"),
                            shape=(
                                f.get_slice(f"{b}.hi2_row_nbits").get_shape()[0],
                                f.get_slice(f"{b}.hi2_had").get_shape()[0] * 16,
                            ),
                            col_scale=f.get_tensor(cs) if cs in names else None,
                        )
                    )
                small.append(row)
            build_stack(small)
            plan[s] = members
        except StackError as err:
            log(f"restack: {s} kept per expert ({err})")
    return plan


def restack(
    src: Path, out: Path, layout: str, log=print, overwrite: bool = False
) -> dict:
    """Rewrite checkpoint ``src`` into ``out`` with the routed experts in ``layout``
    (``stacked`` or ``per-expert``). Byte-level data movement only: no tensor is
    re-encoded, dense tensors are copied unchanged, ``config.json`` and
    ``hi2_report.json`` record the layout, other files are copied (symlinks kept as
    symlinks). The stacks of one layer go to one shard (the one holding most of that
    layer's expert tensors)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("restack needs a different output directory")
    if out.exists() and any(out.glob("*.safetensors")) and not overwrite:
        raise SystemExit(f"{out} already holds a checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    ck = Hi2Checkpoint(src)
    shards = sorted(set(ck.wmap.values()))
    # every source tensor goes to an output (name, shard) unless it is a stack member
    moves: dict[str, list] = {
        s: [] for s in shards
    }  # output shard -> [("copy", name) | ("stack", base) | ("split", stack)]
    consumed: set[str] = set()
    with ExitStack() as es:
        files = {s: es.enter_context(safe_open(src / s, "pt")) for s in shards}
        if layout == "stacked":
            plan = _plan_stacked(ck, files, log)
            member_of = {}
            for s, members in plan.items():
                for parts in members:
                    for b in parts:
                        member_of[b] = s
            # one shard per layer (= stack prefix): the shard holding most members
            by_pre: dict[str, Counter] = {}
            for s, members in plan.items():
                c = by_pre.setdefault(s.rpartition(".")[0], Counter())
                for parts in members:
                    for b in parts:
                        c[ck.shard(b)] += 1
            target = {
                s: min(
                    by_pre[s.rpartition(".")[0]].items(), key=lambda kv: (-kv[1], kv[0])
                )[0]
                for s in plan
            }
            placed: set[str] = set()
            for sh in shards:
                for name in files[sh].keys():  # noqa: SIM118
                    b = name.rsplit(".hi2_", 1)[0] if ".hi2_" in name else None
                    if b is not None and b in member_of:
                        consumed.add(name)
                        s = member_of[b]
                        if s not in placed and target[s] == sh:
                            moves[sh].append(("stack", s))
                            placed.add(s)
                        continue
                    moves[sh].append(("copy", name))
            for s in plan:
                if s not in placed:
                    moves[target[s]].append(("stack", s))
            stacks_out = plan
        else:
            stack_set = set(ck.stacked)
            for sh in shards:
                done: set[str] = set()
                for name in files[sh].keys():  # noqa: SIM118
                    b = name.rsplit(".hi2_", 1)[0] if ".hi2_" in name else None
                    if b in stack_set:
                        if b not in done:
                            moves[sh].append(("split", b))
                            done.add(b)
                        continue
                    moves[sh].append(("copy", name))
            stacks_out = {}
        new_map: dict[str, str] = {}
        total = 0
        for sh in shards:
            f = files[sh]
            data: dict[str, torch.Tensor] = {}
            for kind, x in moves[sh]:
                if kind == "copy":
                    data[x] = f.get_tensor(x)
                elif kind == "stack":
                    members = [
                        [from_fields(read_fields(files[ck.shard(b)], b)) for b in parts]
                        for parts in stacks_out[x]
                    ]
                    for fld, t in build_stack(members).items():
                        data[f"{x}.hi2_{fld}"] = t
                else:
                    pre = x.rpartition(".")[0]
                    parts = ck.parts(x)
                    for e, ts in enumerate(
                        split_stack(read_fields(f, x, stacked=True), len(parts))
                    ):
                        for pname, t in zip(parts, ts):
                            b = member_base(pre, e, pname)
                            for fld, v in to_fields(t).items():
                                data[f"{b}.hi2_{fld}"] = v
            tmp = out / (sh + ".tmp")
            save_file(data, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, out / sh)
            for k, v in data.items():
                new_map[k] = sh
                total += v.numel() * v.element_size()
            log(f"restack: wrote {sh} ({len(data)} tensors)")
            del data
    src_idx = src / "model.safetensors.index.json"
    meta = (
        json.loads(src_idx.read_text()).get("metadata", {}) if src_idx.exists() else {}
    )
    meta = dict(meta) | {"total_size": total}
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": meta, "weight_map": new_map}, indent=2)
    )
    groups_used = sorted(
        {s.rpartition(".")[2] for s in (stacks_out if layout == "stacked" else [])}
    )
    cfg = json.loads((src / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    qc.pop("moe_stacks", None)
    qc["moe_layout"] = layout
    if groups_used:
        qc["moe_stacks"] = {g: list(GROUPS[g]) for g in groups_used}
    cfg["quantization_config"] = qc
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    rep = src / "hi2_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        r.setdefault("summary", {})["moe_layout"] = layout
        r["summary"]["moe_stacks"] = len(stacks_out)
        (out / "hi2_report.json").write_text(json.dumps(r, indent=2))
    skip = {"config.json", "model.safetensors.index.json", "hi2_report.json", *shards}
    for entry in src.iterdir():
        if (
            entry.name in skip
            or entry.name.startswith(".")
            or entry.name.endswith(".tmp")
        ):
            continue
        dest = out / entry.name
        if dest.is_symlink() or dest.exists():
            if not overwrite:
                continue
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if entry.is_symlink():
            os.symlink(os.readlink(entry), dest)
        elif entry.is_dir():
            shutil.copytree(entry, dest, symlinks=True)
        else:
            shutil.copy2(entry, dest)
    return {
        "layout": layout,
        "stacks": len(stacks_out),
        "before": checkpoint_stats(src),
        "after": checkpoint_stats(out),
    }
