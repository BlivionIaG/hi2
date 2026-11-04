# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Checkpoint I/O: shard enumeration, eligibility, the streaming writer (phase 3)
and verification. Tensor names follow spec 2.4 (``<name>.hi2_<field>``).
"""

import ast
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch

from .calibrate import Config, EncodeRequest, TensorTable, encode_many
from .executor import Executor
from .format import FIELDS, HI2Tensor, dequantize, tensor_bpw
from .paths import check_output_paths
from .runtime import file_digest
from .stats import ColumnStats

# Embeddings, the output head, norms, routers and indexer projections stay dense.
DEFAULT_EXCLUDE = (
    r"(embed|lm_head|norm|\.mlp\.gate(\.weight)?$|router|indexer|conv1d|\.bias$)"
)
SPEC_VERSION = "3.2"


def _json_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON value {value}")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_json_object,
            parse_constant=_invalid_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _local_path(root: Path, name: str, *, contained: bool = True) -> Path:
    """Reject lexical traversal; writes/artifact validation also reject escaping links.

    Read-only source enumeration opts out of physical containment so an explicitly
    chosen Hugging Face snapshot can reference its sibling content-addressed blobs.
    """
    rel = Path(name)
    if not name or rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError(f"{root}: unsafe checkpoint path {name!r}")
    path = root / rel
    if contained and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{path}: checkpoint path escapes {root}")
    return path


def weight_map(d: Path) -> dict[str, str]:
    """Tensor name -> shard file; a single-file checkpoint has no index."""
    from safetensors import safe_open

    d = Path(d)
    idx = _local_path(d, "model.safetensors.index.json", contained=False)
    if idx.exists() or idx.is_symlink():
        result = _read_json(idx).get("weight_map")
        if not isinstance(result, dict) or not result:
            raise ValueError(f"{idx}: weight_map must be a nonempty object")
        for name, shard in result.items():
            if (
                not name
                or not isinstance(shard, str)
                or not shard.endswith(".safetensors")
            ):
                raise ValueError(f"{idx}: invalid weight_map entry {name!r}: {shard!r}")
            _local_path(d, shard, contained=False)
        return result
    with safe_open(_local_path(d, "model.safetensors", contained=False), "pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")  # noqa: SIM118


_LAYER_KEY = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(.*)$")


def make_resolver(wmap: dict[str, str]):
    """Map a module's weight name to the checkpoint's tensor name.

    Multimodal checkpoints often nest the text model (``model.language_model.layers.N``)
    while the text-only class names it ``model.layers.N``. Tries the name itself, the
    usual ``language_model`` rewrites, then a unique match on ``layers.N.<rest>``.
    Returns ``None`` when nothing matches unambiguously.
    """
    index: dict[tuple[int, str], list[str]] = {}
    for key in wmap:
        m = _LAYER_KEY.match(key)
        if m:
            index.setdefault((int(m.group(1)), m.group(2)), []).append(key)

    def resolve(name: str) -> str | None:
        for cand in (
            name,
            name.replace("model.", "model.language_model.", 1),
            "language_model." + name,
        ):
            if cand in wmap:
                return cand
        m = _LAYER_KEY.match(name)
        hits = index.get((int(m.group(1)), m.group(2)), []) if m else []
        return hits[0] if len(hits) == 1 else None

    return resolve


def is_eligible(name: str, shape, dtype, include, exclude) -> bool:
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and dtype.is_floating_point
        and shape[1] % 64 == 0
        and (include is None or include.search(name) is not None)
        and exclude.search(name) is None
    )


def to_tensors(t: HI2Tensor, base: str) -> dict[str, torch.Tensor]:
    out = {f"{base}.hi2_{f}": getattr(t, f) for f in FIELDS if f != "had"}
    out[f"{base}.hi2_had"] = t.had
    if t.col_scale is not None:
        out[f"{base}.hi2_col_scale"] = t.col_scale
    return out


def from_tensors(d: dict[str, torch.Tensor], base: str) -> HI2Tensor:
    g = {f: d[f"{base}.hi2_{f}"] for f in FIELDS}
    return HI2Tensor(
        **g,
        shape=(g["row_nbits"].numel(), g["had"].numel() * 16),
        col_scale=d.get(f"{base}.hi2_col_scale"),
    )


def read_tensors(f, base: str) -> dict[str, torch.Tensor]:
    """The ``<base>.hi2_*`` tensors (including the optional smoothing vector) from an
    open safetensors file."""
    names = [f"{base}.hi2_{fld}" for fld in FIELDS]
    cs = f"{base}.hi2_col_scale"
    if cs in f.keys():  # noqa: SIM118
        names.append(cs)
    return {n: f.get_tensor(n) for n in names}


def table_hessian_path(workdir: Path, name: str) -> Path:
    return workdir / "hessian" / (name.replace("/", "_") + ".pt")


def table_stats_path(workdir: Path, name: str) -> Path:
    return workdir / "stats" / (name.replace("/", "_") + ".pt")


# tensors encoded together (u factors, weights and temporaries on the device)
ENCODE_BATCH_BYTES = int(float(os.environ.get("HI2_ENCODE_BATCH_GB", "6")) * 2**30)


@dataclass
class Output:
    """One checkpoint written by :func:`write_checkpoints`."""

    dst: Path
    picks: dict[str, torch.Tensor]
    meta: dict
    done_dir: Path  # per-shard "finished" markers (resume)
    pre_encoded: dict | None = None
    report: dict = field(default_factory=dict)
    new_map: dict = field(default_factory=dict)
    # trained dense tensors (QAD norms) written instead of the source's
    dense: dict | None = None


def _requests_for(names, path, view, tables, outs, cfg):
    """Load the source tensors, statistics and Hessians of ``names`` (one shard file)
    and build an :class:`EncodeRequest` per tensor and output that still needs it."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return _requests_from(names, f, view, tables, outs, cfg)


def _requests_from(names, f, view, tables, outs, cfg):
    reqs, slots = [], []
    loaded: dict[str, torch.Tensor] = {}
    for name in names:
        need = [
            o for o in outs if not (o.pre_encoded is not None and name in o.pre_encoded)
        ]
        if not need:
            continue
        w = f.get_tensor(name)
        st = view.stats_state(name)
        stats = None
        if st is not None and st["n"] > 0:
            stats = ColumnStats.from_state(st, "cpu")
        sigma = sid = None
        if cfg.rounding == "gptq":
            sid = view.sigma_id(name)
            if sid is not None:  # one load per shared input (gate/up, q/k/v)
                if sid not in loaded:
                    loaded[sid] = view.sigma(name)
                sigma = loaded[sid]
        for o in need:
            reqs.append(
                EncodeRequest(w, stats, tables[name], o.picks[name], sigma, sid)
            )
            slots.append((name, o))
    return reqs, slots


def _batches(names, tables, outs, budget):
    """Consecutive tensors of a shard grouped so that the factors (K^2) and weights of
    a group stay under ``budget`` bytes (fp32) per device."""
    cur, size = [], 0
    for name in names:
        r, k = tables[name].shape
        b = 4 * (k * k + 3 * r * k * len(outs))
        if cur and size + b > budget:
            yield cur
            cur, size = [], 0
        cur.append(name)
        size += b
    if cur:
        yield cur


def _restore_shard_report(o: Output, shard: str, names, tables, log) -> bool:
    """A completed shard is reusable only with its original, bound tensor report."""
    from safetensors import SafetensorError, safe_open

    path, marker = _local_path(o.dst, shard), _local_path(o.done_dir, shard)
    if not path.exists() or not marker.exists():
        return False
    try:
        saved = _read_json(marker)
        expected = {n for n in names if n in tables}
        reports = saved.get("tensors")
        if (
            saved.get("schema_version") != 1
            or saved.get("shard") != shard
            or not isinstance(reports, dict)
            or set(reports) != expected
            or saved.get("sha256") != file_digest(path)
        ):
            raise ValueError("missing or mismatched shard report")
        with safe_open(path, "pt") as f:
            expected_fields = set(names) - expected
            for name in expected:
                base = name.removesuffix(".weight")
                expected_fields.update(f"{base}.hi2_{fld}" for fld in FIELDS)
                if tables[name].col_scale is not None:
                    expected_fields.add(f"{base}.hi2_col_scale")
            if (
                set(f.keys()) != expected_fields
                or set(saved.get("tensor_names", [])) != expected_fields
            ):
                raise ValueError("shard tensor coverage changed")
            for name in expected:
                base = name.removesuffix(".weight")
                d = {
                    n.rsplit(".hi2_", 1)[1]: t for n, t in read_tensors(f, base).items()
                }
                _validate_fields(f"{path}:{base}", d)
                t = next(_container_members(base, d))[1]
                _check_report_entry(name, reports[name], t)
                required = {
                    "shape",
                    "bpw",
                    "pred_rel_err",
                    "frac_int4",
                    "frac_int8",
                    "block_hist",
                    "rotated_groups",
                    "groups",
                    "n_tokens",
                }
                if not required.issubset(reports[name]) or tuple(
                    reports[name]["shape"]
                ) != tuple(tables[name].shape):
                    raise ValueError(f"{name}: incomplete original report")
        o.report.update(reports)
        return True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        SafetensorError,
    ) as exc:
        # Older 'ok' markers do not contain error/calibration statistics. Re-encode
        # instead of inventing a report or publishing reduced coverage.
        log(f"resume {path}: re-encoding ({exc})")
        return False


def write_checkpoints(
    src: Path,
    outs: list[Output],
    workdir: Path,
    tables: dict[str, TensorTable],
    cfg: Config,
    ex: Executor,
    log=print,
) -> None:
    """Phase 3 for one or several outputs (allocations) in one pass over the source:
    each shard is read once, every tensor's Hessian is loaded and factorised once for
    all outputs and all tensors sharing its input, and the tensors of a shard are
    encoded in batches (:func:`hi2.calibrate.encode_many`) while the next batch is read
    from disk. Existing shards are kept only with a complete committed report.
    Destination ownership is established by the pipeline before calling this API."""
    from concurrent.futures import ThreadPoolExecutor

    from safetensors import safe_open
    from safetensors.torch import save_file

    from .cache import StatsView
    from .runtime import atomic_json

    check_output_paths(
        inputs=[src], outputs=[p for o in outs for p in (o.dst, o.done_dir)]
    )
    if not tables:
        raise ValueError("cannot write an HI2 checkpoint with zero expected tensors")

    view = StatsView(workdir)
    wmap = weight_map(src)
    shards = sorted(set(wmap.values()))
    if not set(tables).issubset(wmap):
        raise ValueError("expected quantized tensors are missing from the source index")
    for o in outs:
        for root in (o.dst, o.done_dir):
            for name in [*shards, "config.json", "model.safetensors.index.json"]:
                if _local_path(root, name).is_symlink():
                    raise ValueError(f"{root / name}: refusing to overwrite a symlink")
    for o in outs:
        o.dst.mkdir(parents=True, exist_ok=True)
        o.report.clear()
        o.new_map.clear()
    warned = False
    io = ThreadPoolExecutor(1)
    for shard in shards:
        names = [n for n, s in wmap.items() if s == shard]
        todo = [
            o for o in outs if not _restore_shard_report(o, shard, names, tables, log)
        ]
        if event := getattr(log, "event", None):
            for o in outs:
                if o not in todo:
                    event(
                        "encode",
                        "reused",
                        shard=shard,
                        completed_tensors=len(o.report),
                        total_tensors=len(tables),
                        artifact=str(o.dst),
                    )
        data: dict[int, dict[str, torch.Tensor | None]] = {id(o): {} for o in outs}
        with safe_open(src / shard, "pt") as f:
            for name in names:
                if name in tables:
                    continue
                t = f.get_tensor(name) if todo else None
                for o in outs:
                    v = t
                    if o.dense is not None and name in o.dense and o in todo:
                        v = o.dense[name].contiguous()
                    data[id(o)][name] = v if o in todo else None
            enc_names = [n for n in names if n in tables]
            for o in outs:
                if o in todo:
                    continue
                for name in enc_names:  # report was restored with the committed shard
                    base = name.removesuffix(".weight")
                    for fld in FIELDS:
                        data[id(o)][f"{base}.hi2_{fld}"] = None
                    if tables[name].col_scale is not None:
                        data[id(o)][f"{base}.hi2_col_scale"] = None
            if todo:
                if cfg.rounding == "gptq" and not warned:
                    missing = [n for n in enc_names if view.sigma_path(n) is None]
                    if missing:
                        log(
                            f"warning: {len(missing)} tensors have no Hessian; nearest rounding"
                        )
                        warned = True
                budget = ENCODE_BATCH_BYTES // (
                    2 if cfg.refine_passes else 1
                )  # hs stays next to u
                groups = list(_batches(enc_names, tables, todo, budget))
                sp = src / shard
                nxt = (
                    io.submit(_requests_for, groups[0], sp, view, tables, todo, cfg)
                    if groups
                    else None
                )
                for gi, group in enumerate(groups):
                    reqs, slots = nxt.result()
                    if gi + 1 < len(groups):
                        nxt = io.submit(
                            _requests_for, groups[gi + 1], sp, view, tables, todo, cfg
                        )
                    encoded = encode_many(reqs, cfg, ex) if reqs else []
                    by = {(n, id(o)): t for (n, o), t in zip(slots, encoded)}
                    for name in group:
                        base = name.removesuffix(".weight")
                        for o in todo:
                            if o.pre_encoded is not None and name in o.pre_encoded:
                                t = o.pre_encoded[name]
                            else:
                                t = by[(name, id(o))]
                            data[id(o)].update(to_tensors(t, base))
                            o.report[name] = _tensor_report(
                                tables[name], o.picks[name], t
                            )
                            rep = o.report[name]
                            tag = "" if len(outs) == 1 else f"[{o.dst.name}] "
                            log(
                                f"{tag}{name} {tuple(tables[name].shape)} bpw={rep['bpw']:.3f} "
                                f"rel_err={rep['pred_rel_err']:.4f} "
                                f"int4={rep['frac_int4']:.2f}"
                                + (
                                    f" int8={rep['frac_int8']:.2f}"
                                    if rep["frac_int8"]
                                    else ""
                                )
                            )
                            if event := getattr(log, "event", None):
                                event(
                                    "encode",
                                    "running",
                                    tensor=name,
                                    completed_tensors=len(o.report),
                                    total_tensors=len(tables),
                                    artifact=str(o.dst),
                                )
                    del reqs, encoded, by
        for o in outs:
            out = data[id(o)]
            if o in todo:
                ordered = {}
                for name in names:  # keep the source order of tensors in the shard
                    if name in tables:
                        base = name.removesuffix(".weight") + ".hi2_"
                        ordered.update(
                            {k: v for k, v in out.items() if k.startswith(base)}
                        )
                    else:
                        ordered[name] = out[name]
                path = _local_path(o.dst, shard)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as tmp:
                    temporary = Path(tmp.name)
                try:
                    save_file(ordered, str(temporary), metadata={"format": "pt"})
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                marker = _local_path(o.done_dir, shard)
                atomic_json(
                    marker,
                    {
                        "schema_version": 1,
                        "shard": shard,
                        "sha256": file_digest(path),
                        "tensor_names": list(ordered),
                        "tensors": {n: o.report[n] for n in enc_names},
                    },
                )
                out = ordered
            for k in out:
                o.new_map[k] = shard
    io.shutdown()
    for o in outs:
        if set(o.report) != set(tables):
            raise ValueError(f"{o.dst}: incomplete tensor report after shard recovery")
        _finish_output(src, o, shards)


def _finish_output(src: Path, o: Output, shards) -> None:
    from .runtime import atomic_json

    index = {"metadata": {}, "weight_map": o.new_map}
    atomic_json(o.dst / "model.safetensors.index.json", index)
    cfgj = _read_json(src / "config.json")
    cfgj["quantization_config"] = {
        "quant_method": "rdna_hint_flex",
        "version": SPEC_VERSION,
        **o.meta,
    }
    atomic_json(o.dst / "config.json", cfgj)
    skip = {"config.json", "model.safetensors.index.json", *shards}
    for entry in src.iterdir():
        if entry.name in skip or entry.name.startswith("."):
            continue
        link = o.dst / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)


def _tensor_report(table: TensorTable, picks: torch.Tensor, t: HI2Tensor) -> dict:
    r = table.shape[0]
    chosen = table.sse[picks, torch.arange(r)].sum().item()
    nb = t.row_nbits.to(torch.int64)
    bl = t.row_block.to(torch.int64)
    return {
        "shape": list(table.shape),
        "bpw": tensor_bpw(t),
        "pred_rel_err": chosen / max(table.energy, 1e-30),
        "frac_int4": float((nb == 4).float().mean()),
        "frac_int8": float((nb == 8).float().mean()),
        "block_hist": {int(b): int((bl == b).sum()) for b in (16, 32, 64)},
        "rotated_groups": int(t.had.sum()),
        "groups": int(t.had.numel()),
        "n_tokens": table.n_tokens,
    }


def _finite(label: str, tensor: torch.Tensor) -> None:
    if tensor.is_floating_point() or tensor.is_complex():
        for chunk in tensor.reshape(-1).split(1024 * 1024):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"{label}: nonfinite values")


def _validate_fields(label: str, d: dict, parts: tuple[str, ...] | None = None) -> None:
    """Validate a flat container before any offset is used to slice it."""
    expected = set(FIELDS) | ({"part_off"} if parts is not None else set())
    if not expected.issubset(d) or set(d) - expected - {"col_scale"}:
        raise ValueError(f"{label}: missing or unsupported HI2 fields")
    dtypes = {
        "codes": torch.int32,
        "code_off": torch.int32,
        "blk_params": torch.uint8,
        "param_off": torch.int32,
        "row_consts": torch.float16,
        "row_nbits": torch.uint8,
        "row_block": torch.uint8,
        "had": torch.uint8,
        "col_scale": torch.float16,
        "part_off": torch.int32,
    }
    for name, tensor in d.items():
        if tensor.dtype != dtypes[name]:
            raise ValueError(
                f"{label}.hi2_{name}: expected {dtypes[name]}, got {tensor.dtype}"
            )
        _finite(f"{label}.hi2_{name}", tensor)
    had = d["had"]
    if had.ndim != (2 if parts is not None else 1) or not had.numel():
        raise ValueError(f"{label}.hi2_had: invalid shape {tuple(had.shape)}")
    if bool((had > 1).any()):
        raise ValueError(f"{label}.hi2_had: flags must be 0 or 1")
    k, r = had.shape[-1] * 16, d["row_nbits"].numel()
    shapes = {
        "codes": (d["codes"].numel(),),
        "blk_params": (d["blk_params"].numel(),),
        "code_off": (r + 1,),
        "param_off": (r + 1,),
        "row_consts": (r, 3),
        "row_nbits": (r,),
        "row_block": (r,),
    }
    if not r:
        raise ValueError(f"{label}: zero rows")
    if "col_scale" in d:
        shapes["col_scale"] = (had.shape[0], k) if parts is not None else (k,)
        if bool((d["col_scale"] <= 0).any()):
            raise ValueError(
                f"{label}.hi2_col_scale: smoothing scales must be positive"
            )
    if parts is not None:
        shapes["part_off"] = (had.shape[0] * len(parts) + 1,)
    for name, shape in shapes.items():
        if tuple(d[name].shape) != shape:
            raise ValueError(
                f"{label}.hi2_{name}: shape {tuple(d[name].shape)}, expected {shape}"
            )
    nb, block = d["row_nbits"].long(), d["row_block"].long()
    if not bool(((nb == 2) | (nb == 4) | (nb == 8)).all()):
        raise ValueError(f"{label}.hi2_row_nbits: supported widths are 2, 4, 8")
    if not bool(((block == 16) | (block == 32) | (block == 64)).all()):
        raise ValueError(f"{label}.hi2_row_block: supported blocks are 16, 32, 64")
    if bool((k % block != 0).any()) or bool((k * nb % 32 != 0).any()):
        raise ValueError(f"{label}: row width is not divisible by block/packing widths")
    for name, sizes, total in (
        ("code_off", k * nb // 32, d["codes"].numel()),
        ("param_off", k // block, d["blk_params"].numel() // 2),
    ):
        off = d[name].long()
        if (
            int(off[0]) != 0
            or int(off[-1]) != total
            or not torch.equal(off[1:] - off[:-1], sizes)
        ):
            raise ValueError(
                f"{label}.hi2_{name}: invalid row offsets or buffer bounds"
            )
    if d["blk_params"].numel() % 2:
        raise ValueError(f"{label}.hi2_blk_params: incomplete parameter pair")
    if bool((d["row_consts"][:, 1:] < 0).any()):
        raise ValueError(f"{label}.hi2_row_consts: negative parameter scales")
    if parts is not None:
        off = d["part_off"].long()
        if int(off[0]) != 0 or int(off[-1]) != r or bool((off[1:] <= off[:-1]).any()):
            raise ValueError(f"{label}.hi2_part_off: invalid expert/part row bounds")
    _validate_ladders(label, d, k, nb, block)


def _validate_ladders(label, d, k, nb, block):
    """Finite stored parameters are insufficient if their fp16 decode overflows.

    Scan bounded row batches; only unpacked codes and block extrema are needed,
    never a whole dense checkpoint or a matrix multiply.
    """
    from .format import decode_params, unpack_codes

    batch_rows = max(1, (1024 * 1024) // k)
    for bits in (2, 4, 8):
        for size in (16, 32, 64):
            rows = ((nb == bits) & (block == size)).nonzero().flatten()
            for batch in rows.split(batch_rows):
                if not batch.numel():
                    continue
                ci = d["code_off"][batch].long()[:, None] + torch.arange(k * bits // 32)
                codes = unpack_codes(d["codes"][ci], bits).reshape(-1, k // size, size)
                pi = d["param_off"][batch].long()[:, None] + torch.arange(k // size)
                params = d["blk_params"].reshape(-1, 2)[pi]
                rc = d["row_consts"][batch, None, :]
                lo, step = decode_params(
                    params[..., 0], params[..., 1], rc[..., 0], rc[..., 1], rc[..., 2]
                )
                _finite(f"{label}: decoded lo", lo)
                _finite(f"{label}: decoded step", step)
                low = (lo.double() + codes.amin(-1).double() * step.double()).half()
                high = (lo.double() + codes.amax(-1).double() * step.double()).half()
                _finite(f"{label}: decoded weights", low)
                _finite(f"{label}: decoded weights", high)


def _container_members(base, d, parts=None):
    """Logical views of a validated container; unlike split_stack, no buffer clones."""
    if parts is None:
        yield (
            base + ".weight",
            HI2Tensor(
                **{f: d[f] for f in FIELDS},
                shape=(d["row_nbits"].numel(), d["had"].numel() * 16),
                col_scale=d.get("col_scale"),
            ),
        )
        return
    pre = base.rpartition(".")[0]
    for expert in range(d["had"].shape[0]):
        for part, name in enumerate(parts):
            index = expert * len(parts) + part
            r0, r1 = map(int, d["part_off"][index : index + 2])
            c0, c1 = int(d["code_off"][r0]), int(d["code_off"][r1])
            p0, p1 = int(d["param_off"][r0]), int(d["param_off"][r1])
            yield (
                f"{pre}.{expert}.{name}.weight",
                HI2Tensor(
                    codes=d["codes"][c0:c1],
                    code_off=d["code_off"][r0 : r1 + 1] - c0,
                    blk_params=d["blk_params"][2 * p0 : 2 * p1],
                    param_off=d["param_off"][r0 : r1 + 1] - p0,
                    row_consts=d["row_consts"][r0:r1],
                    row_nbits=d["row_nbits"][r0:r1],
                    row_block=d["row_block"][r0:r1],
                    had=d["had"][expert],
                    shape=(r1 - r0, d["had"].shape[1] * 16),
                    col_scale=d["col_scale"][expert] if "col_scale" in d else None,
                ),
            )


def _check_report_entry(name: str, info, tensor: HI2Tensor) -> None:
    if not isinstance(info, dict) or info.get("shape") != list(tensor.shape):
        raise ValueError(f"{name}: report shape does not match artifact")
    bpw = info.get("bpw")
    if (
        not isinstance(bpw, (int, float))
        or not math.isfinite(bpw)
        or abs(tensor_bpw(tensor) - bpw) > 1e-6
    ):
        raise ValueError(f"{name}: report bpw does not match artifact")


def validate_checkpoint(
    dst: Path,
    *,
    expected_tensors: Sequence[str] | None = None,
    require_report: bool = True,
    log=print,
) -> dict:
    """Full CPU structural validation of every shard, dense tensor and HI2 buffer.

    Expected names are source weight names (including ``.weight``), not physical
    stack names. The report must exactly cover the artifact unless explicitly
    disabled for a pre-report write. This is not a native-runtime qualification.
    Raises ValueError with the failing artifact instead of returning partial success.
    """
    from safetensors import SafetensorError, safe_open

    from .stack import GROUPS

    dst = Path(dst)
    cfg = _read_json(_local_path(dst, "config.json"))
    qc = cfg.get("quantization_config")
    if (
        not isinstance(qc, dict)
        or qc.get("quant_method") != "rdna_hint_flex"
        or qc.get("version") != SPEC_VERSION
    ):
        raise ValueError(
            f"{dst}/config.json: unsupported HI2 quantization config/version"
        )
    if qc.get("mode", "flex") != "flex":
        raise ValueError(f"{dst}/config.json: unsupported HI2 quantization mode")
    layout = qc.get("moe_layout", "per-expert")
    if layout not in ("per-expert", "stacked"):
        raise ValueError(f"{dst}/config.json: unsupported MoE layout {layout!r}")
    groups = qc.get("moe_stacks", {})
    if not isinstance(groups, dict) or any(
        g not in GROUPS or not isinstance(p, list) or tuple(p) != GROUPS[g]
        for g, p in groups.items()
    ):
        raise ValueError(f"{dst}/config.json: unsupported moe_stacks mapping")
    report = None
    report_path = _local_path(dst, "hi2_report.json")
    if require_report or report_path.exists():
        report = _read_json(report_path).get("tensors")
        if not isinstance(report, dict) or not report:
            raise ValueError(f"{report_path}: empty or invalid tensor coverage")
    index_path = _local_path(dst, "model.safetensors.index.json")
    metadata = (
        _read_json(index_path).get("metadata", {})
        if index_path.exists() or index_path.is_symlink()
        else {}
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"{index_path}: metadata must be an object")
    try:
        wmap = weight_map(dst)
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"{dst}: unreadable checkpoint index/shard: {exc}") from exc
    shards = set(wmap.values())
    present = {str(p.relative_to(dst)) for p in dst.rglob("*.safetensors")}
    if present != shards:
        raise ValueError(
            f"{dst}: shard/index coverage mismatch; missing={sorted(shards - present)}, unindexed={sorted(present - shards)}"
        )
    if len({_local_path(dst, sh).resolve() for sh in shards}) != len(shards):
        raise ValueError(f"{dst}: shard paths alias the same file")
    containers: dict[str, tuple[str, set[str]]] = {}
    dense_names = set()
    for name, shard in wmap.items():
        if ".hi2_" not in name:
            dense_names.add(name)
            continue
        base, fld = name.rsplit(".hi2_", 1)
        if not base:
            raise ValueError(f"{dst}: empty HI2 tensor base")
        if base not in containers:
            containers[base] = (shard, set())
        old_shard, fields = containers[base]
        if old_shard != shard:
            raise ValueError(f"{dst}:{base}: HI2 fields are split across shards")
        fields.add(fld)
    logical = set()
    physical_checked = 0
    data_bytes = 0
    stack_groups = set()
    for shard in sorted(shards):
        path = _local_path(dst, shard)
        try:
            with safe_open(path, "pt") as f:
                keys = set(f.keys())
                indexed = {n for n, s in wmap.items() if s == shard}
                if keys != indexed:
                    raise ValueError(
                        f"{path}: tensor/index coverage mismatch; missing={sorted(indexed - keys)}, unindexed={sorted(keys - indexed)}"
                    )
                for name in keys & dense_names:
                    tensor = f.get_tensor(name)
                    _finite(f"{path}:{name}", tensor)
                    data_bytes += tensor.numel() * tensor.element_size()
                for base, (location, fields) in containers.items():
                    if location != shard:
                        continue
                    parts = None
                    if "part_off" in fields:
                        group = base.rpartition(".")[2]
                        if layout != "stacked" or group not in GROUPS:
                            raise ValueError(
                                f"{path}:{base}: stack contradicts config/layout"
                            )
                        parts = GROUPS[group]
                        stack_groups.add(group)
                    d = {fld: f.get_tensor(f"{base}.hi2_{fld}") for fld in fields}
                    _validate_fields(f"{path}:{base}", d, parts)
                    data_bytes += sum(t.numel() * t.element_size() for t in d.values())
                    for name, tensor in _container_members(base, d, parts):
                        if name in logical or name in dense_names:
                            raise ValueError(f"{path}:{name}: duplicate logical tensor")
                        logical.add(name)
                        if report is not None:
                            _check_report_entry(
                                f"{path}:{name}", report.get(name), tensor
                            )
                    physical_checked += 1
        except (OSError, SafetensorError, KeyError, IndexError, RuntimeError) as exc:
            raise ValueError(
                f"{path}: unreadable or invalid tensor data: {exc}"
            ) from exc
    if not logical:
        raise ValueError(f"{dst}: zero expected quantized tensors")
    if "moe_stacks" in qc and set(groups) != stack_groups:
        raise ValueError(
            f"{dst}/config.json: moe_stacks does not match artifact groups"
        )
    if "total_size" in metadata and (
        type(metadata["total_size"]) is not int or metadata["total_size"] != data_bytes
    ):
        raise ValueError(
            f"{index_path}: metadata.total_size does not match tensor data"
        )
    if expected_tensors is not None:
        expected = set(expected_tensors)
        if not expected or expected != logical:
            raise ValueError(
                f"{dst}: expected tensor coverage mismatch; missing={sorted(expected - logical)}, unexpected={sorted(logical - expected)}"
            )
    if report is not None and set(report) != logical:
        raise ValueError(
            f"{report_path}: tensor coverage mismatch; missing={sorted(logical - set(report))}, unexpected={sorted(set(report) - logical)}"
        )
    result = {
        "mode": "full_structural",
        "status": "ok",
        "checked": len(logical),
        "expected": len(logical),
        "coverage": 1.0,
        "physical_containers": physical_checked,
        "physical_tensors": len(wmap),
        "dense_tensors": len(dense_names),
        "tensor_data_bytes": data_bytes,
        "shards": len(shards),
        "report_checked": report is not None,
        "portable_manifest_checked": _validate_portable_manifest(dst),
        "native_runtime": "not_run",
    }
    log(
        f"validate full_structural: {len(logical)}/{len(logical)} logical tensors, {len(shards)} shards; native runtime not run"
    )
    return result


def verify_checkpoint(src: Path, dst: Path, max_tensors: int = 8, log=print) -> dict:
    """Cheap sampled diagnostics with exact logical report coverage, not a full scan."""
    from safetensors import safe_open

    from .stack import Hi2Checkpoint, to_fields

    if max_tensors <= 0:
        raise ValueError(
            "max_tensors must be positive; zero-coverage verification is not valid"
        )
    qc = _read_json(_local_path(dst, "config.json")).get("quantization_config", {})
    if (
        not isinstance(qc, dict)
        or qc.get("version") != SPEC_VERSION
        or qc.get("quant_method") != "rdna_hint_flex"
    ):
        raise ValueError(
            f"{dst}/config.json: unsupported HI2 quantization config/version"
        )
    report = _read_json(_local_path(dst, "hi2_report.json")).get("tensors")
    if not isinstance(report, dict) or not report:
        raise ValueError(f"{dst}/hi2_report.json: empty or invalid tensor coverage")
    ck = Hi2Checkpoint(dst)
    fields = {}
    for name in ck.wmap:
        if ".hi2_" in name:
            base, fld = name.rsplit(".hi2_", 1)
            fields.setdefault(base, set()).add(fld)
    for base, present in fields.items():
        if not set(FIELDS).issubset(present) or present - set(FIELDS) - {
            "part_off",
            "col_scale",
        }:
            raise ValueError(f"{dst}:{base}: incomplete or unsupported HI2 fields")
    logical = {name + ".weight" for name in ck.logical()}
    if not logical or set(report) != logical:
        raise ValueError(
            f"{dst}/hi2_report.json: tensor coverage mismatch with checkpoint"
        )
    wsrc = weight_map(src)
    missing = set(report) - set(wsrc)
    if missing:
        raise ValueError(f"{src}: missing expected source tensors {sorted(missing)}")
    worst, checked = 0.0, 0
    for name in list(report)[:max_tensors]:
        t = ck.get(name.removesuffix(".weight"))
        _validate_fields(f"{dst}:{name}", to_fields(t))
        _check_report_entry(name, report[name], t)
        with safe_open(src / wsrc[name], "pt") as f:
            w = f.get_tensor(name).double()
        if tuple(w.shape) != t.shape:
            raise ValueError(
                f"{name}: source shape {tuple(w.shape)} != checkpoint {t.shape}"
            )
        _finite(f"{src}:{name}", w)
        decoded = dequantize(t)
        _finite(f"{dst}:{name}: dequantized", decoded)
        loss, energy = ((decoded - w) ** 2).sum().item(), (w**2).sum().item()
        if energy == 0 and loss != 0:
            raise ValueError(f"{name}: nonzero reconstruction of a zero-energy source")
        rel = loss / energy if energy else 0.0
        if not math.isfinite(rel):
            raise ValueError(f"{name}: nonfinite numerical diagnostic")
        worst = max(worst, rel)
        checked += 1
        log(f"verify {name}: weight energy loss {rel:.4f} bpw {tensor_bpw(t):.3f}")
    expected = len(report)
    log(
        f"verify sampled_numerical: {checked}/{expected} tensors; native runtime not run"
    )
    return {
        "mode": "sampled_numerical",
        "status": "ok",
        "checked": checked,
        "expected": expected,
        "coverage": checked / expected,
        "worst_unweighted_energy_loss": worst,
        "structural": {"mode": "full_structural", "status": "not_run"},
        "native_runtime": "not_run",
    }


def load_dequantized(dst: Path) -> dict[str, torch.Tensor]:
    """Dense fp32 state dict of an HI2 checkpoint (either MoE layout): quantised
    tensors are dequantised back to ``<name>.weight`` (original basis); the rest are
    copied."""
    from safetensors import safe_open

    from .stack import Hi2Checkpoint

    ck = Hi2Checkpoint(dst)
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(ck.wmap.values())):
        with safe_open(dst / shard, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                if ".hi2_" in name:
                    continue
                out[name] = f.get_tensor(name)
            for base, t in ck.iter_shard(shard, f):
                out[f"{base}.weight"] = dequantize(t).float()
    return out


def export_dense(src: Path, dst: Path, out: Path) -> None:
    """Write the dequantised model as a plain fp16 Hugging Face checkpoint in ``out``
    (``model.safetensors`` + ``config.json`` without ``quantization_config`` + the
    tokenizer/other files of ``src`` symlinked), e.g. to score it in another engine."""
    from safetensors.torch import save_file

    check_output_paths(inputs=[src, dst], outputs=[out])
    if out.exists() or out.is_symlink():
        raise ValueError(
            f"{out}: dense export requires a new, exclusively owned destination"
        )
    out.mkdir(parents=True)
    state = {
        k: v.to(torch.float16).contiguous() for k, v in load_dequantized(dst).items()
    }
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    skip = {"config.json", "model.safetensors", "model.safetensors.index.json"}
    for entry in src.iterdir():
        if (
            entry.name in skip
            or entry.name.startswith(".")
            or entry.suffix == ".safetensors"
        ):
            continue
        link = out / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)


_PORTABLE_MANIFEST = "hi2_portable_manifest.json"
_EVIDENCE_FILES = {"hi2_run.json", "hi2_quality.json"}
_ASSET_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
    "chat_template.jinja",
    "chat_template.json",
}


def _portable_files(src: Path, assets_from: Path | None, shards: set[str]):
    """Resolve only recognized assets and their explicit local dependencies.

    An external asset symlink is not implicit authorization to copy arbitrary files:
    its resolved target must be in src or the caller's explicit assets_from root.
    """
    roots = [src.resolve()]
    if assets_from is not None:
        roots.append(assets_from.resolve())
    files: dict[str, Path] = {}
    rewritten: dict[str, dict] = {}
    reserved = {
        "config.json",
        "hi2_report.json",
        "model.safetensors.index.json",
        *shards,
        *_EVIDENCE_FILES,
    }

    def add(name: str, source: Path, *, asset=False):
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
            raise ValueError(f"{source}: escaping portable asset path {name!r}")
        name = rel.as_posix()
        if name == _PORTABLE_MANIFEST or (
            asset
            and (
                name in reserved
                or rel.suffix in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
            )
        ):
            raise ValueError(
                f"{source}: ancillary asset collides with checkpoint weights/config"
            )
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"{source}: missing or invalid portable dependency"
            ) from exc
        if not resolved.is_file() or not any(
            resolved.is_relative_to(root) for root in roots
        ):
            raise ValueError(
                f"{source}: portable dependency escapes trusted asset roots"
            )
        if name in files and files[name] != resolved:
            raise ValueError(f"{name}: portable asset destination collision")
        for other in files:
            if other != name and (
                rel.is_relative_to(Path(other)) or Path(other).is_relative_to(rel)
            ):
                raise ValueError(
                    f"{name}: portable file/directory collision with {other}"
                )
        files[name] = resolved

    def find(name: str):
        for root in (src, assets_from):
            if root is not None:
                candidate = root / name
                if candidate.exists() or candidate.is_symlink():
                    return candidate
        return src / name

    for name in sorted(reserved):
        if name == "model.safetensors.index.json" and not (src / name).exists():
            continue
        if name in _EVIDENCE_FILES:
            source = _local_path(src, name)
            if not source.exists() and not source.is_symlink():
                continue
            add(name, source)
        else:
            add(name, src / name)
    for name in sorted(_ASSET_NAMES):
        source = find(name)
        if source.exists() or source.is_symlink():
            add(name, source, asset=True)
    templates = find("chat_templates")
    if templates.exists() or templates.is_symlink():
        if not templates.resolve().is_dir() or not any(
            templates.resolve().is_relative_to(root) for root in roots
        ):
            raise ValueError(
                f"{templates}: escaping or invalid chat template directory"
            )
        for entry in sorted(templates.rglob("*")):
            if entry.is_file() or entry.is_symlink():
                add(
                    "chat_templates/" + entry.relative_to(templates).as_posix(),
                    entry,
                    asset=True,
                )

    def dependency(current: str, value: str):
        reference = Path(value)
        if reference.is_absolute():
            resolved = reference.resolve()
            trusted = next(
                (root for root in roots if resolved.is_relative_to(root)), None
            )
            if trusted is None:
                raise ValueError(f"{current}: escaping asset reference {value!r}")
            target = resolved.relative_to(trusted).as_posix()
            source = reference
        else:
            if ".." in reference.parts or "://" in value:
                raise ValueError(
                    f"{current}: escaping or remote asset reference {value!r}"
                )
            target = (Path(current).parent / reference).as_posix()
            source = find(target)
        add(target, source, asset=True)
        return os.path.relpath(target, Path(current).parent).replace(os.sep, "/")

    def rewrite(current: str, obj):
        if isinstance(obj, list):
            return [rewrite(current, value) for value in obj]
        if not isinstance(obj, dict):
            return obj
        result = {}
        for key, value in obj.items():
            if key.endswith("_file") and isinstance(value, str) and value:
                result[key] = dependency(current, value)
            elif key.endswith("_files") and isinstance(value, (list, dict)):
                if isinstance(value, list):
                    result[key] = [
                        dependency(current, v) if isinstance(v, str) and v else v
                        for v in value
                    ]
                else:
                    result[key] = {
                        k: dependency(current, v) if isinstance(v, str) and v else v
                        for k, v in value.items()
                    }
            else:
                result[key] = rewrite(current, value)
            if key == "auto_map" and isinstance(value, dict):
                for classes in value.values():
                    for cls in classes if isinstance(classes, list) else [classes]:
                        if cls is None:
                            continue
                        if not isinstance(cls, str) or "--" in cls or "." not in cls:
                            raise ValueError(
                                f"{current}: unsupported remote/custom code reference {cls!r}"
                            )
                        module = cls.rsplit(".", 1)[0]
                        if not all(part.isidentifier() for part in module.split(".")):
                            raise ValueError(
                                f"{current}: invalid custom module {module!r}"
                            )
                        name = module.replace(".", "/") + ".py"
                        add(name, find(name), asset=True)
        return result

    processed = set()
    while set(files) - processed:
        name = sorted(set(files) - processed)[0]
        source = files[name]
        processed.add(name)
        if (
            name.endswith(".json")
            and name not in shards
            and name not in _EVIDENCE_FILES
            and name != "model.safetensors.index.json"
        ):
            original = _read_json(source)
            changed = rewrite(name, original)
            if changed != original:
                rewritten[name] = changed
        elif name.endswith(".py"):
            # Follow local imports, not installed-library code.
            for parent in Path(name).parents:
                if parent == Path("."):
                    continue
                init = (parent / "__init__.py").as_posix()
                if find(init).exists():
                    add(init, find(init), asset=True)
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [
                        (alias.name.replace(".", "/"), False) for alias in node.names
                    ]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        prefix = Path(name).parent
                        for _ in range(node.level - 1):
                            if prefix == Path("."):
                                raise ValueError(
                                    f"{source}: relative import escapes portable root"
                                )
                            prefix = prefix.parent
                        module = (
                            prefix / (node.module or "").replace(".", "/")
                        ).as_posix()
                        if node.module:
                            modules.append((module, True))
                        modules.extend(
                            ((Path(module) / alias.name).as_posix(), not node.module)
                            for alias in node.names
                            if alias.name != "*"
                        )
                    elif node.module:
                        modules = [(node.module.replace(".", "/"), False)]
                for stem, required in modules:
                    candidates = [stem + ".py", stem + "/__init__.py"]
                    found = next((n for n in candidates if find(n).exists()), None)
                    if found is not None:
                        add(found, find(found), asset=True)
                    elif required:
                        raise ValueError(
                            f"{source}: missing local Python dependency {stem}"
                        )
    tokenizers = {
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
    }
    if not tokenizers.intersection(files) and not {"vocab.json", "merges.txt"}.issubset(
        files
    ):
        raise ValueError(
            f"{src}: missing tokenizer vocabulary/model for portable export"
        )
    return files, rewritten


def _validate_portable_manifest(dst: Path) -> bool:
    path = dst / _PORTABLE_MANIFEST
    if not path.exists() and not path.is_symlink():
        return False
    manifest = _read_json(_local_path(dst, _PORTABLE_MANIFEST))
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != "hi2-portable"
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError(f"{path}: unsupported portable manifest")
    actual = set()
    for entry in dst.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"{entry}: portable bundle contains a symlink")
        if entry.is_file() and entry != path:
            actual.add(entry.relative_to(dst).as_posix())
    if actual != set(files):
        raise ValueError(f"{path}: portable file coverage mismatch")
    for name, info in files.items():
        entry = _local_path(dst, name)
        if (
            not isinstance(info, dict)
            or info.get("size_bytes") != entry.stat().st_size
            or info.get("sha256") != file_digest(entry)
        ):
            raise ValueError(f"{entry}: portable checksum/size mismatch")
    return True


def _artifact_output_lock(dst: Path):
    """Use RunSession's output ownership lock, outside the bundle's file coverage."""
    from .runtime import exclusive_lock, object_digest

    lock_name = object_digest(str(dst.resolve()))[:24]
    return exclusive_lock(dst.parent / f".hi2-output-{lock_name}.lock")


def _publish_quality_report(dst: Path, report: dict) -> None:
    """Publish under the caller's output lock, rolling back ordinary failures.

    Each replacement is atomic, not the pair: a hard kill between them remains
    fail-closed under the portable manifest's complete coverage/checksum checks.
    """
    from .runtime import atomic_file, atomic_json

    portable = _validate_portable_manifest(dst)
    report_path = _local_path(dst, "hi2_quality.json")
    manifest_path = dst / _PORTABLE_MANIFEST
    original_report = report_path.read_bytes() if report_path.exists() else None
    original_manifest = manifest_path.read_bytes() if portable else None
    manifest = _read_json(manifest_path) if portable else None
    try:
        atomic_json(report_path, report)
        if manifest is not None:
            manifest["files"][report_path.name] = {
                "size_bytes": report_path.stat().st_size,
                "sha256": file_digest(report_path),
            }
            atomic_json(manifest_path, manifest)
    except BaseException:
        if original_report is None:
            report_path.unlink(missing_ok=True)
        else:
            with atomic_file(report_path) as stream:
                stream.write(original_report)
        if original_manifest is not None:
            with atomic_file(manifest_path) as stream:
                stream.write(original_manifest)
        raise


def export_portable(src: Path, out: Path, *, assets_from: Path | None = None) -> dict:
    """Copy a validated quantized HI2 checkpoint into a self-contained bundle.

    Shard bytes/layout and source-owned run/quality evidence are retained, never
    dequantized, re-encoded or replaced by evidence from ``assets_from``. Only
    recognized ancillary assets and their local file dependencies are materialized.
    External asset links require an explicit trusted ``assets_from`` directory.
    The output must not exist; failures remove only this invocation's directory.
    """
    from .runtime import atomic_json

    src, out = Path(src), Path(out)
    assets_from = Path(assets_from) if assets_from is not None else None
    check_output_paths(
        inputs=[src] + ([assets_from] if assets_from is not None else []), outputs=[out]
    )
    with _artifact_output_lock(src), _artifact_output_lock(out):
        if out.exists() or out.is_symlink():
            raise ValueError(
                f"{out}: portable export requires a new, exclusively owned destination"
            )
        validate_checkpoint(src, log=lambda *_: None)
        files, rewritten = _portable_files(
            src, assets_from, set(weight_map(src).values())
        )
        out.mkdir(parents=True)
        try:
            checksums = {}
            for name, source in sorted(files.items()):
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if name in rewritten:
                    atomic_json(dest, rewritten[name])
                    digest = file_digest(dest)
                else:
                    digest = file_digest(source)
                    shutil.copyfile(source, dest)
                    if file_digest(dest) != digest:
                        raise ValueError(
                            f"{source}: changed while copying portable asset"
                        )
                checksums[name] = {"size_bytes": dest.stat().st_size, "sha256": digest}
            manifest = {
                "schema_version": 1,
                "format": "hi2-portable",
                "asset_policy": "portable",
                "quantized_weights": "byte_identical",
                "files": checksums,
            }
            atomic_json(out / _PORTABLE_MANIFEST, manifest)
            validation = validate_checkpoint(out, log=lambda *_: None)
        except BaseException:
            shutil.rmtree(out)
            raise
        return {
            "exported": str(out),
            "manifest": str(out / _PORTABLE_MANIFEST),
            "validation": validation,
            **manifest,
        }
