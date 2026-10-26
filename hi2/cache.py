# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Persistent cache of the reusable phase products (``--cache-dir``, default
``$HI2_CACHE_DIR`` or ``~/.cache/hi2``).

Three kinds of entries, each a directory ``<root>/<kind>/<digest>`` where ``digest`` is
a hash of a JSON key (kept in the entry's ``key.json``):

* ``stats``: per Linear input, the second moment ``Sigma = E[x x^T]`` (``sigma/``,
  stored once per shared input as the upper triangle in fp32, or in full if it is not
  exactly symmetric) and the 16x16 block moments (``colstats/``), plus ``layers.json``
  (which Linears share an input, token counts) and the MoE ``expert_tokens.json``.
  Sigma is taken in the *original* basis from the unquantized model, so it depends only
  on the model weights, the calibration tokens and how they are run (dtype, device kind,
  number of devices, the MoE token floor), never on the allocation target, error
  ratio, rounding, smoothing, rotation, candidates or QAD. Key: :func:`stats_key`.
* ``tables``: the phase-1 tables, keyed by the stats key plus the phase-1 settings.
* ``probe``: the sensitivity-probe result, keyed by the tables key plus the probe
  settings and probe tokens.

An entry is built in ``<digest>.tmp-<pid>`` and renamed into place when complete, under
a lock (``<digest>.lock``) so concurrent runs wait for one computation instead of
repeating it. Bump the matching ``*_VERSION`` when the code that produces an entry
changes its output; old entries are then simply never hit again (``hi2 cache prune``).
"""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import torch

STATS_VERSION = 1  # layerwise forward, hooks, ColumnStats / HessianStats, MoE floor
TABLES_VERSION = 1  # build_table / shared_transform / encode_rows
PROBE_VERSION = 1  # sensitivity.probe_sensitivity
KINDS = ("stats", "tables", "probe")


def default_root() -> Path:
    d = os.environ.get("HI2_CACHE_DIR")
    return Path(d) if d else Path.home() / ".cache" / "hi2"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:24]


def model_fingerprint(src: Path) -> str:
    """config.json and the weight files' names and sizes (not their paths or mtimes, so
    a copied checkpoint still hits; a re-saved one with other sizes does not)."""
    h = hashlib.sha256((src / "config.json").read_bytes())
    for f in sorted(src.iterdir()):
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            h.update(f"{f.name}:{f.stat().st_size};".encode())
    return h.hexdigest()[:24]


def ids_fingerprint(ids: torch.Tensor) -> str:
    """The calibration token ids themselves: covers the text, the tokenizer, the
    sample count and the sequence length."""
    a = ids.to(torch.int64).contiguous().cpu()
    h = hashlib.sha256(str(tuple(a.shape)).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()[:24]


def stats_key(rc, ids: torch.Tensor, devices) -> dict:
    return {
        "kind": "stats",
        "version": STATS_VERSION,
        "model": model_fingerprint(rc.src),
        "ids": ids_fingerprint(ids),
        "samples_x_len": list(ids.shape),
        "compute_dtype": rc.compute_dtype,
        "device_kind": sorted({d.type for d in devices}),
        "n_devices": len(devices),
        "include": rc.include,
        "exclude": rc.exclude,
        "limit_layers": rc.limit_layers,
        "expert_min_tokens": rc.cfg.expert_min_tokens,
        **({"sigma_fp16_above": rc.sigma_fp16_above} if rc.sigma_fp16_above else {}),
        # informational only (the content hashes above are what matter)
        "_src": str(rc.src),
        "_calib": str(rc.calib),
    }


def _key_view(key: dict) -> dict:
    return {k: v for k, v in key.items() if not k.startswith("_")}


def tables_key(skey: dict, cfg) -> dict:
    return {
        "kind": "tables",
        "version": TABLES_VERSION,
        "stats": digest(_key_view(skey)),
        "candidates": [list(x) for x in cfg.candidates],
        "rotation": cfg.rotation,
        "smooth": cfg.smooth,
        "effort": cfg.effort,
    }


def probe_key(tkey: dict, rc, probe_ids: torch.Tensor, ref_candidate) -> dict:
    return {
        "kind": "probe",
        "version": PROBE_VERSION,
        "tables": digest(tkey),
        "ids": ids_fingerprint(probe_ids),
        "floor": rc.sens_floor,
        "dtype": rc.sens_dtype,
        "share_floor": rc.expert_share_floor,
        "ref_candidate": list(ref_candidate),
        "effort": rc.cfg.effort,
        "devices": len(rc.devices),
        "batch": rc.sens_batch,
    }


class Entry:
    def __init__(self, root: Path, key: dict):
        self.key = key
        self.kind = key["kind"]
        self.digest = digest(_key_view(key))
        self.path = root / self.kind / self.digest
        self.work = self.path  # where writes go (a tmp dir while building)

    def complete(self, need: str | None = None) -> bool:
        info = self.info()
        return info is not None and (need is None or bool(info.get(need)))

    def info(self) -> dict | None:
        p = self.path / "complete.json"
        return json.loads(p.read_text()) if p.exists() else None

    def touch(self) -> None:
        with contextlib.suppress(OSError):
            (self.path / "complete.json").touch()

    @contextmanager
    def building(self):
        """Lock the entry; yields True if the caller must build it into ``self.work``
        (then :meth:`finish`), False if another process finished it meanwhile."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.parent / f"{self.digest}.lock"
        with open(lock, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def begin(self) -> None:
        self.work = self.path.parent / f"{self.digest}.tmp-{os.getpid()}"
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        (self.work / "key.json").write_text(
            json.dumps(self.key, indent=1, sort_keys=True)
        )

    def finish(self, info: dict) -> None:
        info = {"created": time.time(), **info}
        (self.work / "complete.json").write_text(json.dumps(info, indent=1))
        if self.work != self.path:
            if self.path.exists():  # an older, incomplete or superseded copy
                shutil.rmtree(self.path)
            os.rename(self.work, self.path)
            self.work = self.path

    def abort(self) -> None:
        if self.work != self.path:
            shutil.rmtree(self.work, ignore_errors=True)
            self.work = self.path


# ------------------------------------------------------------------ tensor storage


def _fname(name: str) -> str:
    return name.replace("/", "_") + ".pt"


def _upper(k: int, device="cpu") -> torch.Tensor:
    return torch.ones(k, k, dtype=torch.bool, device=device).triu_()


def sigma_payload(sigma: torch.Tensor, n: int, fp16_above: int = 0) -> dict:
    """What :func:`save_sigma` stores, computed where ``sigma`` lives (the symmetry
    check and the triangle extraction on the GPU halve the device-to-host copy): the
    upper triangle (row-major) when ``sigma`` is exactly symmetric, else the full
    matrix; in fp16 when ``fp16_above`` > 0 and K exceeds it (lossy, opt-in)."""
    s = sigma.detach().float().contiguous()
    k = s.shape[0]
    half = 0 < fp16_above < k
    if torch.equal(s, s.T):
        tri = s[_upper(k, s.device)]
        return {"k": k, "n": n, "tri": (tri.half() if half else tri).cpu()}
    return {"k": k, "n": n, "full": (s.half() if half else s).cpu()}


def save_sigma(
    d: Path, name: str, sigma: torch.Tensor, n: int, fp16_above: int = 0
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    torch.save(sigma_payload(sigma, n, fp16_above), d / _fname(name))


def load_sigma(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=True)
    if isinstance(obj, torch.Tensor):  # legacy workdir file: the full matrix
        return obj
    if "full" in obj:
        return obj["full"].float()
    k = obj["k"]
    up = _upper(k)
    s = torch.zeros(k, k, dtype=torch.float32)
    s.masked_scatter_(up, obj["tri"].float())
    return torch.where(up, s, s.T)  # an exact copy of the upper triangle


class StatsView:
    """Read side of the per-tensor statistics: the workdir's own files (``hessian/``,
    ``stats/``, legacy layout and ``--no-cache``) or a cache entry they point to
    (``workdir/cache_ref.json``)."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.entry: Path | None = None
        self.links: dict[str, str] = {}
        ref = workdir / "cache_ref.json"
        if ref.exists():
            p = Path(json.loads(ref.read_text())["stats"])
            if (p / "complete.json").exists():
                self.entry = p
                self.links = json.loads((p / "layers.json").read_text())["leader"]

    def _cached(self, sub: str, name: str) -> Path | None:
        if self.entry is None or name not in self.links:
            return None
        p = self.entry / sub / _fname(self.links[name])
        return p if p.exists() else None

    def sigma_path(self, name: str) -> Path | None:
        legacy = self.workdir / "hessian" / _fname(name)
        if legacy.exists():
            return legacy
        return self._cached("sigma", name)

    def sigma(self, name: str) -> torch.Tensor | None:
        p = self.sigma_path(name)
        return None if p is None else load_sigma(p)

    def sigma_id(self, name: str) -> str | None:
        """Identifies the Sigma behind ``name`` (shared inputs give one id)."""
        p = self.sigma_path(name)
        return None if p is None else str(p.resolve())

    def expert_tokens_path(self) -> Path | None:
        for p in (
            self.workdir / "expert_tokens.json",
            None if self.entry is None else self.entry / "expert_tokens.json",
        ):
            if p is not None and p.exists():
                return p
        return None

    def stats_state(self, name: str) -> dict | None:
        legacy = self.workdir / "stats" / _fname(name)
        if legacy.exists():
            return torch.load(legacy)
        p = self._cached("colstats", name)
        return None if p is None else torch.load(p)


def write_ref(workdir: Path, **entries: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / "cache_ref.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update({k: str(v) for k, v in entries.items()})
    p.write_text(json.dumps(cur, indent=1))


# ------------------------------------------------------------------ maintenance


def _size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        with contextlib.suppress(OSError):  # a build may rename files meanwhile
            if f.is_file():
                total += f.stat().st_size
    return total


def entries(root: Path) -> list[dict]:
    out = []
    for kind in KINDS:
        base = root / kind
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            info_p = d / "complete.json"
            key_p = d / "key.json"
            out.append(
                {
                    "kind": kind,
                    "digest": d.name,
                    "path": str(d),
                    "complete": info_p.exists(),
                    "bytes": _size(d),
                    "last_used": (info_p if info_p.exists() else d).stat().st_mtime,
                    "key": json.loads(key_p.read_text()) if key_p.exists() else None,
                }
            )
    return out


def _building(e: dict) -> bool:
    """An unfinished ``<digest>.tmp-<pid>`` whose process is still alive."""
    name = e["digest"]
    if ".tmp-" not in name:
        return False
    try:
        os.kill(int(name.rsplit(".tmp-", 1)[1]), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def prune(
    root: Path,
    older_than_days: float | None = None,
    keep_gb: float | None = None,
    digests: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Remove incomplete leftovers (not builds in progress), entries unused for
    ``older_than_days``, named ``digests``, then the least recently used until at most
    ``keep_gb`` remain."""
    now = time.time()
    es = [e for e in entries(root) if not _building(e)]
    drop = [
        e
        for e in es
        if not e["complete"]
        or (digests and e["digest"] in digests)
        or (
            older_than_days is not None
            and now - e["last_used"] > older_than_days * 86400
        )
    ]
    if keep_gb is not None:
        rest = sorted((e for e in es if e not in drop), key=lambda e: e["last_used"])
        total = sum(e["bytes"] for e in rest)
        while rest and total > keep_gb * 2**30:
            e = rest.pop(0)
            drop.append(e)
            total -= e["bytes"]
    if not dry_run:
        for e in drop:
            shutil.rmtree(e["path"], ignore_errors=True)
        for lock in root.glob("*/*.lock"):
            if not (lock.parent / lock.stem).exists():
                lock.unlink(missing_ok=True)
    return drop


def estimate_stats_bytes(src: Path, include, exclude) -> int:
    """Upper bound of a stats entry: an fp32 triangle per eligible Linear input (inputs
    shared by q/k/v or gate/up are counted once per layer and width)."""
    from safetensors import safe_open

    from .checkpoint import is_eligible, weight_map
    from .moe import expert_parts

    wmap = weight_map(src)
    seen: set[tuple] = set()
    total = 0
    for shard in sorted(set(wmap.values())):
        with safe_open(src / shard, "pt") as f:
            for name in [n for n, s in wmap.items() if s == shard]:
                sl = f.get_slice(name)
                shape = sl.get_shape()
                if not is_eligible(name, shape, torch.float32, include, exclude):
                    continue
                k = shape[1]
                p = expert_parts(name)
                parent = name.rsplit(".", 2)[0]
                fam = (p[0], p[1], k) if p else (parent, k)
                if fam in seen and not name.endswith(
                    ("o_proj.weight", "down_proj.weight", "out_proj.weight")
                ):
                    continue
                seen.add(fam)
                total += 2 * k * (k + 1) + 128 * k
    return total


def check_space(root: Path, need: int, log=print) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < need:
        log(
            f"warning: {free / 2**30:.1f} GiB free under {root}, the statistics cache "
            f"needs up to {need / 2**30:.1f} GiB (hi2 cache prune, or --no-cache)"
        )
