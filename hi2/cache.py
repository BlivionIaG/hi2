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
  on the model weights (including auxiliary inputs), the calibration tokens and how
  they are run (dtype, device kind, number of devices, the MoE token floor), never on
  the allocation target, error ratio, rounding, smoothing, rotation, candidates or
  QAD. Key: :func:`stats_key`.
* ``tables``: the phase-1 tables, keyed by the stats key plus the phase-1 settings.
* ``probe``: the sensitivity-probe result, keyed by the tables key plus the probe
  settings and probe tokens.

An entry is built in ``<digest>.tmp-<pid>`` and renamed into place when complete, under
a permanent lock file (``<digest>.lock``) so concurrent runs wait for one computation
instead of repeating it. Pruning takes the same lock and never unlinks lock files:
even an unlocked file may already have waiters with that inode open. Bump the matching
``*_VERSION`` when the code that produces an entry changes its output; old entries
are then simply never hit again (``hi2 cache prune``).
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

STATS_VERSION = 3  # auxiliary model inputs; v2 hashed only the model directory
TABLES_VERSION = 1  # build_table / shared_transform / encode_rows
PROBE_VERSION = 1  # sensitivity.probe_sensitivity
KINDS = ("stats", "tables", "probe")
MODEL_IDENTITY = "sha256-content-v1"


def default_root() -> Path:
    d = os.environ.get("HI2_CACHE_DIR")
    return Path(d) if d else Path.home() / ".cache" / "hi2"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:24]


def model_fingerprint(src: Path) -> str:
    """Hash config, index and weight bytes, including index-referenced nested shards.

    Identity includes relative filenames but not the checkpoint root, inode or
    timestamps: an identical copy hits, a same-size edit misses. Each call streams
    every file in full (O(checkpoint bytes) I/O and hashing, a reusable 1 MiB buffer).
    There is deliberately no stat-based memoization for mutable local checkpoints.
    The current run identity and stats key share this fingerprint; tables and probes
    inherit it through their parent keys. External model inputs are identified
    separately by auxiliary_model_inputs(), not folded into this directory hash.
    As with model loading, the source must remain unchanged throughout the run.
    """
    names = {"config.json"}
    names.update(f.name for f in src.iterdir() if f.suffix == ".safetensors")
    index = src / "model.safetensors.index.json"
    index_bytes = index.read_bytes() if index.exists() else None
    if index_bytes is not None:
        names.add(index.name)
        names.update(json.loads(index_bytes)["weight_map"].values())
    h = hashlib.sha256(MODEL_IDENTITY.encode())
    buffer = bytearray(2**20)
    view = memoryview(buffer)
    for name in sorted(names):
        content = hashlib.sha256()
        if name == index.name and index_bytes is not None:
            content.update(index_bytes)
        else:
            with (src / name).open("rb", buffering=0) as fh:
                while n := fh.readinto(buffer):
                    content.update(view[:n])
        encoded = name.encode()
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
        h.update(content.digest())
    return h.hexdigest()


def auxiliary_model_inputs() -> list[dict[str, str]]:
    """Identify active external model inputs by canonical path and full SHA256.

    Stream each file in bounded chunks, without timestamp/size memoization. Inputs
    must remain unchanged throughout the run, just like the model directory.
    """
    from .runtime import file_digest

    value = os.environ.get("HI2_NGRAM_MMAP")
    if not value:
        return []
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"HI2_NGRAM_MMAP must be a regular file: {path}")
    return [{"role": "ngram_mmap", "path": str(path), "sha256": file_digest(path)}]


def ids_fingerprint(ids: torch.Tensor) -> str:
    """The calibration token ids themselves: covers the text, the tokenizer, the
    sample count and the sequence length."""
    a = ids.to(torch.int64).contiguous().cpu()
    h = hashlib.sha256(str(tuple(a.shape)).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()[:24]


def stats_key(rc, ids: torch.Tensor, devices) -> dict:
    # resolve_config binds these inputs once for this invocation. Direct callers
    # without that identity must read the current bytes, never reuse stat metadata.
    source = (getattr(rc, "_identity", None) or {}).get("source", {})
    fingerprint = source.get("fingerprint")
    if fingerprint is None:
        fingerprint = model_fingerprint(rc.src)
    auxiliary = source.get("auxiliary_inputs")
    if auxiliary is None:
        auxiliary = auxiliary_model_inputs()
    return {
        "kind": "stats",
        "version": STATS_VERSION,
        "model": fingerprint,
        "auxiliary_inputs": auxiliary,
        "ids": ids_fingerprint(ids),
        "samples_x_len": list(ids.shape),
        "compute_dtype": rc.compute_dtype,
        "device_kind": sorted({d.type for d in devices}),
        "n_devices": rc.cache_devices or len(devices),
        "include": rc.include,
        "exclude": rc.exclude,
        "limit_layers": rc.limit_layers,
        "expert_min_tokens": rc.cfg.expert_min_tokens,
        **({"sigma_fp16_above": rc.sigma_fp16_above} if rc.sigma_fp16_above else {}),
        # informational only (the content hashes above are what matter)
        "_src": str(rc.src),
        "_calib": str(rc.calib),
        "_model_identity": MODEL_IDENTITY,
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
        **({"smooth_search": list(cfg.smooth_search)} if cfg.smooth_search else {}),
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
        "devices": rc.cache_devices or len(rc.devices),
        "batch": rc.sens_batch,
    }


@contextmanager
def _entry_lock(path: Path, *, blocking: bool = True):
    """Coordinate builders/pruners on one permanent inode, including tmp siblings."""
    lock = path.parent / f"{path.name.partition('.tmp-')[0]}.lock"
    with lock.open("a+b") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            if blocking:
                raise
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


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
        """Hold the entry's permanent lock through begin/finish/abort. The caller
        must recheck completeness after acquiring it before deciding to build."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _entry_lock(self.path):
            yield

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
    """What a cached Sigma file stores, computed where ``sigma`` lives (the symmetry
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


def _entry_paths(root: Path):
    for kind in KINDS:
        base = root / kind
        if base.is_dir():
            yield from (d for d in sorted(base.iterdir()) if d.is_dir())


def _entry_info(d: Path) -> dict:
    info_p = d / "complete.json"
    key_p = d / "key.json"
    # finish() writes the marker before the atomic rename; a tmp sibling is not
    # published even if a crashed builder left a complete.json inside it.
    complete = ".tmp-" not in d.name and info_p.exists()
    return {
        "kind": d.parent.name,
        "digest": d.name,
        "path": str(d),
        "complete": complete,
        "bytes": _size(d),
        "last_used": (info_p if complete else d).stat().st_mtime,
        "key": json.loads(key_p.read_text()) if key_p.exists() else None,
    }


def entries(root: Path) -> list[dict]:
    out = []
    for d in _entry_paths(root):
        with contextlib.suppress(FileNotFoundError):  # concurrently published/pruned
            out.append(_entry_info(d))
    return out


def prune(
    root: Path,
    older_than_days: float | None = None,
    keep_gb: float | None = None,
    digests: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Remove incomplete leftovers (not builds in progress), entries unused for
    ``older_than_days``, named ``digests``, then the least recently used until at most
    ``keep_gb`` remain. Busy entries are skipped, including published entries whose
    builder still holds the lock. Lock files are permanent; stale tmp directories
    are identified by lock availability, never by the liveness of a recorded PID.
    A dry run also takes these locks but does not remove any products.
    """
    now = time.time()
    es = []
    for d in _entry_paths(root):
        with _entry_lock(d, blocking=False) as held:
            if held:
                with contextlib.suppress(FileNotFoundError):
                    es.append(_entry_info(d))
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
    removed = []
    for e in drop:
        d = Path(e["path"])
        with _entry_lock(d, blocking=False) as held:
            if not held:
                continue
            with contextlib.suppress(FileNotFoundError):
                # A builder/pruner may have changed this entry since selection.
                # Leave changed entries for a later prune rather than deleting
                # a newly published or recently used product on a stale snapshot.
                if _entry_info(d) != e:
                    continue
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
                removed.append(e)
    return removed


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
