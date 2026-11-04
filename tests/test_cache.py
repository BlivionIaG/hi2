# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Statistics/tables/probe cache, batched multi-tensor encode, multi-output mode (CPU).

Everything here must be bit-identical to the uncached, per-tensor path."""

import fcntl
import json
import multiprocessing
import os
import shutil

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from hi2 import cache as C  # noqa: E402
from hi2.calibrate import (  # noqa: E402
    Config,
    EncodeRequest,
    encode_many,
    encode_with_picks_reference,
)
from hi2.executor import Executor  # noqa: E402
from hi2.pipeline import RunConfig, output_dir, run  # noqa: E402


def _save_sigma(d, name, sigma, n, fp16_above=0):
    d.mkdir(parents=True, exist_ok=True)
    torch.save(C.sigma_payload(sigma, n, fp16_above), d / C._fname(name))


QUIET = dict(log=lambda *_: None)


@pytest.fixture
def cache_process():
    """Spawn rather than fork an already initialized Torch runtime."""
    ctx = multiprocessing.get_context("spawn")
    children = []

    def start(target, *args):
        parent, child = ctx.Pipe()
        process = ctx.Process(target=target, args=(*args, child))
        process.start()
        child.close()
        children.append((process, parent))
        return process, parent

    yield start
    for process, pipe in children:
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        pipe.close()


def _cache_message(pipe):
    assert pipe.poll(30), "timed out waiting for the cache worker"
    return pipe.recv()


def _join_cache_process(process):
    process.join(timeout=30)
    assert process.exitcode == 0


def _held_cache_builder(root, key, pipe):
    entry = C.Entry(root, key)
    with entry.building():
        pipe.send("locked")
        assert pipe.recv() == "begin"
        entry.begin()
        (entry.work / "product").write_bytes(b"first builder")
        pipe.send(("building", entry.work))
        command = pipe.recv()
        if command == "abandon-marked":
            (entry.work / "complete.json").write_text("{}")
            return  # simulate interruption between the marker and publication
        if command == "abandon":
            return  # leave an interrupted build, but release the kernel lock
        assert command == "finish"
        entry.finish({})
        pipe.send("published")
        assert pipe.recv() == "release"
    pipe.send("released")


def _competing_cache_builder(root, key, pipe):
    entry = C.Entry(root, key)
    # A real nonblocking acquisition gives a deterministic contention observation;
    # no sleep or negative event timeout is used to infer whether a builder waited.
    with entry.path.with_suffix(".lock").open("a+b") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pipe.send("blocked")
        else:
            fcntl.flock(fh, fcntl.LOCK_UN)
            pipe.send("unlocked")
    with entry.building():
        if entry.complete():
            pipe.send("hit")
        else:
            entry.begin()
            (entry.work / "product").write_bytes(b"second builder")
            entry.finish({})
            pipe.send("built")


def _rewrite_same_size(path, content):
    before = path.stat()
    assert len(content) == before.st_size
    path.write_bytes(content)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def _same_size_weight_edit(path, pipe):
    pipe.send("ready")
    assert pipe.recv() == "edit"
    content = path.read_bytes()
    _rewrite_same_size(path, content[:-1] + bytes([content[-1] ^ 1]))
    pipe.send("edited")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 2 and "proj" in n:
                p.mul_(torch.exp(0.5 * torch.randn(p.shape[0], 1)))
                p.mul_(torch.exp(0.4 * torch.randn(1, p.shape[1])))
    model.save_pretrained(d, safe_serialization=True)
    ids = torch.randint(0, 256, (8, 64), generator=torch.Generator().manual_seed(1))
    return d, ids


def _rc(tiny, tmp_path, tag, cache=True, **kw):
    d, ids = tiny
    cfg = kw.pop("cfg", Config(effort="fast"))
    base = dict(
        src=d,
        dst=tmp_path / f"out-{tag}",
        workdir=tmp_path / f"work-{tag}",
        calib_ids=ids,
        target_bpw=3.0,
        calib_samples=8,
        seq_len=64,
        cfg=cfg,
        cache_dir=tmp_path / "cache" if cache else None,
    )
    base.update(kw)
    return RunConfig(**base)


def _files(dst):
    return {
        p.name: p.read_bytes()
        for p in sorted(dst.iterdir())
        if p.suffix == ".safetensors"
        or p.name in ("config.json", "model.safetensors.index.json")
    }


def _same_checkpoint(a, b):
    fa, fb = _files(a), _files(b)
    assert fa.keys() == fb.keys()
    for k in fa:
        assert fa[k] == fb[k], k


def _no_forward(monkeypatch):
    import hi2.layerwise as lw
    import hi2.pipeline as pl

    def boom(*a, **k):
        raise AssertionError("the model must not run on a cache hit")

    monkeypatch.setattr(lw, "calibrate_layerwise", boom)
    monkeypatch.setattr(pl, "_load_model", boom)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors",
        "shards/extra.safetensors",
    ],
)
def test_model_fingerprint_tracks_all_checkpoint_bytes(tmp_path, filename):
    src = tmp_path / "source"
    (src / "shards").mkdir(parents=True)
    (src / "config.json").write_bytes(b'{"width":1}')
    # The last byte lies beyond the hash buffer; sampling only a prefix is unsafe.
    (src / "model.safetensors").write_bytes(b"x" * (2**20 + 31))
    (src / "shards" / "extra.safetensors").write_bytes(b"extra weights")
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.weight": "shards/extra.safetensors",
                }
            }
        )
    )
    original = C.model_fingerprint(src)
    copied = shutil.copytree(src, tmp_path / "copy")
    os.utime(copied / "config.json", ns=(0, 0))
    assert C.model_fingerprint(copied) == original
    path = src / filename
    content = path.read_bytes()
    if filename == "config.json":
        changed = content.replace(b"1", b"2")
    elif filename == "model.safetensors.index.json":
        changed = content.replace(b"a.weight", b"c.weight")
    else:
        changed = content[:-1] + bytes([content[-1] ^ 1])
    _rewrite_same_size(path, changed)
    assert C.model_fingerprint(src) != original
    assert C.model_fingerprint(copied) == original


def test_same_size_weight_edit_invalidates_cached_products(
    tiny, tmp_path, cache_process
):
    src = shutil.copytree(tiny[0], tmp_path / "source")
    root = tmp_path / "cache"

    def products(source):
        rc = _rc(tiny, tmp_path, "keys", src=source)
        skey = C.stats_key(rc, tiny[1], [torch.device("cpu")])
        tkey = C.tables_key(skey, rc.cfg)
        pkey = C.probe_key(tkey, rc, tiny[1], (2, 32))
        return [C.Entry(root, key) for key in (skey, tkey, pkey)]

    original = products(src)
    for entry in original:
        with entry.building():
            entry.begin()
            (entry.work / "product").write_bytes(b"original model")
            entry.finish({})
    saved_key = json.loads((original[0].path / "key.json").read_text())
    assert saved_key["_model_identity"] == "sha256-content-v1"
    weight = src / "model.safetensors"
    before = weight.stat()
    editor, pipe = cache_process(_same_size_weight_edit, weight)
    assert _cache_message(pipe) == "ready"
    copied = shutil.copytree(src, tmp_path / "copy")
    for old, copy in zip(original, products(copied)):
        assert copy.digest == old.digest
        assert copy.complete()
    pipe.send("edit")
    assert _cache_message(pipe) == "edited"
    _join_cache_process(editor)
    after = weight.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    for old, changed in zip(original, products(src)):
        assert changed.digest != old.digest
        assert not changed.complete(), changed.kind
        assert old.complete()


@pytest.mark.parametrize("change", ["bytes", "path", "unset"])
def test_ngram_table_change_invalidates_cached_products(
    tiny, tmp_path, monkeypatch, cache_process, change
):
    from safetensors.torch import save_file

    table = tmp_path / "ngram.safetensors"
    save_file(
        {"model.layers.1.ple.ngram_embedding.weight": torch.ones(64, 8)},
        table,
    )
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(table))
    rc = _rc(tiny, tmp_path, "ngram")
    fingerprint = C.model_fingerprint(rc.src)

    def products():
        skey = C.stats_key(rc, tiny[1], [torch.device("cpu")])
        tkey = C.tables_key(skey, rc.cfg)
        pkey = C.probe_key(tkey, rc, tiny[1], (2, 32))
        return [C.Entry(rc.cache_dir, key) for key in (skey, tkey, pkey)]

    original = products()
    for entry in original:
        with entry.building():
            entry.begin()
            (entry.work / "product").write_bytes(b"original ngram table")
            entry.finish({})
    saved_key = json.loads((original[0].path / "key.json").read_text())
    assert saved_key["auxiliary_inputs"] == C.auxiliary_model_inputs()
    alias = tmp_path / "ngram-alias.safetensors"
    alias.symlink_to(table)
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(alias))
    for old, aliased in zip(original, products()):
        assert aliased.digest == old.digest
        assert aliased.complete()
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(table))

    if change == "bytes":
        before = table.stat()
        editor, pipe = cache_process(_same_size_weight_edit, table)
        assert _cache_message(pipe) == "ready"
        pipe.send("edit")
        assert _cache_message(pipe) == "edited"
        _join_cache_process(editor)
        after = table.stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
    elif change == "path":
        selected = tmp_path / "other-ngram.safetensors"
        shutil.copy2(table, selected)
        monkeypatch.setenv("HI2_NGRAM_MMAP", str(selected))
    else:
        monkeypatch.delenv("HI2_NGRAM_MMAP")

    # External inputs must not change the directory-only fingerprint API.
    assert C.model_fingerprint(rc.src) == fingerprint
    for old, changed in zip(original, products()):
        assert changed.digest != old.digest, changed.kind
        assert not changed.complete(), changed.kind
        assert old.complete()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize(
    "known",
    [
        (),
        ("fingerprint",),
        ("auxiliary_inputs",),
        ("fingerprint", "auxiliary_inputs"),
    ],
)
def test_stats_key_reuses_available_current_run_identity(
    tiny, tmp_path, monkeypatch, active, known
):
    from safetensors.torch import save_file

    if active:
        table = tmp_path / "ngram.safetensors"
        save_file({"ngram_embedding.weight": torch.ones(64, 8)}, table)
        monkeypatch.setenv("HI2_NGRAM_MMAP", str(table))
    else:
        monkeypatch.delenv("HI2_NGRAM_MMAP", raising=False)
    rc = _rc(tiny, tmp_path, "identity")
    devices = [torch.device("cpu")]
    expected = C.stats_key(rc, tiny[1], devices)
    source = {
        "fingerprint": expected["model"],
        "auxiliary_inputs": expected["auxiliary_inputs"],
    }
    rc._identity = {"source": {key: source[key] for key in known}}
    snapshot = json.loads(json.dumps(rc._identity))
    reads = []
    fingerprint, auxiliary = C.model_fingerprint, C.auxiliary_model_inputs

    def read_model(src):
        reads.append("fingerprint")
        return fingerprint(src)

    def read_auxiliary():
        reads.append("auxiliary_inputs")
        return auxiliary()

    monkeypatch.setattr(C, "model_fingerprint", read_model)
    monkeypatch.setattr(C, "auxiliary_model_inputs", read_auxiliary)
    assert C.stats_key(rc, tiny[1], devices) == expected
    assert sorted(reads) == sorted(set(source) - set(known))
    assert rc._identity == snapshot


@pytest.mark.parametrize("phase", ["locked", "building", "published"])
def test_prune_preserves_active_build_lock(tmp_path, cache_process, phase):
    root = tmp_path / "cache"
    key = {"kind": "stats", "version": C.STATS_VERSION, "test": "active"}
    entry = C.Entry(root, key)
    builder, pipe = cache_process(_held_cache_builder, root, key)
    assert _cache_message(pipe) == "locked"
    work = None
    if phase != "locked":
        pipe.send("begin")
        state, work = _cache_message(pipe)
        assert state == "building"
    if phase == "published":
        pipe.send("finish")
        assert _cache_message(pipe) == "published"
    lock = entry.path.with_suffix(".lock")
    inode = lock.stat().st_ino
    assert C.prune(root, older_than_days=0, keep_gb=0, digests=[entry.digest]) == []
    if phase == "building":
        assert work.is_dir()
    if phase == "published":
        assert entry.complete()
    contender, contender_pipe = cache_process(_competing_cache_builder, root, key)
    assert _cache_message(contender_pipe) == "blocked"
    assert lock.stat().st_ino == inode
    if phase == "locked":
        pipe.send("begin")
        state, work = _cache_message(pipe)
        assert state == "building"
    if phase != "published":
        pipe.send("finish")
        assert _cache_message(pipe) == "published"
    pipe.send("release")
    assert _cache_message(pipe) == "released"
    assert _cache_message(contender_pipe) == "hit"
    _join_cache_process(builder)
    _join_cache_process(contender)
    assert (entry.path / "product").read_bytes() == b"first builder"
    gone = C.prune(root, keep_gb=0)
    assert [e["digest"] for e in gone] == [entry.digest]
    assert not entry.path.exists()
    assert lock.stat().st_ino == inode
    # Rebuilding a pruned digest must also reuse the permanent lock inode.
    with entry.building():
        entry.begin()
        entry.finish({})
    assert entry.complete()
    assert lock.stat().st_ino == inode


@pytest.mark.parametrize("marked", [False, True])
def test_prune_reclaims_abandoned_process_build(tmp_path, cache_process, marked):
    root = tmp_path / "cache"
    key = {"kind": "stats", "version": C.STATS_VERSION, "test": "abandoned"}
    entry = C.Entry(root, key)
    builder, pipe = cache_process(_held_cache_builder, root, key)
    assert _cache_message(pipe) == "locked"
    pipe.send("begin")
    state, work = _cache_message(pipe)
    assert state == "building"
    pipe.send("abandon-marked" if marked else "abandon")
    _join_cache_process(builder)
    lock = entry.path.with_suffix(".lock")
    inode = lock.stat().st_ino
    assert (
        next(e for e in C.entries(root) if e["path"] == str(work))["complete"] is False
    )
    assert [e["digest"] for e in C.prune(root)] == [work.name]
    assert not work.exists()
    assert lock.stat().st_ino == inode


def test_prune_reclaims_unlocked_build_despite_live_pid(tmp_path):
    entry = C.Entry(tmp_path, {"kind": "stats", "test": "live-pid-leftover"})
    with entry.building():
        entry.begin()
    work = entry.work
    assert work.name.endswith(f".tmp-{os.getpid()}")
    preview = C.prune(tmp_path, dry_run=True)
    assert [e["digest"] for e in preview] == [work.name]
    assert work.is_dir()
    assert C.prune(tmp_path) == preview
    assert not work.exists()
    assert entry.path.with_suffix(".lock").exists()


def test_sigma_triangle_roundtrip_is_exact(tmp_path):
    x = torch.randn(300, 48)
    s = x.T @ x / 300
    s = 0.5 * (s + s.T)
    _save_sigma(tmp_path, "a.weight", s, 300)
    assert torch.equal(C.load_sigma(tmp_path / "a.weight.pt"), s)
    obj = torch.load(tmp_path / "a.weight.pt")
    assert "tri" in obj and obj["tri"].numel() == 48 * 49 // 2
    asym = s.clone()
    asym[0, 1] += 1e-3  # not symmetric: stored in full
    _save_sigma(tmp_path, "b.weight", asym, 300)
    assert torch.equal(C.load_sigma(tmp_path / "b.weight.pt"), asym)


@pytest.mark.parametrize("rounding", ["gptq", "nearest"])
def test_cached_run_equals_uncached_and_hits_on_rerun(
    tiny, tmp_path, monkeypatch, rounding
):
    cfg = Config(effort="fast", rounding=rounding)
    plain = _rc(tiny, tmp_path, "plain", cache=False, cfg=cfg)
    cached = _rc(tiny, tmp_path, "cached", cfg=cfg)
    run(plain, **QUIET)
    run(cached, **QUIET)
    _same_checkpoint(plain.dst, cached.dst)
    assert not (cached.workdir / "hessian").exists()  # Sigma lives in the cache
    ref = json.loads((cached.workdir / "cache_ref.json").read_text())
    assert (C.Path(ref["stats"]) / "complete.json").exists()
    # a second run (other workdir, other target) reuses statistics and tables
    logs = []
    _no_forward(monkeypatch)
    again = _rc(tiny, tmp_path, "again", cfg=cfg, target_bpw=3.6)
    run(again, log=logs.append)
    assert any("statistics hit" in m for m in logs) and any(
        "tables hit" in m for m in logs
    )
    plain2 = _rc(tiny, tmp_path, "plain2", cache=False, cfg=cfg, target_bpw=3.6)
    monkeypatch.undo()
    run(plain2, **QUIET)
    _same_checkpoint(plain2.dst, again.dst)


def test_new_phase1_settings_reuse_statistics_without_forward(
    tiny, tmp_path, monkeypatch
):
    run(_rc(tiny, tmp_path, "first"), **QUIET)
    cfg = Config(
        effort="fast",
        smooth=0.25,
        rotation="auto",
        candidates=[(2, 32), (4, 64), (2, 16)],
    )
    logs = []
    _no_forward(monkeypatch)
    other = _rc(tiny, tmp_path, "other", cfg=cfg)
    run(other, log=logs.append)
    assert any("statistics hit" in m for m in logs) and any(
        "tables miss" in m for m in logs
    )
    monkeypatch.undo()
    plain = _rc(tiny, tmp_path, "plain", cache=False, cfg=cfg)
    run(plain, **QUIET)
    _same_checkpoint(plain.dst, other.dst)


def test_other_calibration_tokens_miss(tiny, tmp_path):
    run(_rc(tiny, tmp_path, "a"), **QUIET)
    ids = tiny[1].clone()
    ids[0, 0] = (ids[0, 0] + 1) % 256
    logs = []
    run(_rc(tiny, tmp_path, "b", calib_ids=ids), log=logs.append)
    assert any("statistics miss" in m for m in logs)
    assert len([e for e in C.entries(tmp_path / "cache") if e["kind"] == "stats"]) == 2


def test_probe_is_cached(tiny, tmp_path, monkeypatch):
    kw = dict(sensitivity="probe", sens_seqs=2, sens_len=32)
    first = _rc(tiny, tmp_path, "p1", **kw)
    run(first, **QUIET)
    import hi2.sensitivity as sm

    monkeypatch.setattr(
        sm,
        "probe_sensitivity",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe re-ran")),
    )
    logs = []
    second = _rc(tiny, tmp_path, "p2", target_bpw=3.6, **kw)
    run(second, log=logs.append)
    assert any("sensitivity probe hit" in m for m in logs)
    assert json.loads((first.workdir / "sensitivity.json").read_text()) == json.loads(
        (second.workdir / "sensitivity.json").read_text()
    )


def test_multi_ratio_outputs_equal_single_runs(tiny, tmp_path):
    ratios = [2.0, 5.0]
    multi = _rc(tiny, tmp_path, "m", target_bpw=None, max_error_ratios=ratios)
    out = run(multi, **QUIET)
    assert set(out["outputs"]) == {str(output_dir(multi.dst, r)) for r in ratios}
    for r in ratios:
        single = _rc(
            tiny, tmp_path, f"s{r:g}", cache=False, target_bpw=None, max_error_ratio=r
        )
        run(single, **QUIET)
        _same_checkpoint(single.dst, output_dir(multi.dst, r))


@pytest.mark.parametrize("devices", [["cpu"], ["cpu", "cpu"]])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("rounding", ["gptq", "nearest"])
def test_encode_many_equals_per_tensor_reference(
    tiny, tmp_path, devices, act_order, rounding
):
    """Several tensors (two sharing an input), several devices: identical buffers and
    identical QAD initialisation values."""
    cfg = Config(effort="fast", act_order=act_order, rounding=rounding)
    rc = _rc(tiny, tmp_path, "t", cache=False, cfg=cfg)
    run(rc, **QUIET)
    from safetensors import safe_open

    from hi2.cache import StatsView
    from hi2.checkpoint import weight_map
    from hi2.stats import ColumnStats

    tables = {
        n: torch.load(p, weights_only=False)
        for n, p in ((p.stem, p) for p in (rc.workdir / "tables").iterdir())
    }
    view = StatsView(rc.workdir)
    wmap = weight_map(rc.src)
    names = [n for n in tables if "layers.1." in n]
    cfg, ex = rc.cfg, Executor(devices)
    g = torch.Generator().manual_seed(5)
    reqs, refs = [], []
    for n in names:
        with safe_open(rc.src / wmap[n], "pt") as f:
            w = f.get_tensor(n)
        t = tables[n]
        picks = torch.randint(0, len(t.candidates), (t.shape[0],), generator=g)
        st = view.stats_state(n)
        stats = ColumnStats.from_state(st) if st else None
        sigma = view.sigma(n)
        reqs.append(
            EncodeRequest(w, stats, t, picks, sigma, view.sigma_id(n), comp_out={})
        )
        co: dict = {}
        ref = encode_with_picks_reference(w, stats, t, picks, cfg, ex, sigma, co)
        refs.append((ref, co))
    assert len({r.sigma_id for r in reqs}) < len(reqs)  # q/k/v or gate/up share
    for got, (ref, co), r in zip(encode_many(reqs, cfg, ex), refs, reqs):
        assert torch.equal(r.comp_out["comp"], co["comp"])  # the QAD initialisation
        for f in (
            "codes",
            "code_off",
            "blk_params",
            "param_off",
            "row_consts",
            "row_nbits",
            "row_block",
            "had",
        ):
            assert torch.equal(getattr(got, f), getattr(ref, f)), f


def test_cache_list_and_prune(tiny, tmp_path):
    run(_rc(tiny, tmp_path, "a"), **QUIET)
    root = tmp_path / "cache"
    es = C.entries(root)
    assert {e["kind"] for e in es} == {"stats", "tables"} and all(
        e["complete"] for e in es
    )
    (root / "stats" / "deadbeef.tmp-999999999").mkdir()  # an interrupted build
    with (root / "stats" / "cafe.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (root / "stats" / f"cafe.tmp-{os.getpid()}").mkdir()  # a build in progress
        gone = C.prune(root)
        assert [e["digest"] for e in gone] == ["deadbeef.tmp-999999999"]
        assert (root / "stats" / f"cafe.tmp-{os.getpid()}").exists()
        (root / "stats" / f"cafe.tmp-{os.getpid()}").rmdir()
    gone = C.prune(root, keep_gb=0)
    assert len(gone) == len(es) and C.entries(root) == []


def test_space_estimate_bounds_the_entry(tiny, tmp_path):
    rc = _rc(tiny, tmp_path, "a")
    run(rc, **QUIET)
    import regex as re

    from hi2.checkpoint import DEFAULT_EXCLUDE

    est = C.estimate_stats_bytes(rc.src, None, re.compile(DEFAULT_EXCLUDE))
    entry = C.Path(json.loads((rc.workdir / "cache_ref.json").read_text())["stats"])
    # triangle bytes (a CPU GEMM is not exactly symmetric, so the tiny test stores full)
    ks = [torch.load(p)["k"] for p in (entry / "sigma").iterdir()]
    assert 0 < sum(2 * k * (k + 1) for k in ks) <= est


def test_probe_prefix_replay_is_bit_identical(tiny, tmp_path):
    """Replaying the unchanged decoder layers before a unit gives the same KL as the
    full forward of the original probe."""
    from hi2.sensitivity import probe_sensitivity

    rc = _rc(tiny, tmp_path, "pr", cache=False)
    run(rc, **QUIET)
    tables = {
        p.name.removesuffix(".pt"): torch.load(p, weights_only=False)
        for p in (rc.workdir / "tables").iterdir()
    }
    ids = tiny[1][:3, :32]
    kw = dict(log=lambda *_: None)
    full = probe_sensitivity(
        rc.src,
        tables,
        rc.workdir,
        ids,
        rc.cfg,
        Executor("cpu"),
        skip_prefix=False,
        **kw,
    )
    fast = probe_sensitivity(
        rc.src, tables, rc.workdir, ids, rc.cfg, Executor("cpu"), skip_prefix=True, **kw
    )
    assert full == fast
    batched = probe_sensitivity(
        rc.src, tables, rc.workdir, ids, rc.cfg, Executor("cpu"), batch=3, **kw
    )
    for n in full:
        assert batched[n] == pytest.approx(full[n], rel=1e-3)


def test_sigma_fp16_storage_is_opt_in(tmp_path):
    x = torch.randn(400, 80)
    s = x.T @ x / 400
    s = 0.5 * (s + s.T)
    _save_sigma(tmp_path, "a.weight", s, 400, fp16_above=64)
    got = C.load_sigma(tmp_path / "a.weight.pt")
    assert torch.equal(got, s.half().float())
    _save_sigma(tmp_path, "b.weight", s, 400, fp16_above=80)  # K not above: fp32
    assert torch.equal(C.load_sigma(tmp_path / "b.weight.pt"), s)


def test_cache_keys_can_record_another_device_count(monkeypatch):
    from types import SimpleNamespace

    import torch

    from hi2.cache import probe_key, stats_key

    cpu = torch.device("cpu")
    rc = SimpleNamespace(
        src=".",
        calib=".",
        compute_dtype="float32",
        include=None,
        exclude="x",
        limit_layers=None,
        cfg=SimpleNamespace(expert_min_tokens=2.0, effort="normal"),
        sigma_fp16_above=0,
        devices=["cpu"],
        cache_devices=0,
        sens_floor=0.05,
        sens_dtype="float32",
        expert_share_floor=0.05,
        sens_batch=1,
    )
    ids = torch.zeros(2, 4, dtype=torch.long)
    import hi2.cache as C

    monkeypatch.setattr(C, "model_fingerprint", lambda src: "m")
    one = stats_key(rc, ids, [cpu])
    four = stats_key(rc, ids, [cpu] * 4)
    assert one["n_devices"] == 1 and four["n_devices"] == 4
    rc.cache_devices = 4
    assert stats_key(rc, ids, [cpu])["n_devices"] == 4
    assert probe_key({"a": 1}, rc, ids, (2, 32))["devices"] == 4
