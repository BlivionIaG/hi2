# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Consumer-visible CPU planning, artifact ownership and recovery contracts."""

import dataclasses
import json

import pytest
import torch
from safetensors.torch import save_file

from hi2.calibrate import Config
from hi2.pipeline import RunConfig, plan, run
from hi2.runtime import atomic_file, exclusive_lock, read_run
from hi2.service import config_from_dict, resolve_config, save_recipe


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 64,
                "num_hidden_layers": 1,
            }
        )
    )
    weights = torch.randn(64, 64, generator=torch.Generator().manual_seed(17))
    save_file(
        {"model.layers.0.mlp.up_proj.weight": weights}, root / "model.safetensors"
    )
    return root


def configuration(source, tmp_path, **kwargs):
    values = dict(
        src=source,
        dst=tmp_path / "output",
        workdir=tmp_path / "work",
        cfg=Config(effort="fast", rotation="off"),
        moe_layout="per-expert",
    )
    values.update(kwargs)
    return RunConfig(**values)


def test_plan_resolves_data_free_defaults_without_writes(source, tmp_path):
    rc = configuration(source, tmp_path)
    result = plan(rc, log=lambda _: None)
    assert result["rounding"] == "nearest"
    assert result["max_error_ratio"] == 4
    assert result["phases"] == [
        "data_free_tables",
        "allocation",
        "encode",
        "validation",
    ]
    assert result["outputs"] == [str(rc.dst)]
    assert not rc.workdir.exists() and not rc.dst.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"devices": ["cuda:99999"]},
        {"seq_len": 0},
        {"target_bpw": float("nan")},
        {"max_error_ratios": [2, 2]},
        {"sensitivity": "probe"},
        {"target_bpw": 3, "max_error_ratio": 4},
    ],
)
def test_invalid_plan_rejected_before_work(source, tmp_path, changes):
    rc = configuration(source, tmp_path, **changes)
    with pytest.raises((ValueError, RuntimeError)):
        plan(rc, log=lambda _: None)
    assert not rc.workdir.exists() and not rc.dst.exists()


def test_output_source_alias_is_rejected(source, tmp_path):
    link = tmp_path / "alias"
    link.symlink_to(source, target_is_directory=True)
    rc = configuration(source, tmp_path, dst=link)
    before = (source / "model.safetensors").read_bytes()
    with pytest.raises(ValueError):
        run(rc, log=lambda _: None)
    assert (source / "model.safetensors").read_bytes() == before
    assert not rc.workdir.exists()


def test_unchanged_resume_preserves_complete_report_and_rejects_changed_recipe(
    source, tmp_path
):
    rc = configuration(source, tmp_path)
    first = run(rc, log=lambda _: None)
    report = json.loads((rc.dst / "hi2_report.json").read_text())
    assert set(report["tensors"]) == {"model.layers.0.mlp.up_proj.weight"}
    second = run(rc, log=lambda _: None)
    assert second == first
    assert json.loads((rc.dst / "hi2_report.json").read_text()) == report
    manifest = read_run(rc.workdir)
    assert manifest["state"] == "succeeded"
    assert (rc.dst / "hi2_run.json").is_file()
    with pytest.raises(ValueError, match="incompatible resume"):
        run(dataclasses.replace(rc, max_error_ratio=7), log=lambda _: None)
    assert json.loads((rc.dst / "hi2_report.json").read_text()) == report


def test_same_size_source_change_rejects_resume(source, tmp_path):
    rc = configuration(source, tmp_path)
    run(rc, log=lambda _: None)
    original_size = (source / "model.safetensors").stat().st_size
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.ones(64, 64)},
        source / "model.safetensors",
    )
    assert (source / "model.safetensors").stat().st_size == original_size
    with pytest.raises(ValueError, match="incompatible resume"):
        run(rc, log=lambda _: None)


def test_unowned_output_is_not_overwritten(source, tmp_path):
    rc = configuration(source, tmp_path)
    rc.dst.mkdir()
    sentinel = rc.dst / "valuable.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="unowned output"):
        run(rc, log=lambda _: None)
    assert sentinel.read_text() == "preserve me"
    assert not (rc.workdir / "run.json").exists()


def test_recipe_roundtrip_and_explicit_override(source, tmp_path):
    rc = resolve_config(configuration(source, tmp_path))
    recipe = tmp_path / "recipe.json"
    save_recipe(recipe, rc)
    doc = json.loads(recipe.read_text())
    restored = config_from_dict(doc, {"max_error_ratio": 6})
    assert resolve_config(restored).max_error_ratio == 6
    assert restored.cfg == rc.cfg
    with pytest.raises(ValueError, match="unknown recipe"):
        config_from_dict(doc, {"unexpected": 1})
    with pytest.raises(ValueError, match="schema_version"):
        config_from_dict({**doc, "schema_version": 999})


def test_failed_atomic_write_preserves_previous_artifact(tmp_path):
    artifact = tmp_path / "record.json"
    artifact.write_bytes(b"original")
    with pytest.raises(RuntimeError), atomic_file(artifact) as stream:
        stream.write(b"incomplete")
        raise RuntimeError("interrupted writer")
    assert artifact.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


def test_workdir_lock_excludes_second_writer_without_unlinking(tmp_path):
    lock = tmp_path / ".run.lock"
    with exclusive_lock(lock):
        inode = lock.stat().st_ino
        with (
            pytest.raises(RuntimeError, match="resource is busy"),
            exclusive_lock(lock),
        ):
            pass
    assert lock.stat().st_ino == inode
    with exclusive_lock(lock):
        assert lock.stat().st_ino == inode


def test_quality_rejects_an_unrelated_reference_before_evaluation(source, tmp_path):
    import shutil

    from hi2.quality import evaluate_quality

    rc = configuration(source, tmp_path)
    run(rc, log=lambda _: None)
    other = tmp_path / "different-model"
    shutil.copytree(source, other)
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.ones(64, 64)},
        other / "model.safetensors",
    )
    text = tmp_path / "held-out.txt"
    text.write_text("independent evaluation text")
    with pytest.raises(ValueError, match="reference differs"):
        evaluate_quality(other, rc.dst, text, samples=1, seq_len=8, log=lambda _: None)
    assert not (rc.dst / "hi2_quality.json").exists()


@pytest.mark.parametrize("operation", [plan, run], ids=["plan", "run"])
@pytest.mark.parametrize(
    "placement",
    [
        "output_in_cache",
        "cache_in_output",
        "work_in_cache",
        "cache_in_work",
        "same_output",
        "same_work",
        "symlink_parent",
    ],
)
def test_cache_cannot_overlap_owned_run_paths(source, tmp_path, operation, placement):
    cache, output, work = (tmp_path / name for name in ("cache", "output", "work"))
    if placement == "output_in_cache":
        output = cache / "stats" / "result"
    elif placement == "cache_in_output":
        cache = output / "cache"
    elif placement == "work_in_cache":
        work = cache / "tables" / "work"
    elif placement == "cache_in_work":
        cache = work / "cache"
    elif placement == "same_output":
        output = cache
    elif placement == "same_work":
        work = cache
    else:
        cache.mkdir()
        alias = tmp_path / "cache-alias"
        alias.symlink_to(cache, target_is_directory=True)
        output = alias / "stats" / "result"
    before = set(tmp_path.rglob("*"))
    weights = (source / "model.safetensors").read_bytes()
    rc = configuration(source, tmp_path, dst=output, workdir=work, cache_dir=cache)
    with pytest.raises(ValueError):
        operation(rc, log=lambda _: None)
    assert set(tmp_path.rglob("*")) == before
    assert (source / "model.safetensors").read_bytes() == weights


def test_pruning_disjoint_cache_preserves_published_checkpoint(source, tmp_path):
    from hi2.cache import prune
    from hi2.checkpoint import validate_checkpoint

    cache = tmp_path / "cache"
    rc = configuration(source, tmp_path, cache_dir=cache)
    run(rc, log=lambda _: None)
    published = (rc.dst / "hi2_run.json").read_bytes()
    abandoned = cache / "stats" / "abandoned"
    abandoned.mkdir(parents=True)
    (abandoned / "partial").write_bytes(b"incomplete cache product")
    removed = prune(cache)
    assert [entry["path"] for entry in removed] == [str(abandoned)]
    assert (rc.dst / "hi2_run.json").read_bytes() == published
    assert validate_checkpoint(rc.dst, log=lambda _: None)["status"] == "ok"
