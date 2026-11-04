# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""External model inputs bind cache identity and real CPU run/resume lifecycles."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from hi2 import cache as C
from hi2.calibrate import Config
from hi2.pipeline import RunConfig, run
from hi2.runtime import read_run
from hi2.service import config_dict, resolve_config, run_identity


@pytest.fixture
def auxiliary_run(tmp_path, monkeypatch):
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 1})
    )
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                64, 64, generator=torch.Generator().manual_seed(17)
            )
        },
        source / "model.safetensors",
    )
    table = tmp_path / "ngram.safetensors"
    save_file(
        {
            "model.layers.1.ple.ngram_embedding.weight": torch.ones(
                64, 8, dtype=torch.bfloat16
            )
        },
        table,
    )
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(table))
    rc = resolve_config(
        RunConfig(
            src=source,
            dst=tmp_path / "output",
            workdir=tmp_path / "work",
            cfg=Config(effort="fast", rotation="off"),
            moe_layout="per-expert",
        ),
        check_ownership=False,
    )
    return rc, table


def _run_process(rc):
    """Use a fresh interpreter so no process-local identity can mask changed bytes."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "from hi2.errors import IncompatibleResume\n"
            "from hi2.service import config_from_dict, execute_run\n"
            "rc = config_from_dict(json.loads(sys.argv[1]))\n"
            "try:\n"
            "    result = execute_run(rc, log=lambda _: None)\n"
            "except IncompatibleResume as exc:\n"
            "    print(json.dumps({'error': type(exc).__name__}))\n"
            "    sys.exit(3)\n"
            "print(json.dumps(result, sort_keys=True))\n",
            json.dumps({"schema_version": 1, "config": config_dict(rc)}),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _artifacts(rc):
    return {
        str(path): path.read_bytes()
        for root in (rc.workdir, rc.dst)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_run_identity_and_stats_share_auxiliary_records(auxiliary_run):
    rc, _ = auxiliary_run
    identity = run_identity(rc)
    assert identity["source"]["fingerprint"] == C.model_fingerprint(rc.src)
    assert identity["source"]["auxiliary_inputs"] == C.auxiliary_model_inputs()
    ids = torch.arange(16).reshape(2, 8)
    direct = C.stats_key(rc, ids, [torch.device("cpu")])
    rc._identity = identity
    bound = C.stats_key(rc, ids, [torch.device("cpu")])
    assert bound == direct
    assert bound["model"] == identity["source"]["fingerprint"]
    assert bound["auxiliary_inputs"] == identity["source"]["auxiliary_inputs"]


@pytest.mark.parametrize("change", ["bytes", "path", "unset"])
def test_auxiliary_change_rejects_resume_across_processes(
    auxiliary_run, tmp_path, monkeypatch, change
):
    rc, table = auxiliary_run
    identity = run_identity(rc)
    first = _run_process(rc)
    assert first.returncode == 0, first.stderr
    assert read_run(rc.workdir)["source"] == identity["source"]
    checkpoint = (rc.dst / "model.safetensors").read_bytes()
    report = (rc.dst / "hi2_report.json").read_bytes()
    resumed = _run_process(rc)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == json.loads(first.stdout)
    assert (rc.dst / "model.safetensors").read_bytes() == checkpoint
    assert (rc.dst / "hi2_report.json").read_bytes() == report
    before_artifacts = _artifacts(rc)

    if change == "bytes":
        before = table.stat()
        with table.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"\x40")
        os.utime(table, ns=(before.st_atime_ns, before.st_mtime_ns))
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

    changed = run_identity(rc)
    assert changed["source"]["fingerprint"] == identity["source"]["fingerprint"]
    assert (
        changed["source"]["auxiliary_inputs"] != identity["source"]["auxiliary_inputs"]
    )
    rejected = _run_process(rc)
    assert rejected.returncode == 3, rejected.stdout + rejected.stderr
    assert json.loads(rejected.stdout) == {"error": "IncompatibleResume"}
    assert _artifacts(rc) == before_artifacts


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_auxiliary_input_rejects_execution_before_mutation(
    auxiliary_run, tmp_path, monkeypatch, kind
):
    rc, _ = auxiliary_run
    path = tmp_path / "invalid-ngram.safetensors"
    if kind == "directory":
        path.mkdir()
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(path))
    with pytest.raises(ValueError if kind == "directory" else FileNotFoundError):
        run(rc, log=lambda _: None)
    assert not rc.workdir.exists()
    assert not rc.dst.exists()
