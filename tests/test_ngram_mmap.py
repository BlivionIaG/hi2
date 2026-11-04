# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``$HI2_NGRAM_MMAP``: the PLE table from disk, cast per gather, outside the module tree."""

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from hi2.cache import auxiliary_model_inputs
from hi2.pipeline import _attach_mmap_ngram


class _Ple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = torch.nn.Identity()
        self.scale = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))


def test_attach_gathers_from_disk_and_casts(tmp_path, monkeypatch):
    table = torch.randn(64, 8, dtype=torch.bfloat16)
    save_file(
        {"model.layers.1.ple.ngram_embedding.weight": table},
        tmp_path / "ngram.safetensors",
    )
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(tmp_path / "ngram.safetensors"))
    model = _Ple()

    _attach_mmap_ngram(model, log=lambda _m: None)

    emb = model.ngram_embedding
    assert emb.weight.dtype == torch.bfloat16
    assert emb.weight.device.type == "cpu"
    ids = torch.tensor([[1, 2], [3, 63]])
    out = emb(ids)
    assert out.dtype == torch.float16  # the model dtype, not the file's
    assert torch.equal(out.float(), table[ids].float())


def test_deepcopy_does_not_copy_the_table(tmp_path, monkeypatch):
    """The layer-wise pass deep-copies every layer once per device; a 95 GiB attribute
    would be copied with it, which is what made the Flash-Next pass fill the cards."""
    table = torch.randn(64, 8, dtype=torch.bfloat16)
    save_file(
        {"model.layers.1.ple.ngram_embedding.weight": table},
        tmp_path / "ngram.safetensors",
    )
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(tmp_path / "ngram.safetensors"))
    model = _Ple()

    _attach_mmap_ngram(model, log=lambda _m: None)
    loaded = model.ngram_embedding.weight
    clone = copy.deepcopy(model)

    assert clone.ngram_embedding.weight is loaded
    assert clone.ngram_embedding.weight.data_ptr() == loaded.data_ptr()
    assert (
        loaded.data_ptr() != table.data_ptr()
    )  # read from the file, not the source tensor


def test_no_env_leaves_the_module_alone(monkeypatch):
    monkeypatch.delenv("HI2_NGRAM_MMAP", raising=False)
    model = _Ple()
    before = model.ngram_embedding
    _attach_mmap_ngram(model, log=lambda _m: None)
    assert model.ngram_embedding is before


def test_missing_table_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(tmp_path / "absent.safetensors"))
    with pytest.raises(FileNotFoundError):
        _attach_mmap_ngram(_Ple(), log=lambda _m: None)


@pytest.mark.parametrize("value", [None, ""])
def test_inactive_auxiliary_input_has_no_identity(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("HI2_NGRAM_MMAP", raising=False)
    else:
        monkeypatch.setenv("HI2_NGRAM_MMAP", value)
    assert auxiliary_model_inputs() == []


def test_auxiliary_input_records_canonical_path_and_full_digest(tmp_path, monkeypatch):
    path = tmp_path / "ngram.safetensors"
    save_file({"ngram_embedding.weight": torch.ones(64, 8)}, path)
    alias = tmp_path / "alias.safetensors"
    alias.symlink_to(path.name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HI2_NGRAM_MMAP", alias.name)
    records = auxiliary_model_inputs()
    assert records == [
        {
            "role": "ngram_mmap",
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(path))
    assert auxiliary_model_inputs() == records


def test_auxiliary_input_tracks_mmap_bytes_beyond_first_chunk(tmp_path, monkeypatch):
    path = tmp_path / "ngram.safetensors"
    table = torch.ones(65_537, 8, dtype=torch.bfloat16)
    save_file({"model.layers.1.ple.ngram_embedding.weight": table}, path)
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(path))
    model = _Ple()
    _attach_mmap_ngram(model, log=lambda _: None)
    ids = torch.tensor([[0, table.shape[0] - 1]])
    before_gather = model.ngram_embedding(ids)
    before_identity = auxiliary_model_inputs()
    before = path.stat()
    assert before.st_size > 2**20

    # Change the final bf16 value from 1 to 4 without changing inode/size/mtime.
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"\x40")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    reloaded = _Ple()
    _attach_mmap_ngram(reloaded, log=lambda _: None)
    after_gather = reloaded.ngram_embedding(ids)
    assert before_gather[0, -1, -1].item() == 1
    assert after_gather[0, -1, -1].item() == 4
    assert torch.equal(before_gather[0, 0], after_gather[0, 0])
    after_identity = auxiliary_model_inputs()
    assert after_identity[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert after_identity[0]["sha256"] != before_identity[0]["sha256"]
    assert after_identity[0]["path"] == before_identity[0]["path"]


@pytest.mark.parametrize("kind", ["missing", "dangling_symlink", "directory"])
def test_auxiliary_input_rejects_missing_or_nonregular_path(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / "ngram.safetensors"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling_symlink":
        path.symlink_to(tmp_path / "missing.safetensors")
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(path))
    with pytest.raises(ValueError if kind == "directory" else FileNotFoundError):
        auxiliary_model_inputs()


def test_auxiliary_input_rejects_fifo_without_blocking(tmp_path, monkeypatch):
    path = tmp_path / "ngram.safetensors"
    os.mkfifo(path)
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(path))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hi2.cache import auxiliary_model_inputs\n"
            "try:\n"
            "    auxiliary_model_inputs()\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('accepted a nonregular auxiliary input')\n",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
