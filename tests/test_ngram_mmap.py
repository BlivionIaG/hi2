# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``$HI2_NGRAM_MMAP``: the PLE table from disk, cast per gather, outside the module tree."""

import copy

import pytest
import torch
from safetensors.torch import save_file

from hi2.pipeline import _attach_mmap_ngram


class _Ple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = torch.nn.Identity()
        self.scale = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))


def test_attach_gathers_from_disk_and_casts(tmp_path, monkeypatch):
    table = torch.randn(64, 8, dtype=torch.bfloat16)
    save_file({"model.layers.1.ple.ngram_embedding.weight": table}, tmp_path / "ngram.safetensors")
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
    save_file({"model.layers.1.ple.ngram_embedding.weight": table}, tmp_path / "ngram.safetensors")
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(tmp_path / "ngram.safetensors"))
    model = _Ple()

    _attach_mmap_ngram(model, log=lambda _m: None)
    loaded = model.ngram_embedding.weight
    clone = copy.deepcopy(model)

    assert clone.ngram_embedding.weight is loaded
    assert clone.ngram_embedding.weight.data_ptr() == loaded.data_ptr()
    assert loaded.data_ptr() != table.data_ptr()  # read from the file, not the source tensor


def test_no_env_leaves_the_module_alone(monkeypatch):
    monkeypatch.delenv("HI2_NGRAM_MMAP", raising=False)
    model = _Ple()
    before = model.ngram_embedding
    _attach_mmap_ngram(model, log=lambda _m: None)
    assert model.ngram_embedding is before


def test_missing_table_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HI2_NGRAM_MMAP", str(tmp_path / "absent.safetensors"))
    with pytest.raises(Exception):
        _attach_mmap_ngram(_Ple(), log=lambda _m: None)
