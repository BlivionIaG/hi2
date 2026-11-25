# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU tests for the v2 held-out subset builder."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.inkling_eval_subset import build_subset, select_windows


def _make_source(root: Path) -> Path:
    root.mkdir()
    files = {
        "heldout_a__en__2048_ids.npy": (2048, 3),
        "heldout_b__en__2048_ids.npy": (2048, 2),
        "heldout_long__en__16384_ids.npy": (16384, 2),
        "heldout_long__en__32768_ids.npy": (32768, 2),
    }
    metas = {}
    for name, (length, windows) in files.items():
        ids = np.arange(windows * length, dtype=np.int32).reshape(windows, length)
        np.save(root / name, ids)
        metas[name] = {
            "bytes": (root / name).stat().st_size,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "category": "long_context" if "long" in name else "code",
            "language": "en",
            "length": length,
            "windows": windows,
        }
    manifest = {
        "schema": "hi2.inkling-calib-v2/1",
        "ok": True,
        "gates": {},
        "files": metas,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_select_windows_takes_one_per_file_then_extras_in_sorted_order():
    selected = select_windows({"b": 3, "a": 2, "c": 1}, 31)
    assert selected == {"a": 2, "b": 3, "c": 1}
    selected = select_windows({"b": 3, "a": 2, "c": 1}, 4)
    assert selected == {"a": 2, "b": 1, "c": 1}


def test_build_subset_truncates_and_updates_hash_and_windows(tmp_path: Path):
    src = _make_source(tmp_path / "src")
    dst = tmp_path / "dst"
    manifest = build_subset(src, dst, max_length=16384, windows=5)
    files = manifest["files"]
    assert set(files) == {
        "heldout_a__en__2048_ids.npy",
        "heldout_b__en__2048_ids.npy",
        "heldout_long__en__16384_ids.npy",
    }
    assert files["heldout_a__en__2048_ids.npy"]["windows"] == 2
    assert files["heldout_b__en__2048_ids.npy"]["windows"] == 2
    assert files["heldout_long__en__16384_ids.npy"]["windows"] == 1
    assert sum(i["windows"] for i in files.values()) == 5
    for name, info in files.items():
        data = (dst / name).read_bytes()
        assert info["bytes"] == len(data)
        assert info["sha256"] == hashlib.sha256(data).hexdigest()
    assert "heldout_long__en__32768_ids.npy" not in (dst / "manifest.json").read_text()
    assert (dst / "manifest.json").exists()
