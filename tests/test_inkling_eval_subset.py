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
    manifest = build_subset(src, dst, max_length=16384, windows=5, balanced=False)
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


def _make_balanced_source(root: Path) -> Path:
    root.mkdir()
    files = {}
    for lang in ("c", "cpp", "javascript", "json", "python", "rust", "shell", "sql", "typescript"):
        files[f"heldout_code__{lang}__2048_ids.npy"] = ("code", lang, 2048, 2)
    for lang in ("ar", "de", "es", "fr", "hi", "ja", "ko", "pt", "ru", "tr", "vi", "zh"):
        files[f"heldout_multilingual__{lang}__2048_ids.npy"] = ("multilingual", lang, 2048, 2)
    for cat, lang in (("agentic", "en"), ("math_reasoning", "en"), ("general_chat", "en")):
        files[f"heldout_{cat}__{lang}__2048_ids.npy"] = (cat, lang, 2048, 20)
    for length in (8192, 16384):
        files[f"heldout_long_context__en__{length}_ids.npy"] = ("long_context", "en", length, 8)
    metas = {}
    for name, (cat, lang, length, windows) in files.items():
        ids = np.arange(windows * length, dtype=np.int32).reshape(windows, length)
        np.save(root / name, ids)
        metas[name] = {
            "bytes": (root / name).stat().st_size,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "category": cat,
            "language": lang,
            "length": length,
            "windows": windows,
        }
    (root / "manifest.json").write_text(
        json.dumps({"schema": "hi2.inkling-calib-v2/1", "ok": True, "gates": {}, "files": metas})
    )
    return root


def test_balanced_subset_covers_quotas_languages_and_both_long_lengths(tmp_path: Path):
    src = _make_balanced_source(tmp_path / "src")
    dst = tmp_path / "dst"
    manifest = build_subset(src, dst, max_length=16384, windows=74, balanced=True)
    files = manifest["files"]
    assert manifest["subset"]["windows"] == 74
    cats = {}
    for name, info in files.items():
        cats[info["category"]] = cats.get(info["category"], 0) + info["windows"]
    assert cats["code"] == 18
    assert cats["multilingual"] == 24
    assert cats["agentic"] == 8
    assert cats["math_reasoning"] == 8
    assert cats["general_chat"] == 8
    assert cats["long_context"] == 8
    code_langs = {i["language"]: i["windows"] for n, i in files.items() if i["category"] == "code"}
    multi_langs = {i["language"]: i["windows"] for n, i in files.items() if i["category"] == "multilingual"}
    assert set(code_langs.values()) == {2}
    assert set(multi_langs.values()) == {2}
    long_lens = {i["length"]: i["windows"] for n, i in files.items() if i["category"] == "long_context"}
    assert long_lens == {8192: 4, 16384: 4}
    assert "heldout_long_context__en__32768_ids.npy" not in (dst / "manifest.json").read_text()


def test_no_long_preflight_falls_back_to_101_balanced_windows(tmp_path: Path):
    src = _make_balanced_source(tmp_path / "src")
    dst = tmp_path / "dst"
    manifest = build_subset(src, dst, max_length=2048, windows=101, balanced=True)
    files = manifest["files"]
    cats = {}
    for name, info in files.items():
        cats[info["category"]] = cats.get(info["category"], 0) + info["windows"]
    assert manifest["subset"]["windows"] == 101
    assert cats == {"code": 18, "multilingual": 24, "agentic": 20,
                    "math_reasoning": 20, "general_chat": 19}
    assert all(n >= 8 for n in cats.values())
    assert "long_context" not in cats
    code_langs = {i["language"]: i["windows"] for i in files.values() if i["category"] == "code"}
    multi_langs = {i["language"]: i["windows"] for i in files.values() if i["category"] == "multilingual"}
    assert set(code_langs.values()) == {2}
    assert set(multi_langs.values()) == {2}
    assert "heldout_long_context" not in (dst / "manifest.json").read_text()


def test_category_quotas_at_120():
    from experiments.inkling_eval_subset import category_quotas

    q = category_quotas(120)
    assert q == {"code": 18, "multilingual": 24, "agentic": 20,
                 "math_reasoning": 20, "general_chat": 19, "long_context": 19}
    assert sum(q.values()) == 120
