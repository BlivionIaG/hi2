# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU tests for the v2 calibration-text builder."""

import json
from pathlib import Path

import numpy as np

from experiments.inkling_calib_v2_text import interleave, load_calib


def test_interleave_is_round_robin_over_all_windows():
    entries = [
        ("a", np.zeros((2, 4), dtype=np.int32)),
        ("b", np.zeros((3, 4), dtype=np.int32)),
    ]
    assert interleave(entries) == [
        ("a", 0),
        ("b", 0),
        ("a", 1),
        ("b", 1),
        ("b", 2),
    ]


def test_load_calib_uses_the_verified_manifest_and_only_calib_files(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "hi2.inkling-calib-v2/1",
                "ok": True,
                "excluded_license_classes": ["gpl2"],
                "excluded_records": 60,
                "files": {
                    "calib_code__python__2048_ids.npy": {"bytes": 1},
                    "calib_multilingual__de__2048_ids.npy": {"bytes": 1},
                    "heldout_code__python__2048_ids.npy": {"bytes": 1},
                },
            }
        )
    )
    np.save(tmp_path / "calib_code__python__2048_ids.npy", np.zeros((2, 4), np.int32))
    np.save(
        tmp_path / "calib_multilingual__de__2048_ids.npy", np.ones((1, 4), np.int32)
    )
    manifest, entries = load_calib(tmp_path)
    assert manifest["excluded_license_classes"] == ["gpl2"]
    assert [k for k, _ in entries] == [
        "code__python__2048",
        "multilingual__de__2048",
    ]
    assert interleave(entries) == [
        ("code__python__2048", 0),
        ("multilingual__de__2048", 0),
        ("code__python__2048", 1),
    ]
