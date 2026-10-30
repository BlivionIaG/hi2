# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The probe dtype resolution behind ``--sens-dtype auto``."""

import json

from hi2.sensitivity import fp32_weight_bytes, plan_placement, probe_fits


def test_fp32_weight_bytes_scales_the_stored_bf16(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    assert fp32_weight_bytes(tmp_path) == 2 << 20


def test_fp32_weight_bytes_leaves_float32_alone(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\0" * (1 << 20))
    assert fp32_weight_bytes(tmp_path) == 2 << 20


def test_probe_fits_single_shard_and_the_27b_case():
    gib = 1 << 30
    assert probe_fits(10 * gib, 30 * gib, 1)  # one device holds it
    assert probe_fits(80 * gib, 30 * gib, 4)  # 20 GiB a shard
    assert not probe_fits(108 * gib, 30 * gib, 4)  # 27 GiB a shard, over the guard
    assert probe_fits(108 * gib, 30 * gib, 8)  # 13.5 GiB a shard
    assert plan_placement(10 * gib, 30 * gib, 1) == "single"
    assert plan_placement(80 * gib, 30 * gib, 4) == "shard"
