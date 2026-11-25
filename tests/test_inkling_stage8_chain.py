# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU tests for the stage-8 driver (no GPU, no model)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "stage8_chain.sh"


def test_every_eval_gets_the_stock_source():
    text = CHAIN.read_text()
    # Five real evals and five identical-argv preflight --help checks.
    assert text.count("inkling_kl_eval.py") == 10
    assert text.count('--src "$SRC"') == 11


def test_balanced_120_windows_and_budget():
    text = CHAIN.read_text()
    assert "SUB=${SUB:-120}" in text
    assert "--windows \"$SUB\" --balanced" in text
    assert "25.0 / 32.0" in text
    assert "101 windows" in text
    assert 'GATE_SRC=${GATE_SRC:-$TREE/stage8_GATE_${SUB}.md}' in text
    assert 'cmp -s "$GATE_SRC" "$GATE"' in text


def test_committed_gates_match_their_subset_size():
    for n in (120, 96):
        text = (ROOT / f"stage8_GATE_{n}.md").read_text()
        match = re.search(r"^subset n=(\d+)", text, re.M)
        assert match and int(match.group(1)) == n
        assert "32K is not preflighted and is excluded" in text
        assert "16384" in text and "8192" in text
    text = (ROOT / "stage8_GATE_101.md").read_text()
    match = re.search(r"^subset n=(\d+)", text, re.M)
    assert match and int(match.group(1)) == 101
    assert "Long context is gated separately" in text
    assert "served" in text and "max_model_len 32768" in text


def test_8192_preflight_failure_continues_with_101():
    text = CHAIN.read_text()
    assert "EVAL_MAX=2048" in text
    assert "SUB=101" in text
    assert "no retry" in text
    assert "preflight-eval-8192-failed" not in text


def _quantize_argv():
    return [
        "quantize", "MODEL", "OUT",
        "--workdir", "WORK", "--cache-dir", "CACHE",
        "--devices", "cuda:0,cuda:1", "--compute-dtype", "float16",
        "--moe-layout", "stacked", "--include", r"\.mlp\.experts\.",
        "--exclude", "x", "--target-bpw", "3.5", "--calib", "CALIB",
        "--calib-samples", "256", "--seq-len", "2048",
        "--sens-seqs", "8", "--sens-len", "512", "--sensitivity", "none",
        "--no-qad",
    ]


def _eval_argv(dst, ids, out, windows=0, split=("heldout",), batch="1"):
    return [
        "--src", "STOCK", "--dst", dst, "--ids", ids, "--windows", str(windows),
        *(["--split", *split] if split else []),
        "--devices", "cuda:0", "cuda:1", "--compute-dtype", "float32",
        "--groups", "none", "--batch-windows", batch,
        "--position-buckets", "0-2048,2048-8192", "--out", out,
    ]


def test_every_gpu_argv_parses_against_the_real_clis():
    from hi2.__main__ import _UsageError, build_parser
    from experiments.inkling_kl_eval import parser as eval_parser

    text = CHAIN.read_text()
    assert text.count("--devices cuda:0,cuda:1") == 2
    assert text.count("--devices cuda:0 cuda:1") == 10

    ns = build_parser().parse_args(_quantize_argv())
    assert ns.devices == "cuda:0,cuda:1"

    # The two-value quantize form is the exact mistake that killed the chain;
    # argparse must reject it.
    bad = _quantize_argv()
    i = bad.index("--devices")
    bad[i + 1 : i + 2] = ["cuda:0", "cuda:1"]
    with pytest.raises(_UsageError) as err:
        build_parser().parse_args(bad)
    assert "unrecognized arguments: cuda:1" in str(err.value)

    evals = [
        _eval_argv("OLD", "PRE/window_8192.npy", "PRE/eval.json", 1, split=(), batch="1"),
        _eval_argv("OLD", "EVAL_IDS", "EVAL/old_heldout.json", 0, ("heldout",), "1"),
        _eval_argv("NEW", "EVAL_IDS", "EVAL/new_heldout.json", 0, ("heldout",), "1"),
        _eval_argv("OLD", "OOD.npy", "EVAL/old_ood.json", 8, split=(), batch="1"),
        _eval_argv("NEW", "OOD.npy", "EVAL/new_ood.json", 8, split=(), batch="1"),
    ]
    for argv in evals:
        ns = eval_parser().parse_args(argv)
        assert ns.devices == ["cuda:0", "cuda:1"]
