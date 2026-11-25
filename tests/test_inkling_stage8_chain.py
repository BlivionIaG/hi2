# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""CPU tests for the stage-8 driver (no GPU, no model)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "stage8_chain.sh"


def test_every_eval_gets_the_stock_source():
    text = CHAIN.read_text()
    assert text.count("inkling_kl_eval.py") == 5
    # Five eval invocations plus the quantize call; without --src the eval exits on
    # argparse before using the GPU.
    assert text.count('--src "$SRC"') == 6


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
