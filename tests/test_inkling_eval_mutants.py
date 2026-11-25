# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Scorer mutants: every mutation of the scoring logic must be killed."""

from tools.inkling_eval import mutants


def test_every_scorer_mutant_is_killed():
    results = mutants.run_mutants(verbose=False)
    assert len(results) >= 10
    survivors = [result["name"] for result in results if not result["killed"]]
    assert not survivors, f"surviving mutants: {survivors}"
    assert all(result["by"] for result in results)


def test_mutant_cli_exits_zero():
    assert mutants.main() == 0
