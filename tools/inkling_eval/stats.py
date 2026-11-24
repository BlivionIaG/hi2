# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Paired statistics for the task evaluation.

Reuses the repository's tested inference code (``bench/compare.py``) instead of
carrying a second McNemar/bootstrap implementation, and adds the minimum
detectable difference at a category's own item count.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.compare import bootstrap_diff_ci, mcnemar_exact, n_needed  # noqa: E402

DEFAULT_REPS = 10000
DEFAULT_SEED = 0


def mdd(
    n: int,
    discordance: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    step: float = 0.005,
) -> float | None:
    """Smallest accuracy difference this item count detects at the given discordance."""
    if n <= 0:
        return None
    steps = int(round(0.5 / step))
    for index in range(1, steps + 1):
        delta = index * step
        if n_needed(max(discordance, delta + 1e-9), delta, alpha, power) <= n:
            return round(delta, 3)
    return None


def paired_stats(
    pairs: list[tuple[bool, bool]],
    *,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Accuracy of both runs on the same items, with the paired difference."""
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "ref_ok": 0,
            "cand_ok": 0,
            "ref_acc": None,
            "cand_acc": None,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "mcnemar_p": None,
            "flips_ref_right": 0,
            "flips_cand_right": 0,
            "discordance": None,
            "mdd": None,
            "mdd_assumed_0.10": None,
        }
    ref_ok = sum(1 for ref, _ in pairs if ref)
    cand_ok = sum(1 for _, cand in pairs if cand)
    b = sum(1 for ref, cand in pairs if ref and not cand)
    c = sum(1 for ref, cand in pairs if cand and not ref)
    low, high = bootstrap_diff_ci(b, c, n, reps=reps, seed=seed)
    discordance = (b + c) / n
    return {
        "n": n,
        "ref_ok": ref_ok,
        "cand_ok": cand_ok,
        "ref_acc": round(ref_ok / n, 4),
        "cand_acc": round(cand_ok / n, 4),
        "delta": round((cand_ok - ref_ok) / n, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "mcnemar_p": round(mcnemar_exact(b, c), 6),
        "flips_ref_right": b,
        "flips_cand_right": c,
        "discordance": round(discordance, 4),
        "mdd": mdd(n, discordance) if b + c else None,
        "mdd_assumed_0.10": mdd(n, 0.10),
    }
