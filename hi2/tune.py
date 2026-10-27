# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Pick the quality target from a measured fidelity target (``--target-kld``).

The allocator takes ``--max-error-ratio``, which has no direct meaning for quality. This
module measures the KL(original || quantised) of PTQ checkpoints made at a few ratios,
fits ``log KLD`` against ``log ratio`` (monotone, piecewise linear) and returns the
largest ratio, hence the fewest bits, whose predicted KLD is within the target. QAD only
lowers the KLD of the chosen ratio, so the final model meets the target with margin.
"""

import math

from .evalq import LOSSLESS_KLD  # noqa: F401  (re-exported: the default target)

START_RATIOS = (2.0, 4.0, 8.0, 16.0)
MAX_ROUNDS = 4  # extensions of the measured range after the first pass


def fit_ratio(points: list[tuple[float, float, float]], target: float) -> dict:
    """``points`` are measured ``(ratio, kld, bpw)``. Returns ``ratio`` (the largest whose
    interpolated KLD is at most ``target``), the interpolated ``kld`` and ``bpw`` there,
    and ``status``: ``ok`` (target inside the measured range), ``above`` (even the
    loosest ratio is under target: try larger ratios) or ``below`` (even the tightest is
    over target: try smaller ratios, or the target is out of reach)."""
    pts = sorted(points)
    if len(pts) < 2:
        raise ValueError("need at least two measured points")
    # KLD is monotone in the ratio in expectation; enforce it against measurement noise
    ks, run = [], 0.0
    for _, k, _ in pts:
        run = max(run, k)
        ks.append(max(run, 1e-12))
    rs = [p[0] for p in pts]
    bs = [p[2] for p in pts]
    if ks[0] > target:
        return {"ratio": rs[0], "kld": ks[0], "bpw": bs[0], "status": "below"}
    if ks[-1] <= target:
        return {"ratio": rs[-1], "kld": ks[-1], "bpw": bs[-1], "status": "above"}
    for i in range(len(rs) - 1):
        if ks[i] <= target < ks[i + 1]:
            t = (math.log(target) - math.log(ks[i])) / (
                math.log(ks[i + 1]) - math.log(ks[i])
            )
            ratio = math.exp(math.log(rs[i]) + t * (math.log(rs[i + 1]) - math.log(rs[i])))
            bpw = bs[i] + t * (bs[i + 1] - bs[i])
            return {"ratio": ratio, "kld": target, "bpw": bpw, "status": "ok"}
    raise AssertionError("unreachable")


def next_ratios(points: list[tuple[float, float, float]], fit: dict) -> list[float]:
    """Ratios to measure next when the target lies outside the measured range."""
    rs = sorted(p[0] for p in points)
    if fit["status"] == "above":
        return [rs[-1] * 2, rs[-1] * 4]
    if fit["status"] == "below":
        return [max(rs[0] / 2, 1.0), max(rs[0] / 4, 1.0)] if rs[0] > 1.0 else []
    return []


def tune(measure, target: float, ratios=START_RATIOS, log=print) -> dict:
    """Search the ratio. ``measure(ratios) -> [(ratio, kld, bpw), ...]`` quantises (PTQ,
    sharing the expensive phases) and evaluates one checkpoint per ratio."""
    points = list(measure(sorted(set(ratios))))
    for rnd in range(MAX_ROUNDS + 1):
        fit = fit_ratio(points, target)
        log(
            f"tune round {rnd}: target KLD {target:.4f}, measured "
            + ", ".join(f"r{r:g}: {k:.4f}@{b:.2f}" for r, k, b in sorted(points))
            + f" -> {fit['status']}, ratio {fit['ratio']:.2f}"
        )
        more = [r for r in next_ratios(points, fit) if r not in {p[0] for p in points}]
        if fit["status"] == "ok" or not more or rnd == MAX_ROUNDS:
            break
        points += measure(more)
    fit["points"] = sorted(points)
    return fit
