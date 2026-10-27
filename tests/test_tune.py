"""The fidelity-target search (hi2.tune): interpolation, range extension, monotonicity."""

import math

import pytest

from hi2.tune import fit_ratio, tune


def curve(r):  # KLD grows as a power of the ratio, bpw falls with it
    return 0.004 * r**0.9, 4.5 - 0.5 * math.log2(r)


def test_fit_interpolates_inside_the_measured_range():
    pts = [(r, *curve(r)) for r in (2.0, 4.0, 8.0, 16.0)]
    f = fit_ratio(pts, 0.02)
    assert f["status"] == "ok"
    assert f["ratio"] == pytest.approx((0.02 / 0.004) ** (1 / 0.9), rel=1e-6)
    assert f["bpw"] == pytest.approx(curve(f["ratio"])[1], abs=0.05)


def test_fit_reports_targets_outside_the_range():
    pts = [(r, *curve(r)) for r in (2.0, 4.0)]
    assert fit_ratio(pts, 1.0)["status"] == "above"
    assert fit_ratio(pts, 1e-5)["status"] == "below"


def test_fit_enforces_monotone_kld_against_noise():
    pts = [(2.0, 0.010, 4.0), (4.0, 0.008, 3.8), (8.0, 0.030, 3.4)]
    f = fit_ratio(pts, 0.02)  # the 0.008 dip is lifted to 0.010
    assert f["status"] == "ok" and 4.0 < f["ratio"] < 8.0


def test_tune_extends_the_range_until_the_target_is_bracketed():
    calls = []

    def measure(ratios):
        calls.append(list(ratios))
        return [(r, *curve(r)) for r in ratios]

    f = tune(measure, 0.2, ratios=(2.0, 4.0), log=lambda *_: None)
    assert f["status"] == "ok" and len(calls) >= 2
    assert curve(f["ratio"])[0] == pytest.approx(0.2, rel=0.02)
