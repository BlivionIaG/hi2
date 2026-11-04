import math

from bench.tfkld import mean_logprob_gap


def test_gap_is_the_token_weighted_mean_log_ratio():
    ref = {"t": {"a": [-1.0, -2.0], "b": [-0.5]}}
    cand = {"t": {"a": [-1.5, -2.0], "b": [-1.0]}}
    r = mean_logprob_gap(ref, cand)["t"]
    assert math.isclose(r["mean_logprob_gap"], (0.5 + 0.0 + 0.5) / 3)
    assert r["tokens"] == 3 and r["items"] == 2


def test_gap_skips_errors_and_length_mismatches():
    ref = {"t": {"a": [-1.0], "b": [-1.0, -1.0], "c": [-1.0]}}
    cand = {"t": {"a": {"error": "x"}, "b": [-2.0], "c": [-3.0]}}
    r = mean_logprob_gap(ref, cand)["t"]
    assert r["items"] == 1 and math.isclose(r["mean_logprob_gap"], 2.0)


def test_gap_can_be_negative_when_candidate_prefers_reference_completion():
    ref = {"t": {"answer": [-2.0, -3.0]}}
    cand = {"t": {"answer": [-1.0, -1.0]}}
    result = mean_logprob_gap(ref, cand)["t"]
    assert result["mean_logprob_gap"] == -1.5
    assert result["tokens"] == 2
    assert result["status"] == "complete"
