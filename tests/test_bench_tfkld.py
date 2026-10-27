import math

from bench.tfkld import tf_kld


def test_tf_kld_is_the_token_weighted_mean_log_ratio():
    ref = {"t": {"a": [-1.0, -2.0], "b": [-0.5]}}
    cand = {"t": {"a": [-1.5, -2.0], "b": [-1.0]}}
    r = tf_kld(ref, cand)["t"]
    assert math.isclose(r["kld"], (0.5 + 0.0 + 0.5) / 3)
    assert r["tokens"] == 3 and r["items"] == 2


def test_tf_kld_skips_errors_and_length_mismatches():
    ref = {"t": {"a": [-1.0], "b": [-1.0, -1.0], "c": [-1.0]}}
    cand = {"t": {"a": {"error": "x"}, "b": [-2.0], "c": [-3.0]}}
    r = tf_kld(ref, cand)["t"]
    assert r["items"] == 1 and math.isclose(r["kld"], 2.0)
