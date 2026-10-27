# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench needle: generator determinism, the facts are in the prompt, lengths, scoring."""

import json

from bench.tasks import needle


def test_items_are_deterministic_and_complete():
    a, b = needle.make_items(0), needle.make_items(0)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(a) != json.dumps(needle.make_items(1))
    assert len(a) == len(needle.LENGTHS) * needle.PER_LENGTH
    assert len({i["id"] for i in a}) == len(a)
    groups = {i["group"] for i in a}
    assert groups == {
        f"{n // 1024}k-{k}" for n in needle.LENGTHS for k in ("single", "multikey")
    }


def test_facts_present_once_and_lengths():
    for it in needle.make_items(0):
        assert it["text"].count(it["answer"]) == 1
        for d in it["distractors"]:
            assert it["text"].count(d) == 1
        length = int(it["id"].split("/")[1][:-1]) * 1024
        est = needle.approx_tokens(it["text"])
        assert 0.8 * length < est < 0.95 * length, (it["id"], est)


def test_small_haystack_wraps():
    items = needle.make_items(0, text="line one\nline two\n")
    assert all(len(i["text"]) > 2000 for i in items)


def test_limit_keeps_every_length():
    assert {i["id"].split("/")[1] for i in needle.load(4, 0)} == {
        "2k",
        "8k",
        "16k",
        "32k",
    }


def test_fingerprint_is_stable():
    assert needle.fingerprint() == needle.fingerprint()


def test_score():
    item = {"answer": "4829131", "distractors": ["1234567"]}
    assert needle.score(item, "4829131")["ok"]
    assert needle.score(item, "The value is 4,829,131.")["ok"]
    r = needle.score(item, "1234567")
    assert not r["ok"] and r["distractor"]
