# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench ifeval-lite: every verifier on a passing and a failing reply, and the prompt
generator (determinism, coverage, no conflicting pairs)."""

import json

import pytest

from bench.tasks import ifeval

CASES = {  # kind: (params, passing reply, failing reply)
    "word_min": ({"n": 5}, "one two three four five", "one two"),
    "word_max": ({"n": 3}, "one two", "one two three four"),
    "no_comma": ({}, "no commas here", "one, two"),
    "lowercase": ({}, "all lower case.", "Not lower"),
    "uppercase": ({}, "ALL UPPER.", "Not Upper"),
    "keywords": ({"words": ["river", "orbit"]}, "A River in orbit.", "a river only"),
    "avoid_word": ({"word": "very"}, "quite nice", "Very nice"),
    "paragraphs": ({"n": 2}, "first\n\n***\n\nsecond", "first\n\nsecond"),
    "bullets": ({"n": 2}, "Intro\n* one\n- two\n", "* one\n* two\n* three"),
    "end_phrase": (
        {"phrase": "That is all for today."},
        "Text. That is all for today.\n",
        "That is all for today. More.",
    ),
    "quotes": ({}, '"wrapped"', 'not "wrapped"'),
    "json": ({}, '```json\n{"a": 1}\n```', "{a: 1}"),
    "title": ({}, "<<My Title>>\nbody", "My Title\nbody"),
    "postscript": ({"marker": "P.S."}, "body\n\nP.S. bye", "body ps bye"),
    "highlights": ({"n": 2}, "*one* and *two*", "*one* only"),
    "placeholders": ({"n": 2}, "[name] at [address]", "[name] only"),
    "sections": ({"n": 2}, "SECTION 1\na\nSECTION 2\nb", "SECTION 1\na"),
}


def test_cases_cover_every_kind():
    assert set(CASES) == set(ifeval.C)
    assert len(ifeval.C) >= 15


@pytest.mark.parametrize("kind", sorted(CASES))
def test_verifier(kind):
    params, good, bad = CASES[kind]
    assert ifeval.check(kind, good, params), (kind, good)
    assert not ifeval.check(kind, bad, params), (kind, bad)


def test_generator_is_deterministic_and_consistent():
    a, b = ifeval.generate(500, 0), ifeval.generate(500, 0)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(a) != json.dumps(ifeval.generate(500, 1))
    assert len({i["id"] for i in a}) == 500
    kinds = [c["kind"] for i in a for c in i["cons"]]
    assert set(kinds) == set(ifeval.C)
    for item in a:
        ks = [c["kind"] for c in item["cons"]]
        assert len(ks) == len(set(ks))
        if len(ks) == 2:
            assert frozenset(ks) not in ifeval.CONFLICTS
    assert ifeval.load(34, 0) == a[:34]
    assert {i["group"] for i in a[:34]} == set(ifeval.C)  # a prefix covers every kind


def test_score_strict_and_fraction():
    item = {"cons": [{"kind": "no_comma"}, {"kind": "lowercase"}]}
    assert ifeval.score(item, "all fine here")["ok"]
    r = ifeval.score(item, "Not, fine")
    assert not r["ok"] and r["frac"] == 0.0
    r = ifeval.score(item, "Fine here")
    assert r["frac"] == 0.5 and r["checks"] == {"no_comma": True, "lowercase": False}
