# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``hi2 calib`` (CPU): sources of every kind are read, the token shares follow the weights
at every point of the file, the held-out text shares no document with the calibration text,
and the result is deterministic."""

import json

import pytest

from hi2.calib import build_mix, parse_source, read_documents


@pytest.fixture
def sources(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("\n\n".join(f"alpha {i} " + "x" * 900 for i in range(60)))
    b = tmp_path / "b.jsonl"
    b.write_text(
        "\n".join(json.dumps({"text": f"beta {i} " + "y" * 1500}) for i in range(60))
    )
    c = tmp_path / "c.json"
    c.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": f"gamma {i}"},
                        {"role": "assistant", "content": "z" * 1200},
                    ]
                }
                for i in range(60)
            ]
        )
    )
    return a, b, c


def test_reads_text_jsonl_and_chat_sources(sources):
    a, b, c = sources
    assert all(len(d) >= 2000 for d in read_documents(a)[:-1])
    assert read_documents(b)[0].startswith("beta 0")
    chat = read_documents(c)[0]
    assert chat.startswith("user: gamma 0\nassistant: zzz")
    assert parse_source("data/x.jsonl:2.5") == (parse_source("data/x.jsonl")[0], 2.5)
    assert parse_source("data/x.jsonl")[1] == 1.0


def test_shares_follow_the_weights_through_the_whole_file(sources):
    a, b, c = sources
    text, _, rep = build_mix([(a, 2), (b, 1), (c, 1)], 8000)
    toks = {k.rsplit("/", 1)[1]: v["tokens"] for k, v in rep["sources"].items()}
    total = sum(toks.values())
    assert toks["a.txt"] / total == pytest.approx(0.5, abs=0.1)
    assert toks["b.jsonl"] / total == pytest.approx(0.25, abs=0.1)
    for frac in (0.25, 0.5, 1.0):  # every prefix holds every domain
        head = text[: int(len(text) * frac)]
        assert all(w in head for w in ("alpha", "beta", "gamma"))
    assert rep["tokens"] >= 0.95 * 8000 and "warning" not in rep


def test_held_out_text_is_disjoint_and_deterministic(sources):
    a, b, c = sources
    cal, held, rep = build_mix([(a, 1), (b, 1), (c, 1)], 6000, holdout_share=0.2)
    assert rep["held_out_documents"] > 0
    capped = build_mix(
        [(a, 1), (b, 1), (c, 1)], 6000, holdout_share=0.2, holdout_tokens=1500
    )
    assert capped[2]["held_out_tokens"] < 2500 and capped[1]
    docs = lambda t: set(t.split("\n\n"))  # noqa: E731
    assert not docs(cal) & docs(held)
    again = build_mix([(a, 1), (b, 1), (c, 1)], 6000, holdout_share=0.2)
    assert again[:2] == (cal, held)


def test_warns_when_the_sources_are_too_small_and_rejects_bad_input(sources, tmp_path):
    a, _, _ = sources
    _, _, rep = build_mix([(a, 1)], 10**7)
    assert "fewer than" in rep["warning"]
    with pytest.raises(ValueError, match="at least one source"):
        build_mix([], 100)
    with pytest.raises(ValueError, match="weight above 0"):
        build_mix([(a, 0)], 100)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no documents"):
        build_mix([(empty, 1)], 100)
