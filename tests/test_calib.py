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


def _jsonl(path, documents):
    path.write_text(
        "\n".join(json.dumps({"text": text}) for text in documents), encoding="utf-8"
    )
    return path


def test_five_document_requested_holdout_is_not_empty(tmp_path):
    # The reported reproducer used five documents and a 10% holdout. Positional
    # blocks of ten previously returned an empty holdout without a warning.
    docs = [f"document {i} " * 100 for i in range(5)]
    path = _jsonl(tmp_path / "five.jsonl", docs)
    cal, held, report = build_mix([(path, 1.0)], tokens=1000, holdout_share=0.1)
    assert held
    assert report["held_out_documents"] == 1
    assert report["calibration_documents"] == 4
    assert report["held_out_tokens"] == len(held) // 4
    assert report["tokens"] == len(cal) // 4
    assert report["tokens"] >= 1000
    assert not set(cal.split("\n\n")) & set(held.split("\n\n"))
    assert report["unique_documents"] == 5
    assert report["reserved_holdout_share"] == 0.2
    assert report["preparation"]["token_count"] == "characters/4-estimate"
    assert "holdout has" in report["warning"]
    assert "200000 requested" in report["warning"]
    assert "evaluation requirement" in report["warning"]


def test_explicit_holdout_minimum_checks_the_available_tokens(tmp_path):
    path = _jsonl(tmp_path / "five.jsonl", [f"document {i} " * 100 for i in range(5)])
    with pytest.raises(
        ValueError, match=r"holdout has 1 documents and 275 tokens.*need at least 300"
    ):
        build_mix(
            [(path, 1)],
            1000,
            holdout_share=0.1,
            holdout_tokens=1000,
            min_holdout_tokens=300,
        )
    cal, held, report = build_mix(
        [(path, 1)],
        1000,
        holdout_share=0.1,
        holdout_tokens=275,
        min_holdout_tokens=275,
    )
    assert cal and held
    assert report["held_out_tokens"] == 275
    assert "warning" not in report


def test_reversed_duplicate_twenty_document_sources_stay_disjoint(tmp_path):
    # The audited A + reversed(A) case put all twenty contents in both splits.
    docs = [
        f"Unique document number {i}: " + "calibration example " * 20 for i in range(20)
    ]
    a = _jsonl(tmp_path / "a.jsonl", docs)
    b = _jsonl(tmp_path / "b.jsonl", reversed(docs))
    sources = [(a, 1.0), (b, 1.0)]
    result = build_mix(sources, tokens=100000, holdout_share=0.5)
    cal, held, report = result
    train_docs, holdout_docs = set(cal.split("\n\n")), set(held.split("\n\n"))
    assert len(train_docs) == len(holdout_docs) == 10
    assert not train_docs & holdout_docs
    assert train_docs | holdout_docs == set(docs)
    assert report["documents"] == 40
    assert report["unique_documents"] == 20
    assert report["duplicate_documents"] == 20
    assert report["cross_source_duplicate_documents"] == 20
    assert report["calibration_documents"] == report["held_out_documents"] == 20
    assert report["calibration_unique_documents"] == 10
    assert report["held_out_unique_documents"] == 10
    assert report["split_method"] == "sha256-exact-text-v1"
    assert build_mix(sources, tokens=100000, holdout_share=0.5) == result

    # Assignment is global content-based, independent of source and record order.
    reordered = build_mix(list(reversed(sources)), 100000, holdout_share=0.5)
    assert set(reordered[0].split("\n\n")) == train_docs
    assert set(reordered[1].split("\n\n")) == holdout_docs
    _jsonl(a, reversed(docs))
    _jsonl(b, docs)
    reread = build_mix(sources, 100000, holdout_share=0.5)
    assert set(reread[0].split("\n\n")) == train_docs
    assert set(reread[1].split("\n\n")) == holdout_docs
    for source in report["sources"].values():
        assert source["documents"] == 20
        assert source["used"] == source["held_out_documents"] == 10
        assert source["requested_share"] == 0.5
        assert source["token_share"] == 0.5
        assert source["held_out_token_share"] == 0.5


def test_duplicate_occurrences_within_a_source_share_one_assignment(tmp_path):
    docs = ["alpha " * 50, "beta " * 50, "gamma " * 50, "delta " * 50]
    path = _jsonl(tmp_path / "repeated.jsonl", docs + list(reversed(docs)) + docs)
    cal, held, report = build_mix([(path, 1)], 100000, holdout_share=0.5)
    assert not set(cal.split("\n\n")) & set(held.split("\n\n"))
    assert report["documents"] == 12
    assert report["unique_documents"] == 4
    assert report["duplicate_documents"] == 8
    assert report["cross_source_duplicate_documents"] == 0
    assert report["calibration_documents"] == report["held_out_documents"] == 6


def test_one_unique_document_cannot_supply_both_splits(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", ["same document " * 50] * 5)
    b = _jsonl(tmp_path / "b.jsonl", ["same document " * 50] * 5)
    with pytest.raises(ValueError, match="at least two unique documents"):
        build_mix([(a, 1), (b, 1)], 100, holdout_share=0.1)


@pytest.mark.parametrize("share", [1e-100, 0.1, 0.5, 0.999999999])
def test_two_document_corpus_keeps_both_requested_splits_usable(tmp_path, share):
    docs = ["first document", "second document"]
    path = _jsonl(tmp_path / "two.jsonl", docs)
    cal, held, report = build_mix(
        [(path, 1)], 100, count=len, holdout_share=share, holdout_tokens=10
    )
    assert {cal, held} == set(docs)
    assert report["calibration_documents"] == report["held_out_documents"] == 1
    assert report["reserved_holdout_share"] == 0.5


@pytest.mark.parametrize("share", [-0.1, 1, 1.1, float("inf"), float("nan"), True])
def test_invalid_holdout_boundaries_fail_before_reading_sources(tmp_path, share):
    with pytest.raises(ValueError, match="holdout_share"):
        build_mix([(tmp_path / "not-read.jsonl", 1)], 100, holdout_share=share)


@pytest.mark.parametrize(
    "weight", [0, -1, float("inf"), float("-inf"), float("nan"), True]
)
def test_invalid_weights_fail_before_reading_sources(tmp_path, weight):
    with pytest.raises(ValueError, match="finite weight above 0"):
        build_mix([(tmp_path / "not-read.jsonl", weight)], 100)


@pytest.mark.parametrize("field", ["tokens", "holdout_tokens", "min_holdout_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), float("nan"), True])
def test_invalid_token_requirements_fail_before_reading_sources(tmp_path, field, value):
    kwargs = {"tokens": 100, "holdout_share": 0.2, field: value}
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        build_mix([(tmp_path / "not-read.jsonl", 1)], **kwargs)


def test_holdout_minimum_cannot_exceed_its_target(tmp_path):
    with pytest.raises(ValueError, match="must not exceed holdout_tokens"):
        build_mix(
            [(tmp_path / "not-read.jsonl", 1)],
            100,
            holdout_share=0.2,
            holdout_tokens=10,
            min_holdout_tokens=11,
        )


def test_finite_weights_do_not_overflow_when_normalized(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", ["alpha"] * 4)
    b = _jsonl(tmp_path / "b.jsonl", ["bravo"] * 4)
    cal, held, report = build_mix([(a, 1e308), (b, 1e308)], 20, count=len)
    assert cal == "alpha\n\nbravo\n\nalpha\n\nbravo"
    assert held == ""
    assert all(source["token_share"] == 0.5 for source in report["sources"].values())


def test_no_holdout_preserves_order_repetitions_and_positional_api(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", ["aaaa", "bbbb", "aaaa", "cccc"])
    b = _jsonl(tmp_path / "b.jsonl", ["dddd", "eeee", "dddd", "ffff"])
    # The five existing positional parameters remain valid; duplicates are retained.
    cal, held, report = build_mix([(a, 2), (b, 1)], 12, len, 0.0, 200000)
    assert cal == "aaaa\n\ndddd\n\nbbbb"
    assert held == ""
    assert report["tokens"] == len(cal)
    assert report["document_tokens"] == 12
    assert report["calibration_documents"] == 3
    assert report["held_out_documents"] == report["held_out_tokens"] == 0
    assert report["reserved_holdout_documents"] == 0
    assert report["split_method"] == "none"
    assert "warning" not in report
    full, _, full_report = build_mix([(a, 1)], 100, len)
    assert full == "aaaa\n\nbbbb\n\naaaa\n\ncccc"
    assert full_report["duplicate_documents"] == 1
    single = _jsonl(tmp_path / "one.jsonl", ["one document"])
    assert build_mix([(single, 1)], 1)[0:2] == ("one document", "")


def test_final_token_counts_include_separators_and_minimum_uses_them(tmp_path):
    path = _jsonl(tmp_path / "four.jsonl", ["aa", "bb", "cc", "dd"])
    cal, held, report = build_mix(
        [(path, 1)],
        4,
        count=len,
        holdout_share=0.5,
        holdout_tokens=6,
        min_holdout_tokens=6,
    )
    assert report["tokens"] == len(cal) == 6
    assert report["held_out_tokens"] == len(held) == 6
    assert report["document_tokens"] == report["held_out_document_tokens"] == 4
    assert report["sources"][str(path)]["tokens"] == 4
    assert "warning" not in report


@pytest.mark.parametrize("invalid_count", [-1, 1.5, float("nan"), float("inf"), True])
def test_invalid_token_count_results_are_rejected(tmp_path, invalid_count):
    path = _jsonl(tmp_path / "input.jsonl", ["a document"])
    with pytest.raises(ValueError, match="nonnegative integer token count"):
        build_mix([(path, 1)], 10, count=lambda text: invalid_count)


def test_requested_holdout_with_zero_token_content_fails(tmp_path):
    path = _jsonl(tmp_path / "input.jsonl", ["aa", "bb"])
    _, held, _ = build_mix(
        [(path, 1)], 2, count=len, holdout_share=0.5, holdout_tokens=2
    )

    def count(text):
        return 0 if text == held else len(text)

    with pytest.raises(ValueError, match="holdout has 1 documents and 0 tokens"):
        build_mix([(path, 1)], 2, count=count, holdout_share=0.5, holdout_tokens=2)


def _render_chat(messages):
    return "".join(
        f"<{message['role']}>{message['content']}</{message['role']}>"
        for message in messages
    )


def test_template_preparation_is_explicit_and_does_not_reformat_text(tmp_path):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    path = tmp_path / "mixed.jsonl"
    plain = "  keep this text verbatim\nincluding its spacing  "
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"text": plain, "messages": messages},
                {"messages": messages},
                {
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "world"},
                    ]
                },
            )
        )
    )
    default = read_documents(path)
    assert default == [
        plain,
        "user: hello\nassistant: world",
        "human: hello\ngpt: world",
    ]
    rendered = _render_chat(messages)
    assert read_documents(path, render_chat=_render_chat) == [plain, rendered, rendered]
    text, held, report = build_mix(
        [(path, 1)],
        10000,
        count=len,
        render_chat=_render_chat,
        tokenizer_id="test/character-tokenizer",
        chat_template_id="test/role-tags-v1",
    )
    assert text == "\n\n".join([plain, rendered, rendered])
    assert held == ""
    assert report["tokens"] == len(text)
    assert report["preparation"] == {
        "chat_format": "chat-template",
        "token_count": "tokenizer",
        "tokenizer_id": "test/character-tokenizer",
        "chat_template_id": "test/role-tags-v1",
    }
    assert report["sources"][str(path)]["documents"] == 3
    assert report["unique_documents"] == 2
    assert report["duplicate_documents"] == 1
    txt = tmp_path / "plain.txt"
    txt.write_text(plain)
    assert read_documents(txt, render_chat=_render_chat) == [plain]


def test_rendered_content_identity_protects_chat_and_plain_text_copies(tmp_path):
    messages = [[{"role": "user", "content": f"question {i}"}] for i in range(6)]
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps([{"messages": turns} for turns in messages]))
    rendered = [_render_chat(turns) for turns in messages]
    plain = _jsonl(tmp_path / "text.jsonl", reversed(rendered))
    kwargs = {
        "tokens": 10000,
        "count": len,
        "holdout_share": 0.5,
        "holdout_tokens": 10,
        "render_chat": _render_chat,
        "tokenizer_id": "test/character-tokenizer",
        "chat_template_id": "test/role-tags-v1",
    }
    result = build_mix([(chat, 1), (plain, 1)], **kwargs)
    cal, held, report = result
    assert not set(cal.split("\n\n")) & set(held.split("\n\n"))
    assert report["tokens"] == len(cal)
    assert report["held_out_tokens"] == len(held)
    assert report["unique_documents"] == 6
    assert (
        report["duplicate_documents"] == report["cross_source_duplicate_documents"] == 6
    )
    assert build_mix([(chat, 1), (plain, 1)], **kwargs) == result
    counted_only = build_mix(
        [(plain, 1)], 10000, count=len, tokenizer_id="test/character-tokenizer"
    )
    assert counted_only[0] == "\n\n".join(reversed(rendered))
    assert counted_only[2]["preparation"]["chat_format"] == "role-content"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokenizer_id": "tokenizer"}, "requires a token-counting callback"),
        ({"chat_template_id": "template"}, "requires render_chat"),
        ({"render_chat": _render_chat}, "tokenizer_id and chat_template_id"),
        (
            {"render_chat": _render_chat, "count": len, "tokenizer_id": "tokenizer"},
            "tokenizer_id and chat_template_id",
        ),
        ({"count": len, "tokenizer_id": ""}, "tokenizer_id must be a nonempty"),
    ],
)
def test_preparation_requires_attributable_identities(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_mix([(tmp_path / "not-read.jsonl", 1)], 100, **kwargs)


@pytest.mark.parametrize("rendered", ["", "  ", None, ["token", "ids"]])
def test_chat_renderer_must_return_usable_text(tmp_path, rendered):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps([{"messages": [{"role": "user", "content": "hello"}]}]))
    with pytest.raises(ValueError, match="render_chat must return nonempty text"):
        read_documents(path, render_chat=lambda messages: rendered)


def test_exhausted_sources_report_realized_not_requested_shares(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", ["aaaa", "aaaa"])
    b = _jsonl(tmp_path / "b.jsonl", ["bbbb"])
    cal, _, report = build_mix([(a, 1), (b, 3)], 100, count=len)
    assert report["tokens"] == len(cal)
    assert report["calibration_documents"] == 3
    assert report["duplicate_documents"] == 1
    assert report["sources"][str(a)]["requested_share"] == 0.25
    assert report["sources"][str(b)]["requested_share"] == 0.75
    assert report["sources"][str(a)]["token_share"] == pytest.approx(2 / 3)
    assert report["sources"][str(b)]["token_share"] == pytest.approx(1 / 3)
    assert "fewer than" in report["warning"]


def test_template_preparation_rejects_nontext_chat_content(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps([{"messages": [{"role": "user", "content": [{"type": "image"}]}]}])
    )
    with pytest.raises(ValueError, match="requires text-only messages"):
        read_documents(path, render_chat=_render_chat)
