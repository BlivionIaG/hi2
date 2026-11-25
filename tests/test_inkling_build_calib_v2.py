# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_build_calib_v2.py``: the windows of every category, the provenance
each window records, the document split and the dedupe, and the gates the manifest writes.

CPU only and tiny: a fake tokenizer whose ids are a function of the text, so no checkpoint,
no dataset and no GPU are needed. ``LONG_LENGTHS`` is patched to a few hundred tokens so the
long-context path runs in a test."""

import importlib.util
import json
import random
import shutil
import string
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

CATS = (
    "agentic",
    "code",
    "multilingual",
    "long_context",
    "math_reasoning",
    "general_chat",
)
LONG = (256, 512, 768)


def _import(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def b():
    return _import(
        "hi2_test_build_calib_v2",
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_build_calib_v2.py",
    )


class CharTok:
    """One token per character: the tests can write the windows by hand."""

    name_or_path = "char-test-tokenizer"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + ord(c) % 250 for c in text]}


class BuildTok:
    """``short`` pseudo-random tokens per text, ``long`` for a text that starts with
    ``LONG:``; the stream is seeded by the text, so different documents differ (and share
    almost no 32-token shingle)."""

    name_or_path = "build-test-tokenizer"

    def __init__(self, short: int = 2048, long: int = 800):
        self.short, self.long = short, long

    def __call__(self, text, add_special_tokens=False):
        n = self.long if text.startswith("LONG:") else self.short
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return {"input_ids": rng.integers(0, 250, n).tolist()}


DEFAULT_DOCS = {
    "agentic": 300,
    "multilingual": 40,
    "long_context": 40,
    "math_reasoning": 40,
    "general_chat": 40,
}
CODE_DOCS = {"python": 40, "shell": 80, "sql": 80, "javascript": 80, "typescript": 80}


def _stage(out: Path, docs: dict[str, int] | None = None) -> None:
    """A staged directory: 300 agentic documents (60 held-out windows), 80 documents of
    every small code language, and 40 of the rest. A key like ``code__shell`` overrides
    the code language's count."""
    docs = docs or {}
    src = out / "src"
    src.mkdir(parents=True)
    rng = random.Random(0)

    def line(cat: str, lang: str, i: int) -> str:
        body = "".join(rng.choice(string.ascii_letters) for _ in range(200))
        text = f"LONG:{i}:{body}" if cat == "long_context" else f"{cat}:{i}:{body}"
        return json.dumps(
            {
                "text": text,
                "category": cat,
                "language": lang,
                "source": f"test/{cat}/{lang}",
                "license": "apache-2.0",
            }
        )

    for cat in ("agentic", "math_reasoning", "general_chat", "long_context"):
        n = docs.get(cat, DEFAULT_DOCS[cat])
        (src / f"{cat}.jsonl").write_text(
            "\n".join(line(cat, "en", i) for i in range(n)) + "\n"
        )
    n = docs.get("multilingual", DEFAULT_DOCS["multilingual"])
    (src / "multilingual.jsonl").write_text(
        "\n".join(line("multilingual", "zh" if i % 2 else "en", i) for i in range(n)) + "\n"
    )
    lines = []
    for lang, default in CODE_DOCS.items():
        n = docs.get(f"code__{lang}", default)
        lines.extend(line("code", lang, i) for i in range(n))
    (src / "code.jsonl").write_text("\n".join(lines) + "\n")
    (src / "staging.json").write_text(
        json.dumps({"seed": 0, "sources": {c: {} for c in CATS}})
    )


@pytest.fixture(scope="module")
def built(b, tmp_path_factory):
    """One good build with short long-context lengths, shared by the tests that read it."""
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path_factory.mktemp("calib_v2_build") / "calib_v2"
        _stage(out)
        manifest = b.build(out, BuildTok(), seed=0)
        return out, manifest
    finally:
        b.LONG_LENGTHS = old


# ------------------------------------------------------------------------------- windows


def test_a_short_window_is_a_run_of_documents_and_names_them(b):
    recs = [
        {
            "text": "a" * 120,
            "category": "general_chat",
            "language": "en",
            "source": "s1",
            "license": "apache-2.0",
        },
        {
            "text": "b" * 120,
            "category": "general_chat",
            "language": "en",
            "source": "s2",
            "license": "mit",
        },
    ]
    tok = CharTok()
    out, meta = b._windowize(recs, tok, {"general_chat": (150,)})
    (key,) = out
    assert key == "general_chat__en__150" and len(out[key]) == 1
    a = b.encode(tok, "\n\n" + recs[0]["text"])
    c = b.encode(tok, "\n\n" + recs[1]["text"])
    assert np.array_equal(out[key][0], np.concatenate([a, c[: 150 - len(a)]]))
    row = meta[key][0]
    assert row["category"] == "general_chat" and row["language"] == "en"
    assert row["length"] == 150
    assert row["documents"] == [
        b.sha256_text(recs[0]["text"]),
        b.sha256_text(recs[1]["text"]),
    ]
    assert row["document_records"] == [
        {"id": b.sha256_text(recs[0]["text"]), "source": "s1", "license": "apache-2.0"},
        {"id": b.sha256_text(recs[1]["text"]), "source": "s2", "license": "mit"},
    ]
    assert row["sources"] == ["s1", "s2"] and row["licenses"] == ["apache-2.0", "mit"]
    assert row["license_classes"] == ["permissive"]


def test_a_long_window_is_one_contiguous_slice_of_one_document(b):
    recs = [
        {
            "text": "a" * 300,
            "category": "long_context",
            "language": "en",
            "source": "s1",
            "license": "apache-2.0",
        },
        {
            "text": "b" * 300,
            "category": "long_context",
            "language": "en",
            "source": "s2",
            "license": "mit",
        },
    ]
    tok = CharTok()
    out, meta = b._windowize(recs, tok, {"long_context": (100,)})
    key = "long_context__en__100"
    assert len(out[key]) == 2
    assert np.array_equal(out[key][0], b.encode(tok, "a" * 300)[:100])
    assert np.array_equal(out[key][1], b.encode(tok, "b" * 300)[:100])
    assert meta[key][0]["documents"] == [b.sha256_text("a" * 300)]
    assert meta[key][1]["sources"] == ["s2"]


# ---------------------------------------------------------------- split and dedupe


def test_the_split_is_disjoint_and_keeps_sixteen_held_out_documents(b):
    recs = [
        {
            "text": f"{cat} document {i}",
            "category": cat,
            "language": lang,
            "source": "t",
            "license": "t",
        }
        for cat in ("code", "general_chat")
        for lang in ("en", "python")
        for i in range(40)
    ]
    cal, held = b._split(recs, 0, 0.2)
    for cat in ("code", "general_chat"):
        for lang in ("en", "python"):
            h = {
                b.sha256_text(r["text"])
                for r in held
                if r["category"] == cat and r["language"] == lang
            }
            c = {
                b.sha256_text(r["text"])
                for r in cal
                if r["category"] == cat and r["language"] == lang
            }
            assert len(h) == 16  # the share alone would give 8
            assert len(c) == 24
            assert not h & c
            assert h | c == {
                b.sha256_text(r["text"])
                for r in recs
                if r["category"] == cat and r["language"] == lang
            }


def test_dedupe_drops_exact_and_near_duplicates_within_a_category(b):
    base = "the same document " * 40
    near = base[:-10] + "different tail"
    recs = [
        {"text": base, "category": "code", "language": "python"},
        {"text": base, "category": "code", "language": "python"},
        {"text": near, "category": "code", "language": "python"},
        {"text": base, "category": "general_chat", "language": "en"},
    ]
    kept, report = b._dedupe(recs)
    assert report["exact_dropped"] == 2  # the second copy and the other category's copy
    assert report["near_dropped"] == 1  # the near duplicate of the kept text
    assert [r["category"] for r in kept] == ["code"]


# ------------------------------------------------------------------------------ build


def test_the_manifest_holds_every_gate_and_the_provenance(b, built):
    out, m = built
    assert m["ok"] is True
    assert b.verify(out, BuildTok(), production=True) == []
    assert m["schema"] == "hi2.inkling-calib-v2/1"
    for cat in CATS:
        assert m["gates"][f"heldout_windows:{cat}"]["ok"]
        assert m["gates"][f"calib_windows:{cat}"]["ok"]
    assert m["gates"]["heldout_windows:agentic_floor"]["ok"]
    assert m["gates"]["calib_windows:agentic_floor"]["ok"]
    for lang in ("shell", "sql", "typescript"):
        assert m["gates"][f"heldout_windows:code__{lang}__2048"]["ok"]
        assert m["gates"][f"calib_windows:code__{lang}__2048"]["ok"]
    for length in LONG:
        assert m["gates"][f"heldout_windows:long_context__en__{length}"]["ok"]
    for split, rows in m["meta"].items():
        assert rows
        for key, per_window in rows.items():
            assert len(per_window) == m["splits"][split][key]["windows"]
            for row in per_window:
                assert key == f"{row['category']}__{row['language']}__{row['length']}"
                assert row["documents"] and row["sources"] and row["licenses"]
                assert len(row["document_records"]) == len(row["documents"])
                for doc in row["document_records"]:
                    assert doc["id"] in row["documents"]
                    assert doc["source"] and doc["license"]
                assert set(row["license_classes"]) == {"permissive"}
    for info in m["files"].values():
        assert info["category"] in CATS
        assert info["language"] in ("en", "zh", "python", "shell", "sql", "javascript", "typescript", "rust")


def test_verify_refuses_a_tampered_window_file(b, built, tmp_path):
    out, _ = built
    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    name = next(iter(json.loads((copy / "manifest.json").read_text())["files"]))
    path = copy / name
    a = np.load(path)
    a[0, 0] = (int(a[0, 0]) + 1) % 250
    np.save(path, a)
    bad = b.verify(copy, None, production=True)
    assert any("sha256" in x or "size" in x for x in bad), bad


def test_the_long_context_gate_is_per_length(b, tmp_path):
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path / "short_long"
        _stage(out)
        m = b.build(out, BuildTok(long=400), seed=0)
    finally:
        b.LONG_LENGTHS = old
    # 400 tokens fill the 256 window once per document, and never reach 512 or 768
    assert m["ok"] is False
    assert m["gates"]["heldout_windows:long_context"]["ok"] is True  # 16 x 256
    assert m["gates"]["heldout_windows:long_context__en__512"]["ok"] is False
    assert m["gates"]["heldout_windows:long_context__en__768"]["ok"] is False
    assert "the build recorded a failed gate" in b.verify(out, None, production=True)


def test_a_small_category_cannot_pass_on_other_categories_windows(b, tmp_path):
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path / "small_math"
        _stage(out, {"math_reasoning": 5})
        m = b.build(out, BuildTok(), seed=0)
    finally:
        b.LONG_LENGTHS = old
    assert m["ok"] is False
    assert m["gates"]["heldout_windows:math_reasoning"]["ok"] is False
    assert m["gates"]["calib_windows:math_reasoning"]["ok"] is False
    assert m["gates"]["heldout_windows:agentic"]["ok"] is True


def test_a_long_document_does_not_exceed_the_window_cap(b):
    recs = [
        {
            "text": "x" * 20_000,
            "category": "general_chat",
            "language": "en",
            "source": "s",
            "license": "apache-2.0",
        }
    ]
    out, _ = b._windowize(recs, CharTok(), {"general_chat": (60,)})
    assert len(out["general_chat__en__60"]) == 32, len(out["general_chat__en__60"])


def test_dedupe_uses_dedupe_text_when_present(b):
    shared = "system prompt " * 400
    recs = [
        {
            "text": shared + f"task {i}",
            "dedupe_text": f"task body {i} " * 20,
            "category": "agentic",
            "language": "en",
            "source": "agentic_bench/x",
            "license": "apache-2.0",
        }
        for i in range(3)
    ]
    kept, report = b._dedupe(recs)
    assert len(kept) == 3, report
    assert report["near_dropped"] == 0, report


def test_sql_public_records_read_the_cached_file(b, tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    rows = [
        {
            "question": f"question {i}",
            "context": "CREATE TABLE t" + " " * 20 + f" (id INT, name TEXT) /* {i} */",
            "answer": "SELECT id FROM t",
        }
        for i in range(500)
    ]
    (cache / "sql_create_context_v4.json").write_text(json.dumps(rows))
    recs = b._sql_public_records(0, cache, log=lambda *a: None, docs=2)
    assert len(recs) == 2, len(recs)
    for r in recs:
        assert r["language"] == "sql"
        assert r["source"] == "b-mc2/sql-create-context"
        assert r["license"] == "cc-by-4.0"
        assert len(r["text"]) >= 16_000


def test_ood_falls_back_to_local_docs(b, built, tmp_path, monkeypatch):
    import datasets

    def offline(*a, **k):
        raise ConnectionError("OfflineModeIsEnabled")

    monkeypatch.setattr(datasets, "load_dataset", offline)
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(16):
        (docs / f"{i}.md").write_text("# heading\n" + "text " * 1200)
    out, _ = built
    m = b.ood(out, BuildTok(), docs_roots=(docs,), log=lambda *a: None)
    assert m["splits"]["ood"]["docs__en__2048"]["windows"] == 16
    assert m["ood"]["licenses"] == ["apache-2.0"]
    for row in m["meta"]["ood"]["docs__en__2048"]:
        assert row["document_records"]


def test_the_split_balances_small_and_large_documents(b):
    sizes = [10_000] * 14 + [10] * 16
    recs = [
        {
            "text": "x" * n,
            "category": "code",
            "language": "sql",
            "source": "t",
            "license": "apache-2.0",
        }
        for n in sizes
    ]
    cal, held = b._split(recs, 0)
    held_chars = sum(len(r["text"]) for r in held)
    cal_chars = sum(len(r["text"]) for r in cal)
    assert len(held) == 16 and len(cal) == 14
    assert min(held_chars, cal_chars) >= 0.5 * max(held_chars, cal_chars)
    assert sum(len(r["text"]) >= 5_000 for r in held) >= 5
    assert sum(len(r["text"]) >= 5_000 for r in cal) >= 5


def test_a_unicode_line_separator_does_not_split_a_record(b, tmp_path):
    path = tmp_path / "u2028.jsonl"
    path.write_text(json.dumps({"text": "a\u2028b\u2029c"}, ensure_ascii=False) + "\n")
    assert b._read_jsonl(path) == [{"text": "a\u2028b\u2029c"}]


def test_an_unknown_licence_is_refused(b):
    with pytest.raises(SystemExit, match="unknown licence"):
        b.validate_licenses(
            [{"text": "x", "category": "code", "language": "python", "source": "s", "license": "proprietary"}]
        )


def test_the_agentic_floor_is_sixty(b, tmp_path):
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path / "agentic_59"
        _stage(out, {"agentic": 250})
        m = b.build(out, BuildTok(), seed=0)
    finally:
        b.LONG_LENGTHS = old
    assert m["ok"] is False
    assert m["gates"]["heldout_windows:agentic_floor"]["ok"] is False
    assert "50 >= 60" in m["gates"]["heldout_windows:agentic_floor"]["detail"]


def test_the_small_code_floor_is_sixteen(b, tmp_path):
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path / "shell_12"
        _stage(out, {"code__shell": 20})
        m = b.build(out, BuildTok(), seed=0)
    finally:
        b.LONG_LENGTHS = old
    assert m["ok"] is False
    assert m["gates"]["heldout_windows:code__shell__2048"]["ok"] is False
    assert m["gates"]["calib_windows:code__shell__2048"]["ok"] is False


def test_exclude_license_drops_gpl2(b, tmp_path):
    old = b.LONG_LENGTHS
    b.LONG_LENGTHS = LONG
    try:
        out = tmp_path / "gpl2"
        _stage(out)
        code = out / "src/code.jsonl"
        lines = code.read_text().splitlines()
        rng = random.Random(2)
        for i in range(160):
            body = "".join(rng.choice(string.ascii_letters) for _ in range(200))
            gpl = i % 2 == 0
            lines.append(
                json.dumps(
                    {
                        "text": f"rust:{i}:{body}",
                        "category": "code",
                        "language": "rust",
                        "source": "linux" if gpl else "crate",
                        "license": "gpl-2.0" if gpl else "mit",
                    }
                )
            )
        code.write_text("\n".join(lines) + "\n")
        m = b.build(out, BuildTok(), seed=0, exclude_license=("gpl2",))
    finally:
        b.LONG_LENGTHS = old
    assert m["ok"] is True
    assert m["excluded_license_classes"] == ["gpl2"]
    assert m["excluded_records"] == 80
    for split_rows in m["meta"].values():
        for per_window in split_rows.values():
            for row in per_window:
                assert "gpl2" not in row["license_classes"]
    assert b.verify(out, BuildTok(), production=True) == []
