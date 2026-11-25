# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Mutants of the v2 calibration and evaluation logic: the tests must kill each of them.

Each mutation is applied to a temporary copy of the module, the copy is imported and the
check that the tests make is run against it. CPU only, no checkpoint, no dataset:

    python experiments/inkling_calib_v2_mutants.py

Exit code 0 only when every mutant is killed (its check fails on the mutant and holds on
the original). A surviving mutant is a hole in the tests.
"""

import importlib.util
import json
import random
import string
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mutated(name: str, source: Path, old: str, new: str, tmp: Path):
    text = source.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{name}: the mutation text occurs {text.count(old)} times")
    copy = tmp / f"{name}_{source.name}"
    copy.write_text(text.replace(old, new))
    return _load(f"mut_{name}", copy)


# --------------------------------------------------------------------------- checks


def _check_split(b):
    recs = [
        {
            "text": f"document {i}",
            "category": "code",
            "language": "python",
            "source": "t",
            "license": "t",
        }
        for i in range(40)
    ]
    _, held = b._split(recs, 0, 0.2)
    assert len(held) == 16, f"the held-out split kept {len(held)} documents, not 16"


def _check_dedupe(b):
    base = "the same document " * 40
    near = base[:-10] + "different tail"
    recs = [
        {"text": base, "category": "code", "language": "python"},
        {"text": near, "category": "code", "language": "python"},
    ]
    kept, report = b._dedupe(recs)
    assert report["near_dropped"] == 1, f"near_dropped={report['near_dropped']}"
    assert len(kept) == 1


class _CharTok:
    name_or_path = "char"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + ord(c) % 250 for c in text]}


def _check_window_key(b):
    recs = [
        {
            "text": "a" * 120,
            "category": "general_chat",
            "language": "en",
            "source": "s",
            "license": "apache-2.0",
        }
    ]
    out, _ = b._windowize(recs, _CharTok(), {"general_chat": (60,)})
    assert list(out) == ["general_chat__en__60"], list(out)


class _Tok2048:
    name_or_path = "mut"

    def __call__(self, text, add_special_tokens=False):
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return {"input_ids": rng.integers(0, 250, 2048).tolist()}


def _stage_small(root: Path, groups: dict[str, list[tuple[str, str, str, int]]]) -> None:
    """``groups``: category -> [(language, source, licence, documents)]."""
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    for cat, rows in groups.items():
        lines = []
        for lang, source, lic, n in rows:
            for i in range(n):
                body = "".join(random.choice(string.ascii_letters) for _ in range(200))
                lines.append(
                    json.dumps(
                        {
                            "text": f"{cat}:{lang}:{source}:{i}:{body}",
                            "category": cat,
                            "language": lang,
                            "source": source,
                            "license": lic,
                        }
                    )
                )
        (src / f"{cat}.jsonl").write_text("\n".join(lines) + "\n")
    (src / "staging.json").write_text(
        json.dumps({"seed": 0, "sources": {c: {} for c in groups}})
    )


def _check_exclude(b):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "exclude"
        _stage_small(
            root,
            {"code": [("rust", "linux", "gpl-2.0", 80), ("rust", "crate", "mit", 80)]},
        )
        m = b.build(root, _Tok2048(), exclude_license=("gpl2",))
    assert m["excluded_records"] == 80, m["excluded_records"]
    for per_window in m["meta"]["calib"].values():
        for row in per_window:
            assert "gpl2" not in row["license_classes"], row["license_classes"]


def _check_unknown_licence(b):
    try:
        b.classify_license("proprietary")
    except SystemExit as e:
        assert "unknown licence" in str(e)
    else:
        raise AssertionError("an unknown licence was accepted")


def _check_document_id(b):
    recs = [
        {
            "text": "a" * 120,
            "category": "general_chat",
            "language": "en",
            "source": "s",
            "license": "apache-2.0",
        }
    ]
    _, meta = b._windowize(recs, _CharTok(), {"general_chat": (60,)})
    row = meta["general_chat__en__60"][0]
    assert row["document_records"][0]["id"] == b.sha256_text("a" * 120), row["document_records"]


def _check_agentic_floor(b):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "agentic"
        _stage_small(root, {"agentic": [("en", "bench", "apache-2.0", 250)]})
        m = b.build(root, _Tok2048())
    assert m["gates"]["heldout_windows:agentic_floor"]["ok"] is False, m["gates"]["heldout_windows:agentic_floor"]


def _check_small_code_floor(b):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "shell"
        _stage_small(root, {"code": [("shell", "repo", "apache-2.0", 20)]})
        m = b.build(root, _Tok2048())
    assert m["gates"]["heldout_windows:code__shell__2048"]["ok"] is False, m["gates"]["heldout_windows:code__shell__2048"]


def _check_bucket(kl):
    teacher = nn.Linear(2, 3, bias=False)
    student = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        teacher.weight.zero_()
        student.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    ht = torch.zeros(2, 6, 2)
    hs = torch.zeros(2, 6, 2)
    hs[:, :2, 0] = 1.0
    hs[:, 2:, 1] = 1.0
    out = kl.kl_position_buckets(
        teacher,
        hs,
        ht,
        kl.parse_position_buckets("0-2,2-4,4-"),
        chunk=3,
        lm_student=student,
    )
    assert out["0-2"]["positions"] == 4, out
    assert out["2-4"]["positions"] == 4, out
    assert out["4+"]["positions"] == 4, out
    assert out["2-4"]["kl_mean"] > 0, out


MUTANTS = [
    (
        "split_keeps_the_share",
        "inkling_build_calib_v2.py",
        "max(MIN_HELDOUT_DOCS, int(len(order) * holdout_share))",
        "int(len(order) * holdout_share)",
        _check_split,
        "build",
    ),
    (
        "dedupe_trusts_every_record",
        "inkling_build_calib_v2.py",
        'keep = near_unique([r.get("dedupe_text") or r["text"] for r in recs])',
        "keep = list(range(len(recs)))",
        _check_dedupe,
        "build",
    ),
    (
        "window_key_uses_one_separator",
        "inkling_build_calib_v2.py",
        'key = f"{cat}__{lang}__{length}"',
        'key = f"{cat}_{lang}_{length}"',
        _check_window_key,
        "build",
    ),
    (
        "every_position_in_the_first_bucket",
        "inkling_kl_eval.py",
        "index[lo:hi] = bi",
        "index[lo:hi] = 0",
        _check_bucket,
        "kl",
    ),
    (
        "unknown_license_accepted",
        "inkling_build_calib_v2.py",
        "if not parts or unknown:",
        "if not parts:",
        _check_unknown_licence,
        "build",
    ),
    (
        "exclude_license_keeps_every_record",
        "inkling_build_calib_v2.py",
        'kept = [r for r in records if r["license_class"] not in excluded]',
        "kept = list(records)",
        _check_exclude,
        "build",
    ),
    (
        "document_id_is_the_source",
        "inkling_build_calib_v2.py",
        '{"id": s[4], "source": s[1], "license": s[2]}',
        '{"id": s[1], "source": s[1], "license": s[2]}',
        _check_document_id,
        "build",
    ),
    (
        "agentic_floor_is_sixteen",
        "inkling_build_calib_v2.py",
        "AGENTIC_HELDOUT_WINDOWS = 60",
        "AGENTIC_HELDOUT_WINDOWS = 16",
        _check_agentic_floor,
        "build",
    ),
    (
        "small_code_floor_is_eight",
        "inkling_build_calib_v2.py",
        "SMALL_CODE_WINDOWS = 16",
        "SMALL_CODE_WINDOWS = 8",
        _check_small_code_floor,
        "build",
    ),
]


def main() -> int:
    build = _load("calib_v2_original_build", HERE / "inkling_build_calib_v2.py")
    kl = _load("calib_v2_original_kl", HERE / "inkling_kl_eval.py")
    originals = {"build": build, "kl": kl}
    survivors = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, source, old, new, check, module in MUTANTS:
            original = originals[module]
            check(original)  # the check holds on the code as it is
            mutant = _mutated(name, HERE / source, old, new, tmp)
            try:
                check(mutant)
            except AssertionError as e:
                print(f"killed   {name}: {e}")
            else:
                print(f"SURVIVED {name}")
                survivors.append(name)
    if survivors:
        print(json.dumps({"ok": False, "survivors": survivors}))
        return 1
    print(json.dumps({"ok": True, "mutants": len(MUTANTS)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
