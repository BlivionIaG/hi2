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
import sys
import tempfile
from pathlib import Path

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
            "license": "l",
        }
    ]
    out, _ = b._windowize(recs, _CharTok(), {"general_chat": (60,)})
    assert list(out) == ["general_chat__en__60"], list(out)


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
        'keep = near_unique([r["text"] for r in recs])',
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
