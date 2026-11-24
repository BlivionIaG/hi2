# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""``experiments/inkling_build_calib.py`` (CPU): the calibration and held-out sets of the
Inkling-2-Mini quantization. A toy tokenizer stands in for the real one; every gate is
shown to fail on the input it exists for, and ``verify`` on a tampered set."""

import importlib.util
import json
import random
import re
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ADDED = {
    "<|message_user|>": 9001,
    "<|message_model|>": 9002,
    "<|message_system|>": 9003,
    "<|message_tool|>": 9004,
    "<|content_text|>": 9005,
    "<|content_thinking|>": 9006,
    "<|content_xml|>": 9007,
    "<|end_message|>": 9008,
}


class ToyTok:
    """Words and ``<|...|>`` markers are the tokens; a word's id is its crc32."""

    def __len__(self):
        return 10_000

    def get_added_vocab(self):
        return dict(ADDED)

    def _ids(self, text):
        found = re.findall(r"<\|[a-z_]+\|>|\w+|[^\w\s]", text)
        return [ADDED.get(t) or 100 + zlib.crc32(t.encode()) % 8000 for t in found]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self._ids(text)
        if return_tensors == "pt":
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))
        return {"input_ids": ids}

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        out = ""
        if tools:
            out += f"<|message_system|>tool_declare<|content_xml|>{json.dumps(tools)}<|end_message|>"
        for m in messages:
            role = {"user": "user", "assistant": "model"}.get(m["role"], m["role"])
            if m.get("reasoning_content"):
                out += f"<|message_{role}|><|content_thinking|>{m['reasoning_content']}<|end_message|>"
            out += f"<|message_{role}|><|content_text|>{m['content']}<|end_message|>"
        return out


def _import():
    name = "hi2_test_inkling_build_calib"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[1] / "experiments" / "inkling_build_calib.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod(monkeypatch):
    m = _import()
    for name, value in {
        "SEQ_LEN": 64,
        "MIN_CALIB_TOKENS": 2048,
        "PREFERRED_CALIB_TOKENS": 4096,
        "MIN_HELDOUT_WINDOWS": 4,
        "PREFERRED_HELDOUT_WINDOWS": 8,
        "PIPELINE_WINDOWS": 8,
        "SHINGLE_LIMIT": 0.5,  # scale_down relaxes it: the fixture restores it
    }.items():
        monkeypatch.setattr(m, name, value)
    monkeypatch.setitem(m.SOURCES["gsm8k"], "file", "main/train.json")
    return m


def _words(rng, n):
    return " ".join(f"w{rng.randrange(3000)}" for _ in range(n))


def _fake_hub(root: Path, mod, rng, n=60):
    """The four cached datasets of ``SOURCES`` in the Hugging Face cache layout."""

    def put(repo, file, text):
        p = root / f"datasets--{repo.replace('/', '--')}" / "snapshots" / "s1" / file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    tools = lambda: [  # noqa: E731
        {
            "name": f"tool{k}",
            "description": _words(rng, 6),
            "parameters": {"type": "dict", "properties": {}},
        }
        for k in range(2)
    ]
    records = []
    for i in range(n):
        system = "You are an expert. " + _words(rng, 20)
        if i % 7:
            system += (
                f"\n{mod.TOOL_MARK}{json.dumps(tools())}\nShould you decide. "
                + _words(rng, 8)
            )
        records.append(
            {
                "system": system,
                "conversations": [
                    {"from": "user", "value": _words(rng, 40)},
                    {"from": "assistant", "value": '[tool0(q="x")]'},
                    {"from": "tool", "value": json.dumps({"r": _words(rng, 10)})},
                    {"from": "assistant", "value": _words(rng, 40)},
                ],
            }
        )
    put("Team-ACE/ToolACE", "data.json", json.dumps(records))
    put(
        "timdettmers/openassistant-guanaco",
        "openassistant_best_replies_train.jsonl",
        "\n".join(
            json.dumps(
                {
                    "text": f"### Human: {_words(rng, 30)}### Assistant: {_words(rng, 60)}"
                }
            )
            for _ in range(n)
        ),
    )
    put(
        "WizardLM/WizardLM_evol_instruct_70k",
        "alpaca_evol_instruct_70k.json",
        json.dumps(
            [
                {"instruction": _words(rng, 30), "output": _words(rng, 70)}
                for _ in range(n)
            ]
        ),
    )
    put(
        "openai/gsm8k",
        "main/train.json",
        json.dumps(
            [
                {
                    "question": _words(rng, 30),
                    "answer": f"{_words(rng, 25)}\n#### {rng.randrange(100)}",
                }
                for _ in range(n)
            ]
        ),
    )
    return root


def _docs(path: Path, rng, paragraphs=500):
    path.write_text("\n\n".join(_words(rng, 60) for _ in range(paragraphs)))
    return path


@pytest.fixture
def built(mod, tmp_path):
    """A staged and built toy set: ``(out, manifest, tok)``."""
    rng = random.Random(3)
    hub = _fake_hub(tmp_path / "hub", mod, rng)
    docs = _docs(tmp_path / "docs.txt", rng)
    out, tok = tmp_path / "calib", ToyTok()
    out.mkdir()
    mod.stage(out, tok, hub, docs, seed=0, log=lambda *_: None)
    manifest = mod.build(
        out,
        tok,
        tokens=12_000,
        holdout_tokens=4_000,
        holdout_share=0.25,
        tokenizer_dir="toy",
        log=lambda *_: None,
    )
    return out, manifest, tok


# ----------------------------------------------------------------------- the pieces


def test_guanaco_turns_alternate_or_the_record_is_refused(mod):
    ok = mod.parse_guanaco("### Human: a?### Assistant: b.### Human: c?")
    assert [(m["role"], m["content"]) for m in ok] == [
        ("user", "a?"),
        ("assistant", "b."),
        ("user", "c?"),
    ]
    assert mod.parse_guanaco("### Assistant: first") is None
    assert mod.parse_guanaco("### Human: a### Human: b") is None
    assert mod.parse_guanaco("### Human: ### Assistant: b") is None
    assert mod.parse_guanaco("no markers at all") is None
    assert mod.parse_guanaco("stray text### Human: a### Assistant: b") is None


def test_gsm8k_solution_becomes_the_thinking_turn(mod):
    rec = {"question": "How many?", "answer": "2 + 2 = <<2+2=4>>4\n#### 4"}
    user, reply = mod.gsm8k_messages(rec)
    assert user == {"role": "user", "content": "How many?"}
    assert reply["reasoning_content"] == "2 + 2 = <<2+2=4>>4"
    assert reply["content"] == "The answer is 4."
    plain = mod.gsm8k_messages({"question": "q", "answer": "just text"})[1]
    assert plain == {"role": "assistant", "content": "just text"}


def test_toolace_tool_list_moves_to_the_tool_declaration(mod):
    seen = []
    spy = lambda messages, tools=None: seen.append((messages, tools)) or "x"  # noqa: E731
    tools = [{"name": "get_x", "description": "d", "parameters": {}}]
    system = f"Head.\n{mod.TOOL_MARK}{json.dumps(tools)}\nTail."
    rec = {"system": system, "conversations": [{"from": "user", "value": "hi"}]}
    assert mod.render_toolace(rec, spy) == ("x", True)
    messages, got = seen[-1]
    assert got == tools
    assert messages[0] == {"role": "system", "content": "Head.\n\nTail."}
    assert "get_x" not in messages[0]["content"] and messages[1]["role"] == "user"
    # no list, a list that is not JSON, one without names: the system prompt stays whole
    for bad in ("Head only.", f"H{mod.TOOL_MARK}[not json", f"H{mod.TOOL_MARK}[1, 2]"):
        rec = {"system": bad, "conversations": [{"from": "user", "value": "hi"}]}
        assert mod.render_toolace(rec, spy) == ("x", False)
        assert seen[-1][1] is None and seen[-1][0][0]["content"] == bad
    with pytest.raises(KeyError):  # an unknown role is refused, not guessed
        rec = {"system": "s", "conversations": [{"from": "robot", "value": "hi"}]}
        mod.render_toolace(rec, spy)


def test_near_unique_keeps_one_copy_of_a_shared_section(mod):
    rng = random.Random(0)
    shared = _words(rng, 120)
    a = _words(rng, 100) + " " + shared + " " + _words(rng, 100)  # carries the section
    b = shared  # is the section
    c = _words(rng, 150)  # unrelated
    d = shared + " " + _words(rng, 5)  # nearly only the section
    texts = [b, c, a, d, c]
    keep = mod.near_unique(texts, span=16, every=2)
    assert [texts[i] for i in keep] == [
        c,
        a,
    ]  # b and d are inside a, the second c repeats c
    assert keep == sorted(keep)
    assert mod.near_unique(texts, span=16, every=2) == keep  # deterministic
    assert mod.near_unique([c, a], span=16, every=2) == [
        0,
        1,
    ]  # distinct texts all stay


def test_shingle_overlap_counts_copied_text_and_nothing_else(mod):
    rng = np.random.default_rng(0)
    calib = rng.integers(0, 9000, 4000).astype(np.int32)
    fresh = rng.integers(0, 9000, 64).astype(np.int32)
    copied = calib[1000:1064].copy()
    half = np.concatenate([calib[500:532], fresh[:32]])
    frac = mod.overlap_fractions(np.stack([copied, fresh, half]), calib)
    assert frac[0] == 1.0 and frac[1] == 0.0
    assert frac[2] == pytest.approx(1 / 33, abs=1e-9)  # one 32-token shingle is whole
    # a window shorter than a shingle has none
    assert mod.overlap_fractions(calib[None, :20], calib)[0] == 0.0


def test_the_window_hash_is_order_sensitive_and_unaffected_by_other_windows(mod):
    a = np.arange(1, 65, dtype=np.int32)
    assert mod.sha256_array(a) == mod.sha256_array(a.astype(np.int64))
    assert mod.sha256_array(a) != mod.sha256_array(a[::-1])
    assert mod.windows(np.arange(200, dtype=np.int32)).shape == (3, 64)  # tail dropped
    assert (
        mod.shingle_hashes(a).shape == (33,) and len(set(mod.shingle_hashes(a))) == 33
    )


def test_the_shingle_hash_has_no_false_matches_between_unrelated_texts(mod):
    """Two unrelated random id streams share no 32-token run, so no hash may match: a
    hash linear in small multiples of the ids (1, 3, 5, ...) collides thousands of
    times at this size, which showed up as an overlap of a few percent on fresh text."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3000, 60_000).astype(np.int32)
    b = rng.integers(0, 3000, 60_000).astype(np.int32)
    ha, hb = mod.shingle_hashes(a), mod.shingle_hashes(b)
    assert not np.isin(ha, hb).any()
    assert len(np.unique(ha)) == len(ha)  # and none inside one stream either
    # a copied run of 200 ids matches exactly where a whole shingle lies inside it
    c = np.concatenate([b[:300], a[5000:5200], b[300:600]])
    hit = np.isin(mod.shingle_hashes(c), ha)
    assert hit.sum() == 200 - 32 + 1 and hit[300 : 300 + 169].all()


# ------------------------------------------------------------------------- the build


def test_the_toy_set_builds_verifies_and_matches_its_manifest(mod, built):
    out, m, tok = built
    assert m["ok"] and not m["warnings"], m["gates"]
    assert mod.verify(out, tok) == [] and mod.verify(out) == []
    ids = np.load(out / "calib_ids.npy")
    text_ids = mod.encode(tok, (out / "calib.txt").read_text())
    assert ids.shape == (len(text_ids) // 64, 64)
    assert np.array_equal(ids.reshape(-1), text_ids[: ids.size])  # the file, windowed
    c = m["calibration"]
    assert c["tokens"] == len(text_ids) and c["windows"] == len(ids)
    n = min(mod.PIPELINE_WINDOWS, len(ids))
    assert c["pipeline_windows"] == n
    assert c["ids_sha256_pipeline_windows"] == mod.sha256_array(ids[:n])
    held = np.load(out / "heldout_mixed_ids.npy")
    assert len(held) >= 4 and m["heldout"]["mixed"]["windows"] == len(held)
    # every source is in both splits and the shares follow the weights
    per = m["mix"]["per_source"]
    assert set(per) == {"toolace", "guanaco", "wizard", "gsm8k", "docs"}
    for name, rep in per.items():
        assert rep["calibration_documents"] > 0 and rep["heldout_documents"] > 0, name
        want = mod.SOURCES[name]["weight"]
        assert rep["calibration_token_share"] == pytest.approx(want, abs=0.06), name
    # every file the manifest lists is on disk with the recorded hash
    for name, info in m["files"].items():
        assert (out / name).stat().st_size == info["bytes"]
        assert mod.sha256_file(out / name) == info["sha256"]
    assert m["staging"]["sources"]["toolace"]["native_tool_declarations"] > 0
    assert (
        m["tokenizer"]["vocabulary"] == 10_000
        and m["tokenizer"]["special_token_ids"] == 8
    )
    # the pipeline's own reader returns the stored windows
    from hi2.pipeline import load_calibration_ids

    got = load_calibration_ids(out / "calib.txt", tok, 8, 64)
    assert np.array_equal(got.numpy(), ids[:8])


def test_verify_finds_a_tampered_set(mod, built):
    out, _, tok = built
    cal = out / "calib.txt"
    original = cal.read_text()
    cal.write_text(original + " extra")
    bad = mod.verify(out, tok)
    assert any("calib.txt" in b and "differ" in b for b in bad)
    cal.write_text(original)
    assert mod.verify(out, tok) == []
    ids = np.load(out / "calib_ids.npy")
    ids[0, 0] += 1
    np.save(out / "calib_ids.npy", ids)
    bad = mod.verify(out, tok)
    assert any("calib_ids.npy" in b for b in bad)
    (out / "manifest.json").unlink()
    assert mod.verify(out, tok) == [f"{out / 'manifest.json'}: missing"]


def test_verify_refuses_a_set_whose_build_failed_a_gate(mod, built):
    out, _, _ = built
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["ok"] = False
    m["gates"]["heldout_windows"]["ok"] = False
    path.write_text(json.dumps(m))
    bad = mod.verify(out)
    assert any("failed gate: heldout_windows" in b for b in bad)


def test_a_toy_manifest_is_not_a_production_manifest(mod, built):
    out, m, tok = built
    assert m["thresholds"]["production"] is False and m["thresholds"]["seq_len"] == 64
    assert mod.verify(out, tok) == []
    bad = mod.verify(out, tok, production=True)
    assert len(bad) == 1 and "not a production manifest" in bad[0]
    path = out / "manifest.json"
    old = json.loads(path.read_text())
    del old["thresholds"]  # a manifest from before the field existed
    path.write_text(json.dumps(old))
    assert any("production" in b for b in mod.verify(out, production=True))


def test_the_production_limits_are_the_contract():
    """No fixture: the module's own limits (a lowered floor fails here)."""
    m = _import()
    assert m.SEQ_LEN == 2048
    assert m.MIN_CALIB_TOKENS >= 262_144 and m.PREFERRED_CALIB_TOKENS >= 524_288
    assert m.MIN_HELDOUT_WINDOWS >= 32 and m.PREFERRED_HELDOUT_WINDOWS >= 64
    assert m.SHINGLE_LIMIT <= 0.5 and m.SHINGLE == 32
    assert m.thresholds()["production"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SEQ_LEN", 1024),
        ("MIN_CALIB_TOKENS", 262_143),
        ("MIN_HELDOUT_WINDOWS", 31),
        ("SHINGLE_LIMIT", 0.51),
    ],
)
def test_each_production_limit_decides_the_flag_on_its_own(monkeypatch, name, value):
    m = _import()
    assert m.thresholds()["production"] is True
    monkeypatch.setattr(m, name, value)
    assert m.thresholds()["production"] is False


def test_a_dry_run_scale_changes_the_limits_and_says_so(mod):
    mod.scale_down(32, 3, 2, 4)
    t = mod.thresholds()
    assert t == {
        "seq_len": 32,
        "min_calibration_tokens": 96,
        "preferred_calibration_tokens": 192,
        "min_heldout_windows": 2,
        "preferred_heldout_windows": 4,
        "pipeline_windows": 4,
        "shingle_limit": 1.01,
        "production": False,
    }


def test_verify_tokenizes_with_the_window_length_of_the_manifest(
    mod, built, monkeypatch
):
    out, _, tok = built
    monkeypatch.setattr(mod, "SEQ_LEN", 2048)  # the module's limit is not the set's
    assert mod.verify(out, tok) == []


def test_verify_ties_the_set_to_the_tokenizer_files_it_was_built_with(
    mod, built, tmp_path
):
    out, _, _ = built
    tdir = tmp_path / "tok"
    tdir.mkdir()
    # the toy build recorded no tokenizer hashes and this directory has no such files
    assert mod.verify(out, tokenizer_dir=tdir) == []
    (tdir / "tokenizer.json").write_text("{}")
    bad = mod.verify(out, tokenizer_dir=tdir)
    assert len(bad) == 1 and "tokenizer.json" in bad[0]
    path = (
        out / "manifest.json"
    )  # a manifest that recorded the hash accepts that file only
    m = json.loads(path.read_text())
    m["tokenizer"]["tokenizer_json_sha256"] = mod.sha256_file(tdir / "tokenizer.json")
    path.write_text(json.dumps(m))
    assert mod.verify(out, tokenizer_dir=tdir) == []
    (tdir / "tokenizer.json").write_text('{"changed": 1}')
    assert any("tokenizer.json" in b for b in mod.verify(out, tokenizer_dir=tdir))


def test_too_few_calibration_tokens_fail_the_build(mod, built, monkeypatch, tmp_path):
    out, m, tok = built
    monkeypatch.setattr(mod, "MIN_CALIB_TOKENS", m["calibration"]["tokens"] + 1)
    monkeypatch.setattr(mod, "PREFERRED_CALIB_TOKENS", m["calibration"]["tokens"] + 1)
    again = mod.build(
        out, tok, tokens=12_000, holdout_tokens=4_000, holdout_share=0.25,
        tokenizer_dir="toy", log=lambda *_: None,
    )  # fmt: skip
    assert not again["ok"] and not again["gates"]["calibration_tokens"]["ok"]
    # the preferred size is a soft gate: a warning, never a failure of its own
    assert any(w.startswith("calibration_tokens_preferred") for w in again["warnings"])
    assert not any(w.startswith("calibration_tokens:") for w in again["warnings"])
    assert any("failed gate: calibration_tokens" in b for b in mod.verify(out))


def test_the_held_out_floor_of_32_windows_is_wired_into_the_mix(
    mod, built, monkeypatch
):
    """``build`` asks ``hi2.calib`` for at least ``MIN_HELDOUT_WINDOWS`` x ``SEQ_LEN``
    held-out tokens, and that is a hard error there."""
    out, _, tok = built
    monkeypatch.setattr(mod, "MIN_HELDOUT_WINDOWS", 600)  # 38,400 toy tokens
    with pytest.raises(ValueError, match="holdout has .* need at least 38400"):
        mod.build(
            out, tok, tokens=12_000, holdout_tokens=40_000, holdout_share=0.25,
            tokenizer_dir="toy", log=lambda *_: None,
        )  # fmt: skip


def test_a_held_out_text_copied_from_the_calibration_fails_the_overlap_gate(
    mod, built, tmp_path
):
    out, _, tok = built
    cal = (out / "calib.txt").read_text()
    leak = tmp_path / "leak.txt"
    leak.write_text(cal[1000:9000])  # 8 kB of the calibration text, as a held-out set
    fresh = tmp_path / "fresh.txt"
    fresh.write_text(_words(random.Random(99), 2000))
    again = mod.build(
        out, tok, tokens=12_000, holdout_tokens=4_000, holdout_share=0.25,
        ood={"leak": leak, "fresh": fresh}, tokenizer_dir="toy", log=lambda *_: None,
    )  # fmt: skip
    gates = again["gates"]
    assert not gates["shingle_overlap:heldout_leak.txt"]["ok"]
    assert gates["shingle_overlap:heldout_fresh.txt"]["ok"]
    assert again["heldout"]["leak"]["overlap_with_calibration"]["max"] > 0.9
    assert again["heldout"]["fresh"]["overlap_with_calibration"]["max"] == 0.0
    assert not again["ok"]


def test_a_document_in_both_splits_fails_the_identity_gate(mod, tmp_path):
    """A text and a longer text that contains it are different documents to the split,
    so a pair can land on both sides: the substring gate finds it."""
    rng = random.Random(5)
    out, tok = tmp_path / "calib", ToyTok()
    (out / "src").mkdir(parents=True)
    weights = {}
    for name in mod.SOURCES:
        texts = []
        for _ in range(40):
            short = _words(rng, 80)
            texts += [short, _words(rng, 20) + " " + short + " " + _words(rng, 20)]
        (out / "src" / f"{name}.jsonl").write_text(
            "".join(json.dumps({"text": t}) + "\n" for t in texts)
        )
        weights[name] = {"weight": mod.SOURCES[name]["weight"]}
    (out / "src" / "staging.json").write_text(
        json.dumps({"seed": 0, "sources": weights})
    )
    m = mod.build(
        out, tok, tokens=12_000, holdout_tokens=4_000, holdout_share=0.5,
        tokenizer_dir="toy", log=lambda *_: None,
    )  # fmt: skip
    gate = m["gates"]["no_document_in_both_splits"]
    assert not gate["ok"] and not m["ok"] and "shared" in gate["detail"]


def test_a_source_the_template_mostly_refuses_stops_the_staging(mod, tmp_path):
    class Refusing(ToyTok):
        def apply_chat_template(self, messages, **kw):
            if zlib.crc32(messages[0]["content"].encode()) % 2:  # about half
                raise ValueError("Unknown message role")
            return super().apply_chat_template(messages, **kw)

    rng = random.Random(1)
    hub = _fake_hub(tmp_path / "hub", mod, rng)
    docs = _docs(tmp_path / "docs.txt", rng)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="were refused, more than 5%"):
        mod.stage(tmp_path / "out", Refusing(), hub, docs, seed=0, log=lambda *_: None)


def test_staging_is_deterministic_and_the_seed_changes_the_sample(
    mod, tmp_path, monkeypatch
):
    rng = random.Random(2)
    hub = _fake_hub(tmp_path / "hub", mod, rng, n=80)
    docs = _docs(tmp_path / "docs.txt", rng)
    monkeypatch.setitem(mod.SOURCES["wizard"], "stage", 20)
    files = []
    for seed, name in ((0, "a"), (0, "b"), (1, "c")):
        out = tmp_path / name
        out.mkdir()
        mod.stage(out, ToyTok(), hub, docs, seed=seed, log=lambda *_: None)
        files.append((out / "src" / "wizard.jsonl").read_text())
    assert files[0] == files[1] != files[2]
    assert len(files[0].splitlines()) == 20
