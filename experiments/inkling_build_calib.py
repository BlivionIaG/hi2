# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Calibration and held-out tokens for the Inkling-2-Mini quantization (stage 4).

The routed experts are calibrated with real tokens: an expert is topped up with its
layer's average statistics when it saw fewer than ``expert_min_tokens`` (2.0) x its input
width tokens (2304 for gate/up), so a few thousand tokens leave all 128 experts of a layer
under the floor. The mix is chat- and tool-heavy (the largest evidenced lever of the 4B
study): the records are rendered with the model's own chat template (real special tokens,
the default ``Thinking effort level`` system turn, ToolACE tool lists as ``tool_declare``,
GSM8K worked solutions as a thinking turn), then mixed and split by ``hi2.calib``.

    python experiments/inkling_build_calib.py stage  --out DIR --tokenizer SRC --docs FILE
    python experiments/inkling_build_calib.py build  --out DIR --tokenizer SRC [--ood N=FILE]
    python experiments/inkling_build_calib.py verify --out DIR [--tokenizer SRC]

``stage`` samples (seeded) and renders the cached datasets into ``DIR/src/*.jsonl``; ``build``
mixes them (``calib.txt``, ``heldout_mixed.txt``, token windows as ``*_ids.npy``) and writes
``manifest.json``: source files and licences, token counts per source, sha256 of every file
and of the token windows, and the gates (below). ``verify`` re-hashes every file and, with a
tokenizer, re-tokenizes and compares what ``hi2.pipeline.load_calibration_ids`` would read.
Exit code 0: every gate held. The gates: at least 262,144 calibration tokens and 32 held-out
windows of 2048; every source present in both splits; no prepared document in both texts;
no held-out window sharing more than half of its 32-token shingles with the calibration
tokens; the share of special tokens plausible for chat records; ids inside the vocabulary;
the pipeline reads exactly the stored windows.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

SCHEMA = "hi2.inkling-calib/1"
SEQ_LEN = 2048
MIN_CALIB_TOKENS = 128 * SEQ_LEN  # the contract: 128 windows of 2048
PREFERRED_CALIB_TOKENS = 256 * SEQ_LEN
MIN_HELDOUT_WINDOWS = 32
PREFERRED_HELDOUT_WINDOWS = 64
PIPELINE_WINDOWS = 256  # --calib-samples of the run (524,288 tokens): read and compared
SHINGLE = 32
SHINGLE_LIMIT = 0.5
SPECIAL_SHARE = (0.002, 0.20)  # chat records: a few percent of the tokens are specials
# a set below these is a dry run on a toy model: its manifest says ``production: false``
PRODUCTION = {
    "seq_len": 2048,
    "calibration_tokens": 262_144,
    "heldout_windows": 32,
    "shingle_limit": 0.5,
}
TOOL_MARK = "Here is a list of functions in JSON format that you can invoke:\n"
ROLES = {"user": "user", "assistant": "assistant", "tool": "tool", "system": "system"}

# name -> where the records are, their licence (dataset cards), how many to stage and the
# mixing weight. ``docs`` is a plain text file given by --docs (vLLM documentation, Apache-2.0,
# and the RDNA fork's own notes).
SOURCES: dict[str, dict] = {
    "toolace": {
        "repo": "Team-ACE/ToolACE",
        "file": "data.json",
        "license": "apache-2.0",
        "stage": 600,
        "weight": 0.30,
    },
    "guanaco": {
        "repo": "timdettmers/openassistant-guanaco",
        "file": "openassistant_best_replies_train.jsonl",
        "license": "apache-2.0",
        "stage": 800,
        "weight": 0.20,
    },
    "wizard": {
        "repo": "WizardLM/WizardLM_evol_instruct_70k",
        "file": "alpaca_evol_instruct_70k.json",
        "license": "mit",
        "stage": 800,
        "weight": 0.20,
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "file": "main/train-00000-of-00001.parquet",
        "license": "mit",
        "stage": 500,
        "weight": 0.05,
    },
    "docs": {
        "license": "apache-2.0 (vLLM documentation) + project notes",
        "stage": 0,  # all blocks
        "weight": 0.25,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int32).tobytes()).hexdigest()


def atomic_write(path: Path, data: str | bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


def atomic_npy(path: Path, a: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        np.save(f, a)
    tmp.replace(path)


# ----------------------------------------------------------------------- record rendering


def make_renderer(tok):
    """``render(messages, tools=None)``: the model's chat template, no generation prompt."""

    def render(messages, tools=None):
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("the chat template returned no text")
        return text

    return render


def render_toolace(rec: dict, render) -> tuple[str, bool]:
    """A ToolACE record: its JSON tool list leaves the system prompt for the template's
    ``tool_declare`` turn when it parses, the calls stay the dataset's text. Returns the
    text and whether the tools were declared natively."""
    system, tools = rec["system"], None
    i = system.find(TOOL_MARK)
    if i >= 0:
        j = i + len(TOOL_MARK)
        try:
            obj, end = json.JSONDecoder().raw_decode(system[j:])
        except ValueError:
            obj = None
        if (
            isinstance(obj, list)
            and obj
            and all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in obj)
        ):
            tools = obj
            system = (system[:i] + system[j + end :]).strip()
    messages = [{"role": "system", "content": system}]
    messages += [
        {"role": ROLES[c["from"]], "content": c["value"]} for c in rec["conversations"]
    ]
    return render(messages, tools), tools is not None


_TURN = re.compile(r"### (Human|Assistant): ")


def parse_guanaco(text: str) -> list[dict] | None:
    """``### Human: ...### Assistant: ...`` into alternating user/assistant messages; None
    when the record does not have that shape."""
    parts = _TURN.split(text)
    if parts[0].strip() or len(parts) < 3:
        return None
    out: list[dict] = []
    for who, body in zip(parts[1::2], parts[2::2]):
        role = "user" if who == "Human" else "assistant"
        if (out and out[-1]["role"] == role) or not body.strip():
            return None
        out.append({"role": role, "content": body.strip()})
    return out if out[0]["role"] == "user" else None


def gsm8k_messages(rec: dict) -> list[dict]:
    """The worked solution is the thinking turn, the number after ``####`` the answer."""
    reasoning, sep, final = rec["answer"].rpartition("####")
    reply = {"role": "assistant", "content": rec["answer"].strip()}
    if sep and reasoning.strip() and final.strip():
        reply = {
            "role": "assistant",
            "content": f"The answer is {final.strip()}.",
            "reasoning_content": reasoning.strip(),
        }
    return [{"role": "user", "content": rec["question"].strip()}, reply]


def _records(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return json.loads(path.read_text())


def find_cached(hf_cache: Path, repo: str, file: str) -> Path:
    """``<hf_cache>/datasets--owner--name/snapshots/<hash>/<file>`` (the newest snapshot)."""
    root = hf_cache / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    found = sorted(root.glob(f"*/{file}"))
    if not found:
        raise SystemExit(f"{repo}: {file} is not in the cache {hf_cache}")
    return found[-1]


def near_unique(
    texts: list[str], span: int = 64, every: int = 8, limit: float = 0.5
) -> list[int]:
    """Indices of the texts to keep, in their order: longest first, a text is dropped
    when at least ``limit`` of its sampled ``span``-character windows (a window is
    sampled when its crc32 is a multiple of ``every``) occur in the texts kept before it.
    A section that two pages of a documentation set both carry is kept once, so the
    split cannot put one copy in the calibration text and the other in the held-out."""
    seen: set[int] = set()
    keep = []
    for i in sorted(range(len(texts)), key=lambda j: -len(texts[j])):
        t = texts[i]
        spans = {
            h
            for k in range(max(0, len(t) - span + 1))
            if (h := zlib.crc32(t[k : k + span].encode())) % every == 0
        }
        if spans and len(spans & seen) >= limit * len(spans):
            continue
        seen |= spans
        keep.append(i)
    return sorted(keep)


def stage_source(name: str, spec: dict, path: Path, render, seed: int):
    """Sample, render and return ``(texts, report)`` of one source; records the template
    refuses are dropped and counted."""
    if name == "docs":
        from hi2.calib import read_documents

        records = read_documents(path)
        total = len(records)
        keep = near_unique(records)
        near_dropped = total - len(keep)
        records = [records[i] for i in keep]
    else:
        records = _records(path)
        total, near_dropped = len(records), None
    order = list(range(len(records)))
    random.Random(f"{seed}:{name}").shuffle(order)
    want = spec["stage"] or len(records)
    texts, dropped, native, attempted = [], {}, 0, 0
    for idx in order:
        if len(texts) >= want:
            break
        rec = records[idx]
        attempted += 1
        try:
            if name == "docs":
                text = rec
            elif name == "toolace":
                text, was_native = render_toolace(rec, render)
                native += was_native
            elif name == "guanaco":
                msgs = parse_guanaco(rec["text"])
                if msgs is None:
                    raise ValueError("not a Human/Assistant alternation")
                text = render(msgs)
            elif name == "wizard":
                text = render(
                    [
                        {"role": "user", "content": rec["instruction"]},
                        {"role": "assistant", "content": rec["output"]},
                    ]
                )
            else:  # gsm8k
                text = render(gsm8k_messages(rec))
        except Exception as e:  # a record the template or the parser refuses
            key = f"{type(e).__name__}: {str(e)[:60]}"
            dropped[key] = dropped.get(key, 0) + 1
            continue
        texts.append(text)
    report = {
        "records_total": total,
        "near_duplicates_dropped": near_dropped,
        "attempted": attempted,
        "staged": len(texts),
        "dropped": dropped,
        "sample_seed": f"{seed}:{name}",
        "native_tool_declarations": native if name == "toolace" else None,
    }
    return texts, report


def stage(out: Path, tok, hf_cache: Path, docs: Path | None, seed: int, log=print):
    """Write ``out/src/<name>.jsonl`` (one rendered ``{"text": ...}`` per line) and
    ``out/src/staging.json`` for every source."""
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    render = make_renderer(tok)
    report = {"seed": seed, "sources": {}}
    for name, spec in SOURCES.items():
        if name == "docs":
            if docs is None:
                raise SystemExit("--docs FILE is required (the plain text source)")
            path = docs
        else:
            path = find_cached(hf_cache, spec["repo"], spec["file"])
        texts, rep = stage_source(name, spec, path, render, seed)
        if not texts:
            raise SystemExit(f"{name}: nothing staged ({rep['dropped']})")
        if sum(rep["dropped"].values()) > 0.05 * rep["attempted"]:
            raise SystemExit(
                f"{name}: {sum(rep['dropped'].values())} of {rep['attempted']} records "
                f"were refused, more than 5%: {rep['dropped']}"
            )
        target = src / f"{name}.jsonl"
        atomic_write(
            target,
            "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        )
        rep |= {
            "license": spec["license"],
            "weight": spec["weight"],
            "origin": str(path),
            "origin_sha256": sha256_file(path),
            "origin_bytes": path.stat().st_size,
            "staged_file": f"src/{name}.jsonl",
            "staged_sha256": sha256_file(target),
            "staged_chars": sum(len(t) for t in texts),
        }
        if spec.get("repo"):
            rep["repo"] = spec["repo"]
            rep["snapshot"] = path.parent.name if "snapshots" in str(path) else None
        report["sources"][name] = rep
        log(
            f"staged {name}: {len(texts)} of {rep['records_total']} records {rep['dropped']}"
        )
    atomic_write(src / "staging.json", json.dumps(report, indent=2, sort_keys=True))
    return report


# ------------------------------------------------------------------------ token windows


def encode(tok, text: str) -> np.ndarray:
    """Token ids of ``text`` as the pipeline reads them (no special tokens added)."""
    return np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int32)


def windows(ids: np.ndarray, seq_len: int | None = None) -> np.ndarray:
    """Whole windows of ``seq_len`` ids (default ``SEQ_LEN``), the tail is dropped."""
    seq_len = seq_len or SEQ_LEN
    n = len(ids) // seq_len
    return ids[: n * seq_len].reshape(n, seq_len)


# One independent random odd 64-bit multiplier per position of a shingle: two different
# runs of ids collide only when a weighted sum of their differences is 0 mod 2**64, about
# 2**-64 per pair. (Multipliers in a small ratio, such as 1, 3, 5, ..., collide whenever
# two runs have equal weighted sums: thousands of false matches on a few hundred
# thousand tokens.)
_MULT = np.random.default_rng(0x51D61E).integers(
    0, 2**63, size=SHINGLE, dtype=np.uint64
) * np.uint64(2) + np.uint64(1)


def shingle_hashes(ids: np.ndarray, n: int = SHINGLE) -> np.ndarray:
    """A 64-bit hash of every run of ``n`` consecutive ids (positions ``0..len-n``)."""
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    view = np.lib.stride_tricks.sliding_window_view(ids.astype(np.uint64) + 1, n)
    return (view * _MULT[:n]).sum(axis=1, dtype=np.uint64)


def overlap_fractions(held: np.ndarray, calib_ids: np.ndarray) -> np.ndarray:
    """Per held-out window: the share of its 32-token shingles that occur in the
    calibration tokens (shared boilerplate such as a system prompt counts, a copied
    document counts for all of its length)."""
    known = np.unique(shingle_hashes(calib_ids))
    out = np.zeros(len(held))
    for i, w in enumerate(held):
        h = shingle_hashes(w)
        out[i] = float(np.isin(h, known).mean()) if len(h) else 0.0
    return out


def special_ids(tok) -> set[int]:
    """Ids of the ``<|...|>`` added tokens (the chat structure)."""
    added = tok.get_added_vocab()
    return {i for t, i in added.items() if t.startswith("<|") and t.endswith("|>")}


# ------------------------------------------------------------------------------- build


def _rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return path


def gate(gates: dict, name: str, ok: bool, detail: str, hard: bool = True) -> None:
    gates[name] = {"ok": bool(ok), "hard": hard, "detail": detail}


def thresholds() -> dict:
    """The limits this build gates on, and whether they are the production ones."""
    t = {
        "seq_len": SEQ_LEN,
        "min_calibration_tokens": MIN_CALIB_TOKENS,
        "preferred_calibration_tokens": PREFERRED_CALIB_TOKENS,
        "min_heldout_windows": MIN_HELDOUT_WINDOWS,
        "preferred_heldout_windows": PREFERRED_HELDOUT_WINDOWS,
        "pipeline_windows": PIPELINE_WINDOWS,
        "shingle_limit": SHINGLE_LIMIT,
    }
    t["production"] = (
        PRODUCTION["seq_len"] == SEQ_LEN
        and PRODUCTION["calibration_tokens"] <= MIN_CALIB_TOKENS
        and PRODUCTION["heldout_windows"] <= MIN_HELDOUT_WINDOWS
        and PRODUCTION["shingle_limit"] >= SHINGLE_LIMIT
    )
    return t


def scale_down(
    seq_len: int,
    min_calib_windows: int,
    min_heldout_windows: int,
    pipeline_windows: int,
) -> None:
    """Dry run on a toy model: windows of ``seq_len`` tokens and smaller floors. A window
    shorter than a boilerplate paragraph of the chat records (the tool-use instruction) can
    match the calibration set entirely, so the overlap gate reports but does not fail. The
    manifest is not a ``production`` one and ``verify --production`` refuses it."""
    global SEQ_LEN, MIN_CALIB_TOKENS, PREFERRED_CALIB_TOKENS, SHINGLE_LIMIT
    global MIN_HELDOUT_WINDOWS, PREFERRED_HELDOUT_WINDOWS, PIPELINE_WINDOWS
    SHINGLE_LIMIT = 1.01
    SEQ_LEN = seq_len
    MIN_CALIB_TOKENS = min_calib_windows * seq_len
    PREFERRED_CALIB_TOKENS = 2 * MIN_CALIB_TOKENS
    MIN_HELDOUT_WINDOWS = min_heldout_windows
    PREFERRED_HELDOUT_WINDOWS = 2 * min_heldout_windows
    PIPELINE_WINDOWS = pipeline_windows


def build(
    out: Path,
    tok,
    *,
    tokens: int = 540_672,
    holdout_tokens: int = 196_608,
    holdout_share: float = 0.2,
    ood: dict[str, Path] | None = None,
    tokenizer_dir: str = "",
    log=print,
) -> dict:
    """Mix the staged sources, write the texts, windows and ``manifest.json``; returns the
    manifest (``ok`` false when a hard gate failed)."""
    from hi2.calib import build_mix

    staging = json.loads((out / "src" / "staging.json").read_text())
    sources = [
        (out / "src" / f"{n}.jsonl", s["weight"]) for n, s in staging["sources"].items()
    ]
    count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    tok_id = tokenizer_dir or "tokenizer"
    cal_text, held_text, mix = build_mix(
        sources,
        tokens,
        count,
        holdout_share,
        holdout_tokens,
        tokenizer_id=tok_id,
        min_holdout_tokens=MIN_HELDOUT_WINDOWS * SEQ_LEN,
    )
    log(f"mixed: {mix['tokens']} calibration tokens, {mix['held_out_tokens']} held-out")
    texts = {"calib.txt": cal_text, "heldout_mixed.txt": held_text}
    for name, path in (ood or {}).items():
        texts[f"heldout_{name}.txt"] = path.read_text(encoding="utf-8", errors="ignore")
    for fname, text in texts.items():
        atomic_write(out / fname, text)
    ids = {f: encode(tok, t) for f, t in texts.items()}
    cal_ids = ids["calib.txt"]
    win = {f: windows(a) for f, a in ids.items()}
    for fname, w in win.items():
        atomic_npy(out / fname.replace(".txt", "_ids.npy"), w)
    specials = special_ids(tok)
    gates: dict[str, dict] = {}
    n_cal, n_win = len(cal_ids), len(win["heldout_mixed.txt"])
    gate(
        gates,
        "calibration_tokens",
        n_cal >= MIN_CALIB_TOKENS,
        f"{n_cal} >= {MIN_CALIB_TOKENS}",
    )
    gate(
        gates,
        "calibration_tokens_preferred",
        n_cal >= PREFERRED_CALIB_TOKENS,
        f"{n_cal} >= {PREFERRED_CALIB_TOKENS}",
        hard=False,
    )
    gate(
        gates,
        "heldout_windows",
        n_win >= MIN_HELDOUT_WINDOWS,
        f"{n_win} >= {MIN_HELDOUT_WINDOWS}",
    )
    gate(
        gates,
        "heldout_windows_preferred",
        n_win >= PREFERRED_HELDOUT_WINDOWS,
        f"{n_win} >= {PREFERRED_HELDOUT_WINDOWS}",
        hard=False,
    )
    per_source = {}
    for path, _ in sources:
        rep = mix["sources"][str(path)]
        per_source[path.stem] = {
            "documents_available": rep["documents"],
            "calibration_documents": rep["used"],
            "calibration_tokens": rep["tokens"],
            "calibration_token_share": rep["token_share"],
            "requested_share": rep["requested_share"],
            "heldout_documents": rep["held_out_documents"],
            "heldout_tokens": rep["held_out_tokens"],
            "heldout_token_share": rep["held_out_token_share"],
        }
    missing = [
        n
        for n, s in per_source.items()
        if s["calibration_tokens"] == 0 or s["heldout_tokens"] == 0
    ]
    gate(gates, "every_source_in_both_splits", not missing, f"missing: {missing}")
    # no prepared document in both texts: every staged document, found by substring
    both, seen = [], 0
    for path, _ in sources:
        for ln in path.read_text(encoding="utf-8").splitlines():
            doc = json.loads(ln)["text"]
            in_cal, in_held = doc in cal_text, doc in held_text
            seen += in_cal or in_held
            if in_cal and in_held:
                both.append(hashlib.sha256(doc.encode()).hexdigest()[:16])
    gate(
        gates,
        "no_document_in_both_splits",
        not both,
        f"{len(both)} shared of {seen} used: {both[:3]}",
    )
    overlap = {}
    for fname, w in win.items():
        if fname == "calib.txt":
            continue
        frac = overlap_fractions(w, cal_ids)
        overlap[fname] = {
            "windows": len(w),
            "mean": float(frac.mean()) if len(frac) else 0.0,
            "p99": float(np.quantile(frac, 0.99)) if len(frac) else 0.0,
            "max": float(frac.max()) if len(frac) else 0.0,
        }
        gate(
            gates,
            f"shingle_overlap:{fname}",
            bool(len(frac) == 0 or frac.max() < SHINGLE_LIMIT),
            f"max {overlap[fname]['max']:.3f} < {SHINGLE_LIMIT}",
        )
    special = float(np.isin(cal_ids, list(specials)).mean()) if len(cal_ids) else 0.0
    gate(
        gates,
        "special_token_share",
        SPECIAL_SHARE[0] <= special <= SPECIAL_SHARE[1],
        f"{special:.4f} in {SPECIAL_SHARE}",
    )
    vocab = len(tok)
    top = int(max(int(a.max()) for a in ids.values() if len(a)))
    gate(gates, "ids_in_vocabulary", top < vocab, f"max id {top} < {vocab}")
    pipeline = _pipeline_windows(out / "calib.txt", tok, win["calib.txt"])
    gate(
        gates, "pipeline_reads_the_stored_windows", pipeline is not False, str(pipeline)
    )
    files = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.name != "manifest.json":
            files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    for p in sorted((out / "src").iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files[f"src/{p.name}"] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    tdir = Path(tokenizer_dir) if tokenizer_dir else None
    manifest = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seq_len": SEQ_LEN,
        "thresholds": thresholds(),
        "tokenizer": {
            "dir": tokenizer_dir,
            "vocabulary": vocab,
            "tokenizer_json_sha256": sha256_file(tdir / "tokenizer.json")
            if tdir and (tdir / "tokenizer.json").exists()
            else None,
            "chat_template_sha256": sha256_file(tdir / "chat_template.jinja")
            if tdir and (tdir / "chat_template.jinja").exists()
            else None,
            "special_token_ids": len(specials),
        },
        "staging": staging,
        "mix": {
            "requested": {
                "tokens": tokens,
                "holdout_tokens": holdout_tokens,
                "holdout_share": holdout_share,
                "weights": {n: s["weight"] for n, s in staging["sources"].items()},
            },
            "report": {k: v for k, v in mix.items() if k != "sources"},
            "per_source": per_source,
        },
        "calibration": {
            "file": "calib.txt",
            "tokens": n_cal,
            "windows": len(win["calib.txt"]),
            "special_token_share": special,
            "ids_file": "calib_ids.npy",
            "ids_sha256_first_4_windows": sha256_array(win["calib.txt"][:4]),
            "pipeline_windows": min(PIPELINE_WINDOWS, len(win["calib.txt"])),
            "ids_sha256_pipeline_windows": sha256_array(
                win["calib.txt"][:PIPELINE_WINDOWS]
            ),
            "ids_sha256_all_windows": sha256_array(win["calib.txt"]),
        },
        "heldout": {
            fname.removeprefix("heldout_").removesuffix(".txt"): {
                "file": fname,
                "tokens": int(len(ids[fname])),
                "windows": len(win[fname]),
                "ids_file": fname.replace(".txt", "_ids.npy"),
                "ids_sha256_all_windows": sha256_array(win[fname]),
                "ids_sha256_first_32_windows": sha256_array(win[fname][:32]),
                "overlap_with_calibration": overlap[fname],
            }
            for fname in win
            if fname != "calib.txt"
        },
        "files": files,
        "gates": gates,
    }
    manifest["ok"] = all(g["ok"] for g in gates.values() if g["hard"])
    manifest["warnings"] = [
        f"{k}: {g['detail']}" for k, g in gates.items() if not g["ok"] and not g["hard"]
    ]
    atomic_write(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _pipeline_windows(path: Path, tok, stored: np.ndarray):
    """What ``hi2.pipeline.load_calibration_ids`` reads from ``path`` equals the stored
    first windows (True/False), or a note when torch is not available."""
    try:
        import torch

        from hi2.pipeline import load_calibration_ids
    except ImportError as e:
        return f"skipped ({e})"
    n = min(PIPELINE_WINDOWS, len(stored))
    got = load_calibration_ids(path, tok, n, SEQ_LEN)
    return bool(
        torch.equal(got.to(torch.int64), torch.from_numpy(stored[:n]).to(torch.int64))
    )


def verify(
    out: Path, tok=None, production: bool = False, tokenizer_dir: Path | None = None
) -> list[str]:
    """Problems found (empty = the set is what the manifest says): every file's sha256,
    the gates the build recorded, and with a tokenizer the token windows. ``production``
    also refuses a manifest built with scaled-down thresholds (a toy-model dry run)."""
    mpath = out / "manifest.json"
    if not mpath.exists():
        return [f"{mpath}: missing"]
    m = json.loads(mpath.read_text())
    bad = []
    if production and not m.get("thresholds", {}).get("production"):
        bad.append("not a production manifest (scaled-down or without thresholds)")
    if m.get("schema") != SCHEMA:
        bad.append(f"schema {m.get('schema')!r} != {SCHEMA!r}")
    if not m.get("ok"):
        bad.append(
            "the build recorded a failed gate: "
            + ", ".join(k for k, g in m["gates"].items() if g["hard"] and not g["ok"])
        )
    for name, info in m["files"].items():
        p = out / name
        if not p.exists():
            bad.append(f"{name}: missing")
        elif p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
            bad.append(f"{name}: sha256/size differ from the manifest")
    cal = m["calibration"]
    if (out / cal["ids_file"]).exists():
        ids = np.load(out / cal["ids_file"])
        if sha256_array(ids) != cal["ids_sha256_all_windows"]:
            bad.append(f"{cal['ids_file']}: windows differ from the manifest")
    if tok is not None and (out / "calib.txt").exists():
        text = (out / "calib.txt").read_text(encoding="utf-8")
        w = windows(encode(tok, text), m["seq_len"])
        if sha256_array(w) != cal["ids_sha256_all_windows"]:
            bad.append("calib.txt tokenizes to other windows than the manifest's")
    if tokenizer_dir is not None:  # the windows belong to this tokenizer and template
        for key, fname in (
            ("tokenizer_json_sha256", "tokenizer.json"),
            ("chat_template_sha256", "chat_template.jinja"),
        ):
            p = Path(tokenizer_dir) / fname
            got = sha256_file(p) if p.exists() else None
            if got != m["tokenizer"].get(key):
                bad.append(
                    f"{fname} of {tokenizer_dir} is not the one the set was built with"
                )
    return bad


# -------------------------------------------------------------------------------- CLI


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("stage", "build", "verify"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None, help="Inkling-2 directory")
    ap.add_argument(
        "--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub"
    )
    ap.add_argument(
        "--docs", type=Path, default=None, help="plain text source of `docs`"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=540_672)
    ap.add_argument("--holdout-tokens", type=int, default=196_608)
    ap.add_argument("--holdout-share", type=float, default=0.2)
    ap.add_argument(
        "--ood",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="held-out only text",
    )
    ap.add_argument(
        "--production",
        action="store_true",
        help="verify: refuse a manifest built with scaled-down thresholds",
    )
    ap.add_argument(
        "--dry-run-scale",
        nargs=4,
        type=int,
        metavar=("SEQ_LEN", "CALIB_WINDOWS", "HELDOUT_WINDOWS", "PIPELINE_WINDOWS"),
        help="build: toy-model dry run with these (non-production) limits",
    )
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.dry_run_scale:
        scale_down(*a.dry_run_scale)
    tok = None
    if a.tokenizer is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if a.command == "verify":
        bad = verify(a.out, tok, a.production, a.tokenizer)
        print(json.dumps({"ok": not bad, "problems": bad}, indent=2))
        return 1 if bad else 0
    if tok is None:
        raise SystemExit("--tokenizer is required")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "stage":
        stage(a.out, tok, a.hf_cache, a.docs, a.seed)
        return 0
    ood = {}
    for item in a.ood:
        name, _, path = item.partition("=")
        ood[name] = Path(path)
    m = build(
        a.out,
        tok,
        tokens=a.tokens,
        holdout_tokens=a.holdout_tokens,
        holdout_share=a.holdout_share,
        ood=ood,
        tokenizer_dir=str(a.tokenizer),
    )
    print(
        json.dumps(
            {"ok": m["ok"], "gates": m["gates"], "warnings": m["warnings"]}, indent=2
        )
    )
    return 0 if m["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
