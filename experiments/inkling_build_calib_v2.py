# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Calibration and held-out/OOD sets v2 for Inkling-2-Mini (long context, code, agentic,
multilingual).

The stage-4 set was 2,048-token windows of mostly English chat and tool use. This builder
adds the domains the production target needs: native tool-call turns rendered with the
Inkling chat template, code in several languages from local project trees, natural text and
chat in 12 languages, long coherent documents at 8k/16k/32k tokens, math/reasoning and
general chat. Every window is labelled in the manifest with its category, language, source
and licence. Calibration and held-out are disjoint by document (sha256) and by near-duplicate
shingles; the held-out floor is 16 windows per category (and per long-context length).

    python experiments/inkling_build_calib_v2.py stage  --out DIR --tokenizer SRC --local REPO
    python experiments/inkling_build_calib_v2.py build  --out DIR --tokenizer SRC
    python experiments/inkling_build_calib_v2.py ood    --out DIR --tokenizer SRC
    python experiments/inkling_build_calib_v2.py verify --out DIR --tokenizer SRC --production

``stage`` writes ``DIR/src/<category>.jsonl`` (one prepared ``{"text", "category",
"language", "source", "license"}`` per line). ``build`` dedupes, splits by document, cuts
windows of the category's lengths, writes ``<split>_<category>__<language>__<length>_ids.npy``
and ``manifest.json`` with the per-window provenance. ``ood`` (after ``build``) adds the
separate out-of-distribution set, English Wikipedia, a source in no other split. ``verify``
re-hashes every file and re-checks the hard gates.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SCHEMA = "hi2.inkling-calib-v2/1"
LONG_LENGTHS = (8192, 16384, 32768)
SHORT_LENGTH = 2048
MIN_HELDOUT_WINDOWS = 16
MIN_CALIB_WINDOWS = 16
MIN_HELDOUT_DOCS = 16  # per (category, language), when the group has the documents
MIN_CALIB_DOCS = 8
SHINGLE = 32
SHINGLE_LIMIT = 0.5
SPECIAL_SHARE = (0.002, 0.20)
LANGUAGES = ("zh", "ja", "ko", "ru", "ar", "hi", "es", "fr", "de", "pt", "tr", "vi")
CODE_LANGUAGES = ("python", "c", "cpp", "rust", "typescript", "javascript", "shell", "sql", "yaml", "json")
PRODUCTION = {
    "min_heldout_windows": MIN_HELDOUT_WINDOWS,
    "min_calib_windows": MIN_CALIB_WINDOWS,
    "shingle_limit": SHINGLE_LIMIT,
    "long_lengths": list(LONG_LENGTHS),
}
ROLES = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system", "tool": "tool"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def encode(tok, text: str) -> np.ndarray:
    return np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int32)


def windows(ids: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(ids) // seq_len
    return ids[: n * seq_len].reshape(n, seq_len)


def near_unique(texts: list[str], span: int = 64, every: int = 8, limit: float = 0.5) -> list[int]:
    """Indices to keep, longest first; a text is dropped when ``limit`` of its sampled
    ``span``-character shingles occur in the texts kept before it."""
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


_MULT = np.random.default_rng(0x51D61E).integers(0, 2**63, size=SHINGLE, dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def shingle_hashes(ids: np.ndarray, n: int = SHINGLE) -> np.ndarray:
    """The rolling hash of every ``n``-token shingle, in chunks: a 32k-token calibration
    window and a concatenation of them would otherwise materialise a large temporary."""
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    step = 1 << 16
    out = []
    for start in range(0, len(ids) - n + 1, step):
        chunk = ids[start : start + step + n - 1].astype(np.uint64) + 1
        view = np.lib.stride_tricks.sliding_window_view(chunk, n)
        out.append((view * _MULT[:n]).sum(axis=1, dtype=np.uint64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


def overlap_fractions(held: np.ndarray, calib_ids: np.ndarray) -> np.ndarray:
    known = np.unique(shingle_hashes(calib_ids))
    out = np.zeros(len(held))
    if not len(known):  # the calibration windows are shorter than one shingle
        return out
    for i, w in enumerate(held):
        h = shingle_hashes(w)
        if not len(h):
            continue
        at = np.searchsorted(known, h)
        at[at == len(known)] = 0
        out[i] = float((known[at] == h).mean())
    return out


def special_ids(tok) -> set[int]:
    added = tok.get_added_vocab()
    return {i for t, i in added.items() if t.startswith("<|") and t.endswith("|>")}


def gate(gates: dict, name: str, ok: bool, detail: str, hard: bool = True) -> None:
    gates[name] = {"ok": bool(ok), "hard": hard, "detail": detail}


# ----------------------------------------------------------------------- source loading

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_texts(path: Path, min_chars: int = 2000) -> list[str]:
    """Plain text blocks separated by blank lines, grouped to ``min_chars``."""
    blocks = [b.strip() for b in path.read_text(encoding="utf-8", errors="ignore").split("\n\n") if b.strip()]
    out, cur = [], ""
    for b in blocks:
        cur = f"{cur}\n\n{b}" if cur else b
        if len(cur) >= min_chars:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out


def _iter_local_files(root: Path, suffixes: tuple[str, ...], limit: int = 200) -> list[Path]:
    found = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes and p.stat().st_size < 2_000_000:
            found.append(p)
            if len(found) >= limit:
                break
    return found


def _chat_messages(record: dict) -> list[dict] | None:
    turns = record.get("messages") or record.get("conversations") or record.get("chat")
    if not turns:
        return None
    out = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        role = t.get("role") or t.get("from") or "user"
        content = t.get("content") if "content" in t else t.get("value")
        if not isinstance(content, str):
            return None
        out.append({"role": ROLES.get(role, role), "content": content})
    return out if out and out[0]["role"] in ("system", "user") else None


def _render(tok, messages: list[dict], tools: list | None = None) -> str:
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the chat template returned no text")
    return text


def _stage_agentic(tok, local: Path, hf_cache: Path, seed: int, log=print) -> list[dict]:
    """Native tool-call turns: ToolACE (native tool_declare where the JSON parses), the
    local agentic_bench system prompts and tasks, and Hermes tool-call records."""
    records = []
    # ToolACE: split the system prompt's tool list into a native tool_declare turn.
    toolace = sorted(hf_cache.glob("datasets--Team-ACE--ToolACE/snapshots/*/data.json"))
    if toolace:
        data = json.loads(toolace[-1].read_text())
        rng = random.Random(f"{seed}:toolace")
        rng.shuffle(data)
        for rec in data[:400]:
            system = rec.get("system", "")
            tools = None
            mark = "Here is a list of functions in JSON format that you can invoke:\n"
            i = system.find(mark)
            if i >= 0:
                try:
                    obj, end = json.JSONDecoder().raw_decode(system[i + len(mark):])
                    if isinstance(obj, list) and obj and all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in obj):
                        tools, system = obj, (system[:i] + system[i + len(mark) + end:]).strip()
                except ValueError:
                    pass
            msgs = [{"role": "system", "content": system}] if system else []
            for c in rec.get("conversations", []):
                msgs.append({"role": ROLES.get(c.get("from"), "user"), "content": c.get("value", "")})
            try:
                text = _render(tok, msgs, tools)
            except Exception:
                continue
            records.append({"text": text, "category": "agentic", "language": "en", "source": "Team-ACE/ToolACE", "license": "apache-2.0"})
            if len(records) >= 120:
                break
    # Local agentic_bench: system prompt + task, as a user turn.
    prompts = local / "tools/agentic_bench/system_prompts"
    tasks = local / "tools/agentic_bench/tasks"
    if prompts.exists() and tasks.exists():
        for sp in sorted(prompts.glob("*.md"))[:3]:
            sys_text = sp.read_text(encoding="utf-8", errors="ignore")[:8000]
            for tp in sorted(tasks.glob("*.md"))[:8]:
                task = tp.read_text(encoding="utf-8", errors="ignore")[:6000]
                msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": task}]
                try:
                    text = _render(tok, msgs)
                except Exception:
                    continue
                records.append({"text": text, "category": "agentic", "language": "en", "source": f"agentic_bench/{sp.name}", "license": "apache-2.0"})
    # Hermes function-calling records (native tool calls in the dataset).
    try:
        from datasets import load_dataset

        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
        for i, rec in enumerate(ds):
            if i >= 200:
                break
            msgs = _chat_messages(rec)
            if not msgs:
                continue
            try:
                text = _render(tok, msgs)
            except Exception:
                continue
            records.append({"text": text, "category": "agentic", "language": "en", "source": "NousResearch/hermes-function-calling-v1", "license": "apache-2.0"})
    except Exception as e:
        log(f"agentic: hermes skipped ({e})")
    log(f"agentic: {len(records)} records")
    return records


def _stage_code(tok, local: Path, hf_cache: Path, seed: int, log=print) -> list[dict]:
    """Code from local project trees and public permissive datasets."""
    records = []
    roots = {
        "python": [local / "hi2", local / "experiments"],
        "c": [Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp")],
        "cpp": [Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp")],
        "rust": [Path("/home/chenco_adm/rpmbuild/BUILD/linux-6.19.12/drivers")],
        "typescript": [Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp/tools/ui")],
        "javascript": [Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp/scripts")],
        "shell": [local / "experiments", local / "bench"],
        "sql": [Path("/home/chenco_adm/tensorfold-pr1694/rocm-10.1/share/rocprofiler-sdk-rocpd")],
        "yaml": [local / "experiments"],
        "json": [local / "experiments"],
    }
    suffixes = {".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".sh": "shell", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json"}

    def license_of(p: Path) -> str:
        s = str(p)
        if "llama.cpp" in s:
            return "mit"
        if "linux-" in s:
            return "gpl-2.0"  # local Linux tree: public but copyleft, see docs
        if "rocprofiler" in s:
            return "mit"
        return "apache-2.0"  # this repository's own trees
    for lang, dirs in roots.items():
        found = []
        for d in dirs:
            if d.exists():
                found.extend(_iter_local_files(d, tuple(k for k, v in suffixes.items() if v == lang), limit=120))
        rng = random.Random(f"{seed}:{lang}")
        rng.shuffle(found)
        for p in found[:60]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(text) < 200:
                continue
            records.append({"text": text, "category": "code", "language": lang, "source": str(p), "license": license_of(p)})
    # Public code datasets (permissive).
    for repo, split in (("deepmind/code_contests", "train"), ("open-r1/codeforces", "train")):
        try:
            from datasets import load_dataset

            ds = load_dataset(repo, split=split, streaming=True)
            for i, rec in enumerate(ds):
                if i >= 40:
                    break
                sols = rec.get("solutions") or []
                text = sols[0].get("solution", "") if sols and isinstance(sols[0], dict) else ""
                if not text:
                    text = rec.get("description", "") or rec.get("statement", "")
                if not text or len(text) < 200:
                    continue
                records.append({"text": text, "category": "code", "language": "python", "source": repo, "license": "apache-2.0"})
        except Exception as e:
            log(f"code: {repo} skipped ({e})")
    log(f"code: {len(records)} records")
    return records


def _stage_multilingual(tok, seed: int, log=print) -> list[dict]:
    """Natural text and chat in 12 languages: Wikipedia (natural text) and OPUS-100
    (translation as a chat turn)."""
    records = []
    for lang in LANGUAGES:
        try:
            from datasets import load_dataset

            ds = load_dataset("wikimedia/wikipedia", name=f"20231101.{lang}", split="train", streaming=True)
            for i, rec in enumerate(ds):
                if i >= 20:
                    break
                text = rec.get("text", "")
                if len(text) < 2000:
                    continue
                records.append({"text": text, "category": "multilingual", "language": lang, "source": f"wikimedia/wikipedia/{lang}", "license": "cc-by-sa-4.0"})
        except Exception as e:
            log(f"multilingual: wikipedia {lang} skipped ({e})")
        for prefix in (f"en-{lang}", f"{lang}-en"):
            try:
                from datasets import load_dataset

                ds = load_dataset("Helsinki-NLP/opus-100", name=prefix, split="train", streaming=True)
                for i, rec in enumerate(ds):
                    if i >= 10:
                        break
                    tr = rec.get("translation", {})
                    if prefix.startswith("en-"):
                        src, tgt = tr.get("en", ""), tr.get(lang, "")
                        prompt = f"Translate to {lang}:\n{src}"
                    else:
                        src, tgt = tr.get(lang, ""), tr.get("en", "")
                        prompt = f"Translate to en:\n{src}"
                    if not src or not tgt:
                        continue
                    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": tgt}]
                    try:
                        text = _render(tok, msgs)
                    except Exception:
                        continue
                    records.append({"text": text, "category": "multilingual", "language": lang, "source": f"Helsinki-NLP/opus-100/{prefix}", "license": "cc-by-4.0"})
            except Exception as e:
                log(f"multilingual: opus {prefix} skipped ({e})")
    log(f"multilingual: {len(records)} records")
    return records


def _stage_long_context(tok, local: Path, hf_cache: Path, seed: int, log=print) -> list[dict]:
    """Coherent long documents: PG19 books, local repos, and agent transcripts."""
    records = []
    try:
        from datasets import load_dataset

        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        for i, rec in enumerate(ds):
            if i >= 400:
                break
            text = rec.get("text", "")
            if len(text) < 20000:
                continue
            records.append({"text": text, "category": "long_context", "language": "en", "source": "emozilla/pg19", "license": "public-domain"})
            if len(records) >= 160:
                break
    except Exception as e:
        log(f"long_context: pg19 skipped ({e})")
    # A local repo as a coherent document: concatenate the files of one directory.
    for root in (Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp/src"), local / "hi2"):
        if not root.exists():
            continue
        files = _iter_local_files(root, (".py", ".c", ".cpp", ".h", ".hpp"), limit=20)
        text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
        if len(text) >= 20000:
            lic = "mit" if "llama.cpp" in str(root) else "apache-2.0"
            records.append({"text": text, "category": "long_context", "language": "en", "source": str(root), "license": lic})
    # Agent transcripts: one long multi-turn session per system prompt (eight tasks), so a
    # transcript can fill an 8k window and is a coherent session, not one short exchange.
    prompts = local / "tools/agentic_bench/system_prompts"
    tasks = local / "tools/agentic_bench/tasks"
    if prompts.exists() and tasks.exists():
        for sp in sorted(prompts.glob("*.md"))[:2]:
            sys_text = sp.read_text(encoding="utf-8", errors="ignore")[:6000]
            msgs = [{"role": "system", "content": sys_text}]
            for tp in sorted(tasks.glob("*.md"))[:8]:
                task = tp.read_text(encoding="utf-8", errors="ignore")[:6000]
                msgs.append({"role": "user", "content": task})
                msgs.append(
                    {
                        "role": "assistant",
                        "content": "I will inspect the repository, make the smallest fix, and run the tests.",
                    }
                )
            try:
                text = _render(tok, msgs)
            except Exception:
                continue
            records.append({"text": text, "category": "long_context", "language": "en", "source": f"agentic_bench/{sp.name}", "license": "apache-2.0"})
    log(f"long_context: {len(records)} records")
    return records


def _stage_math(tok, hf_cache: Path, seed: int, log=print) -> list[dict]:
    records = []
    gsm = sorted(hf_cache.glob("datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"))
    if gsm:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(gsm[-1]).to_pylist()
            rng = random.Random(f"{seed}:gsm8k")
            rng.shuffle(table)
            for rec in table[:1500]:
                msgs = [{"role": "user", "content": rec.get("question", "")}, {"role": "assistant", "content": rec.get("answer", "")}]
                try:
                    text = _render(tok, msgs)
                except Exception:
                    continue
                records.append({"text": text, "category": "math_reasoning", "language": "en", "source": "openai/gsm8k", "license": "mit"})
        except Exception as e:
            log(f"math: gsm8k skipped ({e})")
    try:
        from datasets import load_dataset

        ds = load_dataset("open-r1/codeforces", split="train", streaming=True)
        for i, rec in enumerate(ds):
            if i >= 120:
                break
            text = rec.get("description", "") or rec.get("statement", "")
            if not text or len(text) < 200:
                continue
            records.append({"text": text, "category": "math_reasoning", "language": "en", "source": "open-r1/codeforces", "license": "apache-2.0"})
    except Exception as e:
        log(f"math: codeforces skipped ({e})")
    log(f"math_reasoning: {len(records)} records")
    return records


def _stage_general(tok, hf_cache: Path, seed: int, log=print) -> list[dict]:
    records = []
    for repo, pat, lic in (
        ("timdettmers/openassistant-guanaco", "datasets--timdettmers--openassistant-guanaco/snapshots/*/openassistant_best_replies_train.jsonl", "apache-2.0"),
        ("WizardLM/WizardLM_evol_instruct_70k", "datasets--WizardLM--WizardLM_evol_instruct_70k/snapshots/*/alpaca_evol_instruct_70k.json", "mit"),
    ):
        found = sorted(hf_cache.glob(pat))
        if not found:
            continue
        try:
            if found[-1].suffix == ".jsonl":
                data = _read_jsonl(found[-1])
            else:
                data = json.loads(found[-1].read_text())
            rng = random.Random(f"{seed}:{repo}")
            rng.shuffle(data)
            for rec in data[:200]:
                if "text" in rec:
                    msgs = _chat_messages(rec)
                else:
                    msgs = [{"role": "user", "content": rec.get("instruction", "")}, {"role": "assistant", "content": rec.get("output", "")}]
                if not msgs:
                    continue
                try:
                    text = _render(tok, msgs)
                except Exception:
                    continue
                records.append({"text": text, "category": "general_chat", "language": "en", "source": repo, "license": lic})
        except Exception as e:
            log(f"general: {repo} skipped ({e})")
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for i, rec in enumerate(ds):
            if i >= 200:
                break
            msgs = _chat_messages(rec)
            if not msgs:
                continue
            try:
                text = _render(tok, msgs)
            except Exception:
                continue
            records.append({"text": text, "category": "general_chat", "language": "en", "source": "HuggingFaceH4/ultrachat_200k", "license": "mit"})
    except Exception as e:
        log(f"general: ultrachat skipped ({e})")
    log(f"general_chat: {len(records)} records")
    return records


def stage(out: Path, tok, local: Path, hf_cache: Path, seed: int, log=print) -> dict:
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed, "sources": {}}
    for name, fn in (
        ("agentic", lambda: _stage_agentic(tok, local, hf_cache, seed, log)),
        ("code", lambda: _stage_code(tok, local, hf_cache, seed, log)),
        ("multilingual", lambda: _stage_multilingual(tok, seed, log)),
        ("long_context", lambda: _stage_long_context(tok, local, hf_cache, seed, log)),
        ("math_reasoning", lambda: _stage_math(tok, hf_cache, seed, log)),
        ("general_chat", lambda: _stage_general(tok, hf_cache, seed, log)),
    ):
        recs = fn()
        if not recs:
            raise SystemExit(f"{name}: nothing staged")
        target = src / f"{name}.jsonl"
        atomic_write(target, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        report["sources"][name] = {
            "records": len(recs),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "sources": sorted({r["source"] for r in recs}),
            "licenses": sorted({r["license"] for r in recs}),
            "languages": sorted({r["language"] for r in recs}),
        }
        log(f"staged {name}: {len(recs)} records, {target.stat().st_size} bytes")
    atomic_write(src / "staging.json", json.dumps(report, indent=2, sort_keys=True))
    return report


# ------------------------------------------------------------------------------- build

def _dedupe(records: list[dict], log=print) -> tuple[list[dict], dict]:
    """Exact hash dedupe, then near-duplicate shingle dedupe within each category."""
    seen: dict[str, int] = {}
    exact_kept = []
    for r in records:
        h = sha256_text(r["text"])
        if h in seen:
            continue
        seen[h] = len(exact_kept)
        exact_kept.append(r)
    by_cat = defaultdict(list)
    for r in exact_kept:
        by_cat[r["category"]].append(r)
    out = []
    near_dropped = 0
    for cat, recs in by_cat.items():
        keep = near_unique([r["text"] for r in recs])
        near_dropped += len(recs) - len(keep)
        out.extend(recs[i] for i in keep)
    return out, {"exact_dropped": len(records) - len(exact_kept), "near_dropped": near_dropped}


def _split(records: list[dict], seed: int, holdout_share: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split by document hash: every document goes to one split. ``MIN_HELDOUT_DOCS`` is
    applied per (category, language) when the group has the documents, so no language is
    held out with only the accidents of a category-wide hash order; the calibration keeps
    at least ``MIN_CALIB_DOCS`` documents of every group. The category floor is a gate."""
    by_group = defaultdict(list)
    for r in records:
        by_group[(r["category"], r["language"])].append(r)
    cal, held = [], []
    for recs in by_group.values():
        order = sorted(range(len(recs)), key=lambda i: sha256_text(recs[i]["text"]))
        n_held = min(
            max(MIN_HELDOUT_DOCS, int(len(order) * holdout_share)),
            max(0, len(order) - MIN_CALIB_DOCS),
        )
        held_idx = set(order[:n_held])
        for i, r in enumerate(recs):
            (held if i in held_idx else cal).append(r)
    return cal, held


def _windowize(records: list[dict], tok, lengths: dict[str, tuple[int, ...]], log=print) -> tuple[dict, dict]:
    """Cut windows per category and length. A long-context window is a contiguous slice of
    one document (or a concatenation of files from one repo); a short window is a
    concatenation of documents until the length is reached."""
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["category"], r["language"])].append(r)
    out = {}
    meta = {}
    for (cat, lang), recs in by_key.items():
        want = lengths.get(cat, (SHORT_LENGTH,))
        keys = {}
        for length in want:
            key = f"{cat}__{lang}__{length}"
            # the key exists even with no window, so its gate fails instead of vanishing
            out.setdefault(key, [])
            meta.setdefault(key, [])
            keys[length] = key
        if cat == "long_context":
            # one contiguous slice per document, encoded once for every length
            for r in recs:
                text = r["text"]
                sha = sha256_text(text)
                # only the first 32k tokens can become a window: encoding a 1M-token book
                # to slice its first 32k is minutes of tokenizer for nothing
                ids = encode(tok, text[:200_000])
                for length in want:
                    key = keys[length]
                    if len(ids) < length or len(out[key]) >= 32:
                        continue
                    out[key].append(ids[:length])
                    meta[key].append(
                        {
                            "category": cat,
                            "language": lang,
                            "length": length,
                            "documents": [sha],
                            "sources": [r["source"]],
                            "licenses": [r["license"]],
                        }
                    )
        else:
            # concatenate documents until the window is full, then slide; every window
            # records the documents it is made of
            for length in want:
                key = keys[length]
                label = {"category": cat, "language": lang, "length": length}
                ids = np.zeros(0, dtype=np.int32)
                spans = []  # (end offset, source, license, sha256 of the document)
                for r in recs:
                    text = r["text"]
                    more = encode(tok, "\n\n" + text)
                    start = len(ids)
                    ids = np.concatenate([ids, more]) if len(ids) else more
                    spans.append((start + len(more), r["source"], r["license"], sha256_text(text)))
                    while len(ids) >= length:
                        contributing = [s for s in spans if s[0] > 0]
                        out[key].append(ids[:length])
                        meta[key].append(
                            {
                                **label,
                                "documents": [s[3] for s in contributing],
                                "sources": sorted({s[1] for s in contributing}),
                                "licenses": sorted({s[2] for s in contributing}),
                            }
                        )
                        ids = ids[length:]
                        spans = [(e - length, src, lic, sha) for e, src, lic, sha in spans if e > length]
                    if len(out[key]) >= 32:
                        break
        for length in want:
            log(f"windows {cat}/{lang}/{length}: {len(out[keys[length]])}")
    return out, meta


def build(out: Path, tok, seed: int = 0, holdout_share: float = 0.2, log=print) -> dict:
    src = out / "src"
    staging = json.loads((src / "staging.json").read_text())
    records = []
    for name in staging["sources"]:
        records.extend(_read_jsonl(src / f"{name}.jsonl"))
    deduped, dedupe_report = _dedupe(records, log)
    cal, held = _split(deduped, seed, holdout_share)
    lengths = {"long_context": LONG_LENGTHS}
    cal_w, cal_meta = _windowize(cal, tok, lengths, log)
    held_w, held_meta = _windowize(held, tok, lengths, log)
    gates = {}
    # A held-out window that shares more than SHINGLE_LIMIT of its shingles with the
    # calibration of its category is a leak (a duplicated or contained document the
    # sampled text dedupe did not catch): drop the window and record the count.
    overlap = {}
    for key in list(held_w):
        if not held_w[key]:
            overlap[key] = {"mean": None, "max": None, "p99": None, "dropped": 0}
            gate(gates, f"shingle_overlap:{key}", False, "no held-out window", hard=False)
            continue
        cat = key.split("__")[0]
        cal_ids = np.concatenate(
            [w for k, arrs in cal_w.items() if k.split("__")[0] == cat for w in arrs],
            axis=0,
        ) if any(k.split("__")[0] == cat for k in cal_w) else np.zeros(0, dtype=np.int32)
        frac = overlap_fractions(np.stack(held_w[key]), cal_ids) if len(cal_ids) else np.ones(len(held_w[key]))
        keep = frac < SHINGLE_LIMIT
        kept = frac[keep]
        overlap[key] = {
            "mean": float(kept.mean()) if len(kept) else None,
            "max": float(kept.max()) if len(kept) else None,
            "p99": float(np.quantile(kept, 0.99)) if len(kept) else None,
            "leaked_max": float(frac.max()),
            "dropped": int((~keep).sum()),
        }
        if not keep.all():
            log(f"held-out {key}: {int((~keep).sum())} window(s) share the calibration and are dropped")
            held_w[key] = [w for w, ok in zip(held_w[key], keep) if ok]
            held_meta[key] = [m for m, ok in zip(held_meta[key], keep) if ok]
        gate(
            gates,
            f"shingle_overlap:{key}",
            overlap[key]["dropped"] == 0,
            f"kept max {overlap[key]['max']:.3f}, leaked {overlap[key]['leaked_max']:.3f}, dropped {overlap[key]['dropped']}",
            hard=False,
        )
    # require at least MIN_HELDOUT_WINDOWS per category in aggregate (a language may be
    # small, the category may not), and additionally per long-context length
    held_cat = defaultdict(int)
    for key, arr in held_w.items():
        held_cat[key.split("__")[0]] += len(arr)
    cal_cat = defaultdict(int)
    for key, arr in cal_w.items():
        cal_cat[key.split("__")[0]] += len(arr)
    # every staged category is gated even when the split leaves one out entirely
    for cat in sorted({r["category"] for r in deduped}):
        n = held_cat.get(cat, 0)
        gate(gates, f"heldout_windows:{cat}", n >= MIN_HELDOUT_WINDOWS, f"{n} >= {MIN_HELDOUT_WINDOWS}")
        n = cal_cat.get(cat, 0)
        gate(gates, f"calib_windows:{cat}", n >= MIN_CALIB_WINDOWS, f"{n} >= {MIN_CALIB_WINDOWS}")
    for lang in sorted({r["language"] for r in held if r["category"] == "long_context"}):
        for length in LONG_LENGTHS:
            key = f"long_context__{lang}__{length}"
            n = len(held_w.get(key, []))
            gate(gates, f"heldout_windows:{key}", n >= MIN_HELDOUT_WINDOWS, f"{n} >= {MIN_HELDOUT_WINDOWS}")
    # write arrays and manifest
    files = {}
    for split, wmap, mmap in (("calib", cal_w, cal_meta), ("heldout", held_w, held_meta)):
        for key, arr in wmap.items():
            if not arr:  # the key is in the manifest with zero windows and a failed gate
                continue
            a = np.stack(arr).astype(np.int32)
            fname = f"{split}_{key}_ids.npy"
            atomic_npy(out / fname, a)
            cat, lang, _length = key.split("__")
            files[fname] = {"bytes": (out / fname).stat().st_size, "sha256": sha256_file(out / fname), "windows": len(a), "length": int(a.shape[1]), "category": cat, "language": lang}
            mmap[key] = mmap.get(key, [])
    manifest = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": str(tok.name_or_path) if hasattr(tok, "name_or_path") else "tokenizer",
        "staging": staging,
        "dedupe": dedupe_report,
        "lengths": {"short": SHORT_LENGTH, "long": list(LONG_LENGTHS)},
        "splits": {
            "calib": {k: {"windows": len(v), "length": int(k.split("__")[2])} for k, v in cal_w.items()},
            "heldout": {k: {"windows": len(v), "length": int(k.split("__")[2])} for k, v in held_w.items()},
        },
        "files": files,
        "overlap": overlap,
        "gates": gates,
        "meta": {"calib": cal_meta, "heldout": held_meta},
    }
    manifest["ok"] = all(g["ok"] for g in gates.values() if g["hard"])
    atomic_write(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def ood(out: Path, tok, log=print) -> dict:
    """A separate out-of-distribution set: English Wikipedia natural text, a source that
    appears in no calibration and no held-out window. One article per window, 16 windows
    of ``SHORT_LENGTH`` tokens, added to the manifest under the ``ood`` split."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", name="20231101.en", split="train", streaming=True)
    windows, meta = [], []
    for rec in ds:
        text = rec.get("text", "")
        if len(text) < 4000:
            continue
        ids = encode(tok, text)
        if len(ids) < SHORT_LENGTH:
            continue
        windows.append(ids[:SHORT_LENGTH])
        meta.append(
            {
                "category": "ood_wiki",
                "language": "en",
                "length": SHORT_LENGTH,
                "documents": [sha256_text(text)],
                "sources": ["wikimedia/wikipedia/20231101.en"],
                "licenses": ["cc-by-sa-4.0"],
            }
        )
        if len(windows) >= MIN_HELDOUT_WINDOWS:
            break
    if len(windows) < MIN_HELDOUT_WINDOWS:
        raise SystemExit(f"ood: {len(windows)} windows, need {MIN_HELDOUT_WINDOWS}")
    key = f"wiki__en__{SHORT_LENGTH}"  # under the `ood` split prefix
    name = f"ood_{key}_ids.npy"
    a = np.stack(windows).astype(np.int32)
    atomic_npy(out / name, a)
    mpath = out / "manifest.json"
    m = json.loads(mpath.read_text())
    m.setdefault("files", {})[name] = {
        "bytes": (out / name).stat().st_size,
        "sha256": sha256_file(out / name),
        "windows": len(a),
        "length": int(a.shape[1]),
        "category": "ood_wiki",
        "language": "en",
    }
    m.setdefault("splits", {}).setdefault("ood", {})[key] = {
        "windows": len(a),
        "length": SHORT_LENGTH,
    }
    m.setdefault("meta", {}).setdefault("ood", {})[key] = meta
    m["ood"] = {
        "sources": ["wikimedia/wikipedia/20231101.en"],
        "licenses": ["cc-by-sa-4.0"],
        "reason": "English Wikipedia natural text is in no staging source of the other splits",
    }
    atomic_write(mpath, json.dumps(m, indent=2, sort_keys=True))
    log(f"ood: {len(windows)} windows of {SHORT_LENGTH} tokens in {name}")
    return m


def verify(out: Path, tok=None, production: bool = False) -> list[str]:
    mpath = out / "manifest.json"
    if not mpath.exists():
        return [f"{mpath}: missing"]
    m = json.loads(mpath.read_text())
    bad = []
    if production and m.get("schema") != SCHEMA:
        bad.append(f"schema {m.get('schema')!r} != {SCHEMA!r}")
    if not m.get("ok"):
        bad.append("the build recorded a failed gate")
    for name, info in m.get("files", {}).items():
        p = out / name
        if not p.exists():
            bad.append(f"{name}: missing")
        elif p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
            bad.append(f"{name}: sha256/size differ")
    if tok is not None:
        for key, info in m.get("splits", {}).get("heldout", {}).items():
            fname = f"heldout_{key}_ids.npy"
            p = out / fname
            if p.exists():
                a = np.load(p)
                if a.shape[1] != info["length"]:
                    bad.append(f"{fname}: length {a.shape[1]} != {info['length']}")
    return bad


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("stage", "build", "ood", "verify"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--local", type=Path, default=Path("/home/chenco_adm/w4a8_runs/inkling-hi2/tree"))
    ap.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-share", type=float, default=0.2)
    ap.add_argument("--production", action="store_true")
    return ap


def _exit_after_streaming() -> None:
    """Leave without interpreter finalization. In the build box's vLLM venv, datasets +
    pyarrow abort in ``PyGILState_Release`` at shutdown after a streaming read (exit 134),
    after every artefact has been written; a clean zero exit is what the caller needs."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "stage":
        stage(a.out, tok, a.local, a.hf_cache, a.seed)
        _exit_after_streaming()
        return 0
    if a.command == "build":
        m = build(a.out, tok, a.seed, a.holdout_share)
        print(json.dumps({"ok": m["ok"], "gates": m["gates"]}, indent=2))
        return 0 if m["ok"] else 1
    if a.command == "ood":
        m = ood(a.out, tok)
        print(json.dumps({"ood": m["splits"]["ood"]}, indent=2))
        _exit_after_streaming()
        return 0
    bad = verify(a.out, tok, a.production)
    print(json.dumps({"ok": not bad, "problems": bad}, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
