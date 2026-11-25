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
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
import tomllib
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
AGENTIC_HELDOUT_WINDOWS = 60
AGENTIC_CALIB_WINDOWS = 60
# The languages the local trees supply enough held-out documents for. JavaScript is absent:
# the trees supply 13 small .js/.mjs/.cjs documents, so its count is recorded but no floor or
# language-level claim is made.
SMALL_CODE_LANGUAGES = ("shell", "sql", "typescript")
SMALL_CODE_WINDOWS = 16
MAX_WINDOWS = {"agentic": 64}
SHINGLE = 32
SHINGLE_LIMIT = 0.5
SPECIAL_SHARE = (0.002, 0.20)
LANGUAGES = ("zh", "ja", "ko", "ru", "ar", "hi", "es", "fr", "de", "pt", "tr", "vi")
CODE_LANGUAGES = ("python", "c", "cpp", "rust", "typescript", "javascript", "shell", "sql", "yaml", "json")
# The SPDX ids a staged document may carry. ``gpl-2.0`` is kept only for the tagged
# Linux Rust text; everything else must be permissive, and an unknown id is refused.
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "public-domain",
    "unlicense",
    "zlib",
}
LICENSE_CLASSES = ("permissive", "gpl2")
PRODUCTION = {
    "min_heldout_windows": MIN_HELDOUT_WINDOWS,
    "min_calib_windows": MIN_CALIB_WINDOWS,
    "agentic_heldout_windows": AGENTIC_HELDOUT_WINDOWS,
    "agentic_calib_windows": AGENTIC_CALIB_WINDOWS,
    "small_code_windows": SMALL_CODE_WINDOWS,
    "shingle_limit": SHINGLE_LIMIT,
    "long_lengths": list(LONG_LENGTHS),
    "permissive_licenses": sorted(PERMISSIVE_LICENSES),
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


def classify_license(license: str) -> str:
    """``permissive`` or ``gpl2``; anything not in the allowlist is refused."""
    lic = str(license).strip().lower()
    if lic in ("gpl-2.0", "gpl2"):
        return "gpl2"
    parts = [p.strip() for p in re.split(r"\s+or\s+|/", lic) if p.strip()]
    unknown = [p for p in parts if p not in PERMISSIVE_LICENSES]
    if not parts or unknown:
        raise SystemExit(f"unknown licence {license!r}: {unknown or parts}")
    return "permissive"


def validate_licenses(records: list[dict], where: str = "stage") -> None:
    for r in records:
        try:
            r["license_class"] = classify_license(r.get("license", ""))
        except SystemExit as e:
            raise SystemExit(f"{where}: {r.get('source', '?')}: {e}") from None


# ----------------------------------------------------------------------- source loading

def _read_jsonl(path: Path) -> list[dict]:
    # Split on "\n", not str.splitlines: U+2028/U+2029 inside a JSON string (they occur in
    # source files) are line breaks to splitlines and would cut the record in half.
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


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


_TOOL_CALL_TEXT = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _normalise_tool_calls(raw, start: int = 0) -> list[dict]:
    """The template wants ``function.arguments`` as an object, not a JSON string."""
    out = []
    for i, tc in enumerate(raw or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        out.append(
            {
                "id": tc.get("id") or f"call_{start + i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return out


def _chat_messages(record: dict) -> list[dict] | None:
    turns = record.get("messages") or record.get("conversations") or record.get("chat")
    if not turns:
        return None
    out = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        role = t.get("role") or t.get("from") or "user"
        role = ROLES.get(role, role)
        content = t.get("content") if "content" in t else t.get("value")
        calls = _normalise_tool_calls(t.get("tool_calls"), start=len(out))
        if role == "assistant" and isinstance(content, str):
            found = _TOOL_CALL_TEXT.findall(content)
            if found:
                parsed = []
                for blob in found:
                    try:
                        obj = json.loads(blob)
                    except ValueError:
                        continue
                    parsed.append(obj)
                calls = calls + _normalise_tool_calls(parsed, start=len(out) + len(calls))
                content = _TOOL_CALL_TEXT.sub("", content).strip()
        if not isinstance(content, str):
            if role == "assistant" and calls:
                content = ""
            else:
                return None
        msg = {"role": role, "content": content}
        if calls:
            msg["tool_calls"] = calls
        if isinstance(t.get("tool_call_id"), str) and t["tool_call_id"]:
            msg["tool_call_id"] = t["tool_call_id"]
        if isinstance(t.get("name"), str) and t["name"]:
            msg["name"] = t["name"]
        out.append(msg)
    return out if out and out[0]["role"] in ("system", "user") else None


def _render(tok, messages: list[dict], tools: list | None = None) -> str:
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the chat template returned no text")
    return text


_TOOLACE_CALL = re.compile(r"^\s*\[([^\]\(]+)\((.*)\)\]\s*$", re.S)


def _toolace_arguments(body: str) -> dict | None:
    """``Name(a=1, b="x")`` -> ``{"a": 1, "b": "x"}``; only keyword arguments."""
    try:
        call = ast.parse(f"f({body})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(call, ast.Call) or call.args or not call.keywords:
        return None
    out = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return out


def _toolace_messages(rec: dict) -> tuple[list[dict], list | None]:
    """ToolACE stores calls as ``[Name(k=v)]`` text and results as ``tool`` turns; convert
    them to the template's ``tool_calls`` objects and link each result by id."""
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
    pending: list[str] = []
    for turn in rec.get("conversations", []):
        role = ROLES.get(turn.get("from"), turn.get("from"))
        value = turn.get("value", "")
        if not isinstance(value, str) or not value:
            return [], None
        if role == "assistant":
            m = _TOOLACE_CALL.match(value)
            args = _toolace_arguments(m.group(2)) if m else None
            if m and args is not None:
                cid = f"call_{len(msgs)}"
                pending.append(cid)
                msgs.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": cid, "type": "function", "function": {"name": m.group(1).strip(), "arguments": args}}
                        ],
                    }
                )
                continue
        if role == "tool":
            msg = {"role": "tool", "content": value}
            if pending:
                msg["tool_call_id"] = pending.pop(0)
            msgs.append(msg)
            continue
        msgs.append({"role": role, "content": value})
    return msgs, tools


def _code_agent_traces(tok, local: Path, log=print) -> list[dict]:
    """Code-agent traces with real repository tool results: the system prompt and task are
    the agentic_bench ones, ``read_file``/``list_dir`` return real bytes of this repository,
    and the calls are the template's native JSON-argument turns."""
    prompts = local / "tools/agentic_bench/system_prompts"
    tasks = local / "tools/agentic_bench/tasks"
    ctx_path = local / "hi2/looped.py"
    if not (prompts.exists() and tasks.exists() and ctx_path.exists()):
        return []
    ctx = ctx_path.read_text(encoding="utf-8", errors="ignore")[:4000]
    listing = "\n".join(sorted(p.name for p in (local / "hi2").glob("*.py"))[:60])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 file from the repository.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List the files of a directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
    ]
    out = []
    for sp in sorted(prompts.glob("*.md"))[:3]:
        sys_text = sp.read_text(encoding="utf-8", errors="ignore")[:8000]
        for tp in sorted(tasks.glob("*.md"))[:8]:
            task = tp.read_text(encoding="utf-8", errors="ignore")[:6000]
            msgs = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": task + "\n\nThe repository root is /repo."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_read", "type": "function", "function": {"name": "read_file", "arguments": {"path": "hi2/looped.py"}}}
                    ],
                },
                {"role": "tool", "content": ctx, "tool_call_id": "call_read"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_ls", "type": "function", "function": {"name": "list_dir", "arguments": {"path": "hi2"}}}
                    ],
                },
                {"role": "tool", "content": listing, "tool_call_id": "call_ls"},
                {"role": "assistant", "content": "I have the relevant code; I will make the change and run the tests."},
            ]
            try:
                text = _render(tok, msgs, tools)
            except Exception as e:
                log(f"agentic: code trace {sp.name}/{tp.name} skipped ({e})")
                continue
            out.append(
                {
                    "text": text,
                    # The 24 traces share three system prompts; near-duplicate on the task
                    # body so the shared prompt does not collapse them to one document.
                    "dedupe_text": f"{sp.name}\n{tp.name}\n{task}",
                    "category": "agentic",
                    "language": "en",
                    "source": f"agentic_bench/{sp.name}/{tp.name}+repo",
                    "license": "apache-2.0",
                }
            )
    return out


def _stage_agentic(tok, local: Path, hf_cache: Path, seed: int, log=print) -> list[dict]:
    """Native tool-call turns with JSON-object arguments and tool results fed back:
    ToolACE (multi-turn calls parsed from its text notation, tools as tool_declare), Hermes
    (``<tool_call>`` JSON parsed to native turns), and code-agent traces built from the real
    agentic_bench prompts/tasks and real repository tool results."""
    records = []
    toolace = sorted(hf_cache.glob("datasets--Team-ACE--ToolACE/snapshots/*/data.json"))
    if toolace:
        data = json.loads(toolace[-1].read_text())
        rng = random.Random(f"{seed}:toolace")
        rng.shuffle(data)
        for rec in data[:12000]:
            msgs, tools = _toolace_messages(rec)
            if not msgs or not any(m.get("tool_calls") for m in msgs):
                continue
            try:
                text = _render(tok, msgs, tools)
            except Exception:
                continue
            records.append(
                {"text": text, "category": "agentic", "language": "en", "source": "Team-ACE/ToolACE", "license": "apache-2.0"}
            )
            if len(records) >= 3000:
                break
    records.extend(_code_agent_traces(tok, local, log))
    try:
        from datasets import load_dataset

        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
        added = 0
        for i, rec in enumerate(ds):
            if i >= 2000 or added >= 800:
                break
            msgs = _chat_messages(rec)
            if not msgs or not any(m.get("tool_calls") for m in msgs):
                continue
            try:
                text = _render(tok, msgs)
            except Exception:
                continue
            records.append(
                {"text": text, "category": "agentic", "language": "en", "source": "NousResearch/hermes-function-calling-v1", "license": "apache-2.0"}
            )
            added += 1
    except Exception as e:
        log(f"agentic: hermes skipped ({e})")
    log(f"agentic: {len(records)} records")
    return records


def _cargo_rust_records(seed: int, log=print, limit: int = 400) -> list[dict]:
    """Permissively licensed Rust from the local crates.io registry. A crate is used only
    when every part of its declared licence is in the allowlist; the source records the
    crate and version for attribution."""
    registries = sorted((Path.home() / ".cargo/registry/src").glob("index.crates.io-*"))
    crates = []
    for root in registries:
        for d in root.iterdir():
            tom = d / "Cargo.toml"
            if not d.is_dir() or not tom.exists():
                continue
            try:
                pkg = tomllib.loads(tom.read_text(encoding="utf-8", errors="ignore")).get("package", {})
            except Exception:
                continue
            lic = pkg.get("license", "")
            try:
                classify_license(lic)
            except SystemExit:
                continue
            crates.append((d, lic))
    rng = random.Random(f"{seed}:cargo")
    rng.shuffle(crates)
    records = []
    for d, lic in crates:
        files = sorted(p for p in d.rglob("*.rs") if p.is_file() and p.stat().st_size < 200_000)
        rng.shuffle(files)
        for p in files[:2]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(text) < 200:
                continue
            records.append(
                {
                    "text": text,
                    "category": "code",
                    "language": "rust",
                    "source": f"crates.io/{d.name}/{p.relative_to(d)}",
                    "license": lic,
                }
            )
            if len(records) >= limit:
                log(f"code: {len(records)} permissive rust records from {len(crates)} crates")
                return records
    log(f"code: {len(records)} permissive rust records from {len(crates)} crates")
    return records


def _sql_public_records(seed: int, hf_cache: Path, log=print, docs: int = 80) -> list[dict]:
    """SQL a permissive public corpus can supply: ``sql-create-context`` (CC-BY-4.0),
    grouped into documents so each is long enough to carry windows. The box is offline: use
    the cached JSON file (``sql_create_context_v4.json``) when it is there, and fall back to
    the streaming loader when it is not."""
    cached = sorted(hf_cache.glob("datasets--b-mc2--sql-create-context/**/sql_create_context_v4.json"))
    if not cached and (hf_cache / "sql_create_context_v4.json").exists():
        cached = [hf_cache / "sql_create_context_v4.json"]
    try:
        if cached:
            rows = iter(json.loads(cached[-1].read_text()))
            log(f"code: sql-create-context from {cached[-1]}")
        else:
            from datasets import load_dataset

            rows = iter(load_dataset("b-mc2/sql-create-context", split="train", streaming=True))
    except Exception as e:
        log(f"code: sql-create-context skipped ({e})")
        return []
    records, buf = [], []
    for i, rec in enumerate(rows):
        if i >= 40_000 or len(records) >= docs:
            break
        question = str(rec.get("question", "")).strip()
        context = str(rec.get("context", "")).strip()
        answer = str(rec.get("answer", "")).strip()
        if not context or not answer:
            continue
        buf.append(f"-- {question}\n{context}\n{answer}")
        if sum(len(x) for x in buf) >= 16_000:
            records.append(
                {
                    "text": "\n\n".join(buf),
                    "category": "code",
                    "language": "sql",
                    "source": "b-mc2/sql-create-context",
                    "license": "cc-by-4.0",
                }
            )
            buf = []
    if buf and len(records) < docs:
        records.append(
            {
                "text": "\n\n".join(buf),
                "category": "code",
                "language": "sql",
                "source": "b-mc2/sql-create-context",
                "license": "cc-by-4.0",
            }
        )
    log(f"code: {len(records)} sql documents from sql-create-context")
    return records


def _stage_code(tok, local: Path, hf_cache: Path, seed: int, log=print) -> list[dict]:
    """Code from local project trees and public permissive datasets. The code roots are
    permissive trees (MIT/Apache-2.0); the Linux Rust tree is GPL-2.0 and is tagged so the
    build can drop it with ``--exclude-license gpl2``."""
    records = []
    llama = Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp")
    vllm = Path("/home/chenco_adm/w4a8_runs/leapquant/tree-inkling2")
    roots = {
        "python": [local / "hi2", local / "experiments"],
        "c": [llama],
        "cpp": [llama],
        "rust": [Path("/home/chenco_adm/rpmbuild/BUILD/linux-6.19.12/drivers")],
        "typescript": [llama / "tools/ui"],
        "javascript": [llama, vllm],
        "shell": [llama, vllm, local / "experiments", local / "bench"],
        "sql": [Path("/home/chenco_adm/tensorfold-pr1694/rocm-10.1/share/rocprofiler-sdk-rocpd")],
        "yaml": [local / "experiments"],
        "json": [local / "experiments"],
    }
    suffixes = {".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript", ".svelte": "typescript", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".sh": "shell", ".bash": "shell", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json"}
    limits = {"shell": 400, "typescript": 400, "javascript": 200, "sql": 200}

    def license_of(p: Path) -> str:
        s = str(p)
        if "llama.cpp" in s:
            return "mit"
        if "linux-" in s:
            return "gpl-2.0"  # local Linux tree: public but copyleft, see docs
        if "rocprofiler" in s:
            return "mit"
        return "apache-2.0"  # this repository's own trees and the vLLM fork
    for lang, dirs in roots.items():
        found = []
        want = limits.get(lang, 60)
        for d in dirs:
            if d.exists():
                found.extend(_iter_local_files(d, tuple(k for k, v in suffixes.items() if v == lang), limit=want))
        rng = random.Random(f"{seed}:{lang}")
        rng.shuffle(found)
        for p in found[:want]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(text) < 200:
                continue
            records.append({"text": text, "category": "code", "language": lang, "source": str(p), "license": license_of(p)})
    records.extend(_cargo_rust_records(seed, log))
    records.extend(_sql_public_records(seed, hf_cache, log))
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


STAGED_CATEGORIES = (
    "agentic",
    "code",
    "multilingual",
    "long_context",
    "math_reasoning",
    "general_chat",
)


def stage(
    out: Path,
    tok,
    local: Path,
    hf_cache: Path,
    seed: int,
    categories: tuple[str, ...] | None = None,
    log=print,
) -> dict:
    """Stage every category, or only ``categories``. A partial restage keeps the
    ``staging.json`` entries of the categories it does not touch, so a full build still reads
    all six files."""
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    stagers = {
        "agentic": lambda: _stage_agentic(tok, local, hf_cache, seed, log),
        "code": lambda: _stage_code(tok, local, hf_cache, seed, log),
        "multilingual": lambda: _stage_multilingual(tok, seed, log),
        "long_context": lambda: _stage_long_context(tok, local, hf_cache, seed, log),
        "math_reasoning": lambda: _stage_math(tok, hf_cache, seed, log),
        "general_chat": lambda: _stage_general(tok, hf_cache, seed, log),
    }
    names = tuple(categories) if categories else STAGED_CATEGORIES
    report = {"seed": seed, "sources": {}}
    previous = src / "staging.json"
    if categories and previous.exists():
        report["sources"].update(json.loads(previous.read_text()).get("sources") or {})
    for name in names:
        recs = stagers[name]()
        if not recs:
            raise SystemExit(f"{name}: nothing staged")
        validate_licenses(recs, name)
        target = src / f"{name}.jsonl"
        atomic_write(target, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        report["sources"][name] = {
            "records": len(recs),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "sources": sorted({r["source"] for r in recs}),
            "licenses": sorted({r["license"] for r in recs}),
            "license_classes": sorted({r["license_class"] for r in recs}),
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
        keep = near_unique([r.get("dedupe_text") or r["text"] for r in recs])
        near_dropped += len(recs) - len(keep)
        out.extend(recs[i] for i in keep)
    return out, {"exact_dropped": len(records) - len(exact_kept), "near_dropped": near_dropped}


def _split(records: list[dict], seed: int, holdout_share: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split by document: every document goes to one split, size-stratified within each
    (category, language) group. ``MIN_HELDOUT_DOCS`` is applied per group when it has the
    documents, so no language is held out with only the accidents of a category-wide order;
    the calibration keeps at least ``MIN_CALIB_DOCS`` documents of every group. The category
    floor is a gate."""
    by_group = defaultdict(list)
    for r in records:
        by_group[(r["category"], r["language"])].append(r)
    cal, held = [], []
    for recs in by_group.values():
        # Largest first, hash as the tiebreak. Both sides then get a comparable size mix:
        # a pure hash order can put every large document on one side (the SQL group did),
        # leaving the other side too short to make a single window.
        order = sorted(
            range(len(recs)),
            key=lambda i: (-len(recs[i]["text"]), sha256_text(recs[i]["text"])),
        )
        n_held = min(
            max(MIN_HELDOUT_DOCS, int(len(order) * holdout_share)),
            max(0, len(order) - MIN_CALIB_DOCS),
        )
        held_idx, cal_idx = [], []
        for i in order:
            if len(held_idx) == n_held:
                cal_idx.append(i)
            elif len(cal_idx) == len(order) - n_held:
                held_idx.append(i)
            elif len(held_idx) * (len(order) - n_held) <= len(cal_idx) * n_held:
                held_idx.append(i)
            else:
                cal_idx.append(i)
        held_set = set(held_idx)
        for i, r in enumerate(recs):
            (held if i in held_set else cal).append(r)
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
        max_windows = MAX_WINDOWS.get(cat, 32)
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
                    if len(ids) < length or len(out[key]) >= max_windows:
                        continue
                    out[key].append(ids[:length])
                    meta[key].append(
                        {
                            "category": cat,
                            "language": lang,
                            "length": length,
                            "documents": [sha],
                            "document_records": [{"id": sha, "source": r["source"], "license": r["license"]}],
                            "sources": [r["source"]],
                            "licenses": [r["license"]],
                            "license_classes": [r.get("license_class") or classify_license(r["license"])],
                        }
                    )
        else:
            # concatenate documents until the window is full, then slide; every window
            # records the documents it is made of
            for length in want:
                key = keys[length]
                label = {"category": cat, "language": lang, "length": length}
                ids = np.zeros(0, dtype=np.int32)
                spans = []  # (end offset, source, license, license class, sha256 of the document)
                for r in recs:
                    text = r["text"]
                    more = encode(tok, "\n\n" + text)
                    start = len(ids)
                    ids = np.concatenate([ids, more]) if len(ids) else more
                    cls = r.get("license_class") or classify_license(r["license"])
                    spans.append((start + len(more), r["source"], r["license"], cls, sha256_text(text)))
                    while len(ids) >= length and len(out[key]) < max_windows:
                        contributing = [s for s in spans if s[0] > 0]
                        out[key].append(ids[:length])
                        meta[key].append(
                            {
                                **label,
                                "documents": [s[4] for s in contributing],
                                "document_records": [
                                    {"id": s[4], "source": s[1], "license": s[2]} for s in contributing
                                ],
                                "sources": sorted({s[1] for s in contributing}),
                                "licenses": sorted({s[2] for s in contributing}),
                                "license_classes": sorted({s[3] for s in contributing}),
                            }
                        )
                        ids = ids[length:]
                        spans = [(e - length, src, lic, cls, sha) for e, src, lic, cls, sha in spans if e > length]
                    if len(out[key]) >= max_windows:
                        break
        for length in want:
            log(f"windows {cat}/{lang}/{length}: {len(out[keys[length]])}")
    return out, meta


def build(
    out: Path,
    tok,
    seed: int = 0,
    holdout_share: float = 0.2,
    exclude_license: tuple[str, ...] = (),
    log=print,
) -> dict:
    src = out / "src"
    staging = json.loads((src / "staging.json").read_text())
    records = []
    for name in staging["sources"]:
        records.extend(_read_jsonl(src / f"{name}.jsonl"))
    validate_licenses(records, "staged")
    excluded = tuple(dict.fromkeys(exclude_license))
    unknown = set(excluded) - set(LICENSE_CLASSES)
    if unknown:
        raise SystemExit(f"--exclude-license: unknown class(es) {sorted(unknown)}")
    kept = [r for r in records if r["license_class"] not in excluded]
    excluded_records = len(records) - len(kept)
    records = kept
    deduped, dedupe_report = _dedupe(records, log)
    # The local coding-agent transcripts are first in their split, so the capped agentic
    # windows include them before ToolACE's larger volume fills the cap. The sort is stable:
    # every other record keeps the staging order.
    deduped.sort(key=lambda r: 0 if str(r.get("source", "")).startswith("agentic_bench/") else 1)
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
    # the native tool-call set and the four small code languages have their own floors
    n = held_cat.get("agentic", 0)
    gate(gates, "heldout_windows:agentic_floor", n >= AGENTIC_HELDOUT_WINDOWS, f"{n} >= {AGENTIC_HELDOUT_WINDOWS}")
    n = cal_cat.get("agentic", 0)
    gate(gates, "calib_windows:agentic_floor", n >= AGENTIC_CALIB_WINDOWS, f"{n} >= {AGENTIC_CALIB_WINDOWS}")
    for lang in SMALL_CODE_LANGUAGES:
        key = f"code__{lang}__{SHORT_LENGTH}"
        n = len(held_w.get(key, []))
        gate(gates, f"heldout_windows:{key}", n >= SMALL_CODE_WINDOWS, f"{n} >= {SMALL_CODE_WINDOWS}")
        n = len(cal_w.get(key, []))
        gate(gates, f"calib_windows:{key}", n >= SMALL_CODE_WINDOWS, f"{n} >= {SMALL_CODE_WINDOWS}")
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
        "excluded_license_classes": list(excluded),
        "excluded_records": excluded_records,
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


OOD_DOCS_ROOTS = (
    Path("/home/chenco_adm/w4a8_runs/leapquant/tree-inkling2/docs"),
    Path("/home/chenco_adm/Apps/llama.cpp/llama.cpp/docs"),
)


def _ood_from_docs(tok, roots: tuple[Path, ...], log=print) -> tuple[list, list, list, list]:
    """The offline fallback: one contiguous 2048-token window per English technical-docs
    document. No `.md` file is staged by any category (only `.py`/`.sh`/`.c`/... are), so the
    source stays out of the calibration and held-out splits."""
    windows, meta, sources, licenses = [], [], set(), set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(text) < 4000:
                continue
            ids = encode(tok, text)
            if len(ids) < SHORT_LENGTH:
                continue
            lic = "mit" if "llama.cpp" in str(p) else "apache-2.0"
            windows.append(ids[:SHORT_LENGTH])
            sources.add(str(p))
            licenses.add(lic)
            meta.append(
                {
                    "category": "ood_docs",
                    "language": "en",
                    "length": SHORT_LENGTH,
                    "documents": [sha256_text(text)],
                    "document_records": [{"id": sha256_text(text), "source": str(p), "license": lic}],
                    "sources": [str(p)],
                    "licenses": [lic],
                    "license_classes": ["permissive"],
                }
            )
            if len(windows) >= MIN_HELDOUT_WINDOWS:
                return windows, meta, sorted(sources), sorted(licenses)
    return windows, meta, sorted(sources), sorted(licenses)


def ood(out: Path, tok, docs_roots: tuple[Path, ...] | None = None, log=print) -> dict:
    """A separate out-of-distribution set: English Wikipedia natural text, a source that
    appears in no calibration and no held-out window. One article per window, 16 windows
    of ``SHORT_LENGTH`` tokens, added to the manifest under the ``ood`` split. Offline, fall
    back to English technical documentation that no staged category contains."""
    try:
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
                    "document_records": [
                        {
                            "id": sha256_text(text),
                            "source": "wikimedia/wikipedia/20231101.en",
                            "license": "cc-by-sa-4.0",
                        }
                    ],
                    "sources": ["wikimedia/wikipedia/20231101.en"],
                    "licenses": ["cc-by-sa-4.0"],
                }
            )
            if len(windows) >= MIN_HELDOUT_WINDOWS:
                break
        if len(windows) < MIN_HELDOUT_WINDOWS:
            raise SystemExit(f"ood: {len(windows)} wikipedia windows, need {MIN_HELDOUT_WINDOWS}")
        category, key_prefix = "ood_wiki", "wiki"
        sources, licenses = ["wikimedia/wikipedia/20231101.en"], ["cc-by-sa-4.0"]
        reason = "English Wikipedia natural text is in no staging source of the other splits"
    except (Exception, SystemExit) as e:
        log(f"ood: wikipedia unavailable ({e}); using local docs")
        windows, meta, sources, licenses = _ood_from_docs(tok, docs_roots or OOD_DOCS_ROOTS, log)
        if len(windows) < MIN_HELDOUT_WINDOWS:
            raise SystemExit(f"ood: {len(windows)} local-docs windows, need {MIN_HELDOUT_WINDOWS}") from None
        category, key_prefix = "ood_docs", "docs"
        reason = "English technical documentation from the vLLM/llama.cpp docs trees; no .md document is staged"
    key = f"{key_prefix}__en__{SHORT_LENGTH}"  # under the `ood` split prefix
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
        "category": category,
        "language": "en",
    }
    m.setdefault("splits", {}).setdefault("ood", {})[key] = {
        "windows": len(a),
        "length": SHORT_LENGTH,
    }
    m.setdefault("meta", {}).setdefault("ood", {})[key] = meta
    m["ood"] = {"sources": sources, "licenses": licenses, "reason": reason}
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
    excluded = set(m.get("excluded_license_classes") or [])
    for split in ("calib", "heldout", "ood"):
        for key, rows in (m.get("meta", {}).get(split) or {}).items():
            for row in rows:
                cls = set(row.get("license_classes") or [])
                if excluded & cls:
                    bad.append(f"{split}/{key}: an excluded licence class is in a window")
                    break
                for rec in row.get("document_records") or []:
                    if not rec.get("id") or not rec.get("source") or not rec.get("license"):
                        bad.append(f"{split}/{key}: incomplete document provenance")
                        break
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
    ap.add_argument(
        "--exclude-license",
        action="append",
        default=[],
        choices=LICENSE_CLASSES,
        metavar="CLASS",
        help="drop a licence class from the build (gpl2); repeatable",
    )
    ap.add_argument(
        "--category",
        action="append",
        choices=STAGED_CATEGORIES,
        help="stage only these categories; the other staging.json entries are kept",
    )
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
        stage(a.out, tok, a.local, a.hf_cache, a.seed, tuple(a.category) if a.category else None)
        _exit_after_streaming()
        return 0
    if a.category:
        raise SystemExit("--category is only meaningful for stage")
    if a.command == "build":
        m = build(a.out, tok, a.seed, a.holdout_share, tuple(a.exclude_license))
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
