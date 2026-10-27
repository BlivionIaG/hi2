# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Mixed-domain calibration text (``hi2 calib``).

PTQ reads the first tokens of the calibration file and QAD draws windows from all of it,
so the domains must be mixed through the whole file, not appended one after another. This
builds such a file from any number of sources: ``.txt`` (blank-line separated blocks),
``.jsonl`` and ``.json`` with ``text`` records or chat records (``messages`` or
``conversations``). Each source gets its weight's share of the token budget, and its
documents are interleaved by token share, so every stretch of the output holds every
domain. An optional held-out text takes a block of documents out of every source,
for ``hi2 eval``; the two texts never share a document.
"""

import json
from pathlib import Path

CHARS_PER_TOKEN = 4  # the estimate when no tokenizer is given
MIN_DOC_CHARS = 2000  # blocks of a .txt file are grouped up to this size
HOLDOUT_BLOCK = 10  # documents per held-out block


def _chat(record) -> str | None:
    turns = record.get("messages") or record.get("conversations")
    if not turns:
        return None
    lines = []
    for t in turns:
        role = t.get("role") or t.get("from") or "user"
        content = t.get("content") if "content" in t else t.get("value")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or None


def _record(record) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return record.get("text") or _chat(record)
    return None


def read_documents(path: Path) -> list[str]:
    """The documents of one source file."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".jsonl":
        records = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    elif path.suffix == ".json":
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError(f"{path}: a .json source must be a list of records")
    else:
        docs, cur = [], ""
        for block in raw.split("\n\n"):
            cur = f"{cur}\n\n{block}" if cur else block
            if len(cur) >= MIN_DOC_CHARS:
                docs.append(cur)
                cur = ""
        if cur.strip():
            docs.append(cur)
        return [d for d in docs if d.strip()]
    docs = [_record(r) for r in records]
    return [d for d in docs if d and d.strip()]


def parse_source(spec: str) -> tuple[Path, float]:
    """``path`` or ``path:weight``."""
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            return Path(head), float(tail)
        except ValueError:
            pass
    return Path(spec), 1.0


def _interleave(lists, weights, tokens, count, names=None, report=None):
    """Documents of every list, the list furthest behind its token share first, until each
    list has its share of ``tokens`` (or runs out)."""
    quota = [tokens * w / sum(weights) for w in weights]
    pos = [0] * len(lists)
    used = [0.0] * len(lists)
    out: list[str] = []
    while True:
        live = [
            k
            for k in range(len(lists))
            if pos[k] < len(lists[k]) and used[k] < quota[k]
        ]
        if not live:
            return out
        k = min(live, key=lambda j: used[j] / weights[j])
        doc = lists[k][pos[k]]
        pos[k] += 1
        n = count(doc)
        used[k] += n
        out.append(doc)
        if report is not None:
            report["sources"][names[k]]["used"] += 1
            report["sources"][names[k]]["tokens"] += n


def build_mix(
    sources: list[tuple[Path, float]],
    tokens: int,
    count=None,
    holdout_share: float = 0.0,
    holdout_tokens: int = 200_000,
) -> tuple[str, str, dict]:
    """``(calibration text, held-out text, report)``. ``count(text)`` returns a token
    count (default: characters / 4). The held-out text takes ``holdout_share`` of every
    source's documents (blocks of 10), never used for calibration, up to ``holdout_tokens``."""
    if not sources or any(w <= 0 for _, w in sources):
        raise ValueError("give at least one source, each with a weight above 0")
    count = count or (lambda t: max(1, len(t) // CHARS_PER_TOKEN))
    period = round(1 / holdout_share) if holdout_share > 0 else 0
    weights = [w for _, w in sources]
    names = [str(p) for p, _ in sources]
    cal: list[list[str]] = []
    held: list[list[str]] = []
    report = {"sources": {}, "tokens_target": tokens}
    for (path, _), name in zip(sources, names):
        docs = read_documents(path)
        if not docs:
            raise ValueError(f"{path}: no documents")
        keep, out = [], []
        for i, d in enumerate(docs):
            held_out = period and (i // HOLDOUT_BLOCK) % period == period - 1
            (out if held_out else keep).append(d)
        if not keep:
            raise ValueError(f"{path}: every document is held out")
        cal.append(keep)
        held.append(out)
        report["sources"][name] = {"documents": len(docs), "used": 0, "tokens": 0}
    text = _interleave(cal, weights, tokens, count, names, report)
    held_text = _interleave(held, weights, holdout_tokens, count) if period else []
    got = sum(v["tokens"] for v in report["sources"].values())
    report["tokens"] = got
    report["held_out_documents"] = len(held_text)
    report["held_out_tokens"] = sum(count(d) for d in held_text)
    if got < 0.95 * tokens:
        report["warning"] = (
            f"the sources hold {got} tokens, fewer than the {tokens} requested"
        )
    return "\n\n".join(text), "\n\n".join(held_text), report
