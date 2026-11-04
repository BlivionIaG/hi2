# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Mixed-domain calibration text (``hi2 calib``).

PTQ reads the first tokens of the calibration file and QAD draws windows from all of it,
so the domains must be mixed through the whole file, not appended one after another. This
builds such a file from any number of sources: ``.txt`` (blank-line separated blocks),
``.jsonl`` and ``.json`` with ``text`` records or chat records (``messages`` or
``conversations``). Each source gets its weight's share of the token budget, and its
documents are interleaved by token share, so every stretch of the output holds every
domain. Holdouts reserve exact prepared-document identities globally, including copies
in other sources. Plain text stays unchanged; chat templates and tokenizer counts are
explicit callbacks, with their identities recorded in the report.
"""

import hashlib
import json
import math
from collections.abc import Callable, Iterator
from numbers import Integral, Real
from pathlib import Path

CHARS_PER_TOKEN = 4  # the estimate when no tokenizer is given
MIN_DOC_CHARS = 2000  # blocks of a .txt file are grouped up to this size


def _chat(record, render_chat=None) -> str | None:
    turns = record.get("messages") or record.get("conversations")
    if not turns:
        return None
    if render_chat is not None:
        if not isinstance(turns, list):
            raise ValueError("chat turns must be a list of role/content messages")
        messages = []
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("each chat turn must be a role/content object")
            role = turn.get("role") or turn.get("from") or "user"
            content = turn.get("content") if "content" in turn else turn.get("value")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    "chat-template preparation requires text-only messages"
                )
            if not turn.get("role"):
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
            messages.append({"role": role, "content": content})
        text = render_chat(messages)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("render_chat must return nonempty text")
        return text
    lines = []
    for t in turns:
        role = t.get("role") or t.get("from") or "user"
        content = t.get("content") if "content" in t else t.get("value")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or None


def _record(record, render_chat=None) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        text = record.get("text") or _chat(record, render_chat)
        if text is not None and not isinstance(text, str):
            raise ValueError("a text record must contain a string")
        return text
    return None


def _iter_documents(path: Path, render_chat=None) -> Iterator[str]:
    # Stream JSONL instead of retaining raw text, parsed records and prepared copies.
    with path.open(encoding="utf-8", errors="ignore") as stream:
        if path.suffix == ".jsonl":
            records = (json.loads(line) for line in stream if line.strip())
        elif path.suffix == ".json":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f"{path}: a .json source must be a list of records")
        else:
            cur = ""
            for block in stream.read().split("\n\n"):
                cur = f"{cur}\n\n{block}" if cur else block
                if len(cur) >= MIN_DOC_CHARS:
                    if cur.strip():
                        yield cur
                    cur = ""
            if cur.strip():
                yield cur
            return
        for record in records:
            text = _record(record, render_chat)
            if text and text.strip():
                yield text


def read_documents(
    path: Path,
    *,
    render_chat: Callable[[list[dict[str, str]]], str] | None = None,
) -> list[str]:
    """Read a source, leaving text records and .txt blocks unchanged.

    By default chat records become ``role: content`` lines. An explicit renderer gets
    role/content messages (including ShareGPT human/gpt -> user/assistant aliases)
    instead. It must return nonempty text; no template or tokenizer is loaded here.
    """
    return list(_iter_documents(path, render_chat))


def parse_source(spec: str) -> tuple[Path, float]:
    """``path`` or ``path:weight``."""
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            return Path(head), float(tail)
        except ValueError:
            pass
    return Path(spec), 1.0


def _interleave(lists, shares, tokens, count):
    """Take whole documents, the source furthest behind its token share first."""
    streams = [iter(docs) for docs in lists]
    quota = [tokens * share for share in shares]
    used = [0] * len(streams)
    documents = [0] * len(streams)
    live = [k for k, q in enumerate(quota) if q > 0]
    out = []
    while live:
        k = min(live, key=lambda j: used[j] / shares[j])
        doc = next(streams[k], None)
        if doc is None:
            live.remove(k)
            continue
        used[k] += count(doc)
        documents[k] += 1
        out.append(doc)
        if used[k] >= quota[k]:
            live.remove(k)
    return out, used, documents


def _positive_tokens(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer token count")


def build_mix(
    sources: list[tuple[Path, float]],
    tokens: int,
    count: Callable[[str], int] | None = None,
    holdout_share: float = 0.0,
    holdout_tokens: int = 200_000,
    *,
    render_chat: Callable[[list[dict[str, str]]], str] | None = None,
    tokenizer_id: str | None = None,
    chat_template_id: str | None = None,
    min_holdout_tokens: int = 1,
) -> tuple[str, str, dict]:
    """Return calibration text, held-out text and their preparation/split report.

    ``count(text)`` must deterministically return a nonnegative integer; without it,
    counts are explicitly character/4 estimates. Final token counts include document
    separators. Per-source counts/shares use document counts, since tokenization of
    concatenated text need not be additive. Targets select whole documents, so they
    can overshoot. Shortfalls are reported; ``min_holdout_tokens`` is a hard minimum.

    A requested holdout reserves floor(unique_documents * holdout_share) identities,
    clamped to [1, unique_documents - 1], in SHA256 order (exact text breaks ties).
    Assignment depends only on prepared content, not source/document order. All
    occurrences of an identity stay in one split. Duplicates are counted, not removed,
    preserving source weighting and the no-holdout document order.

    ``render_chat`` is opt-in and requires ``count``, ``tokenizer_id`` and
    ``chat_template_id`` for attributable preparation. Text records are never
    templated. Callers should render without a generation prompt and count without
    adding special tokens, matching direct quantization of the prepared .txt file.
    """
    if not sources or any(
        isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w <= 0
        for _, w in sources
    ):
        raise ValueError("give at least one source, each with a finite weight above 0")
    if (
        isinstance(holdout_share, bool)
        or not isinstance(holdout_share, Real)
        or not math.isfinite(holdout_share)
        or not 0 <= holdout_share < 1
    ):
        raise ValueError("holdout_share must be finite and satisfy 0 <= share < 1")
    _positive_tokens("tokens", tokens)
    _positive_tokens("holdout_tokens", holdout_tokens)
    _positive_tokens("min_holdout_tokens", min_holdout_tokens)
    if holdout_share and min_holdout_tokens > holdout_tokens:
        raise ValueError("min_holdout_tokens must not exceed holdout_tokens")
    for name, identity in (
        ("tokenizer_id", tokenizer_id),
        ("chat_template_id", chat_template_id),
    ):
        if identity is not None and (
            not isinstance(identity, str) or not identity.strip()
        ):
            raise ValueError(f"{name} must be a nonempty identity")
    if count is not None and not callable(count):
        raise ValueError("count must be a token-counting callback")
    if tokenizer_id is not None and count is None:
        raise ValueError("tokenizer_id requires a token-counting callback")
    if render_chat is not None:
        if not callable(render_chat) or not chat_template_id or not tokenizer_id:
            raise ValueError(
                "render_chat requires a callback, tokenizer_id and chat_template_id"
            )
    elif chat_template_id is not None:
        raise ValueError("chat_template_id requires render_chat")

    count_method = (
        "characters/4-estimate"
        if count is None
        else "tokenizer"
        if tokenizer_id is not None
        else "callback"
    )
    counter = (
        count if count is not None else lambda t: max(1, len(t) // CHARS_PER_TOKEN)
    )

    def token_count(text):
        n = counter(text)
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError(
                "count(text) must return a nonnegative integer token count"
            )
        return int(n)

    # Normalize before summing: valid finite weights can otherwise overflow their sum.
    scale = max(w for _, w in sources)
    weights = [w / scale for _, w in sources]
    weight_sum = sum(weights)
    shares = [w / weight_sum for w in weights]
    names = [str(p) for p, _ in sources]
    documents: list[str] = []
    identities: dict[str, int] = {}
    owners: list[str] = []
    shared: set[int] = set()
    by_source: list[list[int]] = []
    report = {
        "sources": {},
        "tokens_target": tokens,
        "holdout_tokens_target": holdout_tokens if holdout_share else 0,
        "min_holdout_tokens": min_holdout_tokens if holdout_share else 0,
        "holdout_share": holdout_share,
        "split_method": "sha256-exact-text-v1" if holdout_share else "none",
        "preparation": {
            "chat_format": "chat-template"
            if render_chat is not None
            else "role-content",
            "token_count": count_method,
            "tokenizer_id": tokenizer_id,
            "chat_template_id": chat_template_id,
        },
    }
    for (path, _), name, share in zip(sources, names, shares):
        ids = []
        for doc in _iter_documents(path, render_chat):
            identity = identities.get(doc)
            if identity is None:
                identity = len(documents)
                identities[doc] = identity
                documents.append(doc)
                owners.append(name)
            elif owners[identity] != name:
                shared.add(identity)
            ids.append(identity)
        if not ids:
            raise ValueError(f"{path}: no documents")
        by_source.append(ids)
        source = report["sources"].setdefault(
            name,
            {
                "documents": 0,
                "used": 0,
                "tokens": 0,
                "held_out_documents": 0,
                "held_out_tokens": 0,
                "requested_share": 0.0,
            },
        )
        source["documents"] += len(ids)
        source["requested_share"] += share
    report["documents"] = sum(len(ids) for ids in by_source)
    report["unique_documents"] = len(documents)
    report["duplicate_documents"] = report["documents"] - len(documents)
    report["cross_source_duplicate_documents"] = len(shared)
    del identities, owners, shared

    reserved = set()
    if holdout_share:
        if len(documents) < 2:
            raise ValueError(
                "a requested holdout needs at least two unique documents; "
                "add distinct source documents or set holdout_share=0"
            )
        n_held = min(
            len(documents) - 1, max(1, math.floor(len(documents) * holdout_share))
        )
        order = sorted(
            range(len(documents)),
            key=lambda i: (
                hashlib.sha256(documents[i].encode("utf-8")).digest(),
                documents[i],
            ),
        )
        reserved.update(order[:n_held])
        del order
    report["reserved_holdout_documents"] = len(reserved)
    report["reserved_holdout_share"] = len(reserved) / len(documents)
    counts = {}

    def document_count(identity):
        if identity not in counts:
            counts[identity] = token_count(documents[identity])
        return counts[identity]

    cal, cal_tokens, cal_used = _interleave(
        ((i for i in ids if i not in reserved) for ids in by_source),
        shares,
        tokens,
        document_count,
    )
    held, held_tokens, held_used = (
        _interleave(
            ((i for i in ids if i in reserved) for ids in by_source),
            shares,
            holdout_tokens,
            document_count,
        )
        if holdout_share
        else ([], [0] * len(sources), [0] * len(sources))
    )
    text = "\n\n".join(documents[i] for i in cal)
    held_text = "\n\n".join(documents[i] for i in held)
    got = token_count(text) if text else 0
    held_got = token_count(held_text) if held_text else 0
    if got == 0:
        raise ValueError(
            "calibration has no usable tokens; add source text or check count"
        )
    if holdout_share and (not held or held_got < min_holdout_tokens):
        raise ValueError(
            f"holdout has {len(held)} documents and {held_got} tokens ({count_method}); "
            f"need at least {min_holdout_tokens}. Add distinct source documents or "
            "increase holdout_share; lower min_holdout_tokens only if evaluation "
            "needs fewer tokens"
        )
    document_tokens = sum(cal_tokens)
    held_document_tokens = sum(held_tokens)
    for k, name in enumerate(names):
        source = report["sources"][name]
        source["used"] += cal_used[k]
        source["tokens"] += cal_tokens[k]
        source["held_out_documents"] += held_used[k]
        source["held_out_tokens"] += held_tokens[k]
    for source in report["sources"].values():
        source["token_share"] = (
            source["tokens"] / document_tokens if document_tokens else 0.0
        )
        source["held_out_token_share"] = (
            source["held_out_tokens"] / held_document_tokens
            if held_document_tokens
            else 0.0
        )
    report["tokens"] = got
    report["document_tokens"] = document_tokens
    report["calibration_documents"] = len(cal)
    report["calibration_unique_documents"] = len(set(cal))
    report["held_out_documents"] = len(held)
    report["held_out_unique_documents"] = len(set(held))
    report["held_out_tokens"] = held_got
    report["held_out_document_tokens"] = held_document_tokens
    warnings = []
    if got < 0.95 * tokens:
        warnings.append(
            f"the sources hold {got} tokens, fewer than the {tokens} requested"
        )
    if holdout_share and held_got < holdout_tokens:
        warnings.append(
            f"holdout has {held_got} tokens ({count_method}), fewer than the "
            f"{holdout_tokens} requested; add distinct source documents, increase "
            "holdout_share, or lower holdout_tokens to your evaluation requirement"
        )
    if warnings:
        report["warnings"] = warnings
        report["warning"] = "; ".join(warnings)
    return text, held_text, report
