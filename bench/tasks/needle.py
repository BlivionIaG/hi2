# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Long-context retrieval: facts hidden in a haystack made of this repository's own
markdown documents, at 2k, 8k, 16k and 32k tokens, 40 items per length. Half the items
hide one access code and ask for it; half hide four key-value pairs and ask for one of
them (the other three are distractors). Everything follows from the seed; the haystack's
hash is the task fingerprint, so a changed document shows up in the comparison.

Lengths are approximate and tokenizer-free: the haystack is cut at a count of ``PIECE``
matches, which is within about 8% of the Qwen3 token count on these documents (plain
characters vary by 25%); the documents fill ``FILL`` of each length, so measured with the
Qwen3 tokenizer the prompts land at 78-95% of it (2k) and 88-93% (32k). ``usage.prompt_tokens`` in the results
has the real count. The 32k items need a ``max_model_len`` of at least 33k.
"""

import hashlib
import random
import re
from pathlib import Path

NAME = "needle"
VERSION = "1"
MAX_TOKENS = 32
DEFAULT_LIMIT = None
LENGTHS = (2048, 8192, 16384, 32768)
PER_LENGTH = 40
PIECE = re.compile(r"[A-Za-z]{1,6}|\d|[^\w\s]")  # about one Qwen3 token each
FILL = 0.9
RESERVE = 200  # tokens for the instructions and the question
ROOT = Path(__file__).resolve().parents[2]
ADJ = [
    "amber",
    "silent",
    "copper",
    "northern",
    "velvet",
    "hollow",
    "scarlet",
    "frozen",
    "golden",
    "quiet",
    "rapid",
    "ivory",
    "misty",
    "bright",
    "lunar",
    "rusty",
]
NOUN = [
    "falcon",
    "harbor",
    "lantern",
    "meadow",
    "glacier",
    "orchard",
    "beacon",
    "canyon",
    "willow",
    "comet",
    "anchor",
    "summit",
    "thistle",
    "ember",
    "quarry",
    "delta",
]
HEAD = (
    "Below is a long document. A few facts have been hidden in it. Read it, then "
    "answer the question at the end.\n\n<document>\n"
)


def haystack() -> str:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    return "\n\n".join(f.read_text() for f in files if f.exists())


def fingerprint() -> str:
    return hashlib.sha256(haystack().encode()).hexdigest()[:16]


def approx_tokens(text: str) -> int:
    return len(PIECE.findall(text))


def _cut(text: str, start: int, n: int) -> str:
    """About ``n`` tokens of text from the line after ``start``, wrapping around."""
    start = text.find("\n", start) + 1
    reps = 1 + n // max(1, approx_tokens(text))
    s = "\n\n".join([text[start:]] + [text] * reps)
    for k, m in enumerate(PIECE.finditer(s), 1):
        if k == n:
            return s[: m.end()]
    return s


def _insert(doc: str, facts: list[tuple[float, str]]) -> str:
    """Place each fact on its own line at the line break nearest its depth."""
    for depth, fact in sorted(facts, reverse=True):
        at = doc.rfind("\n", 0, max(1, int(len(doc) * depth))) + 1
        doc = doc[:at] + fact + "\n" + doc[at:]
    return doc


def make_items(seed: int, text: str | None = None) -> list[dict]:
    text = text if text is not None else haystack()
    rng = random.Random(seed)
    items = []
    for k in range(PER_LENGTH):
        for length in LENGTHS:  # interleaved, so a limit keeps every length
            n = int((length - RESERVE) * FILL)
            doc = _cut(text, rng.randrange(len(text)), n)
            names = rng.sample([f"{a}-{b}" for a in ADJ for b in NOUN], 4)
            values = rng.sample(range(1_000_000, 10_000_000), 4)
            kind = "single" if k % 2 == 0 else "multikey"
            if kind == "single":
                facts = [
                    (
                        rng.uniform(0.05, 0.95),
                        f"The access code for the {names[0]} vault is {values[0]}.",
                    )
                ]
                q = (
                    f"What is the access code for the {names[0]} vault? Answer with "
                    f"the number only."
                )
            else:
                facts = [
                    (rng.uniform(0.05, 0.95), f"The value of key {nm} is {v}.")
                    for nm, v in zip(names, values)
                ]
                q = f"What is the value of key {names[0]}? Answer with the number only."
            doc = _insert(doc, facts)
            items.append(
                {
                    "id": f"needle/{length // 1024}k/{k}",
                    "group": f"{length // 1024}k-{kind}",
                    "text": f"{HEAD}{doc}\n</document>\n\nQuestion: {q}",
                    "answer": str(values[0]),
                    "distractors": [str(v) for v in values[1:]]
                    if kind == "multikey"
                    else [],
                }
            )
    return items


def load(limit, seed):
    items = make_items(seed)
    return items[:limit] if limit else items


def prompt(item):
    return [{"role": "user", "content": item["text"]}]


def score(item, completion):
    nums = re.findall(r"\d+", completion.replace(",", ""))
    return {
        "ok": item["answer"] in nums,
        "distractor": any(d in nums for d in item["distractors"]),
    }
