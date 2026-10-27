# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Instruction following with verifiable constraints (an IFEval-style subset written
here, no external package): 17 constraint types attached to 40 writing tasks, one or two
constraints per prompt, 500 prompts generated from the seed. ``ok`` = every constraint
holds (strict, prompt level); ``frac`` = the share of constraints that hold."""

import json
import random
import re

NAME = "ifeval"
VERSION = "1"
MAX_TOKENS = 1024
DEFAULT_LIMIT = None
SIZE = 500
METRICS = ("frac",)

TASKS = [
    "Write a short blog post about a weekend trip to Lisbon.",
    "Write an email to a landlord asking them to repair a broken heater.",
    "Explain how a refrigerator keeps food cold, for a ten-year-old.",
    "Write a product description for a solar-powered phone charger.",
    "Write a cover letter for a junior data analyst position.",
    "Describe the water cycle.",
    "Give advice to a student who is starting university.",
    "Write a review of an imaginary Italian restaurant.",
    "Pitch a mobile app that reminds people to drink water.",
    "Explain why the sky is blue.",
    "Write a story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the benefits of regular exercise.",
    "Write a speech for the opening of a community garden.",
    "Explain what a hash table is and when to use one.",
    "Write a letter to your future self ten years from now.",
    "Describe how to make a cup of tea.",
    "Argue for or against school uniforms.",
    "Write a guide to choosing a first houseplant.",
    "Explain how vaccines train the immune system.",
    "Write a short biography of an invented jazz musician.",
    "Describe a typical day for a beekeeper.",
    "Write a complaint to an airline about lost luggage.",
    "Explain the rules of chess to a beginner.",
    "Write a travel itinerary for three days in Kyoto.",
    "Explain the difference between weather and climate.",
    "Write a motivational message for a team before a product launch.",
    "Describe the history of the bicycle.",
    "Write a job advertisement for a bakery assistant.",
    "Explain how to set up a monthly budget.",
    "Write a poem about autumn in a city.",
    "Describe how a bill becomes a law, in general terms.",
    "Write instructions for assembling a simple bookshelf.",
    "Explain what causes earthquakes.",
    "Write a thank-you note to a teacher.",
    "Compare trains and planes for a 600 km trip.",
    "Explain how compound interest works.",
    "Write a short mystery set on a night train.",
    "Describe the benefits and risks of social media for teenagers.",
    "Write an announcement that a library is extending its opening hours.",
    "Explain how a search engine ranks pages, in simple terms.",
]
KEYWORDS = [
    "river",
    "balance",
    "window",
    "signal",
    "garden",
    "copper",
    "harbor",
    "lantern",
    "bridge",
    "orbit",
    "maple",
    "compass",
]
AVOID = ["very", "really", "thing", "good", "important", "just"]
ENDINGS = [
    "Is there anything else I can help with?",
    "Let me know if you have questions.",
    "That is all for today.",
]


def _words(t):
    return re.findall(r"\b[\w'-]+\b", t)


def _unfence(t):
    m = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", t, re.S)
    return m.group(1) if m else t


def _json_ok(t):
    try:
        json.loads(_unfence(t))
    except ValueError:
        return False
    return True


def _bullets(t):
    return len(re.findall(r"^\s*[*-]\s+\S", t, re.M))


def _paragraphs(t):
    return len([p for p in re.split(r"\s*\*\*\*\s*", t.strip()) if p.strip()])


# kind -> (make(rng) -> params, instruction(params), check(text, params))
C = {
    "word_min": (
        lambda r: {"n": r.choice([150, 200, 300])},
        lambda p: f"Answer with at least {p['n']} words.",
        lambda t, p: len(_words(t)) >= p["n"],
    ),
    "word_max": (
        lambda r: {"n": r.choice([50, 80, 120])},
        lambda p: f"Answer with fewer than {p['n']} words.",
        lambda t, p: len(_words(t)) < p["n"],
    ),
    "no_comma": (
        lambda r: {},
        lambda p: "Do not use any commas in your response.",
        lambda t, p: "," not in t,
    ),
    "lowercase": (
        lambda r: {},
        lambda p: (
            "Your entire response must be in lowercase letters; no "
            "capital letters are allowed."
        ),
        lambda t, p: t == t.lower() and any(c.isalpha() for c in t),
    ),
    "uppercase": (
        lambda r: {},
        lambda p: "Your entire response must be in English capital letters only.",
        lambda t, p: t == t.upper() and any(c.isalpha() for c in t),
    ),
    "keywords": (
        lambda r: {"words": r.sample(KEYWORDS, 2)},
        lambda p: "Include the keywords {} and {} in the response.".format(*p["words"]),
        lambda t, p: all(re.search(rf"\b{w}\b", t, re.I) for w in p["words"]),
    ),
    "avoid_word": (
        lambda r: {"word": r.choice(AVOID)},
        lambda p: f'Do not use the word "{p["word"]}" anywhere in the response.',
        lambda t, p: not re.search(rf"\b{p['word']}\b", t, re.I),
    ),
    "paragraphs": (
        lambda r: {"n": r.choice([2, 3, 4])},
        lambda p: (
            f"Write exactly {p['n']} paragraphs, separated from each "
            f"other by the markdown divider ***."
        ),
        lambda t, p: _paragraphs(t) == p["n"] and "***" in t,
    ),
    "bullets": (
        lambda r: {"n": r.choice([3, 4, 5])},
        lambda p: (
            f"Your answer must contain exactly {p['n']} bullet points, "
            f"written as markdown bullets such as:\n* This is a point."
        ),
        lambda t, p: _bullets(t) == p["n"],
    ),
    "end_phrase": (
        lambda r: {"phrase": r.choice(ENDINGS)},
        lambda p: (
            f"Finish your response with this exact phrase: "
            f'"{p["phrase"]}". No other words should follow it.'
        ),
        lambda t, p: t.strip().endswith(p["phrase"]),
    ),
    "quotes": (
        lambda r: {},
        lambda p: "Wrap your entire response in double quotation marks.",
        lambda t, p: len(t.strip()) > 1 and t.strip()[0] == t.strip()[-1] == '"',
    ),
    "json": (
        lambda r: {},
        lambda p: (
            "Your entire output must be valid JSON; you may wrap it in a "
            "```json code block."
        ),
        lambda t, p: _json_ok(t),
    ),
    "title": (
        lambda r: {},
        lambda p: (
            "Your answer must contain a title wrapped in double angular "
            "brackets, such as <<poem of joy>>."
        ),
        lambda t, p: re.search(r"<<[^\n<>]+>>", t) is not None,
    ),
    "postscript": (
        lambda r: {"marker": r.choice(["P.S.", "P.P.S."])},
        lambda p: (
            f"At the end of your response, add a postscript starting with {p['marker']}"
        ),
        lambda t, p: re.search(rf"^\s*{re.escape(p['marker'])}", t, re.M) is not None,
    ),
    "highlights": (
        lambda r: {"n": r.choice([2, 3])},
        lambda p: (
            f"Highlight at least {p['n']} sections of your answer with "
            f"markdown, for example *highlighted section*."
        ),
        lambda t, p: len(re.findall(r"\*[^\n*]+\*", t)) >= p["n"],
    ),
    "placeholders": (
        lambda r: {"n": r.choice([2, 3])},
        lambda p: (
            f"The response must contain at least {p['n']} "
            f"placeholders in square brackets, such as [address]."
        ),
        lambda t, p: len(re.findall(r"\[[^\]\n]+\]", t)) >= p["n"],
    ),
    "sections": (
        lambda r: {"n": r.choice([2, 3])},
        lambda p: (
            f"Your response must have {p['n']} sections. Mark the "
            f"beginning of each section with SECTION X, such as: SECTION 1."
        ),
        lambda t, p: len(re.findall(r"\bSECTION\s+\d+", t)) >= p["n"],
    ),
}
CONFLICTS = {
    frozenset(p)
    for p in [
        ("word_min", "word_max"),
        ("lowercase", "uppercase"),
        ("lowercase", "sections"),
        ("lowercase", "postscript"),
        ("lowercase", "end_phrase"),
        ("uppercase", "end_phrase"),
        ("uppercase", "keywords"),
        ("uppercase", "avoid_word"),
        ("quotes", "end_phrase"),
        ("quotes", "json"),
        ("paragraphs", "sections"),
        ("paragraphs", "bullets"),
        ("word_max", "word_min"),
        ("no_comma", "json"),
        ("word_max", "paragraphs"),
        ("word_max", "sections"),
    ]
} | {
    frozenset(("json", k))
    for k in C
    if k not in ("json", "keywords", "lowercase", "avoid_word", "word_max")
}


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    kinds = sorted(C)
    items = []
    for i in range(n):
        first = kinds[i % len(kinds)]  # every type about equally often
        chosen = [first]
        if rng.random() < 0.5:
            ok = [
                k
                for k in kinds
                if k != first and frozenset((first, k)) not in CONFLICTS
            ]
            chosen.append(rng.choice(ok))
        cons = [{"kind": k, **C[k][0](rng)} for k in chosen]
        task = rng.choice(TASKS)
        text = " ".join([task, *(C[c["kind"]][1](c) for c in cons)])
        items.append({"id": f"ifeval/{i}", "group": first, "text": text, "cons": cons})
    return items


def load(limit, seed):
    items = generate(SIZE, seed)
    return items[:limit] if limit else items


def prompt(item):
    return [{"role": "user", "content": item["text"]}]


def check(kind: str, text: str, params: dict) -> bool:
    return bool(C[kind][2](text, params))


def score(item, completion):
    res = {c["kind"]: check(c["kind"], completion, c) for c in item["cons"]}
    return {
        "ok": all(res.values()),
        "frac": sum(res.values()) / len(res),
        "checks": res,
    }
