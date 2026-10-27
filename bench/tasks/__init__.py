# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Task registry. A task is a module with:

- ``NAME``, ``VERSION`` (bump when the items, the prompt or the scoring change),
  ``MAX_TOKENS``, ``DEFAULT_LIMIT`` (None: all items);
- ``load(limit, seed) -> list[dict]``: items with a unique ``id`` and optionally a
  ``group`` (a sub-table in the comparison);
- ``prompt(item)``: a message list (chat endpoint) or a string (text completion);
- ``score(item, completion) -> dict`` with ``ok`` (bool) and any details;

and optionally ``DATASETS`` (``(path, name, split)`` for ``bench.data export``),
``request(client, item) -> (completion, usage)`` to replace plain generation,
``fingerprint()`` (identifies generated content), ``AGREE`` (detail-key tuples compared
for agreement with the reference) and ``METRICS`` (numeric detail keys to average).
"""

import importlib

NAMES = ("gsm8k", "humaneval", "mbpp", "ifeval", "toolcall", "needle", "mc", "decision")
ALIASES = {"code": ("humaneval", "mbpp"), "all": NAMES}


def get(name: str):
    if name not in NAMES:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(NAMES)}")
    return importlib.import_module(f"bench.tasks.{name}")


def resolve(spec: str) -> dict:
    names = []
    for n in (s.strip() for s in spec.split(",")):
        for m in ALIASES.get(n, (n,)):
            if m and m not in names:
                names.append(m)
    return {n: get(n) for n in names}
