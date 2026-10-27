# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Decision fidelity (``experiments/decision_fidelity.py``): 66 coding-agent prompts
over this repository, one XML tool call each. ``ok`` = a well-formed call and no garbage;
the comparison also reports agreement with the reference run on the first tool and on
the tool with its key argument (``AGREE``). Groups are the conversation stages."""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXP))
import decision_fidelity as df  # noqa: E402

NAME = "decision"
VERSION = "1"
MAX_TOKENS = 384
DEFAULT_LIMIT = None
AGREE = (("tool",), ("tool", "arg"))


def load(limit, seed):
    prompts = json.loads(df.PROMPTS.read_text())
    items = [
        {"id": p["id"], "group": p["stage"], "messages": p["messages"]} for p in prompts
    ]
    return items[:limit] if limit else items


def prompt(item):
    return item["messages"]


def score(item, completion):
    call = df.parse(completion)
    bad = df.garbage(completion)
    return {"ok": call["valid"] and not bad, **call, "garbage": bad}
