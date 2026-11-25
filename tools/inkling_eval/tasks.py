# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Build the five task categories into one deterministic, renderable plan.

Every item is a JSON-ready dict with ``id``, ``category``, ``subcategory``,
``prompt`` (rendered with the Inkling chat template), ``max_tokens``,
``scorer``, ``expected`` and a ``gold`` completion used only by the dry-run.
The plan's hash covers every item id and prompt, so a different plan refuses to
resume and a comparison warns.

Data provenance is in ``data/SOURCES.md``.  Code problems come from HumanEval
(MIT), MBPP (CC-BY-4.0) and authored C/TypeScript tasks; math from GSM8K (MIT)
and authored problems; multilingual from authored references; long-context
haystacks from this repository's own markdown (Apache-2.0).
"""

import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from tools.inkling_eval import chat_template, multilingual_data
from tools.inkling_eval.scoring import canonical_number

DATA = Path(__file__).resolve().parent / "data"
PIECE = re.compile(r"[A-Za-z]{1,6}|\d|[^\w\s]")  # about one Inkling token each
FILL = 0.9
RESERVE = 300
CATEGORIES = ("agentic", "code", "multilingual", "long_context", "math")
# Completion budgets cover the model's plain-text reasoning preamble plus the
# final answer.  Measured on 2026-11-25 over stock, stock2 and hi2-v1 (735
# completions): p99 completion tokens of items that stopped before their old
# budget were agentic 189/384, code 506/512, translation 156/160, math 464/512;
# qa_numeric (0/84 stopped at 32), qa_word (0/42 at 24) and both needle tasks
# (0/72 at 48) never stopped, and summarize was 4/9 at 320, so those p99 are
# censored.  Rule: min(1024, ceil(2 * p99 / 128) * 128); censored short-answer QA
# gets 512 and censored long-context/summarize tasks get the 1024 cap.  A later
# max_tokens=1024 probe can tighten this; do not lower a budget below a proven
# p99 without checking the truncation gate.
MAX_TOKENS = 1024
BUDGETS = {
    ("agentic", "*"): 384,
    ("code", "*"): 1024,
    ("multilingual", "translation"): 384,
    ("multilingual", "qa_numeric"): 512,
    ("multilingual", "qa_word"): 512,
    ("long_context", "*"): 1024,
    ("math", "*"): 1024,
}


def budget_for(category: str, subcategory: str) -> int:
    return BUDGETS.get((category, subcategory), BUDGETS.get((category, "*"), MAX_TOKENS))
TEMPLATE_ID = "inkling2-mini chat_template.jinja transcription v1"
SYSTEM = {
    "role": "system",
    "content": "You are a precise operations assistant. Call exactly one of the provided tools when the user asks for an action.",
}


def load_jsonl(name: str) -> list[dict]:
    path = DATA / name
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return len(PIECE.findall(text))


# --------------------------------------------------------------------------- tools


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = {
    "get_weather": _tool(
        "get_weather",
        "Current weather for a location over the next days.",
        {
            "location": {"type": "string", "description": "City name."},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        ["location", "unit", "days"],
    ),
    "search_flights": _tool(
        "search_flights",
        "Search flights between two airports.",
        {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "date": {"type": "string", "description": "ISO date, YYYY-MM-DD."},
            "passengers": {"type": "integer", "minimum": 1, "maximum": 9},
        },
        ["origin", "destination", "date", "passengers"],
    ),
    "book_hotel": _tool(
        "book_hotel",
        "Book a hotel room.",
        {
            "city": {"type": "string"},
            "check_in": {"type": "string", "description": "ISO date, YYYY-MM-DD."},
            "nights": {"type": "integer", "minimum": 1, "maximum": 30},
            "guests": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        ["city", "check_in", "nights", "guests"],
    ),
    "send_email": _tool(
        "send_email",
        "Send an email.",
        {
            "to": {"type": "string", "description": "Email address."},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        ["to", "subject", "body"],
    ),
    "create_invoice": _tool(
        "create_invoice",
        "Create an invoice for a customer.",
        {
            "customer_id": {"type": "string"},
            "amount_cents": {"type": "integer", "minimum": 1},
            "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            "due_date": {"type": "string", "description": "ISO date, YYYY-MM-DD."},
        },
        ["customer_id", "amount_cents", "currency", "due_date"],
    ),
    "lookup_customer": _tool(
        "lookup_customer",
        "Look up a customer by email address.",
        {"email": {"type": "string"}},
        ["email"],
    ),
    "schedule_meeting": _tool(
        "schedule_meeting",
        "Schedule a meeting.",
        {
            "participants": {"type": "array", "items": {"type": "string"}},
            "start_iso": {"type": "string", "description": "ISO date-time."},
            "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480},
        },
        ["participants", "start_iso", "duration_minutes"],
    ),
    "convert_currency": _tool(
        "convert_currency",
        "Convert an amount between currencies.",
        {
            "amount": {"type": "number", "minimum": 0},
            "from_currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            "to_currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        },
        ["amount", "from_currency", "to_currency"],
    ),
}

CITIES = ["Lisbon", "Berlin", "Osaka", "Toronto", "Nairobi", "Lima", "Helsinki", "Auckland", "Prague", "Seoul"]
PEOPLE = ["Ada", "Bruno", "Chen", "Dara", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas"]
PROJECTS = ["Falcon", "Harbor", "Lantern", "Meadow", "Glacier", "Orchard", "Beacon", "Canyon"]
CURRENCIES = ["USD", "EUR", "GBP"]
DATES = ["2026-12-03", "2026-12-10", "2026-12-17", "2027-01-07", "2027-01-14"]


def _email(person: str) -> str:
    return person.lower() + "@example.com"


def _assistant_call(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }


def _tool_result(name: str, content: str) -> dict:
    return {"role": "tool", "name": name, "content": content}


def _conversation(final: str, prior: list[dict]) -> list[dict]:
    return [SYSTEM, *prior, {"role": "user", "content": final}]


def _weather(k):
    city, other = CITIES[k % 10], CITIES[(k + 3) % 10]
    days, unit = k + 2, ("fahrenheit" if k % 2 else "celsius")
    prior = [
        {"role": "user", "content": f"What is the weather in {other} for the next 2 days in celsius?"},
        _assistant_call("get_weather", {"location": other, "unit": "celsius", "days": 2}),
        _tool_result("get_weather", "8 degrees celsius, light rain"),
    ]
    final = f"Now check {city} for the next {days} days in {unit}."
    return ["get_weather", "search_flights", "lookup_customer"], _conversation(final, prior), {
        "name": "get_weather",
        "args": {"location": city, "unit": unit, "days": days},
    }


def _flights(k):
    origin, destination = CITIES[k % 10], CITIES[(k + 4) % 10]
    day = DATES[k % len(DATES)]
    passengers = 1 + (k % 4)
    prior = [
        {"role": "user", "content": f"Find flights from {destination} to {origin} on {day}."},
        _assistant_call("search_flights", {"origin": destination, "destination": origin, "date": day, "passengers": 1}),
        _tool_result("search_flights", "2 results: 06:15 and 18:40"),
    ]
    final = f"Book the search in reverse: from {origin} to {destination} on {DATES[(k + 1) % len(DATES)]} for {passengers} passengers."
    return ["search_flights", "get_weather", "book_hotel"], _conversation(final, prior), {
        "name": "search_flights",
        "args": {"origin": origin, "destination": destination, "date": DATES[(k + 1) % len(DATES)], "passengers": passengers},
    }


def _hotel(k):
    city, other = CITIES[(k + 2) % 10], CITIES[k % 10]
    check_in, nights, guests = DATES[k % len(DATES)], 2 + (k % 4), 1 + (k % 3)
    prior = [
        {"role": "user", "content": f"Book a hotel in {other} for {nights} nights starting {check_in}."},
        _assistant_call("book_hotel", {"city": other, "check_in": check_in, "nights": nights, "guests": 2}),
        _tool_result("book_hotel", f"Booked in {other}: check-in {check_in}, {nights} nights."),
    ]
    final = f"Book a hotel in {city} for the same check-in date and the same number of nights as that reservation, for {guests} guests."
    return ["book_hotel", "search_flights", "send_email"], _conversation(final, prior), {
        "name": "book_hotel",
        "args": {"city": city, "check_in": check_in, "nights": nights, "guests": guests},
    }


def _email_after_lookup(k):
    person = PEOPLE[k % 10]
    address, cid = _email(person), f"C-{1040 + k}"
    prior = [
        {"role": "user", "content": f"Look up the customer {address}."},
        _assistant_call("lookup_customer", {"email": address}),
        _tool_result("lookup_customer", json.dumps({"customer_id": cid, "email": address, "name": person})),
    ]
    subject, body = f"Invoice {cid}", f"Your invoice {cid} is ready."
    final = f'Send an email to the customer you just looked up with subject "{subject}" and body "{body}"'
    return ["lookup_customer", "send_email", "create_invoice"], _conversation(final, prior), {
        "name": "send_email",
        "args": {"to": address, "subject": subject, "body": body},
    }


def _invoice_after_lookup(k):
    person = PEOPLE[(k + 1) % 10]
    address, cid = _email(person), f"C-{2000 + k}"
    amount = 500 * (k + 1)
    currency, due = CURRENCIES[k % 3], DATES[(k + 2) % len(DATES)]
    prior = [
        {"role": "user", "content": f"Look up {address}."},
        _assistant_call("lookup_customer", {"email": address}),
        _tool_result("lookup_customer", json.dumps({"customer_id": cid, "email": address})),
    ]
    final = f"Create an invoice for that customer for {amount} cents in {currency}, due {due}."
    return ["lookup_customer", "create_invoice", "send_email"], _conversation(final, prior), {
        "name": "create_invoice",
        "args": {"customer_id": cid, "amount_cents": amount, "currency": currency, "due_date": due},
    }


def _meeting_from_result(k):
    people = [PEOPLE[(k + i) % 10] for i in range(3)]
    participants = [_email(person) for person in people]
    start = f"2026-12-{4 + k:02d}T15:00"
    prior = [
        {"role": "user", "content": "Who is on the account team?"},
        _assistant_call("lookup_customer", {"email": participants[0]}),
        _tool_result("lookup_customer", "Account team: " + ", ".join(participants)),
    ]
    final = f"Schedule a 45-minute meeting with those three participants starting {start}."
    return ["lookup_customer", "schedule_meeting", "send_email"], _conversation(final, prior), {
        "name": "schedule_meeting",
        "args": {"participants": participants, "start_iso": start, "duration_minutes": 45},
    }


def _currency_chain(k):
    amount = 100 + 50 * k
    prior = [
        {"role": "user", "content": "Convert 100 USD to EUR."},
        _assistant_call("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}),
        _tool_result("convert_currency", "92.50 EUR"),
    ]
    final = f"Now convert {amount} USD to EUR."
    return ["convert_currency", "create_invoice", "get_weather"], _conversation(final, prior), {
        "name": "convert_currency",
        "args": {"amount": amount, "from_currency": "USD", "to_currency": "EUR"},
    }


def _distractor(k):
    city, person = CITIES[(k + 5) % 10], PEOPLE[(k + 2) % 10]
    cid, amount = f"C-{3000 + k}", 1200 + 100 * k
    due = DATES[(k + 3) % len(DATES)]
    prior = [
        {"role": "user", "content": f"What is the weather in {city}?"},
        _assistant_call("get_weather", {"location": city, "unit": "celsius", "days": 1}),
        _tool_result("get_weather", "11 degrees celsius, clear"),
    ]
    final = f"Create an invoice for customer {cid} for {amount} cents in GBP, due {due}."
    return ["get_weather", "create_invoice", "search_flights", "book_hotel"], _conversation(final, prior), {
        "name": "create_invoice",
        "args": {"customer_id": cid, "amount_cents": amount, "currency": "GBP", "due_date": due},
    }


def _retry_after_error(k):
    person = PEOPLE[(k + 3) % 10]
    good = _email(person)
    prior = [
        {"role": "user", "content": f"Look up {person.lower()}@exmaple.com."},
        _assistant_call("lookup_customer", {"email": person.lower() + "@exmaple.com"}),
        _tool_result("lookup_customer", "No customer found."),
    ]
    final = f"The address had a typo. Look it up again with {good}."
    return ["lookup_customer", "send_email", "create_invoice"], _conversation(final, prior), {
        "name": "lookup_customer",
        "args": {"email": good},
    }


def _date_shift(k):
    people = [PEOPLE[(k + i) % 10] for i in range(2)]
    participants = [_email(person) for person in people]
    start = date(2026, 12, 3) + timedelta(days=k)
    prior = [
        {"role": "user", "content": f"Schedule a 30-minute meeting with {participants[0]} and {participants[1]} on {start.isoformat()} at 10:00."},
        _assistant_call("schedule_meeting", {"participants": participants, "start_iso": f"{start.isoformat()}T10:00", "duration_minutes": 30}),
        _tool_result("schedule_meeting", "Scheduled."),
    ]
    final = "Move that meeting exactly one week later, keeping the same participants and duration."
    shifted = start + timedelta(days=7)
    return ["schedule_meeting", "lookup_customer", "send_email"], _conversation(final, prior), {
        "name": "schedule_meeting",
        "args": {"participants": participants, "start_iso": f"{shifted.isoformat()}T10:00", "duration_minutes": 30},
    }


def _unit_change(k):
    city, days = CITIES[(k + 6) % 10], 2 + (k % 5)
    prior = [
        {"role": "user", "content": f"What is the weather in {city} for the next {days} days in celsius?"},
        _assistant_call("get_weather", {"location": city, "unit": "celsius", "days": days}),
        _tool_result("get_weather", "9 degrees celsius on average"),
    ]
    final = "Same place and the same number of days, but give it in fahrenheit."
    return ["get_weather", "convert_currency", "search_flights"], _conversation(final, prior), {
        "name": "get_weather",
        "args": {"location": city, "unit": "fahrenheit", "days": days},
    }


def _two_prior_calls(k):
    person, cid = PEOPLE[(k + 4) % 10], f"C-{4000 + k}"
    address = _email(person)
    prior = [
        {"role": "user", "content": f"Look up {address}."},
        _assistant_call("lookup_customer", {"email": address}),
        _tool_result("lookup_customer", json.dumps({"customer_id": cid, "email": address})),
        {"role": "user", "content": "Convert 100 USD to GBP."},
        _assistant_call("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "GBP"}),
        _tool_result("convert_currency", "78.00 GBP"),
    ]
    subject = f"Transfer {cid}"
    body = f"Converted 100 USD to 78 GBP for {cid}."
    final = f'Send an email to {address} with subject "{subject}" and body "{body}"'
    return ["lookup_customer", "convert_currency", "send_email"], _conversation(final, prior), {
        "name": "send_email",
        "args": {"to": address, "subject": subject, "body": body},
    }


SCENARIOS = (
    _weather,
    _flights,
    _hotel,
    _email_after_lookup,
    _invoice_after_lookup,
    _meeting_from_result,
    _currency_chain,
    _distractor,
    _retry_after_error,
    _date_shift,
    _unit_change,
    _two_prior_calls,
)


def _agentic_items(reasoning_effort: float) -> list[dict]:
    items = []
    for index in range(len(SCENARIOS) * 5):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        variant = index // len(SCENARIOS)
        tool_names, messages, expected = scenario(variant)
        tools = [TOOLS[name] for name in tool_names]
        prompt = chat_template.render(messages, tools=tools, reasoning_effort=reasoning_effort)
        gold_args = json.dumps(expected["args"], sort_keys=True, separators=(",", ":"))
        gold = (
            f"<|message_model|>{expected['name']}<|content_invoke_tool_json|>"
            f'{{"name":{json.dumps(expected["name"])},"args":{gold_args}}}'
            f"<|end_message|><|content_model_end_sampling|>"
        )
        items.append(
            {
                "id": f"agentic-{index:02d}",
                "category": "agentic",
                "subcategory": scenario.__name__.lstrip("_"),
                "scorer": "agentic",
                "max_tokens": budget_for("agentic", scenario.__name__.lstrip("_")),
                "prompt": prompt,
                "expected": {"name": expected["name"], "args": expected["args"], "tools": tools},
                "gold": gold,
                "meta": {"turn": scenario.__name__.lstrip("_")},
            }
        )
    return items


# --------------------------------------------------------------------------- code


def _code_items(reasoning_effort: float) -> list[dict]:
    items = []
    for row in load_jsonl("humaneval.jsonl"):
        entry = row["entry_point"]
        prompt = (
            "Complete the Python function below. Return the complete function definition "
            "(same name and signature) in one ```python code block and nothing else.\n\n"
            f"```python\n{row['prompt']}```"
        )
        tests = row["test"] + f"\ncheck({entry})\nprint('OK')\n"
        items.append(
            {
                "id": f"code-py-{row['task_id'].replace('/', '-')}",
                "category": "code",
                "subcategory": "python_humaneval",
                "scorer": "code",
                "max_tokens": budget_for("code", "python_humaneval"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {
                    "language": "python",
                    "entry": entry,
                    "tests": tests,
                    "source": f"openai/openai_humaneval:{row['task_id']}",
                },
                "gold": "```python\n" + row["prompt"] + row["canonical_solution"] + "\n```",
                "meta": {"source": "openai/openai_humaneval", "task_id": row["task_id"]},
            }
        )
    for row in load_jsonl("mbpp.jsonl"):
        signature = row["code"].splitlines()[0].strip()
        prompt = (
            "Write Python code that solves the task. Return one ```python code block and "
            f"nothing else.\n\nTask: {row['prompt']}\nRequired signature: {signature}"
        )
        imports = "\n".join(row.get("test_imports") or [])
        assertions = "\n".join(row["test_list"])
        tests = (imports + "\n" if imports else "") + assertions + "\nprint('OK')\n"
        items.append(
            {
                "id": f"code-py-mbpp-{row['task_id']}",
                "category": "code",
                "subcategory": "python_mbpp",
                "scorer": "code",
                "max_tokens": budget_for("code", "python_mbpp"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {
                    "language": "python",
                    "entry": row["entry_point"],
                    "tests": tests,
                    "source": f"google-research-datasets/mbpp:{row['task_id']}",
                },
                "gold": "```python\n" + row["code"] + "\n```",
                "meta": {"source": "google-research-datasets/mbpp", "task_id": row["task_id"]},
            }
        )
    for row in load_jsonl("c_code.jsonl"):
        prompt = (
            "Write a C11 translation unit for the task. Include the headers you need and "
            "do not define main. Return one ```c code block and nothing else.\n\n"
            f"Task: {row['task']}\nRequired signature: {row['signature']}"
        )
        items.append(
            {
                "id": f"code-c-{row['id']}",
                "category": "code",
                "subcategory": "c",
                "scorer": "code",
                "max_tokens": budget_for("code", "c"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {
                    "language": "c",
                    "entry": row["id"],
                    "tests": row["prototype"] + "\n" + row["tests"],
                    "source": "authored",
                },
                "gold": "```c\n" + row["solution"] + "\n```",
                "meta": {"source": "authored", "task_id": row["id"]},
            }
        )
    for row in load_jsonl("ts_code.jsonl"):
        prompt = (
            "Write TypeScript for the task. Use a single function declaration; do not use "
            "classes, enums, namespaces, interfaces or export statements. Return one "
            "```typescript code block and nothing else.\n\n"
            f"Task: {row['task']}\nRequired signature: {row['signature']}"
        )
        items.append(
            {
                "id": f"code-ts-{row['id']}",
                "category": "code",
                "subcategory": "typescript",
                "scorer": "code",
                "max_tokens": budget_for("code", "typescript"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {
                    "language": "typescript",
                    "entry": row["id"],
                    "tests": row["tests"],
                    "source": "authored",
                },
                "gold": "```typescript\n" + row["solution"] + "\n```",
                "meta": {"source": "authored", "task_id": row["id"]},
            }
        )
    return items


# --------------------------------------------------------------------------- multilingual


def _multilingual_items(reasoning_effort: float) -> list[dict]:
    items = []
    for spec in multilingual_data.TRANSLATIONS:
        for lang, name in multilingual_data.LANGUAGES:
            prompt = (
                f"Translate the following English sentence into {name}. Reply with the "
                f"translation only.\n\n{spec['en']}"
            )
            reference = spec["refs"][lang]
            items.append(
                {
                    "id": f"multi-{lang}-translate-{spec['id']}",
                    "category": "multilingual",
                    "subcategory": "translation",
                    "scorer": "chrf",
                    "max_tokens": budget_for("multilingual", "translation"),
                    "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                    "expected": {"reference": reference, "lang": lang, "min_chrf": 0.2},
                    "gold": reference,
                    "meta": {"lang": lang, "language": name, "source": "authored"},
                }
            )
    for spec in multilingual_data.QA:
        for lang, name in multilingual_data.LANGUAGES:
            prompt = spec["prompts"][lang]
            if spec["kind"] == "numeric":
                items.append(
                    {
                        "id": f"multi-{lang}-num-{spec['id']}",
                        "category": "multilingual",
                        "subcategory": "qa_numeric",
                        "scorer": "qa_numeric",
                        "max_tokens": budget_for("multilingual", "qa_numeric"),
                        "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                        "expected": {"answer": spec["answer"], "lang": lang},
                        "gold": spec["answer"],
                        "meta": {"lang": lang, "language": name, "source": "authored"},
                    }
                )
            else:
                answers = spec["answers"][lang]
                items.append(
                    {
                        "id": f"multi-{lang}-word-{spec['id']}",
                        "category": "multilingual",
                        "subcategory": "qa_word",
                        "scorer": "qa_word",
                        "max_tokens": budget_for("multilingual", "qa_word"),
                        "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                        "expected": {"answers": answers, "lang": lang},
                        "gold": answers[0],
                        "meta": {"lang": lang, "language": name, "source": "authored"},
                    }
                )
    return items


# --------------------------------------------------------------------------- long context


HAYSTACK = DATA / "haystack.txt"


def haystack_text() -> str:
    """The frozen snapshot of this repository's markdown used as filler.

    It is vendored so a documentation edit never changes long-context prompts
    between paired runs; ``data/SOURCES.md`` records how the snapshot is made.
    """
    return HAYSTACK.read_text(encoding="utf-8")


def haystack_sha256() -> str:
    return sha256_text(haystack_text())


def _take_pieces(corpus: str, pieces: int) -> str:
    text = corpus
    while estimate_tokens(text) < pieces:
        text = text + "\n\n" + corpus
    count = 0
    for match in PIECE.finditer(text):
        count += 1
        if count >= pieces:
            return text[: match.end()]
    return text


def _insert_at(text: str, fraction: float, snippet: str) -> str:
    position = max(0, min(len(text), int(len(text) * fraction)))
    newline = text.find("\n", position)
    if newline == -1:
        newline = len(text)
    return text[:newline] + "\n\n" + snippet + "\n\n" + text[newline:]


def _needle(rng: random.Random, project: str) -> tuple[str, str]:
    value = (
        "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
        + f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    )
    return value, f"The access code for project {project} is {value}."


def _long_context_items(contexts, seed: int, reasoning_effort: float) -> list[dict]:
    corpus = haystack_text()
    corpus_hash = sha256_text(corpus)
    items = []
    for length in contexts:
        pieces = int(FILL * (length - RESERVE))
        haystack_base = _take_pieces(corpus, pieces)
        for position in (0.05, 0.25, 0.5, 0.75, 0.95):
            rng = random.Random(f"{seed}:{length}:{position}:single")
            project = PROJECTS[int(position * 100) % len(PROJECTS)]
            value, sentence = _needle(rng, project)
            document = _insert_at(haystack_base, position, sentence)
            question = (
                f"What is the access code for project {project}? Answer with the code only."
            )
            prompt = (
                "Read the document below and answer the question at the end.\n\n"
                f"<document>\n{document}\n</document>\n\n{question}"
            )
            items.append(
                {
                    "id": f"long-{length // 1024}k-single-p{int(position * 100):02d}",
                    "category": "long_context",
                    "subcategory": "single_needle",
                    "scorer": "needle",
                    "max_tokens": budget_for("long_context", "single_needle"),
                    "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                    "expected": {"value": value, "distractors": []},
                    "gold": value,
                    "meta": {"length": length, "position": position, "haystack_sha256": corpus_hash},
                }
            )
        for position in (0.15, 0.5, 0.85):
            rng = random.Random(f"{seed}:{length}:{position}:multi")
            pairs = []
            sentinel = set()
            while len(pairs) < 4:
                project = PROJECTS[(len(pairs) * 3 + int(position * 10)) % len(PROJECTS)]
                value, sentence = _needle(rng, project)
                if value in sentinel or project in [p for p, _, _ in pairs]:
                    continue
                sentinel.add(value)
                pairs.append((project, value, sentence))
            document = haystack_base
            offsets = (position - 0.15, position - 0.05, position + 0.05, position + 0.15)
            for pair, offset in sorted(zip(pairs, offsets), key=lambda item: -item[1]):
                document = _insert_at(document, min(0.95, max(0.02, offset)), pair[2])
            asked_project, asked_value = pairs[0][0], pairs[0][1]
            distractors = [value for _, value, _ in pairs[1:]]
            question = (
                f"What is the access code for project {asked_project}? Answer with the code only."
            )
            prompt = (
                "Read the document below and answer the question at the end.\n\n"
                f"<document>\n{document}\n</document>\n\n{question}"
            )
            items.append(
                {
                    "id": f"long-{length // 1024}k-multi-p{int(position * 100):02d}",
                    "category": "long_context",
                    "subcategory": "multi_needle",
                    "scorer": "needle",
                    "max_tokens": budget_for("long_context", "multi_needle"),
                    "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                    "expected": {"value": asked_value, "distractors": distractors},
                    "gold": asked_value,
                    "meta": {"length": length, "position": position, "haystack_sha256": corpus_hash},
                }
            )
        memo = (
            "Operations memo. The December maintenance window runs from 03:00 to 05:00 UTC "
            "on 2026-12-05. The rollback owner is Team Cedar. The change freeze starts on "
            "2026-12-01."
        )
        document = _insert_at(haystack_base, 0.5, memo)
        prompt = (
            "Read the document below. Then reply with a summary of at most 60 words followed "
            "by the answer, and end with 'Answer: <name>'.\n\n"
            f"<document>\n{document}\n</document>\n\n"
            "Who is the rollback owner for the December maintenance window?"
        )
        items.append(
            {
                "id": f"long-{length // 1024}k-summarize",
                "category": "long_context",
                "subcategory": "summarize_then_answer",
                "scorer": "summarize",
                "max_tokens": budget_for("long_context", "summarize_then_answer"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {"answer": "Team Cedar", "summary_max_words": 60},
                "gold": (
                    "The document is long; the December maintenance window is 03:00-05:00 UTC "
                    "on 2026-12-05 and Team Cedar owns the rollback. Answer: Team Cedar"
                ),
                "meta": {"length": length, "position": 0.5, "haystack_sha256": corpus_hash},
            }
        )
    return items


# --------------------------------------------------------------------------- math


def _math_items(reasoning_effort: float) -> list[dict]:
    items = []
    for row in load_jsonl("gsm8k.jsonl"):
        found = re.search(r"####\s*([-0-9.,]+)", row["answer"])
        if not found:
            raise ValueError(f"GSM8K row {row['row']} has no final answer")
        answer = canonical_number(found.group(1).replace(",", ""))
        prompt = (
            "Solve the problem. End with a final line exactly 'The answer is <number>'.\n\n"
            + row["question"]
        )
        items.append(
            {
                "id": f"math-gsm-{row['row']:04d}",
                "category": "math",
                "subcategory": "gsm8k",
                "scorer": "math",
                "max_tokens": budget_for("math", "gsm8k"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {"answer": answer},
                "gold": f"The answer is {answer}.",
                "meta": {"source": "openai/gsm8k", "row": row["row"]},
            }
        )
    for row in load_jsonl("math_authored.jsonl"):
        answer = canonical_number(row["answer"])
        prompt = (
            "Solve the problem. End with a final line exactly 'The answer is <number>'.\n\n"
            + row["question"]
        )
        items.append(
            {
                "id": f"math-authored-{row['id']}",
                "category": "math",
                "subcategory": "authored",
                "scorer": "math",
                "max_tokens": budget_for("math", "authored"),
                "prompt": chat_template.user_prompt(prompt, reasoning_effort=reasoning_effort),
                "expected": {"answer": answer},
                "gold": f"The answer is {answer}.",
                "meta": {"source": "authored"},
            }
        )
    return items


# --------------------------------------------------------------------------- plan


def build_plan(
    *,
    seed: int = 0,
    contexts=(8192, 16384, 32768),
    max_model_len: int | None = None,
    categories=CATEGORIES,
    limit: int | None = None,
    reasoning_effort: float = 0.0,
) -> dict:
    """The full item list, deterministically ordered and hashed."""
    wanted = [category for category in CATEGORIES if category in set(categories)]
    usable = []
    skipped_contexts = []
    for length in contexts:
        if (
            max_model_len is not None
            and int(FILL * (length - RESERVE)) + RESERVE + 128 > max_model_len
        ):
            skipped_contexts.append({"length": length, "reason": f"max_model_len={max_model_len}"})
        else:
            usable.append(length)
    items: list[dict] = []
    if "agentic" in wanted:
        items.extend(_agentic_items(reasoning_effort))
    if "code" in wanted:
        items.extend(_code_items(reasoning_effort))
    if "multilingual" in wanted:
        items.extend(_multilingual_items(reasoning_effort))
    if "long_context" in wanted:
        items.extend(_long_context_items(usable, seed, reasoning_effort))
    if "math" in wanted:
        items.extend(_math_items(reasoning_effort))
    for item in items:
        item["prompt_sha256"] = sha256_text(item["prompt"])
        item["approx_prompt_tokens"] = estimate_tokens(item["prompt"])
    ids = [item["id"] for item in items]
    duplicate = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate:
        raise ValueError(f"duplicate item ids: {duplicate[:5]}")
    if limit is not None:
        kept, counts = [], {}
        for item in items:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
            if counts[item["category"]] <= limit:
                kept.append(item)
        items = kept
    items.sort(key=lambda item: item["id"])
    # The prompt-only hash proves a budget-only plan change did not alter the
    # item set or any prompt: it is the plan hash used before budgets were added.
    prompt_set_sha = sha256_text(
        json.dumps(
            [{"id": item["id"], "prompt": item["prompt_sha256"]} for item in items],
            sort_keys=True,
        )
    )
    plan_sha = sha256_text(
        json.dumps(
            [
                {"id": item["id"], "prompt": item["prompt_sha256"], "max_tokens": item["max_tokens"]}
                for item in items
            ],
            sort_keys=True,
        )
    )
    by_category = {}
    for item in items:
        by_category.setdefault(item["category"], {"items": 0, "subcategories": {}})
        by_category[item["category"]]["items"] += 1
        sub = item["subcategory"]
        by_category[item["category"]]["subcategories"][sub] = (
            by_category[item["category"]]["subcategories"].get(sub, 0) + 1
        )
    return {
        "items": items,
        "meta": {
            "schema": 1,
            "seed": seed,
            "categories": wanted,
            "contexts": usable,
            "skipped_contexts": skipped_contexts,
            "max_model_len": max_model_len,
            "reasoning_effort": reasoning_effort,
            "template": TEMPLATE_ID,
            "plan_sha256": plan_sha,
            "prompt_set_sha256": prompt_set_sha,
            "haystack_sha256": haystack_sha256() if "long_context" in wanted else None,
            "counts": by_category,
            "limit": limit,
        },
    }
