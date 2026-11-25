# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""A separate agentic probe for reference-to-an-earlier-call tool choice.

The main plan has five hotel items; the v2 run showed four of them flipping
from ``content_invoke_tool_json`` on stock to a plain ``content_text`` answer on
hi2.  This set tests that family with 100 new, deterministically generated
items: 80 tool calls that copy a value from an earlier call, and 20 plain-chat
controls where no tool call is expected.  It is deliberately separate from the
245-item plan (own ids, prompts and ``plan_sha256``).

    python -m tools.inkling_eval.cli plan --set agentic-probe
    python -m tools.inkling_eval.cli dry-run --set agentic-probe
"""

import json

from tools.inkling_eval import chat_template, stats, tasks

BUDGET = 768
CITIES = tasks.CITIES
PEOPLE = tasks.PEOPLE
DATES = tasks.DATES

PROBE_TOOLS = dict(tasks.TOOLS)
PROBE_TOOLS.update(
    {
        "rent_car": tasks._tool(
            "rent_car",
            "Rent a car.",
            {
                "city": {"type": "string"},
                "pickup_date": {"type": "string", "description": "ISO date, YYYY-MM-DD."},
                "days": {"type": "integer", "minimum": 1, "maximum": 30},
                "car_class": {"type": "string", "enum": ["economy", "compact", "suv"]},
            },
            ["city", "pickup_date", "days", "car_class"],
        ),
        "order_food": tasks._tool(
            "order_food",
            "Order food for delivery.",
            {
                "restaurant": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "deliver_by": {"type": "string", "description": "ISO date-time."},
                "total_cents": {"type": "integer", "minimum": 1},
            },
            ["restaurant", "items", "deliver_by", "total_cents"],
        ),
        "transfer_money": tasks._tool(
            "transfer_money",
            "Transfer money between two accounts.",
            {
                "from_account": {"type": "string"},
                "to_account": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            },
            ["from_account", "to_account", "amount_cents", "currency"],
        ),
    }
)

CONTROL_SYSTEM = {
    "role": "system",
    "content": (
        "You are a precise operations assistant. Answer questions about the "
        "conversation directly. Call a tool only when the user asks for an action."
    ),
}


def _prior(user: str, name: str, args: dict, result: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        tasks._assistant_call(name, args),
        tasks._tool_result(name, result),
    ]


def _extras(k: int, count: int, tool_names: list[str], target: str) -> list[dict]:
    """Zero to two unrelated completed exchanges, chosen deterministically."""
    if count <= 0:
        return []
    choices = [name for name in tool_names if name != target]
    blocks = []
    for offset in range(count):
        name = choices[(k + offset) % len(choices)]
        if name == "get_weather":
            city = CITIES[(k + offset + 1) % 10]
            blocks += _prior(
                f"What is the weather in {city} for the next 2 days in celsius?",
                "get_weather",
                {"location": city, "unit": "celsius", "days": 2},
                "9 degrees celsius, light rain",
            )
        elif name == "lookup_customer":
            person = PEOPLE[(k + offset + 2) % 10]
            address = f"{person.lower()}@example.com"
            blocks += _prior(
                f"Look up {address}.",
                "lookup_customer",
                {"email": address},
                json.dumps({"customer_id": f"C-{6100 + k}", "email": address}),
            )
        elif name == "convert_currency":
            blocks += _prior(
                "Convert 100 USD to GBP.",
                "convert_currency",
                {"amount": 100, "from_currency": "USD", "to_currency": "GBP"},
                "78.00 GBP",
            )
        elif name == "search_flights":
            origin, destination = CITIES[(k + offset) % 10], CITIES[(k + offset + 4) % 10]
            day = DATES[(k + offset) % len(DATES)]
            blocks += _prior(
                f"Search flights from {origin} to {destination} on {day} for 1 passenger.",
                "search_flights",
                {"origin": origin, "destination": destination, "date": day, "passengers": 1},
                "2 results: 06:15 and 18:40",
            )
        elif name == "book_hotel":
            city = CITIES[(k + offset + 3) % 10]
            blocks += _prior(
                f"Book a hotel in {city} for 1 night starting {DATES[0]}.",
                "book_hotel",
                {"city": city, "check_in": DATES[0], "nights": 1, "guests": 1},
                f"Booked in {city}: check-in {DATES[0]}, 1 night.",
            )
        elif name == "send_email":
            person = PEOPLE[(k + offset + 5) % 10]
            address = f"{person.lower()}@example.com"
            blocks += _prior(
                f'Send an email to {address} with subject "Update" and body "No change."',
                "send_email",
                {"to": address, "subject": "Update", "body": "No change."},
                "Sent.",
            )
        else:  # pragma: no cover - the tool lists below only use the named branches
            raise ValueError(f"no distractor for {name}")
    return blocks


def _ref_book_hotel(k: int):
    other = CITIES[k % 10]
    city = CITIES[(k + 2) % 10]
    check_in = DATES[k % len(DATES)]
    nights, guests = 2 + (k % 4), 1 + (k % 3)
    prior = _prior(
        f"Book a hotel in {other} for {nights} nights starting {check_in}.",
        "book_hotel",
        {"city": other, "check_in": check_in, "nights": nights, "guests": 2},
        f"Booked in {other}: check-in {check_in}, {nights} nights.",
    )
    final = (
        f"Book a hotel in {city} for the same check-in date and the same number "
        f"of nights as that reservation, for {guests} guests."
    )
    return final, prior, ["book_hotel", "search_flights", "send_email", "get_weather"], {
        "name": "book_hotel",
        "args": {"city": city, "check_in": check_in, "nights": nights, "guests": guests},
    }


def _ref_rent_car(k: int):
    other = CITIES[(k + 4) % 10]
    city = CITIES[(k + 7) % 10]
    pickup = DATES[(k + 1) % len(DATES)]
    days = 2 + (k % 3)
    car_class = ("economy", "compact", "suv")[k % 3]
    prior = _prior(
        f"Rent a car in {other} from {pickup} for {days} days.",
        "rent_car",
        {"city": other, "pickup_date": pickup, "days": days, "car_class": car_class},
        f"Reserved a {car_class} car in {other} from {pickup} for {days} days.",
    )
    final = (
        f"Rent a car in {city} for the same pickup date and the same number of "
        f"days as that rental, in the same class."
    )
    return final, prior, ["rent_car", "book_hotel", "get_weather", "send_email"], {
        "name": "rent_car",
        "args": {"city": city, "pickup_date": pickup, "days": days, "car_class": car_class},
    }


def _ref_schedule(k: int):
    people = [PEOPLE[(k + i) % 10] for i in range(2)]
    participants = [f"{person.lower()}@example.com" for person in people]
    start = f"2026-12-{3 + (k % 20):02d}T{9 + (k % 6):02d}:00"
    duration = 30 + 15 * (k % 3)
    prior = _prior(
        f"Schedule a {duration}-minute meeting with {participants[0]} and {participants[1]} "
        f"starting {start}.",
        "schedule_meeting",
        {"participants": participants, "start_iso": start, "duration_minutes": duration},
        "Scheduled.",
    )
    final = "Move that meeting exactly one week later, keeping the same participants and duration."
    day = int(start[8:10]) + 7
    shifted = f"2026-12-{day:02d}T{start[11:]}"
    return final, prior, ["schedule_meeting", "lookup_customer", "send_email", "get_weather"], {
        "name": "schedule_meeting",
        "args": {"participants": participants, "start_iso": shifted, "duration_minutes": duration},
    }


def _ref_order(k: int):
    restaurant = ("Cedar Table", "Harbor Kitchen", "Lantern Grill", "Meadow Cafe")[k % 4]
    items = [("margherita", "salad"), ("ramen", "gyoza"), ("pilaf", "tea"), ("taco", "soup")][k % 4]
    total = 1800 + 125 * k
    deliver = f"2026-12-{5 + (k % 20):02d}T19:30"
    prior = _prior(
        f"Order {', '.join(items)} from {restaurant} for delivery by {deliver}.",
        "order_food",
        {"restaurant": restaurant, "items": list(items), "deliver_by": deliver, "total_cents": total},
        f"Order confirmed: {restaurant}, {', '.join(items)}, total {total} cents.",
    )
    new_date = f"2026-12-{6 + (k % 20):02d}T20:00"
    final = (
        "Order the same items from that restaurant again with the same total, "
        f"delivering by {new_date}."
    )
    return final, prior, ["order_food", "get_weather", "send_email", "search_flights"], {
        "name": "order_food",
        "args": {
            "restaurant": restaurant,
            "items": list(items),
            "deliver_by": new_date,
            "total_cents": total,
        },
    }


def _ref_transfer(k: int):
    from_account = f"AC-{700 + k:03d}"
    to_account = f"AC-{800 + k:03d}"
    amount = 2500 + 500 * (k % 6)
    currency = ("USD", "EUR", "GBP")[k % 3]
    prior = _prior(
        f"Transfer {amount} cents in {currency} from {from_account} to {to_account}.",
        "transfer_money",
        {
            "from_account": from_account,
            "to_account": to_account,
            "amount_cents": amount,
            "currency": currency,
        },
        f"Transferred {amount} cents in {currency}.",
    )
    final = "Send the same amount from the same source account to the same destination account again."
    return final, prior, ["transfer_money", "lookup_customer", "send_email", "get_weather"], {
        "name": "transfer_money",
        "args": {
            "from_account": from_account,
            "to_account": to_account,
            "amount_cents": amount,
            "currency": currency,
        },
    }


def _ref_invoice(k: int):
    person = PEOPLE[(k + 3) % 10]
    cid = f"C-{7100 + k}"
    amount = 3000 + 700 * (k % 5)
    currency = ("USD", "EUR", "GBP")[k % 3]
    due = DATES[(k + 2) % len(DATES)]
    prior = _prior(
        f"Create an invoice for customer {cid} for {amount} cents in {currency}, due {due}.",
        "create_invoice",
        {
            "customer_id": cid,
            "amount_cents": amount,
            "currency": currency,
            "due_date": due,
        },
        f"Invoice created for {cid}, {amount} cents in {currency}.",
    )
    new_due = DATES[(k + 3) % len(DATES)]
    final = (
        f"Create another invoice for the same customer and amount in {currency}, "
        f"with the same customer id, due {new_due}."
    )
    return final, prior, ["create_invoice", "lookup_customer", "send_email", "get_weather"], {
        "name": "create_invoice",
        "args": {
            "customer_id": cid,
            "amount_cents": amount,
            "currency": currency,
            "due_date": new_due,
        },
    }


def _ref_flights(k: int):
    origin = CITIES[(k + 5) % 10]
    destination = CITIES[(k + 8) % 10]
    day = DATES[(k + 3) % len(DATES)]
    passengers = 1 + (k % 4)
    prior = _prior(
        f"Search flights from {origin} to {destination} on {day} for {passengers} passengers.",
        "search_flights",
        {"origin": origin, "destination": destination, "date": day, "passengers": passengers},
        "3 results: 07:10, 12:25 and 19:40",
    )
    final = "Search the reverse route on the same date for the same number of passengers."
    return final, prior, ["search_flights", "get_weather", "book_hotel", "send_email"], {
        "name": "search_flights",
        "args": {
            "origin": destination,
            "destination": origin,
            "date": day,
            "passengers": passengers,
        },
    }


def _ref_email(k: int):
    person = PEOPLE[(k + 1) % 10]
    address = f"{person.lower()}@example.com"
    cid = f"C-{7200 + k}"
    subject = f"Booking {cid}"
    body = f"Your booking {cid} is confirmed."
    prior = _prior(
        f'Send an email to {address} with subject "{subject}" and body "{body}"',
        "send_email",
        {"to": address, "subject": subject, "body": body},
        "Sent.",
    )
    final = "Resend that exact email to the same address."
    return final, prior, ["send_email", "lookup_customer", "get_weather", "convert_currency"], {
        "name": "send_email",
        "args": {"to": address, "subject": subject, "body": body},
    }


TOOL_SCENARIOS = (
    _ref_book_hotel,
    _ref_rent_car,
    _ref_schedule,
    _ref_order,
    _ref_transfer,
    _ref_invoice,
    _ref_flights,
    _ref_email,
)


def _tool_items(reasoning_effort: float) -> list[dict]:
    items = []
    for index in range(len(TOOL_SCENARIOS) * 10):
        scenario = TOOL_SCENARIOS[index % len(TOOL_SCENARIOS)]
        variant = index // len(TOOL_SCENARIOS)
        final, prior, tool_names, expected = scenario(variant)
        prior = _extras(variant, variant % 3, tool_names, expected["name"]) + prior
        tools = [PROBE_TOOLS[name] for name in tool_names]
        prompt = chat_template.render(
            [tasks.SYSTEM, *prior, {"role": "user", "content": final}],
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        gold_args = json.dumps(expected["args"], sort_keys=True, separators=(",", ":"))
        gold = (
            f"<|message_model|>{expected['name']}<|content_invoke_tool_json|>"
            f'{{"name":{json.dumps(expected["name"])},"args":{gold_args}}}'
            "<|end_message|><|content_model_end_sampling|>"
        )
        items.append(
            {
                "id": f"probe-{len(items):02d}",
                "category": "agentic",
                "subcategory": "ref_" + expected["name"],
                "scorer": "agentic",
                "max_tokens": BUDGET,
                "prompt": prompt,
                "expected": {"name": expected["name"], "args": expected["args"], "tools": tools},
                "gold": gold,
                "meta": {"turn": "reference_earlier_call", "prior_calls": 1 + variant % 3},
            }
        )
    return items


CONTROLS = (
    (
        "reservation",
        "What check-in date and how many nights did that reservation use? "
        "Answer in one short sentence and do not call a tool.",
        "check-in",
    ),
    (
        "rental",
        "Which city was that rental in, and for how many days? "
        "Answer in one short sentence and do not call a tool.",
        "days",
    ),
    (
        "meeting",
        "What duration was that meeting and when did it start? "
        "Answer in one short sentence and do not call a tool.",
        "minutes",
    ),
    (
        "order",
        "What was the total on that order? Answer in one short sentence and do not call a tool.",
        "cents",
    ),
    (
        "transfer",
        "Which source account did that transfer leave from? "
        "Answer in one short sentence and do not call a tool.",
        "AC-",
    ),
    (
        "invoice",
        "What was the amount on that invoice? "
        "Answer in one short sentence and do not call a tool.",
        "cents",
    ),
    (
        "flight",
        "What route did that flight search use? "
        "Answer in one short sentence and do not call a tool.",
        "to",
    ),
    (
        "email",
        "What was the subject of the email you sent? "
        "Answer in one short sentence and do not call a tool.",
        "Booking",
    ),
)


def _control_items(reasoning_effort: float) -> list[dict]:
    items = []
    for index in range(20):
        name, question, marker = CONTROLS[index % len(CONTROLS)]
        k = index % 10
        if name == "reservation":
            city, check_in, nights = CITIES[k], DATES[k % len(DATES)], 2 + (k % 4)
            prior = _prior(
                f"Book a hotel in {city} for {nights} nights starting {check_in}.",
                "book_hotel",
                {"city": city, "check_in": check_in, "nights": nights, "guests": 2},
                f"Booked in {city}: check-in {check_in}, {nights} nights.",
            )
            answer = f"The reservation used check-in {check_in} for {nights} nights."
            must = [check_in, str(nights)]
            tool_names = ["book_hotel", "search_flights", "send_email"]
        elif name == "rental":
            city, pickup, days = CITIES[(k + 1) % 10], DATES[(k + 1) % len(DATES)], 2 + (k % 3)
            prior = _prior(
                f"Rent a car in {city} from {pickup} for {days} days.",
                "rent_car",
                {"city": city, "pickup_date": pickup, "days": days, "car_class": "compact"},
                f"Reserved a compact car in {city} for {days} days.",
            )
            answer = f"The rental was in {city} for {days} days."
            must = [city, str(days)]
            tool_names = ["rent_car", "book_hotel", "get_weather"]
        elif name == "meeting":
            person = PEOPLE[(k + 2) % 10]
            address = f"{person.lower()}@example.com"
            start = f"2026-12-{3 + (k % 20):02d}T10:00"
            duration = 30 + 15 * (k % 3)
            prior = _prior(
                f"Schedule a {duration}-minute meeting with {address} starting {start}.",
                "schedule_meeting",
                {"participants": [address], "start_iso": start, "duration_minutes": duration},
                "Scheduled.",
            )
            answer = f"The meeting lasted {duration} minutes and started at {start[11:16]}."
            must = [str(duration), start[11:16]]
            tool_names = ["schedule_meeting", "lookup_customer", "send_email"]
        elif name == "order":
            total = 1500 + 100 * k
            prior = _prior(
                "Order ramen and tea from Harbor Kitchen for delivery by 2026-12-06T19:00.",
                "order_food",
                {
                    "restaurant": "Harbor Kitchen",
                    "items": ["ramen", "tea"],
                    "deliver_by": "2026-12-06T19:00",
                    "total_cents": total,
                },
                f"Order confirmed, total {total} cents.",
            )
            answer = f"The order total was {total} cents."
            must = [str(total), "cents"]
            tool_names = ["order_food", "get_weather", "send_email"]
        elif name == "transfer":
            from_account, to_account = f"AC-{730 + k:03d}", f"AC-{830 + k:03d}"
            amount = 2000 + 250 * (k % 5)
            prior = _prior(
                f"Transfer {amount} cents from {from_account} to {to_account}.",
                "transfer_money",
                {
                    "from_account": from_account,
                    "to_account": to_account,
                    "amount_cents": amount,
                    "currency": "USD",
                },
                f"Transferred {amount} cents.",
            )
            answer = f"The transfer left {from_account}."
            must = [from_account]
            tool_names = ["transfer_money", "lookup_customer", "send_email"]
        elif name == "invoice":
            cid, amount = f"C-{7400 + k}", 2600 + 300 * (k % 5)
            prior = _prior(
                f"Create an invoice for customer {cid} for {amount} cents in EUR, due 2027-01-07.",
                "create_invoice",
                {
                    "customer_id": cid,
                    "amount_cents": amount,
                    "currency": "EUR",
                    "due_date": "2027-01-07",
                },
                f"Invoice created for {cid}, {amount} cents in EUR.",
            )
            answer = f"The invoice amount was {amount} cents."
            must = [str(amount), "cents"]
            tool_names = ["create_invoice", "lookup_customer", "send_email"]
        elif name == "flight":
            origin, destination = CITIES[(k + 3) % 10], CITIES[(k + 6) % 10]
            day = DATES[(k + 2) % len(DATES)]
            prior = _prior(
                f"Search flights from {origin} to {destination} on {day} for 2 passengers.",
                "search_flights",
                {"origin": origin, "destination": destination, "date": day, "passengers": 2},
                "2 results: 08:05 and 17:30",
            )
            answer = f"The search was from {origin} to {destination}."
            must = [origin, destination]
            tool_names = ["search_flights", "get_weather", "book_hotel"]
        else:  # email
            person = PEOPLE[(k + 4) % 10]
            address = f"{person.lower()}@example.com"
            subject = f"Booking C-{7500 + k}"
            prior = _prior(
                f'Send an email to {address} with subject "{subject}" and body "Confirmed."',
                "send_email",
                {"to": address, "subject": subject, "body": "Confirmed."},
                "Sent.",
            )
            answer = f"The subject was {subject}."
            must = [subject]
            tool_names = ["send_email", "lookup_customer", "get_weather"]
        tools = [PROBE_TOOLS[name] for name in tool_names]
        prompt = chat_template.render(
            [CONTROL_SYSTEM, *prior, {"role": "user", "content": question}],
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        items.append(
            {
                "id": f"probe-{80 + index:02d}",
                "category": "agentic",
                "subcategory": "control",
                "scorer": "agentic",
                "max_tokens": BUDGET,
                "prompt": prompt,
                "expected": {
                    "name": None,
                    "args": None,
                    "tools": tools,
                    "control": True,
                    "must_include": must,
                },
                "gold": (
                    "<|message_model|><|content_text|>"
                    + answer
                    + "<|end_message|><|content_model_end_sampling|>"
                ),
                "meta": {"turn": "plain_chat_control"},
            }
        )
    return items


def build_plan(*, reasoning_effort: float = 0.0, max_model_len: int | None = None) -> dict:
    """The 100-item probe plan, with its own hash and a separate ``set`` name."""
    items = _tool_items(reasoning_effort) + _control_items(reasoning_effort)
    mdd = stats.mdd(len(items), 0.10)
    plan = tasks.finalize_plan(
        items,
        reasoning_effort=reasoning_effort,
        wanted=("agentic",),
        usable=(),
        max_model_len=max_model_len,
        set_name="agentic-probe",
        extra_meta={
            "probe": {
                "items": len(items),
                "tool_call_items": len(TOOL_SCENARIOS) * 10,
                "control_items": 20,
                "assumed_discordance": 0.10,
                "mdd_pp": round(100 * mdd, 1) if mdd is not None else None,
                "note": (
                    "MDE at n=100 and 10% discordance; the observed hotel "
                    "discordance (stock right, hi2 wrong) is 4/5."
                ),
            }
        },
    )
    return plan
