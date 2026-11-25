# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Deterministic scoring for the task-level evaluation.

Every scorer returns ``{"ok": bool, "detail": {...}}`` and never looks at a
clock, the network or the model's identity.  ``ok`` is what the paired
comparison counts; ``detail`` keeps the evidence (extracted answer, metrics,
sandbox status) next to the completion in the run JSONL.

The scorers are deliberately narrow: a wrong extraction is a measurement bug,
so each one states its rule and stores what it extracted.
"""

import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

from tools.inkling_eval import sandbox

# --------------------------------------------------------------------------- text


TEXT_CHANNEL = re.compile(
    r"<\|content_text\|>(.*?)(?=<\|end_message\|>|<\|message_model\|>"
    r"|<\|content_model_end_sampling\|>|\Z)",
    re.S,
)
_SPECIAL = re.compile(r"<\|[a-z_0-9]+\|>")


def remove_thinking(text: str) -> str:
    """Drop Inkling thinking channels, including a truncated unterminated one."""
    text = re.sub(
        r"<\|content_thinking\|>.*?(?:<\|end_message\|>|\Z)", " ", text, flags=re.S
    )
    if "</think>" in text:
        text = text.rsplit("", 1)[-1]
    return text


def _clean(text: str) -> str:
    for token in (
        "<|content_model_end_sampling|>",
        "<|end_message|>",
        "<|message_model|>",
        "<|message_user|>",
        "<|message_system|>",
        "<|content_thinking|>",
        "<|content_text|>",
    ):
        text = text.replace(token, "\n")
    return _SPECIAL.sub(" ", text).strip()


def answer_text(text: str) -> str:
    """The last Inkling ``<|content_text|>`` channel, else the cleaned fallback.

    vLLM strips special tokens by default, which concatenates the reasoning
    channel and the answer (``... should be "Paris".Paris``).  The client now
    requests ``skip_special_tokens=false``; this function keeps the committed
    text channel and drops thinking.  Old runs without markers still fall back
    to the cleaned whole completion.
    """
    text = text or ""
    blocks = TEXT_CHANNEL.findall(text)
    if blocks:
        return _clean(blocks[-1])
    return _clean(remove_thinking(text))


_SENTENCE_END = re.compile(r"(?<=[.!?。！？؟।])\s*")
_TRIM = " \t\r\n\"'`*_[](){}<>«»“”"


def final_segment(text: str) -> str:
    """The last sentence of the last non-empty answer line.

    This is the committed answer after the model's plain-text preamble, whether
    the preamble and answer share a content channel or the preamble sits in the
    thinking channel.  A fenced/code answer still goes through ``extract_code``.
    """
    body = answer_text(text)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    line = lines[-1] if lines else ""
    parts = [part.strip() for part in _SENTENCE_END.split(line) if part.strip()]
    segment = parts[-1] if parts else line
    return segment.strip(_TRIM).strip()


_FENCE = re.compile(r"```([A-Za-z0-9_+.-]*)[ \t]*\n(.*?)(```|\Z)", re.S)


def _fenced(text: str) -> list[tuple[str, str, bool]]:
    blocks = []
    for match in _FENCE.finditer(text):
        language = match.group(1).lower()
        if language in ("typescript", "tsx"):
            language = "ts"
        elif language in ("javascript", "js"):
            language = "js"
        blocks.append((language, match.group(2), match.group(3) == "```"))
    return blocks


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """``(language, code)`` for each fenced block with a newline after the fence.

    Requiring the newline keeps a prose mention such as ``one ```python code
    block`` from swallowing the real block that follows.
    """
    return [(language, code) for language, code, _ in _fenced(text)]


CODE_HINTS = {
    "python": ("python", "py", ""),
    "c": ("c", "h", ""),
    "typescript": ("ts", "typescript", "js", "javascript", ""),
}


def extract_code(text: str, entry: str, language: str) -> str:
    """The last complete block defining ``entry``, else the largest complete block.

    A truncated completion can end in an unterminated block; a complete earlier
    block is better evidence than the cut-off one.  If every block is
    unterminated, keep the largest.
    """
    text = remove_thinking(text).replace("<|content_model_end_sampling|>", "")
    hints = CODE_HINTS[language]
    blocks = [
        (code, closed)
        for lang, code, closed in _fenced(text)
        if lang in hints
    ]

    def defines(code: str) -> bool:
        return bool(
            re.search(
                rf"\b(?:def|function|int|float|double|char|void|struct)\b[^\n]*\b{re.escape(entry)}\b",
                code,
            )
        )

    complete = [code for code, closed in blocks if closed and defines(code)]
    if complete:
        return complete[-1].strip()
    partial = [code for code, closed in blocks if defines(code)]
    if partial:
        return partial[-1].strip()
    complete = [code for code, closed in blocks if closed]
    if complete:
        return max(complete, key=len).strip()
    if blocks:
        return max(blocks, key=lambda pair: len(pair[0]))[0].strip()
    return text.strip()


# --------------------------------------------------------------------------- numbers


def last_number(text: str) -> str | None:
    """The final numeric answer, in the order the model usually commits to it."""
    body = answer_text(text)
    patterns = (
        r"the answer is\s*\$?-?([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\\boxed\{\s*-?([0-9][0-9,]*(?:\.[0-9]+)?)\s*\}",
    )
    for pattern in patterns:
        found = re.findall(pattern, body, flags=re.I)
        if found:
            return found[-1].replace(",", "")
    found = re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", body)
    return found[-1].replace(",", "") if found else None


def canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None
    text = format(number.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalize_word(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u064b-\u0652]", "", text)  # Arabic diacritics
    text = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", " ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------- chrf

_CHRF_ORDERS = range(1, 7)
_CHRF_BETA = 2.0


def chrf(hypothesis: str, reference: str) -> float:
    """chrF2 over character n-grams 1..6, with whitespace runs collapsed."""
    hyp = " ".join(hypothesis.split())
    ref = " ".join(reference.split())
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for order in _CHRF_ORDERS:
        hyp_grams = Counter(hyp[i : i + order] for i in range(len(hyp) - order + 1))
        ref_grams = Counter(ref[i : i + order] for i in range(len(ref) - order + 1))
        match = sum((hyp_grams & ref_grams).values())
        precisions.append(match / max(1, sum(hyp_grams.values())))
        recalls.append(match / max(1, sum(ref_grams.values())))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    beta2 = _CHRF_BETA**2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


# --------------------------------------------------------------------------- language id

SCRIPT_RANGES = {
    "ru": ((0x0400, 0x04FF),),
    "zh": ((0x3400, 0x9FFF),),
    "ja": ((0x3040, 0x30FF),),
    "ko": ((0xAC00, 0xD7A3), (0x1100, 0x11FF)),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hi": ((0x0900, 0x097F),),
}
LATIN_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "den", "dem", "des", "ein", "eine", "zu", "sich", "auf", "für", "vor", "wird", "sie", "um", "im", "am", "einer", "einem"},
    "fr": {"le", "la", "les", "de", "des", "du", "et", "est", "une", "un", "dans", "pour", "pas", "que", "qui", "avant", "elle", "chaque", "à", "au", "aux"},
    "es": {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es", "por", "para", "con", "no", "antes", "cada", "ella", "a", "al"},
    "pt": {"o", "a", "os", "as", "de", "que", "e", "em", "um", "uma", "é", "do", "da", "para", "com", "não", "antes", "cada", "ela", "às", "ao", "todas"},
    "pl": {"i", "w", "z", "na", "do", "nie", "jest", "się", "to", "że", "ale", "jak", "po", "co", "przed", "każdego", "książkę", "o"},
    "nl": {"de", "het", "een", "en", "van", "dat", "is", "niet", "op", "te", "met", "aan", "er", "maar", "voor", "elke", "zij", "leest", "om"},
    "tr": {"ve", "bir", "bu", "da", "de", "için", "ile", "değil", "olarak", "çok", "daha", "ne", "her", "önce", "kitap", "okur"},
    "vi": {"và", "của", "là", "không", "được", "trong", "những", "có", "người", "một", "cho", "với", "mỗi", "trước", "cô", "đọc"},
}


def _script_ratio(text: str, lang: str) -> float:
    ranges = SCRIPT_RANGES[lang]
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0
    inside = sum(
        1
        for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in ranges)
    )
    return inside / total


LATIN_MARKS = {
    "de": "äöüß",
    "fr": "àâçéèêëîïôùûüœ",
    "es": "áéíóúñ¿¡",
    "pt": "áâãàçéêíóôõú",
    "pl": "ąćęłńóśźż",
    "nl": "éëïöü",
    "tr": "çğıöşü",
    "vi": "ăâđêôơưàáảãạằắẳẵặèéẻẽẹìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ",
}


def language_signal(text: str, lang: str) -> dict:
    """A deterministic language guard: ``match``, ``mismatch`` or ``undetermined``.

    Script languages use the script share of non-space characters (>= 25%; Japanese
    also accepts >= 5% kana; Chinese rejects kana).  Latin-script languages use two
    hits from a short function-word list or two characteristic diacritics.  English
    text is left ``undetermined`` here and is caught by the chrF floor; a reply in
    another Latin language that scores two more hits than the target is a mismatch.
    """
    body = answer_text(text)
    if not body:
        return {"state": "mismatch", "reason": "empty"}
    if lang in SCRIPT_RANGES:
        han = _script_ratio(body, "zh")
        kana = _script_ratio(body, "ja")
        if lang == "zh":
            if kana >= 0.05:
                return {"state": "mismatch", "reason": "kana", "han": round(han, 3)}
            state = "match" if han >= 0.25 else ("mismatch" if han < 0.05 else "undetermined")
            return {"state": state, "han": round(han, 3)}
        if lang == "ja":
            if kana >= 0.05:
                return {"state": "match", "kana": round(kana, 3)}
            if han >= 0.25:
                return {"state": "undetermined", "kana": round(kana, 3)}
            return {"state": "mismatch", "reason": "no_script"}
        ratio = _script_ratio(body, lang)
        state = "match" if ratio >= 0.25 else ("mismatch" if ratio < 0.05 else "undetermined")
        return {"state": state, "script_ratio": round(ratio, 3)}
    words = re.findall(r"[^\W\d_]+", body.casefold())
    if not words:
        return {"state": "mismatch", "reason": "no_words"}
    latin = sum(
        1
        for ch in body
        if "a" <= ch.casefold() <= "z"
        or ch in "àâäçéèêëîïôöùûüñáíóúãõåæøœłżźćńś"
    )
    if latin / max(1, len([c for c in body if not c.isspace()])) < 0.6:
        return {"state": "mismatch", "reason": "not_latin"}
    hits = {code: len(set(words) & stop) for code, stop in LATIN_STOPWORDS.items()}
    marks = sum(1 for ch in body.casefold() if ch in LATIN_MARKS[lang])
    expected_hits = hits[lang]
    if expected_hits >= 2 or marks >= 2:
        return {"state": "match", "hits": expected_hits, "marks": marks}
    other_code, other_hits = max(hits.items(), key=lambda pair: pair[1])
    if other_code != lang and other_hits >= 2 and other_hits >= expected_hits + 2:
        return {"state": "mismatch", "hits": expected_hits, "other": other_code}
    return {"state": "undetermined", "hits": expected_hits, "marks": marks}


# --------------------------------------------------------------------------- tool calls

_JSON = json.JSONDecoder()


def _raw_object(text: str, start: int) -> dict | list | None:
    while start < len(text) and text[start] not in "{[":
        start += 1
    if start >= len(text):
        return None
    try:
        value, _ = _JSON.raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return value


def _calls_from_payload(payload, fmt: str) -> list[dict]:
    if isinstance(payload, dict) and "tool_calls" in payload:
        payload = payload["tool_calls"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    calls = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name") if isinstance(fn, dict) else None
        args = None
        if isinstance(fn, dict):
            args = fn.get("arguments", fn.get("args"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        calls.append(
            {
                "name": name if isinstance(name, str) else None,
                "args": args if isinstance(args, dict) else None,
                "format": fmt,
                "json_ok": True,
                "raw": json.dumps(entry, sort_keys=True)[:400],
            }
        )
    return calls


def parse_tool_calls(text: str) -> list[dict]:
    """Tool calls in the Inkling invoke marker, ``<tool_call>`` blocks or JSON fences."""
    body = remove_thinking(text)
    calls = []
    for match in re.finditer(r"<\|content_invoke_tool_json\|>", body):
        payload = _raw_object(body, match.end())
        calls.extend(_calls_from_payload(payload, "inkling"))
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", body, re.S):
        payload = _raw_object(match.group(1), 0)
        calls.extend(_calls_from_payload(payload, "tool_call"))
    for language, code in fenced_blocks(body):
        if language in ("json", ""):
            payload = _raw_object(code, 0)
            if payload is not None:
                calls.extend(_calls_from_payload(payload, "json"))
    if not calls:
        # Last resort: a bare object near the end of the reply, still schema-checked.
        position = body.rfind('{"name"')
        if position == -1:
            position = body.rfind('{"tool"')
        if position != -1:
            payload = _raw_object(body, position)
            calls.extend(_calls_from_payload(payload, "bare_json"))
    if not calls:
        return [
            {"name": None, "args": None, "format": "none", "json_ok": False, "raw": body[:200]}
        ]
    return calls


def _schema_errors(value, schema: dict, path: str = "args") -> list[str]:
    errors = []
    if not isinstance(schema, dict):
        return errors
    expected = schema.get("type")
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }
    if expected in checks and not checks[expected](value):
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, entry in enumerate(value):
            errors.extend(_schema_errors(entry, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key}: required")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: not declared")
        for key, sub in properties.items():
            if key in value:
                errors.extend(_schema_errors(value[key], sub, f"{path}.{key}"))
    return errors


def validate_args(args: dict | None, tool: dict) -> list[str]:
    if args is None:
        return ["arguments are not a JSON object"]
    schema = tool.get("function", tool).get("parameters") or {}
    return _schema_errors(args, schema)


def canon(value):
    """A comparison form: whitespace/case-insensitive strings, rounded numbers."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", round(float(value), 9))
    if isinstance(value, str):
        return ("str", " ".join(value.strip().casefold().split()))
    if isinstance(value, list):
        return ("list", tuple(canon(item) for item in value))
    if isinstance(value, dict):
        return ("obj", tuple(sorted((key, canon(item)) for key, item in value.items())))
    return ("null", None)


def arg_triples(name: str, args: dict) -> set:
    triples = {(name, "", "")}
    for key, value in args.items():
        triples.add((name, key, canon(value)))
    return triples


# --------------------------------------------------------------------------- scorers


def score_math(item: dict, completion: str) -> dict:
    expected = canonical_number(item["expected"]["answer"])
    got = canonical_number(last_number(completion))
    return {"ok": got is not None and got == expected, "detail": {"answer": got, "expected": expected}}


def score_qa_numeric(item: dict, completion: str) -> dict:
    expected = canonical_number(item["expected"]["answer"])
    got = canonical_number(last_number(completion))
    return {
        "ok": got is not None and got == expected,
        "detail": {"answer": got, "expected": expected, "lang": item["expected"].get("lang")},
    }


def score_vision_chart(item: dict, completion: str) -> dict:
    """A chart-read value with a stated absolute tolerance (no judge)."""
    expected = item["expected"]
    want = Decimal(canonical_number(expected["answer"]))
    got_text = last_number(completion)
    got = Decimal(canonical_number(got_text)) if got_text is not None else None
    tolerance = Decimal(str(expected.get("tolerance", 0)))
    ok = got is not None and abs(got - want) <= tolerance
    return {
        "ok": ok,
        "detail": {
            "answer": str(got) if got is not None else None,
            "expected": str(want),
            "tolerance": str(tolerance),
            "label": expected.get("label"),
        },
    }


def score_qa_word(item: dict, completion: str) -> dict:
    variants = {normalize_word(v) for v in item["expected"]["answers"]}
    segment = final_segment(completion)
    words = normalize_word(segment).split()
    tails = {" ".join(words[index:]) for index in range(max(0, len(words) - 3), len(words))}
    got = " ".join(words)
    return {
        "ok": bool(words) and bool((tails | {got}) & variants),
        "detail": {
            "answer": got[:120],
            "segment": segment[:160],
            "expected": sorted(variants),
            "lang": item["expected"].get("lang"),
        },
    }


def score_chrf(item: dict, completion: str) -> dict:
    expected = item["expected"]
    body = final_segment(completion)
    score = chrf(body, expected["reference"])
    signal = language_signal(body, expected["lang"])
    return {
        "ok": bool(body) and score >= expected["min_chrf"] and signal["state"] != "mismatch",
        "detail": {
            "chrf": round(score, 4),
            "min_chrf": expected["min_chrf"],
            "language_state": signal["state"],
            "language_signal": signal,
            "lang": expected["lang"],
            "answer": body[:300],
        },
    }


def score_needle(item: dict, completion: str) -> dict:
    expected = item["expected"]
    body = answer_text(completion)
    value = expected["value"]
    found = value.casefold() in body.casefold()
    distractors = [d for d in expected.get("distractors", []) if d.casefold() in body.casefold()]
    return {
        "ok": found,
        "detail": {
            "found": found,
            "distractors_present": len(distractors),
            "answer": body[:300],
        },
    }


def score_summarize(item: dict, completion: str) -> dict:
    expected = item["expected"]
    body = answer_text(completion)
    marker = None
    for match in re.finditer(r"(?is)\banswer\s*:\s*(.+)$", body):
        marker = match
    if marker is None:
        marker = None
        for match in re.finditer(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+)$", body):
            marker = match
    if marker is not None:
        summary, answer = body[: marker.start()], marker.group(1)
    else:
        summary, answer = "", body.splitlines()[-1] if body else ""
    words = len(summary.split())
    got = normalize_word(answer)
    want = normalize_word(expected["answer"])
    return {
        "ok": 0 < words <= expected["summary_max_words"] and bool(got) and got == want,
        "detail": {
            "summary_words": words,
            "max_words": expected["summary_max_words"],
            "answer": answer.strip()[:200],
            "expected": expected["answer"],
        },
    }


def score_agentic(item: dict, completion: str) -> dict:
    expected = item["expected"]
    tools = {tool["function"]["name"]: tool for tool in expected["tools"]}
    calls = parse_tool_calls(completion)
    if expected.get("control"):
        # A plain-chat control is correct only when the reply answers directly:
        # an invoke marker (even malformed) or any parsed call is the opposite error.
        body = answer_text(completion)
        attempted = any(
            marker in (completion or "")
            for marker in ("<|content_invoke_tool_json|>", "<tool_call>")
        ) or any(call["format"] != "none" for call in calls)
        missing = [
            token
            for token in expected.get("must_include") or []
            if token.casefold() not in body.casefold()
        ]
        return {
            "ok": not attempted and not missing and bool(body.strip()),
            "detail": {
                "control": True,
                "attempted_tool_call": attempted,
                "missing": missing,
                "answer": body[:200],
            },
        }
    graded = None
    for call in calls:
        if call["name"] == expected["name"] and call["args"] is not None:
            graded = call
            break
    if graded is None:
        graded = calls[0]
    errors = validate_args(graded["args"], tools.get(graded["name"], {}))
    name_ok = graded["name"] == expected["name"]
    args_ok = graded["args"] is not None and canon(graded["args"]) == canon(expected["args"])
    expected_triples = arg_triples(expected["name"], expected["args"])
    actual_triples = arg_triples(graded["name"] or "", graded["args"] or {})
    matched = len(expected_triples & actual_triples)
    precision = matched / max(1, len(actual_triples))
    recall = matched / max(1, len(expected_triples))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "ok": bool(graded["json_ok"]) and name_ok and args_ok and not errors,
        "detail": {
            "parsed": len([c for c in calls if c["json_ok"]]),
            "format": graded["format"],
            "schema_ok": not errors,
            "schema_errors": errors[:4],
            "name_ok": name_ok,
            "args_ok": args_ok,
            "arg_f1": round(f1, 4),
            "call": {"name": graded["name"], "args": graded["args"]},
            "expected": {"name": expected["name"], "args": expected["args"]},
        },
    }


def score_code(item: dict, completion: str) -> dict:
    expected = item["expected"]
    language = expected["language"]
    code = extract_code(completion, expected["entry"], language)
    if not code:
        return {"ok": False, "detail": {"status": "empty", "language": language}}
    if language == "python":
        result = sandbox.run_python(code, expected["tests"], timeout=expected.get("timeout", 10.0))
    elif language == "c":
        result = sandbox.run_c(code, expected["tests"], timeout=expected.get("timeout", 10.0))
    elif language == "typescript":
        result = sandbox.run_typescript(code, expected["tests"], timeout=expected.get("timeout", 10.0))
    else:  # pragma: no cover - plan validation rejects this
        raise ValueError(f"unknown code language: {language}")
    detail = {
        "status": result["status"],
        "language": language,
        "engine": result.get("engine"),
        "transform": result.get("transform"),
        "network_isolation": result.get("network_isolation"),
        "stdout": (result.get("stdout") or "")[-400:],
        "stderr": (result.get("stderr") or result.get("error") or "")[-800:],
        "seconds": result.get("seconds"),
        "entry": expected["entry"],
        "source": expected.get("source"),
        "code_chars": len(code),
    }
    return {"ok": result["status"] == "pass", "detail": detail}


SCORERS = {
    "math": score_math,
    "qa_numeric": score_qa_numeric,
    "qa_word": score_qa_word,
    "vision_chart": score_vision_chart,
    "chrf": score_chrf,
    "needle": score_needle,
    "summarize": score_summarize,
    "agentic": score_agentic,
    "code": score_code,
}


def score(item: dict, completion: str) -> dict:
    """Score one completion; any traceback is returned as an explicit score error."""
    try:
        return SCORERS[item["scorer"]](item, completion)
    except Exception as error:  # pragma: no cover - exercised by the runner contract
        return {
            "ok": False,
            "detail": {"score_error": f"{type(error).__name__}: {error}"},
        }
