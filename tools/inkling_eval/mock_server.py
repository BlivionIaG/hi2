# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Canned OpenAI-compatible server for tests and the dry-run.

Answers are keyed by the SHA-256 of the prompt, so a task is served its own gold
completion without embedding an item id in the prompt.  An unknown prompt gets a
deliberately wrong answer, which makes a prompt mismatch visible as a failed
item instead of a false pass.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UNKNOWN_ANSWER = "I don't know."


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_answers(items, path: Path) -> int:
    seen = {}
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in items:
            key = prompt_key(item["prompt"])
            if key in seen and seen[key] != item["gold"]:
                raise ValueError(f"two different answers for prompt {key[:12]}")
            seen[key] = item["gold"]
            handle.write(json.dumps({"prompt_sha256": key, "completion": item["gold"]}) + "\n")
    return len(seen)


def load_answers(path: Path) -> dict[str, str]:
    answers = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["prompt_sha256"]
            if key in answers and answers[key] != row["completion"]:
                raise ValueError(f"conflicting canned answers for {key[:12]}")
            answers[key] = row["completion"]
    return answers


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    answers: dict[str, str] = {}
    model: str = "canned-inkling"
    max_model_len: int = 32768
    finish_reason: str = "stop"

    def log_message(self, *args):  # quiet
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model,
                            "object": "model",
                            "created": 0,
                            "max_model_len": self.max_model_len,
                        }
                    ],
                },
            )
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON"}})
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            self._json(400, {"error": {"message": "prompt must be a string"}})
            return
        completion = self.answers.get(prompt_key(prompt), UNKNOWN_ANSWER)
        self._json(
            200,
            {
                "id": "cmpl-canned",
                "object": "text_completion",
                "created": 0,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "text": completion,
                        "finish_reason": self.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max(1, len(completion) // 4),
                    "total_tokens": max(1, len(prompt) // 4) + max(1, len(completion) // 4),
                },
            },
        )


def serve(
    answers: dict[str, str],
    *,
    model: str = "canned-inkling",
    max_model_len: int = 32768,
    finish_reason: str = "stop",
    host: str = "127.0.0.1",
    port: int = 0,
):
    """Start the server on a thread; returns ``(server, thread, url)``.

    ``finish_reason="length"`` turns every canned reply into a truncation, which
    the runner must count separately from a wrong answer.
    """
    handler = type(
        "CannedHandler",
        (_Handler,),
        {
            "answers": dict(answers),
            "model": model,
            "max_model_len": max_model_len,
            "finish_reason": finish_reason,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://{host}:{server.server_address[1]}"
