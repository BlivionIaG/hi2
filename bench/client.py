# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Minimal client for an OpenAI-compatible server (vLLM): greedy chat and text
completions, plus the vLLM ``/tokenize`` and ``prompt_logprobs`` extensions that the
multiple-choice task uses."""

import json
import time
import urllib.error
import urllib.request


class Client:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 900,
        retries: int = 2,
        chat_extra: dict | None = None,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.chat_extra = chat_extra or {}

    def post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.url + path, data, {"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:  # the server answered: do not retry
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def generate(self, prompt, max_tokens: int) -> tuple[str, dict]:
        """Greedy reply to a message list (chat) or a string (text completion)."""
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens}
        if isinstance(prompt, str):
            out = self.post("/v1/completions", body | {"prompt": prompt})
            text = out["choices"][0]["text"]
        else:
            out = self.post(
                "/v1/chat/completions", body | {"messages": prompt} | self.chat_extra
            )
            text = out["choices"][0]["message"]["content"] or ""
        return text, out.get("usage") or {}

    def tokenize(self, text: str) -> list[int]:
        body = {"model": self.model, "prompt": text, "add_special_tokens": False}
        return self.post("/tokenize", body)["tokens"]

    def prompt_logprobs(self, ids: list[int]) -> list[float | None]:
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
        plp = self.post("/v1/completions", body)["choices"][0]["prompt_logprobs"]
        return [None if e is None else e[str(t)]["logprob"] for t, e in zip(ids, plp)]


def strip_think(text: str) -> str:
    """The answer part of a reply that carries a reasoning block."""
    return text.rsplit("</think>", 1)[-1].lstrip() if "</think>" in text else text
