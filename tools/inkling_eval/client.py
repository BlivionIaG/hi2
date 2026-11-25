# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Minimal OpenAI-compatible client: greedy text/chat and model metadata.

The main harness renders the Inkling prompt itself (``chat_template.py``) and
sends it to ``/v1/completions``, so both endpoints receive byte-identical
prompts and no server-side template version can change the comparison.  The
vision set sends messages (including base64 ``image_url`` parts) to
``/v1/chat/completions``; that endpoint must render the template and attach the
image to the prompt.  Only the standard library is needed.
"""

import json
import time
import urllib.error
import urllib.request


class TransportError(RuntimeError):
    """The request failed after retries; the item is incomplete, not wrong."""


class Client:
    def __init__(
        self,
        url: str,
        model: str | None = None,
        *,
        timeout: float = 1800.0,
        retries: int = 2,
        seed: int = 0,
        stop: list[str] | None = None,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.seed = seed
        self.stop = stop or ["<|content_model_end_sampling|>"]

    def _request(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url + path, data, headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read()[:600].decode("utf-8", "replace")
                if error.code < 500 and error.code != 429:
                    raise TransportError(f"HTTP {error.code}: {detail}") from error
                if attempt == self.retries:
                    raise TransportError(f"HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if attempt == self.retries:
                    raise TransportError(str(error)) from error
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def model_name(self) -> str:
        if self.model:
            return self.model
        payload = self._request("/v1/models")
        models = payload.get("data") or []
        if not models or not models[0].get("id"):
            raise TransportError("/v1/models returned no model id; pass --model")
        self.model = models[0]["id"]
        return self.model

    def max_model_len(self) -> int | None:
        try:
            payload = self._request("/v1/models")
        except TransportError:
            return None
        for entry in payload.get("data") or []:
            if entry.get("max_model_len"):
                return int(entry["max_model_len"])
        return None

    def complete(self, prompt: str, max_tokens: int) -> dict:
        """One greedy completion; ``text`` plus usage and finish reason."""
        body = {
            "model": self.model_name(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "seed": self.seed,
            "stop": self.stop,
            # Keep <|content_thinking|>/<|content_text|>/<|end_message|> so the
            # scorer can separate the reasoning preamble from the committed
            # answer. vLLM defaults this to true and strips the boundary.
            "skip_special_tokens": False,
        }
        payload = self._request("/v1/completions", body)
        choice = (payload.get("choices") or [{}])[0]
        return {
            "text": choice.get("text") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") or {},
            "model": payload.get("model") or self.model,
        }

    def chat(
        self,
        messages: list[dict],
        max_tokens: int,
        *,
        reasoning_effort: str = "none",
    ) -> dict:
        """One greedy multimodal chat completion.

        The Inkling chat template inserts the thinking-effort system turn from
        ``reasoning_effort``; ``"none"`` maps to numeric 0.0 in the server's
        Inkling renderer.  ``skip_special_tokens=false`` keeps the answer
        channel so the same scorers can separate a reasoning preamble.
        """
        body = {
            "model": self.model_name(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "seed": self.seed,
            "stop": self.stop,
            "skip_special_tokens": False,
            "reasoning_effort": reasoning_effort,
            "stream": False,
        }
        payload = self._request("/v1/chat/completions", body)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text") or "" if isinstance(part, dict) else str(part)
                for part in content
            )
        reasoning = message.get("reasoning_content") or ""
        if reasoning and "<|content_thinking|>" not in content:
            content = (
                f"<|content_thinking|>{reasoning}<|end_message|>" + content
            )
        return {
            "text": content,
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") or {},
            "model": payload.get("model") or self.model,
        }
