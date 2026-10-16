"""Geekbot's Ollama adapter for the ``csv_analyst`` package (step C2, lesson 06).

``OllamaCsvProvider`` fits the package's ``LLMProvider`` protocol (``csv_analyst.llm.base``) by
shape: it has a ``generate_json`` method with the right signature and does not inherit from
anything. It replaces the package's own ``OllamaProvider`` so that every model call uses
Geekbot's settings (host, ``num_ctx``, ``keep_alive``, timeout) and Geekbot's HTTP client
(httpx), and so a test can fake the server with ``httpx.MockTransport``.

The request is the same one the package's provider sends (that is what Gigalogy measured), plus
``keep_alive`` so the model stays loaded between questions (PLAN.md §6.5, R-2).
"""

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Self

import httpx

from csv_analyst.llm.base import LLMProviderError, PromptTooLongError
from csv_analyst.llm.ollama_provider import OUTPUT_RESERVE_TOKENS, estimate_tokens

# Called with Ollama's parsed response after each successful call. Ollama puts timings there,
# including ``load_duration`` (nanoseconds spent loading the model), which is how a model load
# during a question gets noticed (R-2).
ResponseHook = Callable[[Mapping[str, object]], None]


class OllamaCsvProvider:
    """Sends the CSV analyst's prompts to one Ollama model. Use as a context manager."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        num_ctx: int,
        keep_alive: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
        on_response: ResponseHook | None = None,
    ) -> None:
        self.model = model
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._on_response = on_response
        # transport lets tests plug in httpx.MockTransport instead of a real server.
        self._client = httpx.Client(base_url=host, transport=transport, timeout=timeout_s)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _check_fits(self, prompt: str, system: str | None) -> None:
        """Refuse a prompt that would not fit in the context window, before anything is sent.

        Ollama would otherwise silently keep only the end of an over-long prompt, which drops
        the schema at the top and makes the model answer from what is left. The estimate and the
        reply reserve are the package's own numbers (one source of truth); see
        ``csv_analyst.llm.ollama_provider``.
        """
        needed = estimate_tokens(prompt) + estimate_tokens(system or "")
        budget = self._num_ctx - OUTPUT_RESERVE_TOKENS
        if needed > budget:
            raise PromptTooLongError(
                f"prompt too long for the model's context window: about {needed} tokens, "
                f"{budget} available (num_ctx {self._num_ctx} minus {OUTPUT_RESERVE_TOKENS} "
                "for the reply)"
            )

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the model's raw reply, constrained to JSON by Ollama.

        ✍️ Written by Claude in prototyping mode (D-047); on the ✍️ backlog in STATUS.md. It
        does these, in order:

        1. Call ``self._check_fits(prompt, system)``. If it raises ``PromptTooLongError``,
           nothing may be sent.
        2. Build the JSON body for ``POST /api/generate``:
           - ``model`` (``self.model``), ``prompt``, ``system``;
           - ``format``: the ``schema`` if one was given, otherwise the string ``"json"``;
           - ``stream``: ``False`` (one reply, not a stream of pieces);
           - ``keep_alive``: ``self._keep_alive``;
           - ``options``: ``temperature`` and ``num_ctx`` (``self._num_ctx``), plus ``seed``
             **only when it is not None** (a missing seed means "any").
        3. Send it with ``self._client`` (its ``base_url`` is already the Ollama host).
        4. A timeout, a connection error, a non-2xx status, or a body that is not a JSON
           object must raise ``LLMProviderError``. Put the model name in the message and chain
           the original error (``raise ... from error``). The package counts this as one lost
           sample, not a crash.
        5. If ``self._on_response`` is set, call it with the parsed response.
        6. Return the reply text: ``data["response"]``; if that is empty, ``data.get("thinking")``
           (some models put the whole answer there); if that is missing too, ``""``. If
           ``response`` is missing or not text, raise ``LLMProviderError``.

        Never parse, trim or repair the model's JSON: the caller validates it and counts a bad
        reply as a failure. The tests are in ``tests/unit/test_csv_provider.py``.
        """
        self._check_fits(prompt, system)
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "format": schema if schema is not None else "json",
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self._num_ctx,
                # 0 is a real seed, so test for None, not for falsiness.
                **({"seed": seed} if seed is not None else {}),
            },
        }
        try:
            response = self._client.post("/api/generate", json=body)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as error:  # ValueError: the body was not JSON
            raise LLMProviderError(f"Ollama request failed ({self.model}): {error}") from error
        if not isinstance(data, dict):
            raise LLMProviderError(f"Ollama reply for {self.model} is not a JSON object")
        if self._on_response is not None:
            self._on_response(data)
        text = data.get("response")
        if not isinstance(text, str):
            raise LLMProviderError(f"Ollama reply for {self.model} has no response text")
        if text:
            return text
        thinking = data.get("thinking")
        return thinking if isinstance(thinking, str) else ""
