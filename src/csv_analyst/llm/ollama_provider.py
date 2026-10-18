"""Ollama-backed LLMProvider, using the /api/generate endpoint in JSON mode."""

import math
from typing import Callable, Optional

import requests

from csv_analyst.config import settings
from csv_analyst.llm.base import LLMProviderError, PromptTooLongError

# Conservative token estimate for the overflow guard: qwen2.5 measured ~3.35
# characters per token on our prompts (schema, rules, CSV values), so 3.0
# over-estimates by ~10 %. Text with many non-Latin characters packs fewer
# characters per token and could still slip past -- a known limit.
CHARS_PER_TOKEN = 3.0
# Room left for the model's reply (planner output ~100 tokens, answers less).
OUTPUT_RESERVE_TOKENS = 512


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class OllamaProvider:
    def __init__(
        self, model: str, host: Optional[str] = None, timeout_s: Optional[int] = None,
        num_ctx: Optional[int] = None,
        on_response: Optional[Callable[[str, dict], None]] = None,
    ):
        self.model = model
        self.host = host or settings.ollama_host
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.num_ctx = num_ctx or settings.llm_num_ctx
        # Measurement hook (card H-S timing): called with (prompt, Ollama's
        # response JSON) after each successful call -- token counts and
        # durations live there. Unused by the app.
        self.on_response = on_response

    def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        schema: Optional[dict] = None,
    ) -> str:
        # `num_ctx` is set explicitly -- some models (e.g. deepseek-coder-v2:16b)
        # default to a 160K+ token context window, which forces llama.cpp to
        # allocate a multi-GB KV cache before it can serve a single request
        # (observed: 43GB, on CPU, taking long enough to look like a hang).
        # Our prompts are a few thousand tokens at most.
        #
        # PLAN.md Phase 6B.2: Ollama 0.32+ accepts a JSON Schema object here,
        # not only the string "json" -- "json" only constrains the response
        # to *be* JSON, not to match a required shape, which is why a model
        # could previously return valid JSON with a truncated/malformed
        # `code` string. A schema constrains that too.
        needed = estimate_tokens(prompt) + estimate_tokens(system or "")
        budget = self.num_ctx - OUTPUT_RESERVE_TOKENS
        if needed > budget:
            raise PromptTooLongError(
                f"prompt too long for the model's context window: about {needed} tokens, "
                f"{budget} available (num_ctx {self.num_ctx} minus {OUTPUT_RESERVE_TOKENS} for the reply)"
            )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "format": schema if schema is not None else "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                **({"seed": seed} if seed is not None else {}),
            },
        }
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if self.on_response is not None:
                self.on_response(prompt, data)
            # A "thinking" model (e.g. qwen3.5) can put its entire structured
            # answer in `thinking` and leave `response` empty -- Ollama splits
            # a model's own <think> tags into that field, it isn't reasoning
            # that got cut off (observed: done_reason="stop", a two-digit
            # eval_count). Only used when `response` is empty, so a normal
            # model's behavior is completely unchanged.
            return data["response"] or data.get("thinking") or ""
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"Ollama request failed ({self.model}): {exc}") from exc
