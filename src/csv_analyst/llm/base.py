"""LLMProvider protocol. Swapping backends (Ollama today, a cloud adapter later)
means implementing this interface — nothing above it should import a provider directly.
"""

from typing import Optional, Protocol


class LLMProviderError(Exception):
    """Raised by a provider on a transport-level failure (timeout, connection
    error, non-2xx response, ...). Callers treat this exactly like a
    malformed-output failure -- one sample lost, not a crash -- so a slow or
    momentarily unreachable backend degrades the run instead of killing it."""


class PromptTooLongError(LLMProviderError):
    """The prompt would not fit in the model's context window. Raised before
    the request is sent: Ollama would otherwise silently keep only the end of
    the prompt (losing the schema), and the model would answer from what was
    left. Deterministic -- retrying the same prompt cannot help."""


class LLMProvider(Protocol):
    def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        schema: Optional[dict] = None,
    ) -> str:
        """Return raw model output, constrained to JSON via the backend's JSON mode.
        Caller is responsible for parsing/validating — a provider must not silently
        repair malformed JSON.

        `schema` (PLAN.md Phase 6B.2) is an optional JSON Schema object. A
        backend that supports constrained/structured decoding (Ollama 0.32+)
        should use it to constrain the response shape, not just "is this
        JSON". A provider without that capability is free to ignore it and
        fall back to plain JSON mode -- the caller's one-shot repair path
        exists precisely for that case."""
        ...
