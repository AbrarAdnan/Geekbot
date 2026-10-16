"""A thin Ollama HTTP client: version, pulled models, warm-up and unload.

PLAN.md §6.5 rules R-2 and R-4: models are loaded (warmed up) on purpose, with a long
``keep_alive`` and the *same* ``num_ctx`` the real queries will use, and unloaded explicitly.
It is synchronous on purpose: the CLI is synchronous, and the server (S4.1) will call it
from worker threads, never on the event loop (PLAN.md §6.3).
"""

import time
from types import TracebackType
from typing import Self

import httpx


class OllamaError(RuntimeError):
    """Ollama was unreachable or answered with something unexpected."""


def model_is_pulled(model: str, available: list[str]) -> bool:
    """Whether ``model`` is in ``available``. Ollama treats ``name`` as ``name:latest``."""
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in available


class OllamaClient:
    """Talks to one Ollama server. Use as a context manager so the connection is closed."""

    def __init__(
        self,
        host: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 300.0,
    ) -> None:
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
        self._client.close()

    def _call(
        self, method: str, path: str, body: dict[str, object] | None = None
    ) -> dict[str, object]:
        try:
            response = self._client.request(method, path, json=body)
            response.raise_for_status()
            data: object = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OllamaError(f"Ollama {method} {path} failed: {error}") from error
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama {method} {path} returned {type(data).__name__}, not a dict")
        return data

    def version(self) -> str:
        version = self._call("GET", "/api/version").get("version")
        if not isinstance(version, str):
            raise OllamaError("Ollama /api/version has no version string")
        return version

    def list_models(self) -> list[str]:
        """Names of the pulled models, e.g. ``["gemma3:4b", "nomic-embed-text:latest"]``."""
        models = self._call("GET", "/api/tags").get("models")
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags has no model list")
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.append(entry["name"])
        return names

    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:
        """Load ``model`` into memory without generating text; return the seconds it took.

        An empty prompt makes Ollama load the model and stop. ``num_ctx`` must equal what
        real queries send: a different value makes Ollama reload the model (lesson 03 §4).
        """
        started = time.perf_counter()
        self._call(
            "POST",
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_ctx": num_ctx},
            },
        )
        return time.perf_counter() - started

    def unload(self, model: str) -> None:
        """Free the model's memory now (``keep_alive: 0``). Used for explicit swaps (R-4)."""
        self._call(
            "POST",
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )
