"""A thin Ollama HTTP client: version, pulled models, warm-up, unload and explicit swaps.

PLAN.md §6.5 rules R-2 and R-4: models are loaded (warmed up) on purpose, with a long
``keep_alive`` and the *same* ``num_ctx`` the real queries will use, and unloaded explicitly.
It is synchronous on purpose: the CLI is synchronous, and the server (S4.1) will call it
from worker threads, never on the event loop (PLAN.md §6.3).

Step C3 adds the swap between the chat model and the CSV model. The two do not fit in 10 GB
together, and if Ollama is left to evict one to make room for the other it can keep part of the
new model on the CPU, which wrecks speed. So a swap is three explicit moves: ask for the other
model to be unloaded, *wait until Ollama stops listing it*, then warm up the new one.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import httpx

from geekbot.config import Mode, Settings


class OllamaError(RuntimeError):
    """Ollama was unreachable or answered with something unexpected."""


def _with_tag(name: str) -> str:
    """Ollama treats ``name`` as ``name:latest``."""
    return name if ":" in name else f"{name}:latest"


def same_model(a: str, b: str) -> bool:
    """Whether two model names are the same model (``gemma3`` and ``gemma3:latest`` are)."""
    return _with_tag(a) == _with_tag(b)


def model_is_pulled(model: str, available: list[str]) -> bool:
    """Whether ``model`` is in ``available``. Ollama treats ``name`` as ``name:latest``."""
    return any(same_model(model, name) for name in available)


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
        return self._names(self._call("GET", "/api/tags").get("models"), "/api/tags")

    def loaded_models(self) -> list[str]:
        """Names of the models Ollama holds in memory right now (``GET /api/ps``).

        The HTTP twin of the ``ollama ps`` command. A swap uses it to be sure an unload has
        really finished; the doctor's residency check still uses the command's text, because
        only that shows the GPU/CPU split.
        """
        return self._names(self._call("GET", "/api/ps").get("models"), "/api/ps")

    @staticmethod
    def _names(models: object, path: str) -> list[str]:
        if not isinstance(models, list):
            raise OllamaError(f"Ollama {path} has no model list")
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


# --- explicit swaps (step C3, PLAN.md §6.5 R-4) ---------------------------------------------


class SwapClient(Protocol):
    """What a swap needs from the Ollama client (so tests can pass a fake)."""

    def loaded_models(self) -> list[str]: ...
    def unload(self, model: str) -> None: ...
    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float: ...


@dataclass(frozen=True)
class Patience:
    """How long to wait for an unload to finish, and how. ``sleep`` and ``clock`` are
    replaceable so a test can run a 30-second timeout in no time."""

    timeout_s: float = 30.0
    poll_s: float = 0.25
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class SwapResult:
    """What one swap did and how long each part took (seconds)."""

    loaded: str  # the model that is now warm
    unloaded: tuple[str, ...]  # the models that were loaded and have been unloaded
    unload_s: float  # asking for the unload, up to Ollama no longer listing the model
    warm_up_s: float  # loading the new model

    @property
    def total_s(self) -> float:
        return self.unload_s + self.warm_up_s


def unload_and_wait(
    client: SwapClient, names: Iterable[str], *, patience: Patience | None = None
) -> tuple[str, ...]:
    """Unload each of ``names`` that is loaded now, then wait until none of them is listed.

    Returns the names that were unloaded (empty when none was loaded). Raises ``OllamaError``
    if one is still listed after ``patience.timeout_s``: warming the next model up on top of it
    is exactly what the explicit swap exists to avoid.
    """
    wait = patience or Patience()
    wanted = list(names)
    to_unload = [m for m in client.loaded_models() if any(same_model(m, w) for w in wanted)]
    for name in to_unload:
        client.unload(name)
    deadline = wait.clock() + wait.timeout_s
    while to_unload:
        still = [m for m in client.loaded_models() if any(same_model(m, u) for u in to_unload)]
        if not still:
            break
        if wait.clock() >= deadline:
            raise OllamaError(
                f"{', '.join(still)} still loaded {wait.timeout_s:.0f} s after the unload request"
            )
        wait.sleep(wait.poll_s)
    return tuple(to_unload)


def swap_to(
    client: SwapClient,
    model: str,
    *,
    release: Iterable[str],
    num_ctx: int,
    keep_alive: str,
    patience: Patience | None = None,
) -> SwapResult:
    """Make ``model`` the loaded one: unload the ``release`` models, then warm ``model`` up.

    ✍️ Written by Claude in prototyping mode (D-047, step C3); on the backlog in STATUS.md.

    The order is the point (R-4): the other model must be *gone* before the new one loads, so
    the new one never has to share the GPU and is never left partly on the CPU. Only models that
    are loaded are unloaded (asking Ollama to unload one that isn't loaded is wasted time and
    would make the unload timing meaningless), and ``model`` itself is never unloaded, even if
    it appears in ``release``. ``num_ctx`` and ``keep_alive`` are what real queries use for
    ``model``, or Ollama would reload it on the first question (R-2).

    ``unload_s`` covers the unload request plus the wait for Ollama to stop listing the model,
    and is ``0.0``-ish when nothing was loaded. ``warm_up_s`` is the load itself.
    """
    wait = patience or Patience()
    started = wait.clock()
    others = [name for name in release if not same_model(name, model)]
    unloaded = unload_and_wait(client, others, patience=wait)
    unload_s = wait.clock() - started
    warm_up_s = client.warm_up(model, num_ctx=num_ctx, keep_alive=keep_alive)
    return SwapResult(loaded=model, unloaded=unloaded, unload_s=unload_s, warm_up_s=warm_up_s)


def swap_to_mode(
    client: SwapClient, settings: Settings, mode: Mode, *, patience: Patience | None = None
) -> SwapResult:
    """Swap to ``mode``'s model (chat or CSV), releasing the other mode's model."""
    if mode == "csv":
        model, other, num_ctx = (
            settings.resolved_csv_model,
            settings.resolved_chat_model,
            settings.csv_num_ctx,
        )
    else:
        model, other, num_ctx = (
            settings.resolved_chat_model,
            settings.resolved_csv_model,
            settings.chat_num_ctx,
        )
    return swap_to(
        client,
        model,
        release=[other],
        num_ctx=num_ctx,
        keep_alive=settings.keep_alive,
        patience=patience,
    )
