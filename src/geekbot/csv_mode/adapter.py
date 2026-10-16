"""Geekbot's side of a CSV conversation: one file, one model (step C2, lesson 06).

``CsvAnalyst`` joins the vendored ``csv_analyst`` package to Geekbot's settings. The package does
the analysis; this module decides three things around it:

- *what it talks to*: our own Ollama provider, with Geekbot's host, context size, ``keep_alive``
  and timeout (not the package's separate ``CSV_ANALYST_*`` settings);
- *how long a question may take*: ``csv_question_time_limit_s``, passed in per question;
- *what gets reported*: the model that answered and the time spent loading it, because a number
  without its model is not a result (D-043) and a model load during a question breaks R-2.

``CsvAnalyst.ask`` blocks for seconds. A caller on the event loop must run it in a worker
thread (PLAN.md §6.3); the CLI is synchronous, so it calls it directly.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from csv_analyst import CsvAnalystSession, IngestionError, RunRecord, Status, to_markdown
from csv_analyst.llm.base import LLMProvider
from geekbot.config import Settings
from geekbot.csv_mode.provider import OllamaCsvProvider
from geekbot.llm.models import OllamaClient, OllamaError, model_is_pulled

_NS_PER_MS = 1_000_000


class CsvLoadError(Exception):
    """The file is not a usable CSV. Geekbot's own type, so callers outside this package never
    need the vendored package's ``IngestionError`` (the import boundary, ``csv_mode/__init__``)."""


class LoadTimer:
    """Adds up Ollama's ``load_duration`` over every reply it is shown.

    Each Ollama reply says how long that call spent loading the model into memory (nanoseconds).
    The provider hands every reply to this object, so a question that needed a load shows it in
    the total. What counts as a "load event" is decided at C3, from measurements; here we only
    add the numbers up.
    """

    def __init__(self) -> None:
        self.total_ns = 0

    def __call__(self, response: Mapping[str, object]) -> None:
        load = response.get("load_duration")
        if isinstance(load, int) and not isinstance(load, bool):
            self.total_ns += load


@dataclass(frozen=True)
class CsvAnswer:
    """One answered question, with what the CLI footer and later the evals need."""

    record: RunRecord
    markdown: str  # the chat-ready reply: answer, confidence, SQL and result table
    model: str  # the model tag that produced it (D-043)
    seconds: float
    model_load_ms: int  # summed ``load_duration`` of this question's model calls

    @property
    def status(self) -> str:
        """``answered``, ``abstained``, ``needs_clarification`` or ``failed``."""
        return self.record.status.value

    @property
    def failed(self) -> bool:
        return self.record.status == Status.FAILED

    @property
    def reason(self) -> str | None:
        """Why there is no answer (abstained, needs clarification or failed); else ``None``."""
        return self.record.reason

    @property
    def retries(self) -> int:
        """How many times the package retried after a failed attempt."""
        reliability = self.record.reliability
        return reliability.retries if reliability is not None else 0

    @property
    def failed_steps(self) -> int:
        """How many steps of the pipeline failed on the way (a model call that errored or timed
        out, a rejected query, ...), even if a later attempt then succeeded."""
        return sum(not entry.ok for entry in self.record.trace)

    @property
    def slowest_step_ms(self) -> int:
        """The longest single step of this question (a model call, a query, ...), in ms.

        A question is slow either because it took many steps or because one step ran away; this
        tells the two apart (found at C3: two questions took exactly 61.2 s, a 60 s model-call
        timeout plus the rest).
        """
        return max((entry.ms for entry in self.record.trace), default=0)


def find_problem(settings: Settings, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Why ``csv ask`` cannot work right now (Ollama down, model not pulled), or ``None``.

    Without this, a missing model comes back as a ``failed`` answer with a service-error reason
    after the CSV is already loaded. One clear line up front is kinder.
    """
    model = settings.resolved_csv_model
    try:
        with OllamaClient(settings.ollama_host, transport=transport) as client:
            available = client.list_models()
    except OllamaError as error:
        return f"cannot reach Ollama at {settings.ollama_host}: {error}"
    if not model_is_pulled(model, available):
        return f"the CSV model {model} is not pulled: run `ollama pull {model}`"
    return None


class CsvAnalyst:
    """A loaded CSV plus the model that answers questions about it. Use as a context manager."""

    def __init__(
        self,
        session: CsvAnalystSession,
        settings: Settings,
        load_timer: LoadTimer,
        owned_provider: OllamaCsvProvider | None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._load_timer = load_timer
        self._owned_provider = owned_provider

    @classmethod
    def open(
        cls,
        path: Path,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> Self:
        """Load and profile the CSV at ``path``.

        With no ``provider``, builds ours from ``settings`` (``transport`` is for tests, to fake
        the Ollama server). A ``provider`` passed in is used as it is and is not closed here.
        Raises ``CsvLoadError`` if the file is not a usable CSV.
        """
        load_timer = LoadTimer()
        owned: OllamaCsvProvider | None = None
        if provider is None:
            owned = OllamaCsvProvider(
                host=settings.ollama_host,
                model=settings.resolved_csv_model,
                num_ctx=settings.csv_num_ctx,
                keep_alive=settings.keep_alive,
                timeout_s=float(settings.csv_llm_timeout_s),
                transport=transport,
                on_response=load_timer,
            )
            provider = owned
        try:
            # The session reads the model name separately from the provider (it puts it in the
            # record and uses it for its own time-limit lookup), so both come from the same setting.
            session = CsvAnalystSession.from_path(
                path, model=settings.resolved_csv_model, llm=provider
            )
        except IngestionError as error:
            if owned is not None:
                owned.close()
            raise CsvLoadError(str(error)) from error
        except Exception:
            if owned is not None:
                owned.close()
            raise
        return cls(session, settings, load_timer, owned)

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
        if self._owned_provider is not None:
            self._owned_provider.close()

    @property
    def model(self) -> str:
        return self._settings.resolved_csv_model

    def describe(self) -> str:
        """The package's "Loaded **file**: N rows, M columns…" message. No model call."""
        return self._session.describe()

    def ask(self, question: str, *, seed_offset: int = 0) -> CsvAnswer:
        """Answer one question from the data, within the per-question time cap.

        ``seed_offset`` shifts the seeds the package uses. Normal callers leave it at 0; a
        reproduction run gives each of its repeats a different one (step C3), because with the
        same seeds Ollama would return the same output and the repeats would measure nothing.
        """
        load_before = self._load_timer.total_ns
        started = time.perf_counter()
        record = self._session.ask(
            question,
            time_limit_s=self._settings.csv_question_time_limit_s,
            seed_offset=seed_offset,
        )
        seconds = time.perf_counter() - started
        return CsvAnswer(
            record=record,
            markdown=to_markdown(record),
            model=self.model,
            seconds=seconds,
            model_load_ms=(self._load_timer.total_ns - load_before) // _NS_PER_MS,
        )
