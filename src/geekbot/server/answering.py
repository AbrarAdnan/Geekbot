"""What the streamed answer route needs, built once when the server starts (step S4.2).

``geekbot ask`` prepares everything for one question and exits. A server answers many, so it does that
preparation once, in its lifespan, and keeps the result on ``app.state``:

- the **retriever** (hybrid search plus the cross-encoder, over the workspace the lifespan opened);
- the **chat model** and the model that rewrites a follow-up, as LangChain objects (cheap to build,
  they only hold settings; the model itself lives in Ollama);
- the **warm-up**: the chat model is loaded onto the GPU through the explicit swap (PLAN.md §6.5 R-2
  and R-4: the other mode's model is unloaded first, then this one is loaded with the context size
  real questions will use) and the reranker's ONNX model is loaded on the CPU. After that a question
  never waits for a load. ``geekbot serve --no-warmup`` skips it: the first question then pays it
  (the stream says so with a ``status`` event), which is what you want while another program, such
  as the Gigalogy CSV project, is using the GPU.

**One question at a time** (``TurnGate``). There is one chat model on one GPU, and a thread's
history is read, extended and written back by the last step of a turn; two turns on the same thread
at once would both start from the same history and one would overwrite the other. Rather than a queue
(a question that waits for a minute while the user looks at a blank screen) the second asker is
refused at once, with HTTP 409 and a sentence saying why. A gate with one slot is also all that C5
needs for the swap to the CSV model, which must not happen under a running chat turn.

**Which model is on the GPU (C5).** CSV conversations use another model (``CsvSetup``), and the two
never share the GPU (PLAN.md §6.5 R-2, R-4). ``ready`` says the chat model is loaded and ``csv_ready``
that the CSV model is; at most one is true. ``ensure_ready`` and ``ensure_csv_ready`` swap explicitly
(the other model is unloaded first, then this one is loaded) and flip both flags under one lock, so
the first question of the other kind pays the swap (2.7 s to the CSV model, 4.5 s back, measured in
C3 on the target models) and the next of the same kind pays nothing. The warm-up at start-up loads the
chat model only; the CSV model is loaded when the first CSV question comes.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.runtime import (
    AgentContext,
    CsvTool,
    UnavailableChat,
    UnavailableRetriever,
)
from geekbot.config import Settings
from geekbot.index.workspace import Workspace
from geekbot.llm.chat import ChatModel, StructuredChat, make_chat_model, make_structured_model
from geekbot.llm.models import OllamaClient, swap_to_mode
from geekbot.retrieval import SearchMode
from geekbot.retrieval.factory import build_retriever, citation_scorer_of
from geekbot.retrieval.rerank import PassageScorer

__all__ = ["Answering", "CsvSetup", "TurnGate", "open_chat_answering"]

log = logging.getLogger(__name__)

_WARM_TEXT = (
    "warm up"  # scored once so the cross-encoder's model is loaded before the first question
)


@dataclass(frozen=True)
class CsvSetup:
    """What the server needs to answer CSV questions (step C5): the CSV model, the model call that
    rewrites a follow-up (it must use the CSV model, the one on the GPU, with the context size it was
    loaded with: R-2), and the swap that loads the CSV model."""

    model: str
    condenser: StructuredChat | None
    load: Callable[[], float]


class Answering:
    """The server's answering tools: the graph context, a way to make the chat model, a warm-up.

    ``context`` is the context of an ordinary turn (thinking off). ``context_for(think=True)`` is the
    same with a chat model built to reason, for the "think harder" toggle of PLAN.md §6.5.
    ``ensure_ready`` runs the warm-up once (it blocks for seconds: call it from a worker thread) and
    ``ready`` says whether it has run. ``csv`` (C5) is the CSV side: ``None`` when the server has no
    CSV model set up; ``csv_context`` makes the graph context of a CSV turn and ``ensure_csv_ready``
    swaps the GPU to the CSV model (the module docstring says how the two flags work together).
    """

    def __init__(
        self,
        context: AgentContext,
        *,
        make_llm: Callable[[bool], ChatModel],
        load: Callable[[], float],
        csv: CsvSetup | None = None,
    ) -> None:
        self.context = context
        self._make_llm = make_llm
        self._load = load
        self.csv = csv
        self._lock = threading.Lock()
        self.ready = False
        self.csv_ready = False

    @property
    def model(self) -> str:
        """The chat model's tag, for the stream and the health route."""
        return self.context.model

    def context_for(self, *, think: bool) -> AgentContext:
        """The graph context for one turn: the shared one, or with a reasoning chat model."""
        return replace(self.context, llm=self._make_llm(True)) if think else self.context

    def csv_context(self, analyst: CsvTool) -> AgentContext:
        """The graph context of a CSV turn: no knowledge base, no chat model, the loaded CSV.

        What a CSV run must not have is replaced by objects that refuse to be used (the same ones the
        command line passes), so a bug that reached for the retriever would fail loudly instead of
        answering from documents in a CSV conversation. ``top_k`` is unused. The citation scorer is
        dropped: a CSV answer has no sources to check.
        """
        if self.csv is None:
            raise RuntimeError("this server has no CSV model set up")
        return replace(
            self.context,
            retriever=UnavailableRetriever(),
            llm=UnavailableChat(),
            model=self.csv.model,
            top_k=1,
            condenser=self.csv.condenser,
            csv=analyst,
            citation_scorer=None,
        )

    def ensure_ready(self) -> float:
        """Load the chat model if it is not the one on the GPU; return the seconds it took (0 if it
        is). The swap unloads the CSV model first, and the two flags change together."""
        with self._lock:
            if self.ready:
                return 0.0
            seconds = self._load()
            self.ready = True
            self.csv_ready = False
            return seconds

    def ensure_csv_ready(self) -> float:
        """Like ``ensure_ready``, for the CSV model: swap the GPU to it if it is not there."""
        if self.csv is None:
            raise RuntimeError("this server has no CSV model set up")
        with self._lock:
            if self.csv_ready:
                return 0.0
            seconds = self.csv.load()
            self.csv_ready = True
            self.ready = False
            return seconds


class TurnGate:
    """Lets one question at a time through (see the module docstring).

    ``try_acquire`` returns a token, or ``None`` when a question is already being answered;
    ``release`` gives the slot back, and only to the holder of that token: the route releases twice
    (when the stream ends, and again as a backstop after the response), and the second call must not
    free a slot a newer question has taken in the meantime. It is used from the event loop only, so
    it needs no lock.
    """

    def __init__(self) -> None:
        self._holder: tuple[object, str] | None = None

    def try_acquire(self, thread_id: str) -> object | None:
        """Take the slot for ``thread_id``; ``None`` if somebody holds it."""
        if self._holder is not None:
            return None
        token = object()
        self._holder = (token, thread_id)
        return token

    def release(self, token: object) -> None:
        """Give the slot back if ``token`` is the holder's; otherwise do nothing."""
        if self._holder is not None and self._holder[0] is token:
            self._holder = None

    @property
    def busy(self) -> bool:
        """Whether a question is being answered right now."""
        return self._holder is not None

    @property
    def thread_id(self) -> str | None:
        """The thread being answered right now, if any."""
        return None if self._holder is None else self._holder[1]


def _loader(settings: Settings, scorer: PassageScorer | None) -> Callable[[], float]:
    """The warm-up: swap the GPU to the chat model, then load the reranker's model."""

    def load() -> float:
        started = time.perf_counter()
        with OllamaClient(settings.ollama_host) as client:
            swap = swap_to_mode(client, settings, "chat")
        released = f", unloaded {', '.join(swap.unloaded)}" if swap.unloaded else ""
        log.info("%s loaded in %.1f s%s", settings.resolved_chat_model, swap.warm_up_s, released)
        if scorer is not None:
            scorer.score(_WARM_TEXT, [_WARM_TEXT])
        return time.perf_counter() - started

    return load


def _csv_loader(settings: Settings) -> Callable[[], float]:
    """The swap to the CSV model: unload the chat model, then load this one with its own context."""

    def load() -> float:
        started = time.perf_counter()
        with OllamaClient(settings.ollama_host) as client:
            swap = swap_to_mode(client, settings, "csv")
        released = f", unloaded {', '.join(swap.unloaded)}" if swap.unloaded else ""
        log.info("%s loaded in %.1f s%s", settings.resolved_csv_model, swap.warm_up_s, released)
        return time.perf_counter() - started

    return load


def open_chat_answering(
    settings: Settings, workspace: Workspace, *, warm_up: bool = True
) -> Answering:
    """Build the answering tools over ``workspace`` and (unless ``warm_up`` is false) load the models.

    Blocking (the warm-up takes seconds, the first time ever it also downloads the reranker): the
    lifespan calls it in a worker thread. A missing model or an Ollama that went away since
    ``geekbot serve`` looked raises ``OllamaError`` and the server does not start.
    """
    retriever = build_retriever(SearchMode.HYBRID_RERANK, workspace, settings)
    scorer = citation_scorer_of(retriever)
    context = AgentContext(
        retriever=retriever,
        llm=make_chat_model(settings),
        model=settings.resolved_chat_model,
        top_k=settings.answer_top_k,
        context_budget_tokens=settings.context_budget_tokens,
        condenser=make_structured_model(
            settings, CondenseOutput, num_predict=settings.condense_num_predict
        ),
        history_budget_tokens=settings.history_budget_tokens,
        history_keep_messages=settings.history_keep_messages,
        citation_scorer=scorer,
        citation_weak_below=settings.cite_weak_below,
    )
    csv_model = settings.resolved_csv_model
    answering = Answering(
        context,
        make_llm=lambda think: make_chat_model(settings, think=think),
        load=_loader(settings, scorer),
        csv=CsvSetup(
            model=csv_model,
            condenser=make_structured_model(
                settings,
                CondenseOutput,
                num_predict=settings.condense_num_predict,
                model=csv_model,
                num_ctx=settings.csv_num_ctx,
            ),
            load=_csv_loader(settings),
        ),
    )
    if warm_up:
        seconds = answering.ensure_ready()
        log.info("answering ready in %.1f s: %s", seconds, answering.model)
    else:
        log.info("answering built without a warm-up: the first question loads %s", answering.model)
    return answering
