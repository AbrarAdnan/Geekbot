"""Tests for step S4.2: the answering tools the server builds at start-up, and the one-at-a-time gate.

No Ollama and no model: the swap, the Ollama client and the retriever are replaced where the module
looks them up, so what is tested is the wiring (what is built from which setting, what the warm-up
does and in which order, that it runs once).
"""

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self

import pytest
from chatkit import ScriptedChat, ScriptedRetriever
from serverkit import open_fake_workspace, settings_for

from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.llm.models import SwapResult
from geekbot.retrieval.rerank import Reranker
from geekbot.server import answering as answering_module
from geekbot.server.answering import Answering, TurnGate, open_chat_answering

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


# --- the gate ----------------------------------------------------------------------------------


def test_the_gate_lets_one_through_and_refuses_the_next() -> None:
    gate = TurnGate()
    assert gate.busy is False
    assert gate.thread_id is None
    token = gate.try_acquire("t-a")
    assert token is not None
    assert gate.busy is True
    assert gate.thread_id == "t-a"
    assert gate.try_acquire("t-b") is None  # another thread, same refusal: one chat model
    assert gate.try_acquire("t-a") is None  # the same thread twice: the history would be torn
    gate.release(token)
    assert gate.busy is False
    assert gate.try_acquire("t-b") is not None


def test_a_token_that_is_not_the_holders_releases_nothing() -> None:
    """The route releases twice (stream end, then a backstop); the second must not free a newer slot."""
    gate = TurnGate()
    first = gate.try_acquire("t-a")
    assert first is not None
    gate.release(first)
    second = gate.try_acquire("t-b")
    assert second is not None
    gate.release(first)  # the old token, again
    assert gate.busy is True
    assert gate.thread_id == "t-b"
    gate.release(object())  # a stranger
    assert gate.busy is True
    gate.release(second)
    assert gate.busy is False


# --- Answering -----------------------------------------------------------------------------------


def make_answering(loads: list[float] | None = None) -> tuple[Answering, list[bool]]:
    thinks: list[bool] = []
    reasoning_chat = ScriptedChat("I reasoned.")
    plain_chat = ScriptedChat("Plain.")
    context = AgentContext(
        retriever=ScriptedRetriever(),
        llm=plain_chat,
        model="fake:1b",
        top_k=4,
        context_budget_tokens=1234,
    )

    def make_llm(think: bool) -> ScriptedChat:
        thinks.append(think)
        return reasoning_chat

    def load() -> float:
        if loads is not None:
            loads.append(1.0)
        return 2.5

    return Answering(context, make_llm=make_llm, load=load), thinks


def test_the_warm_up_runs_once_and_says_how_long_it_took() -> None:
    loads: list[float] = []
    answering, _ = make_answering(loads)
    assert answering.ready is False
    assert answering.ensure_ready() == 2.5
    assert answering.ready is True
    assert answering.ensure_ready() == 0.0  # already loaded: nothing to wait for
    assert len(loads) == 1


def test_a_failed_warm_up_leaves_the_tools_not_ready_and_can_be_tried_again() -> None:
    attempts: list[int] = []

    def load() -> float:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Ollama is down")
        return 0.5

    context = AgentContext(
        retriever=ScriptedRetriever(),
        llm=ScriptedChat(),
        model="fake:1b",
        top_k=1,
        context_budget_tokens=100,
    )
    answering = Answering(context, make_llm=lambda think: ScriptedChat(), load=load)
    with pytest.raises(RuntimeError, match="Ollama is down"):
        answering.ensure_ready()
    assert answering.ready is False
    assert answering.ensure_ready() == 0.5
    assert answering.ready is True


def test_an_ordinary_turn_uses_the_shared_context_and_asks_for_no_new_model() -> None:
    answering, thinks = make_answering()
    assert answering.context_for(think=False) is answering.context
    assert thinks == []
    assert answering.model == "fake:1b"


def test_a_thinking_turn_swaps_only_the_chat_model() -> None:
    answering, thinks = make_answering()
    thinking = answering.context_for(think=True)
    assert thinks == [True]
    assert thinking is not answering.context
    assert thinking.llm is not answering.context.llm
    # everything else is the shared context's, so the budgets and the retriever are not repeated
    assert thinking.retriever is answering.context.retriever
    assert (thinking.model, thinking.top_k, thinking.context_budget_tokens) == ("fake:1b", 4, 1234)


# --- open_chat_answering -------------------------------------------------------------------------


class SpyScorer:
    """Records what it is asked to score; stands in for the cross-encoder."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [0.0] * len(passages)


class StubClient:
    """Replaces ``OllamaClient`` in ``answering``: a context manager that takes the host and nothing else."""

    hosts: ClassVar[list[str]] = []

    def __init__(self, host: str) -> None:
        StubClient.hosts.append(host)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def patch_the_outside(monkeypatch: pytest.MonkeyPatch, scorer: SpyScorer, swaps: list[str]) -> None:
    """The retriever is a real ``Reranker`` over a scripted one (so ``citation_scorer_of`` finds the
    scorer); the Ollama client and the swap are fakes that record the mode."""
    StubClient.hosts = []
    monkeypatch.setattr(answering_module, "OllamaClient", StubClient)

    def fake_swap(client: Any, settings: Settings, mode: str, **kwargs: Any) -> SwapResult:
        swaps.append(mode)
        return SwapResult(loaded="m", unloaded=("other",), unload_s=0.1, warm_up_s=0.2)

    monkeypatch.setattr(answering_module, "swap_to_mode", fake_swap)
    monkeypatch.setattr(
        answering_module,
        "build_retriever",
        lambda mode, workspace, settings, **kwargs: Reranker(ScriptedRetriever(), scorer, depth=3),
    )


def test_the_tools_are_built_from_the_settings_and_not_warmed_up_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer, swaps = SpyScorer(), []
    patch_the_outside(monkeypatch, scorer, swaps)
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", answer_top_k=7)
    with open_fake_workspace(settings) as workspace:
        answering = open_chat_answering(settings, workspace, warm_up=False)
    context = answering.context
    assert answering.ready is False
    assert (swaps, scorer.calls) == ([], [])  # --no-warmup loads nothing
    assert answering.model == context.model == settings.resolved_chat_model
    assert context.top_k == 7
    assert context.context_budget_tokens == settings.context_budget_tokens
    assert context.history_budget_tokens == settings.history_budget_tokens
    assert context.history_keep_messages == settings.history_keep_messages
    assert context.condenser is not None  # a thread's follow-ups are rewritten
    assert context.citation_scorer is scorer  # the citation check shares the reranker's model
    assert context.citation_weak_below == settings.cite_weak_below


def test_the_warm_up_swaps_the_gpu_to_the_chat_model_then_loads_the_reranker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer, swaps = SpyScorer(), []
    patch_the_outside(monkeypatch, scorer, swaps)
    settings = settings_for(tmp_path)
    with open_fake_workspace(settings) as workspace:
        answering = open_chat_answering(settings, workspace)
    assert answering.ready is True
    assert swaps == ["chat"]  # the explicit swap of R-4, in chat mode
    assert StubClient.hosts == [settings.ollama_host]
    assert len(scorer.calls) == 1  # the cross-encoder was used once, so its model is loaded
    assert answering.ensure_ready() == 0.0
    assert swaps == ["chat"]  # and a question never swaps again


def test_the_real_thing_builds_with_no_network_when_the_warm_up_is_off(tmp_path: Path) -> None:
    """The retriever, the chat model and the rewrite model are only configured, not contacted."""
    settings = settings_for(tmp_path)
    with open_fake_workspace(settings) as workspace:
        answering = open_chat_answering(settings, workspace, warm_up=False)
    assert isinstance(answering.context.retriever, Reranker)
    assert answering.context.citation_scorer is answering.context.retriever.scorer
    assert answering.context.top_k == settings.answer_top_k == 5
    assert answering.ready is False
