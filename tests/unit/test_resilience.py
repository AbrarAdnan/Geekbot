"""Tests for step S3.6: which failures are retried, and the steps that carry the retry.

Two layers. ``is_transient`` is a pure function of an exception and its causes, tested as a table.
Then the graph is run with a retry policy that does not wait (``initial_interval=0``) and fakes that
fail on a schedule, to show what each step does with a failure: a transient one before anything was
shown is run again, a permanent one is not, one after the first piece was sent is not, and a staged
search runs again only the stage that failed.
"""

import threading
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
import pytest
from chatkit import ScriptedChat, scored
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.runtime import Runtime

from geekbot.agent.graph import build_graph
from geekbot.agent.nodes import generate, source_refs
from geekbot.agent.resilience import (
    RETRY_ATTEMPTS,
    RETRY_FIRST_WAIT_S,
    TRANSIENT_STATUS,
    is_transient,
    transient_retry,
)
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState
from geekbot.domain import Chunk, ScoredChunk
from geekbot.index.embedder import EmbeddingError
from geekbot.llm.chat import ChatError
from geekbot.llm.models import OllamaError
from geekbot.retrieval import RetrievalResult, Retriever
from geekbot.retrieval.staged import StagedSearch

REQUEST = httpx.Request("POST", "http://localhost:11434/api/chat")
CHUNKS = [scored(1, "Chat history is kept for 90 days."), scored(2, "Backups: 30 days.")]


def status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"HTTP {code}", request=REQUEST, response=httpx.Response(code, request=REQUEST)
    )


def because[E: BaseException](outer: E, cause: BaseException) -> E:
    """``outer`` as if it had been raised ``from cause`` (what ``raise outer from cause`` leaves)."""
    outer.__cause__ = cause
    return outer


class ResponseError(Exception):
    """Looks like the ``ollama`` client's own error: a status code on the exception."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


# --- which failures are transient ------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("reset"),
        ConnectionError("reset by peer"),
        TimeoutError("timed out"),
        status_error(503),
        status_error(500),
        status_error(429),
        ResponseError(500),
        because(ChatError("the chat model failed"), httpx.ConnectError("refused")),
        because(OllamaError("Ollama POST /api/embed failed"), httpx.ConnectError("refused")),
        because(EmbeddingError("no vectors"), status_error(502)),
        because(ChatError("outer"), because(OllamaError("middle"), httpx.ReadTimeout("slow"))),
    ],
)
def test_a_failure_that_may_pass_is_transient(error: BaseException) -> None:
    assert is_transient(error) is True


@pytest.mark.parametrize(
    "error",
    [
        status_error(404),  # the model is not pulled
        status_error(400),
        ResponseError(404),
        ValueError("not json"),
        RuntimeError("a bug"),
        KeyError("n"),
        ChatError("the chat model returned no text"),  # no cause: nothing to wait out
        because(ChatError("failed"), status_error(404)),
        because(ChatError("failed"), ValueError("bad reply")),
        # A failure after part of the reply was shown is never retried, whatever caused it.
        because(ChatError("failed half way", partial=True), httpx.ConnectError("reset")),
    ],
)
def test_a_failure_that_will_come_again_is_not(error: BaseException) -> None:
    assert is_transient(error) is False


def test_a_cycle_in_the_causes_ends_the_search() -> None:
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first
    assert is_transient(first) is False


def test_the_statuses_that_are_retried_are_the_busy_and_broken_ones() -> None:
    assert TRANSIENT_STATUS == {429, 500, 502, 503, 504}


def test_the_policy_retries_only_what_is_transient_and_gives_up_after_three_tries() -> None:
    policy = transient_retry()
    assert policy.max_attempts == RETRY_ATTEMPTS == 3
    assert policy.initial_interval == RETRY_FIRST_WAIT_S == 0.5
    assert policy.retry_on is is_transient
    assert transient_retry(max_attempts=5, initial_interval=0.0, jitter=False).max_attempts == 5


# --- fakes that fail on a schedule -------------------------------------------------------------


class FlakyChat:
    """Fails with the given errors, one per call, then answers once. ``half`` sends a piece first."""

    def __init__(self, *errors: Exception, half: bool = False) -> None:
        self.errors = list(errors)
        self.half = half
        self.calls = 0

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        self.calls += 1
        if self.errors:
            error = self.errors.pop(0)
            if self.half:
                yield AIMessageChunk(content="Ninety ")
            raise error
        yield AIMessageChunk(content="Ninety days [1].")


class FlakyRetriever:
    """A whole retriever that fails with the given errors, one per call, then returns the chunks."""

    def __init__(self, *errors: Exception, chunks: Sequence[ScoredChunk] = CHUNKS) -> None:
        self.errors = list(errors)
        self.chunks = tuple(chunks)
        self.calls = 0
        self.lock = threading.Lock()

    def search(self, query: str, k: int) -> RetrievalResult:
        with self.lock:
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 1.0})


def context_for(retriever: Retriever, chat: Any) -> AgentContext:
    return AgentContext(
        retriever=retriever, llm=chat, model="fake:1b", top_k=2, context_budget_tokens=3000
    )


def run(retriever: Retriever, chat: Any) -> AgentState:
    graph = build_graph(retry=transient_retry(initial_interval=0.0, jitter=False))
    final: AgentState = graph.invoke(
        {"question": "How long?"}, context=context_for(retriever, chat)
    )
    return final


def staged_with(dense: FlakyRetriever, lexical: FlakyRetriever) -> StagedSearch:
    index: dict[str, Chunk] = {item.chunk.chunk_id: item.chunk for item in CHUNKS}
    return StagedSearch(
        dense,
        lexical,
        lambda ids: [index[cid] for cid in ids if cid in index],
        None,
        candidates=4,
        depth=2,
    )


# --- the model call -----------------------------------------------------------------------------


def test_a_transient_failure_before_any_piece_is_retried_and_the_answer_written_once() -> None:
    chat = FlakyChat(httpx.ConnectError("refused"))
    graph = build_graph(retry=transient_retry(initial_interval=0.0, jitter=False))
    events = graph.stream(
        {"question": "How long?"},
        context=context_for(FlakyRetriever(), chat),
        stream_mode="custom",
    )
    texts = [event["text"] for event in events if event["event"] == "text"]
    assert chat.calls == 2
    assert "".join(texts) == "Ninety days [1]."  # once, from the top: nothing had been shown


def test_it_gives_up_after_the_last_attempt_and_reports_the_failure() -> None:
    chat = FlakyChat(*[httpx.ConnectError("refused")] * 5)
    with pytest.raises(ChatError, match="the chat model 'fake:1b' failed"):
        run(FlakyRetriever(), chat)
    assert chat.calls == RETRY_ATTEMPTS


@pytest.mark.parametrize("error", [status_error(404), ValueError("bad")])
def test_a_permanent_failure_is_not_run_again(error: Exception) -> None:
    chat = FlakyChat(error)
    with pytest.raises(ChatError):
        run(FlakyRetriever(), chat)
    assert chat.calls == 1


def test_a_failure_after_the_first_piece_is_not_run_again() -> None:
    chat = FlakyChat(httpx.ConnectError("reset"), half=True)
    with pytest.raises(ChatError) as caught:
        run(FlakyRetriever(), chat)
    assert caught.value.partial is True
    assert chat.calls == 1  # a second run would have written "Ninety Ninety days" on the screen


def test_generate_says_whether_anything_had_been_sent_when_it_failed() -> None:
    state: AgentState = {"question": "How long?", "sources": source_refs(CHUNKS)}

    def failure(half: bool) -> tuple[ChatError, list[Any]]:
        sent: list[Any] = []
        runtime = Runtime(
            context=context_for(FlakyRetriever(), FlakyChat(httpx.ConnectError("x"), half=half)),
            stream_writer=sent.append,
        )
        with pytest.raises(ChatError) as caught:
            generate(state, runtime)
        return caught.value, sent

    early, sent_early = failure(half=False)
    late, sent_late = failure(half=True)
    assert early.partial is False
    assert [e["name"] for e in sent_early if e["event"] == "step"] == ["writing"]
    assert not [e for e in sent_early if e["event"] == "text"]
    assert late.partial is True
    assert "".join(e["text"] for e in sent_late if e["event"] == "text").strip() == "Ninety"


# --- the search ----------------------------------------------------------------------------------


def test_a_whole_retriever_that_fails_transiently_is_called_again() -> None:
    retriever = FlakyRetriever(because(OllamaError("embed failed"), httpx.ConnectError("refused")))
    final = run(retriever, ScriptedChat("Ninety days [1]."))
    assert retriever.calls == 2
    assert final["used"] == 2


def test_in_a_staged_search_only_the_stage_that_failed_runs_again() -> None:
    dense = FlakyRetriever(because(EmbeddingError("no vectors"), status_error(503)))
    lexical = FlakyRetriever()
    final = run(staged_with(dense, lexical), ScriptedChat("Ninety days [1]."))
    assert dense.calls == 2
    assert lexical.calls == 1  # it had finished; the retry did not redo it
    assert final["used"] == 2


def test_a_permanent_failure_in_the_dense_stage_is_not_run_again() -> None:
    dense = FlakyRetriever(RuntimeError("a bug"))
    with pytest.raises(RuntimeError, match="a bug"):
        run(staged_with(dense, FlakyRetriever()), ScriptedChat("never used"))
    assert dense.calls == 1
