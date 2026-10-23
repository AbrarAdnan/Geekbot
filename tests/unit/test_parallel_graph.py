"""Tests for step S3.6: the answer graph with a staged search (dense ‖ lexical → fuse → rerank).

No model and no database. The two searches are fakes; the graph is run for real, so what is tested is
the wiring: that the two searches really run at the same moment, that a staged turn ends with the same
sources as a sequential one, that what the checkpointer is given is plain data, which steps a turn
announces, and that a retriever without stages still takes the old single step.
"""

import json
import threading
from collections.abc import Sequence
from typing import Any

import pytest
from chatkit import ScriptedChat, scored
from langgraph.checkpoint.memory import InMemorySaver

from geekbot.agent.graph import build_graph
from geekbot.agent.nodes import NO_CONTEXT_ANSWER, route_after_retrieve
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState
from geekbot.domain import Chunk, ScoredChunk
from geekbot.retrieval import RetrievalResult, Retriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.rerank import Reranker
from geekbot.retrieval.staged import StagedSearch

BARRIER_TIMEOUT_S = 3.0  # long for a loaded machine, short enough that a failure is no hang

DENSE = [
    scored(1, "Chat history is kept for 90 days."),
    scored(2, "Audit logs are kept for a year."),
    scored(3, "Backups are kept for 30 days."),
]
LEXICAL = [scored(3, "Backups are kept for 30 days."), scored(4, "Passwords rotate yearly.")]
EVERY_CHUNK = [*DENSE, *LEXICAL]


class Side:
    """A search that returns canned chunks. With a barrier it waits for the other side to arrive."""

    def __init__(
        self, chunks: Sequence[ScoredChunk], *, barrier: threading.Barrier | None = None
    ) -> None:
        self.chunks = tuple(chunks)
        self.barrier = barrier
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        if self.barrier is not None:
            self.barrier.wait(timeout=BARRIER_TIMEOUT_S)  # BrokenBarrierError if nobody joins
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 2.0})


def resolver(chunks: Sequence[ScoredChunk]) -> Any:
    index: dict[str, Chunk] = {item.chunk.chunk_id: item.chunk for item in chunks}

    def get_chunks(chunk_ids: Sequence[str]) -> list[Chunk]:
        return [index[cid] for cid in chunk_ids if cid in index]

    return get_chunks


class YearFirst:
    """A scorer that prefers the passage about a year, which the fusion does not."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [5.0 if "year" in text else 0.0 for text in passages]


def staged_search(
    dense: Sequence[ScoredChunk] = DENSE,
    lexical: Sequence[ScoredChunk] = LEXICAL,
    *,
    scorer: YearFirst | None = None,
    barrier: threading.Barrier | None = None,
) -> StagedSearch:
    return StagedSearch(
        Side(dense, barrier=barrier),
        Side(lexical, barrier=barrier),
        resolver(EVERY_CHUNK),
        scorer,
        candidates=4,
        depth=3,
    )


def context_for(retriever: Retriever, chat: ScriptedChat | None = None) -> AgentContext:
    return AgentContext(
        retriever=retriever,
        llm=chat or ScriptedChat("Ninety days [1]."),
        model="fake:1b",
        top_k=3,
        context_budget_tokens=3000,
    )


def run(retriever: Retriever, question: str = "How long are chats kept?") -> AgentState:
    final: AgentState = build_graph().invoke({"question": question}, context=context_for(retriever))
    return final


# --- the two searches overlap ------------------------------------------------------------------


def test_the_two_searches_run_at_the_same_time() -> None:
    """Each side waits for the other at a barrier: run one after the other, the first would time out.

    A timing comparison would be flaky on a loaded machine; a barrier is not. The turn only
    finishes if both searches were in flight at once.
    """
    final = run(staged_search(barrier=threading.Barrier(2)))
    assert final["used"] == 3
    assert final["answer"] == "Ninety days [1]."


# --- the same result as the sequential schedule -----------------------------------------------


@pytest.mark.parametrize("with_scorer", [False, True])
def test_a_staged_turn_ends_with_the_sources_a_sequential_turn_ends_with(
    with_scorer: bool,
) -> None:
    scorer = YearFirst() if with_scorer else None
    parallel = run(staged_search(scorer=scorer))
    hybrid = HybridRetriever(Side(DENSE), Side(LEXICAL), candidates=4)
    whole: Retriever = Reranker(hybrid, scorer, depth=3) if scorer else hybrid
    sequential = run(whole)
    assert parallel["sources"] == sequential["sources"]
    assert parallel["answer"] == sequential["answer"]
    assert parallel["staged"] is True
    assert sequential["staged"] is False


def test_the_scorer_changes_the_order_here_so_the_comparison_above_proves_something() -> None:
    plain = [source["chunk_id"] for source in run(staged_search())["sources"]]
    reranked = [source["chunk_id"] for source in run(staged_search(scorer=YearFirst()))["sources"]]
    assert plain != reranked
    assert reranked[0] == "chunk2"  # "Audit logs are kept for a year."


def test_the_reported_times_are_the_stages_and_the_wait() -> None:
    final = run(staged_search(scorer=YearFirst()))
    timings = final["retrieval_ms"]
    assert set(timings) == {"dense", "lexical", "fusion", "rerank", "total"}
    assert timings["dense"] == timings["lexical"] == 2.0  # what each fake side reported
    # the user waited for the slower search, then fusion, then the rerank
    expected = max(timings["dense"], timings["lexical"]) + timings["fusion"] + timings["rerank"]
    assert timings["total"] == pytest.approx(expected)


# --- the state ---------------------------------------------------------------------------------


def test_the_candidate_lists_do_not_outlive_the_search_and_the_state_is_plain_data() -> None:
    final = run(staged_search(scorer=YearFirst()))
    assert (final["dense_hits"], final["lexical_hits"], final["fused_hits"]) == ([], [], [])
    assert json.loads(json.dumps(final)) == final


def test_the_checkpointer_can_save_a_staged_turn_and_the_next_turn_runs_staged_again() -> None:
    """The candidate IDs and the join between the two searches are saved after every step. A second
    question on the same thread starts from that saved state (and goes through ``condense`` first)."""
    graph = build_graph(InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t-staged"}}
    context = context_for(staged_search(scorer=YearFirst()))
    first = graph.invoke({"question": "How long are chats kept?"}, config, context=context)
    second = graph.invoke({"question": "And audit logs?"}, config, context=context)
    assert (first["turn"], second["turn"]) == (1, 2)
    assert second["staged"] is True
    assert second["used"] == 3
    assert second["dense_hits"] == []  # reset by the new turn, not left over from the first


def test_a_turn_that_finds_nothing_answers_without_the_model() -> None:
    chat = ScriptedChat("never used")
    search = staged_search(dense=[], lexical=[], scorer=YearFirst())
    final = build_graph().invoke({"question": "Anything?"}, context=context_for(search, chat))
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["abstained"] is True
    assert chat.calls == []


def test_a_retriever_without_stages_is_called_whole_and_never_enters_the_staged_steps() -> None:
    side = Side(DENSE)
    final = run(side)
    assert side.calls == [("How long are chats kept?", 3)]
    assert final["staged"] is False
    assert final["retrieval_ms"] == {"total": 2.0}
    assert (final["dense_ms"], final["lexical_ms"], final["fusion_ms"]) == (0.0, 0.0, 0.0)


# --- the route ---------------------------------------------------------------------------------


def test_the_edge_after_retrieve_fans_out_for_a_staged_turn_and_decides_otherwise() -> None:
    assert route_after_retrieve({"staged": True}) == ["search_dense", "search_lexical"]
    source: Any = {"n": 1}
    assert route_after_retrieve({"staged": False, "sources": [source]}) == "generate"
    assert route_after_retrieve({"staged": False, "sources": []}) == "no_context"
    assert route_after_retrieve({}) == "no_context"


# --- the steps a turn announces --------------------------------------------------------------


def steps_of(retriever: Retriever, question: str = "How long are chats kept?") -> list[str]:
    events = build_graph().stream(
        {"question": question}, context=context_for(retriever), stream_mode="custom"
    )
    return [event["name"] for event in events if event["event"] == "step"]


def test_a_staged_turn_announces_searching_ranking_and_writing_in_that_order() -> None:
    assert steps_of(staged_search(scorer=YearFirst())) == ["searching", "ranking", "writing"]


def test_a_whole_retriever_has_nothing_to_rank_and_a_turn_without_sources_never_writes() -> None:
    assert steps_of(Side(DENSE)) == ["searching", "writing"]
    assert steps_of(Side([])) == ["searching"]
