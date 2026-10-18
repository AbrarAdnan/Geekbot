"""Live tests for steps S3.1 to S3.3: ask a question and stream a grounded answer from the real model.

Needs Ollama running with ``nomic-embed-text`` and the active profile's chat model pulled
(``gemma3:4b`` on the default ``proto`` profile). They load the chat model onto the **GPU**, so run
them when Gigalogy is not using the GPU: ``pytest -m live -s tests/live/test_ask_live.py``. Deselected
by default.

Each test builds its own three-document knowledge base in a temporary folder (the embedder runs on
the CPU, the reranker is not used, so nothing is downloaded), then runs questions through the whole
chain: swap to the chat model, retrieve, pack, generate (streaming), check the citations. **They are
plumbing checks, not quality results:** a few questions on three one-sentence documents say whether the
pieces fit together (does Ollama accept ``reasoning=False`` on this model, does the model load once and
stay loaded, does the stream arrive in pieces with a time to the first token, is the answer grounded in
the right document), not how good the answers are or how fast the target model is.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.graph import build_graph
from geekbot.agent.latency import (
    LatencyRun,
    count_loads,
    format_latency_report,
    measure_latency,
)
from geekbot.agent.render import format_answer
from geekbot.agent.runner import read_thread, run_turn
from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.domain import RunStats
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.llm.chat import ChatError, make_chat_model, make_structured_model
from geekbot.llm.models import OllamaClient, swap_to_mode, unload_and_wait
from geekbot.retrieval import SearchMode
from geekbot.retrieval.factory import build_retriever

CORPUS = {
    "sourdough.md": (
        "# Sourdough\n\nA sourdough starter is flour and water left to ferment until it bubbles. "
        "Feed it every day and keep it warm."
    ),
    "rivers.md": (
        "# Rivers of Bangladesh\n\nThe Ganges and the Brahmaputra meet in Bangladesh and flow "
        "together into the Bay of Bengal."
    ),
    "passwords.md": (
        "# Passwords\n\nRotate every password at least every ninety days and never reuse one "
        "across services."
    ),
}
MAX_LOAD_SECONDS = 2.0  # R-2: the warm-up loaded the model; a query should not load it again
PASSWORD_QUESTION = "How often should I rotate my passwords?"


@dataclass(frozen=True)
class Live:
    """The settings and the ready-to-use graph context of a live run."""

    settings: Settings
    context: AgentContext


@pytest.fixture
def live(tmp_path: Path) -> Iterator[Live]:
    """A tiny knowledge base ingested for real, the chat model warm on the GPU, a graph context."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data")
    files, _ = collect_files([docs])
    model = settings.resolved_chat_model
    with Workspace.open(settings) as workspace:
        ingestor = Ingestor(workspace.store, workspace.dense, workspace.embedder)
        assert not ingestor.ingest(files).has_failures
        with OllamaClient(settings.ollama_host) as client:
            swap = swap_to_mode(client, settings, "chat")
        print(f"\n{model} warm in {swap.warm_up_s:.1f} s (unloaded: {list(swap.unloaded)})")
        yield Live(
            settings=settings,
            context=AgentContext(
                retriever=build_retriever(SearchMode.HYBRID, workspace, settings),
                llm=make_chat_model(settings),
                model=model,
                top_k=3,
                context_budget_tokens=settings.context_budget_tokens,
            ),
        )


@pytest.mark.live
def test_a_real_question_gets_a_grounded_answer_from_the_right_document(live: Live) -> None:
    context = live.context
    graph = build_graph()
    answered = graph.invoke({"question": PASSWORD_QUESTION}, context=context)
    print(format_answer(answered, model=context.model))

    assert answered["sources"][0]["title"] == "passwords.md"
    assert answered["used"] >= 1
    text = answered["answer"].lower()
    assert "ninety" in text or "90" in text
    assert answered["generation"].get("load_seconds", 0.0) < MAX_LOAD_SECONDS
    # S3.2: whatever the model wrote, no mark left in the answer points at a missing source.
    # Whether it cited at all is only shown (a 4B model may not): that is a quality question.
    assert all(1 <= n <= answered["used"] for n in answered["cited"])
    assert answered["answer_labels"] == ["KB", "USER"]
    print(f"\nraw answer : {answered['raw_answer']!r}")
    print(f"cited      : {answered['cited']}, removed: {answered['dropped_citations']}")

    # Not asserted, only shown: a question the documents cannot answer. What the model does
    # here is S3.5's concern (abstain); S3.1's prompt only asks it to say it could not find it.
    unanswerable = graph.invoke({"question": "What is the capital of France?"}, context=context)
    print(f"\nunanswerable question:\n{format_answer(unanswerable, model=context.model)}")


@pytest.mark.live
def test_a_real_answer_arrives_in_pieces_with_a_time_to_the_first_token(live: Live) -> None:
    """S3.3: the stream is live, in several pieces, and the first token comes before the end."""
    graph = build_graph()
    events = list(
        graph.stream(
            {"question": PASSWORD_QUESTION},
            context=live.context,
            stream_mode=["custom", "values"],
        )
    )
    pieces = [p["text"] for mode, p in events if mode == "custom" and p["event"] == "text"]
    final = [payload for mode, payload in events if mode == "values"][-1]
    stats = RunStats.from_mapping(final["generation"])
    print(f"\n{len(pieces)} pieces: {pieces[:6]}...")
    print(format_answer(final, model=live.context.model))

    assert len(pieces) > 1  # a real model streams token by token
    assert "".join(pieces).strip() == final["raw_answer"]
    assert stats.first_token_s is not None
    assert 0 < stats.first_token_s < stats.seconds
    assert stats.load_s is not None
    assert stats.load_s < MAX_LOAD_SECONDS
    assert stats.output_tokens is not None
    assert stats.prompt_tokens is not None
    assert stats.tokens_per_second is not None


@pytest.mark.live
def test_the_latency_harness_measures_a_real_cold_start_and_warm_runs(live: Live) -> None:
    """S3.3: one real cold round and two warm passes; prints the report, asserts the plumbing."""
    settings, context = live.settings, live.context
    graph = build_graph()

    def run_once(question: str) -> LatencyRun:
        final = graph.invoke({"question": question}, context=context)
        return LatencyRun(
            question=question,
            stats=RunStats.from_mapping(final["generation"]),
            retrieval_ms=final["retrieval_ms"].get("total"),
        )

    def cold_start() -> None:
        with OllamaClient(settings.ollama_host) as client:
            unload_and_wait(client, [context.model])
            swap_to_mode(client, settings, "chat")

    report = measure_latency(
        run_once,
        cold_start,
        [PASSWORD_QUESTION, "What is a sourdough starter?"],
        model=context.model,
        cold_rounds=1,
        warm_runs=2,
    )
    print("\n" + format_latency_report(report, setup=f"live plumbing check, {context.model}"))

    assert len(report.cold) == 1
    assert len(report.warm) == 4
    for run in report.runs:
        assert run.stats.first_token_s is not None
        assert 0 < run.stats.first_token_s <= run.stats.seconds
    assert count_loads(report.runs) == 0  # R-2: no question had Ollama load the model again


@pytest.mark.live
def test_a_follow_up_is_rewritten_by_the_real_model_inside_a_saved_thread(
    live: Live, tmp_path: Path
) -> None:
    """S3.4: two turns of one thread, the second a fragment that needs the first to make sense.

    Plumbing is asserted (the real model answers in the rewrite's JSON shape, the thread is saved
    and read back, the earlier turn reaches the prompt); whether the rewrite is *good* is only shown.
    """
    settings = live.settings
    condenser = make_structured_model(
        settings, CondenseOutput, num_predict=settings.condense_num_predict
    )
    context = replace(live.context, condenser=condenser)
    db = tmp_path / "threads.db"
    follow_up = "And can I reuse one across services?"

    first = asyncio.run(run_turn(PASSWORD_QUESTION, context=context, thread="live", threads_db=db))
    second = asyncio.run(run_turn(follow_up, context=context, thread="live", threads_db=db))
    print(
        f"\nturn 1: {first.final['answer']!r}  (rewrite seconds: {first.final['condense_seconds']})"
    )
    print(format_answer(second.final, model=context.model, thread="live"))
    print(f"\nrewritten as: {second.final['standalone_question']!r}")

    assert (
        first.final["condense_seconds"] == 0.0
    )  # the first turn makes no model call before retrieval
    final = second.final
    assert final["condense_seconds"] > 0
    assert final["condense_error"] == ""  # the real model answered in the shape of the schema
    assert final["standalone_question"]
    assert final["history_used"] == 2  # the earlier question and answer went into the prompt
    assert final["turn"] == 2
    saved = asyncio.run(read_thread(db, "live"))
    assert saved is not None
    assert [m["id"] for m in saved["history"]] == [
        "t1-user",
        "t1-assistant",
        "t2-user",
        "t2-assistant",
    ]
    assert saved["thread_taint"] == ["KB", "USER"]
    stats = RunStats.from_mapping(final["generation"])
    assert stats.load_s is not None
    assert stats.load_s < MAX_LOAD_SECONDS  # the rewrite did not make Ollama reload the model
    # Shown, not asserted: does the rewrite carry the topic, and did retrieval find the right file?
    print(f"mentions passwords: {'password' in final['standalone_question'].lower()}")
    print(f"first source: {final['sources'][0]['title']}")


@pytest.mark.live
def test_asking_a_model_to_think_either_works_or_fails_with_a_clear_error(live: Live) -> None:
    """S3.3: what Ollama really says. ``gemma3:4b`` has no thinking mode; ``gemma4:12b`` may.

    Only shown and loosely checked: the point is to see the real wording of the refusal (the
    command's hint looks for the words "support thinking") and, on a model that does think,
    that the reasoning arrives apart from the answer.
    """
    settings, context = live.settings, live.context
    thinking_context = AgentContext(
        retriever=context.retriever,
        llm=make_chat_model(settings, think=True),
        model=context.model,
        top_k=context.top_k,
        context_budget_tokens=context.context_budget_tokens,
    )
    try:
        final = build_graph().invoke({"question": PASSWORD_QUESTION}, context=thinking_context)
    except ChatError as error:
        print(f"\nthe model refused think=True: {error}")
        assert "thinking" in str(error).lower()
        return
    print(f"\nthe model accepted think=True; reasoning: {final.get('thinking', '')[:200]!r}")
    print(format_answer(final, model=context.model))
    assert final["answer"]
    assert "reasoning_content" not in final["answer"]
