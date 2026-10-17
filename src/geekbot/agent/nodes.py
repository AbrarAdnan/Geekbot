"""The steps of the answer graph (steps S3.1 to S3.3, PLAN.md §7).

A node is a plain function: it receives the state (and the run's context), does one thing, and
returns *only the keys it changes*. LangGraph merges the returned keys into the state and follows the
edges. Nothing here knows about LangGraph's machinery beyond the ``Runtime`` argument, which is how a
node reaches the retriever, the chat model and the stream writer, so each node can be called directly
in a test.

The nodes are ordinary (synchronous) functions on purpose. When S3.4 runs the graph with ``astream``
and an async checkpointer, LangGraph runs a synchronous node in a worker thread, so the event loop
stays free (PLAN.md §6.3) without rewriting a line here.
"""

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, Literal

from langgraph.runtime import Runtime

from geekbot.agent.citations import check_citations
from geekbot.agent.packing import pack_prompt
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState, SourceRef, StreamEvent
from geekbot.domain import Label, ScoredChunk
from geekbot.llm.chat import ChatError
from geekbot.llm.streaming import read_stream

__all__ = [
    "NO_CONTEXT_ANSWER",
    "generate",
    "has_sources",
    "label_names",
    "no_context",
    "retrieve",
    "section_trail",
    "source_refs",
    "validate_citations",
]

# What the user is told when there is nothing to answer from. S3.5 replaces this stop-gap with a real
# decision (grade the sources, abstain below a calibrated threshold); until then the one rule is that
# a model is never asked to answer without sources, because a small model would answer from memory.
NO_CONTEXT_ANSWER: Final = "I could not find anything about that in the documents."


def _same_words(left: str, right: str) -> bool:
    return left.casefold().split() == right.casefold().split()


def section_trail(title: str, section_path: Sequence[str]) -> str:
    """The heading trail as ``A > B``, without a first heading that only repeats the title.

    A Markdown file whose first heading is its title would otherwise show up as
    ``Lantern staff FAQ | Lantern staff FAQ > Your conversations``, in the prompt and under the
    answer. Only the first element is compared, ignoring case and spacing.
    """
    trail = list(section_path)
    if trail and _same_words(trail[0], title):
        trail = trail[1:]
    return " > ".join(trail)


def source_refs(chunks: Sequence[ScoredChunk]) -> list[SourceRef]:
    """Retrieved chunks as plain, numbered source records (see ``agent.state.SourceRef``)."""
    return [
        SourceRef(
            n=rank,
            chunk_id=scored.chunk.chunk_id,
            title=scored.chunk.title,
            path=scored.chunk.path,
            page=scored.chunk.page,
            section=section_trail(scored.chunk.title, scored.chunk.section_path),
            text=scored.chunk.text,
            score=scored.score,
        )
        for rank, scored in enumerate(chunks, start=1)
    ]


def label_names(labels: Iterable[Label]) -> list[str]:
    """Labels as sorted plain strings, the form the state keeps them in (a set is not JSON)."""
    return sorted(label.value for label in labels)


def retrieve(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Find the chunks that may answer the question, with the time each stage took."""
    context = runtime.context
    result = context.retriever.search(state["question"], context.top_k)
    return {"sources": source_refs(result.chunks), "retrieval_ms": dict(result.timings_ms)}


def has_sources(state: AgentState) -> Literal["generate", "no_context"]:
    """The graph's one decision (a conditional edge): answer, or say nothing was found."""
    return "generate" if state.get("sources") else "no_context"


def no_context(state: AgentState) -> dict[str, Any]:
    """Answer 'nothing found' without calling the model.

    The answer is the application's own sentence, so it cites nothing and carries no labels: no
    model call happened, and nothing a document wrote went into it (PLAN.md §8.2).
    """
    del state  # the answer does not depend on it
    return {
        "answer": NO_CONTEXT_ANSWER,
        "used": 0,
        "cited": [],
        "dropped_citations": [],
        "answer_labels": [],
        "generation": {},
    }


def generate(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Ask the chat model to answer from the packed sources, streaming. The S3.1 ✍️ piece.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. S3.3 made it
    stream; the reading of the stream is ``llm.streaming.read_stream``. In order:

    1. ``pack_prompt`` builds the messages from the question and the retrieved sources, within the
       context budget. If not even one source fits, the model is not called and the answer is
       ``NO_CONTEXT_ANSWER`` (a model must never answer without sources).
    2. The chat model is called once, as a stream, and every piece is sent to the graph's stream
       writer as it arrives (a ``StreamEvent``; nobody is listening under plain ``invoke``, and the
       writer then does nothing). Any failure (Ollama down, model missing, a server error, a stream
       that breaks half way) becomes a single ``ChatError`` that names the model and keeps the
       original as its cause, so the caller handles one kind of exception whatever library
       raised it.
    3. The reply's text must not be empty: a model that spent its whole budget thinking would
       otherwise produce a blank answer that looks like success. That is a ``ChatError`` too, and
       the message says so when reasoning text did arrive.
    4. The node returns the answer, how many sources the prompt held (``used``), the labels of
       the prompt (they become the answer's, PLAN.md §8.2), the call's readings (``RunStats``,
       time to first token included) and, if the model reasoned, its reasoning. Nothing else
       changes in the state. The answer is still the model's raw text: ``validate_citations`` runs
       next, after the stream, on the finished text.
    """
    context = runtime.context
    packed = pack_prompt(
        state["question"], state["sources"], budget_tokens=context.context_budget_tokens
    )
    if packed.used == 0:
        return no_context(state)
    emit = runtime.stream_writer

    def send(event: Literal["text", "thinking"]) -> Callable[[str], None]:
        def write(piece: str) -> None:
            emit(StreamEvent(event=event, text=piece))

        return write

    # The clock starts before the stream is created, so a model whose `stream` already sends the
    # request when called is not credited with a time to first token it did not have.
    started = time.perf_counter()
    try:
        result = read_stream(
            context.llm.stream(packed.messages),
            started=started,
            on_text=send("text"),
            on_thinking=send("thinking"),
        )
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        raise ChatError(f"the chat model {context.model!r} failed ({detail})") from error
    if not result.text:
        spent = (
            " (it spent its whole reply budget thinking; raise GEEKBOT_CHAT_THINK_NUM_PREDICT)"
            if result.thinking
            else ""
        )
        raise ChatError(f"the chat model {context.model!r} returned no text{spent}")
    update: dict[str, Any] = {
        "answer": result.text,
        "used": packed.used,
        "answer_labels": label_names(packed.labels),
        "generation": result.stats.as_dict(),
    }
    if result.thinking:
        update["thinking"] = result.thinking
    return update


def validate_citations(state: AgentState) -> dict[str, Any]:
    """Remove the ``[n]`` marks from the answer that name no source the prompt held (S3.2).

    The model's own text is kept as ``raw_answer`` so nothing is lost (an evaluation can compare the
    two); ``answer`` becomes the checked text, ``cited`` the marks that remain and
    ``dropped_citations`` the numbers that were removed. A reply that was nothing but invalid marks
    would become an empty answer, which is an error like any other empty reply, not a blank result.
    """
    raw = state["answer"]
    check = check_citations(raw, state["used"])
    if not check.text:
        raise ChatError("the chat model's reply held nothing but citations to missing sources")
    return {
        "answer": check.text,
        "raw_answer": raw,
        "cited": list(check.cited),
        "dropped_citations": list(check.dropped),
    }
