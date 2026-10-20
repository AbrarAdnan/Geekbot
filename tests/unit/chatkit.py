"""Shared fakes for the tests that run the answer graph without a model (step S4.2).

``test_conversation.py`` and ``test_agent_graph.py`` keep their own small copies from earlier steps;
these are the ones the server tests share. Nothing here talks to Ollama. A fake satisfies the same
Protocol the real object does (``Retriever``, ``ChatModel``, ``StructuredChat``).
"""

import time
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessage

from geekbot.agent.condense import CondenseOutput
from geekbot.domain import Chunk, ScoredChunk
from geekbot.retrieval import RetrievalResult

REWRITE = "How long does Lantern keep audit logs?"


def scored(
    n: int, text: str, *, title: str = "Handbook", path: str = "/docs/handbook.md"
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"chunk{n}",
        doc_id="doc",
        path=path,
        title=title,
        content_hash="h" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=len(text),
        page=None,
        section_path=(),
        header="",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / n)


class ScriptedRetriever:
    """Returns ``chunks`` (at most ``k``), remembers every query, and can be slow like a real one.

    ``delay_s`` is slept inside ``search`` with ``time.sleep``: it blocks the thread it runs in, which
    is exactly what a real search (SQLite, Chroma, the reranker) does.
    """

    def __init__(self, *chunks: ScoredChunk, delay_s: float = 0.0) -> None:
        self.chunks = chunks or (
            scored(1, "Chat history is kept for 90 days."),
            scored(2, "Audit logs are kept for a year."),
        )
        self.delay_s = delay_s
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        if self.delay_s:
            time.sleep(self.delay_s)
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 1.0})


class ScriptedChat:
    """Streams the replies in order (an exception is raised); the last one repeats.

    A reply is one string (sent as one piece) or a list of strings (one piece each). ``delay_s`` is
    slept before every piece, in the worker thread, with ``time.sleep``: blocking, like a socket read.
    """

    def __init__(self, *replies: str | list[str] | Exception, delay_s: float = 0.0) -> None:
        self.replies = list(replies) or ["Ninety days [1]."]
        self.delay_s = delay_s
        self.calls: list[list[BaseMessage]] = []

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        for piece in [reply] if isinstance(reply, str) else reply:
            if self.delay_s:
                time.sleep(self.delay_s)
            yield AIMessageChunk(content=piece)


class ScriptedCondenser:
    """Stands in for the structured rewrite call: replies in order, the last one repeats."""

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies) or [CondenseOutput(standalone_question=REWRITE)]
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], /) -> Any:
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply
