"""Lexical retrieval: BM25 over the ready chunks (step S2.2, PLAN.md §9).

The store already does the work (``Store.search_bm25``: an FTS5 ``MATCH`` ranked by ``bm25()``,
ready documents only, ties broken by ``chunk_id``; S1.4). This class gives it the shape every
retriever has (``Retriever``), so that the hybrid retriever, the ablation and the answer graph can
use it without knowing it is SQLite underneath. It needs no embedder, so it works with Ollama
switched off.

BM25 matches *words*: it scores a chunk higher for containing more of the query's words, for
containing them more often, for the words being rare across the whole index, and for the chunk
being short. It cannot match a synonym, which is what dense retrieval is for, and it is unbeaten at
the exact rare token: a name, an error code, a part number.
"""

import time
from collections.abc import Callable

from geekbot.index.sqlite import Store
from geekbot.retrieval import RetrievalResult

__all__ = ["LexicalRetriever"]


class LexicalRetriever:
    """BM25 search over ready chunks. Satisfies ``Retriever``. ``clock`` is for tests."""

    def __init__(self, store: Store, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._store = store
        self._clock = clock

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` chunks that best match the query's words, best first (positive BM25 scores).

        A query with no words gives no results. ``ValueError`` if ``k < 1``.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        start = self._clock()
        chunks = tuple(self._store.search_bm25(query, limit=k))
        elapsed = (self._clock() - start) * 1000
        return RetrievalResult(chunks=chunks, timings_ms={"bm25": elapsed, "total": elapsed})
