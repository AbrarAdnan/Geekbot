"""Dense retrieval: embed the question, find the nearest vectors, keep what may be shown (step S2.1).

Dense (or "semantic") retrieval compares meaning, not words: the question and every chunk are
vectors, and the chunks whose vectors point the same way come first. It finds "how do I cancel my
subscription" in a chunk that says "ending your plan" and shares no word with it, which is exactly
what BM25 cannot do and why S2.2 fuses the two.

**Asymmetric.** A question and a passage are different kinds of text, so the embedder adds a
different prefix to each (``search_query: `` here, ``search_document: `` at ingest; S1.5). The query
side of that lives in ``CachedEmbedder.embed_query``; this module never touches a prefix.

**Chroma finds candidates, SQLite decides.** Chroma returns chunk IDs and similarities; it does not
know whether a chunk's document is ``ready`` (that state lives in SQLite, PLAN.md §6.4), and
copying the state into Chroma would be a second thing to keep in step. So every hit is looked up
through ``Store.get_chunks``, which returns only chunks of ready documents; an ID it does not
return is a chunk to drop (its document is being re-ingested or failed, or the vector is an
orphan that the repair has not swept yet).

**No short answers.** Dropping hits *after* taking the top ``k`` would return fewer than ``k``
results whenever the best matches belong to a document that is not ready, even though ready chunks
exist. So the search asks Chroma for more candidates, four times as many each round, until it has
``k`` that survive or the index is exhausted. In the usual case (everything is ready) that is one
round.
"""

import time
from collections.abc import Callable, Sequence
from typing import Final

from geekbot.domain import ScoredChunk
from geekbot.index.dense import DenseHit, DenseIndex
from geekbot.index.embed_cache import CachedEmbedder
from geekbot.index.sqlite import Store
from geekbot.retrieval import RetrievalResult

__all__ = ["DenseRetriever", "resolve_hits"]

_GROWTH: Final = 4


def resolve_hits(store: Store, hits: Sequence[DenseHit]) -> list[ScoredChunk]:
    """Turn Chroma hits back into stored chunks, in the same order, dropping any that may not be shown.

    ✍️ Written by Claude (prototyping mode, D-047). A hit becomes a ``ScoredChunk`` (the stored
    chunk and the similarity Chroma gave it) only if ``Store.get_chunks`` returns its chunk, which
    it does only for a chunk whose document is ``ready``. A hit whose chunk is missing from the
    store, or whose document is not ready, is dropped without an error: both are expected states
    (a document being re-ingested; a vector left behind by a crash). The order of ``hits`` is the
    order of the result, so the ranking survives. An empty list gives an empty list.
    """
    if not hits:
        return []
    found = {chunk.chunk_id: chunk for chunk in store.get_chunks([hit.chunk_id for hit in hits])}
    return [
        ScoredChunk(chunk=found[hit.chunk_id], score=hit.score)
        for hit in hits
        if hit.chunk_id in found
    ]


class DenseRetriever:
    """Nearest-vector search over the ready chunks. Satisfies ``Retriever``.

    The store, the dense index and the cached embedder are the caller's to open and close (see
    ``Workspace``). ``clock`` is for tests, a function returning seconds like ``time.perf_counter``.
    """

    def __init__(
        self,
        store: Store,
        dense: DenseIndex,
        embedder: CachedEmbedder,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._store = store
        self._dense = dense
        self._embedder = embedder
        self._clock = clock

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` ready chunks nearest to ``query``, most similar first (cosine similarity).

        A query with no words gives no results (and no embedder call). ``ValueError`` if ``k < 1``;
        ``OllamaError`` if the embedder cannot be reached.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not query.strip():
            return RetrievalResult(chunks=(), timings_ms={})
        start = self._clock()
        vector = self._embedder.embed_query(query)
        embedded = self._clock()
        total = self._dense.count()
        fetch = min(k, total)
        resolved: list[ScoredChunk] = []
        searching = 0.0
        resolving = 0.0
        while fetch > 0:
            before = self._clock()
            hits = self._dense.query(vector, fetch)
            middle = self._clock()
            resolved = resolve_hits(self._store, hits)
            after = self._clock()
            searching += middle - before
            resolving += after - middle
            if len(resolved) >= k or fetch >= total:
                break
            fetch = min(total, fetch * _GROWTH)
        timings = {
            "embed_query": (embedded - start) * 1000,
            "vector_search": searching * 1000,
            "resolve": resolving * 1000,
            "total": (self._clock() - start) * 1000,
        }
        return RetrievalResult(chunks=tuple(resolved[:k]), timings_ms=timings)
