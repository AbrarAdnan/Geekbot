"""Rung R0 of the ablation: v1's retrieval, rebuilt inside v2 (step S2.6, PLAN.md §9, D-034).

The ablation (PLAN.md §9) starts from "legacy v1 replica" and changes one thing per rung, so the
starting point has to be v1 as it really behaved, not v1 as it was meant to behave. What v1 did
(``rag_pipeline.py``, ``storage_utils.py``; those files stay untouched until S4.4):

- **Chunks.** LangChain's ``RecursiveCharacterTextSplitter``, 800 characters with an overlap of 80,
  separators ``\\n\\n``, ``\\n``, space, nothing, applied to each loaded page on its own. No token
  count, no heading awareness, no header.
- **Embedding.** ``nomic-embed-text`` with no task prefix at all, in Chroma's default L2 space.
- **Query.** Five template queries ("Explain: ...", "Details about: ...", ...), the three nearest
  chunks of each, duplicates dropped, then the survivors re-sorted by the cosine between the
  original question and each chunk *embedded again as if it were a query*. The best five went to
  the model.

Everything else in v1 (the prompt, the web search, the model) is outside retrieval and is not
replicated here. The one thing this file cannot prove is that the splitter cuts exactly as v1's did:
v1 pinned ``langchain-text-splitters`` 0.3.11, which needs ``langchain-core`` 0.3, and v2 runs on
the 1.x line. Both are the same algorithm; CP2's v1 reference run cuts the golden corpus with the
real 0.3.11 and compares (PLAN.md §9, "One-time v1 reference run").

The splitter is imported when it is first needed, so ``import geekbot.evals.legacy`` stays cheap.
"""

import time
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from geekbot.domain import ScoredChunk
from geekbot.index.dense import DenseIndex
from geekbot.index.embedder import EmbeddingsClient
from geekbot.index.sqlite import Store
from geekbot.ingest.chunking import ChunkDraft
from geekbot.ingest.loaders.model import SEGMENT_SEPARATOR, LoadedDocument
from geekbot.retrieval import RetrievalResult
from geekbot.retrieval.dense import resolve_hits

__all__ = [
    "LEGACY_CHUNK_CHARS",
    "LEGACY_OVERLAP_CHARS",
    "LEGACY_PER_QUERY",
    "LegacyMultiQueryRetriever",
    "legacy_chunks",
    "legacy_queries",
    "split_legacy_text",
]

LEGACY_CHUNK_CHARS: Final = 800
LEGACY_OVERLAP_CHARS: Final = 80
LEGACY_SEPARATORS: Final = ("\n\n", "\n", " ", "")
LEGACY_PER_QUERY: Final = 3  # ``similarity_search_with_score(q, k=3)``


# --- chunks ----------------------------------------------------------------------------------


def split_legacy_text(text: str) -> list[str]:
    """Cut one page of text exactly as v1's ``TEXT_SPLITTER`` did."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=LEGACY_CHUNK_CHARS,
        chunk_overlap=LEGACY_OVERLAP_CHARS,
        length_function=len,
        separators=list(LEGACY_SEPARATORS),
    )
    return splitter.split_text(text)


def legacy_chunks(doc: LoadedDocument) -> list[ChunkDraft]:
    """v1's chunks of a loaded document, as drafts the store can hold.

    v1 split each page (PDF) or the whole file (everything else) on its own, which is what a
    segment is here. LangChain returns no positions, so ``char_start`` is found by searching for the
    piece after the previous piece's start; the offsets are for the store's sake and do not affect
    any score. A piece that cannot be found (the splitter strips whitespace, so this should not
    happen) is placed at the previous position rather than failing the run.
    """
    drafts: list[ChunkDraft] = []
    base = 0
    for segment in doc.segments:
        cursor = 0
        for piece in split_legacy_text(segment.text):
            found = segment.text.find(piece, cursor)
            start = found if found >= 0 else cursor
            drafts.append(
                ChunkDraft(
                    text=piece,
                    char_start=base + start,
                    char_end=base + start + len(piece),
                    page=segment.page,
                    section_path=(),
                    header="",
                    ordinal=len(drafts),
                )
            )
            cursor = start
        base += len(segment.text) + len(SEGMENT_SEPARATOR)
    return drafts


# --- the query -------------------------------------------------------------------------------


def legacy_queries(query: str) -> list[str]:
    """The five queries v1 searched with, the question itself first. Wording copied from v1."""
    return [
        query,
        f"Explain: {query}",
        f"Details about: {query}",
        f"What is the background of: {query}?",
        f"Summarize {query} in simple terms.",
    ]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


class LegacyMultiQueryRetriever:
    """v1's retrieval over a dense index. Satisfies ``Retriever``.

    ``embeddings`` is the raw query embedder (no prefix for R0 and R1, which are defined by having
    none). It is called the way v1 called it: once per template query, once more for the question,
    and once per surviving chunk to re-sort them. That is a lot of embedder calls for one question,
    and it is the honest cost of rung R0 (the latency table shows it).
    """

    def __init__(
        self,
        store: Store,
        dense: DenseIndex,
        embeddings: EmbeddingsClient,
        *,
        per_query: int = LEGACY_PER_QUERY,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if per_query < 1:
            raise ValueError(f"per_query must be at least 1, got {per_query}")
        self._store = store
        self._dense = dense
        self._embeddings = embeddings
        self._per_query = per_query
        self._clock = clock

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` best candidates after v1's re-sort (v1 itself used the best 5)."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not query.strip():
            return RetrievalResult(chunks=(), timings_ms={})
        start = self._clock()
        candidates: dict[str, ScoredChunk] = {}
        embed_ms = search_ms = 0.0
        for text in legacy_queries(query):
            before = self._clock()
            vector = self._embeddings.embed_query(text)
            middle = self._clock()
            hits = self._dense.query(vector, self._per_query)
            resolved = resolve_hits(self._store, hits)
            after = self._clock()
            embed_ms += (middle - before) * 1000
            search_ms += (after - middle) * 1000
            for scored in resolved:
                candidates.setdefault(scored.chunk.chunk_id, scored)  # first occurrence wins
        before = self._clock()
        question = self._embeddings.embed_query(query)
        ranked = sorted(
            (
                ScoredChunk(
                    scored.chunk, _cosine(question, self._embeddings.embed_query(scored.chunk.text))
                )
                for scored in candidates.values()
            ),
            key=lambda scored: -scored.score,  # stable: ties keep the order they were found in
        )
        end = self._clock()
        timings = {
            "embed_queries": embed_ms,
            "vector_search": search_ms,
            "resort": (end - before) * 1000,
            "total": (end - start) * 1000,
        }
        return RetrievalResult(chunks=tuple(ranked[:k]), timings_ms=timings)
