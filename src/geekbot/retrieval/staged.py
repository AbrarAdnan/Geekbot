"""Hybrid search cut into stages the answer graph can schedule (step S3.6, PLAN.md §7).

``HybridRetriever`` and ``Reranker`` do their work inside one ``search`` call, one stage after the
other. PLAN.md §7 draws the same work as a graph: ``dense ‖ lexical → fuse → rerank``, with the two
searches side by side. This module is the same retrieval with each stage a method of its own, so the
graph can run the first two at the same time (LangGraph runs the nodes of one step in parallel), show
the user which stage is running (step events) and retry one stage without redoing the others.

**Same answer as the sequential classes, by construction and by test.** ``search`` runs the four
stages in order and returns what ``Reranker(HybridRetriever(dense, lexical), scorer)`` returns for the
same inputs (``HybridRetriever`` when there is no scorer). ``tests/unit/test_staged_search.py`` checks
that on fakes. It keeps the S2.6 measurements honest: the parallel path is a different *schedule*, not
a different retriever.

**What moves between stages is IDs.** The graph's state is saved by the checkpointer after every step.
Fifty chunk texts per side, saved again after each step, would be megabytes per turn in a file that
keeps every checkpoint. So a stage's output is a list of ``SearchHit`` (an ID and a score); fusion
works on IDs (``fusion.fuse_ranks``, the same ranking as the chunk version); and only ``rerank_stage``
reads text, for the handful of fused candidates (``resolve``, normally ``Store.get_chunks``, which
returns ready chunks only, so the readiness rule of PLAN.md §6.4 is applied once more at the end).

**What parallel search can and cannot save**, stated before anyone measures it: the two searches take
as long as the slower one instead of the sum. The BM25 search is a few milliseconds and the dense one
(an embedding call to Ollama, a Chroma query) tens, so the saving is at most the smaller of the two.
Whether it is worth its own overhead (a thread hop and a saved checkpoint per node) is what
``tests/live/test_slo_smoke_live.py`` measures; ``GEEKBOT_PARALLEL_SEARCH=false`` keeps the
sequential classes.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from geekbot.domain import Chunk, ScoredChunk
from geekbot.retrieval import RetrievalResult, Retriever, SearchHit
from geekbot.retrieval.fusion import DEFAULT_RRF_K, fuse_ranks
from geekbot.retrieval.hybrid import DEFAULT_CANDIDATES
from geekbot.retrieval.rerank import DEFAULT_RERANK_DEPTH, PassageScorer, RerankError

__all__ = ["StageHits", "StageRanked", "StagedSearch", "hits_of"]

_MILLISECONDS: Final = 1000.0


def hits_of(chunks: Sequence[ScoredChunk]) -> list[SearchHit]:
    """Retrieved chunks as plain candidate records, best first."""
    return [SearchHit(chunk_id=scored.chunk.chunk_id, score=scored.score) for scored in chunks]


@dataclass(frozen=True)
class StageHits:
    """What one stage found (IDs and scores, best first) and how long it took."""

    hits: list[SearchHit]
    ms: float


@dataclass(frozen=True)
class StageRanked:
    """The last stage's result: the final chunks, best first, and how long it took."""

    chunks: tuple[ScoredChunk, ...]
    ms: float


class StagedSearch:
    """Dense and lexical search, RRF fusion and an optional cross-encoder, as four callable stages.

    ``dense`` and ``lexical`` are ordinary retrievers (``DenseRetriever`` and ``LexicalRetriever``);
    ``resolve`` turns chunk IDs back into stored chunks (``Store.get_chunks``). Without a ``scorer``
    the result is the fused order (the ``hybrid`` mode); with one, the best ``depth`` fused candidates
    are re-scored (``hybrid_rerank``). ``candidates`` is how many chunks each side contributes before
    fusion, and ``rrf_k`` the constant of the fusion; the defaults are the sequential classes' own.
    The object satisfies ``Retriever`` (its ``search`` runs the stages in order), so it can stand
    anywhere the sequential retrievers do, and the graph recognises it by its type and schedules the
    stages itself. ``clock`` is for tests.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        resolve: Callable[[Sequence[str]], Sequence[Chunk]],
        scorer: PassageScorer | None = None,
        *,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
        depth: int = DEFAULT_RERANK_DEPTH,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._dense = dense
        self._lexical = lexical
        self._resolve = resolve
        self._scorer = scorer
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._depth = depth
        self._clock = clock

    @property
    def scorer(self) -> PassageScorer | None:
        """The cross-encoder, if there is one (the answer graph's citation check shares it, S3.5)."""
        return self._scorer

    def fused_depth(self, k: int) -> int:
        """How many fused candidates move on to the last stage for a request of ``k`` results."""
        return max(k, self._depth) if self._scorer is not None else k

    def side_depth(self, k: int) -> int:
        """How many chunks each side is asked for, so that fusion has enough to promote from."""
        return max(self.fused_depth(k), self._candidates)

    def dense_stage(self, query: str, k: int) -> StageHits:
        """Stage 1a: the chunks nearest to the question by meaning. ``ValueError`` if ``k < 1``."""
        _check_k(k)
        return self._side(self._dense, query, k)

    def lexical_stage(self, query: str, k: int) -> StageHits:
        """Stage 1b: the chunks that best match the question's words (BM25)."""
        _check_k(k)
        return self._side(self._lexical, query, k)

    def _side(self, side: Retriever, query: str, k: int) -> StageHits:
        started = self._clock()
        result = side.search(query, self.side_depth(k))
        elapsed = (self._clock() - started) * _MILLISECONDS
        return StageHits(hits_of(result.chunks), result.timings_ms.get("total", elapsed))

    def fuse_stage(
        self, dense: Sequence[SearchHit], lexical: Sequence[SearchHit], k: int
    ) -> StageHits:
        """Stage 2: merge the two lists by rank (RRF) and keep the best ``fused_depth(k)``."""
        _check_k(k)
        started = self._clock()
        fused = fuse_ranks(
            [[hit["chunk_id"] for hit in dense], [hit["chunk_id"] for hit in lexical]],
            k=self._rrf_k,
            limit=self.fused_depth(k),
        )
        elapsed = (self._clock() - started) * _MILLISECONDS
        return StageHits([SearchHit(chunk_id=cid, score=score) for cid, score in fused], elapsed)

    def rerank_stage(self, query: str, fused: Sequence[SearchHit], k: int) -> StageRanked:
        """Stage 3: read the fused candidates' text and keep the best ``k``.

        With a scorer the cross-encoder orders them (equal scores keep the fused order: ``sorted`` is
        stable); without one the fused order stands. A candidate the store no longer returns (its
        document stopped being ``ready`` in the meantime) is skipped. ``RerankError`` if the scorer
        answers with the wrong number of scores.
        """
        _check_k(k)
        started = self._clock()
        found = {chunk.chunk_id: chunk for chunk in self._resolve([h["chunk_id"] for h in fused])}
        ready = [(hit, found[hit["chunk_id"]]) for hit in fused if hit["chunk_id"] in found]
        if self._scorer is None or not ready:
            chunks = tuple(ScoredChunk(chunk, hit["score"]) for hit, chunk in ready[:k])
        else:
            scores = self._scorer.score(query, [chunk.text for _, chunk in ready])
            if len(scores) != len(ready):
                raise RerankError(
                    f"asked for {len(ready)} scores, the scorer returned {len(scores)}"
                )
            order = sorted(range(len(ready)), key=lambda i: -scores[i])
            chunks = tuple(ScoredChunk(ready[i][1], scores[i]) for i in order[:k])
        return StageRanked(chunks, (self._clock() - started) * _MILLISECONDS)

    def timings(
        self,
        dense_ms: float,
        lexical_ms: float,
        fusion_ms: float,
        rerank_ms: float,
        *,
        parallel: bool,
    ) -> dict[str, float]:
        """The stage times in the shape every retriever reports (``dense``, ``lexical``, ``fusion``,
        ``rerank`` when there is a scorer, ``total``).

        ``total`` is the time the user waits: the slower search plus fusion and reranking when the two
        searches ran side by side (``parallel``), the sum of all four when they ran one after the
        other. The two numbers are what a comparison of the schedules needs.
        """
        searching = max(dense_ms, lexical_ms) if parallel else dense_ms + lexical_ms
        timings = {"dense": dense_ms, "lexical": lexical_ms, "fusion": fusion_ms}
        if self._scorer is not None:
            timings["rerank"] = rerank_ms
        timings["total"] = searching + fusion_ms + rerank_ms
        return timings

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` best chunks, the four stages run in order (the sequential schedule)."""
        dense = self.dense_stage(query, k)
        lexical = self.lexical_stage(query, k)
        fused = self.fuse_stage(dense.hits, lexical.hits, k)
        ranked = self.rerank_stage(query, fused.hits, k)
        timings = self.timings(dense.ms, lexical.ms, fused.ms, ranked.ms, parallel=False)
        return RetrievalResult(chunks=ranked.chunks, timings_ms=timings)


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
