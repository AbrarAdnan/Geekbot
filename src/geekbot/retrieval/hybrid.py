"""Hybrid retrieval: dense and lexical, fused by rank (step S2.2, PLAN.md §9 rung R6).

Each side is asked for more candidates than the final ``k`` (50 by default), because fusion can
only promote what it is shown: a chunk that is rank 12 in both lists belongs near the top of the
fused list, but only if both sides returned it. The two lists are merged with Reciprocal Rank Fusion
(``retrieval/fusion.py``) and cut to ``k``.

The two searches run one after the other here. Running them side by side is S3.6's job and it was
done in the answer graph, not here: ``retrieval.staged.StagedSearch`` is this retriever and the
reranker cut into stages, which the graph runs as "dense ‖ lexical → fuse → rerank". Doing it in the
retriever would have put threads in a place that has to stay easy to test and to measure (the ablation
rungs of S2.6 are built from this class), and the stage timings below say what the overlap can save:
the total here is roughly the sum of ``dense`` and ``lexical``.
"""

import time
from collections.abc import Callable
from typing import Final

from geekbot.retrieval import RetrievalResult, Retriever
from geekbot.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

__all__ = ["DEFAULT_CANDIDATES", "HybridRetriever"]

DEFAULT_CANDIDATES: Final = 50


class HybridRetriever:
    """Fuse a dense and a lexical retriever with RRF. Satisfies ``Retriever``."""

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates}")
        self._dense = dense
        self._lexical = lexical
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._clock = clock

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` best chunks by fused rank; the score is the RRF score (higher is better).

        Each side is asked for ``max(k, candidates)`` chunks. ``timings_ms`` has the time of each
        side (``dense``, ``lexical``), of the fusion, and the ``total``. ``ValueError`` if
        ``k < 1``.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        depth = max(k, self._candidates)
        start = self._clock()
        dense = self._dense.search(query, depth)
        after_dense = self._clock()
        lexical = self._lexical.search(query, depth)
        after_lexical = self._clock()
        fused = reciprocal_rank_fusion([dense.chunks, lexical.chunks], k=self._rrf_k, limit=k)
        end = self._clock()
        timings = {
            "dense": dense.timings_ms.get("total", (after_dense - start) * 1000),
            "lexical": lexical.timings_ms.get("total", (after_lexical - after_dense) * 1000),
            "fusion": (end - after_lexical) * 1000,
            "total": (end - start) * 1000,
        }
        return RetrievalResult(chunks=tuple(fused), timings_ms=timings)
