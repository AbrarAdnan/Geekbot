"""Reranking: re-score the best candidates with a model that reads question and passage together
(step S2.3, PLAN.md §6.5, §9 rung R7).

Dense and BM25 compare a question with a chunk *separately*: each is turned into a vector or a bag of
words first, and the two are compared afterwards. That is fast (the chunk side is computed once, at
ingest) and loses detail: nothing in a vector lets "not" in the question meet "not" in the passage.
A **cross-encoder** reads the question and one passage *together* in a single pass and outputs how
well the passage answers the question. It is far more accurate and far too slow to run over every
chunk, so it runs only on the few dozen candidates the fast retrievers found: the retrieval is a
funnel, wide and cheap first, narrow and careful last.

The price is time on the CPU (the GPU belongs to the chat model, PLAN.md §6.5), which grows with the
number of candidates: the **depth**. The plan gives retrieval plus reranking 400 ms at p95, so the
depth is tuned against that budget (``geekbot rerank-timing`` measures it), not guessed.

``PassageScorer`` is the one thing the reranker needs from a model: scores for a list of passages.
``FastembedScorer`` is the real one (ONNX, CPU, weights downloaded on first use); tests use a fake.
"""

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from geekbot.domain import ScoredChunk
from geekbot.retrieval import RetrievalResult, Retriever

__all__ = [
    "DEFAULT_RERANK_DEPTH",
    "DEFAULT_RERANK_MODEL",
    "FastembedScorer",
    "PassageScorer",
    "RerankError",
    "Reranker",
]

DEFAULT_RERANK_MODEL: Final = "Xenova/ms-marco-MiniLM-L-6-v2"
# 10, not the 30 it started at (D-062). The plan gives retrieval plus reranking 400 ms at p95 (PLAN.md
# §6.5). Timed through the whole pipeline on an idle PC (D-077; STATUS.md, "Reranker timings"), 10
# candidates take about 330 ms at p50 and 380 ms at p95, 30 candidates about 1,065 ms at p50. That leaves
# little room at 10 and none at 30. What the smaller depth costs in recall is *not* measured: S2.6
# sweeps the depth on the development items and may change this again.
DEFAULT_RERANK_DEPTH: Final = 10


class RerankError(RuntimeError):
    """The reranker could not score: its model did not load, or it answered with the wrong shape."""


class PassageScorer(Protocol):
    """Scores passages against a question: a higher score means the passage answers it better."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One score per passage, in the same order. Scores are only comparable within one call."""
        ...


class FastembedScorer:
    """A cross-encoder from ``fastembed``: ONNX on the CPU, weights fetched on first use.

    Nothing is loaded when the object is created. The first ``score`` call imports ``fastembed``
    and loads (and, the first time ever, downloads) the model, so a command that never reranks never
    pays for it, and a missing model is reported when it matters. ``cache_dir`` is where the weights
    live (``<data_dir>/models``). The first run needs network access to Hugging Face; later runs
    are offline.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._encoder: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._encoder is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._encoder = TextCrossEncoder(
                        model_name=self.model_name,
                        cache_dir=None if self._cache_dir is None else str(self._cache_dir),
                        threads=self._threads,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as error:
                    raise RerankError(
                        f"the reranker model {self.model_name!r} could not be loaded "
                        f"({type(error).__name__}: {error}). The first use downloads it from "
                        "Hugging Face, so it needs network access once."
                    ) from error
            return self._encoder

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Scores for ``passages`` against ``query``, from the cross-encoder."""
        if not passages:
            return []
        encoder = self._load()
        try:
            scores = [float(s) for s in encoder.rerank(query, list(passages))]
        except Exception as error:
            raise RerankError(f"the reranker failed: {type(error).__name__}: {error}") from error
        if len(scores) != len(passages):
            raise RerankError(
                f"asked for {len(passages)} scores, the reranker returned {len(scores)}"
            )
        return scores


class Reranker:
    """Re-order another retriever's best candidates with a cross-encoder. Satisfies ``Retriever``.

    ``base`` is usually the hybrid retriever. The reranker asks it for ``max(k, depth)`` chunks,
    scores each chunk's text against the question, and returns the ``k`` best by that score (the
    score is the model's, higher is better, and only comparable within one search). Chunks the
    model scores equally keep the order the base retriever gave them. The chunk's text is scored
    as it is stored, without the contextual header, so that rung R7 changes one thing.
    """

    def __init__(
        self,
        base: Retriever,
        scorer: PassageScorer,
        *,
        depth: int = DEFAULT_RERANK_DEPTH,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._base = base
        self._scorer = scorer
        self._depth = depth
        self._clock = clock

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        """The ``k`` best of the base retriever's top ``max(k, depth)`` chunks, by cross-encoder score.

        With no candidates the scorer is not called. ``ValueError`` if ``k < 1``; ``RerankError``
        if the scorer fails or returns the wrong number of scores.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        start = self._clock()
        candidates = self._base.search(query, max(k, self._depth))
        after_base = self._clock()
        chunks = candidates.chunks
        if not chunks:
            return RetrievalResult(
                chunks=(), timings_ms=_timings(candidates, start, after_base, after_base)
            )
        scores = self._scorer.score(query, [scored.chunk.text for scored in chunks])
        if len(scores) != len(chunks):
            raise RerankError(f"asked for {len(chunks)} scores, the scorer returned {len(scores)}")
        # sorted() is stable: equal scores keep the base retriever's order.
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        reranked = tuple(ScoredChunk(chunks[i].chunk, scores[i]) for i in order[:k])
        end = self._clock()
        return RetrievalResult(
            chunks=reranked, timings_ms=_timings(candidates, start, after_base, end)
        )


def _timings(
    candidates: RetrievalResult, start: float, after_base: float, end: float
) -> dict[str, float]:
    """The base retriever's own stages (its total is dropped), then ``rerank`` and the ``total``."""
    timings = {name: ms for name, ms in candidates.timings_ms.items() if name != "total"}
    if not timings:  # a base retriever that reports nothing: show its time as one stage
        timings["candidates"] = (after_base - start) * 1000
    timings["rerank"] = (end - after_base) * 1000
    timings["total"] = (end - start) * 1000
    return timings
