"""Run retrievers over question sets and score them (step S2.6, PLAN.md §9).

This is the part of the ablation that is the same for every phase (the ladder, the sweeps, the
bake-offs, the abstain calibration): turn a rung into a retriever, run a question set through it
with the clock and the CPU gate around the loop, and turn the rankings into per-question metrics.
The phases themselves (which rungs, which indexes, what is written where) are in ``phases.py``.

Two choices that shape every number:

- **Every system returns its top ``k`` chunks** (10 by default), and a ranking of documents is made
  from those chunks by keeping each document at its best chunk (PLAN.md §9, "max"). That is what
  the answer graph would be handed, so it measures the system as a user meets it. The price is that
  chunk size changes how many *distinct* documents ten chunks cover; ``docs_per_query`` is reported
  next to every score so that effect is visible instead of hidden inside rung R4.
- **Per-question scores are kept**, not only means: the statistics (``evals.stats``) are paired by
  question, and the report recomputes them without re-running anything.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from geekbot.evals.ablation import Strategy
from geekbot.evals.cpugate import CpuSnapshot, GateReading, read_snapshot, wait_until_idle
from geekbot.evals.indexing import BuiltIndex
from geekbot.evals.legacy import LegacyMultiQueryRetriever
from geekbot.evals.metrics import (
    MetricFn,
    aggregate_to_documents,
    mean_defined,
    ndcg_at_k,
    per_query,
    recall_at_k,
    reciprocal_rank,
)
from geekbot.retrieval import Retriever
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.lexical import LexicalRetriever
from geekbot.retrieval.rerank import PassageScorer, Reranker
from geekbot.retrieval.rerank_timing import percentile

__all__ = [
    "METRICS",
    "MemoScorer",
    "QuerySet",
    "RunOutput",
    "build_retriever_for",
    "record_of",
    "run_queries",
    "score_rankings",
]

# name -> (function, cutoff). The headline metrics are nDCG@10 and Recall@5 (PLAN.md §9).
METRICS: Final[dict[str, tuple[MetricFn, int]]] = {
    "recall@5": (recall_at_k, 5),
    "recall@10": (recall_at_k, 10),
    "ndcg@10": (ndcg_at_k, 10),
    "mrr@10": (reciprocal_rank, 10),
}


@dataclass(frozen=True)
class QuerySet:
    """Questions and what answers them. ``qrels`` is keyed by ``doc_id`` (the store's document ID)."""

    name: str
    queries: Mapping[str, str]
    qrels: Mapping[str, Mapping[str, int]]


class MemoScorer:
    """Remembers what a scorer said about each (question, passage), so a grid costs one pass.

    The depth and fusion sweeps ask the same reranker about mostly the same passages again and
    again. A cross-encoder is deterministic, so remembering its answers changes no number; it only
    saves the time. Never used when timing (a remembered answer takes no time).
    """

    def __init__(self, inner: PassageScorer) -> None:
        self._inner = inner
        self._seen: dict[tuple[str, str], float] = {}

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        missing = [p for p in dict.fromkeys(passages) if (query, p) not in self._seen]
        if missing:
            for passage, value in zip(missing, self._inner.score(query, missing), strict=True):
                self._seen[(query, passage)] = value
        return [self._seen[(query, p)] for p in passages]


def build_retriever_for(
    strategy: Strategy,
    built: BuiltIndex,
    scorer: PassageScorer | None,
    *,
    depth: int,
    candidates: int,
) -> Retriever:
    """The retriever a rung's strategy names, reading from ``built``.

    ``scorer`` is needed (``ValueError`` otherwise) for the two strategies that rerank.
    """
    if strategy == "legacy_multiquery":
        return LegacyMultiQueryRetriever(built.store, built.dense, built.embeddings)
    dense = DenseRetriever(built.store, built.dense, built.cached)
    if strategy == "dense":
        return dense
    if strategy == "hybrid":
        return HybridRetriever(dense, LexicalRetriever(built.store), candidates=candidates)
    if scorer is None:
        raise ValueError(f"the {strategy} strategy needs a reranker")
    if strategy == "dense_rerank":
        return Reranker(dense, scorer, depth=depth)
    hybrid = HybridRetriever(dense, LexicalRetriever(built.store), candidates=candidates)
    return Reranker(hybrid, scorer, depth=depth)


@dataclass(frozen=True)
class RunOutput:
    """What running one question set through one retriever produced."""

    rankings: dict[str, list[str]]  # question -> doc_id of each returned chunk, best first
    top_scores: dict[str, float | None]  # question -> the best chunk's score (calibration)
    latencies_ms: list[float]
    stage_ms: dict[str, float]  # mean ms per stage the retriever reported
    gate: GateReading


def run_queries(
    retriever: Retriever,
    queries: Mapping[str, str],
    *,
    k: int,
    warmup: int = 3,
    idle_percent: float = 15.0,
    gate: Callable[..., GateReading] = wait_until_idle,
    snapshot: Callable[[], CpuSnapshot] = read_snapshot,
    clock: Callable[[], float] = time.perf_counter,
) -> RunOutput:
    """Search every question once, timing each search; the CPU gate brackets the whole loop.

    The first ``warmup`` questions are searched once beforehand and not timed: the first embedding
    request loads the model into memory and the first rerank loads the ONNX session, and neither is
    what a user's tenth question costs. (The first request of a *session* is a real cost, but it is
    measured by ``geekbot ask-timing``, not here.)
    """
    reading = gate(idle_percent=idle_percent)
    ids = list(queries)
    for question_id in ids[:warmup]:
        retriever.search(queries[question_id], k)
    rankings: dict[str, list[str]] = {}
    top_scores: dict[str, float | None] = {}
    latencies: list[float] = []
    stage_totals: dict[str, float] = {}
    for question_id in ids:
        started = clock()
        result = retriever.search(queries[question_id], k)
        latencies.append((clock() - started) * 1000)
        rankings[question_id] = [scored.chunk.doc_id for scored in result.chunks]
        top_scores[question_id] = result.chunks[0].score if result.chunks else None
        for stage, ms in result.timings_ms.items():
            if stage != "total":
                stage_totals[stage] = stage_totals.get(stage, 0.0) + ms
    count = max(len(ids), 1)
    return RunOutput(
        rankings=rankings,
        top_scores=top_scores,
        latencies_ms=latencies,
        stage_ms={stage: total / count for stage, total in stage_totals.items()},
        gate=replace(reading, after=snapshot()),
    )


def score_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, float]]:
    """Every metric of ``METRICS`` for every question that has a defined score: ``{metric: {id: v}}``."""
    return {
        name: per_query(function, rankings, qrels, cutoff)
        for name, (function, cutoff) in METRICS.items()
    }


def record_of(
    output: RunOutput, qrels: Mapping[str, Mapping[str, int]], **identity: Any
) -> dict[str, Any]:
    """The JSON record of one (rung, question set) run: means, per-question scores, latency, gate.

    ``identity`` (rung name, variant key, strategy, ...) is stored as given, so a result file says
    what it measured without the reader needing the code. A question whose metric is undefined is
    left out of that metric's mean and is counted (``scored`` against ``questions``).
    """
    per_question = score_rankings(output.rankings, qrels)
    means = {
        name: mean_defined(scores, total=len(qrels)).mean if scores else None
        for name, scores in per_question.items()
    }
    distinct = [len(aggregate_to_documents(ranking)) for ranking in output.rankings.values()]
    latencies = output.latencies_ms
    latency = (
        {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": sum(latencies) / len(latencies),
        }
        if latencies
        else {}
    )
    return {
        **identity,
        "questions": len(output.rankings),
        "scored": {name: len(scores) for name, scores in per_question.items()},
        "metrics": means,
        "per_query": per_question,
        "docs_per_query": sum(distinct) / len(distinct) if distinct else None,
        "latency_ms": latency,
        "stage_ms": output.stage_ms,
        "gate": output.gate.to_json(),
    }
