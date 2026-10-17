"""Retrieval metrics: how good is a ranked list? (step S2.5, PLAN.md §9.)

An evaluation of retrieval needs three things for every question: the **ranking** a system returned
(document IDs, best first), the **judgements** saying which documents are relevant (``qrels``, from
"query relevance judgements": a mapping from document ID to a relevance grade, 0 meaning not
relevant), and a **metric** that turns the two into one number. This module is the third thing.

- **Recall@k**: of the relevant documents, what fraction is in the top ``k``? It answers "did the
  system find what it should have?" and ignores the order inside the top ``k``.
- **nDCG@k** (normalized discounted cumulative gain): a relevant document is worth its grade, but
  worth less the lower it is ranked (divided by ``log2(rank + 1)``), and the sum is divided by the
  best sum any ranking could reach, so 1.0 is a perfect ranking and the number is comparable between
  questions with different numbers of relevant documents. It answers "did the relevant documents
  come first?". It is the primary metric of the plan on the BEIR datasets.
- **MRR** (mean reciprocal rank): ``1 / rank`` of the first relevant document: how far down a
  person has to look to find *something* right.

**Documents, not chunks.** The judgements are about documents (BEIR's, and the golden set's), while
Geekbot retrieves chunks. A chunk ranking is turned into a document ranking by keeping each
document at its best chunk's position (``aggregate_to_documents``; PLAN.md §9: "chunk hits are
aggregated to doc level (max)"). Every metric below also ignores a document's repeats, so a ranking
that still lists the same document twice is scored as if it were listed once.

**Undefined is not zero.** A question with no relevant document at all (an *unanswerable* question,
D-033) has no recall and no nDCG: dividing by nothing. These functions return ``None`` for it, never
0.0, so that averaging cannot quietly count it as a failure; ``mean_defined`` averages the rest and
says how many were skipped.

**The definition of nDCG used here** is trec_eval's: the gain of a document is its grade (linear, not
``2**grade - 1``) and the discount is ``log2(rank + 1)``. For binary judgements (SciFact) the two
conventions coincide; for graded ones (NFCorpus) they differ. Before a number from this module is
quoted next to a published BEIR number, one dataset should be cross-checked against BEIR's own
evaluator (S2.6): the definition is stated here, it has not been compared with that tool.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_KS",
    "MeanScore",
    "MetricFn",
    "QueryScores",
    "aggregate_to_documents",
    "evaluate_query",
    "mean_defined",
    "ndcg_at_k",
    "per_query",
    "recall_at_k",
    "reciprocal_rank",
]

DEFAULT_KS: Final = (5, 10)

# One metric on one question: the ranking, the judgements (document -> grade) and the cutoff.
MetricFn = Callable[[Sequence[str], Mapping[str, int], int], float | None]


def aggregate_to_documents(ranked_doc_ids: Sequence[str]) -> list[str]:
    """Turn a chunk ranking into a document ranking: each document at its best chunk's position.

    ``ranked_doc_ids`` has one entry per chunk, best first (the ``doc_id`` of each retrieved
    chunk). A document that appears several times is kept once, where it first appeared, which is
    where its best-ranked chunk was: max aggregation. The order of first appearances is preserved.
    """
    return list(dict.fromkeys(ranked_doc_ids))


def _relevant(qrels: Mapping[str, int]) -> set[str]:
    """The documents with a grade above 0."""
    return {doc for doc, grade in qrels.items() if grade > 0}


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    """The fraction of the relevant documents that are in the top ``k`` of the ranking.

    ``None`` if the question has no relevant document (the metric is undefined, not zero).
    A document listed twice counts once; a document missing from ``qrels`` is not relevant.
    ``ValueError`` if ``k < 1``.
    """
    _check_k(k)
    relevant = _relevant(qrels)
    if not relevant:
        return None
    top = aggregate_to_documents(ranking)[:k]
    return len(relevant.intersection(top)) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    """``1 / rank`` of the first relevant document in the top ``k``, or 0.0 if there is none there.

    ``None`` if the question has no relevant document at all. ``ValueError`` if ``k < 1``.
    """
    _check_k(k)
    relevant = _relevant(qrels)
    if not relevant:
        return None
    for rank, doc in enumerate(aggregate_to_documents(ranking)[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    """Discounted cumulative gain of grades listed best-ranked first: ``sum(g / log2(rank + 1))``."""
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    """Normalized discounted cumulative gain of the top ``k``: 1.0 is a perfect ranking.

    ✍️ Written by Claude (prototyping mode, D-047). The gain of a document is its grade in
    ``qrels`` (0 if it is not there, so an unjudged document counts as not relevant), and the
    document at rank ``r`` (1 is the best) is discounted by ``log2(r + 1)``, so rank 1 counts
    fully, rank 3 counts half. The sum over the top ``k`` is divided by the largest sum any ranking
    of these judgements could reach in ``k`` places (the grades sorted high to low). A document
    listed twice counts once, at its first position. The result is between 0.0 and 1.0.

    ``None`` if no document has a grade above 0: the best sum is 0, and 0 / 0 is undefined, which
    must not be averaged in as a failure. ``ValueError`` if ``k < 1``.
    """
    _check_k(k)
    ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:k]
    if not ideal:
        return None
    gains = [max(qrels.get(doc, 0), 0) for doc in aggregate_to_documents(ranking)[:k]]
    return _dcg(gains) / _dcg(ideal)


def per_query(
    metric: MetricFn,
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> dict[str, float]:
    """One metric for every question that has a judgement and a defined score, by question ID.

    A question in ``qrels`` with no ranking is scored as if the system returned nothing (0.0, not a
    skip: not answering is a failure). A question whose metric is undefined is left out. These
    per-question numbers are what ``evals.stats`` compares between two systems.
    """
    scores: dict[str, float] = {}
    for query_id, judgements in qrels.items():
        value = metric(rankings.get(query_id, ()), judgements, k)
        if value is not None:
            scores[query_id] = value
    return scores


@dataclass(frozen=True)
class MeanScore:
    """A mean over the questions where the metric was defined, and how many were left out."""

    mean: float
    count: int
    skipped: int


def mean_defined(values: Mapping[str, float], *, total: int | None = None) -> MeanScore:
    """The mean of per-question scores. ``total`` is how many questions there were, to count skips.

    ``ValueError`` if there are no scores: the mean of nothing is not a number worth printing.
    """
    if not values:
        raise ValueError("no question has a defined score")
    count = len(values)
    return MeanScore(
        mean=sum(values.values()) / count,
        count=count,
        skipped=0 if total is None else total - count,
    )


@dataclass(frozen=True)
class QueryScores:
    """The standard metrics of one ranking, by cutoff. ``None`` means undefined (nothing relevant)."""

    recall: Mapping[int, float | None]
    ndcg: Mapping[int, float | None]
    mrr: Mapping[int, float | None]


def evaluate_query(
    ranking: Sequence[str], qrels: Mapping[str, int], ks: Sequence[int] = DEFAULT_KS
) -> QueryScores:
    """Recall, nDCG and MRR of one ranking at each cutoff in ``ks``."""
    return QueryScores(
        recall={k: recall_at_k(ranking, qrels, k) for k in ks},
        ndcg={k: ndcg_at_k(ranking, qrels, k) for k in ks},
        mrr={k: reciprocal_rank(ranking, qrels, k) for k in ks},
    )
