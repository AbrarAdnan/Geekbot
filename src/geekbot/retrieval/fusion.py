"""Reciprocal Rank Fusion: merge ranked lists by rank, not by score (step S2.2, PLAN.md §9 rung R6).

Dense and lexical retrieval each return a ranked list, and each is good at what the other misses:
dense finds a passage that says the same thing in other words, BM25 finds the exact rare word, the
error code, the name. To use both, the lists have to be merged, and the obvious merge, adding the
scores, does not work: a cosine similarity lives in [-1, 1] and a BM25 score in [0, 30] or so, and
the second would drown the first. Rescaling them (min-max, z-scores) depends on the other results of
the same query and is fragile.

**RRF** (Cormack, Clarke and Buettcher, 2009) throws the scores away and keeps the ranks. A chunk's
fused score is the sum, over every list it appears in, of ``1 / (k + rank)``, with rank 1 for the
best. A chunk near the top of both lists wins; a chunk at the top of one and absent from the other
still scores well; and nothing depends on the scale of the original scores. ``k`` (60 in the
original paper and in this plan) flattens the curve: with a small ``k`` rank 1 dominates rank 2, with
a large one the ranks matter less and agreement between lists matters more.
"""

from collections.abc import Sequence
from typing import Final

from geekbot.domain import Chunk, ScoredChunk

__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]

DEFAULT_RRF_K: Final = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Merge best-first rankings into one, scoring each chunk by ``sum(1 / (k + rank))``.

    ✍️ Written by Claude (prototyping mode, D-047). ``rankings`` are lists of chunks, best first;
    the scores inside them are ignored. A chunk is identified by its ``chunk_id``; if a list holds
    the same chunk twice only its first, best, position counts. The result holds every chunk that
    appears in any list once, with its fused score (higher is better), sorted by that score.

    The order is fully deterministic, which a benchmark needs. Equal fused scores are broken by
    the chunk's best rank in any list (a better rank first), then by which list it came from first
    (the earlier list in ``rankings`` first), then by ``chunk_id``. The *scores* do not depend on the
    order of ``rankings``; only these ties do.

    ``limit`` keeps the first ``limit`` results. ``ValueError`` if ``k`` is negative or ``limit`` is
    below 1. Empty input, or lists that are all empty, give an empty result.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_list: dict[str, int] = {}
    chunks: dict[str, Chunk] = {}
    for list_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        rank = 0
        for scored in ranking:
            chunk_id = scored.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = scored.chunk
                best_rank[chunk_id] = rank
                first_list[chunk_id] = list_index
            else:
                best_rank[chunk_id] = min(best_rank[chunk_id], rank)
    # Sums of reciprocals that are equal on paper can differ in the last bits when they are added in
    # another order (floating point addition is not associative), so the sort looks at the score
    # rounded to 12 places: a tie on paper is then a tie in the code and the tie-breakers apply.
    ordered = sorted(
        fused,
        key=lambda cid: (-round(fused[cid], 12), best_rank[cid], first_list[cid], cid),
    )
    result = [ScoredChunk(chunk=chunks[cid], score=fused[cid]) for cid in ordered]
    return result if limit is None else result[:limit]
