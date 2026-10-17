"""How long does reranking take at each depth? (step S2.3, PLAN.md §6.5, §9 rung R7.)

The plan gives retrieval plus reranking 400 ms at p95 on the CPU, and the one knob that trades
quality for time is the **depth**: how many candidates the cross-encoder re-scores. A cross-encoder
reads every candidate with the question, so its time grows about in step with the depth. Choosing
the depth by feel would put an unmeasured number into the design, so this module measures it: the
same queries, the same passages, a growing prefix of them, many repeats, and the p50 and p95 of the
wall-clock time of each scoring call.

Why percentiles and not an average. One slow call (the machine was busy with something else) moves
an average a lot and says little; the median (p50) is the typical call and p95 is the slow end that a
user actually notices, which is also what the plan's budget is written in.

The scorer is a parameter, so the harness is tested with a fake and a fake clock; only the command
that wraps it (``geekbot rerank-timing``) loads the real model. Whatever it prints is a measurement
of that machine at that moment: on a PC shared with other work the numbers are only as steady as
the CPU is idle, and any number quoted must name the model.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from geekbot.retrieval.rerank import PassageScorer

__all__ = ["DepthTiming", "format_depth_timings", "measure_rerank_depths", "percentile"]


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``values`` (0 to 100), interpolating between the closest ranks.

    ✍️ Written by Claude (prototyping mode, D-047), with ``measure_rerank_depths`` below. The
    method is the common "linear" one (NumPy's default): sort the values, put the percentile at
    position ``q / 100 * (n - 1)`` and, if that falls between two values, take the proportional
    point between them. So the 0th percentile is the smallest value, the 100th the largest, the
    50th of an even number of values the mean of the middle two, and one value is every percentile
    of itself. ``ValueError`` for no values or a ``q`` outside 0 to 100.
    """
    if not values:
        raise ValueError("percentile of no values")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    position = q / 100 * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class DepthTiming:
    """The scoring time at one depth, in milliseconds, over ``runs`` timed calls."""

    depth: int
    runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def measure_rerank_depths(
    scorer: PassageScorer,
    queries: Sequence[str],
    passages: Sequence[str],
    depths: Sequence[int],
    *,
    repeats: int = 5,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> list[DepthTiming]:
    """Time ``scorer.score(query, passages[:depth])`` for each depth, ``repeats`` times per query.

    ✍️ Written by Claude (prototyping mode, D-047). ``warmup`` untimed calls at the largest depth run
    first, so that loading the model (and, the first time, downloading it) is not counted in any
    depth. Each depth then makes ``len(queries) * repeats`` timed calls, and every depth scores the
    *same* leading passages, so the only thing that changes between rows is how many there are.
    Returns one ``DepthTiming`` per depth, in the order given. ``ValueError`` for no queries, no
    depths, a depth below 1 or above the number of passages, ``repeats`` below 1 or a negative
    ``warmup``.
    """
    if not queries:
        raise ValueError("give at least one query")
    if not depths:
        raise ValueError("give at least one depth")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must not be negative, got {warmup}")
    for depth in depths:
        if depth < 1:
            raise ValueError(f"a depth must be at least 1, got {depth}")
        if depth > len(passages):
            raise ValueError(f"depth {depth} is more than the {len(passages)} passages available")
    deepest = max(depths)
    for _ in range(warmup):
        scorer.score(queries[0], passages[:deepest])
    results: list[DepthTiming] = []
    for depth in depths:
        samples: list[float] = []
        for query in queries:
            for _ in range(repeats):
                start = clock()
                scorer.score(query, passages[:depth])
                samples.append((clock() - start) * 1000)
        results.append(
            DepthTiming(
                depth=depth,
                runs=len(samples),
                p50_ms=percentile(samples, 50),
                p95_ms=percentile(samples, 95),
                max_ms=max(samples),
            )
        )
    return results


def format_depth_timings(model: str, timings: Sequence[DepthTiming]) -> str:
    """A table for the terminal, with the model named and what the numbers are."""
    lines = [
        f"reranker {model} on the CPU",
        f"{'depth':>6} {'runs':>5} {'p50 ms':>9} {'p95 ms':>9} {'max ms':>9}",
    ]
    for row in timings:
        lines.append(
            f"{row.depth:>6} {row.runs:>5} {row.p50_ms:>9.1f} {row.p95_ms:>9.1f} {row.max_ms:>9.1f}"
        )
    lines.append(
        "wall-clock time of one scoring call on this machine, now. The plan's budget is 400 ms "
        "at p95 for retrieval and reranking together (PLAN.md section 6.5)."
    )
    return "\n".join(lines)
