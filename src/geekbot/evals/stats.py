"""Is system B really better than system A, or is it luck? (step S2.5, PLAN.md §9.)

A retrieval metric averaged over a few hundred questions is a noisy number: run the same two
systems on another few hundred questions and the gap changes. So a gap (Δ) is never reported alone.
The plan fixes the recipe, in this order (PLAN.md §9): **Δ first**, then its **95% confidence
interval** from a *paired bootstrap*, then a **paired sign-flip permutation p-value**, then a
**Holm correction** across all the comparisons made. "Significant" is never written without the Δ and
the interval.

*Paired* means both systems are scored on the same questions, and the unit of analysis is the
per-question difference ``b - a``. That removes the variation between easy and hard questions, which
affects both systems alike, and leaves only what the change did.

- **Paired bootstrap** (``paired_bootstrap_ci``): pretend the questions are a sample from a larger
  population. Resample them with replacement many times, recompute the mean difference each time,
  and take the middle 95% of those means as the interval. If the interval excludes 0, the sign of the
  gap is stable under resampling of the questions.
- **Sign-flip permutation test** (``paired_permutation_pvalue``): if the two systems were really
  equivalent, the sign of each question's difference would be a coin flip. Flip the signs at random
  many times; the p-value is the share of flips whose mean difference is at least as large (in size)
  as the real one. Few questions make exact enumeration possible; the code does it then.
- **Holm correction** (``holm_adjust``): ask twenty questions of the data and one will look
  significant by luck. Holm's step-down method adjusts a list of p-values so that the chance of *any*
  false alarm stays below the chosen level, and it is never less powerful than plain Bonferroni.

**Which comparisons form the Holm family** (rungs × primary metrics) is an open decision of the plan
(D-033). This module supplies the mechanism only: the caller passes the list of p-values it decided
belong together. Defining that list is part of S2.4 to S2.6 and is the user's to settle.

All randomness comes from a seed the caller passes, so a result can be regenerated exactly.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "BootstrapCI",
    "Comparison",
    "compare_paired",
    "format_comparison",
    "holm_adjust",
    "paired_bootstrap_ci",
    "paired_permutation_pvalue",
]

DEFAULT_RESAMPLES: Final = 10_000
DEFAULT_CONFIDENCE: Final = 0.95
_BATCH: Final = 500  # resamples drawn at once, to bound memory for a large number of questions


def _differences(a: Sequence[float], b: Sequence[float]) -> NDArray[np.float64]:
    """``b - a`` per question, checked: same length, at least one question, no NaN or infinity."""
    if len(a) != len(b):
        raise ValueError(f"paired scores must have the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("no questions to compare")
    diffs = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(diffs)):
        raise ValueError("scores must be finite numbers (no NaN, no infinity)")
    return diffs


@dataclass(frozen=True)
class BootstrapCI:
    """The mean difference ``b - a`` and its confidence interval."""

    delta: float
    low: float
    high: float
    confidence: float
    resamples: int


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapCI:
    """The mean of ``b - a`` over the questions, with a percentile bootstrap interval.

    ✍️ Written by Claude (prototyping mode, D-047). ``a`` and ``b`` are per-question scores of the
    baseline and the new system, in the same question order. The questions are resampled with
    replacement ``resamples`` times (10,000 in the plan); each resample has as many questions as the
    original and gives one mean difference. The interval is the percentile range of those means that
    holds ``confidence`` of them: for 0.95, from the 2.5th to the 97.5th percentile. ``delta`` is the
    mean difference of the real data, not of the resamples. The same ``seed`` gives the same
    interval. With one question, or when every difference is equal, the interval collapses to
    ``delta``.

    ``ValueError`` if the lists differ in length, are empty or hold a NaN, if ``resamples < 1`` or
    if ``confidence`` is not strictly between 0 and 1.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    diffs = _differences(a, b)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    done = 0
    while done < resamples:
        batch = min(_BATCH, resamples - done)
        picks = rng.integers(0, n, size=(batch, n))
        means[done : done + batch] = diffs[picks].mean(axis=1)
        done += batch
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return BootstrapCI(
        delta=float(diffs.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
    )


def paired_permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value for "the two systems are equivalent".

    The statistic is the absolute mean of the per-question differences ``b - a``. Under the claim
    that the systems are equivalent each difference is as likely to be negative as positive, so
    flipping any subset of the signs gives an equally likely dataset. The p-value is the share of
    sign patterns whose statistic is at least as large as the real one.

    With few questions (when ``2**n`` is not more than ``resamples``: up to 13 questions at the
    default) all ``2**n`` patterns are enumerated and the p-value is exact. With more, ``resamples`` random patterns are drawn and
    the real data counts as one more pattern (``(hits + 1) / (resamples + 1)``), which is the correct
    estimator and means the p-value is never 0 and cannot be below ``1 / (resamples + 1)``. All
    differences equal to zero gives 1.0. ``ValueError`` as for ``paired_bootstrap_ci``.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    diffs = _differences(a, b)
    n = len(diffs)
    observed = abs(diffs.mean())
    if observed == 0.0 and not np.any(diffs):
        return 1.0
    # The tolerance keeps a pattern that ties with the real data on paper but differs in the last
    # bits from being counted as smaller.
    tolerance = 1e-12 * max(1.0, float(np.abs(diffs).sum()))
    threshold = observed * n - tolerance  # compare sums, not means, to avoid a division per pattern
    if n <= 20 and (1 << n) <= resamples:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        sums = np.abs(signs @ diffs)
        return int(np.count_nonzero(sums >= threshold)) / (1 << n)
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < resamples:
        batch = min(_BATCH, resamples - done)
        signs = rng.choice((-1.0, 1.0), size=(batch, n))
        hits += int(np.count_nonzero(np.abs(signs @ diffs) >= threshold))
        done += batch
    return (hits + 1) / (resamples + 1)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Sort the ``m`` p-values from smallest to largest. The smallest is multiplied by ``m``, the next
    by ``m - 1``, and so on down to 1 for the largest; each is capped at 1; and no adjusted value
    may be smaller than the one before it (an adjusted p-value cannot be better than that of a
    result that was more convincing). Rejecting every hypothesis whose adjusted p-value is at most
    ``alpha`` keeps the chance of any false rejection at or below ``alpha``, whatever the
    dependence between the tests. The empty list gives the empty list. ``ValueError`` for a value
    outside 0 to 1 or NaN.
    """
    for p in pvalues:
        if math.isnan(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"a p-value must be between 0 and 1, got {p}")
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * pvalues[index]))
        adjusted[index] = running
    return adjusted


@dataclass(frozen=True)
class Comparison:
    """One paired comparison: the gap, its interval and its p-value (raw, before Holm)."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Comparison:
    """Δ, its bootstrap interval and its permutation p-value for ``b`` against ``a``.

    Both tests use the same ``seed`` and number of resamples. The p-value is *not* adjusted: collect
    the p-values of the whole family and pass them to ``holm_adjust``.
    """
    ci = paired_bootstrap_ci(a, b, resamples=resamples, confidence=confidence, seed=seed)
    p = paired_permutation_pvalue(a, b, resamples=resamples, seed=seed)
    return Comparison(
        n=len(a),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        delta=ci.delta,
        ci_low=ci.low,
        ci_high=ci.high,
        confidence=confidence,
        p_value=p,
    )


def format_comparison(name: str, comparison: Comparison, *, adjusted_p: float | None = None) -> str:
    """One line in the order the plan requires: Δ, then its interval, then p (and the adjusted p).

    The words "significant" or "not significant" are deliberately not produced: a reader decides
    from the numbers.
    """
    c = comparison
    line = (
        f"{name}: n={c.n}, A={c.mean_a:.4f}, B={c.mean_b:.4f}, "
        f"delta={c.delta:+.4f} [{c.ci_low:+.4f}, {c.ci_high:+.4f}] ({c.confidence:.0%} CI), "
        f"p={c.p_value:.4f}"
    )
    if adjusted_p is not None:
        line += f", Holm-adjusted p={adjusted_p:.4f}"
    return line
