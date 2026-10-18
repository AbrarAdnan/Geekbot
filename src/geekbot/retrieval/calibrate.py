"""Calibrating the abstain threshold (step S2.6, PLAN.md §7 "Abstain", used by S3.5).

**The decision being calibrated.** After retrieval and rerank, the answer graph has to choose:
answer from the sources, or say "the documents do not answer this". The cheapest honest signal is
the reranker's score for its best chunk: a cross-encoder reads the question and the passage together
and returns how well the passage answers it (``retrieval/rerank.py``). The graph abstains when that
best score is **below a threshold**. The threshold is not a constant of nature: it depends on the
reranker model and, a little, on the corpus. It has to be *measured*, once, on questions whose
right answer (answer or abstain) is known, and the measurement must be kept apart from the questions
used to report the result (D-033, D-065).

**What "F1-optimal" means here.** The class to find is *should abstain* (an unanswerable question).
For a threshold ``t``:

- a *true positive* is an unanswerable question whose best score is below ``t`` (it abstains: right);
- a *false positive* is an answerable question that abstains (a good answer was withheld);
- a *false negative* is an unanswerable question that does not abstain (the model is asked to answer
  from unrelated text: the case the whole mechanism exists to prevent);
- precision is ``tp / (tp + fp)``, recall ``tp / (tp + fn)``, and F1 is their harmonic mean. F1
  rewards the threshold that balances the two kinds of mistake; it ignores true negatives, so a
  corpus with many more answerable than unanswerable questions cannot hide a bad threshold behind
  its many easy answers.

**A question with no score** (nothing was retrieved) always abstains: there is nothing to answer
from.

**Ties.** Several thresholds can reach the same best F1. The rule is to prefer the highest accuracy,
then the *lowest* threshold: of two equally good settings, the one that withholds fewer answers.

**How sure is it?** The development set is small (8 unanswerable questions in golden v0), and the
best threshold of a small sample moves when one question changes. ``bootstrap_interval`` redraws the
questions with replacement many times and reports the spread of the best threshold, so the report
can say how stable the number is before anyone builds on it.

✍️ Written by Claude (prototyping mode, D-047); this is the step's ✍️ piece, on the backlog in
``project/STATUS.md`` for the user to redo as an exercise.
"""

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "Labelled",
    "SweepPoint",
    "best_threshold",
    "bootstrap_interval",
    "confusion",
    "sweep",
]

# Scores are cross-encoder logits of no fixed scale; the two thresholds that abstain on nothing and
# on everything sit this far outside the observed range, so they are always strictly outside it.
_MARGIN: Final = 1.0


@dataclass(frozen=True)
class Labelled:
    """One question: its best score (``None`` if nothing was retrieved) and the right behaviour."""

    item_id: str
    score: float | None
    answerable: bool


@dataclass(frozen=True)
class SweepPoint:
    """The outcome of abstaining below ``threshold`` on a set of questions."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0


def abstains(item: Labelled, threshold: float) -> bool:
    """Would the graph abstain on this question at ``threshold``? No score means yes."""
    return item.score is None or item.score < threshold


def confusion(items: Sequence[Labelled], threshold: float) -> SweepPoint:
    """Count the four outcomes of abstaining below ``threshold`` on ``items``."""
    tp = fp = fn = tn = 0
    for item in items:
        abstained = abstains(item, threshold)
        if item.answerable:
            fp += abstained
            tn += not abstained
        else:
            tp += abstained
            fn += not abstained
    return SweepPoint(threshold, tp, fp, fn, tn)


def candidate_thresholds(items: Sequence[Labelled]) -> list[float]:
    """Every threshold worth trying: one in each gap between neighbouring scores, and both ends.

    Between two neighbouring distinct scores every threshold makes the same decisions, so one
    (the midpoint) stands for all. One below the smallest score abstains on no scored question; one
    above the largest abstains on all of them.
    """
    scores = sorted({item.score for item in items if item.score is not None})
    if not scores:
        return [0.0]
    middles = [(low + high) / 2 for low, high in itertools.pairwise(scores)]
    return [scores[0] - _MARGIN, *middles, scores[-1] + _MARGIN]


def sweep(items: Sequence[Labelled]) -> list[SweepPoint]:
    """The outcome at every candidate threshold, from the lowest to the highest.

    ``ValueError`` for no questions, and for a set with no unanswerable question: F1 of the
    abstain class would be 0 everywhere, and "the best threshold" would be an arbitrary choice
    that looks like a result.
    """
    if not items:
        raise ValueError("no questions to calibrate on")
    if all(item.answerable for item in items):
        raise ValueError("no unanswerable question: there is nothing to abstain on")
    return [confusion(items, threshold) for threshold in candidate_thresholds(items)]


def best_threshold(items: Sequence[Labelled]) -> SweepPoint:
    """The F1-optimal point of ``sweep`` (ties: highest accuracy, then lowest threshold)."""
    return max(sweep(items), key=lambda point: (point.f1, point.accuracy, -point.threshold))


def bootstrap_interval(
    items: Sequence[Labelled],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """The central ``confidence`` range of the best threshold over bootstrap resamples of ``items``.

    A resample that happens to contain no unanswerable question has no best threshold and is
    redrawn, so the interval is about thresholds, not about the luck of the draw. ``ValueError``
    if ``items`` itself has none or ``confidence`` is not in (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    sweep(items)  # raises the ValueErrors above for an unusable set
    rng = random.Random(seed)
    found: list[float] = []
    while len(found) < resamples:
        drawn = rng.choices(items, k=len(items))
        if all(item.answerable for item in drawn):
            continue
        found.append(best_threshold(drawn).threshold)
    found.sort()
    tail = (1 - confidence) / 2
    low = found[int(tail * (len(found) - 1))]
    high = found[round((1 - tail) * (len(found) - 1))]
    return low, high
