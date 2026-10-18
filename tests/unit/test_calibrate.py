"""Tests for the abstain-threshold calibration (step S2.6, the step's ✍️ piece).

The questions are tiny hand-made sets whose best threshold can be worked out on paper.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.retrieval.calibrate import (
    Labelled,
    SweepPoint,
    best_threshold,
    bootstrap_interval,
    candidate_thresholds,
    confusion,
    sweep,
)


def q(item_id: str, score: float | None, answerable: bool) -> Labelled:
    return Labelled(item_id, score, answerable)


# Unanswerable questions score low, answerable high, with one overlap: u3 scores above a2.
SEPARABLE = [
    q("a1", 8.0, True),
    q("a2", 5.0, True),
    q("a3", 6.0, True),
    q("u1", -4.0, False),
    q("u2", -2.0, False),
]
OVERLAP = [*SEPARABLE, q("u3", 5.5, False)]


def test_a_threshold_is_counted_into_the_four_outcomes() -> None:
    point = confusion(OVERLAP, 0.0)  # abstain below 0: u1 and u2 only
    assert (point.tp, point.fp, point.fn, point.tn) == (2, 0, 1, 3)
    assert point.precision == 1.0
    assert point.recall == pytest.approx(2 / 3)
    assert point.f1 == pytest.approx(0.8)
    assert point.accuracy == pytest.approx(5 / 6)


def test_the_threshold_itself_does_not_abstain() -> None:
    assert confusion([q("a", 3.0, True)], 3.0).tn == 1  # "below", not "at or below"
    assert confusion([q("a", 3.0, True)], 3.0001).fp == 1


def test_a_question_with_no_score_always_abstains() -> None:
    items = [q("a", None, True), q("u", None, False)]
    point = confusion(items, -1e9)
    assert (point.tp, point.fp) == (1, 1)  # even at the lowest possible threshold


def test_precision_and_recall_are_zero_not_an_error_when_undefined() -> None:
    nobody = confusion(SEPARABLE, -100.0)  # abstains on nothing
    assert (nobody.tp, nobody.fp) == (0, 0)
    assert nobody.precision == 0.0
    assert nobody.f1 == 0.0
    assert SweepPoint(0.0, 0, 0, 0, 0).accuracy == 0.0


def test_the_best_threshold_separates_a_separable_set_perfectly() -> None:
    best = best_threshold(SEPARABLE)
    assert best.f1 == 1.0
    assert -2.0 < best.threshold < 5.0  # anywhere in the gap works; the lowest tied one is chosen
    assert best.threshold == pytest.approx(1.5)  # the midpoint of the gap between -2 and 5


def test_with_an_overlap_the_best_threshold_accepts_the_cheaper_mistake() -> None:
    best = best_threshold(OVERLAP)
    # Abstaining below 5.75 catches all three unanswerable questions (u3 scores 5.5) and withholds
    # one good answer (a2 scores 5.0): precision 3/4, recall 1, F1 = 6/7. Abstaining below 1.5
    # withholds nothing but lets u3 through (F1 0.8); below 7 withholds two answers (F1 0.75).
    assert best.f1 == pytest.approx(6 / 7)
    assert best.threshold == pytest.approx(5.75)
    assert (best.tp, best.fp, best.fn, best.tn) == (3, 1, 0, 2)


def test_equal_f1_goes_to_the_higher_accuracy() -> None:
    # u1=0, a1=2, a2=4, u2=6. Abstaining below 1 (only u1) and below 7 (everything) both give
    # F1 = 2/3, but the first answers three of four questions correctly and the second only two.
    items = [q("u1", 0.0, False), q("a1", 2.0, True), q("a2", 4.0, True), q("u2", 6.0, False)]
    by_threshold = {p.threshold: p for p in sweep(items)}
    assert by_threshold[1.0].f1 == pytest.approx(by_threshold[7.0].f1)
    assert by_threshold[1.0].accuracy > by_threshold[7.0].accuracy
    assert best_threshold(items).threshold == pytest.approx(1.0)


def test_abstaining_on_everything_wins_when_unanswerable_questions_score_highest() -> None:
    items = [q("a", 0.0, True), q("u", 5.0, False)]  # the signal is backwards for this set
    best = best_threshold(items)
    assert best.threshold > 5.0  # the only way to catch u is to abstain on everything
    assert (best.tp, best.fp) == (1, 1)


def test_candidate_thresholds_cover_every_distinct_decision() -> None:
    items = [q("a", 1.0, True), q("b", 1.0, True), q("c", 3.0, False)]
    assert candidate_thresholds(items) == [0.0, 2.0, 4.0]  # below all, between 1 and 3, above all
    assert candidate_thresholds([q("n", None, False)]) == [0.0]  # nothing scored


def test_sweep_refuses_sets_with_nothing_to_calibrate() -> None:
    with pytest.raises(ValueError, match="no questions"):
        sweep([])
    with pytest.raises(ValueError, match="no unanswerable"):
        sweep([q("a", 1.0, True)])


def test_the_bootstrap_interval_is_narrow_for_a_clean_set_and_wide_for_a_noisy_one() -> None:
    clean_low, clean_high = bootstrap_interval(SEPARABLE, resamples=200, seed=1)
    assert -2.0 < clean_low <= clean_high < 9.0
    noisy = [q(f"a{i}", float(i % 7), True) for i in range(20)]
    noisy += [q(f"u{i}", float((i * 3) % 7), False) for i in range(6)]
    low, high = bootstrap_interval(noisy, resamples=200, seed=1)
    assert high - low > 1.0


def test_the_bootstrap_is_repeatable_and_validates_its_input() -> None:
    assert bootstrap_interval(OVERLAP, resamples=100, seed=7) == bootstrap_interval(
        OVERLAP, resamples=100, seed=7
    )
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval(OVERLAP, confidence=1.0)
    with pytest.raises(ValueError, match="no unanswerable"):
        bootstrap_interval([q("a", 1.0, True)])


@given(
    st.lists(
        st.tuples(
            st.one_of(st.none(), st.floats(-20, 20)),
            st.booleans(),
        ),
        min_size=2,
        max_size=40,
    ).filter(lambda rows: any(not answerable for _, answerable in rows))
)
def test_the_best_threshold_is_never_beaten_by_any_candidate(
    rows: list[tuple[float | None, bool]],
) -> None:
    items = [q(f"i{n}", score, answerable) for n, (score, answerable) in enumerate(rows)]
    best = best_threshold(items)
    assert all(point.f1 <= best.f1 + 1e-12 for point in sweep(items))
    total = best.tp + best.fp + best.fn + best.tn
    assert total == len(items)  # every question lands in exactly one cell
