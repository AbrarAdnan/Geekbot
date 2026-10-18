"""Tests for step S2.5: the statistics (the paired bootstrap CI is this step's ✍️ piece, D-047).

Where a number can be worked out by hand it is (the permutation p-values and the Holm example); the
bootstrap, which is random, is tested on cases with a known answer, on reproducibility, and by a
small simulation of how often its interval covers the truth.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.evals.stats import (
    DEFAULT_RESAMPLES,
    compare_paired,
    format_comparison,
    holm_adjust,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)

# --- the paired bootstrap --------------------------------------------------------------------


def test_the_delta_is_the_mean_of_b_minus_a() -> None:
    ci = paired_bootstrap_ci([0.2, 0.4, 0.6], [0.3, 0.7, 0.6], resamples=200)
    assert ci.delta == pytest.approx((0.1 + 0.3 + 0.0) / 3)


def test_the_sign_follows_b_against_a() -> None:
    better = paired_bootstrap_ci([0.1, 0.1], [0.5, 0.9], resamples=100)
    worse = paired_bootstrap_ci([0.5, 0.9], [0.1, 0.1], resamples=100)
    assert better.delta > 0 > worse.delta
    assert better.delta == pytest.approx(-worse.delta)


def test_a_constant_difference_has_a_degenerate_interval() -> None:
    ci = paired_bootstrap_ci([0.0] * 20, [0.25] * 20, resamples=300)
    assert (ci.low, ci.delta, ci.high) == pytest.approx((0.25, 0.25, 0.25))


def test_one_question_collapses_the_interval() -> None:
    ci = paired_bootstrap_ci([0.3], [0.5], resamples=100)
    assert ci.low == ci.high == pytest.approx(0.2)


def test_identical_systems_have_a_zero_delta_and_a_zero_interval() -> None:
    ci = paired_bootstrap_ci([0.1, 0.9, 0.5], [0.1, 0.9, 0.5], resamples=100)
    assert (ci.low, ci.delta, ci.high) == (0.0, 0.0, 0.0)


def test_the_same_seed_gives_the_same_interval_and_another_seed_another() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.random(40), rng.random(40)
    first = paired_bootstrap_ci(a, b, resamples=1000, seed=7)
    assert paired_bootstrap_ci(a, b, resamples=1000, seed=7) == first
    other = paired_bootstrap_ci(a, b, resamples=1000, seed=8)
    assert other.delta == first.delta  # the real data does not change
    assert (other.low, other.high) != (first.low, first.high)


def test_a_clear_gain_excludes_zero_and_pure_noise_does_not() -> None:
    rng = np.random.default_rng(3)
    a = rng.random(200)
    gain = paired_bootstrap_ci(a, a + 0.1 + rng.normal(0, 0.02, 200), resamples=2000)
    assert 0 < gain.low < gain.delta < gain.high
    noise = paired_bootstrap_ci(a, a + rng.normal(0, 0.1, 200), resamples=2000)
    assert noise.low < 0 < noise.high


def test_the_interval_narrows_as_the_confidence_drops() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.random(60), rng.random(60)
    wide = paired_bootstrap_ci(a, b, resamples=2000, confidence=0.99, seed=1)
    narrow = paired_bootstrap_ci(a, b, resamples=2000, confidence=0.80, seed=1)
    assert wide.low < narrow.low < narrow.high < wide.high


def test_the_default_is_ten_thousand_resamples_and_ninety_five_percent() -> None:
    ci = paired_bootstrap_ci([0.0, 1.0], [1.0, 0.0])
    assert (ci.resamples, ci.confidence) == (DEFAULT_RESAMPLES, 0.95) == (10_000, 0.95)


def test_a_large_number_of_questions_is_resampled_in_batches_without_changing_the_result() -> None:
    rng = np.random.default_rng(9)
    a, b = rng.random(300), rng.random(300)
    ci = paired_bootstrap_ci(a, b, resamples=1234, seed=2)  # not a multiple of the batch size
    assert ci.low < ci.delta < ci.high
    assert ci.resamples == 1234


def test_the_interval_covers_the_true_mean_difference_about_as_often_as_it_claims() -> None:
    # A simulation: 300 datasets of 40 questions whose differences truly average 0.05. A 95%
    # interval should contain 0.05 in about 95% of them; the bound is loose on purpose (the
    # percentile bootstrap is a little too narrow for small n) and the seeds are fixed, so this
    # does not flake.
    rng = np.random.default_rng(2024)
    covered = 0
    for trial in range(300):
        diffs = rng.normal(0.05, 0.2, 40)
        ci = paired_bootstrap_ci(np.zeros(40), diffs, resamples=400, seed=trial)
        covered += ci.low <= 0.05 <= ci.high
    assert covered / 300 >= 0.88


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ([1.0], [1.0, 2.0], "same length"),
        ([], [], "no questions"),
        ([1.0, float("nan")], [1.0, 2.0], "finite"),
        ([1.0], [float("inf")], "finite"),
    ],
)
def test_bad_scores_are_refused(a: list[float], b: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_ci(a, b, resamples=10)
    with pytest.raises(ValueError, match=message):
        paired_permutation_pvalue(a, b, resamples=10)


def test_bad_options_are_refused() -> None:
    with pytest.raises(ValueError, match="resamples must be at least 1"):
        paired_bootstrap_ci([1.0], [2.0], resamples=0)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            paired_bootstrap_ci([1.0], [2.0], confidence=bad)
    with pytest.raises(ValueError, match="resamples must be at least 1"):
        paired_permutation_pvalue([1.0], [2.0], resamples=0)


@settings(max_examples=40, deadline=None)
@given(
    diffs=st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=1, max_size=25)
)
def test_the_interval_is_ordered_and_stays_inside_the_range_of_the_data(diffs: list[float]) -> None:
    ci = paired_bootstrap_ci([0.0] * len(diffs), diffs, resamples=200, seed=1)
    assert ci.low <= ci.high
    assert (
        min(diffs) - 1e-9 <= ci.low and ci.high <= max(diffs) + 1e-9
    )  # a mean cannot leave the range


# --- the permutation test, exactly -----------------------------------------------------------


def test_three_equal_gains_have_an_exact_p_of_one_quarter() -> None:
    # differences [1, 1, 1]; the 8 sign patterns give |sum| = 3 for (+++) and (---), else 1:
    # 2 of 8 patterns are at least as extreme as the real one
    assert paired_permutation_pvalue([0, 0, 0], [1, 1, 1]) == pytest.approx(2 / 8)


def test_two_equal_gains_have_an_exact_p_of_one_half() -> None:
    assert paired_permutation_pvalue([0, 0], [1, 1]) == pytest.approx(2 / 4)


def test_unequal_differences_exact_p_by_hand() -> None:
    # differences [2, 1]: the sums are +3, +1, -1, -3; |real sum| = 3; two patterns reach it
    assert paired_permutation_pvalue([0, 0], [2, 1]) == pytest.approx(2 / 4)
    # differences [3, 1, 1]: sums |+-3+-1+-1|: 5, 3, 3, 1 (and mirrors); real 5: 2 of 8
    assert paired_permutation_pvalue([0, 0, 0], [3, 1, 1]) == pytest.approx(2 / 8)


def test_a_mixed_sign_dataset_that_looks_like_chance_has_p_one() -> None:
    assert (
        paired_permutation_pvalue([0, 0], [1, -1]) == 1.0
    )  # the real sum is 0: every pattern ties


def test_no_difference_at_all_is_p_one() -> None:
    assert paired_permutation_pvalue([0.3, 0.7], [0.3, 0.7]) == 1.0


def test_the_sign_of_the_gain_does_not_matter_for_a_two_sided_test() -> None:
    a, b = [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.9]
    assert paired_permutation_pvalue(a, b) == paired_permutation_pvalue(b, a)


def test_a_p_value_is_a_probability_and_larger_effects_give_smaller_p() -> None:
    small = paired_permutation_pvalue(
        [0.0] * 10, [0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1]
    )
    large = paired_permutation_pvalue([0.0] * 10, [0.5] * 10)
    assert 0 < large < small <= 1


# --- the permutation test, by sampling -------------------------------------------------------


def test_with_many_questions_the_p_value_is_estimated_and_never_zero() -> None:
    rng = np.random.default_rng(4)
    a = rng.random(100)
    p = paired_permutation_pvalue(a, a + 0.3, resamples=999, seed=1)
    assert p == pytest.approx(1 / 1000)  # nothing in 999 flips beat it: (0 + 1) / (999 + 1)


def test_with_many_questions_pure_noise_gives_a_large_p() -> None:
    rng = np.random.default_rng(6)
    a = rng.random(80)
    p = paired_permutation_pvalue(a, a + rng.normal(0, 0.1, 80), resamples=2000, seed=1)
    assert p > 0.05


def test_the_sampled_p_is_reproducible_by_seed() -> None:
    rng = np.random.default_rng(8)
    a, b = rng.random(60), rng.random(60)
    assert paired_permutation_pvalue(a, b, resamples=500, seed=3) == paired_permutation_pvalue(
        a, b, resamples=500, seed=3
    )


def test_the_sampled_p_agrees_with_the_exact_p_on_a_small_case() -> None:
    a, b = [0.0] * 10, [0.3, 0.1, -0.1, 0.2, 0.4, 0.0, 0.1, 0.3, -0.2, 0.2]
    exact = paired_permutation_pvalue(a, b, resamples=10_000)  # 2**10 <= 10000: enumerated
    sampled = paired_permutation_pvalue(a, b, resamples=1023, seed=5)  # 2**10 > 1023: sampled
    assert sampled == pytest.approx(exact, abs=0.06)


# --- Holm ------------------------------------------------------------------------------------


def test_holm_worked_out_by_hand() -> None:
    # p = [0.01, 0.04, 0.03, 0.005], m = 4. Sorted: 0.005, 0.01, 0.03, 0.04.
    #   0.005 x 4 = 0.02;  0.01 x 3 = 0.03;  0.03 x 2 = 0.06;  0.04 x 1 = 0.04 -> raised to 0.06
    # back in the original order: [0.03, 0.06, 0.06, 0.02]
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_a_single_p_value_is_unchanged() -> None:
    assert holm_adjust([0.04]) == [0.04]
    assert holm_adjust([]) == []


def test_an_adjusted_p_is_capped_at_one() -> None:
    assert holm_adjust([0.6, 0.7, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_ties_are_adjusted_alike() -> None:
    adjusted = holm_adjust([0.02, 0.02, 0.5])
    assert adjusted[0] == adjusted[1] == pytest.approx(0.06)  # 0.02 x 3
    assert adjusted[2] == pytest.approx(0.5)


def test_p_values_must_be_probabilities() -> None:
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="between 0 and 1"):
            holm_adjust([0.5, bad])


_P = st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=12)


@given(p=_P)
def test_adjusted_p_values_are_never_smaller_never_above_one_and_keep_the_order(
    p: list[float],
) -> None:
    adjusted = holm_adjust(p)
    assert all(raw <= adj <= 1.0 for raw, adj in zip(p, adjusted, strict=True))
    # a smaller raw p never gets a larger adjusted p
    pairs = sorted(zip(p, adjusted, strict=True))
    assert [adj for _, adj in pairs] == sorted(adj for _, adj in pairs)


@given(p=_P)
def test_holm_is_never_more_conservative_than_bonferroni(p: list[float]) -> None:
    m = len(p)
    assert all(adj <= min(1.0, raw * m) + 1e-12 for raw, adj in zip(p, holm_adjust(p), strict=True))


# --- the whole comparison --------------------------------------------------------------------


def test_compare_paired_puts_the_pieces_together() -> None:
    a, b = [0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.6, 1.0]
    result = compare_paired(a, b, resamples=500, seed=1)
    assert result.n == 4
    assert result.mean_a == pytest.approx(0.5)
    assert result.mean_b == pytest.approx(0.625)
    assert result.delta == pytest.approx(0.125)
    assert result.ci_low <= result.delta <= result.ci_high
    assert result.p_value == paired_permutation_pvalue(a, b, resamples=500, seed=1)
    assert result.confidence == 0.95


def test_the_line_gives_delta_then_the_interval_then_p_and_never_the_word_significant() -> None:
    result = compare_paired([0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.6, 1.0], resamples=500, seed=1)
    line = format_comparison("R6 vs R5, nDCG@10", result, adjusted_p=0.0312)
    assert line.startswith("R6 vs R5, nDCG@10: n=4, A=0.5000, B=0.6250, delta=+0.1250 [")
    assert "(95% CI), p=" in line
    assert line.endswith("Holm-adjusted p=0.0312")
    assert line.index("delta=") < line.index("CI") < line.index("p=")
    assert "significant" not in line.lower()
    assert "Holm" not in format_comparison("x", result)
