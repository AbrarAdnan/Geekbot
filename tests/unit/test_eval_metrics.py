"""Tests for step S2.5: retrieval metrics (nDCG@k is this step's ✍️ piece, written by Claude, D-047).

Every expected number in the first groups is worked out by hand in the comment next to it; the
hypothesis properties at the end check what must hold for any ranking.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.evals.metrics import (
    aggregate_to_documents,
    evaluate_query,
    mean_defined,
    ndcg_at_k,
    per_query,
    recall_at_k,
    reciprocal_rank,
)

# --- nDCG, by hand ---------------------------------------------------------------------------


def test_a_perfect_ranking_scores_one() -> None:
    qrels = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "c", "x", "y"], qrels, 10) == pytest.approx(1.0)


def test_graded_judgements_worked_out_by_hand() -> None:
    # qrels a=3 b=2 c=0 d=1; ranking c a d b; k=4
    #   DCG  = 0/log2(2) + 3/log2(3) + 1/log2(4) + 2/log2(5)
    #        = 0 + 1.89279 + 0.5 + 0.86135 = 3.25414
    #   IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5 = 4.76186
    #   nDCG = 3.25414 / 4.76186 = 0.68338
    qrels = {"a": 3, "b": 2, "c": 0, "d": 1}
    assert ndcg_at_k(["c", "a", "d", "b"], qrels, 4) == pytest.approx(0.68338, abs=1e-4)


def test_binary_judgements_worked_out_by_hand() -> None:
    # relevant {a, b}; ranking x a y b; k=4
    #   DCG  = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
    #   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    #   nDCG = 0.65091
    assert ndcg_at_k(["x", "a", "y", "b"], {"a": 1, "b": 1}, 4) == pytest.approx(0.65091, abs=1e-4)


def test_the_discount_halves_at_rank_three_and_is_one_at_rank_one() -> None:
    # one relevant document: at rank 1 the score is 1, at rank 3 it is 1/log2(4) = 0.5
    qrels = {"a": 1}
    assert ndcg_at_k(["a", "x", "y"], qrels, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "a"], qrels, 3) == pytest.approx(0.5)
    assert ndcg_at_k(["x", "a", "y"], qrels, 3) == pytest.approx(1 / math.log2(3))


def test_only_the_top_k_counts() -> None:
    qrels = {"a": 1}
    assert ndcg_at_k(["x", "y", "a"], qrels, 2) == 0.0  # a is at rank 3, outside the top 2
    assert ndcg_at_k(["x", "y", "a"], qrels, 3) > 0


def test_the_ideal_is_limited_to_k_too() -> None:
    # four relevant documents but k = 2: the best possible top 2 is two of them, so two at the top
    # is perfect
    qrels = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert ndcg_at_k(["c", "a", "x"], qrels, 2) == pytest.approx(1.0)


def test_unjudged_and_irrelevant_documents_are_worth_nothing() -> None:
    assert ndcg_at_k(["u1", "u2"], {"a": 1}, 5) == 0.0
    assert ndcg_at_k(["z", "a"], {"a": 1, "z": 0}, 5) == pytest.approx(1 / math.log2(3))


def test_a_ranking_shorter_than_k_is_fine() -> None:
    assert ndcg_at_k(["a"], {"a": 1, "b": 1}, 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([], {"a": 1}, 10) == 0.0


def test_a_document_listed_twice_counts_once() -> None:
    qrels = {"a": 1, "b": 1}
    assert (
        ndcg_at_k(["a", "a", "b"], qrels, 3)
        == ndcg_at_k(["a", "b"], qrels, 3)
        == pytest.approx(1.0)
    )


def test_no_relevant_document_means_undefined_not_zero() -> None:
    assert ndcg_at_k(["a", "b"], {}, 5) is None
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 5) is None


def test_k_below_one_is_refused() -> None:
    for metric in (ndcg_at_k, recall_at_k, reciprocal_rank):
        with pytest.raises(ValueError, match="k must be at least 1"):
            metric(["a"], {"a": 1}, 0)


# --- recall and MRR --------------------------------------------------------------------------


def test_recall_is_the_fraction_of_the_relevant_documents_found() -> None:
    qrels = {"a": 1, "b": 1, "c": 1, "x": 0}
    ranking = ["a", "x", "b", "y", "c"]
    assert recall_at_k(ranking, qrels, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, qrels, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, qrels, 5) == pytest.approx(1.0)


def test_recall_counts_a_graded_document_as_relevant_when_its_grade_is_positive() -> None:
    assert recall_at_k(["b"], {"a": 0, "b": 2}, 1) == 1.0


def test_recall_ignores_repeats_so_it_cannot_exceed_one() -> None:
    assert recall_at_k(["a", "a", "a"], {"a": 1, "b": 1}, 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "a", "b"], {"a": 1, "b": 1}, 2) == pytest.approx(1.0)


def test_recall_with_nothing_relevant_is_undefined() -> None:
    assert recall_at_k(["a"], {}, 5) is None


def test_reciprocal_rank_is_one_over_the_rank_of_the_first_relevant() -> None:
    assert reciprocal_rank(["a", "x"], {"a": 1}, 5) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "a", "b"], {"a": 1, "b": 1}, 5) == 0.5  # only the first counts


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_in_the_top_k() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, 2) == 0.0
    assert reciprocal_rank([], {"a": 1}, 5) == 0.0
    assert reciprocal_rank(["a"], {}, 5) is None


# --- chunks to documents ---------------------------------------------------------------------


def test_a_chunk_ranking_becomes_a_document_ranking_at_the_best_chunks_position() -> None:
    assert aggregate_to_documents(["d1", "d1", "d2", "d1", "d3"]) == ["d1", "d2", "d3"]
    assert aggregate_to_documents([]) == []


def test_repeated_chunks_of_one_document_do_not_crowd_the_others_out_of_the_top_k() -> None:
    chunk_docs = ["d1", "d1", "d1", "d2"]
    assert recall_at_k(chunk_docs, {"d1": 1, "d2": 1}, 2) == 1.0  # two documents in the top 2


# --- many questions --------------------------------------------------------------------------


def test_per_query_scores_every_judged_question_and_leaves_out_the_undefined_ones() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {}}  # q3 is unanswerable
    rankings = {"q1": ["a"], "q2": ["x", "b"]}
    scores = per_query(recall_at_k, rankings, qrels, 1)
    assert scores == {"q1": 1.0, "q2": 0.0}  # q3 undefined: left out, not scored 0


def test_a_question_the_system_did_not_answer_counts_as_a_failure() -> None:
    scores = per_query(ndcg_at_k, {}, {"q1": {"a": 1}}, 5)
    assert scores == {"q1": 0.0}


def test_mean_defined_reports_how_many_were_skipped() -> None:
    result = mean_defined({"q1": 1.0, "q2": 0.0, "q3": 0.5}, total=5)
    assert result.mean == pytest.approx(0.5)
    assert (result.count, result.skipped) == (3, 2)
    assert mean_defined({"q": 1.0}).skipped == 0


def test_the_mean_of_nothing_is_an_error_not_a_number() -> None:
    with pytest.raises(ValueError, match="no question has a defined score"):
        mean_defined({})


def test_evaluate_query_gives_every_metric_at_every_cutoff() -> None:
    scores = evaluate_query(["x", "a"], {"a": 1}, ks=(1, 2))
    assert scores.recall == {1: 0.0, 2: 1.0}
    assert scores.mrr == {1: 0.0, 2: 0.5}
    assert scores.ndcg[1] == 0.0
    assert scores.ndcg[2] == pytest.approx(1 / math.log2(3))
    assert evaluate_query(["a"], {}).recall == {5: None, 10: None}


# --- properties ------------------------------------------------------------------------------

_DOCS = st.sampled_from("abcdefgh")
_RANKINGS = st.lists(_DOCS, max_size=12)
_QRELS = st.dictionaries(_DOCS, st.integers(min_value=0, max_value=3), min_size=1, max_size=8)
_KS = st.integers(min_value=1, max_value=12)


@given(ranking=_RANKINGS, qrels=_QRELS, k=_KS)
def test_every_metric_is_between_zero_and_one_or_undefined(
    ranking: list[str], qrels: dict[str, int], k: int
) -> None:
    for metric in (ndcg_at_k, recall_at_k, reciprocal_rank):
        value = metric(ranking, qrels, k)
        assert value is None or 0.0 <= value <= 1.0 + 1e-12


@given(ranking=_RANKINGS, qrels=_QRELS, k=_KS)
def test_all_three_are_undefined_together_and_exactly_when_nothing_is_relevant(
    ranking: list[str], qrels: dict[str, int], k: int
) -> None:
    none_relevant = not any(grade > 0 for grade in qrels.values())
    for metric in (ndcg_at_k, recall_at_k, reciprocal_rank):
        assert (metric(ranking, qrels, k) is None) == none_relevant


@given(qrels=_QRELS, k=_KS)
def test_ranking_by_grade_is_perfect(qrels: dict[str, int], k: int) -> None:
    ideal = [doc for doc, _ in sorted(qrels.items(), key=lambda item: -item[1])]
    value = ndcg_at_k(ideal, qrels, k)
    assert value is None or value == pytest.approx(1.0)


@given(ranking=_RANKINGS, qrels=_QRELS, k=_KS)
def test_recall_never_decreases_as_k_grows(
    ranking: list[str], qrels: dict[str, int], k: int
) -> None:
    low, high = recall_at_k(ranking, qrels, k), recall_at_k(ranking, qrels, k + 1)
    assert (low is None) == (high is None)
    if low is not None and high is not None:
        assert low <= high + 1e-12


@given(ranking=_RANKINGS, qrels=_QRELS, k=_KS)
def test_ndcg_is_unchanged_by_repeating_a_document(
    ranking: list[str], qrels: dict[str, int], k: int
) -> None:
    doubled = [doc for doc in ranking for _ in range(2)]
    assert ndcg_at_k(doubled, qrels, k) == ndcg_at_k(ranking, qrels, k)


@given(
    qrels=_QRELS,
    ranking=st.lists(_DOCS, min_size=2, max_size=10, unique=True),
    k=st.integers(min_value=2, max_value=10),
    position=st.integers(min_value=1, max_value=9),
)
def test_moving_a_more_relevant_document_above_a_less_relevant_one_never_lowers_ndcg(
    qrels: dict[str, int], ranking: list[str], k: int, position: int
) -> None:
    i = position % len(ranking)
    if i == 0:
        return
    above, below = ranking[i - 1], ranking[i]
    if qrels.get(below, 0) <= qrels.get(above, 0):
        return  # only the case where the lower document is the more relevant one
    swapped = list(ranking)
    swapped[i - 1], swapped[i] = below, above
    before, after = ndcg_at_k(ranking, qrels, k), ndcg_at_k(swapped, qrels, k)
    if before is not None and after is not None:
        assert after >= before - 1e-12
