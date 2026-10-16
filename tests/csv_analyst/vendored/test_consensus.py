"""Cross-sample consensus comparator: unanimous / majority / none, per the
comparison semantics in PLAN.md Phase 3 (numeric tolerance, order-insensitive
tables, empty vs non-empty is disagreement)."""

from __future__ import annotations

import pandas as pd

from csv_analyst.contracts import ConsensusVerdict, ResultKind
from csv_analyst.execution.runner import ExecutionResult
from csv_analyst.validation.consensus import evaluate_consensus, results_equal


def _scalar(v):
    return ExecutionResult(ok=True, kind=ResultKind.SCALAR, value=v)


def _table(rows, cols=("region", "revenue")):
    return ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=pd.DataFrame(rows, columns=list(cols)))


def test_unanimous_scalars():
    outcome = evaluate_consensus([_scalar(125.0), _scalar(125.0), _scalar(125.0)])
    assert outcome.verdict == ConsensusVerdict.UNANIMOUS
    assert outcome.chosen_index == 0


def test_scalar_within_tolerance_counts_as_equal():
    outcome = evaluate_consensus([_scalar(125.0), _scalar(125.0000001), _scalar(125.0)])
    assert outcome.verdict == ConsensusVerdict.UNANIMOUS


def test_majority_of_three():
    outcome = evaluate_consensus([_scalar(125.0), _scalar(125.0), _scalar(999.0)])
    assert outcome.verdict == ConsensusVerdict.MAJORITY
    assert len(outcome.agreeing_indices) == 2


def test_none_when_all_disagree():
    outcome = evaluate_consensus([_scalar(1.0), _scalar(2.0), _scalar(3.0)])
    assert outcome.verdict == ConsensusVerdict.NONE
    assert outcome.chosen_index is None


def test_single_sample_is_unanimous():
    outcome = evaluate_consensus([_scalar(125.0)])
    assert outcome.verdict == ConsensusVerdict.UNANIMOUS


def test_no_samples_is_none():
    outcome = evaluate_consensus([])
    assert outcome.verdict == ConsensusVerdict.NONE
    assert outcome.chosen_index is None


def test_tables_equal_ignoring_row_order():
    a = _table([["North", 100.0], ["South", 200.0]])
    b = _table([["South", 200.0], ["North", 100.0]])
    assert results_equal(a, b)


def test_tables_different_values_disagree():
    a = _table([["North", 100.0]])
    b = _table([["North", 999.0]])
    assert not results_equal(a, b)


def test_empty_vs_nonempty_disagree():
    empty = _table([])
    nonempty = _table([["North", 100.0]])
    assert not results_equal(empty, nonempty)


def test_different_kinds_disagree():
    scalar = _scalar(1.0)
    table = _table([["North", 100.0]])
    assert not results_equal(scalar, table)


def test_dicts_compared_by_equality():
    a = ExecutionResult(ok=True, kind=ResultKind.DICT, value={"total": 500.0})
    b = ExecutionResult(ok=True, kind=ResultKind.DICT, value={"total": 500.0})
    c = ExecutionResult(ok=True, kind=ResultKind.DICT, value={"total": 999.0})
    assert results_equal(a, b)
    assert not results_equal(a, c)


# --- PLAN.md Phase 6A.1: abstention votes join the consensus vote ----------


def test_two_abstain_one_result_abstains():
    outcome = evaluate_consensus([_scalar(125.0)], abstain_votes=2)
    assert outcome.abstain_won
    assert outcome.chosen_index is None
    assert outcome.verdict == ConsensusVerdict.MAJORITY


def test_two_agreeing_results_one_abstain_still_answers():
    outcome = evaluate_consensus([_scalar(125.0), _scalar(125.0)], abstain_votes=1)
    assert not outcome.abstain_won
    assert outcome.verdict == ConsensusVerdict.MAJORITY
    assert outcome.chosen_index == 0
    assert outcome.agreeing_indices == [0, 1]


def test_one_abstain_one_result_is_none_not_abstained():
    outcome = evaluate_consensus([_scalar(125.0)], abstain_votes=1)
    assert outcome.verdict == ConsensusVerdict.NONE
    assert not outcome.abstain_won
    assert outcome.chosen_index is None


def test_all_abstain_is_unanimous_abstention():
    outcome = evaluate_consensus([], abstain_votes=3)
    assert outcome.abstain_won
    assert outcome.verdict == ConsensusVerdict.UNANIMOUS
