"""Cross-sample result comparison (PLAN.md Phase 3 / locked decision:
N-sample consensus, PLAN.md Sec2). Three independent generations either
agree -- reasonable evidence the answer is correct -- or disagree, which is
a real signal to retry or abstain, per CLAUDE.md's core reliability
principle: the LLM proposes a computation, execution + comparison decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from csv_analyst.contracts import ConsensusVerdict, ResultKind
from csv_analyst.execution.runner import ExecutionResult

SCALAR_ABS_TOL = 1e-9
SCALAR_REL_TOL = 1e-6


def _scalar_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    a_is_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_is_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_is_num and b_is_num:
        return abs(a - b) <= max(SCALAR_ABS_TOL, SCALAR_REL_TOL * abs(b))
    return a == b


def _table_equal(da: pd.DataFrame, db: pd.DataFrame) -> bool:
    if da.empty != db.empty:
        return False
    if da.empty and db.empty:
        return True
    if set(str(c) for c in da.columns) != set(str(c) for c in db.columns):
        return False

    cols = sorted(str(c) for c in da.columns)
    da = da.copy()
    db = db.copy()
    da.columns = [str(c) for c in da.columns]
    db.columns = [str(c) for c in db.columns]

    try:
        da_sorted = da[cols].sort_values(by=cols).reset_index(drop=True)
        db_sorted = db[cols].sort_values(by=cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(da_sorted, db_sorted, rtol=SCALAR_REL_TOL, check_dtype=False)
        return True
    except (AssertionError, TypeError, ValueError):
        return False


def results_equal(a: ExecutionResult, b: ExecutionResult) -> bool:
    if a.kind != b.kind:
        return False
    if a.kind == ResultKind.SCALAR:
        return _scalar_equal(a.value, b.value)
    if a.kind == ResultKind.DICT:
        return a.value == b.value
    return _table_equal(a.value, b.value)  # SERIES / DATAFRAME


@dataclass
class ConsensusOutcome:
    verdict: ConsensusVerdict
    chosen_index: Optional[int]  # index into the `results` list passed to evaluate_consensus
    agreeing_indices: list[int] = field(default_factory=list)
    abstain_votes: int = 0
    abstain_won: bool = False


def evaluate_consensus(results: list[ExecutionResult], abstain_votes: int = 0) -> ConsensusOutcome:
    """`results` must contain only successfully-executed samples -- callers
    drop failed samples before calling this.

    `abstain_votes` (PLAN.md Phase 6A.1) counts samples the model tagged
    non-answerable. Those votes join the same largest-group-wins count as
    the executed results -- an abstention is a vote, not a fallback used only
    when nothing executed -- so a question two of three samples correctly
    refuse to answer must not be decided by the one outlier that ran.
    """
    n = len(results) + abstain_votes
    if n == 0:
        return ConsensusOutcome(verdict=ConsensusVerdict.NONE, chosen_index=None, abstain_votes=abstain_votes)

    groups: list[list[int]] = []
    for i, r in enumerate(results):
        for group in groups:
            if results_equal(results[group[0]], r):
                group.append(i)
                break
        else:
            groups.append([i])

    # (size, indices) for each real result group, plus a synthetic abstain
    # "group" (indices=None marks it as not indexing into `results`).
    sized_groups: list[tuple[int, Optional[list[int]]]] = [(len(g), g) for g in groups]
    if abstain_votes > 0:
        sized_groups.append((abstain_votes, None))

    sized_groups.sort(key=lambda t: t[0], reverse=True)
    largest_size, largest_group = sized_groups[0]

    if largest_size == n:
        verdict = ConsensusVerdict.UNANIMOUS
    elif largest_size > n / 2:
        verdict = ConsensusVerdict.MAJORITY
    else:
        verdict = ConsensusVerdict.NONE

    if verdict == ConsensusVerdict.NONE:
        return ConsensusOutcome(verdict=verdict, chosen_index=None, agreeing_indices=[], abstain_votes=abstain_votes)

    if largest_group is None:
        return ConsensusOutcome(
            verdict=verdict, chosen_index=None, agreeing_indices=[],
            abstain_votes=abstain_votes, abstain_won=True,
        )
    return ConsensusOutcome(
        verdict=verdict, chosen_index=largest_group[0], agreeing_indices=largest_group,
        abstain_votes=abstain_votes,
    )
