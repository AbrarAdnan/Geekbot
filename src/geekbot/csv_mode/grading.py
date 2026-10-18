"""Grading one CSV answer against a gold answer (step C3, PLAN.md §9).

This is a port of the grader in Gigalogy's ``evaluation/evaluator/runner.py`` and
``evaluation/gold/spot_check.py`` (the project the ``csv_analyst`` package comes from). It is
copied in behaviour, not imported, because that code lives in a git-ignored reference copy that
may move, and a grader that quietly changed would make "the copy reproduces the original" mean
nothing. Gigalogy's own grading tests are ported alongside (``tests/unit/test_csv_grading.py``).

The grading idea (Gigalogy CLAUDE.md §14): judge the *result*, not how it was computed.

- A question that should be refused is right only if the package did not answer it.
- An answerable question is right if its result matches the gold value: row order does not
  matter, column names do not matter (only the values), and floats may differ by a tolerance.
- A one-row table counts as a scalar answer (same value, different container), but a
  many-row table is never squeezed into a scalar.

Lives in ``csv_mode`` because it reads the package's ``RunRecord``; callers outside this package
get a plain ``(correct, note)`` pair.
"""

import datetime
import json
from collections.abc import Mapping, Sequence
from typing import Any

from csv_analyst import Status
from csv_analyst.contracts import ResultKind
from geekbot.csv_mode.adapter import CsvAnswer

# Floats that differ by less than this are equal even when the gold tolerance is 0.
SCALAR_ABS_TOL = 1e-9


def _as_datetime(value: Any) -> datetime.datetime:
    """A date, a datetime (``pandas.Timestamp`` is one) or an ISO date text, as a datetime."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)
    return datetime.datetime.fromisoformat(str(value))


def _coerce_container_mismatch(a: Any, b: Any) -> tuple[Any, Any]:
    """Make a value that is right but arrived in another container comparable.

    Exactly two pairs, as in Gigalogy (its Phase 6D.3): a number against numeric text (gold
    ``1918``, computed ``'1918'``), and a date against an ISO date text. Anything else is left
    alone, so two genuinely different answers are never graded as the same.
    """
    a_num, b_num = isinstance(a, int | float), isinstance(b, int | float)
    if a_num and isinstance(b, str):
        try:
            return a, float(b)
        except ValueError:
            return a, b
    if b_num and isinstance(a, str):
        try:
            return float(a), b
        except ValueError:
            return a, b

    a_date, b_date = isinstance(a, datetime.date), isinstance(b, datetime.date)
    if (a_date and isinstance(b, str)) or (b_date and isinstance(a, str)):
        try:
            return _as_datetime(a), _as_datetime(b)
        except ValueError:
            return a, b
    return a, b


def scalar_close(a: Any, b: Any, tol: float) -> bool:
    a, b = _coerce_container_mismatch(a, b)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= max(SCALAR_ABS_TOL, tol)
    return bool(a == b)


def table_close(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], tol: float) -> bool:
    """Compare two tables by row values only.

    Same number of rows, same number of columns, and the same multiset of rows (order does not
    matter). Column *names* are ignored: production code cannot know gold's naming, and a
    differently labelled but value-correct column is still correct. (One difference from
    Gigalogy's version: rows of different lengths are never equal, instead of being compared
    up to the shorter one. It only matters for malformed rows.)
    """
    if len(a) != len(b):
        return False
    a_rows = [tuple(row.values()) for row in a]
    b_rows = [tuple(row.values()) for row in b]

    def sort_key(row: tuple[Any, ...]) -> str:
        return json.dumps(row, sort_keys=False, default=str)

    for row_a, row_b in zip(
        sorted(a_rows, key=sort_key), sorted(b_rows, key=sort_key), strict=True
    ):
        if len(row_a) != len(row_b):
            return False
        if not all(scalar_close(va, vb, tol) for va, vb in zip(row_a, row_b, strict=True)):
            return False
    return True


def _actual_value(answer: CsvAnswer, compare: str) -> Any:
    """The answer's result as a plain value: a scalar, or a list of dicts for a table."""
    result = answer.record.result
    if result is None:
        return None
    if compare == "scalar":
        if result.kind != ResultKind.SCALAR:
            return None
        return result.preview[0][0] if result.preview else None
    if not result.columns:
        # A dict-like or one-column result may carry no column names; it will simply fail to
        # match a gold table with named columns, which is a correct "wrong shape" grade.
        return [{"value": row[0] if len(row) == 1 else row} for row in result.preview]
    return [dict(zip(result.columns, row, strict=False)) for row in result.preview]


def grade(answer: CsvAnswer, gold_entry: Mapping[str, Any]) -> tuple[bool, str]:
    """Return ``(correct, note)`` for one answer against one gold entry."""
    record = answer.record
    if gold_entry["expected_abstain"]:
        if record.status != Status.ANSWERED:
            return True, "correctly abstained"
        return False, "HALLUCINATION: answered a question that should have been refused"

    if record.status != Status.ANSWERED:
        return False, f"false abstention: status={record.status.value} reason={record.reason!r}"

    compare = gold_entry["compare"]
    tol = gold_entry.get("tolerance") or 0
    actual = _actual_value(answer, compare)
    result = record.result

    if (
        compare == "scalar"
        and result is not None
        and result.kind in (ResultKind.SERIES, ResultKind.DATAFRAME)
        and len(result.preview) == 1
    ):
        # The model returned a natural one-row table instead of a bare scalar: same value,
        # different container. Penalising that would grade the shape of the code.
        row = result.preview[0]
        ok = any(scalar_close(gold_entry["value"], value, tol) for value in row)
        actual = row
    elif compare == "scalar":
        ok = scalar_close(gold_entry["value"], actual, tol)
    else:
        ok = actual is not None and table_close(gold_entry["value"], actual, tol)
    return ok, "matches gold" if ok else f"expected {gold_entry['value']!r}, got {actual!r}"
