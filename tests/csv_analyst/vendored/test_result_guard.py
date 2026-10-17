"""Result payload construction: preview cap, truncation flag, NaN -> null,
and the "empty result is legitimate" rule (PLAN.md Phase 3)."""

from __future__ import annotations

import pandas as pd

from csv_analyst.contracts import ResultKind
from csv_analyst.execution.runner import ExecutionResult
from csv_analyst.validation.result_guard import PREVIEW_ROW_CAP, build_result_payload


def test_scalar_payload():
    payload = build_result_payload(ExecutionResult(ok=True, kind=ResultKind.SCALAR, value=125.0))
    assert payload.shape == [1]
    assert payload.preview == [[125.0]]
    assert not payload.truncated


def test_dataframe_payload_under_cap():
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [125.0, 250.0]})
    payload = build_result_payload(ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=df))
    assert payload.columns == ["region", "revenue"]
    assert payload.shape == [2, 2]
    assert payload.preview == [["North", 125.0], ["South", 250.0]]
    assert not payload.truncated


def test_dataframe_payload_truncated_over_cap():
    df = pd.DataFrame({"n": list(range(PREVIEW_ROW_CAP + 10))})
    payload = build_result_payload(ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=df))
    assert payload.truncated
    assert len(payload.preview) == PREVIEW_ROW_CAP
    assert payload.shape == [PREVIEW_ROW_CAP + 10, 1]


def test_empty_dataframe_is_legitimate_not_an_error():
    df = pd.DataFrame({"region": [], "revenue": []})
    payload = build_result_payload(ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=df))
    assert payload.shape == [0, 2]
    assert payload.preview == []


def test_nan_in_dataframe_becomes_none():
    df = pd.DataFrame({"revenue": [1.0, float("nan")]})
    payload = build_result_payload(ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=df))
    assert payload.preview[1] == [None]


def test_dict_payload():
    payload = build_result_payload(
        ExecutionResult(ok=True, kind=ResultKind.DICT, value={"total": 500.0, "count": 4})
    )
    assert payload.shape == [2]
    assert payload.columns == ["total", "count"]
    assert payload.preview == [["total", 500.0], ["count", 4]]
