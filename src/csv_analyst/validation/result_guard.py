"""Result-shape validation and preview construction (PLAN.md Phase 3,
CLAUDE.md Sec9). Turns a raw ExecutionResult into the frozen ResultPayload
contract. An empty result is a legitimate first-class outcome, never an
error and never backfilled with a number.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from csv_analyst.contracts import ResultKind, ResultPayload
from csv_analyst.execution.runner import ExecutionResult

PREVIEW_ROW_CAP = 100


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def build_result_payload(exec_result: ExecutionResult) -> ResultPayload:
    """`exec_result.ok` must be True -- callers only reach here after a
    successful sandbox execution."""
    kind = exec_result.kind
    value = exec_result.value

    if kind in (ResultKind.SERIES, ResultKind.DATAFRAME):
        df: pd.DataFrame = value
        shape = [int(df.shape[0]), int(df.shape[1])]
        columns = [str(c) for c in df.columns]
        preview_df = df.head(PREVIEW_ROW_CAP)
        preview = [
            [_json_safe(v) for v in row]
            for row in preview_df.itertuples(index=False, name=None)
        ]
        truncated = df.shape[0] > PREVIEW_ROW_CAP

    elif kind == ResultKind.DICT:
        d = value or {}
        shape = [len(d)]
        columns = [str(k) for k in d]
        items = list(d.items())[:PREVIEW_ROW_CAP]
        preview = [[str(k), _json_safe(v)] for k, v in items]
        truncated = len(d) > PREVIEW_ROW_CAP

    else:  # scalar
        shape = [1]
        columns = []
        preview = [[_json_safe(value)]]
        truncated = False

    return ResultPayload(kind=kind, shape=shape, columns=columns, preview=preview, truncated=truncated)
