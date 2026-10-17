"""Isolated child process that executes one generated pandas snippet
(PLAN.md Phase 3, CLAUDE.md Sec8/Sec9).

Invoked as a subprocess by execution/runner.py, never imported directly.
The DataFrame crosses the process boundary as a Parquet file, and the
snippet's `result` comes back the same way -- only data crosses, never a
live Python object.

Known limitation (documented, not hidden): the parent starts this process
with `-I` (isolated mode -- ignores env vars and user site-packages) but
not `-S`, because `-S` also skips the venv's own site-packages
initialization on Windows, which would break the `import pandas`/`pyarrow`
below before any untrusted code runs.

The AST guard (validation/ast_guard.py) runs upstream of this process and
blocks `import` statements, any `_`-prefixed identifier or attribute access,
and a denylist of known I/O attribute names. But it validates attribute
*names* in the parsed expression, not what object an attribute chain
resolves to at runtime. A chain that starts from a public, allowed name
(e.g. `pd` or `np`) and walks through a public pandas/numpy submodule can
still resolve to stdlib modules the guard never sees named in the source,
including `os` and `socket` -- confirmed by a breadth-first walk of `pd`/
`np` attributes that reaches dozens of non-pandas/numpy modules while
respecting the guard's own name-based rules. So the AST guard does not
block arbitrary-module reach from inside the snippet the way this docstring
used to claim.

What does still hold: the restricted `__builtins__` dict below has no
`__import__`, so a nested `exec`/`eval` reached via such a chain fails with
`ImportError: __import__ not found` -- but a module object reached via
attribute traversal is already imported and usable without calling
`import` again, so this protects less than it sounds like.

Net effect: this process boundary isolates the untrusted snippet from this
Python process's live objects (only Parquet data crosses), and gives it a
wall-clock timeout, but it is a prototype execution boundary, not a
security sandbox. Network egress and filesystem access from inside the
snippet are not prevented. Memory is not hard-capped either.
"""

from __future__ import annotations

import builtins
import datetime
import json
import math
import sys

import numpy as np
import pandas as pd

SAFE_BUILTIN_NAMES = (
    "abs", "min", "max", "sum", "len", "round", "sorted", "range",
    "enumerate", "zip", "list", "dict", "set", "tuple", "float", "int",
    "str", "bool", "isinstance", "any", "all",
)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}


def _json_safe_number(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _date_like_to_str(value):
    """ISO string for a date/time scalar (e.g. the `idxmax()` of a series
    grouped by `.dt.date`), or None if `value` isn't one. Date-only when there
    is no time part, so a day comes back as '2018-09-09', not
    '2018-09-09T00:00:00'. Before this, such a result failed every attempt
    with "unsupported result type: date" even when the code was correct."""
    if value is pd.NaT:
        return None
    if isinstance(value, (np.datetime64, datetime.datetime)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()
    if isinstance(value, (datetime.date, pd.Period)):
        return str(value)
    if isinstance(value, (np.timedelta64, datetime.timedelta)):
        td = pd.Timedelta(value)
        return None if pd.isna(td) else str(td)
    return None


def _is_date_like(value) -> bool:
    return value is pd.NaT or isinstance(
        value,
        (np.datetime64, np.timedelta64, datetime.date, datetime.timedelta, pd.Period),
    )


def _to_python_scalar(value):
    if _is_date_like(value):
        return _date_like_to_str(value)
    if isinstance(value, np.generic):
        value = value.item()
    return _json_safe_number(value)


def _is_missing(value) -> bool:
    return value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _cell_to_str(value):
    if _is_missing(value):
        return None
    if _is_date_like(value):
        return _date_like_to_str(value)
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Make object columns holding mixed value types safe for parquet.

    pyarrow infers an object column's type from its values: a date next to
    ints makes it a date column, and the ints are then read as days since
    1970 (a row picked with `t.loc[i]` -- date, count 16, pct 220 -- came
    back as '1970-01-17' and '1970-08-09'). Mixed numbers become one numeric
    column; any other mix becomes text, dates as ISO strings. Single-type
    columns are left alone."""
    frame = frame.copy()
    for col in frame.columns:
        if frame[col].dtype != object:
            continue
        present = [v for v in frame[col] if not _is_missing(v)]
        if len({type(v) for v in present}) <= 1:
            continue
        if all(_is_number(v) for v in present):
            frame[col] = pd.to_numeric(frame[col])
        else:
            frame[col] = frame[col].map(_cell_to_str)
    return frame


def _serialize(result, result_path) -> dict:
    if isinstance(result, pd.DataFrame):
        # A named index holds real labels (e.g. `groupby(...).unstack()`
        # leaves the group keys there, not in a column) -- the preview only
        # shows columns, so those labels would silently vanish from the
        # answer. An unnamed index is just row positions left over from a
        # filter and stays dropped.
        if any(name is not None for name in result.index.names):
            try:
                result = result.reset_index()
            except ValueError:
                pass  # the label is already a column too (set_index(drop=False))
        result.columns = [str(c) for c in result.columns]
        result = _parquet_safe(result)
        result.to_parquet(result_path)
        return {
            "ok": True,
            "kind": "dataframe",
            "shape": list(result.shape),
            "columns": [str(c) for c in result.columns],
        }

    if isinstance(result, pd.Series):
        # A bare 0..n-1 index (after `reset_index(drop=True)` or on a series
        # built from a list) carries no labels; keeping it put a meaningless
        # `index` column next to the values. Any other index is kept.
        plain = isinstance(result.index, pd.RangeIndex) and result.index.name is None and result.index.start == 0 and result.index.step == 1
        frame = result.to_frame() if plain else result.reset_index()
        frame.columns = [str(c) for c in frame.columns]
        frame = _parquet_safe(frame)
        frame.to_parquet(result_path)
        return {
            "ok": True,
            "kind": "series",
            "shape": [int(result.shape[0]), len(frame.columns)],
            "columns": list(frame.columns),
        }

    if isinstance(result, dict):
        safe = {str(k): _to_python_scalar(v) for k, v in result.items()}
        return {"ok": True, "kind": "dict", "shape": [len(safe)], "value": safe}

    if isinstance(result, (np.generic, int, float, str, bool)) or result is None or _is_date_like(result):
        return {"ok": True, "kind": "scalar", "shape": [1], "value": _to_python_scalar(result)}

    return {"ok": False, "error": f"unsupported result type: {type(result).__name__}"}


def main() -> None:
    df_path, code_path, result_path, meta_path = sys.argv[1:5]
    meta: dict

    try:
        df = pd.read_parquet(df_path)
        code = open(code_path, "r", encoding="utf-8").read()

        namespace = {"__builtins__": SAFE_BUILTINS, "df": df, "pd": pd, "np": np}
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - AST-gated upstream

        if "result" not in namespace:
            raise ValueError("code did not assign a variable named 'result'")

        meta = _serialize(namespace["result"], result_path)
    except Exception as exc:  # noqa: BLE001 - must always emit a metadata file, never crash silently
        meta = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, default=str)


if __name__ == "__main__":
    main()
