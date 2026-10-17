"""Error-class to targeted retry hint (PLAN.md Phase 6A.5).

The generic retry feedback ("write different code that avoids this specific
error") never names the fix, so the model reproduces the same mistake. This
table is seeded from errors actually captured in evaluation/reports/ and
turns a recognized error into a concrete instruction, appended to (not
replacing) the existing feedback carrying the previous code and exact
exception.
"""

from __future__ import annotations

import re

_HINTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"cannot insert (\w+), already exists"),
        "You grouped by a column and then aggregated that same column under its "
        "original name. Use `df.groupby('X').size().reset_index(name='count')` "
        "instead of `df.groupby('X')['X'].count().reset_index()`.",
    ),
    (
        re.compile(r"Can only use \.dt accessor"),
        "That column is a string, not a datetime, so `.dt` fails. Convert it first "
        "with `pd.to_datetime(df['col'])`, or extract a year/substring directly "
        "from the string instead.",
    ),
    (
        re.compile(r"unterminated string literal|unexpected character after line continuation|invalid syntax"),
        "`code` must be a single valid Python snippet: use single quotes only for "
        "string literals, and separate statements with real newlines, not escaped "
        "`\\n` sequences.",
    ),
    (
        re.compile(r"KeyError: \('.*?'(?:, '.*?')+\)"),
        "Selecting multiple columns needs double brackets: "
        "`df[['col_a', 'col_b']]`, not a comma inside single brackets.",
    ),
    (
        re.compile(r"'numpy\.ndarray' object has no attribute"),
        "`.unique()` returns a numpy array, not a pandas Series, so it has no "
        "`.sort_values()`/`.reset_index()`. For a sorted list of distinct "
        "values use `df['col'].drop_duplicates().sort_values().to_frame()` -- "
        "one column. Do not call `.reset_index(name=...)` on it: that adds an "
        "extra `index` column to the result.",
    ),
    (
        re.compile(r"disallowed attribute: (merge|join|concat)\b"),
        "merge/join/concat are not allowed -- `df` already holds every column "
        "in one table. To compare counts of two values side by side, reshape "
        "instead: `t = df.groupby(['key', 'kind']).size().unstack(fill_value=0)` "
        "turns each value of 'kind' into its own column, then compare them "
        "directly, e.g. `t[t['x'] > t['y']]`. Do not filter out one of the "
        "values before unstacking.",
    ),
    (
        re.compile(r"'(Series|DataFrame)GroupBy' object has no attribute"),
        "A groupby object cannot do row-wise math. Compute the per-row value "
        "as a new column first, then group it: "
        "`df.assign(new_col=df['a'] - df['b']).groupby('key')['new_col'].mean()`. "
        "For a difference of two datetime columns in days use "
        "`(df['end'] - df['start']).dt.days`.",
    ),
]

_KEY_ERROR = re.compile(r"KeyError: '?([^'\s,)]+)'?")


def _key_error_hint(key: str, columns: list[str] | None) -> str:
    if columns is not None and key in columns:
        return (
            f"'{key}' is a real column of `df`, but the intermediate result you "
            "indexed no longer has it -- an earlier select, groupby, or "
            "reset_index dropped or renamed it. Keep it through each step, or "
            "index `df` itself."
        )
    if columns is not None and "_" not in key:
        return (
            f"'{key}' is not a column of this dataset. If it is a data value you "
            "expected to become a column (via pivot/unstack), check that you did "
            "not filter that value out before reshaping. Otherwise use one of "
            "the exact column names listed in the schema."
        )
    return (
        "That column name does not exist in this dataset. If you built it by "
        "joining words from the question (e.g. 'product_category' from "
        "\"product category\"), the real column is likely just the last word "
        "on its own (e.g. 'category') -- check the exact column names listed "
        "in the schema and use one of those, not a compound guess."
    )


def retry_hint(error: str, columns: list[str] | None = None) -> str | None:
    """Return a targeted instruction for a recognized error class, or None.

    `columns` (the dataset's real column names) lets a `KeyError` hint tell a
    guessed compound column name apart from a column dropped mid-pipeline or
    a data value that was expected to become a column. Without it, a
    `KeyError` gets the original compound-name hint."""
    for pattern, hint in _HINTS:
        if pattern.search(error):
            return hint
    match = _KEY_ERROR.search(error)
    if match:
        return _key_error_hint(match.group(1), columns)
    return None
