"""Targeted retry hints for SQLite errors -- the pandas hint table does not
apply here. Without them, hospital/h01 resent the same `^` query on every
retry (only the raw error was fed back)."""

from __future__ import annotations

import re

from csv_analyst.sql_engine.database import EXTRA_AGGREGATES, EXTRA_FUNCTIONS

_EXTRA = ", ".join(sorted({*EXTRA_FUNCTIONS, *EXTRA_AGGREGATES}))

_HINTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r'unrecognized token: "\^"'),
        "SQLite has no ^ operator. Use POWER(x, 2) or x * x for a square.",
    ),
    (
        re.compile(r"no such function: (\w+)"),
        f"That function does not exist here. Besides SQLite's built-ins (ABS, ROUND, "
        f"LOWER, LENGTH, SUBSTR, strftime, julianday, date, COALESCE, ...) the available "
        f"extras are: {_EXTRA}.",
    ),
    (
        re.compile(r"misuse of (aggregate|window) function"),
        "An aggregate cannot be used in WHERE or nested directly in another aggregate. "
        "Filter groups with HAVING, or compute the aggregate in a subquery/CTE first.",
    ),
    (
        re.compile(r"ambiguous column name"),
        "Qualify the column with its subquery/CTE alias, or rename one of them.",
    ),
    (
        re.compile(r"single SELECT|only one SQL statement"),
        "Write exactly one SELECT (a WITH ... SELECT is fine) -- no second statement, "
        "no INSERT/UPDATE/DELETE/CREATE/PRAGMA.",
    ),
    (
        re.compile(r"not authorized|access to .* is prohibited"),
        "Only reading from the table `data` is allowed -- no writes, PRAGMA, ATTACH, "
        "or system tables.",
    ),
]

_NO_SUCH_COLUMN = re.compile(r"no such column: ([\w.]+)")


def sql_retry_hint(error: str, columns: list[str] | None = None) -> str | None:
    for pattern, hint in _HINTS:
        if pattern.search(error or ""):
            return hint
    match = _NO_SUCH_COLUMN.search(error or "")
    if match:
        name = match.group(1).split(".")[-1]
        if columns is not None and name in columns:
            return (f"'{name}' is a real column of `data`, but not visible where you used it -- "
                    "select it in the subquery/CTE you read from.")
        return ("That column does not exist. An alias defined in a SELECT list cannot be used "
                "in the same SELECT's WHERE or in another expression of that list -- wrap it in "
                "a subquery/CTE. Otherwise use one of the exact column names in the schema.")
    return None
