"""Safe execution of one model-written SQLite query. Same contract as the
pandas `runner.execute_code` -- returns an ExecutionResult -- so consensus
and the answer stage are unchanged.

Safety (prototype-grade, see README limitations):
- `validate_sql`: exactly one statement, starting with SELECT or WITH.
- The query runs on a private in-memory copy of the CSV (table `data`);
  there is no file to reach.
- sqlite3 authorizer allowlist: SELECT, READ (only of `data`), FUNCTION,
  RECURSIVE. Writes, PRAGMA, ATTACH, schema changes, transactions, and reads
  of sqlite_master are refused by SQLite itself even if the static check is
  fooled. Extension loading is off by default in Python's sqlite3.
- A progress handler aborts queries that run past the sandbox timeout.
Runs in-process: SQLite cannot execute arbitrary Python, unlike generated
pandas code.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Optional

import pandas as pd

from csv_analyst.config import settings
from csv_analyst.contracts import ResultKind
from csv_analyst.execution.runner import ExecutionResult
from csv_analyst.sql_engine.database import TABLE_NAME, connect
from csv_analyst.validation.ast_guard import GuardResult

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/)*\s*", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _authorizer_for(readable: frozenset):
    def _authorizer(action, arg1, arg2, db_name, trigger):
        if action not in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_DENY
        # Reads only from the CSV's table (and the H-B1 entity tables built
        # from it) -- not sqlite_master or other internals.
        if action == sqlite3.SQLITE_READ and arg1 not in readable:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return _authorizer


_authorizer = _authorizer_for(frozenset({TABLE_NAME}))


def validate_sql(sql: str) -> GuardResult:
    """Static check before execution; the authorizer is the real enforcement."""
    body = _LEADING_COMMENTS.sub("", sql or "").strip().rstrip(";").strip()
    if not body:
        return GuardResult(ok=False, reason="empty SQL")
    if ";" in _STRING_LITERAL.sub("''", body):
        return GuardResult(ok=False, reason="only one SQL statement is allowed")
    if not re.match(r"(?i)(select|with)\b", body):
        return GuardResult(ok=False, reason="SQL must be a single SELECT (or WITH ... SELECT) statement")
    return GuardResult(ok=True)


def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_sql(
    df: pd.DataFrame, sql: str, timeout_s: Optional[int] = None, real_columns=(), tables=(),
) -> ExecutionResult:
    """`tables`: card H-B1 entity tables (`csv_analyst.sql_engine.entities.EntityTable`)
    created next to `data` and readable by the query."""
    timeout_s = settings.sandbox_timeout_s if timeout_s is None else timeout_s
    start = time.monotonic()
    deadline = start + timeout_s
    try:
        conn = connect(df, real_columns, tuple(tables))
    except Exception as exc:  # loading the CSV itself failed -- not the model's fault
        return ExecutionResult(ok=False, error=f"could not load data into SQLite: {exc}",
                               ms=int((time.monotonic() - start) * 1000))
    conn.set_authorizer(_authorizer_for(frozenset({TABLE_NAME, *(t.name for t in tables)})))
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        rows = [[_json_safe(v) for v in row] for row in cur.fetchall()]
    except sqlite3.OperationalError as exc:
        ms = int((time.monotonic() - start) * 1000)
        if "interrupted" in str(exc):
            return ExecutionResult(ok=False, error=f"TimeoutError: query exceeded {timeout_s}s", ms=ms)
        return ExecutionResult(ok=False, error=f"OperationalError: {exc}", ms=ms)
    except sqlite3.DatabaseError as exc:
        return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                               ms=int((time.monotonic() - start) * 1000))
    finally:
        conn.close()

    ms = int((time.monotonic() - start) * 1000)
    if not columns:
        return ExecutionResult(ok=False, error="query returned no result set", ms=ms)
    # A single value is a scalar answer, like `result = df['x'].mean()` on the pandas path.
    if len(rows) == 1 and len(columns) == 1:
        return ExecutionResult(ok=True, kind=ResultKind.SCALAR, value=rows[0][0], ms=ms)
    return ExecutionResult(ok=True, kind=ResultKind.DATAFRAME, value=pd.DataFrame(rows, columns=columns), ms=ms)
