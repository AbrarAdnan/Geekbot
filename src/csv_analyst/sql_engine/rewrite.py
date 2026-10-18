"""Deterministic rewrite of a model-written query to pandas semantics.

`drop_null_groups`: pandas `groupby` drops NaN keys by default
(dropna=True); SQLite keeps a NULL group, and on a LEFT-JOIN-flattened CSV
that group can be the biggest one (hospital: 1,148 never-admitted rows with
no diagnosis). Prompt rules asking the 7b to add `IS NOT NULL` were ignored
(PLAN_v2.md, SQL v2), so this adds `<key> IS NOT NULL` to the WHERE of every
SELECT whose GROUP BY key references a column that has nulls.

Left alone: keys the model already handles (COALESCE/IFNULL/CASE/IS NULL),
keys with aggregates or window functions, and any query sqlglot cannot parse
(returned unchanged -- the rewrite never blocks execution).
"""

from __future__ import annotations

from typing import Iterable

import sqlglot
from sqlglot import exp

_HANDLES_NULL = (exp.Coalesce, exp.Case, exp.If, exp.Is)


def _resolve(key: exp.Expression, select: exp.Select) -> exp.Expression:
    """GROUP BY 1 / GROUP BY <alias> -> the select-list expression it names."""
    projections = select.expressions
    if isinstance(key, exp.Literal) and not key.is_string:
        i = int(key.this) - 1
        if 0 <= i < len(projections):
            return projections[i].unalias()
    if isinstance(key, exp.Column) and not key.table:
        for p in projections:
            if isinstance(p, exp.Alias) and p.alias == key.name:
                return p.this
    return key


def _needs_guard(key: exp.Expression, nullable: set[str]) -> bool:
    if key.find(exp.AggFunc, exp.Window, *_HANDLES_NULL):
        return False
    return any(c.name in nullable for c in key.find_all(exp.Column))


def drop_null_groups(sql: str, nullable_columns: Iterable[str]) -> str:
    nullable = set(nullable_columns)
    if not nullable:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return sql
    changed = False
    for select in tree.find_all(exp.Select):
        group = select.args.get("group")
        if not group:
            continue
        where = select.args.get("where")
        existing = where.this.sql("sqlite") if where else ""
        for raw_key in group.expressions:
            key = _resolve(raw_key, select)
            if not _needs_guard(key, nullable):
                continue
            guard = exp.Not(this=exp.Is(this=key.copy(), expression=exp.Null()))
            if guard.sql("sqlite") in existing:
                continue
            select.where(guard, dialect="sqlite", copy=False)
            existing = select.args["where"].this.sql("sqlite")
            changed = True
    return tree.sql("sqlite", pretty=True) if changed else sql
