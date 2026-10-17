"""Card H-B1 (PLAN_v2.md): entity tables for the SQL engine.

The CSV is one flat table (`data`); an entity's attributes repeat on every
row of that entity (an invoice total on every invoice line, a patient's
weight on every admission). Summing or averaging such a column over `data`
counts an entity once per row -- the main failure of the 7b on the dev set.

From the profiler's verified functional dependencies
(`SchemaContext.entity_attributes`: exactly one value per id on every row)
this builds one extra table per entity -- `invoices`, `patients`, ... -- with
exactly one row per id, next to the unchanged `data`. Nothing depends on the
question or the benchmark.

Rules (fixed before the first run):
- an entity gets a table when its id repeats (fewer distinct ids than rows)
  and it has at least one attribute with more than one distinct value;
- columns: the id, then its attributes in profiler order; constant columns
  (one distinct value in the whole CSV) are left out -- they "depend" on
  every id trivially and would put line-level columns into entity tables;
- name: the id without its `_id` suffix, pluralized (`invoice_id` ->
  `invoices`); `<id>_entity` when that does not apply or would clash;
- rows: one per non-null id; each attribute is MAX() over that id's rows,
  which is its single value (NULLs ignored).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from csv_analyst.config import settings
from csv_analyst.profiling.schema import SchemaContext

DATA_TABLE = "data"


@dataclass(frozen=True)
class EntityTable:
    name: str
    id_name: str
    columns: tuple[str, ...]  # id first, then the attributes
    rows: int


def _plural(word: str) -> str:
    if re.search(r"(s|x|z|ch|sh)$", word):
        return word + "es"
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"
    return word + "s"


def table_name(id_name: str, taken: set[str]) -> str:
    m = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*?)_id", id_name)
    name = _plural(m.group(1)) if m else f"{id_name}_entity"
    if name in taken or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        name = f"{id_name}_entity"
    return name


def _prefix(id_name: str) -> str | None:
    m = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*?)_id", id_name)
    return m.group(1) + "_" if m else None


def entity_tables(schema: SchemaContext, constants: str | None = None) -> tuple[EntityTable, ...]:
    """`constants` (H-B1 fix F1, settings.sql_entity_constants): "off" leaves
    every constant column out; "prefix" keeps a constant column in the table
    of the entity its name starts with (`support_rep_title` -> `support_reps`;
    `quantity` names no entity and stays out)."""
    constants = settings.sql_entity_constants if constants is None else constants
    by_name = {c.name: c for c in schema.columns}
    taken = {DATA_TABLE}
    out = []
    for id_name, attrs in schema.entity_attributes.items():
        col = by_name.get(id_name)
        if col is None or col.distinct_count >= schema.row_count:
            continue
        prefix = _prefix(id_name) if constants == "prefix" else None
        keep = [a for a in attrs if a in by_name and a != id_name
                and (by_name[a].distinct_count > 1 or (prefix is not None and a.startswith(prefix)))]
        if not keep:
            continue
        name = table_name(id_name, taken)
        taken.add(name)
        out.append(EntityTable(name=name, id_name=id_name, columns=(id_name, *keep), rows=col.distinct_count))
    return tuple(out)


def create_sql(table: EntityTable) -> str:
    q = lambda c: '"' + c.replace('"', '""') + '"'  # noqa: E731
    cols = ", ".join([q(table.id_name)] + [f"MAX({q(c)}) AS {q(c)}" for c in table.columns[1:]])
    return (f"CREATE TABLE {q(table.name)} AS SELECT {cols} FROM {DATA_TABLE} "
            f"WHERE {q(table.id_name)} IS NOT NULL GROUP BY {q(table.id_name)}")


def tables_block(tables: tuple[EntityTable, ...]) -> str:
    """Schema-block section listing the entity tables (types as in `data`)."""
    if not tables:
        return ""
    lines = ["Entity tables (built from `data`, one row per id -- an entity's attributes appear once "
             "here instead of on every one of its rows in `data`):"]
    for t in tables:
        lines.append(f"- {t.name}: one row per {t.id_name} ({t.rows} rows) -> " + ", ".join(t.columns))
    return "\n".join(lines)
