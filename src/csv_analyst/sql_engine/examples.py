"""Card H-G (PLAN_v2.md): worked examples generated from the loaded CSV's own
schema by fixed templates.

The static worked examples in the planner prompt live on an invented table
(H-C C1). These show the same grain techniques with the real column names:
distinct entities per entity attribute, per-entity average, and distinct
entities vs rows on a row-level column. Everything comes from the profiler's
verified entity attributes; nothing depends on the question or the benchmark.

Selection rules (fixed before the first run):
- entities: repeating identifiers with entity attributes, most attributes
  first (ties: column order), at most MAX_ENTITIES;
- categorical column: role dimension, not a 1:1 label, not an identifier,
  2..MAX_DISTINCT distinct values; fewest distinct values first (ties: column
  order);
- numeric attribute: role measure, name not code/id-like, first in column order;
- an example whose query repeats an earlier one is skipped.
"""

from __future__ import annotations

import json
import re

from csv_analyst.profiling.schema import ColumnProfile, ColumnRole, SchemaContext

MAX_ENTITIES = 2
MAX_DISTINCT = 50
_ID_LIKE = re.compile(r"(^|_)(id|code|zip|no|number)($|_)")

HEADER = (
    "Worked examples on YOUR table (generated from its schema by fixed templates; they show "
    "techniques with your column names, they are not answers to the question below):"
)


def _entity_noun(id_name: str) -> str:
    base = re.sub(r"_?id$", "", id_name).replace("_", " ").strip() or id_name
    return base + "s"


def _row_text(id_name: str) -> str:
    # Short on purpose (H-G step G2): the 7b copies this text into its own
    # `row_represents`, so every word here is an output token on each call.
    return f"one row ({id_name} repeats)"


def _categorical(cols: list[ColumnProfile], labels: set[str]) -> ColumnProfile | None:
    ok = [c for c in cols if c.role == ColumnRole.DIMENSION and c.name not in labels
          and 2 <= c.distinct_count <= MAX_DISTINCT]
    return min(ok, key=lambda c: c.distinct_count) if ok else None  # min() keeps column order on ties


def _numeric(cols: list[ColumnProfile]) -> ColumnProfile | None:
    ok = [c for c in cols if c.role == ColumnRole.MEASURE and not _ID_LIKE.search(c.name.lower())]
    return ok[0] if ok else None


def _not_null(col: ColumnProfile) -> str:
    return f" WHERE {col.name} IS NOT NULL" if col.null_pct > 0 else ""


def _example(question: str, why: str, row: str, entity: str, plan: list[str], cols: list[str], sql: str) -> dict:
    return {"question": question, "why": why, "plan": {
        "answerability": "answerable", "row_represents": row, "question_entity": entity,
        "plan": plan, "columns_used": cols, "code": sql,
    }}


def generate_examples(schema: SchemaContext, tables: tuple = ()) -> list[dict]:
    """`tables`: card H-B1 entity tables. When an entity has one, its two
    per-entity examples read that table (one row per id) instead of
    collapsing `data`; the row-level examples stay on `data`."""
    by_name = {c.name: c for c in schema.columns}
    table_of = {t.id_name: t for t in tables}
    labels = {lab for labs in schema.entity_labels.values() for lab in labs}
    entities = [
        (i, name) for i, name in enumerate(by_name)
        if name in schema.entity_attributes and schema.entity_attributes[name]
        and by_name[name].distinct_count < schema.row_count
    ]
    entities.sort(key=lambda e: (-len(schema.entity_attributes[e[1]]), e[0]))

    out: list[dict] = []
    seen_sql: set[str] = set()

    def add(ex: dict) -> None:
        if ex["plan"]["code"] not in seen_sql:
            seen_sql.add(ex["plan"]["code"])
            out.append(ex)

    for _, id_name in entities[:MAX_ENTITIES]:
        noun = _entity_noun(id_name)
        row = _row_text(id_name)
        attrs = [by_name[a] for a in schema.entity_attributes[id_name] if a in by_name]
        owned = {id_name, *schema.entity_attributes[id_name]}
        row_level = [c for c in schema.columns if c.name not in owned and c.role != ColumnRole.IDENTIFIER]
        table = table_of.get(id_name)
        if table:
            attrs = [by_name[a] for a in table.columns[1:]]

        attr = _categorical(attrs, labels)
        if attr and table:
            add(_example(
                f"How many {noun} are there per {attr.name}?",
                f"{attr.name} is a per-{noun[:-1]} attribute and `{table.name}` has exactly one row per "
                f"{id_name}, so count its rows:",
                row, noun, [f"Use {table.name} (one row per {id_name})", f"Group by {attr.name}",
                            "Count rows per group"],
                [id_name, attr.name],
                f"SELECT {attr.name}, COUNT(*) AS count FROM {table.name}{_not_null(attr)} GROUP BY {attr.name}",
            ))
        elif attr:
            add(_example(
                f"How many {noun} are there per {attr.name}?",
                f"{attr.name} is a per-{noun[:-1]} attribute (it repeats identically on every row of the "
                f"same {id_name}) and the question counts {noun}, so count distinct {id_name}:",
                row, noun, [f"Group by {attr.name}", f"Count distinct {id_name} per group"],
                [id_name, attr.name],
                f"SELECT {attr.name}, COUNT(DISTINCT {id_name}) AS count FROM data{_not_null(attr)} "
                f"GROUP BY {attr.name}",
            ))
        num = _numeric(attrs)
        if num and table:
            add(_example(
                f"What is the average {num.name} of {noun}?",
                f"{num.name} is a per-{noun[:-1]} attribute: `{table.name}` has it once per {id_name}, "
                f"while `data` repeats it on every row of the same {id_name}:",
                row, noun, [f"Use {table.name} (one row per {id_name})", f"Take the average of {num.name}"],
                [id_name, num.name],
                f"SELECT AVG({num.name}) AS avg_{num.name} FROM {table.name}",
            ))
        elif num:
            add(_example(
                f"What is the average {num.name} of {noun}?",
                f"{num.name} is a per-{noun[:-1]} attribute: reduce to one row per {id_name} before "
                f"averaging, or {noun} with more rows count more than once:",
                row, noun, [f"Reduce to one row per {id_name}", f"Take the average of {num.name}"],
                [id_name, num.name],
                f"SELECT AVG({num.name}) AS avg_{num.name} FROM (SELECT DISTINCT {id_name}, {num.name} FROM data)",
            ))
        col = _categorical(row_level, labels)
        if col:
            add(_example(
                f"How many different {noun} appear in each {col.name}?",
                f"{col.name} can differ between rows of the same {id_name} and the question counts "
                f"{noun}, so count distinct {id_name}:",
                row, noun, [f"Group by {col.name}", f"Count distinct {id_name} per group"],
                [id_name, col.name],
                f"SELECT {col.name}, COUNT(DISTINCT {id_name}) AS count FROM data{_not_null(col)} "
                f"GROUP BY {col.name}",
            ))
            add(_example(
                f"How many rows are there for each {col.name}?",
                f"Here the question counts the rows themselves, not {noun}, so no collapse:",
                row, f"rows by {col.name}", [f"Group by {col.name}", "Count rows per group"],
                [col.name],
                f"SELECT {col.name}, COUNT(*) AS count FROM data{_not_null(col)} GROUP BY {col.name}",
            ))
    return out


def examples_block(schema: SchemaContext, tables: tuple = ()) -> str:
    examples = generate_examples(schema, tables)
    if not examples:
        return ""
    parts = [HEADER, ""]
    for ex in examples:
        parts += [f'Q: "{ex["question"]}"', ex["why"], json.dumps(ex["plan"], ensure_ascii=False), ""]
    return "\n".join(parts)
