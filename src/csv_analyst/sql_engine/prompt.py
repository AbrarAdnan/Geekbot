"""SQL schema rendering and the SQL planner prompt.

Two schema styles (`settings.sql_schema_style`):
- "profile": the profiler's facts as a bulleted list (the same facts the
  pandas prompt gets, in SQL terms).
- "ddl": a `CREATE TABLE data (...)` statement with the same facts as column
  comments -- the layout text-to-SQL models are mostly trained on.
Both describe the SQLite table as it really is: SQL types, dates as ISO
text, text columns that hold ISO dates labelled as dates, and grain notes in
COUNT(DISTINCT)/SELECT DISTINCT terms.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from csv_analyst.config import settings
from csv_analyst.profiling.schema import ColumnProfile, ColumnRole, SchemaContext
from csv_analyst.sql_engine.entities import entity_tables, tables_block
from csv_analyst.sql_engine.examples import examples_block

PROMPT_VERSION = "planner-sql-v4"  # shipped default; settings.sql_planner_prompt selects the file
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"

SAMPLE_VALUES = 5
TOP_VALUES = 8
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$")

ID_REPEATS_NOTE = (
    "Note: a column marked id_repeats=true is a finer-grained identifier than one row "
    "per entity (e.g. one row per transaction/admission/order, not one per "
    "customer/patient). A per-entity statistic must count DISTINCT that id, or reduce to "
    "one row per id first, or it will overweight entities that repeat more often."
)
# Card H-C C1: same note without the benchmark datasets' words, used with planner-sql-v3.
ID_REPEATS_NOTE_V3 = (
    "Note: a column marked id_repeats=true is a finer-grained identifier than one row "
    "per entity (e.g. one row per loan, not one per member). A per-entity statistic must "
    "count DISTINCT that id, or reduce to one row per id first, or it will overweight "
    "entities that repeat more often."
)


def _id_repeats_note() -> str:
    return ID_REPEATS_NOTE if settings.sql_planner_prompt == "planner-sql-v2" else ID_REPEATS_NOTE_V3


ENTITY_RULE = (
    "If the question counts, averages, ranks, or lists the ENTITY itself (the thing an id "
    "on the left identifies) using its attributes, count DISTINCT <that id> or first reduce "
    "to one row per id (SELECT DISTINCT <that id>, <attributes> FROM data) -- otherwise "
    "entities with more rows are counted more than once. If the question counts or sums "
    "the rows/events themselves, do not deduplicate. When reporting an entity, include its "
    "1:1 readable label if one is listed."
)


def _value(v: Any) -> str:
    if isinstance(v, pd.Timestamp):
        return repr(v.strftime("%Y-%m-%d") if v == v.normalize() else v.strftime("%Y-%m-%d %H:%M:%S"))
    return repr(v)


def _is_text_date(col: ColumnProfile) -> bool:
    samples = [str(v) for v in col.sample_values if v is not None]
    return bool(samples) and all(_ISO_DATE.match(s) for s in samples)


def real_measure_columns(schema: SchemaContext) -> list[str]:
    """Integer measure columns stored as REAL when settings.sql_real_measures is on."""
    if not settings.sql_real_measures:
        return []
    return [c.name for c in schema.columns if str(c.dtype).startswith("int") and c.role == ColumnRole.MEASURE]


def nullable_columns(schema: SchemaContext) -> list[str]:
    return [c.name for c in schema.columns if c.null_pct > 0]


def _sql_type(col: ColumnProfile, real: bool = False) -> str:
    dtype = str(col.dtype)
    if dtype.startswith("datetime") or (dtype in ("object", "str") and _is_text_date(col)):
        return "TEXT date 'YYYY-MM-DD'"
    if dtype.startswith("int") and real:
        return "REAL"
    if dtype.startswith("int"):
        return "INTEGER"
    if dtype.startswith("float"):
        return "REAL"
    if dtype == "bool":
        return "INTEGER 0/1"
    return "TEXT"


def _date_text(value: Optional[str]) -> Optional[str]:
    return value[:-9] if value and value.endswith(" 00:00:00") else value


def _facts(col: ColumnProfile, row_count: int) -> tuple[list[str], bool]:
    """Profiler facts for one column, and whether it is a repeating id."""
    detail = [f"role={col.role.value}", f"null={col.null_pct:.1f}%"]
    repeats = False
    if col.role == ColumnRole.MEASURE and col.numeric_min is not None:
        detail.append(f"range=[{col.numeric_min}, {col.numeric_max}]")
    if col.role == ColumnRole.IDENTIFIER:
        if col.distinct_count < row_count:
            detail.append(f"id_repeats=true ({col.distinct_count}/{row_count} distinct)")
            repeats = True
        else:
            detail.append(f"{col.distinct_count} distinct (unique per row)")
    if col.role == ColumnRole.DATE:
        if col.date_ambiguous:
            detail.append("date_ambiguous=true (do not assume day/month order)")
        elif col.date_format:
            detail.append(f"range=[{_date_text(col.date_min)}, {_date_text(col.date_max)}]")
    if col.role == ColumnRole.DIMENSION and col.top_values:
        top = ", ".join(_value(v.value) for v in col.top_values[:TOP_VALUES])
        detail.append(f"values={{{top}}} ({col.distinct_count} distinct)")
    if col.sample_values and col.role != ColumnRole.DIMENSION:
        detail.append("samples=[" + ", ".join(_value(v) for v in col.sample_values[:SAMPLE_VALUES]) + "]")
    return detail, repeats


def _warnings(schema: SchemaContext) -> list[str]:
    out = []
    for w in schema.warnings:
        m = re.match(r"Column '(\w+)' parsed from \w+ to datetime64", w)
        out.append(f"{m.group(1)} holds ISO text dates" if m else w)
    return out


def _entity_lines(schema: SchemaContext) -> list[str]:
    if not schema.entity_attributes:
        return []
    distinct = {c.name: c.distinct_count for c in schema.columns}
    lines = ["Entity attributes (derived from the data: each column on the right has exactly one "
             "value per value of the id on the left, so it repeats on every row of that entity):"]
    for id_name, attrs in schema.entity_attributes.items():
        line = f"- {id_name} ({distinct.get(id_name, '?')} distinct) -> " + ", ".join(attrs)
        labels = schema.entity_labels.get(id_name)
        if labels:
            line += f"  [1:1 readable label: {', '.join(labels)}]"
        lines.append(line)
    owned = set(schema.entity_attributes) | {a for attrs in schema.entity_attributes.values() for a in attrs}
    row_level = [c.name for c in schema.columns if c.name not in owned and c.distinct_count > 1]
    lines.append("Row-level columns (can differ between rows of the same entity): " + ", ".join(row_level))
    lines.append(ENTITY_RULE)
    return lines


def render_profile(schema: SchemaContext) -> str:
    lines = [f"Rows: {schema.row_count}", f"Columns: {schema.column_count}", ""]
    any_repeats = False
    real = set(real_measure_columns(schema))
    for col in schema.columns:
        detail, repeats = _facts(col, schema.row_count)
        any_repeats |= repeats
        lines.append(f"- {col.name} ({_sql_type(col, col.name in real)}): {', '.join(detail)}")
    if any_repeats:
        lines += ["", _id_repeats_note()]
    warnings = _warnings(schema)
    if warnings:
        lines += ["", "Data quality warnings: " + "; ".join(warnings)]
    entity = _entity_lines(schema)
    if entity:
        lines += [""] + entity
    return "\n".join(lines)


_NA_WARNING = re.compile(r"(\d+) NA-like sentinel value\(s\) converted to null across (\d+) column\(s\)")
_DATE_PARSE_WARNING = re.compile(r"Column '\w+' parsed from \w+ to datetime64")


def _plain_facts(col: ColumnProfile, row_count: int) -> list[str]:
    """Card H-C C2: what the fact bank does not already say. No profiler role
    tags, no value lists, numeric or date ranges (all in the fact bank)."""
    detail = [f"null={col.null_pct:.1f}%"]
    if col.role == ColumnRole.IDENTIFIER:
        detail.append(f"repeats ({col.distinct_count} distinct in {row_count} rows)"
                      if col.distinct_count < row_count else "unique per row")
    if col.role == ColumnRole.DATE and col.date_ambiguous:
        detail.append("day/month order ambiguous (do not assume one)")
    if col.sample_values:
        detail.append("samples=[" + ", ".join(_value(v) for v in col.sample_values[:SAMPLE_VALUES]) + "]")
    return detail


def render_plain(schema: SchemaContext) -> str:
    lines = [f"Rows: {schema.row_count}", f"Columns: {schema.column_count}", ""]
    real = set(real_measure_columns(schema))
    for col in schema.columns:
        lines.append(f"- {col.name} ({_sql_type(col, col.name in real)}): "
                     + ", ".join(_plain_facts(col, schema.row_count)))
    notes, other = [], []
    for w in schema.warnings:
        m = _NA_WARNING.match(w)
        if m:
            notes.append(f"Cells such as 'NA', 'N/A' or empty were stored as NULL ({m.group(1)} cells in "
                         f"{m.group(2)} columns).")
        elif not _DATE_PARSE_WARNING.match(w):  # the column type already says it is a date
            other.append(w)
    if other:
        notes.append("Data quality warnings: " + "; ".join(other))
    if notes:
        lines += [""] + notes
    entity = _entity_lines(schema)
    if entity:
        lines += [""] + entity[:-1]  # attribute lists and row-level columns; the rule is in the prompt (rule 18)
    return "\n".join(lines)


def render_ddl(schema: SchemaContext) -> str:
    lines = [f"CREATE TABLE data (  -- {schema.row_count} rows"]
    any_repeats = False
    real = set(real_measure_columns(schema))
    for i, col in enumerate(schema.columns):
        detail, repeats = _facts(col, schema.row_count)
        any_repeats |= repeats
        comma = "," if i < len(schema.columns) - 1 else ""
        full_type = _sql_type(col, col.name in real)
        sql_type = full_type.split(" ")[0]
        note = "date as 'YYYY-MM-DD' text; " if "date" in full_type else ""
        lines.append(f"  {col.name} {sql_type}{comma}  -- {note}{'; '.join(detail)}")
    lines.append(");")
    if any_repeats:
        lines += ["", _id_repeats_note()]
    warnings = _warnings(schema)
    if warnings:
        lines += ["", "Data quality warnings: " + "; ".join(warnings)]
    entity = _entity_lines(schema)
    if entity:
        lines += [""] + entity
    return "\n".join(lines)


def render_sql_schema(schema: SchemaContext, style: Optional[str] = None) -> str:
    style = style or settings.sql_schema_style
    if style == "ddl":
        return render_ddl(schema)
    if style == "profile":
        return render_profile(schema)
    if style == "plain":
        return render_plain(schema)
    raise ValueError(f"unknown sql_schema_style {style!r}; expected 'profile', 'plain' or 'ddl'")


def build_sql_prompt(
    question: str, schema: SchemaContext, style: Optional[str] = None, extra_context: str = "",
) -> str:
    template = (_PROMPTS_DIR / f"{settings.sql_planner_prompt}.txt").read_text(encoding="utf-8")
    schema_block = render_sql_schema(schema, style)
    tables = entity_tables(schema) if settings.sql_entity_tables else ()
    if tables:  # card H-B1: the entity tables, right after the columns of `data`
        schema_block = f"{schema_block}\n\n{tables_block(tables)}"
    if extra_context:
        schema_block = f"{schema_block}\n\n{extra_context}"
    prompt = template.format(schema_block=schema_block, question=question)
    if settings.sql_generated_examples:
        # Card H-G: examples on the CSV's own columns go after the static ones.
        block = examples_block(schema, tables)
        if block:
            marker = "Respond with ONLY a JSON object"
            prompt = prompt.replace(marker, f"{block}\n{marker}", 1)
    return prompt


def normalize_sql(sql: str) -> str:
    """A query escaped like a string literal (literal backslash-n) gets real
    newlines back. Only applied when the query has no real newline at all, so
    a valid multi-line query is returned unchanged."""
    if "\\n" in sql and "\n" not in sql:
        return sql.replace("\\n", "\n").replace("\\t", "\t")
    return sql
