"""Card H-R experiment (PLAN_v2.md): ingestion-time fact bank.

Facts the schema block does not carry, computed once per dataset from the
loaded DataFrame. Seven generic templates, fixed before any run (never written
from benchmark failures):

1. every value + row count of low-cardinality text columns;
2. numeric quantiles, zero and negative counts;
3. date columns (real or ISO text): parse rate, range, granularity, rows per year;
4. per repeating id: rows-per-id spread; label collisions of its free-text attributes;
5. columns null on exactly the same rows;
6. pairwise relations: date-pair order counts, many-to-one maps between text columns;
7. exact duplicate rows.

Each fact is one short sentence plus the columns it is about. Facts are
observations about this data, never instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from csv_analyst.profiling.schema import ColumnRole, SchemaContext

MAX_LISTED_VALUES = 50
MAX_YEARS_LISTED = 12
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$")


@dataclass(frozen=True)
class Fact:
    text: str
    columns: tuple[str, ...] = field(default_factory=tuple)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or str(s.dtype) in ("str", "string")


def _dates(s: pd.Series):
    """The column as datetimes if it holds dates (real or ISO text), else None."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if not _is_text(s):
        return None
    sample = s.dropna().astype(str).head(50)
    if sample.empty or not all(_ISO_DATE.match(v) for v in sample):
        return None
    return pd.to_datetime(s, errors="coerce")


def _value_facts(df, schema) -> list[Fact]:
    out = []
    for col in schema.columns:
        s = df[col.name]
        if col.role == ColumnRole.IDENTIFIER or not _is_text(s) or _dates(s) is not None:
            continue
        counts = s.value_counts(dropna=True)
        if len(counts) == 1:
            out.append(Fact(f"Column `{col.name}` has one single value on every non-null row: "
                            f"'{counts.index[0]}'.", (col.name,)))
        elif 2 <= len(counts) <= MAX_LISTED_VALUES:
            listed = ", ".join(f"'{v}' ({n})" for v, n in counts.items())
            nulls = f"; {int(s.isna().sum())} null rows" if s.isna().any() else ""
            out.append(Fact(f"Column `{col.name}` has exactly {len(counts)} distinct values "
                            f"(rows per value): {listed}{nulls}.", (col.name,)))
    return out


def _numeric_facts(df, schema) -> list[Fact]:
    out = []
    for col in schema.columns:
        s = df[col.name]
        if col.role == ColumnRole.IDENTIFIER or not pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
            continue
        v = s.dropna()
        if v.empty:
            continue
        q = v.quantile([0.05, 0.5, 0.95])
        text = (f"Column `{col.name}`: min {_fmt(v.min())}, 5th percentile {_fmt(q[0.05])}, "
                f"median {_fmt(q[0.5])}, 95th percentile {_fmt(q[0.95])}, max {_fmt(v.max())}")
        extra = []
        if (v == 0).any():
            extra.append(f"{int((v == 0).sum())} rows are 0")
        if (v < 0).any():
            extra.append(f"{int((v < 0).sum())} rows are negative")
        if (v == v.round()).all() and not pd.api.types.is_integer_dtype(s):
            extra.append("all values are whole numbers")
        out.append(Fact(text + ("; " + ", ".join(extra) if extra else "") + ".", (col.name,)))
    return out


def _date_facts(df, schema) -> list[Fact]:
    out = []
    for col in schema.columns:
        d = _dates(df[col.name])
        if d is None:
            continue
        present = int(df[col.name].notna().sum())
        parsed = d.dropna()
        if parsed.empty:
            continue
        has_time = bool((parsed != parsed.dt.normalize()).any())
        per_year = parsed.dt.year.value_counts().sort_index()
        years = (", ".join(f"{y}: {n}" for y, n in per_year.items()) if len(per_year) <= MAX_YEARS_LISTED
                 else f"{len(per_year)} distinct years")
        stored = "ISO text 'YYYY-MM-DD'" if _is_text(df[col.name]) else "dates"
        out.append(Fact(
            f"Column `{col.name}` holds {stored} ({'with' if has_time else 'no'} time of day): "
            f"{len(parsed)} of {present} non-null values parse as dates, range "
            f"{parsed.min().date()} to {parsed.max().date()}; rows per year: {years}.",
            (col.name,),
        ))
    return out


def _id_facts(df, schema) -> list[Fact]:
    out = []
    for col in schema.columns:
        if col.role != ColumnRole.IDENTIFIER or col.distinct_count >= schema.row_count:
            continue
        per_id = df.groupby(col.name, dropna=True).size()
        if per_id.empty:
            continue
        multi = int((per_id > 1).sum())
        out.append(Fact(
            f"Rows per `{col.name}`: min {per_id.min()}, median {_fmt(per_id.median())}, max {per_id.max()}; "
            f"{multi} of {len(per_id)} `{col.name}` values appear on more than one row.",
            (col.name,),
        ))
        # Label collisions only for name-like (free-text) attributes that are
        # not already a verified 1:1 label.
        roles = {c.name: c.role for c in schema.columns}
        labels = set(schema.entity_labels.get(col.name, []))
        text_attrs = [a for a in schema.entity_attributes.get(col.name, [])
                      if a not in labels and roles.get(a) == ColumnRole.FREE_TEXT
                      and _is_text(df[a]) and _dates(df[a]) is None]
        ids = df.dropna(subset=[col.name]).drop_duplicates(col.name)
        for a in text_attrs:
            n = ids[a].nunique()
            if n < len(ids):
                out.append(Fact(
                    f"`{a}` does not identify a `{col.name}`: {n} distinct `{a}` values across "
                    f"{len(ids)} `{col.name}` values.", (a, col.name),
                ))
        by_card = sorted(text_attrs, key=lambda a: -ids[a].nunique())[:2]
        if len(by_card) == 2:
            a, b = by_card
            combo = ids.groupby([a, b], dropna=True)[col.name].nunique()
            shared = int((combo > 1).sum())
            out.append(Fact(
                f"`{a}` + `{b}` together: {len(combo)} distinct combinations across {len(ids)} "
                f"`{col.name}` values; {shared} combinations belong to more than one `{col.name}`.",
                (a, b, col.name),
            ))
    return out


def _null_block_facts(df, schema) -> list[Fact]:
    groups: dict[tuple, list[str]] = {}
    for col in schema.columns:
        mask = df[col.name].isna()
        if mask.any():
            groups.setdefault(tuple(mask.to_numpy()), []).append(col.name)
    out = []
    for mask, cols in groups.items():
        if len(cols) >= 2:
            n = sum(mask)
            out.append(Fact(
                f"Columns {', '.join(f'`{c}`' for c in cols)} are null on exactly the same {n} rows "
                f"({100 * n / len(df):.1f}% of rows) and present together on the other rows.",
                tuple(cols),
            ))
    return out


def _relation_facts(df, schema) -> list[Fact]:
    out = []
    dates = {c.name: d for c in schema.columns if (d := _dates(df[c.name])) is not None}
    for a, b in combinations(dates, 2):
        both = dates[a].notna() & dates[b].notna()
        if not both.any():
            continue
        da, db = dates[a][both], dates[b][both]
        out.append(Fact(
            f"Comparing `{a}` with `{b}` on the {int(both.sum())} rows where both are present: "
            f"`{a}` is earlier on {int((da < db).sum())} rows, equal on {int((da == db).sum())}, "
            f"later on {int((da > db).sum())}.", (a, b),
        ))
    small = [c.name for c in schema.columns
             if c.role != ColumnRole.IDENTIFIER and _is_text(df[c.name]) and _dates(df[c.name]) is None
             and 2 <= c.distinct_count <= MAX_LISTED_VALUES]
    for a, b in combinations(small, 2):
        pair = df[[a, b]].dropna()
        a_to_b = pair.groupby(a)[b].nunique().max() == 1
        b_to_a = pair.groupby(b)[a].nunique().max() == 1
        if a_to_b and b_to_a:
            out.append(Fact(f"`{a}` and `{b}` map 1:1 (each value of one has exactly one value of the other).", (a, b)))
        elif a_to_b:
            out.append(Fact(f"Each `{a}` value belongs to exactly one `{b}` value (many-to-one).", (a, b)))
        elif b_to_a:
            out.append(Fact(f"Each `{b}` value belongs to exactly one `{a}` value (many-to-one).", (b, a)))
    return out


def _duplicate_facts(df, schema) -> list[Fact]:
    n = int(df.duplicated().sum())
    if n:
        return [Fact(f"{n} rows are exact duplicates of an earlier row (every column equal).", ())]
    return [Fact("No two rows are identical across all columns.", ())]


def build_fact_bank(df: pd.DataFrame, schema: SchemaContext) -> list[Fact]:
    facts: list[Fact] = []
    for template in (_value_facts, _numeric_facts, _date_facts, _id_facts,
                     _null_block_facts, _relation_facts, _duplicate_facts):
        facts.extend(template(df, schema))
    return facts
