"""DataFrame -> SchemaContext profiler (PLAN.md Phase 1, section 6).

Never asserts a role or a date format as fact — `role_confidence` and
`date_ambiguous` exist precisely so the planner (Phase 2) knows what is a
guess versus what is certain.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from csv_analyst.ingestion.loader import LoadResult
from csv_analyst.profiling.schema import ColumnProfile, ColumnRole, SchemaContext, ValueCount

CATEGORICAL_MAX_DISTINCT = 50
TOP_VALUES_LIMIT = 20
SAMPLE_VALUES_LIMIT = 5

_ID_NAME_RE = re.compile(r"(^|_)id$", re.IGNORECASE)

# Tried in order; the first format that parses every sampled value cleanly wins,
# unless it belongs to an ambiguous pair (see below), in which case we refuse to guess.
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%B %d, %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%d-%b-%Y",
]

_AMBIGUOUS_FORMAT_PAIRS = [
    ("%m/%d/%Y", "%d/%m/%Y"),
    ("%m-%d-%Y", "%d-%m-%Y"),
]

_DATE_LIKE_RE = re.compile(
    r"^\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}$"
    r"|^\d{8}$"
    r"|^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"
    r"|^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"
)


def profile_dataset(load_result: LoadResult, dataset_id: str | None = None) -> SchemaContext:
    df = load_result.df
    dataset_id = dataset_id or str(uuid.uuid4())
    row_count = len(df)

    warnings = [w.message for w in load_result.warnings]
    columns = []
    for col in df.columns:
        profile, conversion_warning = _profile_column(df, col, row_count)
        columns.append(profile)
        if conversion_warning:
            warnings.append(conversion_warning)

    entity_attributes, entity_labels = _entity_attributes(df, columns, row_count)

    return SchemaContext(
        dataset_id=dataset_id,
        source_filename=load_result.source_path.name,
        row_count=row_count,
        column_count=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        encoding=load_result.encoding,
        delimiter=load_result.delimiter,
        columns=columns,
        warnings=warnings,
        entity_attributes=entity_attributes,
        entity_labels=entity_labels,
        profiled_at=datetime.now(timezone.utc),
    )


def _profile_column(df: pd.DataFrame, col: str, row_count: int) -> tuple[ColumnProfile, Optional[str]]:
    series = df[col]
    non_null = series.dropna()
    null_count = int(series.isna().sum())
    null_pct = (null_count / row_count * 100) if row_count else 0.0
    distinct_count = int(non_null.nunique())

    date_info = _detect_date_format(non_null) if pd.api.types.is_string_dtype(series) else None
    role, confidence = _infer_role(col, series, non_null, distinct_count, row_count, date_info)

    converted_from = None
    conversion_warning = None
    if role == ColumnRole.DATE and date_info is not None and not date_info["ambiguous"] and date_info["format"]:
        # PLAN.md Phase 6B.1: an unambiguous, fully-parsing date format is
        # detected but the column is still `object` dtype (loader.py leaves
        # datetime parsing to the profiler on purpose) -- convert it here so
        # the planner sees real datetime64 and never has to be told not to
        # call `.dt` on a string.
        converted_from = str(series.dtype)
        converted = pd.to_datetime(series.astype(str).str.strip(), format=date_info["format"], errors="coerce")
        converted[series.isna()] = pd.NaT
        df[col] = converted
        series = df[col]
        non_null = series.dropna()
        conversion_warning = (
            f"Column '{col}' parsed from {converted_from} to datetime64 "
            f"(detected format {date_info['format']})"
        )

    numeric_min = numeric_max = numeric_mean = numeric_std = None
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and not non_null.empty:
        numeric_min = float(non_null.min())
        numeric_max = float(non_null.max())
        numeric_mean = float(non_null.mean())
        numeric_std = float(non_null.std()) if len(non_null) > 1 else 0.0

    top_values = None
    if role == ColumnRole.DIMENSION and 0 < distinct_count <= CATEGORICAL_MAX_DISTINCT:
        counts = non_null.value_counts().head(TOP_VALUES_LIMIT)
        top_values = [ValueCount(value=_jsonable(v), count=int(c)) for v, c in counts.items()]

    sample_values = [_jsonable(v) for v in non_null.drop_duplicates().head(SAMPLE_VALUES_LIMIT).tolist()]

    return ColumnProfile(
        name=col,
        original_name=col,
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=round(null_pct, 4),
        distinct_count=distinct_count,
        role=role,
        role_confidence=confidence,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        sample_values=sample_values,
        top_values=top_values,
        date_format=date_info["format"] if date_info else None,
        date_min=date_info["min"] if date_info else None,
        date_max=date_info["max"] if date_info else None,
        date_ambiguous=date_info["ambiguous"] if date_info else False,
        converted_from=converted_from,
    ), conversion_warning


# Evidence floor for "column X has one value per id": only ids that actually
# repeat can show it (a one-row id is trivially constant), so an id needs at
# least this many rows sitting in its repeating groups before any of its
# attributes are trusted.
_MIN_ROWS_IN_REPEATED_GROUPS = 20


def _entity_attributes(
    df: pd.DataFrame, columns: list[ColumnProfile], row_count: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Which columns are attributes of which repeating identifier, from the
    data itself.

    A column with exactly one value per value of a repeating id is an
    attribute of that id's entity (e.g. allergies per patient_id: the same
    value repeats on every admission row of that patient). Columns that are
    constant across the whole file are skipped -- they describe nothing.
    Returns ({id: [attributes]}, {id: [1:1 readable labels]}), finest id
    first; both empty when no id repeats enough to trust."""
    ids = sorted(
        (c for c in columns if c.role == ColumnRole.IDENTIFIER and 1 < c.distinct_count < row_count),
        key=lambda c: c.distinct_count,
        reverse=True,
    )
    candidates = [c for c in columns if df[c.name].nunique(dropna=False) > 1]
    attributes: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    for id_col in ids:
        keyed = df[df[id_col.name].notna()]
        sizes = keyed.groupby(id_col.name).size()
        if sizes[sizes > 1].sum() < _MIN_ROWS_IN_REPEATED_GROUPS:
            continue
        others = [c for c in candidates if c.name != id_col.name]
        per_id = keyed.groupby(id_col.name)[[c.name for c in others]].nunique(dropna=False)
        determined = [c for c in others if (per_id[c.name] <= 1).all()]
        if not determined:
            continue
        attributes[id_col.name] = [c.name for c in determined]
        same_count = [c.name for c in determined if c.distinct_count == id_col.distinct_count]
        if same_count:
            labels[id_col.name] = same_count
    return attributes, labels


_LONG_TEXT_AVG_LEN = 30


def _infer_role(
    name: str,
    series: pd.Series,
    non_null: pd.Series,
    distinct_count: int,
    row_count: int,
    date_info: dict | None,
) -> tuple[ColumnRole, float]:
    name_lower = name.lower()
    id_like_name = bool(_ID_NAME_RE.search(name_lower))
    uniqueness_ratio = (distinct_count / row_count) if row_count else 0.0

    if id_like_name and uniqueness_ratio > 0.9:
        return ColumnRole.IDENTIFIER, 0.9
    if id_like_name and distinct_count > 1 and uniqueness_ratio <= 0.9:
        # id-named but repeats within the file (e.g. one row per admission
        # instead of one row per patient) -- still an identifier, just of a
        # finer-grained entity than the row itself. Flagged at lower
        # confidence so the planner treats the repetition as informative,
        # not noise: see render_schema_block's "repeats" note.
        return ColumnRole.IDENTIFIER, 0.6
    if date_info is not None:
        return (ColumnRole.DATE, 0.5) if date_info["ambiguous"] else (ColumnRole.DATE, 0.9)
    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.DIMENSION, 0.8
    if pd.api.types.is_numeric_dtype(series):
        return ColumnRole.MEASURE, 0.8

    # Remaining columns are text (dtype == object). Distinguish a short,
    # structured, fully-unique value (identifier-like, e.g. a SKU or UUID
    # without "id" in its name) from long-form free text that happens to be
    # unique per row (notes, comments) by looking at average string length.
    is_long_text = False
    if len(non_null):
        avg_len = non_null.astype(str).str.len().mean()
        is_long_text = avg_len > _LONG_TEXT_AVG_LEN

    if uniqueness_ratio == 1.0 and row_count > 1 and not is_long_text:
        return ColumnRole.IDENTIFIER, 0.6
    if 0 < distinct_count <= CATEGORICAL_MAX_DISTINCT:
        return ColumnRole.DIMENSION, 0.7
    return ColumnRole.FREE_TEXT, 0.6


def _detect_date_format(non_null: pd.Series) -> dict | None:
    """Try a fixed list of explicit formats (never `infer_datetime_format`,
    which silently guesses). If a value fits two formats that disagree on
    day/month order (e.g. 01/02/2024), refuse to pick one."""
    if non_null.empty:
        return None

    sample = non_null.astype(str).str.strip()
    likely = sample.str.match(_DATE_LIKE_RE)
    if likely.mean() < 0.9:
        return None

    working_formats = []
    for fmt in _DATE_FORMATS:
        try:
            pd.to_datetime(sample, format=fmt, errors="raise")
            working_formats.append(fmt)
        except (ValueError, TypeError):
            continue

    if not working_formats:
        return None

    is_ambiguous = any(a in working_formats and b in working_formats for a, b in _AMBIGUOUS_FORMAT_PAIRS)
    if is_ambiguous:
        return {"format": None, "ambiguous": True, "min": None, "max": None}

    chosen = working_formats[0]
    parsed = pd.to_datetime(sample, format=chosen, errors="coerce")
    return {"format": chosen, "ambiguous": False, "min": str(parsed.min()), "max": str(parsed.max())}


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def save_schema(schema: SchemaContext, schemas_dir: Path) -> Path:
    schemas_dir.mkdir(parents=True, exist_ok=True)
    out_path = schemas_dir / f"{schema.dataset_id}.json"
    out_path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_schema(dataset_id: str, schemas_dir: Path) -> SchemaContext:
    path = schemas_dir / f"{dataset_id}.json"
    return SchemaContext.model_validate_json(path.read_text(encoding="utf-8"))
