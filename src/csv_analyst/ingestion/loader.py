"""Guarded CSV -> DataFrame loader (PLAN.md Phase 1).

`load_csv` is the single entry point production code should call. It never
raises a bare pandas/OS exception — failures are always an `IngestionError`
with a machine-readable code, and non-fatal issues are collected as
`LoadWarning`s on the returned `LoadResult`.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csv_analyst.config import settings
from csv_analyst.ingestion.validators import (
    IngestionError,
    LoadWarning,
    check_binary_signature,
    check_extension,
    check_file_exists,
    check_file_size,
    detect_encoding,
    is_na_sentinel,
    sniff_delimiter,
)

SAMPLE_BYTES_FOR_SNIFF = 64 * 1024


@dataclass
class LoadResult:
    df: pd.DataFrame
    source_path: Path
    encoding: str
    delimiter: str
    warnings: list[LoadWarning] = field(default_factory=list)
    column_rename_map: dict[str, str] = field(default_factory=dict)
    bad_lines_skipped: int = 0
    na_conversion_counts: dict[str, int] = field(default_factory=dict)


def load_csv_bytes(data: bytes, filename: str = "upload.csv", max_mb: int | None = None) -> LoadResult:
    """`load_csv` for an upload held in memory (API, host chat apps). The guards read a real file,
    so the bytes go to a temporary file; only the upload's suffix is used, never its name."""
    suffix = Path(filename or "upload.csv").suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return load_csv(tmp_path, max_mb=max_mb)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_csv(path: str | Path, max_mb: int | None = None) -> LoadResult:
    path = Path(path)
    max_mb = settings.max_csv_mb if max_mb is None else max_mb

    check_file_exists(path)
    check_extension(path)
    check_file_size(path, max_mb)
    check_binary_signature(path)

    encoding = detect_encoding(path)

    with open(path, "r", encoding=encoding, errors="strict") as f:
        sample = f.read(SAMPLE_BYTES_FOR_SNIFF)
    delimiter = sniff_delimiter(sample)

    raw_header = _read_raw_header(path, encoding, delimiter)
    if not raw_header:
        raise IngestionError("zero_columns", "CSV has no header row / zero columns")
    final_columns, column_rename_map = _normalize_columns(raw_header)

    bad_lines: list[Any] = []

    def _on_bad_line(bad_line):
        bad_lines.append(bad_line)
        return None

    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            engine="python",
            header=0,
            names=final_columns,
            on_bad_lines=_on_bad_line,
            keep_default_na=False,
            na_values=[],
            dtype=str,
        )
    except pd.errors.EmptyDataError as exc:
        raise IngestionError("empty_file", "CSV has no columns to parse") from exc
    except pd.errors.ParserError as exc:
        raise IngestionError("parse_error", f"Failed to parse CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise IngestionError("encoding_error", f"Failed to decode file as {encoding}: {exc}") from exc

    warnings: list[LoadWarning] = []

    if df.shape[1] == 0:
        raise IngestionError("zero_columns", "CSV has zero columns")
    if df.shape[0] == 0:
        raise IngestionError("zero_rows", "CSV has zero data rows (header only)")

    if bad_lines:
        warnings.append(
            LoadWarning("ragged_rows", f"{len(bad_lines)} malformed row(s) were skipped during parsing")
        )

    if column_rename_map:
        warnings.append(
            LoadWarning(
                "column_names_normalized",
                f"{len(column_rename_map)} column name(s) were renamed (blank/duplicate headers)",
            )
        )

    na_counts = _apply_na_sentinels(df)
    if na_counts:
        total = sum(na_counts.values())
        warnings.append(
            LoadWarning(
                "na_sentinels_converted",
                f"{total} NA-like sentinel value(s) converted to null across {len(na_counts)} column(s)",
            )
        )

    _coerce_dtypes(df)

    return LoadResult(
        df=df,
        source_path=path,
        encoding=encoding,
        delimiter=delimiter,
        warnings=warnings,
        column_rename_map=column_rename_map,
        bad_lines_skipped=len(bad_lines),
        na_conversion_counts=na_counts,
    )


def _read_raw_header(path: Path, encoding: str, delimiter: str) -> list[str]:
    """Read the literal header row before pandas gets a chance to silently
    mangle duplicate names itself, so de-duplication is fully our decision."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return next(reader, [])


def _normalize_columns(raw_columns: list[str]) -> tuple[list[str], dict[str, str]]:
    """De-duplicate and fill in blank column headers. Returns the final
    column list and a map of final name -> original raw name for every
    column that was changed."""
    rename_map: dict[str, str] = {}
    seen: dict[str, int] = {}
    new_cols: list[str] = []

    for i, raw in enumerate(raw_columns):
        name = raw.strip()
        if not name:
            name = f"col_{i + 1}"

        if name in seen:
            seen[name] += 1
            final = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
            final = name

        if final != raw:
            rename_map[final] = raw
        new_cols.append(final)

    return new_cols, rename_map


def _apply_na_sentinels(df: pd.DataFrame) -> dict[str, int]:
    """Replace NA-like string sentinels with real nulls, per column, and
    report the conversion count per column."""
    counts: dict[str, int] = {}
    for col in df.columns:
        mask = df[col].map(is_na_sentinel)
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            counts[col] = n
    return counts


def _coerce_dtypes(df: pd.DataFrame) -> None:
    """Every column is loaded as string (dtype=str) to keep NA-sentinel
    handling exact and predictable. This converts columns that are cleanly
    numeric or boolean back to their natural dtype; anything that doesn't
    convert cleanly stays as text. Datetime parsing is deliberately left to
    the profiler, which refuses to guess on ambiguous formats."""
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            df[col] = pd.to_numeric(series, errors="coerce")
            continue

        lowered = non_null.str.strip().str.lower()
        if lowered.isin({"true", "false"}).all():
            df[col] = series.str.strip().str.lower().map({"true": True, "false": False})
