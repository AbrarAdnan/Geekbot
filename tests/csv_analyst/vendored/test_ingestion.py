"""Phase 1 exit gate: 8 hostile fixtures plus a handful of baseline guard checks.

BOM file, cp1252 file, semicolon-delimited, duplicate headers, ragged rows,
all-null column, single-column file, and a .xlsx renamed to .csv (rejected).
"""

from __future__ import annotations

import pytest

from csv_analyst.ingestion.loader import load_csv
from csv_analyst.ingestion.validators import IngestionError


def _write(tmp_path, name: str, content: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# --- the 8 hostile fixtures -------------------------------------------------


def test_bom_file(tmp_path):
    content = "id,name\n1,Alice\n2,Bob\n".encode("utf-8-sig")
    path = _write(tmp_path, "bom.csv", content)

    result = load_csv(path)

    assert result.encoding == "utf-8-sig"
    assert list(result.df.columns) == ["id", "name"]
    assert len(result.df) == 2


def test_cp1252_file(tmp_path):
    # 'é' is not valid utf-8 when encoded as cp1252's single byte 0xE9. The exact
    # single-byte codepage a statistical detector lands on for a tiny sample can
    # vary (cp1252 vs a lookalike like cp1250) — what matters is it falls back
    # off utf-8 and still parses the row instead of raising.
    content = "id,city\n1,Montr\xe9al\n".encode("cp1252")
    path = _write(tmp_path, "cp1252.csv", content)

    result = load_csv(path)

    assert result.encoding not in ("utf-8", "utf-8-sig")
    assert len(result.df) == 1
    assert result.df["city"].iloc[0].startswith("Montr")


def test_semicolon_delimited(tmp_path):
    content = "id;name;revenue\n1;Alice;100\n2;Bob;200\n".encode("utf-8")
    path = _write(tmp_path, "semicolon.csv", content)

    result = load_csv(path)

    assert result.delimiter == ";"
    assert list(result.df.columns) == ["id", "name", "revenue"]
    assert len(result.df) == 2


def test_duplicate_headers(tmp_path):
    content = "id,value,value\n1,10,20\n2,30,40\n".encode("utf-8")
    path = _write(tmp_path, "dupe_headers.csv", content)

    result = load_csv(path)

    assert list(result.df.columns) == ["id", "value", "value_1"]
    assert result.column_rename_map.get("value_1") == "value"
    assert len(result.df) == 2


def test_ragged_rows(tmp_path):
    content = "id,name,age\n1,Alice,30\n2,Bob\n3,Carol,40,extra\n".encode("utf-8")
    path = _write(tmp_path, "ragged.csv", content)

    result = load_csv(path)

    assert result.bad_lines_skipped >= 1
    assert any(w.code == "ragged_rows" for w in result.warnings)


def test_all_null_column(tmp_path):
    content = "id,notes\n1,\n2,NA\n3,null\n".encode("utf-8")
    path = _write(tmp_path, "all_null.csv", content)

    result = load_csv(path)

    assert result.df["notes"].isna().all()
    assert result.na_conversion_counts.get("notes") == 3


def test_single_column_file(tmp_path):
    content = "id\n1\n2\n3\n".encode("utf-8")
    path = _write(tmp_path, "single_col.csv", content)

    result = load_csv(path)

    assert list(result.df.columns) == ["id"]
    assert len(result.df) == 3


def test_xlsx_renamed_to_csv_rejected(tmp_path):
    # Real xlsx files are zip archives; the PK signature is enough to detect this.
    content = b"PK\x03\x04" + b"\x00" * 32
    path = _write(tmp_path, "fake.csv", content)

    with pytest.raises(IngestionError) as exc_info:
        load_csv(path)

    assert exc_info.value.code == "not_csv"


# --- baseline guard checks ---------------------------------------------------


def test_file_not_found(tmp_path):
    with pytest.raises(IngestionError) as exc_info:
        load_csv(tmp_path / "missing.csv")
    assert exc_info.value.code == "file_not_found"


def test_bad_extension(tmp_path):
    path = _write(tmp_path, "data.txt", b"id,name\n1,Alice\n")
    with pytest.raises(IngestionError) as exc_info:
        load_csv(path)
    assert exc_info.value.code == "bad_extension"


def test_empty_file(tmp_path):
    path = _write(tmp_path, "empty.csv", b"")
    with pytest.raises(IngestionError) as exc_info:
        load_csv(path)
    assert exc_info.value.code == "empty_file"


def test_file_too_large(tmp_path):
    path = _write(tmp_path, "big.csv", b"id,name\n1,Alice\n")
    with pytest.raises(IngestionError) as exc_info:
        load_csv(path, max_mb=0)
    assert exc_info.value.code == "file_too_large"


def test_zero_rows_header_only(tmp_path):
    path = _write(tmp_path, "header_only.csv", b"id,name\n")
    with pytest.raises(IngestionError) as exc_info:
        load_csv(path)
    assert exc_info.value.code == "zero_rows"


def test_numeric_dtype_coercion(tmp_path):
    content = "id,price\n1,10.5\n2,20.25\n".encode("utf-8")
    path = _write(tmp_path, "numeric.csv", content)

    result = load_csv(path)

    assert str(result.df["price"].dtype).startswith("float")
