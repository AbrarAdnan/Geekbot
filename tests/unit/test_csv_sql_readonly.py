"""Tests for step C2: the CSV package's SQL path cannot write (PLAN.md §8.1, D-041).

This is the one security property claimed for CSV mode, and it is claimed only as far as this
file proves it: a statement that writes, attaches another database, runs a PRAGMA, opens a
transaction or reads the schema is refused by SQLite's own authorizer, and nothing reaches the
disk. Everything else about the SQL checks is "prototype-grade" (INTEGRATION.md).

``execute_sql`` is called directly on purpose. It skips ``validate_sql`` (the static check), so a
refusal here comes from the authorizer, which is the real enforcement. What is *not* tested, and
so not claimed: the table-valued ``pragma_*()`` functions, memory or CPU use beyond the query
time limit, and anything a model might do through the allowed SELECT functions.
"""

from pathlib import Path

import pandas as pd
import pytest

from csv_analyst.sql_engine import execute_sql, validate_sql

DF = pd.DataFrame({"region": ["North", "South", "North"], "revenue": [100.0, 200.0, 150.0]})

REFUSED_BY_THE_AUTHORIZER = {
    "insert": "INSERT INTO data VALUES ('East', 1.0)",
    "insert behind a WITH": (
        "WITH x AS (SELECT 'East' AS region, 1.0 AS revenue) INSERT INTO data SELECT * FROM x"
    ),
    "replace": "REPLACE INTO data VALUES ('East', 1.0)",
    "update": "UPDATE data SET revenue = 0",
    "delete": "DELETE FROM data",
    "drop table": "DROP TABLE data",
    "create table": "CREATE TABLE evil (a)",
    "create view": "CREATE VIEW evil AS SELECT * FROM data",
    "alter table": "ALTER TABLE data ADD COLUMN extra",
    "attach": "ATTACH DATABASE 'evil.db' AS evil",
    "pragma": "PRAGMA table_info(data)",
    "pragma that writes": "PRAGMA writable_schema = 1",
    "transaction": "BEGIN",
    "read the schema": "SELECT * FROM sqlite_master",
}


# SQLite words an authorizer refusal in two ways (seen on the first run): "not authorized" when a
# whole statement is denied, and "access to <column> is prohibited" when reading a column is
# denied, which is how the read of `sqlite_master` is refused.
AUTHORIZER_MESSAGES = ("not authorized", "is prohibited")


@pytest.mark.parametrize(
    "sql", REFUSED_BY_THE_AUTHORIZER.values(), ids=REFUSED_BY_THE_AUTHORIZER.keys()
)
def test_statement_is_refused_by_the_authorizer(sql: str) -> None:
    result = execute_sql(DF, sql)
    assert not result.ok
    assert result.error is not None
    assert any(message in result.error for message in AUTHORIZER_MESSAGES), result.error


def test_attach_does_not_create_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # a relative ATTACH path would land here
    result = execute_sql(DF, "ATTACH DATABASE 'evil.db' AS evil")
    assert not result.ok
    assert list(tmp_path.iterdir()) == []


def test_two_statements_in_one_string_are_refused() -> None:
    sql = "SELECT 1; DROP TABLE data"
    assert not execute_sql(DF, sql).ok  # SQLite itself runs one statement at a time
    assert not validate_sql(sql).ok  # and the static check says so first


def test_reading_the_csv_still_works() -> None:
    """The control: without it, every test above would also pass if SQL were simply broken."""
    count = execute_sql(DF, "SELECT COUNT(*) FROM data")
    assert count.ok
    assert count.value == 3
    with_clause = execute_sql(
        DF,
        "WITH north AS (SELECT revenue FROM data WHERE region = 'North') "
        "SELECT SUM(revenue) FROM north",
    )
    assert with_clause.ok
    assert with_clause.value == 250.0
