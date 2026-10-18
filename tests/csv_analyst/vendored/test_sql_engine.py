"""SQL engine option (PLAN_v2.md B4): static validation, the sqlite3
authorizer allowlist, result kinds, date handling, and the timeout."""

from __future__ import annotations

import pandas as pd

from csv_analyst.contracts import ResultKind
from csv_analyst.sql_engine import execute_sql, validate_sql


def _df():
    return pd.DataFrame({
        "region": ["North", "South", "North"],
        "revenue": [100.0, 200.0, 150.0],
        "order_date": pd.to_datetime(["2017-01-05", "2017-02-10", "2017-02-11"]),
    })


def test_scalar_result_for_single_value():
    result = execute_sql(_df(), "SELECT AVG(revenue) FROM data")
    assert result.ok
    assert result.kind == ResultKind.SCALAR
    assert result.value == 150.0


def test_table_result():
    result = execute_sql(_df(), "SELECT region, SUM(revenue) AS total FROM data GROUP BY region ORDER BY region")
    assert result.ok
    assert result.kind == ResultKind.DATAFRAME
    assert result.value.to_dict("records") == [
        {"region": "North", "total": 250.0}, {"region": "South", "total": 200.0},
    ]


def test_dates_are_iso_text_without_midnight_time():
    result = execute_sql(_df(), "SELECT MIN(order_date) FROM data")
    assert result.value == "2017-01-05"
    days = execute_sql(_df(), "SELECT julianday(MAX(order_date)) - julianday(MIN(order_date)) FROM data")
    assert days.value == 37.0


def test_window_functions_and_ctes_work():
    sql = (
        "WITH daily AS (SELECT order_date AS d, COUNT(*) AS c FROM data GROUP BY order_date) "
        "SELECT d FROM (SELECT d, c - LAG(c) OVER (ORDER BY d) AS inc FROM daily) "
        "ORDER BY inc DESC LIMIT 1"
    )
    result = execute_sql(_df(), sql)
    assert result.ok, result.error


def test_validate_rejects_non_select_and_multiple_statements():
    assert not validate_sql("DELETE FROM data").ok
    assert not validate_sql("PRAGMA table_info(data)").ok
    assert not validate_sql("SELECT 1; DROP TABLE data").ok
    assert not validate_sql("").ok
    assert validate_sql("SELECT 'a;b' AS x").ok  # semicolon inside a string literal is fine
    assert validate_sql("-- comment\nSELECT 1;").ok


def test_authorizer_blocks_writes_even_without_static_check():
    for sql in ("DELETE FROM data", "CREATE TABLE x(a)", "PRAGMA table_info(data)",
                "ATTACH DATABASE 'x.db' AS x", "SELECT * FROM sqlite_master"):
        result = execute_sql(_df(), sql)
        assert not result.ok, sql


def test_runaway_query_times_out():
    big = pd.DataFrame({"x": range(3000)})
    result = execute_sql(big, "SELECT COUNT(*) FROM data a, data b, data c", timeout_s=1)
    assert not result.ok
    assert "TimeoutError" in result.error


def test_sql_error_is_reported_not_raised():
    result = execute_sql(_df(), "SELECT no_such_column FROM data")
    assert not result.ok
    assert "no such column" in result.error


# --- sql-v2: extra functions, load-once cache, hints, schema rendering ---

def test_extra_math_and_aggregate_functions():
    df = pd.DataFrame({"w": [90.0, 60.0, 80.0], "h": [180, 150, 200]})
    bmi = execute_sql(df, "SELECT ROUND(w / POWER(h * 1.0 / 100, 2), 2) FROM data WHERE h = 180")
    assert bmi.value == 27.78
    assert execute_sql(df, "SELECT SQRT(16)").value == 4.0
    assert execute_sql(df, "SELECT MEDIAN(w) FROM data").value == 80.0
    assert execute_sql(df, "SELECT POWER(NULL, 2)").value is None


def test_database_is_built_once_per_dataframe(monkeypatch):
    from csv_analyst.sql_engine import database

    df = _df()
    calls = []
    real = database._sql_frame
    monkeypatch.setattr(database, "_sql_frame", lambda *a: calls.append(1) or real(*a))
    execute_sql(df, "SELECT COUNT(*) FROM data")
    execute_sql(df, "SELECT SUM(revenue) FROM data")
    assert len(calls) == 1
    execute_sql(_df(), "SELECT COUNT(*) FROM data")  # a different frame is loaded separately
    assert len(calls) == 2


def test_each_query_gets_an_isolated_copy():
    df = _df()
    assert execute_sql(df, "SELECT COUNT(*) FROM data").value == 3
    assert not execute_sql(df, "DELETE FROM data").ok
    assert execute_sql(df, "SELECT COUNT(*) FROM data").value == 3


def test_sql_retry_hints():
    from csv_analyst.sql_engine import sql_retry_hint

    assert "POWER" in sql_retry_hint('OperationalError: unrecognized token: "^"')
    assert "MEDIAN" in sql_retry_hint("OperationalError: no such function: percentile")
    assert "HAVING" in sql_retry_hint("OperationalError: misuse of aggregate function COUNT()")
    assert "subquery" in sql_retry_hint("OperationalError: no such column: total", columns=["sales"])
    assert sql_retry_hint("something unrelated") is None


def _hospital_schema():
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset

    return profile_dataset(load_csv("data/hospital/hospital.csv"))


def test_profile_schema_is_sql_native():
    from csv_analyst.sql_engine.prompt import render_sql_schema

    block = render_sql_schema(_hospital_schema(), "profile")
    assert "- patient_id (INTEGER)" in block
    assert "- admission_date (TEXT date 'YYYY-MM-DD')" in block  # text column holding ISO dates
    assert "Timestamp(" not in block and "datetime64" not in block and "00:00:00" not in block
    assert "drop_duplicates" not in block
    assert "count DISTINCT <that id>" in block


def test_ddl_schema_is_a_create_table_with_facts_as_comments():
    from csv_analyst.sql_engine.prompt import render_sql_schema

    block = render_sql_schema(_hospital_schema(), "ddl")
    assert block.startswith("CREATE TABLE data (")
    assert "  patient_id INTEGER,  -- role=identifier" in block
    assert "id_repeats=true" in block
    assert ");" in block


# --- SQL v3 arms: REAL measure storage, NULL GROUP BY keys dropped ---

def test_real_columns_make_division_match_pandas():
    df = pd.DataFrame({"patient_id": [1, 2], "height": [156, 180]})
    assert execute_sql(df, "SELECT height / 100 FROM data WHERE patient_id = 1").value == 1
    real = execute_sql(df, "SELECT height / 100 FROM data WHERE patient_id = 1", real_columns=["height"])
    assert real.value == 1.56
    assert execute_sql(df, "SELECT typeof(patient_id) FROM data LIMIT 1", real_columns=["height"]).value == "integer"


def test_real_measure_columns_follow_setting_and_role(monkeypatch):
    from csv_analyst.config import settings
    from csv_analyst.sql_engine.prompt import real_measure_columns, render_sql_schema

    schema = _hospital_schema()
    monkeypatch.setattr(settings, "sql_real_measures", False)
    assert real_measure_columns(schema) == []
    monkeypatch.setattr(settings, "sql_real_measures", True)
    assert real_measure_columns(schema) == ["height", "weight"]
    block = render_sql_schema(schema, "profile")
    assert "- height (REAL)" in block and "- patient_id (INTEGER)" in block


def test_null_group_rewrite():
    from csv_analyst.sql_engine.rewrite import drop_null_groups

    out = drop_null_groups("SELECT diagnosis, COUNT(*) AS c FROM data GROUP BY diagnosis", ["diagnosis"])
    assert "NOT diagnosis IS NULL" in out
    by_position = drop_null_groups("SELECT diagnosis AS d, COUNT(*) FROM data WHERE x > 1 GROUP BY 1", ["diagnosis"])
    assert "x > 1 AND NOT diagnosis IS NULL" in by_position
    by_alias = drop_null_groups("SELECT strftime('%Y', d) AS y, COUNT(*) FROM data GROUP BY y", ["d"])
    assert "NOT STRFTIME('%Y', d) IS NULL" in by_alias
    unchanged = [
        "SELECT gender, COUNT(*) FROM data GROUP BY gender",  # no nulls in key
        "SELECT COALESCE(a, 'none') AS k, COUNT(*) FROM data GROUP BY k",  # model handled nulls
        "SELECT a, COUNT(*) FROM data WHERE a IS NOT NULL GROUP BY a",  # already filtered
        "SELECT broken (((",  # unparseable -> left to the executor
    ]
    for sql in unchanged:
        assert drop_null_groups(sql, ["a", "diagnosis"]) == sql


def test_null_group_rewrite_executes_like_pandas_groupby():
    from csv_analyst.sql_engine.rewrite import drop_null_groups

    df = pd.DataFrame({"diagnosis": [None, None, None, "Flu", "Flu", "Cold"]})
    sql = drop_null_groups(
        "SELECT diagnosis, COUNT(*) AS c FROM data GROUP BY diagnosis ORDER BY c DESC LIMIT 1", ["diagnosis"],
    )
    assert execute_sql(df, sql).value.to_dict("records") == [{"diagnosis": "Flu", "c": 2}]


# --- Card H-C C1: planner prompt without examples that mirror the benchmark ---

BENCHMARK_WORDS = [
    "patient", "patients", "admission", "admissions", "admitted", "allergy", "allergies", "province",
    "diagnosis", "hospital", "order_date", "category", "region", "East", "customer", "customers",
    "product", "important",
]


def test_planner_sql_v3_has_no_benchmark_words():
    import re
    from pathlib import Path

    for version in ("planner-sql-v3", "planner-sql-v4"):
        text = Path(f"csv_analyst/llm/prompts/{version}.txt").read_text(encoding="utf-8")
        found = [w for w in BENCHMARK_WORDS if re.search(rf"\b{w}\b", text, flags=re.IGNORECASE)]
        assert found == [], version
        assert "library-loans" in text


def test_planner_prompt_version_is_selectable(monkeypatch):
    from csv_analyst.config import settings
    from csv_analyst.sql_engine.prompt import build_sql_prompt

    assert settings.sql_planner_prompt == "planner-sql-v4-tables"  # default since 2026-10-08 (H-B1)
    assert settings.sql_schema_style == "plain"
    monkeypatch.setattr(settings, "sql_schema_style", "profile")
    monkeypatch.setattr(settings, "sql_generated_examples", False)
    monkeypatch.setattr(settings, "sql_planner_prompt", "planner-sql-v2")
    schema = _hospital_schema()
    v2 = build_sql_prompt("How many rows?", schema)
    assert "How many patients have each allergy?" in v2
    assert "transaction/admission/order" in v2

    monkeypatch.setattr(settings, "sql_planner_prompt", "planner-sql-v3")
    v3 = build_sql_prompt("How many rows?", schema)
    assert "How many members are in each membership tier?" in v3
    assert "one row per loan, not one per member" in v3
    assert "transaction/admission/order" not in v3
    assert v3.rstrip().endswith(">>>")


# --- Card H-G: worked examples generated from the CSV's own schema ---

def test_generated_examples_use_real_columns_and_run():
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset
    from csv_analyst.sql_engine.examples import generate_examples

    for path in ("data/hospital/hospital.csv", "data/sales/sales.csv"):
        loaded = load_csv(path)
        schema = profile_dataset(loaded)
        examples = generate_examples(schema)
        assert 1 <= len(examples) <= 8
        assert len({e["plan"]["code"] for e in examples}) == len(examples)
        for e in examples:
            assert set(e["plan"]["columns_used"]) <= set(loaded.df.columns)
            assert execute_sql(loaded.df, e["plan"]["code"]).ok, e["plan"]["code"]


def test_generated_examples_follow_the_fixed_selection_rules():
    from csv_analyst.sql_engine.examples import generate_examples

    questions = [e["question"] for e in generate_examples(_hospital_schema())]
    assert questions[:4] == [
        "How many patients are there per gender?",  # fewest-distinct dimension attribute
        "What is the average height of patients?",  # first measure attribute
        "How many different patients appear in each doctor_specialty?",  # row-level column
        "How many rows are there for each doctor_specialty?",
    ]


def test_generated_examples_are_added_only_with_the_flag(monkeypatch):
    from csv_analyst.config import settings
    from csv_analyst.sql_engine.examples import HEADER
    from csv_analyst.sql_engine.prompt import build_sql_prompt

    assert settings.sql_generated_examples is True  # default since 2026-10-07
    monkeypatch.setattr(settings, "sql_generated_examples", False)
    schema = _hospital_schema()
    assert HEADER not in build_sql_prompt("How many rows?", schema)

    monkeypatch.setattr(settings, "sql_generated_examples", True)
    prompt = build_sql_prompt("How many rows?", schema)
    assert prompt.index("How many members are in each membership tier?") < prompt.index(HEADER)
    assert prompt.index(HEADER) < prompt.index("Respond with ONLY a JSON object")


def test_no_generated_examples_without_repeating_ids(tmp_path):
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset
    from csv_analyst.sql_engine.examples import examples_block

    path = tmp_path / "flat.csv"
    path.write_text("id,region,revenue\n1,N,10\n2,S,20\n3,N,30\n", encoding="utf-8")
    assert examples_block(profile_dataset(load_csv(path))) == ""


def test_generated_row_text_is_short():
    from csv_analyst.sql_engine.examples import generate_examples

    assert generate_examples(_hospital_schema())[0]["plan"]["row_represents"] == "one row (patient_id repeats)"


# --- Card H-C C2: plain schema block + planner-sql-v4 ---

def test_plain_schema_says_each_fact_once():
    from csv_analyst.sql_engine.prompt import render_sql_schema

    block = render_sql_schema(_hospital_schema(), "plain")
    assert "- patient_id (INTEGER): null=0.0%, repeats (4530 distinct in 6215 rows)" in block
    for gone in ("role=", "values={", "range=[", "id_repeats", "NA-like", "holds ISO text dates",
                 "count DISTINCT <that id>"):
        assert gone not in block
    assert "Cells such as 'NA', 'N/A' or empty were stored as NULL (10849 cells in 8 columns)." in block
    assert "- patient_id (4530 distinct) -> first_name" in block  # entity attribute lists stay
    assert "Row-level columns" in block


def test_planner_sql_v4_explains_grain_and_nulls_once():
    from pathlib import Path

    v3 = Path("csv_analyst/llm/prompts/planner-sql-v3.txt").read_text(encoding="utf-8")
    v4 = Path("csv_analyst/llm/prompts/planner-sql-v4.txt").read_text(encoding="utf-8")
    assert "id_repeats" not in v4 and "marked `repeats`" in v4
    assert "\n20. " in v3 and "\n20. " not in v4
    assert v4.count("COUNT(loan_date)") == 1
    assert "Grain checkpoint: decide answerability first (rules 1-6)." in v4
    assert "include its 1:1 readable label" in v4
    assert len(v4) < len(v3)


def test_planner_prompt_version_reports_the_shipped_sql_prompt():
    from csv_analyst.pipeline import planner_prompt_version

    assert planner_prompt_version("sql") == "planner-sql-v4-tables+gen"
    assert planner_prompt_version("sql", use_schema=False) == "planner-baseline-v1"
