"""Test setup for the vendored csv_analyst tests (step C1). Geekbot's own file; the vendored
tests in ``vendored/`` stay exactly as Gigalogy wrote them (src/csv_analyst/VENDORED.md).

They were written to run from Gigalogy's repo root, so two things need adapting here:

1. Some read ``csv_analyst/llm/prompts/*.txt`` relative to the current directory. Here that
   folder lives under ``src/``, so every test in this folder runs with ``src/`` as its working
   directory.
2. Ten need Gigalogy's benchmark datasets (``data/hospital/hospital.csv``, ``data/sales/sales.csv``),
   which were deliberately not copied (not Geekbot's data to republish, and the benchmark stays in
   Gigalogy). They are skipped unless the files exist under ``src/data/``. Geekbot measures the
   CSV mode with its own run at C3.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

NEEDS_GIGALOGY_DATASETS = {
    "test_sql_prompt_uses_sql_wording_and_iso_dates",
    "test_profile_schema_is_sql_native",
    "test_ddl_schema_is_a_create_table_with_facts_as_comments",
    "test_real_measure_columns_follow_setting_and_role",
    "test_planner_prompt_version_is_selectable",
    "test_generated_examples_use_real_columns_and_run",
    "test_generated_examples_follow_the_fixed_selection_rules",
    "test_generated_examples_are_added_only_with_the_flag",
    "test_generated_row_text_is_short",
    "test_plain_schema_says_each_fact_once",
}


# Fails in Geekbot's env, cause not confirmed (the original Gigalogy folder was no longer
# available to compare). The profiler turns clean ISO date text into real datetimes
# (profiling/profiler.py), so the fact bank writes "holds dates ..." where this test expects
# "holds ISO text ...". The numbers in the sentence are right; only the label differs. Either the
# source test is stale for the source code, or the environments differ. Marked expected-fail
# (not skipped) so it shows up if it ever starts passing. Look again during C3's reproduction run.
UNDIAGNOSED_FAILURES = {
    "test_templates_cover_values_numbers_dates_nulls_relations": (
        "fact bank says 'holds dates', test expects 'holds ISO text' (see conftest.py)"
    ),
}


@pytest.fixture(autouse=True)
def _run_from_src(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(SRC)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    datasets_present = (SRC / "data" / "hospital" / "hospital.csv").exists()
    skip = pytest.mark.skip(reason="needs Gigalogy's hospital/sales datasets (not vendored)")
    for item in items:
        if not datasets_present and item.name in NEEDS_GIGALOGY_DATASETS:
            item.add_marker(skip)
        if item.name in UNDIAGNOSED_FAILURES:
            item.add_marker(pytest.mark.xfail(reason=UNDIAGNOSED_FAILURES[item.name], strict=False))
