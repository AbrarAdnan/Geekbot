"""Card H-R (PLAN_v2.md): fact bank templates, selection modes, prompt placement."""

from __future__ import annotations

import pytest

from csv_analyst.generation.planner import build_prompt
from csv_analyst.ingestion.loader import load_csv
from csv_analyst.profiling import fact_index
from csv_analyst.profiling.facts import build_fact_bank
from csv_analyst.profiling.profiler import profile_dataset


@pytest.fixture
def visits(tmp_path):
    path = tmp_path / "visits.csv"
    rows = ["patient_id,first_name,last_name,region,area,visit_date,leave_date,doctor,cost"]
    data = [
        (1, "Ann", "Lee", "North", "N1", "2024-01-01", "2024-01-02", "Kim", 10.0),
        (1, "Ann", "Lee", "North", "N1", "2024-02-01", "2024-01-30", "Kim", 0.0),
        (2, "Ann", "Lee", "South", "S1", "", "", "", 5.0),
        (3, "Bob", "Ray", "South", "S2", "2023-05-05", "2023-05-06", "Roe", -2.0),
        (4, "Cid", "Fox", "North", "N2", "", "", "", 7.5),
    ]
    rows += [",".join(str(v) for v in r) for r in data]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    lr = load_csv(path)
    return lr.df, profile_dataset(lr)


def _texts(df, schema):
    return [f.text for f in build_fact_bank(df, schema)]


def test_templates_cover_values_numbers_dates_nulls_relations(visits):
    df, schema = visits
    texts = _texts(df, schema)
    joined = "\n".join(texts)
    assert "Column `region` has exactly 2 distinct values" in joined
    assert "1 rows are 0" in joined and "1 rows are negative" in joined
    assert any(t.startswith("Column `visit_date` holds ISO text") and "2023: 1, 2024: 2" in t for t in texts)
    assert any("`visit_date`, `leave_date`, `doctor` are null on exactly the same 2 rows" in t for t in texts)
    assert any("`visit_date` is earlier on 2 rows, equal on 0, later on 1" in t for t in texts)
    assert "Each `area` value belongs to exactly one `region` value (many-to-one)." in texts
    assert "No two rows are identical across all columns." in texts


def test_label_collision_counts_names_shared_by_different_ids(visits):
    from csv_analyst.profiling.schema import ColumnRole

    df, schema = visits
    # Too few rows for the profiler's evidence floor -> set the verified
    # entity attributes the template reads.
    schema.entity_attributes = {"patient_id": ["first_name", "last_name"]}
    for col in schema.columns:
        if col.name in ("first_name", "last_name"):
            col.role = ColumnRole.FREE_TEXT
        if col.name == "patient_id":
            col.role = ColumnRole.IDENTIFIER
            col.distinct_count = 4
    joined = "\n".join(_texts(df, schema))
    assert "1 combinations belong to more than one `patient_id`" in joined


def test_block_modes_and_prompt_placement(visits):
    df, schema = visits
    bank = build_fact_bank(df, schema)
    fact_index.clear_cache()
    block = fact_index.facts_block(schema, df, "all")
    assert block.splitlines() == [fact_index.HEADER] + [f"- {f.text}" for f in bank]
    assert fact_index.facts_block(schema, df, "off") == ""
    with pytest.raises(ValueError):
        fact_index.facts_block(schema, df, "rag")  # retrieval removed after H-R

    prompt = build_prompt("cost by region?", schema, engine="sql", extra_context=block)
    assert prompt.index("Rows: 5") < prompt.index(fact_index.HEADER) < prompt.index("Rules:")
    assert "patient_id" in prompt  # full column list still present
    fact_index.clear_cache()


def test_block_is_capped_for_wide_csvs(visits, monkeypatch):
    df, schema = visits
    fact_index.clear_cache()
    full = fact_index.facts_block(schema, df, "all")
    assert "more facts left out" not in full
    monkeypatch.setattr(fact_index, "MAX_BLOCK_CHARS", len(fact_index.HEADER) + 120)
    capped = fact_index.facts_block(schema, df, "all")
    assert len(capped) < len(full) and capped.rstrip().endswith("more facts left out to fit the prompt.)")
    fact_index.clear_cache()


def test_fact_bank_is_the_shipped_default():
    from csv_analyst.config import Settings

    assert Settings.model_fields["schema_facts"].default == "all"  # H-R kept, PLAN_v2.md D12
