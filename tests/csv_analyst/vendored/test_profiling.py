"""Profiler tests: role inference, numeric/categorical stats, date-ambiguity
handling, and schema persistence round-trip."""

from __future__ import annotations

from csv_analyst.ingestion.loader import load_csv
from csv_analyst.profiling.profiler import load_schema, profile_dataset, save_schema
from csv_analyst.profiling.schema import ColumnRole


def _profile(tmp_path, csv_text: str):
    path = tmp_path / "data.csv"
    path.write_text(csv_text, encoding="utf-8")
    result = load_csv(path)
    return profile_dataset(result)


def _col(schema, name):
    return next(c for c in schema.columns if c.name == name)


def test_identifier_role(tmp_path):
    csv_text = "customer_id,region\n1,North\n2,South\n3,East\n4,West\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "customer_id")
    assert col.role == ColumnRole.IDENTIFIER
    assert col.role_confidence >= 0.9


def test_measure_role_and_stats(tmp_path):
    csv_text = "id,revenue\n1,100\n2,200\n3,300\n4,150\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "revenue")
    assert col.role == ColumnRole.MEASURE
    assert col.numeric_min == 100.0
    assert col.numeric_max == 300.0
    assert col.numeric_mean == 187.5


def test_dimension_role_and_top_values(tmp_path):
    csv_text = "id,region\n1,North\n2,South\n3,North\n4,North\n5,South\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "region")
    assert col.role == ColumnRole.DIMENSION
    assert col.top_values is not None
    top = {v.value: v.count for v in col.top_values}
    assert top["North"] == 3
    assert top["South"] == 2


def test_free_text_role(tmp_path):
    rows = "\n".join(f"{i},{'note text that is different every single row ' + str(i)}" for i in range(1, 60))
    csv_text = "id,comment\n" + rows + "\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "comment")
    assert col.role == ColumnRole.FREE_TEXT


def test_unambiguous_date_format(tmp_path):
    csv_text = "id,order_date\n1,2024-01-15\n2,2024-02-20\n3,2024-03-05\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "order_date")
    assert col.role == ColumnRole.DATE
    assert col.date_ambiguous is False
    assert col.date_format == "%Y-%m-%d"
    assert col.date_min is not None
    assert col.date_max is not None


def test_ambiguous_date_format_not_guessed(tmp_path):
    # Every day and month value is <= 12, so %m/%d/%Y and %d/%m/%Y both parse cleanly.
    csv_text = "id,order_date\n1,01/02/2024\n2,03/04/2024\n3,05/06/2024\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "order_date")
    assert col.role == ColumnRole.DATE
    assert col.date_ambiguous is True
    assert col.date_format is None
    assert col.role_confidence <= 0.5


def test_null_stats(tmp_path):
    csv_text = "id,notes\n1,hello\n2,\n3,NA\n4,world\n"
    schema = _profile(tmp_path, csv_text)

    col = _col(schema, "notes")
    assert col.null_count == 2
    assert col.null_pct == 50.0


def test_schema_persistence_roundtrip(tmp_path):
    schema = _profile(tmp_path, "id,region\n1,North\n2,South\n")
    schemas_dir = tmp_path / "schemas"

    out_path = save_schema(schema, schemas_dir)
    assert out_path.exists()

    loaded = load_schema(schema.dataset_id, schemas_dir)
    assert loaded.dataset_id == schema.dataset_id
    assert loaded.row_count == schema.row_count
    assert [c.name for c in loaded.columns] == [c.name for c in schema.columns]


# --- entity levels: which columns have one value per repeating id -----------


def _admissions_csv(n_patients: int = 12, per_patient: int = 3) -> str:
    """One row per admission; name/gender are per-patient, diagnosis per-row."""
    rows = ["patient_id,patient_name,gender,diagnosis,ward"]
    for p in range(1, n_patients + 1):
        for a in range(per_patient):
            gender = "F" if p % 2 else "M"
            rows.append(f"{p},name{p},{gender},dx{(p + a) % 5},W")
    return "\n".join(rows) + "\n"


def test_entity_attributes_found_from_the_data(tmp_path):
    schema = _profile(tmp_path, _admissions_csv())

    # per-patient columns, not the per-row diagnosis, not the file-wide constant ward
    assert schema.entity_attributes == {"patient_id": ["patient_name", "gender"]}
    assert schema.entity_labels == {"patient_id": ["patient_name"]}


def test_entity_attributes_list_every_id_finest_first(tmp_path):
    rows = ["order_id,customer_id,customer_name,region,sales"]
    for o in range(1, 31):
        cust = (o % 6) + 1
        for line in range(2):
            rows.append(f"{o},C{cust},cust{cust},R{o % 4},{o * 10 + line}")
    schema = _profile(tmp_path, "\n".join(rows) + "\n")

    assert list(schema.entity_attributes) == ["order_id", "customer_id"]
    assert schema.entity_attributes["order_id"] == ["customer_id", "customer_name", "region"]
    assert schema.entity_attributes["customer_id"] == ["customer_name"]
    assert schema.entity_labels["customer_id"] == ["customer_name"]


def test_entity_attributes_need_enough_repeated_rows(tmp_path):
    # 3 rows, one repeated pair: too little evidence to call anything an attribute
    csv_text = "order_id,region,revenue\n1,North,100\n1,North,50\n2,South,200\n"
    schema = _profile(tmp_path, csv_text)

    assert schema.entity_attributes == {}


def test_entity_attributes_absent_without_a_repeating_id(tmp_path):
    csv_text = "order_id,region,revenue\n" + "".join(f"{i},R{i % 3},{i}\n" for i in range(1, 40))
    schema = _profile(tmp_path, csv_text)

    assert schema.entity_attributes == {}
