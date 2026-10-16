"""Card H-B1: entity tables for the SQL engine (one row per repeating id)."""

from __future__ import annotations

import pandas as pd

from csv_analyst.config import settings
from csv_analyst.sql_engine import execute_sql
from csv_analyst.sql_engine.entities import entity_tables, table_name, tables_block


def _orders_schema(tmp_path):
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset
    rows = []  # 40 orders, 1-4 lines each (enough repeated rows for the profiler's evidence floor)
    for order_id in range(1, 41):
        region, items, total = ("East", "West")[order_id % 2], 1 + order_id % 4, 10.0 * order_id
        rows += [{"order_id": order_id, "region": region, "order_total": total, "qty": 1,
                  "item": f"item{order_id}-{k}", "price": round(total / items, 2)} for k in range(items)]
    path = tmp_path / "orders.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    loaded = load_csv(str(path))
    return loaded.df, profile_dataset(loaded)


def test_table_names():
    taken = {"data"}
    assert table_name("invoice_id", taken) == "invoices"
    assert table_name("support_rep_id", taken) == "support_reps"
    assert table_name("city_id", taken) == "cities"
    assert table_name("box_id", taken) == "boxes"
    assert table_name("sku", taken) == "sku_entity"
    assert table_name("invoice_id", {"data", "invoices"}) == "invoice_id_entity"


def test_one_row_per_id_and_constant_columns_left_out(tmp_path):
    df, schema = _orders_schema(tmp_path)
    tables = entity_tables(schema)
    orders = next(t for t in tables if t.id_name == "order_id")
    assert orders.name == "orders" and orders.rows == 40
    assert "qty" not in orders.columns and "item" not in orders.columns
    assert {"region", "order_total"} <= set(orders.columns)
    assert "orders: one row per order_id (40 rows)" in tables_block(tables)

    # The grain trap: summing the order total over `data` counts each order once per line.
    on_data = execute_sql(df, "SELECT SUM(order_total) FROM data", tables=tables).value
    on_table = execute_sql(df, "SELECT SUM(order_total) FROM orders", tables=tables).value
    assert on_table == 8200.0 and on_data > on_table
    per_region = execute_sql(df, "SELECT region, COUNT(*) AS n FROM orders GROUP BY region ORDER BY region",
                             tables=tables).value.values.tolist()
    assert per_region == [["East", 20], ["West", 20]]


def test_entity_tables_are_read_only_and_absent_by_default(tmp_path):
    df, schema = _orders_schema(tmp_path)
    tables = entity_tables(schema)
    assert not execute_sql(df, "SELECT COUNT(*) FROM orders").ok  # not created without `tables`
    for sql in ("DELETE FROM orders", "SELECT * FROM sqlite_master", "DROP TABLE orders"):
        assert not execute_sql(df, sql, tables=tables).ok, sql


def test_prompt_and_examples_use_the_tables_only_with_the_flag(tmp_path, monkeypatch):
    from csv_analyst.sql_engine.prompt import build_sql_prompt
    _, schema = _orders_schema(tmp_path)
    monkeypatch.setattr(settings, "sql_generated_examples", True)
    monkeypatch.setattr(settings, "sql_entity_tables", False)
    monkeypatch.setattr(settings, "sql_planner_prompt", "planner-sql-v4")
    plain = build_sql_prompt("What is the total order value?", schema)
    assert "Entity tables" not in plain and "FROM orders" not in plain
    monkeypatch.setattr(settings, "sql_entity_tables", True)
    monkeypatch.setattr(settings, "sql_planner_prompt", "planner-sql-v4-tables")
    with_tables = build_sql_prompt("What is the total order value?", schema)
    assert "Entity tables" in with_tables
    assert "SELECT AVG(order_total) AS avg_order_total FROM orders" in with_tables
    assert "never from `data`, where it repeats" in with_tables


def test_entity_tables_are_on_by_default():  # D28, 2026-10-08
    assert settings.sql_entity_tables is True
    assert settings.sql_planner_prompt == "planner-sql-v4-tables"


# --- H-B1 fixes ---

def test_constant_columns_kept_only_when_named_after_the_entity(tmp_path):
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset
    df, _ = _orders_schema(tmp_path)
    # Constant where present, NULL on rows without an order (like Chinook's
    # unsold-track rows) -- that is how a constant reaches the entity lists.
    df = df.assign(order_status="closed")
    extra = pd.DataFrame({"item": [f"catalog-{k}" for k in range(5)]})
    df = pd.concat([df, extra], ignore_index=True)
    path = tmp_path / "orders2.csv"
    df.to_csv(path, index=False)
    schema = profile_dataset(load_csv(str(path)))
    off = next(t for t in entity_tables(schema, "off") if t.id_name == "order_id")
    prefix = next(t for t in entity_tables(schema, "prefix") if t.id_name == "order_id")
    assert "order_status" not in off.columns and "qty" not in off.columns
    assert "order_status" in prefix.columns and "qty" not in prefix.columns


def test_constant_columns_by_prefix_on_by_default():  # D29, 2026-10-08
    assert settings.sql_entity_constants == "prefix"
