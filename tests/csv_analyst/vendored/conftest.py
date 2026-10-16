"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def pandas_engine(monkeypatch):
    """Pin the deprecated pandas engine (and the SQL flags' old defaults) for
    tests whose fake LLM returns pandas code. SQL is the default since
    2026-10-07 (PLAN_v2.md D10); SQL behaviour is tested with engine="sql"."""
    from csv_analyst.config import settings

    monkeypatch.setattr(settings, "query_engine", "pandas")
    monkeypatch.setattr(settings, "sql_real_measures", False)
    monkeypatch.setattr(settings, "sql_drop_null_groups", False)
