"""Pydantic models for the profiled dataset schema (PLAN.md Phase 1, section 6).

This is the SchemaContext the planner (Phase 2) is grounded in. Role hints
are advisory (`role_confidence`), never asserted as fact.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ColumnRole(str, Enum):
    IDENTIFIER = "identifier"
    DATE = "date"
    MEASURE = "measure"
    DIMENSION = "dimension"
    FREE_TEXT = "free_text"


class ValueCount(BaseModel):
    value: Any
    count: int


class ColumnProfile(BaseModel):
    name: str
    original_name: str
    dtype: str
    null_count: int
    null_pct: float
    distinct_count: int

    role: ColumnRole
    role_confidence: float

    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    numeric_mean: Optional[float] = None
    numeric_std: Optional[float] = None

    sample_values: list[Any] = Field(default_factory=list)
    top_values: Optional[list[ValueCount]] = None

    date_format: Optional[str] = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_ambiguous: bool = False

    # PLAN.md Phase 6B.1: set to the original dtype string ("object") when
    # this column was converted from string to real datetime64 at profiling
    # time; None otherwise.
    converted_from: Optional[str] = None


class SchemaContext(BaseModel):
    dataset_id: str
    source_filename: str
    row_count: int
    column_count: int
    memory_bytes: int

    encoding: str
    delimiter: str

    columns: list[ColumnProfile]
    warnings: list[str] = Field(default_factory=list)

    # Derived from the data (profiler._entity_attributes): repeating
    # identifier -> every other column with exactly one value per value of
    # that id, i.e. the entity's attributes, repeated on each of its rows.
    # Empty when there is no usable repeating identifier.
    entity_attributes: dict[str, list[str]] = Field(default_factory=dict)
    # Repeating identifier -> its attributes with exactly as many distinct
    # values as the id (readable 1:1 names, e.g. customer_id -> customer_name).
    entity_labels: dict[str, list[str]] = Field(default_factory=dict)

    profiled_at: datetime
