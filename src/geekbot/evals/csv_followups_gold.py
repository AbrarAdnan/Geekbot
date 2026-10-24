"""Expected results of the CSV follow-up items, computed with pandas (step C6, PLAN.md §9).

Every recipe is a few lines of pandas over the dataset's own CSV: the expected answer is *derived from
the data*, not copied from a website or from the package's own output (the rule of Gigalogy's
benchmark, kept here). A recipe knows the dataset it needs and how its result is compared:

- ``scalar``: one value (a number, a name, a date as ISO text);
- ``table``: a list of rows; only the **values** are compared, row order and column names do not
  matter, floats may differ by ``tolerance`` (the grader of C3, ``csv_mode.grading``).

The recipes are code on purpose, not text in the item file: an expression stored as a string would have
to be evaluated, and evaluating text from a data file is the thing this project is careful about. Each
recipe was checked on the real datasets for what makes a gold answer unreliable: an empty result, a
tie for "the highest", a question whose meaning depends on which rows count (the questions say it:
"distinct orders (order_id)", "admissions (rows with an admission date)").

pandas is imported here and nowhere in ``csv_followups`` (the part that needs no data), so the command
line stays quick to start.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd  # type: ignore[import-untyped, unused-ignore]

__all__ = ["RECIPES", "Recipe", "gold_entry"]


@dataclass(frozen=True)
class Recipe:
    """How to compute one expected result from the dataset's DataFrame."""

    dataset: str
    compare: str  # "scalar" or "table"
    compute: Callable[[pd.DataFrame], Any]
    tolerance: float = 0.0  # absolute, for floating-point sums and means


def _table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as a list of rows of plain Python values (the grader compares values only)."""
    rows: list[dict[str, Any]] = frame.to_dict("records")
    return rows


def _admissions(df: pd.DataFrame) -> pd.DataFrame:
    """The rows that are admissions: a patient with no admission has one row with no admission date."""
    return df[df["admission_date"].notna()]


def _year(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column).dt.year


RECIPES: Final[dict[str, Recipe]] = {
    # --- sales (Superstore: one row per order line) ---
    "sales_profit_per_region": Recipe(
        "sales",
        "table",
        lambda df: _table(df.groupby("region", as_index=False)["profit"].sum()),
        0.01,
    ),
    "sales_distinct_orders_west_by_segment": Recipe(
        "sales",
        "table",
        lambda df: _table(
            df[df["region"] == "West"].groupby("segment", as_index=False)["order_id"].nunique()
        ),
    ),
    "sales_max_discount_by_ship_mode": Recipe(
        "sales",
        "table",
        lambda df: _table(df.groupby("ship_mode", as_index=False)["discount"].max()),
        0.0001,
    ),
    "sales_lowest_category": Recipe(
        "sales", "scalar", lambda df: str(df.groupby("category")["sales"].sum().idxmin())
    ),
    "sales_profit_texas": Recipe(
        "sales", "scalar", lambda df: float(df.loc[df["state"] == "Texas", "profit"].sum()), 0.01
    ),
    "sales_rows_by_ship_mode_consumer": Recipe(
        "sales",
        "table",
        lambda df: _table(
            df[df["segment"] == "Consumer"].groupby("ship_mode", as_index=False).size()
        ),
    ),
    "sales_quantity_by_category_region": Recipe(
        "sales",
        "table",
        lambda df: _table(df.groupby(["category", "region"], as_index=False)["quantity"].sum()),
    ),
    "sales_median_sales_by_segment": Recipe(
        "sales",
        "table",
        lambda df: _table(df.groupby("segment", as_index=False)["sales"].median()),
        0.01,
    ),
    "sales_total_2016": Recipe(
        "sales",
        "scalar",
        lambda df: float(df.loc[_year(df["order_date"]) == 2016, "sales"].sum()),
        0.01,
    ),
    # --- hospital (one row per admission; a patient with none has one row with empty admission fields) ---
    "hospital_female_patients_by_province": Recipe(
        "hospital",
        "table",
        lambda df: _table(
            df[df["gender"] == "F"].groupby("province_name", as_index=False)["patient_id"].nunique()
        ),
    ),
    "hospital_admissions_2019_by_specialty": Recipe(
        "hospital",
        "table",
        lambda df: _table(
            _admissions(df)[_year(_admissions(df)["admission_date"]) == 2019]
            .groupby("doctor_specialty", as_index=False)
            .size()
        ),
    ),
    "hospital_admissions_for_top_diagnosis": Recipe(
        "hospital", "scalar", lambda df: int(_admissions(df)["diagnosis"].value_counts().iloc[0])
    ),
    "hospital_ontario_admissions_by_gender": Recipe(
        "hospital",
        "table",
        lambda df: _table(
            _admissions(df)[_admissions(df)["province_name"] == "Ontario"]
            .groupby("gender", as_index=False)
            .size()
        ),
    ),
    "hospital_latest_admission": Recipe(
        "hospital", "scalar", lambda df: str(_admissions(df)["admission_date"].max())
    ),
    "hospital_ontario_distinct_cities": Recipe(
        "hospital",
        "scalar",
        lambda df: int(df.loc[df["province_name"] == "Ontario", "city"].nunique()),
    ),
}


def gold_entry(recipe: Recipe, frame: pd.DataFrame) -> dict[str, Any]:
    """The gold entry ``csv_mode.grading.grade`` takes, computed from ``frame``.

    Raises ``ValueError`` if the recipe produces nothing to compare (an empty table): an expected
    result that is empty would make every empty answer "correct".
    """
    value = recipe.compute(frame)
    if recipe.compare == "table" and not value:
        raise ValueError("the recipe produced an empty table; the item cannot be graded")
    return {
        "expected_abstain": False,
        "compare": recipe.compare,
        "tolerance": recipe.tolerance,
        "value": value,
    }
