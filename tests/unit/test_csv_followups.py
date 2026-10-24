"""Tests for step C6: the CSV follow-up evaluation, the parts that need no model (``csv_followups``) and
the expected-result recipes (``csv_followups_gold``) on small made-up tables."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.csv_mode.grading import scalar_close, table_close
from geekbot.evals.csv_followups import (
    DATASETS,
    FollowUpItem,
    FollowUpRun,
    FollowUpsError,
    format_followups_report,
    load_followups,
    note_category,
    run_followups,
    solved,
    summarize_followups,
)
from geekbot.evals.csv_followups_gold import RECIPES, gold_entry

REPOSITORY = Path(__file__).resolve().parents[2]
ITEMS_FILE = REPOSITORY / "evals" / "datasets" / "csv-followups.jsonl"

# --- the item file ---------------------------------------------------------------------------


def line(**overrides: Any) -> str:
    data: dict[str, Any] = {
        "id": "F-001",
        "dataset": "sales",
        "turns": ["Total sales per region?", "And profit?"],
        "standalone": "Total profit per region?",
        "gold": "sales_profit_per_region",
    }
    data.update(overrides)
    return json.dumps(data)


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_shipped_item_file_is_valid_and_every_recipe_is_used_once() -> None:
    items = load_followups(ITEMS_FILE, known_gold=list(RECIPES))
    assert len(items) == 15
    assert len({item.id for item in items}) == 15
    assert {item.dataset for item in items} == set(DATASETS)
    assert sorted(item.gold for item in items) == sorted(RECIPES)  # each recipe has its item
    assert all(RECIPES[item.gold].dataset == item.dataset for item in items)
    assert all(item.standalone.strip() != item.follow_up.strip() for item in items)


def test_a_valid_line_is_read(tmp_path: Path) -> None:
    (item,) = load_followups(
        write(tmp_path, line(note="why")), known_gold=["sales_profit_per_region"]
    )
    assert item == FollowUpItem(
        "F-001",
        "sales",
        ("Total sales per region?", "And profit?"),
        "Total profit per region?",
        "sales_profit_per_region",
        "why",
    )
    assert item.follow_up == "And profit?"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("{not json", "not JSON"),
        ("[1, 2]", "not an object"),
        (line(surprise=1), "unknown field"),
        (line(dataset="weather"), "is not one of"),
        (line(turns=["only one"]), "at least 2"),
        (line(turns=["first", "  "]), "non-empty"),
        (line(standalone="And profit?"), "must be written out and differ"),
        (line(standalone="  "), "must be written out and differ"),
        (line(gold="nothing"), "no expected-result recipe"),
        (json.dumps({"id": "F-001"}), "missing or malformed"),
    ],
)
def test_a_bad_line_is_refused_and_the_message_says_which_line(
    tmp_path: Path, bad: str, message: str
) -> None:
    with pytest.raises(FollowUpsError, match=message) as raised:
        load_followups(write(tmp_path, bad), known_gold=["sales_profit_per_region"])
    assert "items.jsonl line 1" in str(raised.value)


def test_a_repeated_id_and_an_empty_file_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FollowUpsError, match="used twice"):
        load_followups(write(tmp_path, line(), line()))
    with pytest.raises(FollowUpsError, match="holds no items"):
        load_followups(write(tmp_path, "", "  "))


def test_blank_lines_are_skipped_and_gold_is_unchecked_unless_asked(tmp_path: Path) -> None:
    items = load_followups(write(tmp_path, line(), "", line(id="F-002", gold="whatever")))
    assert [item.id for item in items] == ["F-001", "F-002"]


# --- the rule and the categories -------------------------------------------------------------


@pytest.mark.parametrize(
    ("right", "runs", "expected"),
    [
        (3, 3, True),
        (2, 3, True),
        (1, 3, False),
        (0, 3, False),
        (1, 1, True),
        (1, 2, False),
        (2, 4, False),
    ],
)
def test_solved_means_right_in_more_than_half_of_the_runs(
    right: int, runs: int, expected: bool
) -> None:
    assert solved(right, runs) is expected


@given(st.integers(1, 9), st.integers(0, 9))
def test_solved_agrees_with_the_majority(runs: int, right: int) -> None:
    right = min(right, runs)
    assert solved(right, runs) == (right > runs / 2)


@pytest.mark.parametrize(
    ("note", "category"),
    [
        ("matches gold", "matches"),
        ("correctly abstained", "abstained"),
        ("false abstention: status=abstained reason='x'", "false abstention"),
        ("HALLUCINATION: answered a question that should have been refused", "hallucination"),
        ("expected [{'region': 'East', 'sales': 1.5}], got [{'a': 2}]", "mismatch"),
        ("", "mismatch"),
    ],
)
def test_the_graders_note_becomes_a_category_that_holds_no_data(note: str, category: str) -> None:
    assert note_category(note) == category


# --- running and summarising -----------------------------------------------------------------


def make_item(identifier: str, dataset: str = "sales") -> FollowUpItem:
    return FollowUpItem(identifier, dataset, ("q1", "follow up"), f"standalone {identifier}", "g")


def make_run(
    identifier: str,
    seed: int,
    *,
    conversation: bool,
    oracle: bool,
    dataset: str = "sales",
    unchanged: bool = False,
    error: str = "",
    seconds: float = 10.0,
) -> FollowUpRun:
    return FollowUpRun(
        identifier,
        dataset,
        seed,
        conversation,
        "matches" if conversation else "mismatch",
        oracle,
        "matches" if oracle else "mismatch",
        "rewritten" if not unchanged else "follow up",
        unchanged,
        error,
        seconds,
    )


def test_runs_go_seed_set_by_seed_set_and_report_each_result() -> None:
    order: list[tuple[str, int]] = []
    seen: list[tuple[str, int]] = []

    def run_item(item: FollowUpItem, seed: int) -> FollowUpRun:
        order.append((item.id, seed))
        return make_run(item.id, seed, conversation=True, oracle=True)

    results = run_followups(
        [make_item("A"), make_item("B")],
        run_item,
        runs=2,
        on_run=lambda r: seen.append((r.item_id, r.seed_set)),
    )
    assert order == [("A", 0), ("B", 0), ("A", 1), ("B", 1)]
    assert seen == order and len(results) == 4


def test_a_run_that_breaks_is_not_counted_as_a_wrong_answer() -> None:
    def broken(item: FollowUpItem, seed: int) -> FollowUpRun:
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="model crashed"):
        run_followups([make_item("A")], broken, runs=3)


def three(
    identifier: str, conversation: list[bool], oracle: list[bool], **kwargs: Any
) -> list[FollowUpRun]:
    return [
        make_run(identifier, seed, conversation=c, oracle=o, **kwargs)
        for seed, (c, o) in enumerate(zip(conversation, oracle, strict=True))
    ]


def test_every_failure_is_placed_on_the_rewrite_or_on_the_package() -> None:
    items = [make_item(i) for i in "ABCDE"]
    runs = [
        *three("A", [True, True, True], [True, True, True]),  # solved both ways
        *three("B", [False, False, True], [True, True, True]),  # the rewrite lost it
        *three("C", [False, False, False], [False, False, True]),  # the package cannot do it
        *three(
            "D", [True, True, False], [False, False, False]
        ),  # solved only through the conversation
        *three(
            "E", [True, False, True], [True, True, True], dataset="hospital"
        ),  # unstable but solved
    ]
    summary = summarize_followups([*items[:4], make_item("E", "hospital")], runs, runs_per_item=3)
    assert (summary.items, summary.solved_conversation, summary.solved_oracle) == (5, 3, 3)
    assert summary.rewrite_loss == ("B",)
    assert summary.package_failures == ("C",)
    assert summary.gained == ("D",)
    assert set(summary.unstable) == {"B", "C", "D", "E"}  # right in some runs and wrong in others
    assert summary.by_dataset == {"sales": (4, 2, 2), "hospital": (1, 1, 1)}


def test_the_rewrites_habits_and_the_time_are_counted() -> None:
    runs = [
        make_run(
            "A",
            0,
            conversation=True,
            oracle=True,
            unchanged=True,
            error="TimeoutError",
            seconds=8.0,
        ),
        make_run("A", 1, conversation=True, oracle=True, unchanged=True, seconds=12.0),
        make_run("A", 2, conversation=True, oracle=True),
    ]
    summary = summarize_followups([make_item("A")], runs, runs_per_item=3)
    assert (summary.rewrites_unchanged, summary.rewrite_errors) == (2, 1)
    assert summary.seconds_mean == pytest.approx(10.0)


def test_an_item_with_no_runs_is_unsolved_on_both_paths_not_a_crash() -> None:
    summary = summarize_followups([make_item("A")], [], runs_per_item=3)
    assert (summary.solved_conversation, summary.solved_oracle) == (0, 0)
    assert summary.package_failures == ("A",) and summary.seconds_mean == 0.0


def test_the_report_names_the_model_both_counts_and_the_failing_rewrites() -> None:
    items = [make_item("A"), make_item("B")]
    runs = [
        *three("A", [True] * 3, [True] * 3),
        *three("B", [False] * 3, [True] * 3),
    ]
    summary = summarize_followups(items, runs, runs_per_item=3)
    text = format_followups_report(
        {"name": "r", "model": "qwen2.5-coder:7b", "profile": "target", "created": "2026-10-24"},
        items,
        runs,
        summary,
    )
    for wanted in (
        "qwen2.5-coder:7b",
        "**1 of 2**",  # the conversation
        "**2 of 2**",  # the oracle
        "Rewrite loss",
        "(B)",
        "## Failing items",
        "| B | follow up | rewritten | standalone B |",
        "Claude-authored",
    ):
        assert wanted in text, wanted
    assert "| A |" not in text  # a solved item is not listed


def test_a_report_with_no_failures_has_no_failing_section() -> None:
    items = [make_item("A")]
    runs = three("A", [True] * 3, [True] * 3)
    text = format_followups_report(
        {"name": "r"}, items, runs, summarize_followups(items, runs, runs_per_item=3)
    )
    assert "## Failing items" not in text


# --- the expected results, on small made-up tables --------------------------------------------


def sales_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "order_date": ["2016-01-05", "2016-01-05", "2017-02-01", "2016-12-31", "2015-03-03"],
            "ship_mode": ["First Class", "First Class", "Same Day", "Same Day", "First Class"],
            "segment": ["Consumer", "Consumer", "Corporate", "Consumer", "Consumer"],
            "region": ["West", "West", "East", "West", "East"],
            "state": ["Texas", "Texas", "Ohio", "Texas", "Ohio"],
            "category": ["Furniture", "Furniture", "Technology", "Office Supplies", "Technology"],
            "sales": [10.0, 20.0, 100.0, 5.0, 1.0],
            "quantity": [1, 2, 3, 4, 5],
            "discount": [0.0, 0.2, 0.1, 0.5, 0.0],
            "profit": [1.0, -2.0, 30.0, 4.0, 0.5],
        }
    )


def hospital_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3, 3, 4],
            "gender": ["F", "F", "M", "F", "F", "F"],
            "province_name": ["Ontario", "Ontario", "Ontario", "Alberta", "Alberta", "Ontario"],
            "city": ["Barrie", "Barrie", "Dundas", "Calgary", "Calgary", "Barrie"],
            "admission_date": [
                "2018-09-20",
                "2019-03-01",
                None,
                "2019-06-05",
                "2018-01-01",
                "2019-02-02",
            ],
            "diagnosis": ["Flu", "Flu", None, "Flu", "Cold", "Cold"],
            "doctor_specialty": [
                "Cardiologist",
                "Cardiologist",
                None,
                "Neurologist",
                "Neurologist",
                "Cardiologist",
            ],
        }
    )


def test_sales_recipes_compute_what_their_names_say() -> None:
    df = sales_frame()
    profit = {
        row["region"]: row["profit"]
        for row in gold_entry(RECIPES["sales_profit_per_region"], df)["value"]
    }
    assert profit == {"East": pytest.approx(30.5), "West": pytest.approx(3.0)}
    orders = {
        r["segment"]: r["order_id"]
        for r in gold_entry(RECIPES["sales_distinct_orders_west_by_segment"], df)["value"]
    }
    assert orders == {"Consumer": 2}  # o1 counted once, o3 once; the East rows are out
    assert gold_entry(RECIPES["sales_lowest_category"], df)["value"] == "Office Supplies"
    assert gold_entry(RECIPES["sales_profit_texas"], df)["value"] == pytest.approx(3.0)
    assert gold_entry(RECIPES["sales_total_2016"], df)["value"] == pytest.approx(35.0)
    quantity = gold_entry(RECIPES["sales_quantity_by_category_region"], df)["value"]
    assert {(r["category"], r["region"]): r["quantity"] for r in quantity}[
        ("Furniture", "West")
    ] == 3
    ship = {
        r["ship_mode"]: r["size"]
        for r in gold_entry(RECIPES["sales_rows_by_ship_mode_consumer"], df)["value"]
    }
    assert ship == {"First Class": 3, "Same Day": 1}


def test_hospital_recipes_count_admissions_not_rows_of_patients_without_one() -> None:
    df = hospital_frame()
    female = {
        r["province_name"]: r["patient_id"]
        for r in gold_entry(RECIPES["hospital_female_patients_by_province"], df)["value"]
    }
    assert female == {
        "Alberta": 1,
        "Ontario": 2,
    }  # distinct patients: 1 and 4 in Ontario, 3 in Alberta
    in_2019 = {
        r["doctor_specialty"]: r["size"]
        for r in gold_entry(RECIPES["hospital_admissions_2019_by_specialty"], df)["value"]
    }
    assert in_2019 == {"Cardiologist": 2, "Neurologist": 1}
    assert gold_entry(RECIPES["hospital_admissions_for_top_diagnosis"], df)["value"] == 3
    by_gender = {
        r["gender"]: r["size"]
        for r in gold_entry(RECIPES["hospital_ontario_admissions_by_gender"], df)["value"]
    }
    assert by_gender == {"F": 3}  # patient 2 (male) has no admission: that row is not one
    assert gold_entry(RECIPES["hospital_latest_admission"], df)["value"] == "2019-06-05"
    assert gold_entry(RECIPES["hospital_ontario_distinct_cities"], df)["value"] == 2


def test_the_values_are_plain_python_and_compare_with_the_graders_functions() -> None:
    entry = gold_entry(RECIPES["sales_profit_per_region"], sales_frame())
    assert entry["compare"] == "table" and entry["expected_abstain"] is False
    rows = entry["value"]
    assert all(type(v) in (int, float, str) for row in rows for v in row.values())
    renamed = [
        {"x": row["region"], "y": row["profit"]} for row in reversed(rows)
    ]  # row order and column names differ; the values in each column do not
    assert table_close(rows, renamed, entry["tolerance"] or 0)
    assert scalar_close(
        gold_entry(RECIPES["hospital_ontario_distinct_cities"], hospital_frame())["value"], 2, 0
    )


def test_a_recipe_that_finds_nothing_is_an_error_not_a_gold_that_every_empty_answer_matches() -> (
    None
):
    nothing = sales_frame().iloc[0:0]
    with pytest.raises(ValueError, match="empty table"):
        gold_entry(RECIPES["sales_profit_per_region"], nothing)


@pytest.mark.skipif(
    not (REPOSITORY / "scratch" / "gigalogy-latest" / "data" / "sales" / "sales.csv").is_file(),
    reason="needs Gigalogy's datasets (not vendored)",
)
def test_every_recipe_works_on_the_real_datasets_and_is_unambiguous() -> None:
    base = REPOSITORY / "scratch" / "gigalogy-latest" / "data"
    frames = {
        "sales": pd.read_csv(base / "sales" / "sales.csv"),
        "hospital": pd.read_csv(base / "hospital" / "hospital.csv"),
    }
    for key, recipe in RECIPES.items():
        entry = gold_entry(recipe, frames[recipe.dataset])
        assert entry["value"] not in ([], None, ""), key
    top = frames["hospital"]["diagnosis"].value_counts()
    assert top.iloc[0] > top.iloc[1]  # no tie for "the most common diagnosis"
    totals = frames["sales"].groupby("category")["sales"].sum().sort_values()
    assert totals.iloc[0] < totals.iloc[1]  # no tie for "the lowest category"
