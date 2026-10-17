"""Tests for step C3: the grader ported from Gigalogy (``geekbot.csv_mode.grading``).

The cases are Gigalogy's own grader tests (``tests/test_evaluator_grading.py`` and
``tests/test_comparator_coercion.py`` in the source project), carried over so the port is held
to the same behaviour. The records are made by hand: no model and no GPU.
"""

import datetime
from typing import Any

import pandas as pd

from csv_analyst.contracts import (
    Answerability,
    ConsensusVerdict,
    NumericFidelity,
    Reliability,
    ResultKind,
    ResultPayload,
    RunRecord,
    Status,
)
from geekbot.csv_mode.adapter import CsvAnswer
from geekbot.csv_mode.grading import grade, scalar_close, table_close

RELIABILITY = Reliability(
    samples=3,
    samples_executed=3,
    consensus=ConsensusVerdict.UNANIMOUS,
    codes_identical=False,
    retries=0,
    numeric_fidelity=NumericFidelity.PASS,
)


def _wrap(record: RunRecord) -> CsvAnswer:
    return CsvAnswer(record=record, markdown="", model="m", seconds=0.1, model_load_ms=0)


def _answered(result: ResultPayload | None) -> CsvAnswer:
    return _wrap(
        RunRecord(
            run_id="r",
            question="q",
            dataset_id="d",
            status=Status.ANSWERED,
            answerability=Answerability.ANSWERABLE,
            answer="some answer",
            result=result,
            reliability=RELIABILITY,
        )
    )


def _abstained(
    status: Status = Status.ABSTAINED, answerability: Answerability = Answerability.DATA_MISSING
) -> CsvAnswer:
    return _wrap(
        RunRecord(
            run_id="r",
            question="q",
            dataset_id="d",
            status=status,
            answerability=answerability,
            reason="no salary column",
        )
    )


def _scalar(value: Any) -> ResultPayload:
    return ResultPayload(kind=ResultKind.SCALAR, shape=[1], columns=[], preview=[[value]])


def _table(columns: list[str], rows: list[list[Any]]) -> ResultPayload:
    return ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[len(rows), len(columns)], columns=columns, preview=rows
    )


def _gold(value: Any, *, compare: str = "scalar", tolerance: float = 0) -> dict[str, Any]:
    return {"expected_abstain": False, "compare": compare, "tolerance": tolerance, "value": value}


# --- scalars and tables ---------------------------------------------------------------------


def test_scalar_match_is_correct() -> None:
    correct, note = grade(_answered(_scalar(5067)), _gold(5067))
    assert correct, note


def test_scalar_mismatch_is_wrong_and_the_note_shows_both_values() -> None:
    correct, note = grade(_answered(_scalar(42)), _gold(5067))
    assert not correct
    assert "42" in note
    assert "5067" in note


def test_scalar_within_tolerance_is_correct() -> None:
    assert grade(_answered(_scalar(54.4801)), _gold(54.48, tolerance=0.01))[0]


def test_table_match_is_correct_regardless_of_row_order() -> None:
    result = _table(["province_name"], [["Ontario"], ["Alberta"]])
    gold = _gold([{"province_name": "Alberta"}, {"province_name": "Ontario"}], compare="table")
    correct, note = grade(_answered(result), gold)
    assert correct, note


def test_table_match_ignores_column_names_but_not_values() -> None:
    result = _table(["doctor_last_name", "total_admissions"], [["Walls", 214]])
    gold = _gold([{"doctor_last_name": "Walls", "admission_count": 214}], compare="table")
    correct, note = grade(_answered(result), gold)
    assert correct, note


def test_table_with_the_right_columns_but_wrong_values_is_wrong() -> None:
    result = _table(["city", "patient_count"], [["Hamilton", 2688]])
    gold = _gold([{"city": "Hamilton", "patient_count": 1938}], compare="table")
    assert not grade(_answered(result), gold)[0]


def test_table_with_the_wrong_row_count_is_wrong() -> None:
    result = _table(["province_name"], [["Ontario"]])
    gold = _gold([{"province_name": "Alberta"}, {"province_name": "Ontario"}], compare="table")
    assert not grade(_answered(result), gold)[0]


def test_table_with_an_extra_column_is_wrong() -> None:
    result = _table(["province_name", "gender_diff"], [["Alberta", 11], ["Ontario", 5]])
    gold = _gold([{"province_name": "Alberta"}, {"province_name": "Ontario"}], compare="table")
    assert not grade(_answered(result), gold)[0]


def test_table_with_a_duplicated_row_is_wrong() -> None:
    result = _table(
        ["patient_id", "first_name"], [[496, "Spiros"], [496, "Spiros"], [629, "Spiros"]]
    )
    gold = _gold(
        [{"patient_id": 496, "first_name": "Spiros"}, {"patient_id": 629, "first_name": "Spiros"}],
        compare="table",
    )
    assert not grade(_answered(result), gold)[0]


def test_table_with_an_id_instead_of_the_label_is_wrong() -> None:
    result = _table(["customer_id", "total_sales"], [["SM-20320", 25043.05]])
    gold = _gold(
        [{"customer_name": "Sean Miller", "total_sales": 25043.05}], compare="table", tolerance=0.01
    )
    assert not grade(_answered(result), gold)[0]


def test_a_table_gold_with_no_result_at_all_is_wrong_not_a_crash() -> None:
    gold = _gold([{"x": 1}], compare="table")
    assert not grade(_answered(None), gold)[0]


# --- one-row tables count as scalars; many rows never do ------------------------------------


def test_a_one_row_table_holding_the_label_counts_as_a_scalar() -> None:
    result = _table(["category", "product_id"], [["Office Supplies", 6026]])
    correct, note = grade(_answered(result), _gold("Office Supplies"))
    assert correct, note


def test_a_one_row_table_holding_the_number_counts_as_a_scalar() -> None:
    result = _table(["state", "profit"], [["California", 76381.387]])
    correct, note = grade(_answered(result), _gold(76381.39, tolerance=0.01))
    assert correct, note


def test_a_one_row_table_with_no_matching_value_is_wrong() -> None:
    result = _table(["category", "product_id"], [["Furniture", 2121]])
    assert not grade(_answered(result), _gold("Office Supplies"))[0]


def test_a_many_row_table_is_not_squeezed_into_a_scalar() -> None:
    result = _table(["index", "0"], [["order_id", 0], ["customer_id", 0], ["sales", 0]])
    assert not grade(_answered(result), _gold(0))[0]


# --- abstention -----------------------------------------------------------------------------


def test_refusing_a_question_that_should_be_refused_is_correct() -> None:
    gold = {"expected_abstain": True, "abstain_reason": "column_absent"}
    correct, note = grade(_abstained(), gold)
    assert correct, note


def test_answering_a_question_that_should_be_refused_is_a_hallucination() -> None:
    gold = {"expected_abstain": True, "abstain_reason": "column_absent"}
    correct, note = grade(_answered(_scalar(99000)), gold)
    assert not correct
    assert "HALLUCINATION" in note


def test_a_failed_run_on_a_question_that_should_be_refused_still_counts_as_not_answering() -> None:
    gold = {"expected_abstain": True, "abstain_reason": "column_absent"}
    assert grade(_abstained(status=Status.FAILED), gold)[0]


def test_refusing_an_answerable_question_is_a_false_abstention() -> None:
    correct, note = grade(_abstained(answerability=Answerability.UNSUPPORTED), _gold(5067))
    assert not correct
    assert "false abstention" in note


# --- the comparator -------------------------------------------------------------------------


def test_an_int_and_numeric_text_match_either_way_round() -> None:
    assert scalar_close(1918, "1918", 0)
    assert scalar_close("1918", 1918, 0)


def test_a_float_and_numeric_text_match_within_tolerance() -> None:
    assert scalar_close(125.0, "125.0", 1e-9)


def test_a_date_and_an_iso_date_text_match_whatever_the_date_type() -> None:
    for date in (
        datetime.date(2018, 9, 9),
        datetime.datetime(2018, 9, 9),
        pd.Timestamp("2018-09-09"),
    ):
        assert scalar_close("2018-09-09", date, 0)
        assert scalar_close(date, "2018-09-09", 0)


def test_a_different_date_is_still_a_different_date() -> None:
    assert not scalar_close(pd.Timestamp("2018-09-08"), "2018-09-09", 0)
    assert not scalar_close(datetime.date(2018, 9, 8), "2018-09-09", 0)


def test_text_that_is_not_a_number_is_not_coerced_into_matching_one() -> None:
    assert not scalar_close(1918, "not a number", 0)


def test_text_that_is_not_a_date_does_not_match_a_date() -> None:
    assert not scalar_close(datetime.date(2018, 9, 9), "yesterday", 0)


def test_different_strings_disagree() -> None:
    assert not scalar_close("North", "South", 0)


def test_the_table_comparison_coerces_each_cell_the_same_way() -> None:
    gold = [{"birth_year": 1918}, {"birth_year": 1923}]
    assert table_close(gold, [{"birth_date": "1918"}, {"birth_date": "1923"}], 0)


def test_rows_of_different_lengths_are_never_equal() -> None:
    assert not table_close([{"a": 1, "b": 2}], [{"a": 1}], 0)
