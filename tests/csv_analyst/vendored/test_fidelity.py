"""Numeric-fidelity check: every number in the answer must trace back to the
executed result, the question, or a bounded counting number (PLAN.md
Phase 3, the anti-hallucination backstop)."""

from __future__ import annotations

from csv_analyst.contracts import NumericFidelity, ResultKind, ResultPayload
from csv_analyst.validation.fidelity import check_numeric_fidelity


def _scalar_result(value):
    return ResultPayload(kind=ResultKind.SCALAR, shape=[1], columns=[], preview=[[value]])


def _table_result(preview, columns):
    return ResultPayload(kind=ResultKind.DATAFRAME, shape=[len(preview), len(columns)],
                          columns=columns, preview=preview)


def test_number_matching_result_passes():
    result = _scalar_result(125.44)
    check = check_numeric_fidelity("The average revenue is 125.44.", "average revenue?", result)
    assert check.verdict == NumericFidelity.PASS


def test_rounded_number_within_tolerance_passes():
    result = _scalar_result(125.4444444)
    check = check_numeric_fidelity("The average is about 125.44.", "average?", result)
    assert check.verdict == NumericFidelity.PASS


def test_fabricated_number_fails():
    result = _scalar_result(125.44)
    check = check_numeric_fidelity("The average revenue is 999.99.", "average revenue?", result)
    assert check.verdict == NumericFidelity.FAILED
    assert 999.99 in check.unmatched


def test_number_from_question_is_allowed():
    result = _scalar_result(3.0)
    check = check_numeric_fidelity("There are 3 regions with revenue above 1000.", "regions above 1000?", result)
    assert check.verdict == NumericFidelity.PASS


def test_positional_counting_number_allowed():
    result = _table_result([["North", 100.0], ["South", 90.0], ["East", 80.0]], ["region", "revenue"])
    check = check_numeric_fidelity("The top 3 regions are North, South, East.", "top regions?", result)
    assert check.verdict == NumericFidelity.PASS


def test_no_numbers_passes_trivially():
    result = _scalar_result(None)
    check = check_numeric_fidelity("No rows matched this analysis.", "average salary?", result)
    assert check.verdict == NumericFidelity.PASS


def test_table_value_match_passes():
    result = _table_result([["North", 125.44]], ["region", "revenue"])
    check = check_numeric_fidelity("North has the highest revenue at 125.44.", "highest revenue region?", result)
    assert check.verdict == NumericFidelity.PASS


def test_half_up_rounding_of_a_binary_float_edge_case_passes():
    # round(370.125, 2) == 370.12 in Python (binary float representation),
    # but 370.13 is the mathematically-correct half-up rounding and must not
    # be flagged as a fabricated number (real bug found via live trace).
    result = _scalar_result(370.125)
    check = check_numeric_fidelity("The average revenue is $370.13.", "average revenue?", result)
    assert check.verdict == NumericFidelity.PASS


def test_wrong_number_still_fails_after_rounding_tolerance_widened():
    result = _scalar_result(370.125)
    check = check_numeric_fidelity("The average revenue is $999.99.", "average revenue?", result)
    assert check.verdict == NumericFidelity.FAILED
    assert 999.99 in check.unmatched


# Card H-F F0: fidelity v2. Each case below is a real false alarm of v1 on the
# protected run (the answer was right, the user got the raw template list).

def test_v2_hyphenated_ids_are_not_negative_numbers():
    result = _table_result([["SM-20320", 25043.05], ["TC-20980", 19052.218]], ["customer_id", "total_sales"])
    answer = "The top customers are SM-20320 ($25,043.05) and TC-20980 ($19,052.22)."
    assert check_numeric_fidelity(answer, "top customers?", result).verdict == NumericFidelity.FAILED
    assert check_numeric_fidelity(answer, "top customers?", result, version="v2").verdict == NumericFidelity.PASS


def test_v2_dates_in_text_cells_pass():
    result = _table_result([["2018-09-09", 30, 15]], ["admission_date", "admissions", "prev_admissions"])
    answer = "The largest increase was on 2018-09-09: 30 admissions, an increase of 15."
    assert check_numeric_fidelity(answer, "which day?", result).verdict == NumericFidelity.FAILED
    assert check_numeric_fidelity(answer, "which day?", result, version="v2").verdict == NumericFidelity.PASS


def test_v2_years_stored_as_text_pass():
    result = _table_result([[str(y)] for y in (1918, 1923, 2018)], ["birth_year"])
    answer = "The unique birth years range from 1918 to 2018."
    assert check_numeric_fidelity(answer, "birth years?", result).verdict == NumericFidelity.FAILED
    assert check_numeric_fidelity(answer, "birth years?", result, version="v2").verdict == NumericFidelity.PASS


def test_v2_rounded_percentage_of_a_fraction_passes():
    result = _table_result([["Furniture", 0.1739226779820839]], ["category", "avg_discount"])
    for answer in ("Furniture has the highest average discount at 17.39%.",
                   "Furniture has the highest average discount at 17.4 percent."):
        assert check_numeric_fidelity(answer, "discount?", result).verdict == NumericFidelity.FAILED
        assert check_numeric_fidelity(answer, "discount?", result, version="v2").verdict == NumericFidelity.PASS


def test_v2_rounded_percentage_needs_a_percent_sign():
    # 17.39 without "%" is not 0.1739 written as a percentage.
    result = _table_result([["Furniture", 0.1739226779820839]], ["category", "avg_discount"])
    check = check_numeric_fidelity("The discount is 17.39.", "discount?", result, version="v2")
    assert check.verdict == NumericFidelity.FAILED


def test_v2_invented_numbers_still_fail():
    result = _table_result([["SM-20320", 25043.05], ["2018-09-09", 30]], ["key", "value"])
    for answer, invented in (("SM-20320 spent $26,043.05.", 26043.05),
                             ("Discounts averaged 18.39%.", 18.39),
                             ("There were 31 admissions on 2018-09-09.", 31.0),
                             ("Customer SM-20321 is first.", 20321.0)):
        check = check_numeric_fidelity(answer, "q", result, version="v2")
        assert check.verdict == NumericFidelity.FAILED, answer
        assert invented in check.unmatched, answer


def test_v2_minus_sign_after_a_space_is_still_negative():
    result = _scalar_result(9)
    check = check_numeric_fidelity("Admissions changed by -9 that day.", "change?", result, version="v2")
    assert check.verdict == NumericFidelity.FAILED
    assert -9.0 in check.unmatched
