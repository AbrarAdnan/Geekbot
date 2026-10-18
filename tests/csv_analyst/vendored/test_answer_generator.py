"""Answer generation: grounded in the result only, JSON parsing, and the
deterministic template fallback (PLAN.md Phase 3, CLAUDE.md Sec12)."""

from __future__ import annotations

import json

from csv_analyst.answer.generator import (
    _result_block,
    build_answer_prompt,
    generate_answer,
    partial_view_note,
    template_answer,
)
from csv_analyst.contracts import ResultKind, ResultPayload
from csv_analyst.llm.base import LLMProviderError


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.calls.append(prompt)
        return self.responses.pop(0)


def _result():
    return ResultPayload(kind=ResultKind.SCALAR, shape=[1], columns=[], preview=[[125.44]])


def test_build_answer_prompt_includes_code_and_result():
    prompt = build_answer_prompt("average revenue?", ["Take mean of revenue"], "result = df['revenue'].mean()", _result())
    assert "average revenue?" in prompt
    assert "df['revenue'].mean()" in prompt
    assert "125.44" in prompt


def test_generate_answer_parses_json():
    llm = FakeLLM(['{"answer": "The average revenue is 125.44."}'])
    answer = generate_answer(llm, "average revenue?", ["step"], "result = ...", _result())
    assert answer == "The average revenue is 125.44."


def test_generate_answer_falls_back_to_template_on_malformed_json():
    llm = FakeLLM(["not json"])
    answer = generate_answer(llm, "average revenue?", ["step"], "result = ...", _result())
    assert "125.44" in answer


def test_generate_answer_falls_back_to_template_on_transport_failure():
    """A timeout on the answer call must fall back to the deterministic
    template, never crash -- the computation already succeeded, only the NL
    wrapper failed, and the template is always grounded in the real result."""
    class FlakyLLM:
        def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
            raise LLMProviderError("Ollama request failed (test-model): timed out")

    answer = generate_answer(FlakyLLM(), "average revenue?", ["step"], "result = ...", _result())
    assert "125.44" in answer


def test_generate_answer_strips_markdown_fences():
    llm = FakeLLM(['```json\n{"answer": "125.44"}\n```'])
    answer = generate_answer(llm, "average revenue?", ["step"], "result = ...", _result())
    assert answer == "125.44"


def test_template_answer_scalar():
    assert "125.44" in template_answer(_result())


def test_template_answer_empty_table():
    empty = ResultPayload(kind=ResultKind.DATAFRAME, shape=[0, 2], columns=["region", "revenue"], preview=[])
    assert template_answer(empty) == "No rows matched this analysis."


def test_template_answer_table_lists_rows():
    table = ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[2, 2], columns=["region", "revenue"],
        preview=[["North", 100.0], ["South", 200.0]],
    )
    text = template_answer(table)
    assert "North" in text and "South" in text


def _wide_table(n_rows: int) -> ResultPayload:
    preview = [[f"city_{i}", float(i)] for i in range(n_rows)]
    return ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[n_rows, 2], columns=["city", "count"], preview=preview,
    )


def test_result_block_sends_full_preview_when_at_or_under_the_cap():
    block = json.loads(_result_block(_wide_table(20)))
    assert len(block["preview"]) == 20
    assert "numeric_summary" not in block


def test_result_block_sends_every_row_up_to_the_cap():
    # A 48-row East/West/Central/South table used to be cut to first/last 10
    # rows, hiding every East row (sales/h02). Up to 100 rows go in full.
    block = json.loads(_result_block(_wide_table(90)))
    assert len(block["preview"]) == 90
    assert block["preview"][0] == ["city_0", 0.0]
    assert block["preview"][-1] == ["city_89", 89.0]
    assert block["partial"] is False
    assert block["numeric_summary_scope"] == "all rows"


def test_result_block_numeric_summary_is_exact_over_the_full_preview():
    block = json.loads(_result_block(_wide_table(90)))
    summary = block["numeric_summary"]["count"]
    assert summary["min"] == 0.0
    assert summary["max"] == 89.0
    assert summary["sum"] == sum(range(90))
    assert summary["mean"] == sum(range(90)) / 90


def test_result_block_large_result_is_not_resorted():
    # Deliberately out-of-order preview -- first/last rows must reflect the
    # executed code's own order, never a second answer-only sort.
    preview = [["b", 2.0], ["a", 1.0]] + [[f"x{i}", float(i)] for i in range(2, 25)]
    result = ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[len(preview), 2], columns=["k", "v"], preview=preview,
    )
    block = json.loads(_result_block(result))
    assert block["preview"][0] == ["b", 2.0]
    assert block["preview"][1] == ["a", 1.0]


def _partial_table() -> ResultPayload:
    preview = [[f"c{i}", float(i)] for i in range(100)]
    return ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[793, 2], columns=["customer", "sales"],
        preview=preview, truncated=True,
    )


def test_result_block_marks_a_partial_result():
    block = json.loads(_result_block(_partial_table()))
    assert block["partial"] is True
    assert block["rows_shown"] == 100
    assert block["rows_total"] == 793
    assert block["numeric_summary_scope"] == "shown rows only"


def test_partial_view_note_added_when_answer_omits_it():
    text = partial_view_note("Customer c0 spent the least.", _partial_table())
    assert "first 100 of 793 result rows" in text


def test_partial_view_note_not_repeated_when_answer_states_the_total():
    answer = "Based on the first 100 of 793 result rows, c99 spent the most."
    assert partial_view_note(answer, _partial_table()) == answer


def test_partial_view_note_skipped_for_complete_result():
    answer = "c89 is highest."
    assert partial_view_note(answer, _wide_table(90)) == answer


def test_partial_view_note_not_fooled_by_total_inside_another_number():
    # 793 appears inside $1793.98 -- that is not a statement of the row total.
    text = partial_view_note("Values range from $2.89 to $1793.98.", _partial_table())
    assert "first 100 of 793 result rows" in text


def test_partial_view_note_accepts_comma_formatted_total():
    table = ResultPayload(
        kind=ResultKind.DATAFRAME, shape=[9994, 1], columns=["x"],
        preview=[[float(i)] for i in range(100)], truncated=True,
    )
    answer = "Based on the first 100 of 9,994 rows, the maximum shown is 99."
    assert partial_view_note(answer, table) == answer


def _region_month_table() -> ResultPayload:
    preview = [["East", "2017-01", 5.0], ["West", "2017-01", 12.0],
               ["East", "2017-02", 40.0], ["West", "2017-02", 9.0]]
    return ResultPayload(kind=ResultKind.DATAFRAME, shape=[4, 3], columns=["region", "month", "sales"], preview=preview)


def test_result_block_group_facts_give_exact_group_totals_and_ranking():
    facts = json.loads(_result_block(_region_month_table()))["group_facts"]["by region"]
    assert facts["groups"]["East"]["sales"]["sum"] == 45.0
    assert facts["groups"]["West"]["sales"]["max"] == 12.0
    assert facts["groups"]["West"]["sales"]["max_at"] == {"month": "2017-01"}
    assert facts["ranked_by_sum_desc"]["sales"] == ["East", "West"]


def test_group_facts_treat_integer_month_as_label_not_measure():
    preview = [["East", 1, 5.0], ["West", 1, 12.0], ["East", 2, 40.0], ["West", 2, 9.0]]
    table = ResultPayload(kind=ResultKind.DATAFRAME, shape=[4, 3], columns=["region", "month", "sales"], preview=preview)
    facts = json.loads(_result_block(table))["group_facts"]
    assert "month" not in facts["by region"]["groups"]["East"]
    assert "by month" in facts


def test_no_group_facts_when_no_text_column_repeats():
    block = json.loads(_result_block(_wide_table(30)))
    assert "group_facts" not in block


def test_fidelity_accepts_a_group_total_quoted_from_group_facts():
    from csv_analyst.answer.generator import derived_numbers
    from csv_analyst.contracts import NumericFidelity
    from csv_analyst.validation.fidelity import check_numeric_fidelity

    table = _region_month_table()
    answer = "East had the higher total, 45 versus 21."
    assert check_numeric_fidelity(answer, "q", table).verdict == NumericFidelity.FAILED
    assert check_numeric_fidelity(answer, "q", table, derived_numbers(table)).verdict == NumericFidelity.PASS


def test_group_facts_never_sum_an_id_column():
    preview = [[496, "Spiros"], [629, "Spiros"], [648, "Stanislaus"], [1273, "Stanislaus"]]
    table = ResultPayload(kind=ResultKind.DATAFRAME, shape=[4, 2], columns=["patient_id", "first_name"], preview=preview)
    assert "group_facts" not in json.loads(_result_block(table))


def test_group_facts_skips_rows_narrower_than_columns():
    from csv_analyst.answer.generator import _group_facts

    assert _group_facts([["East", 1.0], ["West"]], ["region", "sales"]) == {}


def test_answer_prompt_version_is_selectable_and_v4_has_no_pandas_wording(monkeypatch):
    from csv_analyst.answer.generator import build_answer_prompt
    from csv_analyst.config import settings
    from csv_analyst.contracts import ResultKind, ResultPayload

    from csv_analyst.config import Settings

    assert Settings.model_fields["answer_prompt"].default == "answer-v4"  # H-C C0 kept
    result = ResultPayload(kind=ResultKind.SCALAR, shape=[1], preview=[[5]])
    monkeypatch.setattr(settings, "answer_prompt", "answer-v3")
    v3 = build_answer_prompt("How many?", ["count rows"], "SELECT COUNT(*) FROM data", result)
    assert "pandas already ran" in v3
    monkeypatch.setattr(settings, "answer_prompt", "answer-v4")
    v4 = build_answer_prompt("How many?", ["count rows"], "SELECT COUNT(*) FROM data", result)
    assert "pandas" not in v4 and "Analysis that was executed:" in v4
    assert v4.replace("the analysis below was already run", "X").replace("Analysis that was executed:", "Y") == \
        v3.replace("pandas already ran the code below", "X").replace("Code that was executed:", "Y")
