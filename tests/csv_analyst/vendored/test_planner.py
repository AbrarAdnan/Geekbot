"""Planner tests: schema rendering, prompt construction, JSON parsing, and
the single-repair-attempt retry on malformed model output."""

from __future__ import annotations

import json

import pytest

from csv_analyst.contracts import Answerability
from csv_analyst.generation.planner import (
    PlanGenerationError,
    _normalize_code,
    build_prompt,
    generate_plan,
    render_schema_block,
)
from csv_analyst.ingestion.loader import load_csv
from csv_analyst.profiling.profiler import profile_dataset

VALID_PLAN_JSON = """
{
  "answerability": "answerable",
  "reason": null,
  "clarifying_question": null,
  "plan": ["Group rows by region", "Take mean of revenue per group"],
  "columns_used": ["region", "revenue"],
  "code": "result = df.groupby('region')['revenue'].mean().reset_index()"
}
"""


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.calls.append({"prompt": prompt, "temperature": temperature, "seed": seed, "schema": schema})
        return self.responses.pop(0)


@pytest.fixture
def schema(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "order_id,region,revenue,order_date\n"
        "1,North,100.0,2024-01-01\n"
        "2,South,200.0,2024-01-02\n"
        "3,North,150.0,2024-01-03\n",
        encoding="utf-8",
    )
    return profile_dataset(load_csv(path))


def test_render_schema_block_includes_columns(schema):
    block = render_schema_block(schema)
    assert "region" in block
    assert "revenue" in block
    assert "order_date" in block
    assert "Rows: 3" in block


def test_render_schema_block_compacts_repeated_id_repeat_guidance(tmp_path):
    """PLAN.md Phase 6B.3: the repeat-dedup guidance must appear exactly
    once, no matter how many identifier columns repeat -- not once per
    column, which is what dominated the prompt on a wide schema. No
    information may be lost: every column keeps its dtype/null%/role and an
    id_repeats marker."""
    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,customer_id,region,revenue\n"
        "1,A,North,100.0\n"
        "1,A,North,50.0\n"
        "2,B,South,200.0\n",
        encoding="utf-8",
    )
    wide_schema = profile_dataset(load_csv(path))
    block = render_schema_block(wide_schema)

    assert block.count("id_repeats=true (") == 2  # order_id and customer_id both repeat
    assert block.count("A per-entity statistic must deduplicate") == 1

    for col in wide_schema.columns:
        line = next(l for l in block.splitlines() if l.startswith(f"- {col.name} ("))
        assert col.dtype in line
        assert f"null={col.null_pct:.1f}%" in line
        assert f"role={col.role.value}" in line


def test_build_prompt_includes_question_and_schema(schema):
    prompt = build_prompt("What is average revenue by region?", schema)
    assert "What is average revenue by region?" in prompt
    assert "region" in prompt
    assert "result" in prompt  # rule about assigning to `result`


@pytest.mark.parametrize("engine", ["sql", "pandas"])
@pytest.mark.parametrize("mode,text", [("emphasis", "Make no mistakes."), ("placebo", "End of request.")])
def test_build_prompt_suffix_is_last_line_after_question(schema, monkeypatch, mode, text, engine):
    """Card H-P (PLAN_v2.md): the suffix is system text after the untrusted
    question block, and "off" leaves the template prompt byte-identical."""
    from csv_analyst.config import settings

    monkeypatch.setattr(settings, "planner_suffix", "off")
    shipped = build_prompt("Average revenue?", schema, engine=engine)
    monkeypatch.setattr(settings, "planner_suffix", mode)
    prompt = build_prompt("Average revenue?", schema, engine=engine)
    assert prompt.rstrip().splitlines()[-1] == text
    assert prompt.index(">>>") < prompt.index(text)
    assert text not in shipped
    monkeypatch.setattr(settings, "planner_suffix", "off")
    assert build_prompt("Average revenue?", schema, engine=engine) == shipped


def test_build_prompt_without_schema_omits_real_columns(schema):
    """Phase 5 ablation E0 arm (CLAUDE.md §16): include_schema=False must not
    leak any real column name into the prompt -- that's the whole point of
    measuring a naive, ungrounded LLM-writes-pandas loop. Uses a question
    that doesn't itself name a column, so a hit can only come from the
    (suppressed) schema block."""
    prompt = build_prompt("What is the average value grouped by category?", schema, include_schema=False)
    assert "What is the average value grouped by category?" in prompt
    assert "result" in prompt
    assert "Rows:" not in prompt  # only the real schema block prints row count
    for col in ("order_id", "region", "revenue", "order_date"):
        assert col not in prompt


def test_generate_plan_parses_valid_json(schema):
    llm = FakeLLM([VALID_PLAN_JSON])
    plan = generate_plan(llm, "average revenue by region", schema, seed=11)

    assert plan.answerability == Answerability.ANSWERABLE
    assert plan.columns_used == ["region", "revenue"]
    assert "groupby" in plan.code
    assert len(llm.calls) == 1
    assert llm.calls[0]["seed"] == 11


def test_generate_plan_passes_response_json_schema_to_the_provider(schema):
    """PLAN.md Phase 6B.2: a structured-output-capable backend should be
    told the required response shape, not just "respond with JSON"."""
    llm = FakeLLM([VALID_PLAN_JSON])
    generate_plan(llm, "average revenue by region", schema)

    sent_schema = llm.calls[0]["schema"]
    assert sent_schema is not None
    assert "code" in sent_schema["properties"]
    assert "answerability" in sent_schema["properties"]


def test_generate_plan_repairs_malformed_json_once(schema):
    llm = FakeLLM(["not json at all", VALID_PLAN_JSON])
    plan = generate_plan(llm, "average revenue by region", schema)

    assert plan.answerability == Answerability.ANSWERABLE
    assert len(llm.calls) == 2
    assert "could not be parsed" in llm.calls[1]["prompt"]


def test_generate_plan_raises_after_repair_fails(schema):
    llm = FakeLLM(["not json", "still not json"])
    with pytest.raises(PlanGenerationError):
        generate_plan(llm, "average revenue by region", schema)
    assert len(llm.calls) == 2


def test_generate_plan_strips_markdown_fences(schema):
    fenced = "```json\n" + VALID_PLAN_JSON + "\n```"
    llm = FakeLLM([fenced])
    plan = generate_plan(llm, "average revenue by region", schema)
    assert plan.answerability == Answerability.ANSWERABLE


def test_generate_plan_data_missing_when_column_absent(schema):
    response = """
    {
      "answerability": "data_missing",
      "reason": "salary column is not present in the dataset",
      "clarifying_question": null,
      "plan": [],
      "columns_used": [],
      "code": ""
    }
    """
    llm = FakeLLM([response])
    plan = generate_plan(llm, "average employee salary", schema)

    assert plan.answerability == Answerability.DATA_MISSING
    assert plan.code == ""
    assert plan.reason


# --- PLAN.md Phase 6A.3: normalize escaped code before the AST gate ---------


def test_normalize_code_repairs_escaped_newlines():
    escaped = "result = df['revenue'].mean()\\nresult = result * 2"
    normalized = _normalize_code(escaped)
    assert "\\n" not in normalized
    assert "\n" in normalized


def test_normalize_code_leaves_valid_code_byte_identical():
    valid = "result = df.groupby('region')['revenue'].mean().reset_index()"
    assert _normalize_code(valid) == valid


def test_normalize_code_returns_unrepairable_garbage_unchanged():
    garbage = "result = ((("
    assert _normalize_code(garbage) == garbage


def test_generate_plan_repairs_escaped_code_from_the_model(schema):
    # Simulates the h03 failure mode: the model's `code` field literally
    # contains a backslash followed by "n" (two characters), not a newline --
    # what json.dumps produces here is the same shape a model would emit.
    escaped_code = r"result = df['revenue'].mean()\nresult = result"
    payload = {
        "answerability": "answerable", "plan": ["step"],
        "columns_used": ["revenue"], "code": escaped_code,
    }
    llm = FakeLLM([json.dumps(payload)])
    plan = generate_plan(llm, "average revenue", schema)

    assert plan.answerability == Answerability.ANSWERABLE
    assert "\\n" not in plan.code
    assert "\n" in plan.code


def test_render_schema_block_appends_entity_section(tmp_path):
    path = tmp_path / "admissions.csv"
    rows = ["patient_id,patient_name,allergy,diagnosis"]
    for p in range(1, 13):
        for a in range(3):
            rows.append(f"{p},name{p},al{p % 4},dx{(p + a) % 5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    block = render_schema_block(profile_dataset(load_csv(path)))

    assert "- patient_id (12 distinct) -> patient_name, allergy  [1:1 readable label: patient_name]" in block
    assert "Row-level columns (can differ between rows of the same entity): diagnosis" in block
    assert block.count("drop_duplicates(subset='<that id>')") == 1
    # the short id_repeats note stays: replacing it made the model stop deduplicating
    assert "A per-entity statistic must deduplicate" in block


def test_render_schema_block_has_no_entity_section_without_attributes(schema):
    assert "Entity attributes" not in render_schema_block(schema)


def test_sql_prompt_uses_sql_wording_and_iso_dates():
    from csv_analyst.generation.planner import build_prompt
    from csv_analyst.ingestion.loader import load_csv
    from csv_analyst.profiling.profiler import profile_dataset

    schema = profile_dataset(load_csv("data/hospital/hospital.csv"))
    prompt = build_prompt("How many patients?", schema, engine="sql")
    assert "table named `data`" in prompt
    assert "COUNT(DISTINCT" in prompt
    assert "drop_duplicates" not in prompt
    assert "Timestamp(" not in prompt
    assert "(INTEGER)" in prompt


def test_emphasis_suffix_is_the_shipped_default():
    from csv_analyst.config import Settings

    assert Settings.model_fields["planner_suffix"].default == "emphasis"  # H-P kept, PLAN_v2.md
