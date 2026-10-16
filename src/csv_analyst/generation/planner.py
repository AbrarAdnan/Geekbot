"""Question + schema -> PlanResponse (PLAN.md Phase 2).

The planner never states a numeric answer. It proposes `plan` (human-readable
steps) and `code` (pandas assigned to `result`); the caller is responsible for
running that code through the AST guard and the execution sandbox (Phase 3).

A malformed-JSON response gets exactly one repair attempt (re-prompt with the
parse error), per CLAUDE.md's 2-retry ceiling — after that it is a failed
sample, not an infinite loop.
"""

from __future__ import annotations

import ast
import codecs
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from csv_analyst.config import settings
from csv_analyst.contracts import Answerability
from csv_analyst.llm.base import LLMProvider
from csv_analyst.profiling.schema import ColumnRole, SchemaContext
from csv_analyst.sql_engine.prompt import build_sql_prompt, normalize_sql

PROMPT_VERSION = "planner-v6"
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "llm" / "prompts" / f"{PROMPT_VERSION}.txt"

# Used only by the Phase 5 ablation harness's E0 arm (CLAUDE.md §16 / PLAN.md
# Phase 5): no schema block, none of planner-v2's grounding rules -- the
# "naive LLM-writes-pandas loop" the ablation exists to measure. Never used
# by the production pipeline's default configuration.
BASELINE_PROMPT_VERSION = "planner-baseline-v1"
_BASELINE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "llm" / "prompts" / f"{BASELINE_PROMPT_VERSION}.txt"
)

SAMPLE_VALUES_IN_PROMPT = 5
TOP_VALUES_IN_PROMPT = 8

# Card H-P (PLAN_v2.md): experiment-only suffix, selected by
# settings.planner_suffix. The placebo has the same token count as the
# emphasis line, so any flip it causes is prompt-perturbation noise.
PLANNER_SUFFIXES = {
    "off": "",
    "emphasis": "Make no mistakes.",
    "placebo": "End of request.",
}


class PlanResponse(BaseModel):
    answerability: Answerability
    reason: Optional[str] = None
    clarifying_question: Optional[str] = None
    # PLAN.md Phase 11 test 1: positioned BEFORE `plan`/`code` deliberately.
    # Phase 9's Phase B added a similar grain-declaration field *after* `code`
    # in the class -- with Ollama's constrained JSON decoding, keys are
    # generated in schema order, so a field placed after `code` is decided
    # only after the code is already written and can't causally affect it.
    # This test isolates whether forcing the grain judgment first changes
    # what code the model actually writes -- no downstream enforcement here.
    row_represents: str = ""
    question_entity: str = ""
    plan: list[str] = Field(default_factory=list)
    columns_used: list[str] = Field(default_factory=list)
    code: str = ""


class PlanGenerationError(Exception):
    """Raised when the model fails to produce parseable JSON even after the
    one allowed repair attempt."""



# PLAN.md Phase 6B.3: this used to be repeated in full for every repeating
# identifier column -- on a 21-column schema with several such columns, that
# dominated the prompt. Said once, globally, only when at least one column
# needs it.
ID_REPEATS_NOTE = (
    "Note: a column marked id_repeats=true is a finer-grained identifier than one row "
    "per entity (e.g. one row per transaction/admission/order, not one per "
    "customer/patient). A per-entity statistic must deduplicate by that column first, "
    "or it will overweight entities that repeat more often."
)

# Appended after the column list when the profiler found entity attributes.
# Measured on the benchmark (qwen2.5-coder:7b, two seed sets): as a separate
# section, with the old short id_repeats note kept, it fixed hospital/h01 on
# both seed sets (never passed without it); the only other flip, hospital/m01,
# came from sampling on the retry path, not from this section. The same facts
# inlined as per-column `level=` tags plus a longer note replacing the short
# one made the model stop deduplicating where it already did (hospital/e03,
# h03) -- rejected.
ENTITY_SECTION_HEADER = (
    "Entity attributes (derived from the data: each column on the right has exactly one "
    "value per value of the id on the left, so it repeats on every row of that entity):"
)
ENTITY_SECTION_RULE = (
    "If the question counts, averages, ranks, or lists the ENTITY itself (the thing an id "
    "on the left identifies) using its attributes, first df.drop_duplicates(subset='<that id>') "
    "-- otherwise entities with more rows are counted more than once. If the question counts "
    "or sums the rows/events themselves, do not deduplicate. When reporting an entity, "
    "include its 1:1 readable label if one is listed."
)


def _render_entity_section(schema: SchemaContext) -> list[str]:
    if not schema.entity_attributes:
        return []
    distinct = {c.name: c.distinct_count for c in schema.columns}
    lines = ["", ENTITY_SECTION_HEADER]
    for id_name, attrs in schema.entity_attributes.items():
        line = f"- {id_name} ({distinct.get(id_name, '?')} distinct) -> " + ", ".join(attrs)
        labels = schema.entity_labels.get(id_name)
        if labels:
            line += f"  [1:1 readable label: {', '.join(labels)}]"
        lines.append(line)
    owned = set(schema.entity_attributes) | {a for attrs in schema.entity_attributes.values() for a in attrs}
    row_level = [c.name for c in schema.columns if c.name not in owned and c.distinct_count > 1]
    lines.append("Row-level columns (can differ between rows of the same entity): " + ", ".join(row_level))
    lines.append(ENTITY_SECTION_RULE)
    return lines


def render_schema_block(schema: SchemaContext) -> str:
    lines = [
        f"Rows: {schema.row_count}",
        f"Columns: {schema.column_count}",
        "",
    ]
    any_id_repeats = False

    for col in schema.columns:
        detail = [f"role={col.role.value}", f"null={col.null_pct:.1f}%"]

        if col.role == ColumnRole.MEASURE and col.numeric_min is not None:
            detail.append(f"range=[{col.numeric_min}, {col.numeric_max}]")
        if col.role == ColumnRole.IDENTIFIER:
            if col.distinct_count < schema.row_count:
                detail.append(f"id_repeats=true ({col.distinct_count}/{schema.row_count} distinct)")
                any_id_repeats = True
            else:
                detail.append(f"{col.distinct_count} distinct (unique per row)")
        if col.role == ColumnRole.DATE:
            if col.date_ambiguous:
                detail.append("date_ambiguous=true (do not assume day/month order)")
            elif col.date_format:
                detail.append(f"format={col.date_format}, range=[{col.date_min}, {col.date_max}]")
        if col.role == ColumnRole.DIMENSION and col.top_values:
            top = ", ".join(f"{v.value!r}" for v in col.top_values[:TOP_VALUES_IN_PROMPT])
            detail.append(f"values={{{top}}} ({col.distinct_count} distinct)")
        if col.sample_values and col.role not in (ColumnRole.DIMENSION,):
            samples = ", ".join(repr(v) for v in col.sample_values[:SAMPLE_VALUES_IN_PROMPT])
            detail.append(f"samples=[{samples}]")

        lines.append(f"- {col.name} ({col.dtype}): {', '.join(detail)}")

    if any_id_repeats:
        lines.append("")
        lines.append(ID_REPEATS_NOTE)

    if schema.warnings:
        lines.append("")
        lines.append("Data quality warnings: " + "; ".join(schema.warnings))

    lines.extend(_render_entity_section(schema))
    return "\n".join(lines)


def build_prompt(
    question: str, schema: SchemaContext, *, include_schema: bool = True, engine: str = "pandas",
    extra_context: str = "",
) -> str:
    """`extra_context`: a block placed right under the schema block (the
    card H-R fact bank); empty = schema block only."""
    if not include_schema:
        template = _BASELINE_PROMPT_PATH.read_text(encoding="utf-8")
        return template.format(question=question)
    if engine == "sql":
        prompt = build_sql_prompt(question, schema, extra_context=extra_context)
    else:
        template = _PROMPT_PATH.read_text(encoding="utf-8")
        schema_block = render_schema_block(schema)
        if extra_context:
            schema_block = f"{schema_block}\n\n{extra_context}"
        prompt = template.format(schema_block=schema_block, question=question)
    suffix = PLANNER_SUFFIXES[settings.planner_suffix]
    return f"{prompt.rstrip()}\n\n{suffix}\n" if suffix else prompt


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _normalize_code(code: str) -> str:
    """Repair a `code` string the model escaped as if it were embedding it in
    another string (literal `\\n` instead of a newline, etc.) -- PLAN.md
    Phase 6A.3. Parse-gated by construction: a candidate is only returned if
    it parses, so already-valid code is always returned unchanged and
    unrepairable garbage falls through to the AST gate's real error."""
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    candidates = []
    try:
        candidates.append(codecs.decode(code, "unicode_escape"))
    except (UnicodeDecodeError, ValueError, TypeError):
        pass
    candidates.append(
        code.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
    )

    for candidate in candidates:
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return code


def _parse(raw: str, engine: str = "pandas") -> PlanResponse:
    cleaned = _strip_code_fences(raw)
    data = json.loads(cleaned)
    plan = PlanResponse.model_validate(data)
    if plan.code:
        plan.code = normalize_sql(plan.code) if engine == "sql" else _normalize_code(plan.code)
    return plan


def generate_plan(
    llm: LLMProvider,
    question: str,
    schema: SchemaContext,
    *,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    include_schema: bool = True,
    engine: str = "pandas",
    extra_context: str = "",
) -> PlanResponse:
    prompt = build_prompt(
        question, schema, include_schema=include_schema, engine=engine, extra_context=extra_context,
    )

    # PLAN.md Phase 6B.2: pass the response's JSON Schema so a
    # structured-output-capable backend (Ollama 0.32+) constrains the shape
    # of the response, not just "is this JSON" -- a provider without that
    # capability ignores it and the one-shot repair path below still covers it.
    json_schema = PlanResponse.model_json_schema()

    raw = llm.generate_json(prompt, temperature=temperature, seed=seed, schema=json_schema)
    try:
        return _parse(raw, engine)
    except (json.JSONDecodeError, ValidationError) as first_error:
        repair_prompt = (
            f"{prompt}\n\n"
            f"Your previous response could not be parsed as valid JSON matching the "
            f"required schema: {first_error}\n"
            f"Respond again with ONLY the corrected JSON object, no markdown fences."
        )
        raw2 = llm.generate_json(repair_prompt, temperature=temperature, seed=seed, schema=json_schema)
        try:
            return _parse(raw2, engine)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise PlanGenerationError(
                f"malformed planner output after repair attempt: {second_error}"
            ) from second_error
