"""Executed-result -> grounded natural-language answer (PLAN.md Phase 3,
CLAUDE.md Sec12). The answer model never sees the raw DataFrame -- only the
question, the plan, the code that produced the result, and the *executed*
result itself. It cannot state a number the computation did not produce.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from csv_analyst.config import settings
from csv_analyst.contracts import ResultPayload
from csv_analyst.llm.base import LLMProvider, LLMProviderError

PROMPT_VERSION = "answer-v4"  # shipped default; settings.answer_prompt selects the file
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


# Every preview row (result_guard.PREVIEW_ROW_CAP = 100) goes to the answer
# model. The old cap sent only the first/last 10 rows of anything over 20:
# on a 48-row East/West/Central/South table that hid every East row, and the
# model still claimed East was higher (sales/h02, PLAN_v2.md).
ANSWER_PREVIEW_ROW_CAP = 100
# Above this many rows an exact numeric summary is added, so the model does
# not have to scan (and misread) extremes or totals by eye.
SUMMARY_MIN_ROWS = 20


def _numeric_summary(preview: list[list], columns: list[str]) -> dict:
    """min/max/mean/sum per numeric column over the preview rows (never
    re-touches the DataFrame). Exact for a complete result; for a partial one
    it covers the shown rows only, which the result block states."""
    summary: dict[str, dict[str, float]] = {}
    for col_idx, col_name in enumerate(columns):
        values = [
            row[col_idx] for row in preview
            if col_idx < len(row) and isinstance(row[col_idx], (int, float))
            and not isinstance(row[col_idx], bool)
        ]
        if values:
            summary[col_name] = {
                "min": min(values), "max": max(values),
                "mean": sum(values) / len(values), "sum": sum(values),
            }
    return summary


# PLAN_v2.md option B: a grouped table (a text column repeating across rows,
# e.g. region x month) gets exact per-group facts from code. With all 48
# rows of sales/h02 visible the 7b still claimed "East had higher total
# sales" (West was higher) -- it cannot reliably add up or compare groups by
# reading rows, so those facts are computed here instead.
GROUP_FACT_MAX_GROUPS = 12


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# An id is a label, never a measure: summing patient_ids per first name gave
# "Stanislaus has the highest sum of patient IDs" (hospital/m05 replay).
_ID_NAME = re.compile(r"(^|_)id$", re.IGNORECASE)
_DATE_PART = re.compile(r"(^|_)(year|quarter|month|week|day|hour)s?($|_)", re.IGNORECASE)


def _is_date_part(name: str, values: list) -> bool:
    """A whole-number column named like a date part (month 1-12, year) is a
    label to group by, not a measure to sum."""
    return bool(_DATE_PART.search(str(name))) and all(
        v is None or (_is_number(v) and float(v).is_integer()) for v in values
    )


def _group_facts(preview: list[list], columns: list[str]) -> dict:
    """Per group of each repeating text column: row count, sum, max, min per
    numeric column, the other text values on the max/min rows, and the groups
    ranked by sum. Computed from the preview rows, never the DataFrame."""
    if len(preview) < 2 or not columns:
        return {}
    width = len(columns)
    if any(len(r) != width for r in preview):  # e.g. a Series preview whose rows don't match the column list
        return {}
    numeric = [
        i for i in range(width)
        if any(_is_number(r[i]) for r in preview) and all(_is_number(r[i]) or r[i] is None for r in preview)
        and not _is_date_part(columns[i], [r[i] for r in preview])
        and not _ID_NAME.search(str(columns[i]))
    ]
    if not numeric:
        return {}
    text = [i for i in range(width) if i not in numeric]
    facts: dict = {}
    for g in text:
        keys = list(dict.fromkeys(r[g] for r in preview))
        if not 2 <= len(keys) <= GROUP_FACT_MAX_GROUPS or len(keys) == len(preview):
            continue
        others = [i for i in text if i != g]
        groups: dict = {}
        for key in keys:
            rows = [r for r in preview if r[g] == key]
            stats: dict = {}
            for n in numeric:
                vals = [r for r in rows if _is_number(r[n])]
                if not vals:
                    continue
                top = max(vals, key=lambda r: r[n])
                low = min(vals, key=lambda r: r[n])
                stats[columns[n]] = {
                    "rows": len(vals),
                    "sum": sum(r[n] for r in vals),
                    "max": top[n], "max_at": {columns[o]: top[o] for o in others},
                    "min": low[n], "min_at": {columns[o]: low[o] for o in others},
                }
            groups[str(key)] = stats
        ranked = {
            columns[n]: sorted(groups, key=lambda k: groups[k][columns[n]]["sum"], reverse=True)
            for n in numeric if all(columns[n] in groups[k] for k in groups)
        }
        facts[f"by {columns[g]}"] = {"groups": groups, "ranked_by_sum_desc": ranked}
    return facts


def derived_numbers(result: ResultPayload) -> list[float]:
    """Numbers the answer model is handed that are not literally in the
    preview (numeric summary, group facts) -- the numeric-fidelity check must
    accept them, or an answer quoting an exact group total would be flagged
    as invented."""
    rows = result.preview[:ANSWER_PREVIEW_ROW_CAP]
    found: list[float] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif _is_number(node):
            found.append(float(node))

    if len(rows) > SUMMARY_MIN_ROWS:
        walk(_numeric_summary(rows, result.columns))
    walk(_group_facts(rows, result.columns))
    return found


def _rows_total(result: ResultPayload) -> int:
    return result.shape[0] if result.shape else len(result.preview)


def _result_block(result: ResultPayload) -> str:
    # Rows keep the executed code's own order -- a second, answer-only sort
    # would be an ungrounded inference.
    rows = result.preview[:ANSWER_PREVIEW_ROW_CAP]
    partial = result.truncated or len(result.preview) > len(rows)
    block: dict = {
        "kind": result.kind.value,
        "shape": result.shape,
        "columns": result.columns,
        "preview": rows,
        "partial": partial,
    }
    if partial:
        block["rows_shown"] = len(rows)
        block["rows_total"] = _rows_total(result)
    if len(rows) > SUMMARY_MIN_ROWS:
        block["numeric_summary"] = _numeric_summary(rows, result.columns)
        block["numeric_summary_scope"] = "shown rows only" if partial else "all rows"
    group_facts = _group_facts(rows, result.columns)
    if group_facts:
        block["group_facts"] = group_facts
        block["group_facts_scope"] = "shown rows only" if partial else "all rows"
    return json.dumps(block, default=str)


def partial_view_note(answer: str, result: ResultPayload) -> str:
    """Guarantee the partial-view disclosure even when the model leaves it
    out: a result larger than the preview gets a fixed note unless the answer
    already names the total row count."""
    shown = min(len(result.preview), ANSWER_PREVIEW_ROW_CAP)
    if not (result.truncated or len(result.preview) > shown):
        return answer
    total = _rows_total(result)
    # Whole numbers only: "793" must not match inside "$1793.98".
    stated = {n.replace(",", "") for n in re.findall(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?", answer)}
    if str(total) in stated:
        return answer
    return (f"{answer} (This answer is based on a partial view: the first {shown} of "
            f"{total:,} result rows. See the result table in the evidence.)")


def build_answer_prompt(question: str, plan: list[str], code: str, result: ResultPayload) -> str:
    template = (_PROMPTS_DIR / f"{settings.answer_prompt}.txt").read_text(encoding="utf-8")
    return template.format(
        question=question,
        plan="\n".join(f"- {step}" for step in plan) or "(none)",
        code=code,
        result_block=_result_block(result),
    )


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_answer(
    llm: LLMProvider,
    question: str,
    plan: list[str],
    code: str,
    result: ResultPayload,
    *,
    temperature: float = 0.2,
    seed: Optional[int] = None,
) -> str:
    prompt = build_answer_prompt(question, plan, code, result)
    try:
        raw = llm.generate_json(prompt, temperature=temperature, seed=seed)
    except LLMProviderError:
        return template_answer(result)
    try:
        data = json.loads(_strip_code_fences(raw))
        answer = str(data.get("answer", "")).strip()
        if answer:
            return answer
    except (json.JSONDecodeError, AttributeError):
        pass
    return template_answer(result)


def template_answer(result: ResultPayload) -> str:
    """Deterministic fallback answer built directly from the executed
    result. Used when the LLM answer fails the numeric-fidelity check twice
    (CLAUDE.md Sec12) -- never fabricated, always read straight off `result`.
    """
    if result.kind.value == "scalar":
        value = result.preview[0][0] if result.preview else None
        return f"The computed result is {value}."
    if not result.preview:
        return "No rows matched this analysis."
    header = ", ".join(result.columns) if result.columns else ""
    rows = "; ".join(", ".join(str(v) for v in row) for row in result.preview[:5])
    prefix = f"Result ({header})" if header else "Result"
    return f"{prefix}: {rows}."
