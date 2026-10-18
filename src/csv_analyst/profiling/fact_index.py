"""Card H-R (PLAN_v2.md): the fact-bank block added to the planner prompt.

`settings.schema_facts`:
- "all" (default since 2026-10-07, H-R kept): the whole bank, in bank order,
  capped at `MAX_BLOCK_CHARS` so a wide CSV cannot push the prompt past the
  model's context. Deterministic.
- "off": no block.

Picking facts per question by embedding similarity was measured in H-R and
did no better than the whole bank; that code was removed (report:
evaluation/reports/phase14/H-R/report.md).

The bank is built once per dataset and cached. The block goes under the
schema block; the full column list stays (the planner must still see every
column to abstain on a missing one).
"""

from __future__ import annotations

import pandas as pd

from csv_analyst.profiling.facts import Fact, build_fact_bank
from csv_analyst.profiling.schema import SchemaContext

# ~2.5k tokens: the measured banks are 3.9k (sales) and 4.4k (hospital) chars,
# so the cap only bites on wider CSVs than the benchmark ones.
MAX_BLOCK_CHARS = 10_000
HEADER = "Additional facts computed from the full data (they may or may not matter for this question):"

_banks: dict[str, list[Fact]] = {}


def bank_for(schema: SchemaContext, df: pd.DataFrame) -> list[Fact]:
    if schema.dataset_id not in _banks:
        _banks[schema.dataset_id] = build_fact_bank(df, schema)
    return _banks[schema.dataset_id]


def clear_cache() -> None:
    _banks.clear()


def facts_block(schema: SchemaContext, df: pd.DataFrame, mode: str) -> str:
    if mode == "off":
        return ""
    if mode != "all":
        raise ValueError(f"unknown schema_facts mode {mode!r}; expected 'all' or 'off'")
    facts = bank_for(schema, df)
    if not facts:
        return ""
    lines, size = [HEADER], len(HEADER)
    for i, f in enumerate(facts):
        line = f"- {f.text}"
        if size + len(line) + 1 > MAX_BLOCK_CHARS:
            lines.append(f"- ({len(facts) - i} more facts left out to fit the prompt.)")
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)
