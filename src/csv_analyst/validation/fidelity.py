"""Numeric-fidelity check on the generated answer text (PLAN.md Phase 3,
the anti-hallucination backstop). The answer model is grounded in the
executed result, but this is what verifies it did not add or invent a
number in prose. A guardrail with a measured error rate, not a proof --
residual false positives (a legitimately derived ratio flagged as
unmatched) are expected and are reported by the caller, not hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from csv_analyst.contracts import NumericFidelity, ResultPayload

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")
# v2 (card H-F F0): a "-" is a minus sign only when no letter or digit sits
# right before it -- "SM-20320" is the number 20320 and "2018-09-09" is 2018,
# 9, 9 (v1 read -20320 and -9, -9); "a change of -9" is still -9.
_NUMBER_RE_V2 = re.compile(r"(?:(?<![A-Za-z0-9])-)?\d[\d,]*\.?\d*%?")
_PERCENT_AFTER = re.compile(r"\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)


def _extract(text: str, pattern: re.Pattern) -> list[tuple[float, bool]]:
    """(value, written as a percentage) for every number in `text`."""
    numbers = []
    for match in pattern.finditer(text):
        raw = match.group().rstrip("%").replace(",", "")
        if raw in ("", "-", "."):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        percent = match.group().endswith("%") or bool(_PERCENT_AFTER.match(text, match.end()))
        numbers.append((value, percent))
    return numbers


def _extract_numbers(text: str) -> list[float]:
    return [n for n, _ in _extract(text, _NUMBER_RE)]


def _flatten_result_numbers(result: ResultPayload) -> list[float]:
    numbers: list[float] = []
    for row in result.preview:
        for v in row:
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                numbers.append(float(v))
    return numbers


def _text_cell_numbers(result: ResultPayload) -> list[float]:
    """v2: numbers written inside text cells (ids, dates, years stored as
    text) -- the answer may quote them as they appear in the result."""
    numbers: list[float] = []
    for row in result.preview:
        for v in row:
            if isinstance(v, str):
                numbers.extend(n for n, _ in _extract(v, _NUMBER_RE_V2))
    return numbers


def _matches_any(n: float, candidates: list[float], row_count: int, rounded_percent: bool = False) -> bool:
    for c in candidates:
        if abs(n - c) <= max(1e-6, 1e-6 * abs(c)):
            return True
        # n is c rounded to 0-2 decimal places (common answer formatting).
        # A tolerance window, not round(c, dp) == n: Python's round() uses
        # binary-float representation, so e.g. round(370.125, 2) == 370.12
        # even though 370.125 rounds up to 370.13 under ordinary half-up
        # rounding -- a model writing the mathematically-correct 370.13 must
        # not be flagged as having fabricated a number.
        if any(abs(n - c) <= 0.5 * (10 ** -dp) + 1e-9 for dp in (0, 1, 2)):
            return True
        # n expressed as a percentage of candidate c (e.g. c=0.42 -> "42%")
        if abs(n - c * 100) <= 1e-6:
            return True
        # v2: a percentage of a fraction, rounded like any other number
        # (0.17392 -> "17.39%"); only for numbers written as percentages.
        if rounded_percent and any(abs(n - c * 100) <= 0.5 * (10 ** -dp) + 1e-9 for dp in (0, 1, 2)):
            return True
    # positional / counting numbers ("top 5", "1st") bounded by the result size
    return n == int(n) and 0 <= int(n) <= max(row_count, 1)


@dataclass
class FidelityCheck:
    verdict: NumericFidelity
    unmatched: list[float] = field(default_factory=list)


def check_numeric_fidelity(
    answer_text: str, question: str, result: ResultPayload, derived: list[float] | None = None,
    version: str = "v1",
) -> FidelityCheck:
    """`derived`: exact numbers computed from the result by code and handed to
    the answer model (numeric summary, per-group facts) -- as legitimate as
    the result's own cells. `version`: "v1" | "v2" (card H-F F0: hyphens in
    ids / dates, numbers inside text cells, rounded percentages)."""
    v2 = version == "v2"
    pattern = _NUMBER_RE_V2 if v2 else _NUMBER_RE
    answer_numbers = _extract(answer_text, pattern)
    if not answer_numbers:
        return FidelityCheck(verdict=NumericFidelity.PASS)

    question_numbers = {n for n, _ in _extract(question, pattern)}
    result_numbers = _flatten_result_numbers(result) + list(derived or [])
    if v2:
        result_numbers += _text_cell_numbers(result)
    row_count = result.shape[0] if result.shape else 0

    unmatched = [
        n for n, percent in answer_numbers
        if n not in question_numbers and not _matches_any(n, result_numbers, row_count, rounded_percent=v2 and percent)
    ]

    verdict = NumericFidelity.PASS if not unmatched else NumericFidelity.FAILED
    return FidelityCheck(verdict=verdict, unmatched=unmatched)
