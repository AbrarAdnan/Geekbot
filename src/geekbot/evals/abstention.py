"""Measuring abstention on the golden set (step S3.5, D-082 option B, PLAN.md §9).

S3.5 abstains by the model's own refusal (the reply says it could not find the answer and cites
nothing), with no score threshold. That is a design that costs no time and *might* work, so it is
measured, not assumed: every golden question is put to the real graph and the reply is classified.

- An **unanswerable** question should be refused. A refusal is a *correct abstention*; an answer is a
  *wrong answer to an unanswerable question* (the dangerous mistake: the model answers from text that
  does not hold the answer, or from memory).
- An **answerable** question should not be refused. A refusal is a *wrong refusal* (a good answer
  withheld).

Multi-turn items are not run here: they need a conversation and a rewrite (S5.2 measures answers with
threads). The numbers are for the model that was used, which the result names; a number from the small
prototyping model says how the *design* behaves, not how ``gemma4:12b`` will.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from geekbot.evals.datasets import GoldenItem, Split

__all__ = ["AbstentionRow", "run_abstention", "summarize"]

QUESTION_TYPES = ("single_hop", "multi_hop", "unanswerable")


@dataclass(frozen=True)
class AbstentionRow:
    """One golden question and what the graph did with it."""

    id: str
    type: str
    split: str
    abstained: bool
    cited: int
    answer: str  # the start of the reply, for reading
    seconds: float
    weak: int = 0  # how many of its citations the relevance check called weak (S3.5 cite_check)

    @property
    def should_abstain(self) -> bool:
        return self.type == "unanswerable"

    @property
    def correct(self) -> bool:
        return self.abstained == self.should_abstain


def run_abstention(
    items: Iterable[GoldenItem],
    splits: Mapping[str, Split],
    ask: Callable[[str], Mapping[str, Any]],
    *,
    clock: Callable[[], float],
    on_row: Callable[[AbstentionRow], None] | None = None,
) -> list[AbstentionRow]:
    """Put every single-hop, multi-hop and unanswerable question to ``ask`` (the graph, one run each).

    ``ask`` returns the graph's final state. A question the graph raises on is not caught: a broken
    run must not be counted as a refusal.
    """
    rows: list[AbstentionRow] = []
    for item in items:
        if item.type not in QUESTION_TYPES:
            continue
        started = clock()
        final = ask(item.question)
        row = AbstentionRow(
            id=item.id,
            type=item.type,
            split=splits[item.id],
            abstained=bool(final.get("abstained", False)),
            cited=len(final.get("cited", [])),
            answer=" ".join(str(final.get("answer", "")).split())[:160],
            seconds=clock() - started,
            weak=sum(1 for check in final.get("citation_checks", []) if check.get("weak")),
        )
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def summarize(rows: Sequence[AbstentionRow]) -> dict[str, Any]:
    """The counts that matter, overall and per split: correct abstentions, wrong answers, wrong refusals."""

    def counts(chosen: Sequence[AbstentionRow]) -> dict[str, int]:
        unanswerable = [r for r in chosen if r.should_abstain]
        answerable = [r for r in chosen if not r.should_abstain]
        return {
            "unanswerable": len(unanswerable),
            "refused_correctly": sum(r.abstained for r in unanswerable),
            "answered_wrongly": sum(not r.abstained for r in unanswerable),
            "answerable": len(answerable),
            "answered": sum(not r.abstained for r in answerable),
            "refused_wrongly": sum(r.abstained for r in answerable),
            # does the citation relevance check flag the wrong answers, and how often a good one?
            "wrong_answers_with_a_weak_citation": sum(
                (not r.abstained) and r.weak > 0 for r in unanswerable
            ),
            "good_answers_with_a_weak_citation": sum(
                (not r.abstained) and r.weak > 0 for r in answerable
            ),
        }

    return {
        "all": counts(rows),
        "dev": counts([r for r in rows if r.split == "dev"]),
        "test": counts([r for r in rows if r.split == "test"]),
    }
