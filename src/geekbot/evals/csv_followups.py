"""The CSV follow-up evaluation (step C6, PLAN.md §9): does a follow-up question still work in CSV mode?

The copied ``csv_analyst`` package has **no memory**: every question stands alone. A conversation
works only because ``condense`` (C4) rewrites a follow-up ("And what about profit?") into a question
that stands alone ("What is the total profit per region?") before the package sees it, with the CSV
model, in a single short call. INTEGRATION.md lists that rewrite as *unmeasured*; this file measures it.

**The set.** About 15 two-turn items on Gigalogy's two datasets (``sales``, ``hospital``), each with
the first question, a follow-up, and the **standalone question the follow-up means**, written by hand.
The expected result is computed from the CSV with pandas (``csv_followups_gold``), never copied from a
website, and compared with the package's own result by Gigalogy's grader (``csv_mode.grading``, C3):
the *result* is graded, not the SQL.

**Two paths per item, so a failure can be placed.**

- *the conversation*: question one, then the follow-up, through the real graph in CSV mode (the
  rewrite with the CSV model, then the package);
- *the oracle*: the hand-written standalone question put straight to the package.

If the oracle is right and the conversation is wrong, the rewrite lost something (**rewrite loss**). If
both are wrong, the package cannot do that question and no rewrite could have helped (a package
failure, not a rewrite failure). The difference is the number C6 exists to produce.

**The scoring rule is the 2-of-3 rule** of C3 (Gigalogy's): every item is run three times with
different seed sets and counts as *solved* when it is right in more than half of the runs, because a
small local model varies from run to run (C4 saw two turns disagree about "the region that sells the
most").

This file is the part that needs no model and no data: loading and checking the item file, the rule,
the summary and the report. Nothing here copies a row of the datasets into a result: the grader's note
is reduced to a short category (``mismatch``), because the result files are committed to a
repository and the datasets come from third parties.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DATASETS",
    "FollowUpItem",
    "FollowUpRun",
    "FollowUpSummary",
    "FollowUpsError",
    "format_followups_report",
    "load_followups",
    "note_category",
    "run_followups",
    "solved",
    "summarize_followups",
]

DATASETS: Final = ("sales", "hospital")
MIN_TURNS: Final = 2


class FollowUpsError(ValueError):
    """The item file is not usable: a bad line, a missing field, a repeated id, an unknown dataset."""


@dataclass(frozen=True)
class FollowUpItem:
    """One two-turn item. ``standalone`` is what the last turn means when read on its own."""

    id: str
    dataset: str
    turns: tuple[str, ...]  # the questions, in order; the last one is the follow-up that is graded
    standalone: str  # the last turn as a question that needs no history (the oracle's question)
    gold: str  # the key of the expected-result recipe in ``csv_followups_gold``
    note: str = ""

    @property
    def follow_up(self) -> str:
        return self.turns[-1]


def load_followups(path: Path, *, known_gold: Sequence[str] | None = None) -> list[FollowUpItem]:
    """Read and check a JSON Lines file of items. ``FollowUpsError`` says which line is wrong.

    Checks: every line is an object with the fields below and no others (a typo must not be dropped
    silently), ids are unique, the dataset is one of ``DATASETS``, there are at least two turns and
    none is empty, the standalone question differs from the follow-up as typed (an item whose
    follow-up already stands alone tests nothing), and, if ``known_gold`` is given, the recipe exists.
    """
    allowed = {"id", "dataset", "turns", "standalone", "gold", "note"}
    items: list[FollowUpItem] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise FollowUpsError(f"{where}: not JSON ({error})") from error
        if not isinstance(data, dict):
            raise FollowUpsError(f"{where}: not an object")
        extra = set(data) - allowed
        if extra:
            raise FollowUpsError(f"{where}: unknown field(s) {', '.join(sorted(extra))}")
        try:
            item = FollowUpItem(
                id=str(data["id"]),
                dataset=str(data["dataset"]),
                turns=tuple(str(turn) for turn in data["turns"]),
                standalone=str(data["standalone"]),
                gold=str(data["gold"]),
                note=str(data.get("note", "")),
            )
        except (KeyError, TypeError) as error:
            raise FollowUpsError(f"{where}: missing or malformed field {error}") from error
        if item.id in seen:
            raise FollowUpsError(f"{where}: the id {item.id} is used twice")
        seen.add(item.id)
        if item.dataset not in DATASETS:
            raise FollowUpsError(f"{where}: dataset {item.dataset!r} is not one of {DATASETS}")
        if len(item.turns) < MIN_TURNS or any(not turn.strip() for turn in item.turns):
            raise FollowUpsError(f"{where}: needs at least {MIN_TURNS} non-empty turns")
        if not item.standalone.strip() or item.standalone.strip() == item.follow_up.strip():
            raise FollowUpsError(
                f"{where}: the standalone question must be written out and differ from the follow-up"
            )
        if known_gold is not None and item.gold not in known_gold:
            raise FollowUpsError(f"{where}: no expected-result recipe called {item.gold!r}")
        items.append(item)
    if not items:
        raise FollowUpsError(f"{path.name} holds no items")
    return items


# --- one run of one item ---------------------------------------------------------------------


def note_category(note: str) -> str:
    """The grader's note reduced to a category that holds no data (see the module docstring).

    The grader's notes of a wrong answer quote the expected and the actual rows; those are rows of a
    third party's dataset and do not belong in a committed file.
    """
    if note.startswith("matches"):
        return "matches"
    if note.startswith("correctly abstained"):
        return "abstained"
    if note.startswith("false abstention"):
        return "false abstention"
    if note.startswith("HALLUCINATION"):
        return "hallucination"
    return "mismatch"


@dataclass(frozen=True)
class FollowUpRun:
    """One item on one seed set: both paths, and what the rewrite looked like."""

    item_id: str
    dataset: str
    seed_set: int
    conversation_correct: bool
    conversation_note: str  # a category from ``note_category``
    oracle_correct: bool
    oracle_note: str
    rewrite: str  # what the package was asked in the conversation (the rewrite, or the follow-up as typed)
    rewrite_unchanged: (
        bool  # the rewrite equals the follow-up as typed (the rewrite failed or did nothing)
    )
    rewrite_error: str  # why the rewrite fell back, "" if it did not
    seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "item": self.item_id,
            "dataset": self.dataset,
            "seed_set": self.seed_set,
            "conversation_correct": self.conversation_correct,
            "conversation_note": self.conversation_note,
            "oracle_correct": self.oracle_correct,
            "oracle_note": self.oracle_note,
            "rewrite": self.rewrite,
            "rewrite_unchanged": self.rewrite_unchanged,
            "rewrite_error": self.rewrite_error,
            "seconds": round(self.seconds, 2),
        }


def run_followups(
    items: Sequence[FollowUpItem],
    run_item: Callable[[FollowUpItem, int], FollowUpRun],
    *,
    runs: int,
    on_run: Callable[[FollowUpRun], None] | None = None,
) -> list[FollowUpRun]:
    """Every item on every seed set, in the order seed set, then item (so slow drift in the machine hits
    every item alike, as in C3). ``run_item`` does the work; an exception from it is not caught: a run
    that broke must not be counted as a wrong answer."""
    results: list[FollowUpRun] = []
    for seed_set in range(runs):
        for item in items:
            result = run_item(item, seed_set)
            results.append(result)
            if on_run is not None:
                on_run(result)
    return results


# --- the 2-of-3 rule and the summary ---------------------------------------------------------


def solved(correct_runs: int, runs: int) -> bool:
    """The rule of C3: right in more than half of the runs (2 of 3)."""
    return 2 * correct_runs > runs


@dataclass(frozen=True)
class FollowUpSummary:
    """What the runs say: solved counts on both paths, the loss between them, the rewrite's own habits."""

    items: int
    runs_per_item: int
    solved_conversation: int
    solved_oracle: int
    rewrite_loss: tuple[str, ...]  # items the oracle solves and the conversation does not
    package_failures: tuple[
        str, ...
    ]  # items neither path solves (the package cannot do the question)
    gained: tuple[str, ...]  # items the conversation solves and the oracle does not (noise or luck)
    unstable: tuple[str, ...]  # items right in some runs and wrong in others (either path)
    by_dataset: Mapping[
        str, tuple[int, int, int]
    ]  # dataset -> (items, conversation solved, oracle solved)
    runs_total: int
    rewrites_unchanged: int  # runs in which the rewrite equalled the follow-up as typed
    rewrite_errors: int  # runs in which the rewrite call failed and the follow-up was used as typed
    seconds_mean: float


def summarize_followups(
    items: Sequence[FollowUpItem], runs: Sequence[FollowUpRun], *, runs_per_item: int
) -> FollowUpSummary:
    """Apply the 2-of-3 rule to every item on both paths and place each failure."""
    by_item: dict[str, list[FollowUpRun]] = {}
    for run in runs:
        by_item.setdefault(run.item_id, []).append(run)
    loss: list[str] = []
    package: list[str] = []
    gained: list[str] = []
    unstable: list[str] = []
    per_dataset: dict[str, list[int]] = {name: [0, 0, 0] for name in DATASETS}
    conversation_solved = oracle_solved = 0
    for item in items:
        mine = by_item.get(item.id, [])
        conversation_right = sum(r.conversation_correct for r in mine)
        oracle_right = sum(r.oracle_correct for r in mine)
        conversation = solved(conversation_right, runs_per_item)
        oracle = solved(oracle_right, runs_per_item)
        conversation_solved += conversation
        oracle_solved += oracle
        bucket = per_dataset.setdefault(item.dataset, [0, 0, 0])
        bucket[0] += 1
        bucket[1] += conversation
        bucket[2] += oracle
        if oracle and not conversation:
            loss.append(item.id)
        elif not oracle and not conversation:
            package.append(item.id)
        elif conversation and not oracle:
            gained.append(item.id)
        if 0 < conversation_right < len(mine) or 0 < oracle_right < len(mine):
            unstable.append(item.id)
    seconds = [r.seconds for r in runs]
    return FollowUpSummary(
        items=len(items),
        runs_per_item=runs_per_item,
        solved_conversation=conversation_solved,
        solved_oracle=oracle_solved,
        rewrite_loss=tuple(loss),
        package_failures=tuple(package),
        gained=tuple(gained),
        unstable=tuple(unstable),
        by_dataset={name: (b[0], b[1], b[2]) for name, b in per_dataset.items() if b[0]},
        runs_total=len(runs),
        rewrites_unchanged=sum(r.rewrite_unchanged for r in runs),
        rewrite_errors=sum(1 for r in runs if r.rewrite_error),
        seconds_mean=sum(seconds) / len(seconds) if seconds else 0.0,
    )


# --- the report ------------------------------------------------------------------------------


def format_followups_report(
    meta: Mapping[str, Any],
    items: Sequence[FollowUpItem],
    runs: Sequence[FollowUpRun],
    summary: FollowUpSummary,
) -> str:
    """The Markdown report: the two solved counts, where the failures sit, and the failing rewrites.

    Names the CSV model, says it is a measurement of this model on these 15 items on two public datasets
    (not a general claim), and that the item set and its gold recipes are written by Claude.
    """
    lines: list[str] = []
    add = lines.append
    add(f"# CSV follow-up evaluation: {meta.get('name', '?')}")
    add("")
    add(
        f"CSV model `{meta.get('model', '?')}` (profile `{meta.get('profile', '?')}`); "
        f"{summary.items} two-turn items on Gigalogy's `sales` and `hospital` datasets, "
        f"{summary.runs_per_item} runs each, {meta.get('created', '?')}. The items and the expected "
        "results (computed with pandas) are Claude-authored; the grader is Gigalogy's, ported in C3. "
        "A measurement of this model on a small set, not a general claim."
    )
    add("")
    add("## Result (the 2-of-3 rule: solved = right in more than half of the runs)")
    add("| path | solved |")
    add("|---|---|")
    add(
        f"| **the conversation** (question, follow-up, rewrite, package) | **{summary.solved_conversation} of {summary.items}** |"
    )
    add(
        f"| **the oracle** (the hand-written standalone question straight to the package) | **{summary.solved_oracle} of {summary.items}** |"
    )
    add("")
    add(
        f"**Rewrite loss** (the oracle solves it, the conversation does not): "
        f"{len(summary.rewrite_loss)} items {_ids(summary.rewrite_loss)}. "
        f"**Package failures** (neither path solves it; no rewrite could have helped): "
        f"{len(summary.package_failures)} items {_ids(summary.package_failures)}. "
        f"Solved only through the conversation: {len(summary.gained)} {_ids(summary.gained)}. "
        f"Unstable (right in some runs, wrong in others): {len(summary.unstable)} {_ids(summary.unstable)}."
    )
    add("")
    add("| dataset | items | conversation solved | oracle solved |")
    add("|---|---|---|---|")
    for name, (count, conv, oracle) in summary.by_dataset.items():
        add(f"| {name} | {count} | {conv} | {oracle} |")
    add("")
    add(
        f"The rewrite equalled the follow-up as typed in {summary.rewrites_unchanged} of "
        f"{summary.runs_total} runs ({summary.rewrite_errors} of them because the rewrite call failed). "
        f"Mean time per run (both paths): {summary.seconds_mean:.1f} s."
    )
    failing = set(summary.rewrite_loss) | set(summary.package_failures)
    if failing:
        add("")
        add("## Failing items: what was asked")
        add(
            "| item | follow-up as typed | what the package was asked (first failing run) | standalone, by hand |"
        )
        add("|---|---|---|---|")
        by_id = {item.id: item for item in items}
        for item_id in sorted(failing):
            item = by_id[item_id]
            bad = next(
                (r for r in runs if r.item_id == item_id and not r.conversation_correct), None
            )
            asked = "n/a" if bad is None else bad.rewrite
            add(f"| {item_id} | {item.follow_up} | {asked} | {item.standalone} |")
    return "\n".join(lines) + "\n"


def _ids(ids: Sequence[str]) -> str:
    return f"({', '.join(ids)})" if ids else ""
