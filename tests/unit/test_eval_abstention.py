"""Tests for measuring abstention on the golden set (step S3.5, D-082): the counting, not a model."""

from collections.abc import Mapping
from typing import Any

from geekbot.evals.abstention import AbstentionRow, run_abstention, summarize
from geekbot.evals.datasets import GoldenItem, Source


def item(item_id: str, kind: str) -> GoldenItem:
    sources = () if kind == "unanswerable" else (Source("a.txt"),)
    return GoldenItem(
        item_id, kind, f"question {item_id}", sources=sources, reason="r" if sources == () else ""
    )  # type: ignore[arg-type]


ITEMS = [
    item("S-001", "single_hop"),
    item("S-002", "single_hop"),
    item("M-001", "multi_hop"),
    item("T-001", "multi_turn"),
    item("U-001", "unanswerable"),
    item("U-002", "unanswerable"),
]
SPLITS: Mapping[str, Any] = {
    "S-001": "dev",
    "S-002": "test",
    "M-001": "test",
    "T-001": "dev",
    "U-001": "dev",
    "U-002": "test",
}


def fake_ask(question: str) -> dict[str, Any]:
    # the graph refuses U-001 (right), answers U-002 (wrong), refuses S-002 (wrong), answers the rest
    refused = question in {"question U-001", "question S-002"}
    return {"abstained": refused, "cited": [] if refused else [1], "answer": f"reply to {question}"}


def test_every_question_but_the_multi_turn_ones_is_asked_once() -> None:
    asked: list[str] = []
    ticks = iter(range(100))

    def ask(question: str) -> dict[str, Any]:
        asked.append(question)
        return fake_ask(question)

    rows = run_abstention(ITEMS, SPLITS, ask, clock=lambda: float(next(ticks)))
    assert asked == [
        "question S-001",
        "question S-002",
        "question M-001",
        "question U-001",
        "question U-002",
    ]
    assert [r.id for r in rows] == ["S-001", "S-002", "M-001", "U-001", "U-002"]
    assert all(r.seconds == 1.0 for r in rows)  # a fake clock: two reads per question


def test_a_row_is_correct_when_the_reply_matches_what_the_question_needs() -> None:
    rows = run_abstention(ITEMS, SPLITS, fake_ask, clock=lambda: 0.0)
    verdicts = {r.id: r.correct for r in rows}
    assert verdicts == {
        "S-001": True,
        "S-002": False,  # an answerable question refused
        "M-001": True,
        "U-001": True,  # an unanswerable question refused
        "U-002": False,  # an unanswerable question answered
    }
    assert rows[0].cited == 1 and rows[3].cited == 0


def test_the_summary_counts_the_three_things_that_matter_overall_and_by_split() -> None:
    summary = summarize(run_abstention(ITEMS, SPLITS, fake_ask, clock=lambda: 0.0))
    assert summary["all"] == {
        "unanswerable": 2,
        "refused_correctly": 1,
        "answered_wrongly": 1,
        "answerable": 3,
        "answered": 2,
        "refused_wrongly": 1,
        "wrong_answers_with_a_weak_citation": 0,
        "good_answers_with_a_weak_citation": 0,
    }
    assert summary["dev"]["unanswerable"] == 1 and summary["dev"]["refused_correctly"] == 1
    assert summary["test"]["answered_wrongly"] == 1 and summary["test"]["refused_wrongly"] == 1


def test_weak_citations_are_counted_for_wrong_answers_and_for_good_ones() -> None:
    def ask(question: str) -> dict[str, Any]:
        weak = {"question U-002": 1, "question S-001": 2}.get(question, 0)
        return {
            "abstained": question == "question U-001",
            "cited": [1],
            "answer": "x",
            "citation_checks": [{"n": 1, "weak": index < weak} for index in range(2)],
        }

    rows = run_abstention(ITEMS, SPLITS, ask, clock=lambda: 0.0)
    assert {r.id: r.weak for r in rows} == {
        "S-001": 2,
        "S-002": 0,
        "M-001": 0,
        "U-001": 0,
        "U-002": 1,
    }
    summary = summarize(rows)["all"]
    assert summary["wrong_answers_with_a_weak_citation"] == 1  # U-002 was answered and flagged
    assert summary["good_answers_with_a_weak_citation"] == 1  # S-001 was answered well and flagged


def test_a_question_the_graph_cannot_answer_is_an_error_not_a_refusal() -> None:
    def broken(question: str) -> dict[str, Any]:
        raise RuntimeError("the model is down")

    try:
        run_abstention(ITEMS, SPLITS, broken, clock=lambda: 0.0)
    except RuntimeError as error:
        assert "model is down" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a crash was swallowed")


def test_the_answer_is_kept_short_and_on_one_line() -> None:
    long_reply = {"abstained": False, "cited": [1], "answer": "word\n  " * 100}
    (row,) = run_abstention([ITEMS[0]], SPLITS, lambda q: long_reply, clock=lambda: 0.0)
    assert "\n" not in row.answer and len(row.answer) <= 160


def test_the_row_type_is_plain_data() -> None:
    row = AbstentionRow("U-001", "unanswerable", "dev", True, 0, "I could not find it", 1.5)
    assert row.should_abstain and row.correct
