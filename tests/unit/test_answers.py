"""Tests for step S5.2: the deterministic answer checks (``geekbot.evals.answers``), with a fake graph."""

import json
from pathlib import Path
from typing import Any

import pytest

from geekbot.evals.answers import (
    AnswerRecord,
    SourceView,
    corpus_relative,
    format_answers_report,
    generate_answers,
    load_answers,
    record_from_state,
    save_answers,
    score_answer,
    summarize_scores,
)
from geekbot.evals.datasets import GoldenItem, Source

CORPUS = Path("evals/corpus")


def item(
    identifier: str = "S-001",
    kind: str = "single_hop",
    *,
    sources: tuple[str, ...] = ("staff/staff-faq.html",),
    phrases: tuple[str, ...] = ("90 days",),
    earlier: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=identifier,
        type=kind,  # type: ignore[arg-type]
        question=f"question {identifier}?",
        sources=tuple(Source(path) for path in sources),
        must_include=phrases,
        earlier_turns=earlier,
    )


def source(n: int, path: str = "staff/staff-faq.html", text: str = "kept 90 days") -> SourceView:
    return SourceView(n, path, "title", "", text)


def record(
    identifier: str = "S-001",
    kind: str = "single_hop",
    *,
    answer: str = "Chat history is kept for 90 days [1].",
    abstained: bool = False,
    cited: tuple[int, ...] = (1,),
    dropped: tuple[int, ...] = (),
    sources: tuple[SourceView, ...] | None = None,
    split: str = "dev",
) -> AnswerRecord:
    return AnswerRecord(
        id=identifier,
        type=kind,
        split=split,
        question=f"question {identifier}?",
        earlier_turns=(),
        standalone="",
        answer=answer,
        raw_answer=answer,
        abstained=abstained,
        cited=cited,
        dropped=dropped,
        sources=(source(1),) if sources is None else sources,
        seconds=1.5,
        first_token_s=0.4,
        output_tokens=30,
    )


# --- paths -----------------------------------------------------------------------------------


def test_a_path_inside_the_corpus_is_made_relative_with_forward_slashes() -> None:
    inside = str((CORPUS / "staff" / "glossary.md").resolve())
    assert corpus_relative(inside, CORPUS) == "staff/glossary.md"


def test_a_path_outside_the_corpus_keeps_only_its_file_name() -> None:
    assert corpus_relative(str(Path("C:/somewhere/else/secret.md")), CORPUS) == "secret.md"


# --- from the graph's state to a record ------------------------------------------------------


def state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "answer": "Kept for 90 days [1].",
        "raw_answer": "Kept for 90 days [1][7].",
        "abstained": False,
        "cited": [1],
        "dropped_citations": [7],
        "used": 2,
        "standalone_question": "How long is chat history kept?",
        "generation": {"first_token_seconds": 0.42, "output_tokens": 31.0},
        "sources": [
            {
                "n": n,
                "chunk_id": f"c{n}",
                "title": "FAQ",
                "path": str((CORPUS / "staff" / "staff-faq.html").resolve()),
                "page": None,
                "section": "",
                "text": f"text {n}",
                "score": 1.0,
            }
            for n in (1, 2, 3)
        ],
    }
    base.update(overrides)
    return base


def test_a_record_keeps_only_the_sources_the_prompt_held_with_corpus_relative_paths() -> None:
    made = record_from_state(item(), "dev", state(), corpus=CORPUS, seconds=1.2)
    assert [s.n for s in made.sources] == [1, 2]  # `used` is 2: the third was never shown
    assert made.sources[0].path == "staff/staff-faq.html"
    assert made.cited == (1,) and made.dropped == (7,)
    assert made.first_token_s == 0.42 and made.output_tokens == 31 and made.seconds == 1.2


def test_the_rewrite_is_kept_only_for_a_follow_up() -> None:
    first = record_from_state(item(), "dev", state(), corpus=CORPUS, seconds=1.0)
    follow = record_from_state(
        item("T-001", "multi_turn", earlier=("earlier?",)),
        "test",
        state(),
        corpus=CORPUS,
        seconds=1.0,
    )
    assert first.standalone == ""
    assert follow.standalone == "How long is chat history kept?" and follow.split == "test"


def test_a_state_with_nothing_in_it_still_makes_a_record() -> None:
    made = record_from_state(item(), "dev", {}, corpus=CORPUS, seconds=0.1)
    assert made.answer == "" and made.sources == () and not made.abstained
    assert made.first_token_s is None and made.output_tokens is None


def test_a_record_survives_json() -> None:
    original = record_from_state(item(), "dev", state(), corpus=CORPUS, seconds=1.2)
    assert AnswerRecord.from_json(json.loads(json.dumps(original.to_json()))) == original


# --- generating ------------------------------------------------------------------------------


def clock_of(*values: float) -> Any:
    ticks = iter(values)
    return lambda: next(ticks)


def test_every_item_is_asked_once_in_order_and_each_record_is_reported() -> None:
    asked: list[str] = []

    def ask(golden: GoldenItem) -> dict[str, Any]:
        asked.append(golden.id)
        return state()

    seen: list[tuple[str, bool]] = []
    records = generate_answers(
        [item("S-001"), item("S-002")],
        {"S-001": "dev", "S-002": "test"},
        ask,
        corpus=CORPUS,
        clock=clock_of(0.0, 1.0, 1.0, 3.0),
        on_record=lambda r, reused: seen.append((r.id, reused)),
    )
    assert asked == ["S-001", "S-002"]
    assert [(r.id, r.split) for r in records] == [("S-001", "dev"), ("S-002", "test")]
    assert [r.seconds for r in records] == [1.0, 2.0]
    assert seen == [("S-001", False), ("S-002", False)]


def test_an_item_that_already_has_a_record_is_not_asked_again() -> None:
    kept = record("S-001", answer="kept earlier")
    asked: list[str] = []

    def ask(golden: GoldenItem) -> dict[str, Any]:
        asked.append(golden.id)
        return state()

    records = generate_answers(
        [item("S-001"), item("S-002")],
        {"S-001": "dev", "S-002": "dev"},
        ask,
        corpus=CORPUS,
        clock=clock_of(0.0, 1.0),
        existing={"S-001": kept},
    )
    assert asked == ["S-002"]
    assert records[0] is kept


def test_a_graph_that_breaks_is_not_counted_as_a_refusal() -> None:
    def ask(golden: GoldenItem) -> dict[str, Any]:
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="model crashed"):
        generate_answers([item()], {"S-001": "dev"}, ask, corpus=CORPUS, clock=clock_of(0.0, 1.0))


# --- the checks ------------------------------------------------------------------------------


def test_an_answer_that_cites_the_golden_document_is_a_hit_with_full_recall() -> None:
    scores = score_answer(item(), record())
    assert scores.abstention_correct and scores.answered
    assert scores.citation_hit is True and scores.citation_recall == 1.0
    assert scores.must_include == 1.0 and scores.invalid_citations == 0


def test_an_answer_citing_another_document_is_not_a_hit() -> None:
    scores = score_answer(item(), record(sources=(source(1, "ops/log_cleanup.py"),)))
    assert scores.citation_hit is False and scores.citation_recall == 0.0


def test_a_multi_hop_answer_is_scored_on_the_share_of_golden_documents_it_cites() -> None:
    golden = item("M-001", "multi_hop", sources=("a.md", "b.md"), phrases=("one", "two"))
    only_a = record(
        "M-001",
        "multi_hop",
        answer="one [1]",
        cited=(1,),
        sources=(source(1, "a.md"), source(2, "b.md")),
    )
    scores = score_answer(golden, only_a)
    assert scores.citation_hit is True and scores.citation_recall == 0.5
    assert scores.must_include == 0.5


def test_a_cited_number_the_prompt_never_held_counts_for_nothing() -> None:
    scores = score_answer(item(), record(cited=(5,), sources=(source(1),)))
    assert scores.citation_hit is False


def test_a_wrong_refusal_of_an_answerable_question_scores_zero_everywhere() -> None:
    refused = record(
        answer="I could not find it in the documents.", abstained=True, cited=(), sources=()
    )
    scores = score_answer(item(), refused)
    assert not scores.abstention_correct and not scores.answered
    assert scores.citation_hit is False and scores.citation_recall == 0.0
    assert scores.must_include == 0.0


def test_a_right_refusal_of_an_unanswerable_question() -> None:
    golden = item("U-001", "unanswerable", sources=(), phrases=())
    refused = record(
        "U-001",
        "unanswerable",
        answer="I could not find it in the documents.",
        abstained=True,
        cited=(),
        sources=(),
    )
    scores = score_answer(golden, refused)
    assert scores.abstention_correct and not scores.answered
    assert scores.citation_hit is None and scores.citation_recall is None
    assert scores.must_include is None  # an unanswerable item asks for no phrase


def test_an_answer_to_an_unanswerable_question_is_the_dangerous_mistake() -> None:
    golden = item("U-002", "unanswerable", sources=(), phrases=())
    scores = score_answer(golden, record("U-002", "unanswerable", answer="It is 42 [1]."))
    assert not scores.abstention_correct and scores.answered


def test_removed_citations_are_counted() -> None:
    assert score_answer(item(), record(dropped=(7, 9))).invalid_citations == 2


# --- the summary -----------------------------------------------------------------------------


def test_the_summary_has_counts_next_to_every_group() -> None:
    items = [
        item("S-001"),
        item("S-002", sources=("ops/runbook.md",)),
        item("U-001", "unanswerable", sources=(), phrases=()),
        item("U-002", "unanswerable", sources=(), phrases=()),
    ]
    records = [
        record("S-001"),
        record(
            "S-002",
            answer="I could not find it in the documents.",
            abstained=True,
            cited=(),
            sources=(),
        ),
        record(
            "U-001",
            "unanswerable",
            answer="I could not find it.",
            abstained=True,
            cited=(),
            sources=(),
            split="test",
        ),
        record("U-002", "unanswerable", answer="It is 42 [1].", dropped=(4,), split="test"),
    ]
    scores = [score_answer(g, r) for g, r in zip(items, records, strict=True)]
    summary = summarize_scores(scores)
    everything = summary["all"]
    assert everything["n"] == 4 and everything["answerable"] == 2
    assert everything["answerable_refused"] == 1 and everything["answerable_answered"] == 1
    assert everything["unanswerable_refused"] == 1 and everything["unanswerable_answered"] == 1
    assert everything["citation_hit"] == 1 and everything["citation_hit_of"] == 1
    assert everything["answers_with_an_invalid_citation"] == 1 and everything["answers"] == 2
    assert summary["single_hop"]["n"] == 2 and summary["unanswerable"]["n"] == 2
    assert summary["dev"]["n"] == 2 and summary["test"]["n"] == 2
    assert summary["multi_hop"]["n"] == 0 and summary["multi_hop"]["must_include_mean"] is None


def test_a_summary_of_nothing_is_empty_not_an_error() -> None:
    assert summarize_scores([])["all"]["n"] == 0


# --- files and the report --------------------------------------------------------------------


def test_the_answers_file_round_trips_and_a_bad_file_says_so(tmp_path: Path) -> None:
    path = tmp_path / "run" / "answers.json"
    records = [record("S-001"), record("S-002")]
    save_answers(path, {"name": "r", "model": "gemma4:12b"}, records)
    meta, loaded = load_answers(path)
    assert loaded == records and meta["model"] == "gemma4:12b" and meta["version"] == 1
    (tmp_path / "bad.json").write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an answers file"):
        load_answers(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="not an answers file"):
        load_answers(tmp_path / "missing.json")


def test_the_report_has_the_checks_the_judge_and_the_labels_each_in_its_place() -> None:
    summary = summarize_scores([score_answer(item(), record())])
    judge = {
        "model": "qwen2.5-coder:7b",
        "family": "qwen",
        "generator_family": "gemma",
        "judged": 10,
        "skipped_refusals": 2,
        "supported": 8,
        "answers_question": 9,
        "by_type": {"single_hop": {"judged": 10, "supported": 8, "answers_question": 9}},
    }
    kappa = {
        "verdict": "no verdict yet: 3 labels",
        "counts": {
            "both_supported": 1,
            "both_unsupported": 1,
            "human_supported_judge_not": 1,
            "judge_supported_human_not": 0,
        },
    }
    text = format_answers_report(
        {"name": "r", "model": "gemma4:12b", "profile": "target", "top_k": 5}, summary, judge, kappa
    )
    for wanted in (
        "# Answer evaluation: r",
        "gemma4:12b",
        "provisional",
        "primary evidence",
        "| all | 1 | 1 of 1 |",
        "supporting evidence only",
        "family `qwen`",
        "Supported",
        "no verdict yet",
        "both supported 1",
    ):
        assert wanted in text, wanted


def test_a_report_without_a_judge_or_labels_has_only_the_deterministic_part() -> None:
    text = format_answers_report({"name": "r"}, summarize_scores([]), None, None)
    assert "Deterministic checks" in text
    assert "## The judge" not in text and "## Agreement" not in text
