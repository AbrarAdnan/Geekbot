"""Tests for step S2.4: reading the golden-questions markdown template into items."""

from pathlib import Path

import pytest

from geekbot.evals.datasets import GoldenItem, Source, dump_golden, parse_golden, validate_golden
from geekbot.evals.golden_md import import_markdown

TEMPLATE = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "golden-questions.md"

FILLED = """\
# Golden questions

Intro text with a table and a - **Question:** inside prose that is not a block.

## Single-hop (~50)

### S-001
- **Question:** How do I reset the router?
- **Where the answer is:** manual.md (Reset > Factory)
- **Must include:** hold the button; ten seconds
- **Notes:** tricky wording

### S-002
- **Question:**
- **Where the answer is:**
- **Must include:**
- **Notes:**

## Multi-hop (~15)

### M-001
- **Question:** Which plan covers the router and what does it cost?
- **Where the answer is:**
  - warranty.md (Coverage)
  - prices.md (p. 3; appendix)
- **Must include:**
  - two years
  - 49 euros

## Multi-turn (~15)

### T-001
- **Turn 1 (the earlier question):** What is the warranty?
- **Turn 2 (the follow-up that gets graded):** And for refurbished ones?
- **Where the answer is:** warranty.md (p. 2)
- **Must include:** 90 days

## Unanswerable (~20)

### U-001
- **Question:** Does it support Linux?
- **Why it can't be answered:** the manual only covers Windows and macOS
- **Notes:**
"""


def test_a_single_hop_block_is_read_field_by_field() -> None:
    first = import_markdown(FILLED).items[0]
    assert first == GoldenItem(
        id="S-001",
        type="single_hop",
        question="How do I reset the router?",
        sources=(Source("manual.md", "Reset > Factory"),),
        must_include=("hold the button", "ten seconds"),
        notes="tricky wording",
    )


def test_an_untouched_block_is_reported_not_imported() -> None:
    result = import_markdown(FILLED)
    assert result.blank == ("S-002",)
    assert [item.id for item in result.items] == ["S-001", "M-001", "T-001", "U-001"]
    assert result.problems == ()


def test_sub_bullets_make_lists_and_a_semicolon_inside_parentheses_is_kept() -> None:
    multi = import_markdown(FILLED).items[1]
    assert multi.sources == (
        Source("warranty.md", "Coverage"),
        Source("prices.md", "p. 3; appendix"),
    )
    assert multi.must_include == ("two years", "49 euros")


def test_a_multi_turn_block_keeps_the_earlier_turn_and_grades_the_follow_up() -> None:
    turn = import_markdown(FILLED).items[2]
    assert turn.type == "multi_turn"
    assert turn.earlier_turns == ("What is the warranty?",)
    assert turn.question == "And for refurbished ones?"


def test_the_highest_turn_is_the_graded_one() -> None:
    text = """\
### T-001
- **Turn 3 (the follow-up):** and in March?
- **Turn 1 (the earlier question):** total revenue by region?
- **Turn 2 (another):** and by month?
- **Where the answer is:** sales.md
- **Must include:** 42
"""
    item = import_markdown(text).items[0]
    assert item.earlier_turns == ("total revenue by region?", "and by month?")
    assert item.question == "and in March?"


def test_an_unanswerable_block_keeps_its_reason() -> None:
    item = import_markdown(FILLED).items[3]
    assert item.type == "unanswerable"
    assert item.reason == "the manual only covers Windows and macOS"
    assert item.sources == ()


def test_the_type_comes_from_the_id_not_from_the_section_heading() -> None:
    text = "## Multi-hop\n\n### S-005\n- **Question:** q?\n- **Where the answer is:** a.md\n"
    assert import_markdown(text).items[0].type == "single_hop"


def test_a_heading_that_is_not_an_item_id_is_a_problem() -> None:
    text = "### X-001\n- **Question:** q?\n\n### DRAFT-001 (single-hop)\n- **Question:** q?\n"
    result = import_markdown(text)
    assert result.items == ()
    assert [p.item_id for p in result.problems] == ["X-001", "DRAFT-001"]
    assert all(p.severity == "error" for p in result.problems)
    assert "line 1" in result.problems[0].message


def test_an_unknown_field_is_an_error_and_its_text_is_dropped() -> None:
    text = "### S-001\n- **Question:** q?\n- **Source page:** 4\n  continued\n- **Notes:** n\n"
    result = import_markdown(text)
    assert [p.message for p in result.problems] == ["line 3: unknown field 'Source page'"]
    assert result.items[0].notes == "n"


def test_turn_fields_on_a_non_turn_id_are_an_error() -> None:
    text = "### S-001\n- **Question:** q?\n- **Turn 1 (x):** earlier\n"
    result = import_markdown(text)
    assert any("Turn fields" in p.message for p in result.problems)


def test_html_comments_are_ignored() -> None:
    text = (
        "### S-001\n- **Question:** q? <!-- reword -->\n"
        "- **Where the answer is:** a.md\n- **Must include:** <!-- a hint -->\n"
    )
    item = import_markdown(text).items[0]
    assert item.question == "q?"
    assert item.must_include == ()


def test_a_block_with_only_comments_is_blank() -> None:
    result = import_markdown("### S-001\n- **Question:** <!-- later -->\n- **Notes:**\n")
    assert result.blank == ("S-001",)


def test_a_fenced_example_is_not_read() -> None:
    text = "```\n### S-001\n- **Question:** example?\n```\n\n### S-002\n- **Question:** real?\n"
    assert [item.id for item in import_markdown(text).items] == ["S-002"]


def test_a_rule_or_a_heading_ends_a_block() -> None:
    text = "### S-001\n- **Question:** q?\n---\n- **Notes:** belongs to nothing\n## Next\n- **Notes:** x"
    item = import_markdown(text).items[0]
    assert item.notes == ""


def test_a_question_may_continue_on_the_next_lines() -> None:
    text = "### S-001\n- **Question:** How do I\n  reset it,\n  please?\n"
    assert import_markdown(text).items[0].question == "How do I reset it, please?"


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("manual.md", Source("manual.md", "")),
        ("manual.md (p. 3)", Source("manual.md", "p. 3")),
        ("`manual.md` (Setup)", Source("manual.md", "Setup")),
        ("./docs/manual.md", Source("docs/manual.md", "")),
        ("docs\\manual.md (x)", Source("docs/manual.md", "x")),
        ("  manual.md   ( p. 3 )  ", Source("manual.md", "p. 3")),
        ("manual.md ()", Source("manual.md", "")),
    ],
)
def test_a_place_is_a_path_with_an_optional_locator(written: str, expected: Source) -> None:
    text = f"### S-001\n- **Question:** q?\n- **Where the answer is:** {written}\n"
    assert import_markdown(text).items[0].sources == (expected,)


def test_the_semicolon_splits_places_only_outside_parentheses() -> None:
    text = "### M-001\n- **Question:** q?\n- **Where the answer is:** a.md (x; y); b.md\n"
    assert import_markdown(text).items[0].sources == (Source("a.md", "x; y"), Source("b.md", ""))


def test_an_empty_file_has_no_items() -> None:
    result = import_markdown("")
    assert (result.items, result.blank, result.problems) == ((), (), ())


def test_a_filled_template_validates_and_round_trips_through_the_jsonl() -> None:
    items = import_markdown(FILLED).items
    report = validate_golden(items, corpus_files={"manual.md", "warranty.md", "prices.md"})
    assert report.ok
    assert parse_golden(dump_golden(items)) == list(items)


def test_the_template_in_the_repo_is_all_blank_and_has_no_problems() -> None:
    """The shipped template must stay importable: its worked example is fenced, its blocks empty."""
    result = import_markdown(TEMPLATE.read_text(encoding="utf-8"))
    assert result.items == ()
    assert result.blank == ("S-001", "M-001", "T-001", "U-001")
    assert result.problems == ()
