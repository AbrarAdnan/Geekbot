"""Tests for step S3.1: packing, the only place a prompt is assembled.

The budget rule is the one worth a property test: whatever the sources and the budget, the model must
see sources 1 to ``used`` with no gap and no more than the budget allows, and a bigger budget must
never show fewer.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.packing import SYSTEM_PROMPT, pack_prompt, source_block
from geekbot.agent.state import SourceRef
from geekbot.ingest.chunking import estimate_tokens


def source(
    n: int,
    text: str = "Some text.",
    *,
    title: str = "Handbook",
    section: str = "",
    page: int | None = None,
) -> SourceRef:
    return SourceRef(
        n=n,
        chunk_id=f"chunk{n}",
        title=title,
        path=f"/docs/{title}.md",
        page=page,
        section=section,
        text=text,
        score=1.0 / n,
    )


def user_text(question: str, sources: list[SourceRef], budget: int = 3000) -> str:
    packed = pack_prompt(question, sources, budget_tokens=budget)
    return str(packed.messages[1].content)


# --- the layout ------------------------------------------------------------------------------


def test_the_prompt_is_a_system_message_then_one_user_message() -> None:
    packed = pack_prompt("Why?", [source(1)], budget_tokens=3000)
    assert [message.type for message in packed.messages] == ["system", "human"]
    assert packed.messages[0].content == SYSTEM_PROMPT


def test_the_system_message_is_the_same_for_every_question() -> None:
    """A fixed prefix is what lets Ollama reuse its work on it (PLAN.md §6.5)."""
    first = pack_prompt("One?", [source(1)], budget_tokens=3000)
    second = pack_prompt("Two?", [source(1), source(2)], budget_tokens=3000)
    assert first.messages[0].content == second.messages[0].content


def test_the_user_message_has_the_numbered_sources_and_the_question_last() -> None:
    text = user_text("How long?", [source(1, "First body."), source(2, "Second body.")])
    assert text.startswith("Sources:\n\n[1] Handbook")
    assert text.index("[1] Handbook") < text.index("First body.") < text.index("[2] Handbook")
    assert text.index("[2] Handbook") < text.index("Second body.")
    assert text.endswith("\n\nQuestion: How long?")


def test_a_source_header_shows_the_title_the_section_and_the_page() -> None:
    block = source_block(source(3, "Body.", title="Guide", section="Install > Windows", page=7))
    assert block.splitlines()[0] == "[3] Guide | Install > Windows | page 7"


def test_a_source_without_section_or_page_shows_only_its_title() -> None:
    assert source_block(source(1, title="Notes")).splitlines()[0] == "[1] Notes"


def test_the_text_of_a_source_sits_between_fences() -> None:
    assert source_block(source(1, "Body line.")).splitlines()[1:] == ["<<<", "Body line.", ">>>"]


def test_the_system_prompt_asks_for_citations_and_for_honesty_about_gaps() -> None:
    assert "[n]" in SYSTEM_PROMPT
    assert "could not find" in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT  # the sources are data, not orders


# --- the budget ------------------------------------------------------------------------------


def costly(n: int) -> SourceRef:
    return source(n, "word " * 50)


def test_sources_are_added_in_order_until_the_budget_is_used() -> None:
    sources = [costly(1), costly(2), costly(3)]
    cost = estimate_tokens(source_block(sources[0]))
    assert estimate_tokens(source_block(sources[1])) == cost  # same shape, same cost
    assert pack_prompt("q", sources, budget_tokens=3 * cost).used == 3
    assert pack_prompt("q", sources, budget_tokens=3 * cost - 1).used == 2
    assert pack_prompt("q", sources, budget_tokens=2 * cost).used == 2
    assert pack_prompt("q", sources, budget_tokens=cost).used == 1
    assert pack_prompt("q", sources, budget_tokens=cost - 1).used == 0


def test_the_packer_stops_at_the_first_source_that_does_not_fit() -> None:
    """It must not skip to a later, smaller one: that would leave a gap in the numbers."""
    big, small = source(2, "word " * 400), source(3, "tiny")
    sources = [source(1, "short"), big, small]
    budget = estimate_tokens(source_block(sources[0])) + estimate_tokens(source_block(small))
    packed = pack_prompt("q", sources, budget_tokens=budget)
    assert packed.used == 1
    assert "[3]" not in str(packed.messages[1].content)


def test_nothing_given_or_nothing_fitting_says_so_and_counts_zero() -> None:
    assert pack_prompt("q", [], budget_tokens=100).used == 0
    assert "(no sources)" in user_text("q", [], 100)
    too_big = pack_prompt("q", [source(1, "word " * 500)], budget_tokens=10)
    assert too_big.used == 0
    assert "(no sources)" in str(too_big.messages[1].content)


@pytest.mark.parametrize("budget", [0, -5])
def test_a_budget_below_one_is_refused(budget: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        pack_prompt("q", [source(1)], budget_tokens=budget)


TEXTS = st.lists(st.text(alphabet="abc .", max_size=300), max_size=8)


@given(TEXTS, st.integers(min_value=1, max_value=1500))
def test_the_model_sees_exactly_the_longest_prefix_that_fits(texts: list[str], budget: int) -> None:
    sources = [source(n, text) for n, text in enumerate(texts, start=1)]
    packed = pack_prompt("q", sources, budget_tokens=budget)
    expected, spent = 0, 0
    for item in sources:
        spent += estimate_tokens(source_block(item))
        if spent > budget:
            break
        expected += 1
    assert packed.used == expected
    shown = str(packed.messages[1].content)
    for item in sources[:expected]:
        assert f"[{item['n']}] Handbook" in shown
    for item in sources[expected:]:
        assert f"[{item['n']}] Handbook" not in shown


@given(TEXTS, st.integers(min_value=1, max_value=1500), st.integers(min_value=0, max_value=500))
def test_a_bigger_budget_never_shows_fewer_sources(
    texts: list[str], budget: int, extra: int
) -> None:
    sources = [source(n, text) for n, text in enumerate(texts, start=1)]
    smaller = pack_prompt("q", sources, budget_tokens=budget).used
    bigger = pack_prompt("q", sources, budget_tokens=budget + extra).used
    assert bigger >= smaller
