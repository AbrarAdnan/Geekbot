"""Tests for step S3.2: the citation check (this step's ✍️ piece).

The examples pin down each rule of ``check_citations`` one at a time; the properties then check what
must hold for *any* text the model could produce: the numbers it keeps are real sources, the numbers it
drops are not, valid text is left alone, and running the check again changes nothing.
"""

from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.citations import CitationCheck, check_citations


def check(answer: str, used: int = 3) -> CitationCheck:
    return check_citations(answer, used)


# --- marks that are fine ---------------------------------------------------------------------


def test_a_valid_mark_is_kept_exactly_as_written() -> None:
    result = check("Ninety days [1].", 2)
    assert result == CitationCheck(text="Ninety days [1].", cited=(1,), dropped=())


def test_every_source_from_one_to_used_is_valid() -> None:
    result = check("A [1] B [2] C [3].", 3)
    assert result.cited == (1, 2, 3)
    assert result.dropped == ()


def test_marks_are_listed_once_in_order_of_first_appearance() -> None:
    result = check("B [2], then A [1], then B again [2].", 2)
    assert result.cited == (2, 1)


def test_a_group_of_valid_numbers_is_left_untouched() -> None:
    for answer in ("Both [1, 2].", "Both [1,2].", "Both [2 , 1 , 2]."):
        assert check(answer).text == answer
    assert check("Both [2, 1, 2].").cited == (2, 1)


def test_two_marks_side_by_side_are_both_kept() -> None:
    result = check("Both [1][2].")
    assert (result.text, result.cited) == ("Both [1][2].", (1, 2))


def test_an_answer_with_no_marks_is_only_stripped() -> None:
    result = check("  I could not find it in the documents.\n")
    assert result == CitationCheck(
        text="I could not find it in the documents.", cited=(), dropped=()
    )


# --- marks that point nowhere ----------------------------------------------------------------


def test_a_mark_beyond_the_sources_is_removed_and_reported() -> None:
    result = check("Ninety days [1] and a year [7].", 2)
    assert result.text == "Ninety days [1] and a year."
    assert result.cited == (1,)
    assert result.dropped == (7,)


def test_zero_is_not_a_source() -> None:
    result = check("Ninety days [0].", 3)
    assert (result.text, result.cited, result.dropped) == ("Ninety days.", (), (0,))


def test_with_no_sources_every_mark_goes() -> None:
    for used in (0, -1):
        result = check("Anything [1] at all [2].", used)
        assert result.text == "Anything at all."
        assert result.cited == ()
        assert result.dropped == (1, 2)


def test_dropped_numbers_are_listed_once_in_order_of_first_appearance() -> None:
    result = check("a [9] b [8] c [9].", 1)
    assert result.dropped == (9, 8)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("days [9].", "days."),  # before punctuation: no space is left
        ("see [9] below", "see below"),  # between words: one space
        ("see [9]below", "see below"),  # glued to the next word: still one space
        ("days\t[9].", "days."),  # a tab counts as a space
        ("[9] Ninety days.", "Ninety days."),  # at the start: nothing is left in front
        ("a [9] [1].", "a [1]."),  # next to a valid mark: that one is kept
        ("one\n[9]\ntwo", "one\n\ntwo"),  # line breaks are not touched
    ],
)
def test_removing_a_mark_leaves_the_sentence_tidy(answer: str, expected: str) -> None:
    assert check(answer, 1).text == expected


# --- groups with some bad numbers ------------------------------------------------------------


def test_in_a_group_only_the_bad_numbers_go() -> None:
    result = check("Both [1, 7].", 2)
    assert (result.text, result.cited, result.dropped) == ("Both [1].", (1,), (7,))


def test_a_rebuilt_group_has_no_repeats() -> None:
    assert check("Both [2, 2, 9, 1].", 2).text == "Both [2, 1]."


def test_a_group_with_no_good_number_goes_entirely() -> None:
    result = check("None [7, 8].", 2)
    assert (result.text, result.cited, result.dropped) == ("None.", (), (7, 8))


# --- what is not a citation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "Cite as [n] please.",  # the placeholder from the prompt
        "Pages [1-3] apply.",  # a range is not a mark
        "A [-1] sign.",
        "A [a] letter.",
        "Leading zero [01].",  # only plain decimal numbers count
        "Too long [1234567890].",  # more than nine digits
        "Empty [] brackets.",
        "Two [1 2] numbers with no comma.",
        "Not closed [1",
    ],
)
def test_other_bracketed_text_is_left_alone(answer: str) -> None:
    result = check(answer, 3)
    assert result == CitationCheck(text=answer, cited=(), dropped=())


def test_links_are_not_this_checks_job() -> None:
    """``[1](url)`` keeps its mark here; stripping links and images is the output filter (S3.5)."""
    result = check("See [1](http://example.invalid/x).", 1)
    assert result.text == "See [1](http://example.invalid/x)."
    assert result.cited == (1,)


# --- removal must not forge a mark -----------------------------------------------------------


def test_removing_a_mark_cannot_create_a_new_one() -> None:
    """``[[7]1]`` loses its inner mark and would read ``[1]``: the second pass vets that too."""
    result = check("Odd [[7]1].", 2)
    assert result.text == "Odd [1]."
    assert result.cited == (1,)
    assert result.dropped == (7,)


def test_a_forged_mark_that_is_itself_invalid_is_removed_too() -> None:
    result = check("Odd [[7]9].", 2)
    assert result.text == "Odd."
    assert result.dropped == (7, 9)


# --- properties ------------------------------------------------------------------------------

MESSY = st.text(alphabet="ab .,\t\n[]0123456789-", max_size=60)
USED = st.integers(min_value=-1, max_value=6)


@given(MESSY, USED)
def test_the_kept_numbers_are_sources_and_the_dropped_ones_are_not(answer: str, used: int) -> None:
    result = check_citations(answer, used)
    assert all(1 <= number <= used for number in result.cited)
    assert all(not 1 <= number <= used for number in result.dropped)
    assert len(set(result.cited)) == len(result.cited)
    assert len(set(result.dropped)) == len(result.dropped)


@given(MESSY, USED)
def test_checking_a_checked_answer_changes_nothing(answer: str, used: int) -> None:
    first = check_citations(answer, used)
    again = check_citations(first.text, used)
    assert again.text == first.text
    assert again.cited == first.cited
    assert again.dropped == ()


@given(MESSY, USED)
def test_the_check_never_adds_a_digit(answer: str, used: int) -> None:
    """Every digit in the result was in the answer: nothing is invented, only removed."""
    result = check_citations(answer, used)
    before = Counter(character for character in answer if character.isdigit())
    after = Counter(character for character in result.text if character.isdigit())
    assert not after - before


@given(st.text(alphabet="ab .,\t\n]0123456789-", max_size=60), USED)
def test_text_without_an_opening_bracket_is_only_stripped(answer: str, used: int) -> None:
    result = check_citations(answer, used)
    assert result == CitationCheck(text=answer.strip(), cited=(), dropped=())


PIECES = st.one_of(
    st.sampled_from(["alpha", "beta,", "gamma."]),
    st.integers(min_value=1, max_value=5).map(lambda n: f"[{n}]"),
)


@given(st.lists(PIECES, max_size=12))
def test_an_answer_whose_marks_are_all_valid_comes_back_unchanged(pieces: list[str]) -> None:
    answer = " ".join(pieces)
    result = check_citations(answer, 5)
    assert result.text == answer
    assert result.dropped == ()
    expected = [int(piece[1:-1]) for piece in pieces if piece.startswith("[")]
    assert result.cited == tuple(dict.fromkeys(expected))
