"""Tests for step C4: the mode router, a pure function of the thread (the ✍️ piece of this step).

``choose_mode`` decides the mode of a turn from the mode the thread already has and the one the caller
asks for; the two routing functions turn that mode and the thread's history into the next node. None
of them touches a model, a file or the graph, so the whole behaviour is a table.
"""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.mode import (
    MODES,
    ModeError,
    choose_mode,
    route_after_begin,
    route_after_condense,
)

# --- choose_mode: the table ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("saved", "requested", "expected"),
    [
        (None, None, "chat"),  # nothing said anywhere: the default, what every thread was before C4
        ("", "", "chat"),  # the empty string means "not given" (the state resets to it)
        (None, "chat", "chat"),
        (None, "csv", "csv"),  # a first turn takes the mode it asks for
        ("chat", None, "chat"),  # a later turn that asks for nothing keeps the mode
        ("csv", None, "csv"),
        ("csv", "", "csv"),
        ("chat", "chat", "chat"),
        ("csv", "csv", "csv"),  # asking for the mode the thread has is fine
    ],
)
def test_the_mode_of_a_turn(saved: str | None, requested: str | None, expected: str) -> None:
    assert choose_mode(saved, requested) == expected


@pytest.mark.parametrize(("saved", "requested"), [("chat", "csv"), ("csv", "chat")])
def test_a_thread_never_changes_mode_and_the_error_names_both(saved: str, requested: str) -> None:
    with pytest.raises(ModeError) as raised:
        choose_mode(saved, requested)
    message = str(raised.value)
    assert f"a {saved} conversation" in message
    assert f"become a {requested} one" in message
    assert "new thread name" in message  # and says what to do instead


@pytest.mark.parametrize("name", ["CSV", "Chat", "table", "csv ", "agent", "0"])
def test_a_name_that_is_not_a_mode_is_an_error_whichever_argument_carries_it(name: str) -> None:
    with pytest.raises(ModeError, match="unknown mode"):
        choose_mode(None, name)
    with pytest.raises(ModeError, match="unknown saved mode"):
        choose_mode(name, None)
    with pytest.raises(ModeError, match="unknown"):
        choose_mode("chat", name)  # a typo must not silently mean "keep chat"


def test_the_modes_are_chat_and_csv() -> None:
    assert MODES == ("chat", "csv")


# --- choose_mode: properties -----------------------------------------------------------------

GIVEN = st.none() | st.sampled_from(["", "chat", "csv"])


@given(GIVEN, GIVEN)
def test_the_answer_is_always_a_mode_or_an_error_never_anything_else(
    saved: str | None, requested: str | None
) -> None:
    try:
        assert choose_mode(saved, requested) in MODES
    except ModeError:
        assert saved
        assert requested
        assert saved != requested  # the only way to fail among valid names


@given(GIVEN, GIVEN)
def test_a_saved_mode_is_never_overridden(saved: str | None, requested: str | None) -> None:
    if saved:
        try:
            assert choose_mode(saved, requested) == saved
        except ModeError:
            assert requested
            assert requested != saved


@given(GIVEN, GIVEN)
def test_choosing_again_with_the_result_changes_nothing(
    saved: str | None, requested: str | None
) -> None:
    """What ``begin_turn`` stores as ``mode`` and the next turn feeds back in must be stable."""
    try:
        chosen = choose_mode(saved, requested)
    except ModeError:
        return
    assert choose_mode(chosen, requested) == chosen
    assert choose_mode(chosen, None) == chosen


# --- the routing functions -------------------------------------------------------------------

HISTORY: Any = [{"id": "t1-user"}]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"mode": "chat"}, "retrieve"),  # a first turn goes straight to its branch
        ({"mode": "csv"}, "csv_ask"),
        ({}, "retrieve"),  # no mode in the state is chat
        ({"mode": "chat", "history": []}, "retrieve"),
        ({"mode": "chat", "history": HISTORY}, "condense"),  # a follow-up is rewritten first
        ({"mode": "csv", "history": HISTORY}, "condense"),  # in either mode
    ],
)
def test_where_a_turn_goes_first(state: Any, expected: str) -> None:
    assert route_after_begin(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [({"mode": "chat"}, "retrieve"), ({"mode": "csv"}, "csv_ask"), ({}, "retrieve")],
)
def test_where_a_rewritten_follow_up_goes_next(state: Any, expected: str) -> None:
    assert route_after_condense(state) == expected


@given(st.sampled_from(["chat", "csv"]), st.booleans())
def test_a_turn_never_reaches_the_other_modes_branch(mode: str, has_history: bool) -> None:
    state: Any = {"mode": mode, "history": HISTORY if has_history else []}
    first = route_after_begin(state)
    last = first if first != "condense" else route_after_condense(state)
    assert last == ("csv_ask" if mode == "csv" else "retrieve")


def test_a_first_turn_makes_no_rewrite_in_either_mode() -> None:
    for mode in MODES:
        assert route_after_begin({"mode": mode}) != "condense"  # type: ignore[typeddict-item]
