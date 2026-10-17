"""Tests for step S3.2: the labels of PLAN.md §8.2 (``Label``, ``is_tainted``).

The rule worth a property test is the one the whole taint model rests on: taint only ever grows. A
prompt that holds two kinds of text is tainted if either kind is, and adding a label can never make
something untainted.
"""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.domain import TAINTED_LABELS, Label, is_tainted


def test_the_plan_names_five_labels() -> None:
    assert {label.value for label in Label} == {"USER", "KB", "WEB", "TOOL", "UNKNOWN"}


def test_a_label_is_its_name_as_a_string_so_the_state_can_hold_it_as_plain_data() -> None:
    assert Label.KB == "KB"
    assert Label("WEB") is Label.WEB
    assert json.loads(json.dumps([Label.KB, "KB"])) == ["KB", "KB"]


def test_only_the_user_is_trusted() -> None:
    assert TAINTED_LABELS == {Label.KB, Label.WEB, Label.TOOL, Label.UNKNOWN}
    assert Label.USER not in TAINTED_LABELS


def test_nothing_and_the_user_alone_are_not_tainted() -> None:
    assert not is_tainted([])
    assert not is_tainted([Label.USER])


@pytest.mark.parametrize("label", [Label.KB, Label.WEB, Label.TOOL, Label.UNKNOWN])
def test_any_outside_label_taints(label: Label) -> None:
    assert is_tainted([label])
    assert is_tainted([Label.USER, label])


LABELS = st.frozensets(st.sampled_from(list(Label)))


@given(LABELS, LABELS)
def test_taint_of_a_union_is_the_or_of_the_taints(
    first: frozenset[Label], second: frozenset[Label]
) -> None:
    assert is_tainted(first | second) == (is_tainted(first) or is_tainted(second))


@given(LABELS, st.sampled_from(list(Label)))
def test_adding_a_label_never_clears_taint(labels: frozenset[Label], extra: Label) -> None:
    if is_tainted(labels):
        assert is_tainted(labels | {extra})
