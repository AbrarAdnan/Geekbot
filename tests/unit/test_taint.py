"""Tests for step S5.3: the taint rule (G2) and the labelling helpers around it.

``egress_decision`` is a pure function, so it is tested on every combination of its inputs: the table
of the cases that matter, and hypothesis properties that say what can never happen."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.packing import (
    message_labels,
    pack_history,
    pack_prompt,
    source_header,
    source_origin,
)
from geekbot.agent.state import HistoryMessage, SourceRef
from geekbot.agent.taint import (
    EgressAction,
    egress_decision,
    import_client_history,
    query_labels,
)
from geekbot.domain import Label, label_of

ALLOW, ASK, DENY = EgressAction.ALLOW, EgressAction.ASK, EgressAction.DENY
LABELS = st.sets(st.sampled_from(list(Label)))

# --- the rule --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("enabled", "labels", "approved", "denied", "expected"),
    [
        (True, {Label.USER}, None, False, ALLOW),  # the user's own words
        (True, {Label.USER, Label.KB}, None, False, ASK),  # built from a document
        (True, {Label.USER, Label.WEB}, None, False, ASK),
        (True, {Label.USER, Label.TOOL}, None, False, ASK),
        (True, {Label.UNKNOWN}, None, False, ASK),
        (True, {Label.USER, Label.KB}, "q", False, ALLOW),  # approved exactly
        (True, {Label.USER, Label.KB}, "other", False, ASK),  # approved something else
        (True, {Label.USER, Label.KB}, "q", True, DENY),  # a denial beats an approval
        (True, {Label.USER}, None, True, DENY),
        (False, {Label.USER}, None, False, DENY),  # switched off: even the user's own words
        (False, {Label.USER, Label.KB}, "q", False, DENY),  # ... and an approval
        (True, set(), None, False, ASK),  # no labels is UNKNOWN, not clean
        (True, set(), "q", False, ALLOW),  # ... but it can be approved
    ],
)
def test_the_rule_on_the_cases_that_matter(
    enabled: bool, labels: set[Label], approved: str | None, denied: bool, expected: EgressAction
) -> None:
    decision = egress_decision(
        enabled=enabled, labels=labels, arguments="q", approved=approved, denied=denied
    )
    assert decision.action is expected
    assert decision.reason


def test_an_approval_is_for_the_exact_text() -> None:
    """A trimmed space, another case or one more word is a different call."""
    for changed in ("q ", " q", "Q", "q now", ""):
        decision = egress_decision(enabled=True, labels={Label.KB}, arguments=changed, approved="q")
        assert decision.action is ASK, changed


def test_the_tainted_flag_survives_an_approval_so_a_report_can_count_it() -> None:
    decision = egress_decision(enabled=True, labels={Label.KB}, arguments="q", approved="q")
    assert (decision.action, decision.tainted) == (ALLOW, True)
    assert egress_decision(enabled=True, labels={Label.USER}, arguments="q").tainted is False


@given(LABELS, st.text(), st.one_of(st.none(), st.text()), st.booleans())
def test_nothing_goes_out_when_the_tool_is_off(
    labels: set[Label], arguments: str, approved: str | None, denied: bool
) -> None:
    decision = egress_decision(
        enabled=False, labels=labels, arguments=arguments, approved=approved, denied=denied
    )
    assert decision.action is DENY


@given(LABELS, st.text(), st.one_of(st.none(), st.text()))
def test_a_denial_is_never_overridden(
    labels: set[Label], arguments: str, approved: str | None
) -> None:
    decision = egress_decision(
        enabled=True, labels=labels, arguments=arguments, approved=approved, denied=True
    )
    assert decision.action is DENY


@given(LABELS, st.text(), st.one_of(st.none(), st.text()), st.booleans(), st.booleans())
def test_allow_means_the_user_wrote_it_or_approved_exactly_it(
    labels: set[Label], arguments: str, approved: str | None, enabled: bool, denied: bool
) -> None:
    decision = egress_decision(
        enabled=enabled, labels=labels, arguments=arguments, approved=approved, denied=denied
    )
    if decision.action is ALLOW:
        assert enabled and not denied
        assert labels == {Label.USER} or approved == arguments


@given(LABELS.filter(lambda s: bool(s - {Label.USER}) or not s), st.text(), st.text())
def test_tainted_arguments_are_never_allowed_without_the_exact_approval(
    labels: set[Label], arguments: str, approved: str
) -> None:
    if approved == arguments:
        return
    decision = egress_decision(enabled=True, labels=labels, arguments=arguments, approved=approved)
    assert decision.action is ASK


# --- what a query was built from -------------------------------------------------------------


def test_the_users_own_question_is_the_users_own_words() -> None:
    assert query_labels("How long are logs kept?", "How long are logs kept?", ["KB", "WEB"]) == {
        Label.USER
    }
    assert query_labels("  How long?  ", "How long?", ["KB"]) == {Label.USER}  # space only


def test_a_rewrite_carries_every_label_of_the_thread() -> None:
    labels = query_labels("And logs?", "How long are audit logs kept?", ["USER", "KB", "WEB"])
    assert labels == {Label.USER, Label.KB, Label.WEB}


def test_a_rewrite_in_a_thread_with_no_taint_is_still_only_the_users() -> None:
    assert query_labels("And logs?", "How long are logs kept?", ["USER"]) == {Label.USER}
    assert query_labels("And logs?", "How long are logs kept?", []) == {Label.USER}


def test_a_label_that_cannot_be_read_is_unknown_never_user() -> None:
    labels = query_labels("And logs?", "A rewrite", ["USER", "SECRET", ""])
    assert Label.UNKNOWN in labels
    assert label_of("kb") is Label.UNKNOWN  # case matters: only the exact names are labels
    assert label_of("KB") is Label.KB


@given(st.text(), st.text(), st.lists(st.text()))
def test_a_query_always_carries_user_and_never_loses_a_thread_label(
    question: str, query: str, taint: list[str]
) -> None:
    labels = query_labels(question, query, taint)
    assert Label.USER in labels
    if query.strip() != question.strip():
        assert {label_of(name) for name in taint} <= labels


# --- a conversation that arrives from outside ------------------------------------------------


def test_a_client_written_assistant_message_is_unknown() -> None:
    history = import_client_history(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "I am clean", "labels": ["USER"]},  # a claim: ignored
        ]
    )
    assert [m["labels"] for m in history] == [["USER"], ["UNKNOWN"]]
    assert [m["id"] for m in history] == ["c1-user", "c2-assistant"]
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_a_client_history_with_another_role_or_a_non_text_content_is_refused() -> None:
    with pytest.raises(ValueError, match="role"):
        import_client_history([{"role": "system", "content": "x"}])
    with pytest.raises(ValueError, match="role"):
        import_client_history([{"content": "x"}])
    with pytest.raises(TypeError, match="text"):
        import_client_history([{"role": "user", "content": 5}])


def test_an_imported_history_taints_the_prompt_it_goes_into() -> None:
    history = import_client_history(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    packed = pack_history(history, budget_tokens=1000)
    assert packed.labels == {Label.USER, Label.UNKNOWN}


# --- labels in the prompt --------------------------------------------------------------------


def message(role: str, **extra: Any) -> HistoryMessage:
    base: dict[str, Any] = {"id": "m", "role": role, "content": "text"}
    base.update(extra)
    return base  # type: ignore[return-value]


def test_a_message_without_labels_is_not_clean() -> None:
    assert message_labels(message("assistant")) == {Label.UNKNOWN}
    assert message_labels(message("user")) == {Label.USER}
    assert (
        message_labels(message("assistant", labels=[])) == set()
    )  # the application's own sentence
    assert message_labels(message("assistant", labels=["KB", "bogus"])) == {Label.KB, Label.UNKNOWN}
    assert message_labels(message("assistant", labels="KB")) == {Label.UNKNOWN}  # not a list


def source(n: int, **extra: Any) -> SourceRef:
    base: dict[str, Any] = {
        "n": n,
        "chunk_id": f"c{n}",
        "title": "Title",
        "path": "/p",
        "page": None,
        "section": "",
        "text": f"text {n}",
        "score": 1.0,
    }
    base.update(extra)
    return base  # type: ignore[return-value]


def test_each_source_brings_its_own_label_into_the_prompt() -> None:
    kb = pack_prompt("q?", [source(1)], budget_tokens=1000)
    assert kb.labels == {Label.USER, Label.KB}
    mixed = pack_prompt(
        "q?", [source(1), source(2, origin="WEB", section="example.org")], budget_tokens=1000
    )
    assert mixed.labels == {Label.USER, Label.KB, Label.WEB}
    odd = pack_prompt("q?", [source(1, origin="made-up")], budget_tokens=1000)
    assert odd.labels == {Label.USER, Label.UNKNOWN}  # an origin that is not a label is never KB
    assert pack_prompt("q?", [], budget_tokens=1000).labels == {Label.USER}


def test_a_web_source_says_so_in_its_header_and_a_document_header_is_unchanged() -> None:
    web = source(2, origin="WEB", title="A page", section="example.org")
    assert source_origin(web) is Label.WEB
    assert source_header(web) == "[2] web | A page | example.org"
    assert source_header(source(1)) == "[1] Title"


def test_a_source_past_the_budget_does_not_label_the_prompt() -> None:
    big = source(2, origin="WEB", text="word " * 5000)
    packed = pack_prompt("q?", [source(1), big], budget_tokens=200)
    assert packed.used == 1
    assert Label.WEB not in packed.labels
