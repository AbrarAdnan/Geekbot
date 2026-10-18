"""Tests for the cached rewrites of golden follow-ups (step S2.6, D-076)."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from geekbot.agent.condense import CondenseOutput
from geekbot.evals.datasets import GoldenItem, Source
from geekbot.evals.rewrites import (
    RewriteCache,
    cached_rewrites,
    history_of,
    rewrite_followups,
    rewrite_key,
)

ITEM = GoldenItem(
    "T-001",
    "multi_turn",
    "and the logs that only record who used it?",
    earlier_turns=("How long does Lantern keep chat history?",),
    sources=(Source("policy.docx"),),
    must_include=("365 days",),
)
SINGLE = GoldenItem("S-001", "single_hop", "a question", sources=(Source("a.txt"),))


class FakeCondenser:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seen: list[Any] = []

    def invoke(self, messages: list[Any], /) -> CondenseOutput:
        self.seen.append(messages)
        return CondenseOutput(standalone_question=self.replies.pop(0))


def test_the_history_is_the_earlier_questions_as_user_messages() -> None:
    history = history_of(ITEM)
    assert [(m["id"], m["role"], m["labels"]) for m in history] == [("t1-user", "user", ["USER"])]
    assert history[0]["content"] == "How long does Lantern keep chat history?"


def test_the_key_changes_with_anything_that_could_change_the_rewrite() -> None:
    base = rewrite_key(ITEM, model="gemma4:12b", digest="abc")
    assert base == rewrite_key(ITEM, model="gemma4:12b", digest="abc")  # stable
    assert base != rewrite_key(ITEM, model="gemma3:4b", digest="abc")  # another model
    assert base != rewrite_key(ITEM, model="gemma4:12b", digest="def")  # the model was re-pulled
    for changed in (
        dataclasses.replace(ITEM, question="and the other logs?"),
        dataclasses.replace(ITEM, earlier_turns=("A different first question?",)),
        dataclasses.replace(ITEM, id="T-002"),
    ):
        assert rewrite_key(changed, model="gemma4:12b", digest="abc") != base


def test_the_key_changes_when_the_prompt_or_the_schema_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = rewrite_key(ITEM, model="m", digest="d")
    monkeypatch.setattr("geekbot.evals.rewrites.CONDENSE_PROMPT", "a different prompt")
    assert rewrite_key(ITEM, model="m", digest="d") != base


def test_the_cache_round_trips_and_names_the_model_beside_each_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "rewrites.json"
    cache = RewriteCache.load(path)
    assert cache.entries == {}
    cache.put("k1", ITEM, model="gemma4:12b", digest="abc", rewrite="Retention of the access logs?")
    again = RewriteCache.load(path)
    assert again.get("k1") == "Retention of the access logs?"
    assert again.get("missing") is None
    entry = again.entries["k1"]
    assert (entry["item"], entry["model"], entry["digest"]) == ("T-001", "gemma4:12b", "abc")
    assert entry["question"] == ITEM.question
    assert not list(path.parent.glob("*.part"))  # written whole, no half file left behind


def test_a_damaged_cache_is_refused_not_silently_emptied(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a rewrite cache"):
        RewriteCache.load(path)
    path.write_text('{"entries": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a rewrite cache"):
        RewriteCache.load(path)


def test_the_model_is_called_only_for_a_miss_and_only_for_follow_ups(tmp_path: Path) -> None:
    cache = RewriteCache.load(tmp_path / "r.json")
    condenser = FakeCondenser("How long are access logs kept?")
    shown: list[tuple[str, str, bool]] = []
    found = rewrite_followups(
        [SINGLE, ITEM],
        condenser,
        cache,
        model="m",
        digest="d",
        budget_tokens=1000,
        on_item=lambda i, r, c: shown.append((i, r, c)),
    )
    assert found == {"T-001": "How long are access logs kept?"}
    assert len(condenser.seen) == 1
    assert shown == [("T-001", "How long are access logs kept?", False)]
    # the model saw the instructions, the earlier question and the follow-up
    _system, human = condenser.seen[0]
    assert "Last message: and the logs that only record who used it?" in human.content
    assert "How long does Lantern keep chat history?" in human.content

    again = rewrite_followups(
        [SINGLE, ITEM], FakeCondenser(), cache, model="m", digest="d", budget_tokens=1000,
        on_item=lambda i, r, c: shown.append((i, r, c)),
    )  # fmt: skip
    assert again == found  # an empty FakeCondenser would raise if it were called
    assert shown[-1][2] is True
    assert cached_rewrites([SINGLE, ITEM], cache, model="m", digest="d") == found
    assert cached_rewrites([ITEM], cache, model="m", digest="other") == {}


def test_an_unusable_rewrite_falls_back_to_the_follow_up_and_is_cached(tmp_path: Path) -> None:
    cache = RewriteCache.load(tmp_path / "r.json")
    found = rewrite_followups(
        [ITEM], FakeCondenser("   "), cache, model="m", digest="d", budget_tokens=1000
    )
    assert found == {"T-001": ITEM.question}  # as in the product: the user's own words
    assert cache.get(rewrite_key(ITEM, model="m", digest="d")) == ITEM.question


def test_a_model_that_does_not_return_the_expected_shape_is_an_error(tmp_path: Path) -> None:
    class Wrong:
        def invoke(self, messages: list[Any], /) -> object:
            return {"standalone_question": "x"}

    with pytest.raises(TypeError, match="CondenseOutput"):
        rewrite_followups(
            [ITEM], Wrong(), RewriteCache.load(tmp_path / "r.json"), model="m", digest="d",
            budget_tokens=1000,
        )  # fmt: skip
