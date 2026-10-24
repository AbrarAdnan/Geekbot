"""Tests for step S5.2: human labels, the blind order and Cohen's kappa (``geekbot.labelling``)."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.labelling import (
    KAPPA_THRESHOLD,
    LABEL_COUNT,
    MIN_LABELS_FOR_KAPPA,
    LabelStore,
    agreement_counts,
    blind_order,
    cohens_kappa,
    kappa_verdict,
    pick_for_labelling,
)

# --- the blind order -------------------------------------------------------------------------

IDS = [f"{kind}-{n:03d}" for kind in "SMTU" for n in range(1, 26)]


def test_the_order_is_the_same_whatever_order_the_ids_come_in() -> None:
    assert blind_order(IDS) == blind_order(list(reversed(IDS)))
    assert sorted(blind_order(IDS)) == sorted(IDS)  # nothing lost, nothing added


def test_the_order_does_not_follow_the_ids_or_their_kind() -> None:
    first_twenty = blind_order(IDS)[:20]
    assert first_twenty != sorted(first_twenty)
    assert len({item_id[0] for item_id in first_twenty}) > 1  # not all of one kind


def test_another_seed_is_another_order() -> None:
    assert blind_order(IDS, seed="a") != blind_order(IDS, seed="b")


def test_a_repeated_id_is_listed_once() -> None:
    assert blind_order(["S-001", "S-002", "S-001"]).count("S-001") == 1


def test_the_picked_answers_are_the_first_of_the_blind_order() -> None:
    picked = pick_for_labelling(IDS, count=10)
    assert picked == blind_order(IDS)[:10]
    assert len(pick_for_labelling(IDS)) == min(LABEL_COUNT, len(IDS))
    assert pick_for_labelling(IDS, count=0) == []
    assert pick_for_labelling(IDS, count=-3) == []


@given(st.lists(st.text(min_size=1, max_size=6), max_size=40), st.integers(0, 50))
def test_a_pick_is_a_stable_prefix_of_the_unique_ids(ids: list[str], count: int) -> None:
    picked = pick_for_labelling(ids, count=count)
    assert picked == pick_for_labelling(list(reversed(ids)), count=count)
    assert len(picked) == min(count, len(set(ids)))
    assert set(picked) <= set(ids)


# --- the label file --------------------------------------------------------------------------


def test_a_label_is_appended_and_read_back(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels" / "labels.jsonl", "run-1")
    label = store.add("S-001", True, " looks right ")
    assert label.note == "looks right" and label.at  # trimmed, with a time
    store.add("S-002", False)
    found = store.labels()
    assert {k: v.supported for k, v in found.items()} == {"S-001": True, "S-002": False}


def test_the_latest_label_of_an_item_wins_and_nothing_is_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path, "run-1")
    store.add("S-001", True)
    before = path.read_text(encoding="utf-8")
    store.add("S-001", False)  # a corrected click
    assert path.read_text(encoding="utf-8").startswith(before)  # the old line is still there
    assert store.labels()["S-001"].supported is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_labels_of_another_run_are_not_mixed_in(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    LabelStore(path, "run-1").add("S-001", True)
    LabelStore(path, "run-2").add("S-001", False)
    assert LabelStore(path, "run-1").labels()["S-001"].supported is True
    assert LabelStore(path, "run-2").labels()["S-001"].supported is False


def test_a_damaged_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path, "run-1")
    store.add("S-001", True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run": "run-1", "id": "S-002", "supp\n')  # a half-written line
        handle.write('{"run": "run-1", "id": "S-003", "supported": "yes"}\n')  # not a boolean
        handle.write("not json at all\n")
        handle.write("[1, 2]\n")
    store.add("S-004", False)
    assert set(store.labels()) == {"S-001", "S-004"}


def test_a_missing_file_has_no_labels_and_remaining_follows_the_order(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "none.jsonl", "run-1")
    assert store.labels() == {}
    order = ["S-003", "S-001", "S-002"]
    assert store.remaining(order) == order
    store.add("S-001", True)
    assert store.remaining(order) == ["S-003", "S-002"]


def test_a_note_is_cut_to_a_sane_length(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl", "run-1")
    assert len(store.add("S-001", True, "x" * 5000).note) == 500


def test_non_ascii_notes_survive(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl", "run-1")
    store.add("S-001", True, "café – ok")
    assert store.labels()["S-001"].note == "café – ok"


# --- Cohen's kappa ---------------------------------------------------------------------------


def test_perfect_agreement_is_one_and_chance_level_agreement_is_zero() -> None:
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(
        1.0
    )
    # Same yes-rate (half) and half of the answers agree: exactly what chance gives.
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(
        0.0
    )


def test_total_disagreement_is_minus_one_for_balanced_raters() -> None:
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) == pytest.approx(
        -1.0
    )


def test_a_worked_example() -> None:
    """20 items: both yes 15, both no 3, human yes judge no 1, judge yes human no 1."""
    human = [True] * 15 + [False] * 3 + [True] + [False]
    judge = [True] * 15 + [False] * 3 + [False] + [True]
    # p_o = 18/20 = 0.9; human yes 16/20 = 0.8; judge yes 16/20 = 0.8; p_e = 0.64 + 0.04 = 0.68
    assert cohens_kappa(human, judge) == pytest.approx((0.9 - 0.68) / (1 - 0.68))


def test_a_judge_that_always_says_yes_gets_no_credit_for_a_lopsided_set() -> None:
    """The reason κ exists: 9 of 10 answers are supported, and 'always yes' agrees on 90% of them."""
    human = [True] * 9 + [False]
    judge = [True] * 10
    assert cohens_kappa(human, judge) == pytest.approx(0.0)


def test_kappa_is_undefined_with_nothing_to_agree_beyond() -> None:
    assert cohens_kappa([], []) is None
    assert cohens_kappa([True, True], [True, True]) is None  # both always yes
    assert cohens_kappa([False, False], [False, False]) is None  # both always no


def test_the_raters_must_have_rated_the_same_items() -> None:
    with pytest.raises(ValueError, match="must match"):
        cohens_kappa([True], [True, False])
    with pytest.raises(ValueError, match="must match"):
        agreement_counts([True], [True, False])


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=40))
def test_kappa_is_symmetric_and_never_above_one(pairs: list[tuple[bool, bool]]) -> None:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    forward, backward = cohens_kappa(human, judge), cohens_kappa(judge, human)
    assert (forward is None) == (backward is None)
    if forward is not None and backward is not None:
        assert forward == pytest.approx(backward)
        assert forward <= 1.0 + 1e-12 and forward >= -1.0 - 1e-12


@given(st.lists(st.booleans(), min_size=1, max_size=40))
def test_a_rater_agrees_with_itself_perfectly_unless_it_never_varies(rater: list[bool]) -> None:
    value = cohens_kappa(rater, rater)
    if len(set(rater)) == 1:
        assert value is None
    else:
        assert value == pytest.approx(1.0)


def test_the_agreement_table_counts_the_four_cases() -> None:
    counts = agreement_counts([True, True, False, False, True], [True, False, False, True, True])
    assert counts == {
        "both_supported": 2,
        "both_unsupported": 1,
        "human_supported_judge_not": 1,
        "judge_supported_human_not": 1,
    }


# --- the rule of the plan --------------------------------------------------------------------


def test_the_threshold_is_the_plans() -> None:
    assert KAPPA_THRESHOLD == 0.6 and LABEL_COUNT == 60


def test_too_few_labels_get_no_verdict_and_say_how_many_are_missing() -> None:
    text = kappa_verdict(0.9, MIN_LABELS_FOR_KAPPA - 1)
    assert "no verdict yet" in text and f"{MIN_LABELS_FOR_KAPPA}" in text
    assert "EXCLUDED" not in text


def test_an_undefined_kappa_gets_no_verdict() -> None:
    assert "undefined" in kappa_verdict(None, 60)


def test_a_kappa_at_the_threshold_keeps_the_judge_as_supporting_evidence() -> None:
    text = kappa_verdict(0.6, 60)
    assert "supporting evidence" in text and "EXCLUDED" not in text


def test_a_kappa_below_the_threshold_excludes_the_judge_from_headline_claims() -> None:
    text = kappa_verdict(0.59, 60)
    assert "EXCLUDED from headline claims" in text and "0.59" in text
    assert "EXCLUDED" in kappa_verdict(-0.2, 60)
