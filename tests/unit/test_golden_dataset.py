"""Tests for step S2.4: the golden-set format, the validator (this step's ✍️ piece), the split, qrels.

The validator is tested rule by rule, each with one item that breaks only that rule. The split is
tested on the plan's real numbers (100 items -> 40 development) and with hypothesis properties
(a partition, order-free, stable when an item is added). Nothing here needs a model or a corpus.
"""

import random
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from geekbot.evals.datasets import (
    ID_PREFIX,
    ITEM_TYPES,
    TARGET_COUNTS,
    GoldenFormatError,
    GoldenItem,
    Problem,
    Source,
    assign_splits,
    build_qrels,
    corpus_relative,
    describe_set,
    dump_golden,
    list_corpus_files,
    load_golden,
    parse_golden,
    retrieval_items,
    unindexed_sources,
    validate_golden,
    write_golden,
)
from geekbot.evals.metrics import ndcg_at_k, recall_at_k
from geekbot.ingest.ids import doc_id_of

# --- builders: one valid item of each kind ---------------------------------------------------


def single_hop(n: int = 1, *, path: str = "a.md") -> GoldenItem:
    return GoldenItem(
        id=f"S-{n:03d}",
        type="single_hop",
        question=f"Single question {n}?",
        sources=(Source(path, "Setup"),),
        must_include=("a phrase",),
    )


def multi_hop(n: int = 1) -> GoldenItem:
    return GoldenItem(
        id=f"M-{n:03d}",
        type="multi_hop",
        question=f"Multi question {n}?",
        sources=(Source("a.md", "Intro"), Source("b.md", "p. 3")),
        must_include=("first fact", "second fact"),
    )


def multi_turn(n: int = 1) -> GoldenItem:
    return GoldenItem(
        id=f"T-{n:03d}",
        type="multi_turn",
        question=f"And the follow-up {n}?",
        earlier_turns=(f"Earlier question {n}?",),
        sources=(Source("a.md", ""),),
        must_include=("a phrase",),
    )


def unanswerable(n: int = 1) -> GoldenItem:
    return GoldenItem(
        id=f"U-{n:03d}",
        type="unanswerable",
        question=f"Unanswerable question {n}?",
        reason="the manual never mentions it",
    )


def full_set() -> list[GoldenItem]:
    """Exactly the plan's targets: 50 + 15 + 15 + 20 = 100 valid items."""
    return [
        *(single_hop(n) for n in range(1, 51)),
        *(multi_hop(n) for n in range(1, 16)),
        *(multi_turn(n) for n in range(1, 16)),
        *(unanswerable(n) for n in range(1, 21)),
    ]


def messages(problems: tuple[Problem, ...], severity: str = "error") -> list[str]:
    return [p.message for p in problems if p.severity == severity]


# --- the file format -------------------------------------------------------------------------


def test_a_set_survives_dump_and_parse() -> None:
    items = [single_hop(), multi_hop(), multi_turn(), unanswerable()]
    assert parse_golden(dump_golden(items)) == items


def test_blank_lines_are_ignored() -> None:
    text = dump_golden([single_hop(1)]) + "\n   \n" + dump_golden([single_hop(2)])
    assert [i.id for i in parse_golden(text)] == ["S-001", "S-002"]


def test_every_field_is_written_in_a_fixed_order() -> None:
    first = dump_golden([unanswerable()]).splitlines()[0]
    assert first.startswith('{"id": "U-001", "type": "unanswerable", "question": ')
    assert '"earlier_turns": []' in first
    assert '"sources": []' in first


def test_a_file_round_trips_as_utf8_with_unix_line_ends(tmp_path: Path) -> None:
    item = GoldenItem(
        id="S-001",
        type="single_hop",
        question="Wie funktioniert die Garantie für Geräte aus München?",
        sources=(Source("handbuch.md", "Abschnitt 2"),),
        must_include=("zwei Jahre",),
    )
    path = tmp_path / "nested" / "golden.jsonl"  # the folder does not exist yet
    write_golden(path, [item])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM
    assert "München" in raw.decode("utf-8")  # written as text, not as \u escapes
    assert load_golden(path) == [item]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('{"id": "S-001"', "line 1: not valid JSON"),
        ("[1, 2]", "line 1: each line must be a JSON object"),
        ('{"id": "S-001", "type": "single_hop"}', "missing the field 'question'"),
        ('{"id": "S-001", "type": "essay", "question": "q"}', "'type' must be one of"),
        (
            '{"id": "S-001", "type": "single_hop", "question": "q", "must_includes": []}',
            "unknown field(s): must_includes",
        ),
        (
            '{"id": "S-001", "type": "single_hop", "question": "q", "must_include": "x"}',
            "'must_include' must be a list of text",
        ),
        (
            '{"id": "S-001", "type": "single_hop", "question": 5}',
            "'question' must be text, got int",
        ),
        (
            '{"id": "S-001", "type": "single_hop", "question": "q", "sources": ["a.md"]}',
            "each source must be an object",
        ),
        (
            '{"id": "S-001", "type": "single_hop", "question": "q", "sources": [{"file": "a"}]}',
            "unknown source field(s): file",
        ),
        (
            '{"id": "S-001", "type": "single_hop", "question": "q", "sources": [{"locator": "p"}]}',
            "missing the field 'path'",
        ),
    ],
)
def test_a_malformed_file_is_refused_with_its_line_number(text: str, fragment: str) -> None:
    with pytest.raises(GoldenFormatError, match="line 1") as raised:
        parse_golden(text)
    assert fragment in str(raised.value)


def test_the_line_number_counts_blank_lines() -> None:
    good = dump_golden([single_hop()])
    with pytest.raises(GoldenFormatError, match="line 3"):
        parse_golden(good + "\n{broken")


# --- the validator, rule by rule -------------------------------------------------------------


def test_the_plans_full_set_has_no_problems_at_all() -> None:
    report = validate_golden(full_set())
    assert report.problems == ()
    assert report.ok
    assert dict(report.counts) == {
        "single_hop": 50,
        "multi_hop": 15,
        "multi_turn": 15,
        "unanswerable": 20,
    }


def test_an_empty_set_is_an_error() -> None:
    assert messages(validate_golden([]).problems) == ["the golden set is empty"]


@pytest.mark.parametrize("bad_id", ["", "S1", "S-1", "s-001", "X-001", "S_001", "S-001a"])
def test_an_id_must_look_like_a_letter_dash_and_three_digits(bad_id: str) -> None:
    report = validate_golden([GoldenItem(**{**vars(single_hop()), "id": bad_id})])
    assert any("the id must look like S-001" in m for m in messages(report.problems))


def test_an_id_must_start_with_the_letter_of_its_kind() -> None:
    wrong = GoldenItem(**{**vars(unanswerable()), "id": "S-001"})
    assert any("starts with 'U'" in m for m in messages(validate_golden([wrong]).problems))
    assert {ID_PREFIX[k] for k in ITEM_TYPES} == {"S", "M", "T", "U"}


def test_a_repeated_id_is_an_error() -> None:
    report = validate_golden([single_hop(1), single_hop(1, path="b.md")])
    assert "the id is used 2 times" in messages(report.problems)


def test_the_same_question_twice_is_an_error_whatever_the_case_and_spacing() -> None:
    a = single_hop(1)
    b = GoldenItem(**{**vars(single_hop(2)), "question": "  SINGLE   question 1? "})
    assert "the same question as S-001" in messages(validate_golden([a, b]).problems)


def test_a_follow_up_is_compared_together_with_its_earlier_turn() -> None:
    first = GoldenItem(**{**vars(multi_turn(1)), "question": "And by month?"})
    second = GoldenItem(
        **{**vars(multi_turn(2)), "question": "And by month?", "earlier_turns": ("Other topic?",)}
    )
    assert not [m for m in messages(validate_golden([first, second]).problems) if "same" in m]
    same_again = GoldenItem(
        **{**vars(multi_turn(3)), "question": "And by month?", "earlier_turns": first.earlier_turns}
    )
    assert "the same question as T-001" in messages(validate_golden([first, same_again]).problems)


def test_an_empty_question_is_an_error() -> None:
    blank = GoldenItem(**{**vars(single_hop()), "question": "   "})
    assert "the question is empty" in messages(validate_golden([blank]).problems)


@pytest.mark.parametrize("sources", [(), (Source("a.md"), Source("b.md"))])
def test_a_single_hop_question_needs_exactly_one_source(sources: tuple[Source, ...]) -> None:
    item = GoldenItem(**{**vars(single_hop()), "sources": sources})
    assert any("exactly 1 source" in m for m in messages(validate_golden([item]).problems))


def test_a_multi_hop_question_needs_two_sources() -> None:
    item = GoldenItem(**{**vars(multi_hop()), "sources": (Source("a.md", "x"),)})
    assert any("2+ sources" in m for m in messages(validate_golden([item]).problems))


def test_multi_hop_sources_in_one_document_are_only_a_warning() -> None:
    item = GoldenItem(
        **{**vars(multi_hop()), "sources": (Source("a.md", "x"), Source("a.md", "y"))}
    )
    report = validate_golden([item])
    assert not [m for m in messages(report.problems) if "source" in m]
    assert any("one document" in m for m in messages(report.problems, "warning"))


def test_a_multi_turn_item_needs_an_earlier_turn_and_a_source() -> None:
    no_turn = GoldenItem(**{**vars(multi_turn()), "earlier_turns": ()})
    no_source = GoldenItem(**{**vars(multi_turn(2)), "sources": ()})
    blank_turn = GoldenItem(**{**vars(multi_turn(3)), "earlier_turns": (" ",)})
    errors = messages(validate_golden([no_turn, no_source, blank_turn]).problems)
    assert "a multi-turn item needs the earlier turn" in errors
    assert "a multi-turn question needs at least 1 source" in errors
    assert "an earlier turn is empty" in errors


def test_only_a_multi_turn_item_has_earlier_turns() -> None:
    item = GoldenItem(**{**vars(single_hop()), "earlier_turns": ("hi?",)})
    assert "only a multi-turn item has earlier turns" in messages(validate_golden([item]).problems)


def test_an_unanswerable_item_has_no_sources_no_phrases_and_a_reason() -> None:
    bad = GoldenItem(
        **{
            **vars(unanswerable()),
            "sources": (Source("a.md"),),
            "must_include": ("x",),
            "reason": " ",
        }
    )
    errors = messages(validate_golden([bad]).problems)
    assert "an unanswerable question has no sources" in errors
    assert "an unanswerable question has no must-include phrases" in errors
    assert "say why the documents cannot answer it (reason)" in errors


@pytest.mark.parametrize("phrases", [(), ("good", "  ")])
def test_an_answerable_item_needs_real_must_include_phrases(phrases: tuple[str, ...]) -> None:
    item = GoldenItem(**{**vars(single_hop()), "must_include": phrases})
    errors = messages(validate_golden([item]).problems)
    assert any("must-include" in m for m in errors)


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("", "empty path"),
        ("docs\\a.md", "use forward slashes"),
        ("/etc/a.md", "relative to the corpus folder"),
        ("C:/docs/a.md", "relative to the corpus folder"),
        ("../a.md", "would leave the corpus folder"),
        ("docs/../../a.md", "would leave the corpus folder"),
    ],
)
def test_a_source_path_must_be_corpus_relative(path: str, fragment: str) -> None:
    item = GoldenItem(**{**vars(single_hop()), "sources": (Source(path),)})
    assert any(fragment in m for m in messages(validate_golden([item]).problems))


def test_a_source_must_exist_in_the_corpus_when_the_files_are_given() -> None:
    item = single_hop(path="docs/a.md")
    assert not messages(validate_golden([item], corpus_files={"docs/a.md"}).problems)
    errors = messages(validate_golden([item], corpus_files={"docs/b.md"}).problems)
    assert "source 'docs/a.md' is not a file in the corpus folder" in errors
    assert not messages(validate_golden([item], corpus_files=None).problems)  # unchecked


def test_a_source_name_is_compared_exactly_including_case() -> None:
    errors = messages(validate_golden([single_hop(path="A.md")], corpus_files={"a.md"}).problems)
    assert errors  # Windows would find it, Linux would not: be strict everywhere


def test_the_same_place_listed_twice_is_a_warning() -> None:
    item = GoldenItem(
        **{**vars(multi_hop()), "sources": (Source("a.md", "x"), Source("a.md", "x"))}
    )
    assert any("listed twice" in m for m in messages(validate_golden([item]).problems, "warning"))


def test_counts_far_from_the_targets_are_warnings_not_errors() -> None:
    report = validate_golden([single_hop(1), unanswerable(1)])
    assert report.ok
    warnings = messages(report.problems, "warning")
    assert "1 single_hop items; the plan aims for about 50" in warnings
    assert "0 multi_hop items; the plan aims for about 15" in warnings


@pytest.mark.parametrize(("count", "warned"), [(39, True), (40, False), (60, False), (61, True)])
def test_a_count_within_20_percent_of_the_target_is_not_warned_about(
    count: int, warned: bool
) -> None:
    items = [
        *(single_hop(n) for n in range(1, count + 1)),
        *(multi_hop(n) for n in range(1, 16)),
        *(multi_turn(n) for n in range(1, 16)),
        *(unanswerable(n) for n in range(1, 21)),
    ]
    warnings = messages(validate_golden(items).problems, "warning")
    assert any("single_hop" in w for w in warnings) is warned


def test_a_problem_prints_its_severity_and_item() -> None:
    assert str(Problem("error", "S-001", "bad")) == "error: S-001: bad"
    assert str(Problem("warning", None, "few")) == "warning: set: few"


# --- the split -------------------------------------------------------------------------------


def test_a_full_set_splits_into_exactly_40_development_items() -> None:
    items = full_set()
    splits = assign_splits(items)
    assert len(splits) == 100
    assert sum(1 for s in splits.values() if s == "dev") == 40
    for kind, expected in {
        "single_hop": 20,
        "multi_hop": 6,
        "multi_turn": 6,
        "unanswerable": 8,
    }.items():
        dev = [i for i in items if i.type == kind and splits[i.id] == "dev"]
        assert len(dev) == expected
    assert set(TARGET_COUNTS) == set(ITEM_TYPES)


def test_the_split_does_not_depend_on_the_order_of_the_items() -> None:
    items = full_set()
    shuffled = items[:]
    random.Random(3).shuffle(shuffled)
    assert assign_splits(shuffled) == assign_splits(items)


def test_the_split_is_the_same_every_time_and_another_seed_gives_another() -> None:
    items = [single_hop(n) for n in range(1, 51)]
    assert assign_splits(items) == assign_splits(items)
    assert assign_splits(items, seed="other") != assign_splits(items)


def test_a_different_fraction_changes_the_size_of_the_development_part() -> None:
    items = [single_hop(n) for n in range(1, 11)]
    splits = assign_splits(items, dev_fraction=0.5)
    assert sum(1 for s in splits.values() if s == "dev") == 5


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_the_development_fraction_must_be_between_0_and_1(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        assign_splits([single_hop()], dev_fraction=fraction)


def test_a_repeated_id_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        assign_splits([single_hop(1), single_hop(1)])


def test_a_kind_with_one_item_goes_to_the_test_side() -> None:
    assert assign_splits([unanswerable()]) == {"U-001": "test"}  # round(0.4) is 0


@given(st.sets(st.integers(1, 999), max_size=80))
def test_every_item_is_on_exactly_one_side_and_each_kind_is_cut_the_same_way(
    numbers: set[int],
) -> None:
    items = [*(single_hop(n) for n in numbers), *(unanswerable(n) for n in numbers)]
    splits = assign_splits(items)
    assert set(splits) == {i.id for i in items}
    for kind in ("single_hop", "unanswerable"):
        dev = sum(1 for i in items if i.type == kind and splits[i.id] == "dev")
        assert dev == round(len(numbers) * 0.4)


@given(st.sets(st.integers(1, 999), max_size=80), st.integers(1, 999))
def test_adding_an_item_moves_at_most_one_other_item_across_the_line(
    numbers: set[int], extra: int
) -> None:
    assume(extra not in numbers)
    items = [single_hop(n) for n in numbers]
    before = assign_splits(items)
    after = assign_splits([*items, single_hop(extra)])
    assert sum(1 for item_id in before if before[item_id] != after[item_id]) <= 1


def test_retrieval_items_are_the_answerable_single_and_multi_hop_ones_of_one_side() -> None:
    items = full_set()
    splits = assign_splits(items)
    chosen = retrieval_items(items, splits, "dev")
    assert {i.type for i in chosen} == {"single_hop", "multi_hop"}
    assert {splits[i.id] for i in chosen} == {"dev"}
    assert len(chosen) == 20 + 6
    both = retrieval_items(items, splits, "dev") + retrieval_items(items, splits, "test")
    assert len(both) == 50 + 15  # nothing lost, nothing counted twice
    assert [i.id for i in chosen] == [i.id for i in items if i in chosen]  # input order kept


def test_an_item_missing_from_the_split_is_in_neither_side() -> None:
    assert retrieval_items([single_hop()], {}, "dev") == []
    assert retrieval_items([single_hop()], {}, "test") == []


def test_the_summary_table_lists_each_kind_and_the_totals() -> None:
    items = full_set()
    table = describe_set(items, assign_splits(items)).splitlines()
    assert table[0].split() == ["kind", "items", "target", "dev", "test"]
    assert table[1].split() == ["single_hop", "50", "50", "20", "30"]
    assert table[-1].split() == ["total", "100", "100", "40", "60"]


# --- from paths to judgements ----------------------------------------------------------------


def test_qrels_map_each_item_to_the_documents_that_answer_it(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    items = [single_hop(1, path="docs/a.md"), multi_hop(1), unanswerable(1)]
    qrels = build_qrels(items, corpus)
    assert qrels["S-001"] == {doc_id_of(corpus / "docs" / "a.md"): 1}
    assert qrels["M-001"] == {doc_id_of(corpus / "a.md"): 1, doc_id_of(corpus / "b.md"): 1}
    assert qrels["U-001"] == {}


def test_two_sources_in_one_document_are_one_judgement(tmp_path: Path) -> None:
    item = GoldenItem(
        **{**vars(multi_hop()), "sources": (Source("a.md", "x"), Source("a.md", "y"))}
    )
    assert build_qrels([item], tmp_path) == {"M-001": {doc_id_of(tmp_path / "a.md"): 1}}


def test_qrels_work_with_the_metrics(tmp_path: Path) -> None:
    qrels = build_qrels([single_hop(1), unanswerable(1)], tmp_path)
    answer = doc_id_of(tmp_path / "a.md")
    assert recall_at_k(["other", answer], qrels["S-001"], 5) == 1.0
    assert ndcg_at_k([answer], qrels["S-001"], 10) == 1.0
    assert recall_at_k([answer], qrels["U-001"], 5) is None  # undefined, not a zero


def test_a_document_id_depends_on_the_folder_so_the_folder_must_match(tmp_path: Path) -> None:
    item = single_hop()
    assert build_qrels([item], tmp_path / "one") != build_qrels([item], tmp_path / "two")


def test_sources_that_are_not_ready_documents_are_reported(tmp_path: Path) -> None:
    items = [multi_hop(1), single_hop(2)]
    ready = {doc_id_of(tmp_path / "a.md")}
    assert unindexed_sources(items, tmp_path, ready) == [("M-001", "b.md")]
    assert unindexed_sources(items, tmp_path, set()) == [
        ("M-001", "a.md"),
        ("M-001", "b.md"),
        ("S-002", "a.md"),
    ]
    assert unindexed_sources([unanswerable()], tmp_path, set()) == []


def test_corpus_relative_gives_the_golden_files_spelling(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    assert corpus_relative(corpus / "sub" / "a.md", corpus) == "sub/a.md"
    assert corpus_relative(corpus / "sub" / ".." / "b.md", corpus) == "b.md"
    assert corpus_relative(tmp_path / "elsewhere" / "a.md", corpus) is None
    assert corpus_relative(tmp_path / "corpus2" / "a.md", corpus) is None  # a sibling, not inside


def test_corpus_relative_resolves_relative_paths_from_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert corpus_relative("evals/corpus/a.md", Path("evals/corpus")) == "a.md"


def test_the_corpus_files_are_listed_with_forward_slashes(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
    assert list_corpus_files(tmp_path) == {"a.md", "sub/b.txt"}  # folders are not files
