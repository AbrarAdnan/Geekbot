"""Tests for step S2.2: Reciprocal Rank Fusion (this step's ✍️ piece, written by Claude, D-047).

Layout: a hand-computed example first, then the properties of the formula one at a time (scores are
ignored, ``k`` matters, a chunk counted once per list), then the ties, then hypothesis properties
checked against a model written independently of the code.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.domain import Chunk, ScoredChunk
from geekbot.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        path="/d.md",
        title="d.md",
        content_hash="h",
        occurrence=0,
        ordinal=0,
        char_start=0,
        char_end=1,
        page=None,
        section_path=(),
        header="",
        text=chunk_id,
    )


def ranked(*ids: str, scores: list[float] | None = None) -> list[ScoredChunk]:
    """A best-first list of chunks. The scores are made up; fusion must not look at them."""
    values = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return [ScoredChunk(chunk(cid), score) for cid, score in zip(ids, values, strict=True)]


def ids_of(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


# --- by hand ---------------------------------------------------------------------------------


def test_a_hand_computed_example() -> None:
    # list A: a b c    list B: b d a    k = 60
    #   a = 1/61 + 1/63    b = 1/62 + 1/61    c = 1/63    d = 1/62
    fused = reciprocal_rank_fusion([ranked("a", "b", "c"), ranked("b", "d", "a")], k=60)
    assert ids_of(fused) == ["b", "a", "d", "c"]
    scores = {r.chunk.chunk_id: r.score for r in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)


def test_the_default_k_is_the_papers_and_the_plans() -> None:
    assert DEFAULT_RRF_K == 60
    explicit = reciprocal_rank_fusion([ranked("a", "b")], k=60)
    assert reciprocal_rank_fusion([ranked("a", "b")]) == explicit


def test_one_list_keeps_its_order_and_gets_reciprocal_scores() -> None:
    fused = reciprocal_rank_fusion([ranked("x", "y", "z")])
    assert ids_of(fused) == ["x", "y", "z"]
    assert [r.score for r in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_the_chunk_object_is_the_one_from_the_input() -> None:
    first = ranked("a")
    second = ranked("a")
    fused = reciprocal_rank_fusion([first, second])
    assert fused[0].chunk is first[0].chunk  # taken from the first list that has it


# --- the formula's properties ----------------------------------------------------------------


def test_the_original_scores_are_ignored_only_the_ranks_count() -> None:
    plain = reciprocal_rank_fusion([ranked("a", "b", "c"), ranked("c", "a")])
    wild = reciprocal_rank_fusion(
        [
            ranked("a", "b", "c", scores=[0.001, 0.0009, 0.0008]),
            ranked("c", "a", scores=[9000.0, 12.5]),
        ]
    )
    assert plain == [
        ScoredChunk(r.chunk, r.score) for r in wild
    ]  # same chunks, same order, same fused scores


def test_a_chunk_in_both_lists_beats_one_that_tops_a_single_list() -> None:
    # "y" is third in both lists; "x" is first in one and absent from the other.
    fused = reciprocal_rank_fusion([ranked("x", "p", "y"), ranked("q", "r", "y")], k=60)
    assert ids_of(fused)[0] == "y"  # 2/63 beats 1/61


def test_k_decides_how_much_the_top_rank_matters() -> None:
    lists = [ranked("x", "p", "y"), ranked("q", "r", "y")]
    small_k = ids_of(reciprocal_rank_fusion(lists, k=0))
    large_k = ids_of(reciprocal_rank_fusion(lists, k=60))
    assert small_k.index("y") > small_k.index("x")  # k=0: rank 1 is worth 1, rank 3 only 1/3
    assert large_k.index("y") < large_k.index("x")  # k=60: agreement wins


def test_a_chunk_repeated_within_a_list_counts_once_at_its_best_position() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "a", "b")])
    scores = {r.chunk.chunk_id: r.score for r in fused}
    assert scores["a"] == pytest.approx(1 / 61)  # not 1/61 + 1/62
    assert scores["b"] == pytest.approx(1 / 62)  # the repeat did not push b down a rank


def test_three_lists_add_up() -> None:
    fused = reciprocal_rank_fusion([ranked("a"), ranked("a"), ranked("a")])
    assert fused[0].score == pytest.approx(3 / 61)


# --- ties ------------------------------------------------------------------------------------


def test_equal_scores_go_to_the_better_best_rank_then_the_earlier_list_then_the_id() -> None:
    # b: rank 2 in list 1.  c: rank 2 in list 2.  Same score, same best rank: list order decides.
    first_wins = reciprocal_rank_fusion([ranked("a", "b"), ranked("z", "c")])
    assert ids_of(first_wins) == ["a", "z", "b", "c"]  # a and z tie too: list 1 first
    swapped = reciprocal_rank_fusion([ranked("z", "c"), ranked("a", "b")])
    assert ids_of(swapped) == ["z", "a", "c", "b"]
    # the scores themselves do not depend on the order of the lists
    assert {r.chunk.chunk_id: r.score for r in first_wins} == pytest.approx(
        {r.chunk.chunk_id: r.score for r in swapped}
    )


def test_the_earlier_list_wins_before_the_id_does() -> None:
    fused = reciprocal_rank_fusion([ranked("m"), ranked("k")])
    assert ids_of(fused) == ["m", "k"]  # same score, same rank; list 1 first, not alphabetical


def test_a_tie_on_everything_falls_back_to_the_chunk_id() -> None:
    # m and k each add 1/61 + 1/62, both have best rank 1 and both first appear in list 1.
    fused = reciprocal_rank_fusion([ranked("m", "k"), ranked("k", "m")])
    assert ids_of(fused) == ["k", "m"]


def test_a_tie_on_paper_is_a_tie_in_the_code_whatever_the_order_the_terms_are_added_in() -> None:
    # Three-term sums that are equal on paper but are added in a different order for each chunk.
    lists = [ranked("a", "b", "c"), ranked("c", "a", "b"), ranked("b", "c", "a")]
    assert ids_of(reciprocal_rank_fusion(lists)) == ["a", "b", "c"]  # all equal: first list's order


# --- arguments -------------------------------------------------------------------------------


def test_limit_keeps_the_top_results() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "b", "c", "d")], limit=2)
    assert ids_of(fused) == ["a", "b"]
    assert len(reciprocal_rank_fusion([ranked("a")], limit=10)) == 1


def test_nothing_in_nothing_out() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize("bad", [-1, -60])
def test_a_negative_k_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="k must not be negative"):
        reciprocal_rank_fusion([ranked("a")], k=bad)


def test_a_limit_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        reciprocal_rank_fusion([ranked("a")], limit=0)


# --- properties, against a model written independently ---------------------------------------

_IDS = st.sampled_from("abcdefg")
_LISTS = st.lists(st.lists(_IDS, max_size=7), max_size=4)


def model(lists: list[list[str]], k: int) -> dict[str, float]:
    """The textbook formula, written the plain way: rank = position among distinct chunks."""
    totals: dict[str, float] = {}
    for items in lists:
        distinct = list(dict.fromkeys(items))
        for position, chunk_id in enumerate(distinct, start=1):
            totals[chunk_id] = totals.get(chunk_id, 0.0) + 1.0 / (k + position)
    return totals


@given(lists=_LISTS, k=st.integers(min_value=0, max_value=100))
def test_the_scores_equal_the_textbook_formula(lists: list[list[str]], k: int) -> None:
    fused = reciprocal_rank_fusion([ranked(*items) for items in lists if items] or [], k=k)
    got = {r.chunk.chunk_id: r.score for r in fused}
    assert got == pytest.approx(model(lists, k))


@given(lists=_LISTS)
def test_every_chunk_appears_once_and_the_scores_never_increase(lists: list[list[str]]) -> None:
    fused = reciprocal_rank_fusion([ranked(*items) for items in lists if items])
    result_ids = ids_of(fused)
    assert len(result_ids) == len(set(result_ids))
    assert set(result_ids) == {cid for items in lists for cid in items}
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


@given(lists=_LISTS)
def test_the_order_of_the_lists_changes_only_ties(lists: list[list[str]]) -> None:
    forward = reciprocal_rank_fusion([ranked(*items) for items in lists if items])
    backward = reciprocal_rank_fusion([ranked(*items) for items in reversed(lists) if items])
    assert {r.chunk.chunk_id: r.score for r in forward} == pytest.approx(
        {r.chunk.chunk_id: r.score for r in backward}
    )
    assert reciprocal_rank_fusion([ranked(*items) for items in lists if items]) == forward  # stable
