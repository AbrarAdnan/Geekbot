"""Tests for step S3.6: hybrid search as stages (``retrieval.staged.StagedSearch``).

No model, no store: the two searches are small fakes that return canned chunks, the "store" is a
dictionary and the cross-encoder scores by a table. What is tested is that the staged form gives the
same answer as the sequential classes (``HybridRetriever`` and ``Reranker``), that what moves between
stages is plain IDs and scores, and the edges: a chunk that stopped being ready, a scorer that answers
with the wrong number of scores, a bad ``k``.
"""

import json
from collections.abc import Sequence

import pytest

from geekbot.domain import Chunk, ScoredChunk
from geekbot.retrieval import RetrievalResult
from geekbot.retrieval.factory import citation_scorer_of
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.rerank import Reranker, RerankError
from geekbot.retrieval.staged import StagedSearch


def chunk(n: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{n}",
        doc_id="doc",
        path="/docs/handbook.md",
        title="Handbook",
        content_hash="h" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=10,
        page=None,
        section_path=(),
        header="",
        text=f"text {n}",
    )


def scored(*numbers: int) -> list[ScoredChunk]:
    """Chunks c<n> in the given order, with made-up scores that fusion must not look at."""
    return [ScoredChunk(chunk(n), 1.0 / (i + 1)) for i, n in enumerate(numbers)]


class FakeSide:
    """A search that returns canned chunks (at most ``k``) and remembers what it was asked."""

    def __init__(self, *numbers: int, total_ms: float | None = None) -> None:
        self.chunks = scored(*numbers)
        self.total_ms = total_ms
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        timings = {} if self.total_ms is None else {"total": self.total_ms}
        return RetrievalResult(chunks=tuple(self.chunks[:k]), timings_ms=timings)


class FakeStore:
    """``Store.get_chunks`` over a dictionary: the IDs it knows, in the order asked."""

    def __init__(self, numbers: Sequence[int], *, unready: Sequence[int] = ()) -> None:
        self.known = {f"c{n}": chunk(n) for n in numbers if n not in unready}
        self.asked: list[list[str]] = []

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        self.asked.append(list(chunk_ids))
        return [self.known[cid] for cid in chunk_ids if cid in self.known]


class TableScorer:
    """Scores a passage by a table keyed on its text; remembers what it was asked to score."""

    def __init__(self, table: dict[str, float], *, wrong_count: bool = False) -> None:
        self.table = table
        self.wrong_count = wrong_count
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        scores = [self.table.get(text, 0.0) for text in passages]
        return scores[:-1] if self.wrong_count else scores


ALL = range(1, 9)
DENSE = (1, 2, 3, 4, 5)
LEXICAL = (3, 6, 1, 7)
# The cross-encoder disagrees with the fusion (which puts c1, c3, c2, c6, c4, c7 in that order): it
# likes c2 best, then c6, then c1, so the order must change.
TABLE = {"text 2": 9.0, "text 6": 5.0, "text 1": 1.0}


def staged(
    scorer: TableScorer | None,
    *,
    store: FakeStore | None = None,
    candidates: int = 4,
    depth: int = 3,
) -> StagedSearch:
    return StagedSearch(
        FakeSide(*DENSE),
        FakeSide(*LEXICAL),
        (store or FakeStore(ALL)).get_chunks,
        scorer,
        candidates=candidates,
        depth=depth,
    )


def sequential(scorer: TableScorer | None, *, candidates: int = 4, depth: int = 3) -> object:
    hybrid = HybridRetriever(FakeSide(*DENSE), FakeSide(*LEXICAL), candidates=candidates)
    return hybrid if scorer is None else Reranker(hybrid, scorer, depth=depth)


# --- the same answer as the sequential classes ------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9])
def test_with_a_scorer_it_returns_what_reranker_over_hybrid_returns(k: int) -> None:
    got = staged(TableScorer(TABLE)).search("q", k)
    want = sequential(TableScorer(TABLE))
    assert isinstance(want, Reranker)
    expected = want.search("q", k)
    assert [(s.chunk.chunk_id, s.score) for s in got.chunks] == [
        (s.chunk.chunk_id, s.score) for s in expected.chunks
    ]
    assert set(got.timings_ms) == set(expected.timings_ms)  # dense, lexical, fusion, rerank, total


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9])
def test_without_a_scorer_it_returns_what_the_hybrid_retriever_returns(k: int) -> None:
    got = staged(None).search("q", k)
    want = sequential(None)
    assert isinstance(want, HybridRetriever)
    expected = want.search("q", k)
    assert [(s.chunk.chunk_id, s.score) for s in got.chunks] == [
        (s.chunk.chunk_id, s.score) for s in expected.chunks
    ]
    assert set(got.timings_ms) == set(expected.timings_ms)  # no rerank stage to report


def test_the_cross_encoder_really_changes_the_order_in_this_setup() -> None:
    """Guards the two tests above: if the table changed nothing they would prove nothing."""
    plain = [s.chunk.chunk_id for s in staged(None).search("q", 2).chunks]
    reranked = [s.chunk.chunk_id for s in staged(TableScorer(TABLE)).search("q", 2).chunks]
    assert plain != reranked


# --- how much each stage is asked for -----------------------------------------------------------


def test_the_depths_follow_the_sequential_classes() -> None:
    with_scorer = StagedSearch(FakeSide(), FakeSide(), FakeStore(()).get_chunks, TableScorer({}))
    # defaults: 50 candidates a side, rerank depth 10
    assert (with_scorer.fused_depth(5), with_scorer.side_depth(5)) == (10, 50)
    assert (with_scorer.fused_depth(60), with_scorer.side_depth(60)) == (60, 60)
    without = StagedSearch(FakeSide(), FakeSide(), FakeStore(()).get_chunks, None)
    assert (without.fused_depth(5), without.side_depth(5)) == (5, 50)


def test_each_side_is_asked_for_the_side_depth() -> None:
    dense, lexical = FakeSide(*DENSE), FakeSide(*LEXICAL)
    search = StagedSearch(
        dense, lexical, FakeStore(ALL).get_chunks, TableScorer({}), candidates=4, depth=3
    )
    search.search("what?", 2)
    assert dense.calls == lexical.calls == [("what?", 4)]  # max(max(2, 3), 4)


# --- what moves between stages ----------------------------------------------------------------


def test_a_stage_returns_ids_and_scores_only_and_they_are_plain_data() -> None:
    search = staged(TableScorer(TABLE))
    dense = search.dense_stage("q", 2)
    lexical = search.lexical_stage("q", 2)
    fused = search.fuse_stage(dense.hits, lexical.hits, 2)
    for stage in (dense.hits, lexical.hits, fused.hits):
        assert all(set(hit) == {"chunk_id", "score"} for hit in stage)
        assert json.loads(json.dumps(stage)) == stage
    assert [hit["chunk_id"] for hit in dense.hits] == ["c1", "c2", "c3", "c4"]  # side order kept
    assert len(fused.hits) == 3  # fused_depth(2) = max(2, depth 3)


def test_fusion_needs_no_text_and_only_the_last_stage_reads_the_store() -> None:
    store = FakeStore(ALL)
    search = staged(TableScorer(TABLE), store=store)
    dense = search.dense_stage("q", 2)
    lexical = search.lexical_stage("q", 2)
    fused = search.fuse_stage(dense.hits, lexical.hits, 2)
    assert store.asked == []  # the searches and the fusion never asked for a chunk
    search.rerank_stage("q", fused.hits, 2)
    assert len(store.asked) == 1
    assert len(store.asked[0]) == 3  # only the fused candidates were read, not every one found


# --- the last stage ---------------------------------------------------------------------------


def test_the_scorer_reads_the_text_of_the_fused_candidates_in_fused_order() -> None:
    scorer = TableScorer(TABLE)
    search = staged(scorer)
    fused = search.fuse_stage(search.dense_stage("q", 2).hits, search.lexical_stage("q", 2).hits, 2)
    search.rerank_stage("the question", fused.hits, 2)
    ((query, passages),) = scorer.calls
    assert query == "the question"
    assert passages == [f"text {hit['chunk_id'][1:]}" for hit in fused.hits]


def test_a_candidate_the_store_no_longer_returns_is_skipped() -> None:
    scorer = TableScorer(TABLE)
    search = staged(scorer, store=FakeStore(ALL, unready=(3,)))
    result = search.search("q", 5)
    ids = [s.chunk.chunk_id for s in result.chunks]
    assert "c3" not in ids  # fusion put it first; its document stopped being ready
    assert all("text 3" not in passages for _, passages in scorer.calls)


def test_nothing_found_means_the_scorer_is_not_called() -> None:
    scorer = TableScorer(TABLE)
    search = StagedSearch(FakeSide(), FakeSide(), FakeStore(ALL).get_chunks, scorer)
    result = search.search("q", 3)
    assert result.chunks == ()
    assert scorer.calls == []


def test_equal_scores_keep_the_fused_order() -> None:
    scorer = TableScorer({})  # everything scores 0.0
    ordered = staged(scorer).search("q", 3)
    fused_order = staged(None).search("q", 3)
    assert [s.chunk.chunk_id for s in ordered.chunks] == [
        s.chunk.chunk_id for s in fused_order.chunks
    ]


def test_a_scorer_that_returns_the_wrong_number_of_scores_is_an_error() -> None:
    with pytest.raises(RerankError, match="scores, the scorer returned"):
        staged(TableScorer(TABLE, wrong_count=True)).search("q", 2)


# --- timings ---------------------------------------------------------------------------------


def test_total_is_the_slower_search_when_they_overlapped_and_the_sum_when_they_did_not() -> None:
    search = staged(TableScorer(TABLE))
    parallel = search.timings(30.0, 5.0, 1.0, 20.0, parallel=True)
    in_order = search.timings(30.0, 5.0, 1.0, 20.0, parallel=False)
    assert parallel == {"dense": 30.0, "lexical": 5.0, "fusion": 1.0, "rerank": 20.0, "total": 51.0}
    assert in_order["total"] == 56.0
    assert "rerank" not in staged(None).timings(1.0, 1.0, 1.0, 1.0, parallel=True)


def test_a_side_reports_its_own_total_when_it_has_one() -> None:
    search = StagedSearch(
        FakeSide(*DENSE, total_ms=12.5), FakeSide(*LEXICAL), FakeStore(ALL).get_chunks, None
    )
    assert search.dense_stage("q", 1).ms == 12.5
    assert search.lexical_stage("q", 1).ms >= 0.0  # no total reported: the stage's own clock


# --- edges -----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1])
def test_every_stage_refuses_k_below_one(bad: int) -> None:
    search = staged(None)
    with pytest.raises(ValueError, match="k must be at least 1"):
        search.dense_stage("q", bad)
    with pytest.raises(ValueError, match="k must be at least 1"):
        search.lexical_stage("q", bad)
    with pytest.raises(ValueError, match="k must be at least 1"):
        search.fuse_stage([], [], bad)
    with pytest.raises(ValueError, match="k must be at least 1"):
        search.rerank_stage("q", [], bad)


def test_candidates_and_depth_must_be_positive() -> None:
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        StagedSearch(FakeSide(), FakeSide(), FakeStore(()).get_chunks, None, candidates=0)
    with pytest.raises(ValueError, match="depth must be at least 1"):
        StagedSearch(FakeSide(), FakeSide(), FakeStore(()).get_chunks, None, depth=0)


def test_the_citation_check_finds_the_scorer_of_a_staged_search() -> None:
    scorer = TableScorer(TABLE)
    assert citation_scorer_of(staged(scorer)) is scorer
    assert citation_scorer_of(staged(None)) is None
