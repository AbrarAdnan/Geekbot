"""Tests for running retrievers over question sets and scoring them (step S2.6)."""

from collections.abc import Callable
from pathlib import Path

import pytest
from evalkit import DIM, HashEmbeddings, OverlapScorer, fake_backends

from geekbot.domain import Chunk, ScoredChunk
from geekbot.evals.ablation import V2_HEADER
from geekbot.evals.cpugate import CpuSnapshot, GateReading
from geekbot.evals.indexing import BuiltIndex, EmbedderChoice, build_index
from geekbot.evals.runner import (
    METRICS,
    MemoScorer,
    build_retriever_for,
    record_of,
    run_queries,
    score_rankings,
)
from geekbot.index.embedder import EmbedderSpec
from geekbot.ingest.loaders.model import LoadedDocument, Segment
from geekbot.retrieval import RetrievalResult
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.rerank import Reranker

SPEC = EmbedderSpec(
    model="fake",
    digest="d",
    document_prefix="",
    query_prefix="",
    dimension=DIM,
    with_header=True,
)


def chunk(doc_id: str, n: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}{n}".ljust(24, "0"),
        doc_id=doc_id,
        path=f"/{doc_id}",
        title=doc_id,
        content_hash="0" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=1,
        page=None,
        section_path=(),
        header="",
        text="text",
    )


class Scripted:
    """A retriever that returns a fixed ranking per question and counts its calls."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append(query)
        chunks = tuple(
            ScoredChunk(chunk(d, i), 1.0 / (i + 1)) for i, d in enumerate(self.answers[query])
        )
        return RetrievalResult(chunks[:k], {"vector_search": 2.0, "total": 3.0})


@pytest.fixture
def built(tmp_path: Path) -> BuiltIndex:
    docs = [
        LoadedDocument(
            Path("/r") / f"{name}.txt",
            "text",
            name.title(),
            (Segment(text=f"{name} text about the {name} and nothing else"),),
        )
        for name in ("zebra", "quartz", "bread")
    ]
    return build_index(
        docs,
        V2_HEADER,
        EmbedderChoice("fake", "fake"),
        tmp_path / "i",
        tmp_path / "c.db",
        embeddings=HashEmbeddings(),
        spec=SPEC,
    )


def test_the_memo_scorer_asks_the_real_one_only_about_what_it_has_not_seen() -> None:
    inner = OverlapScorer()
    memo = MemoScorer(inner)
    first = memo.score("zebra stripes", ["a zebra", "a rock"])
    assert inner.calls == 1
    again = memo.score("zebra stripes", ["a rock", "a zebra"])
    assert inner.calls == 1  # both were remembered
    assert again == [first[1], first[0]]
    memo.score("zebra stripes", ["a rock", "a new passage"])
    assert inner.calls == 2  # only the new passage went to the scorer
    memo.score("another question", ["a rock"])
    assert inner.calls == 3  # the same passage under another question is a different score


def test_each_strategy_builds_the_retriever_it_names(built: BuiltIndex) -> None:
    scorer = OverlapScorer()
    with built:
        dense = build_retriever_for("dense", built, None, depth=5, candidates=10)
        hybrid = build_retriever_for("hybrid", built, None, depth=5, candidates=10)
        rerank = build_retriever_for("hybrid_rerank", built, scorer, depth=5, candidates=10)
        dense_rerank = build_retriever_for("dense_rerank", built, scorer, depth=5, candidates=10)
        legacy = build_retriever_for("legacy_multiquery", built, None, depth=5, candidates=10)
        assert isinstance(dense, DenseRetriever)
        assert isinstance(hybrid, HybridRetriever)
        assert isinstance(rerank, Reranker)
        assert isinstance(dense_rerank, Reranker)
        assert type(legacy).__name__ == "LegacyMultiQueryRetriever"
        for strategy in ("hybrid_rerank", "dense_rerank"):
            with pytest.raises(ValueError, match="needs a reranker"):
                build_retriever_for(strategy, built, None, depth=5, candidates=10)  # type: ignore[arg-type]


def test_every_strategy_finds_the_right_document_on_a_tiny_index(built: BuiltIndex) -> None:
    with built:
        for strategy in ("dense", "hybrid", "hybrid_rerank", "dense_rerank"):
            retriever = build_retriever_for(
                strategy,  # type: ignore[arg-type]
                built,
                OverlapScorer(),
                depth=3,
                candidates=5,
            )
            top = retriever.search("the zebra", 3).chunks[0]
            assert top.chunk.title == "Zebra", strategy


def gate_returning(idle: bool, calls: list[float]) -> Callable[..., GateReading]:
    def gate(*, idle_percent: float) -> GateReading:
        calls.append(idle_percent)
        return GateReading(CpuSnapshot(1.0, ()), None, idle, 0.0)

    return gate


def test_run_queries_times_each_question_and_leaves_warm_up_out() -> None:
    retriever = Scripted({"q1": ["a", "b"], "q2": ["c"], "q3": []})
    percents: list[float] = []
    ticks = iter(range(0, 1000, 5))  # every clock() call advances 5 seconds -> 5000 ms a search
    output = run_queries(
        retriever,
        {"q1": "q1", "q2": "q2", "q3": "q3"},
        k=5,
        warmup=2,
        idle_percent=12.0,
        gate=gate_returning(True, percents),
        snapshot=lambda: CpuSnapshot(7.0, (("x", 1.0),)),
        clock=lambda: float(next(ticks)),
    )
    assert retriever.calls == ["q1", "q2", "q1", "q2", "q3"]  # 2 warm-up searches, then 3 timed
    assert percents == [12.0]
    assert output.rankings == {"q1": ["a", "b"], "q2": ["c"], "q3": []}
    assert output.top_scores == {"q1": 1.0, "q2": 1.0, "q3": None}
    assert output.latencies_ms == [5000.0, 5000.0, 5000.0]
    assert output.stage_ms == {"vector_search": 2.0}  # the mean of the stage, "total" left out
    assert output.gate.idle is True
    assert output.gate.after == CpuSnapshot(7.0, (("x", 1.0),))


def test_scores_come_from_the_doc_level_ranking() -> None:
    rankings = {"q1": ["a", "a", "b", "c"], "q2": ["x"]}
    qrels = {"q1": {"b": 1}, "q2": {"y": 1}}
    scores = score_rankings(rankings, qrels)
    assert set(scores) == set(METRICS)
    assert scores["recall@5"] == {"q1": 1.0, "q2": 0.0}
    assert scores["mrr@10"]["q1"] == pytest.approx(0.5)  # b is the 2nd document, not the 3rd chunk
    assert scores["ndcg@10"]["q2"] == 0.0


def test_a_record_names_what_it_measured_and_counts_what_it_could_not_score() -> None:
    retriever = Scripted({"q1": ["a", "a", "b"], "q2": ["z"], "q3": ["z"]})
    output = run_queries(
        retriever,
        {"q1": "q1", "q2": "q2", "q3": "q3"},
        k=5,
        warmup=0,
        gate=gate_returning(False, []),
        snapshot=lambda: CpuSnapshot(None, ()),
    )
    qrels = {"q1": {"b": 1}, "q2": {"y": 1}, "q3": {}}  # q3 has nothing relevant: undefined
    record = record_of(output, qrels, rung="R2", variant="legacy-raw", note="x")
    assert (record["rung"], record["variant"], record["note"]) == ("R2", "legacy-raw", "x")
    assert record["questions"] == 3
    assert record["scored"]["ndcg@10"] == 2
    assert record["metrics"]["recall@5"] == pytest.approx(0.5)
    assert record["per_query"]["recall@5"] == {"q1": 1.0, "q2": 0.0}
    assert record["docs_per_query"] == pytest.approx((2 + 1 + 1) / 3)
    assert record["gate"]["idle"] is False
    assert set(record["latency_ms"]) == {"p50", "p95", "mean"}


def test_a_record_with_no_defined_metric_has_no_mean_instead_of_a_wrong_one() -> None:
    retriever = Scripted({"q1": ["a"]})
    output = run_queries(
        retriever, {"q1": "q1"}, k=3, warmup=0, gate=gate_returning(True, []),
        snapshot=lambda: CpuSnapshot(None, ()),
    )  # fmt: skip
    record = record_of(output, {"q1": {}})
    assert record["metrics"] == {name: None for name in METRICS}


def test_fake_backends_are_wired_for_the_phase_tests() -> None:
    backends = fake_backends()
    assert backends.gate(idle_percent=15.0).idle is True
    assert fake_backends(idle=False).gate(idle_percent=15.0).idle is False
