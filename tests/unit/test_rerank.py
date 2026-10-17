"""Tests for step S2.3: the reranker, the fastembed scorer (never loaded) and the commands.

No model is downloaded or run. ``Reranker`` is tested with a fake scorer; ``FastembedScorer`` with
a stand-in for fastembed's ``TextCrossEncoder`` that records how it was called; the commands with
both. Nothing here says anything about how good the real cross-encoder is.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.config import Settings
from geekbot.domain import Chunk, ScoredChunk
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.sqlite import Store
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.retrieval import RetrievalResult
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.lexical import LexicalRetriever
from geekbot.retrieval.rerank import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MODEL,
    FastembedScorer,
    Reranker,
    RerankError,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


def stand_in_chunks(*texts: str) -> tuple[ScoredChunk, ...]:
    return tuple(
        ScoredChunk(
            Chunk(
                chunk_id=f"id-{n}",
                doc_id="d",
                path="/x/doc.md",
                title="doc.md",
                content_hash="h",
                occurrence=0,
                ordinal=n,
                char_start=0,
                char_end=1,
                page=None,
                section_path=(),
                header="h",
                text=text,
            ),
            1.0,
        )
        for n, text in enumerate(texts)
    )


class Base:
    """A retriever that returns prepared chunks and remembers what it was asked."""

    def __init__(
        self, chunks: tuple[ScoredChunk, ...], timings: dict[str, float] | None = None
    ) -> None:
        self.chunks = chunks
        self.timings = timings or {}
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        return RetrievalResult(chunks=self.chunks[:k], timings_ms=self.timings)


class Fixed:
    """A scorer that returns prepared scores and remembers what it was asked."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return self.scores[: len(passages)]


def texts_of(result: RetrievalResult) -> list[str]:
    return [s.chunk.text for s in result.chunks]


# --- the reranker ----------------------------------------------------------------------------


def test_the_scorer_decides_the_order_and_its_scores_are_returned() -> None:
    base = Base(stand_in_chunks("alpha", "beta", "gamma"))
    result = Reranker(base, Fixed([0.1, 0.9, 0.5])).search("q", 3)
    assert texts_of(result) == ["beta", "gamma", "alpha"]
    assert [s.score for s in result.chunks] == [0.9, 0.5, 0.1]


def test_equal_scores_keep_the_order_of_the_base_retriever() -> None:
    base = Base(stand_in_chunks("first", "second", "third"))
    assert texts_of(Reranker(base, Fixed([0.5, 0.5, 0.5])).search("q", 3)) == [
        "first",
        "second",
        "third",
    ]


def test_it_returns_only_the_k_best() -> None:
    base = Base(stand_in_chunks("a", "b", "c", "d"))
    result = Reranker(base, Fixed([0.1, 0.4, 0.3, 0.2])).search("q", 2)
    assert texts_of(result) == ["b", "c"]


def test_the_base_is_asked_for_the_depth_not_for_k() -> None:
    base = Base(stand_in_chunks("a"))
    Reranker(base, Fixed([1.0]), depth=30).search("question", 3)
    assert base.calls == [("question", 30)]
    Reranker(base, Fixed([1.0]), depth=30).search("question", 50)
    assert base.calls[-1] == ("question", 50)  # never fewer than k


def test_the_scorer_reads_the_question_and_each_chunks_text_in_base_order() -> None:
    base = Base(stand_in_chunks("one text", "two text"))
    scorer = Fixed([0.0, 1.0])
    Reranker(base, scorer).search("the question", 2)
    assert scorer.calls == [("the question", ["one text", "two text"])]


def test_no_candidates_means_the_scorer_is_never_called() -> None:
    scorer = Fixed([])
    result = Reranker(Base(()), scorer).search("q", 3)
    assert result.chunks == ()
    assert scorer.calls == []
    assert result.timings_ms["rerank"] == 0.0


def test_a_scorer_that_returns_the_wrong_number_of_scores_is_an_error() -> None:
    base = Base(stand_in_chunks("a", "b", "c"))
    with pytest.raises(RerankError, match="asked for 3 scores, the scorer returned 2"):
        Reranker(base, Fixed([0.1, 0.2])).search("q", 3)


def test_the_timings_keep_the_base_stages_and_add_rerank_and_total() -> None:
    base = Base(
        stand_in_chunks("a", "b"),
        {"dense": 10.0, "lexical": 2.0, "fusion": 0.5, "total": 12.5},
    )
    ticks = iter([0.0, 0.0125, 0.0425])  # start, after the base, end
    result = Reranker(base, Fixed([1.0, 0.0]), clock=lambda: next(ticks)).search("q", 2)
    assert list(result.timings_ms) == ["dense", "lexical", "fusion", "rerank", "total"]
    assert result.timings_ms["dense"] == 10.0  # the base's own numbers, untouched
    assert result.timings_ms["rerank"] == pytest.approx(30.0)
    assert result.timings_ms["total"] == pytest.approx(42.5)


def test_a_base_that_reports_no_stages_is_shown_as_one() -> None:
    ticks = iter([0.0, 0.01, 0.03])
    result = Reranker(Base(stand_in_chunks("a")), Fixed([1.0]), clock=lambda: next(ticks)).search(
        "q", 1
    )
    assert list(result.timings_ms) == ["candidates", "rerank", "total"]
    assert result.timings_ms["candidates"] == pytest.approx(10.0)


def test_reranker_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        Reranker(Base(()), Fixed([]), depth=0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        Reranker(Base(()), Fixed([])).search("q", 0)


def test_the_defaults_are_the_plans() -> None:
    assert DEFAULT_RERANK_MODEL == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert DEFAULT_RERANK_DEPTH == 30
    settings = Settings(_env_file=None)
    assert settings.rerank_model == DEFAULT_RERANK_MODEL
    assert settings.rerank_depth == DEFAULT_RERANK_DEPTH
    assert settings.rerank_threads is None


# --- the fastembed scorer, with fastembed's model class replaced -----------------------------


class FakeCrossEncoder:
    """Stands in for fastembed's ``TextCrossEncoder``: records how it was built and used."""

    instances: ClassVar[list["FakeCrossEncoder"]] = []
    fail_to_load: ClassVar[bool] = False
    fail_to_score: ClassVar[bool] = False
    short_answer: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        if FakeCrossEncoder.fail_to_load:
            raise OSError("no network")
        self.kwargs = kwargs
        self.queries: list[tuple[str, list[str]]] = []
        FakeCrossEncoder.instances.append(self)

    def rerank(self, query: str, documents: list[str]) -> Iterator[float]:
        if FakeCrossEncoder.fail_to_score:
            raise RuntimeError("onnx exploded")
        self.queries.append((query, documents))
        count = len(documents) - (1 if FakeCrossEncoder.short_answer else 0)
        return iter([float(len(d)) for d in documents[:count]])


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> type[FakeCrossEncoder]:
    FakeCrossEncoder.instances = []
    FakeCrossEncoder.fail_to_load = False
    FakeCrossEncoder.fail_to_score = False
    FakeCrossEncoder.short_answer = False
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", FakeCrossEncoder)
    return FakeCrossEncoder


def test_nothing_is_loaded_until_the_first_score(fake_encoder: type[FakeCrossEncoder]) -> None:
    scorer = FastembedScorer("some/model", cache_dir=Path("models"))
    assert fake_encoder.instances == []  # creating the scorer touches no model
    assert scorer.score("q", []) == []
    assert fake_encoder.instances == []  # and scoring nothing does not either


def test_the_model_is_built_once_on_the_cpu_and_scores_come_back_as_floats(
    fake_encoder: type[FakeCrossEncoder],
) -> None:
    scorer = FastembedScorer("some/model", cache_dir=Path("models"), threads=2)
    assert scorer.score("q", ["abc", "de"]) == [3.0, 2.0]
    assert scorer.score("q2", ["x"]) == [1.0]
    assert len(fake_encoder.instances) == 1
    built = fake_encoder.instances[0]
    assert built.kwargs == {
        "model_name": "some/model",
        "cache_dir": str(Path("models")),
        "threads": 2,
        "providers": ["CPUExecutionProvider"],
    }
    assert built.queries == [("q", ["abc", "de"]), ("q2", ["x"])]


def test_no_cache_dir_leaves_fastembeds_default(fake_encoder: type[FakeCrossEncoder]) -> None:
    FastembedScorer("m").score("q", ["a"])
    assert fake_encoder.instances[0].kwargs["cache_dir"] is None


def test_a_model_that_cannot_be_loaded_is_one_clear_error(
    fake_encoder: type[FakeCrossEncoder],
) -> None:
    fake_encoder.fail_to_load = True
    with pytest.raises(RerankError) as caught:
        FastembedScorer("some/model").score("q", ["a"])
    message = str(caught.value)
    assert "'some/model'" in message
    assert "no network" in message
    assert "needs network access once" in message


def test_a_scoring_failure_is_wrapped(fake_encoder: type[FakeCrossEncoder]) -> None:
    scorer = FastembedScorer("m")
    scorer.score("q", ["a"])  # loads fine
    fake_encoder.fail_to_score = True
    with pytest.raises(RerankError, match="onnx exploded"):
        scorer.score("q", ["a"])


def test_too_few_scores_from_the_model_is_an_error(fake_encoder: type[FakeCrossEncoder]) -> None:
    fake_encoder.short_answer = True
    with pytest.raises(RerankError, match="asked for 2 scores, the reranker returned 1"):
        FastembedScorer("m").score("q", ["a", "b"])


# --- with the real hybrid retriever ----------------------------------------------------------

TOPICS = ("zebra", "quartz", "bread", "river")
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=len(TOPICS),
    with_header=True,
)


class TopicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(sum(w.startswith(topic) for w in words)) for topic in TOPICS]
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm if norm else 0.01 for c in counts]


CORPUS = {
    "zebras.md": "Zebras have black and white stripes.",
    "quartz.txt": "Quartz is a hard mineral.",
    "bread.md": "Bread needs flour and sourdough bread uses a starter.",
    "river.md": "The river Brahmaputra flows south. Ticket Qx9812 is about it.",
}


class OverlapScorer:
    """Scores a passage by how many words of the query it contains: a crude stand-in for a model."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        wanted = set(query.lower().split())
        return [float(len(wanted & set(p.lower().replace(".", "").split()))) for p in passages]


def settings_for(root: Path) -> Settings:
    return Settings(_env_file=None, data_dir=root / "data")


@pytest.fixture
def loaded(tmp_path: Path) -> Iterator[Workspace]:
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in CORPUS.items():
        (folder / name).write_text(text, encoding="utf-8")
    files, _ = collect_files([folder])
    with Workspace.open(settings_for(tmp_path), embeddings=prefixed, spec=SPEC) as workspace:
        report = Ingestor(
            workspace.store, workspace.dense, workspace.embedder, chunk_tokens=40, overlap_tokens=2
        ).ingest(files)
        assert not report.has_failures
        yield workspace


def test_the_reranker_can_lift_the_chunk_the_scorer_prefers(loaded: Workspace) -> None:
    dense = DenseRetriever(loaded.store, loaded.dense, loaded.embedder)
    hybrid = HybridRetriever(dense, LexicalRetriever(loaded.store))
    result = Reranker(hybrid, OverlapScorer()).search("river ticket qx9812", 2)
    assert result.chunks[0].chunk.title == "river.md"
    assert list(result.timings_ms) == ["dense", "lexical", "fusion", "rerank", "total"]


# --- the commands ----------------------------------------------------------------------------

runner = CliRunner()


class FakeScorerForCli:
    """Replaces ``FastembedScorer`` in the commands: no model, and a switch to make it fail."""

    fail: ClassVar[bool] = False
    built_with: ClassVar[dict[str, Any]] = {}

    def __init__(
        self, model_name: str, *, cache_dir: Path | None = None, threads: int | None = None
    ) -> None:
        FakeScorerForCli.built_with = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "threads": threads,
        }
        self.model_name = model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if FakeScorerForCli.fail:
            raise RerankError("the reranker model 'x' could not be loaded (OSError: no network)")
        return OverlapScorer().score(query, passages)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    FakeScorerForCli.fail = False
    real_open = Workspace.open.__func__  # type: ignore[attr-defined]
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")

    def fake_open(
        cls: type[Workspace], settings: Settings, *, reset_dense: bool = False, **_: Any
    ) -> Workspace:
        return real_open(cls, settings, reset_dense=reset_dense, embeddings=prefixed, spec=SPEC)

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    monkeypatch.setattr("geekbot.retrieval.rerank.FastembedScorer", FakeScorerForCli)
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(docs)]).exit_code == 0


def test_hybrid_rerank_mode_prints_the_reranked_chunks_and_all_stage_timings(cli: None) -> None:
    result = runner.invoke(
        app, ["search", "river ticket qx9812", "--mode", "hybrid_rerank", "--top-k", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "1. " in result.output
    assert result.output.index("river.md") < result.output.index("2. ")  # river is first
    for stage in ("dense ", "lexical ", "fusion ", "rerank ", "total "):
        assert stage in result.output
    assert FakeScorerForCli.built_with["model_name"] == DEFAULT_RERANK_MODEL
    assert FakeScorerForCli.built_with["cache_dir"].name == "models"


def test_a_reranker_that_cannot_load_is_one_line_not_a_traceback(cli: None) -> None:
    FakeScorerForCli.fail = True
    result = runner.invoke(app, ["search", "bread", "--mode", "hybrid_rerank"])
    assert result.exit_code == 1
    assert "no network" in result.output
    assert "Traceback" not in result.output


def test_rerank_timing_prints_a_table_for_each_depth(cli: None) -> None:
    result = runner.invoke(
        app,
        [
            "rerank-timing",
            "--query",
            "bread",
            "--query",
            "river",
            "--depths",
            "1,3",
            "--repeats",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"reranker {DEFAULT_RERANK_MODEL} on the CPU"
    rows = [line.split() for line in lines if line.split()[:1] in (["1"], ["3"])]
    assert [row[0] for row in rows] == ["1", "3"]
    assert [row[1] for row in rows] == ["4", "4"]  # 2 queries x 2 repeats


def test_rerank_timing_says_when_there_are_too_few_chunks(cli: None) -> None:
    result = runner.invoke(app, ["rerank-timing", "--query", "bread", "--depths", "50"])
    assert result.exit_code == 1
    assert "only 4 chunks are stored but the largest depth is 50" in result.output


def test_rerank_timing_checks_its_depths(cli: None) -> None:
    for bad in ("a,b", ","):
        result = runner.invoke(app, ["rerank-timing", "--query", "x", "--depths", bad])
        assert result.exit_code == 1
        assert "--depths" in result.output


def test_rerank_timing_never_creates_a_knowledge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "mistyped"))
    result = runner.invoke(app, ["rerank-timing", "--query", "x", "--depths", "1"])
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output
    assert not (tmp_path / "mistyped").exists()


def test_rerank_timing_needs_a_query(cli: None) -> None:
    assert runner.invoke(app, ["rerank-timing"]).exit_code == 2


def test_rerank_timing_reports_a_held_lock(cli: None, tmp_path: Path) -> None:
    with Store(tmp_path / "data"):
        result = runner.invoke(app, ["rerank-timing", "--query", "x", "--depths", "1"])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_rerank_timing_reports_a_reranker_that_cannot_load(cli: None) -> None:
    FakeScorerForCli.fail = True
    result = runner.invoke(app, ["rerank-timing", "--query", "x", "--depths", "1"])
    assert result.exit_code == 1
    assert "no network" in result.output
