"""Tests for step S3.1: ``build_retriever`` turns a mode name into the right retriever.

Both ``geekbot search`` and ``geekbot ask`` build their retriever here, so these tests are what keeps
"hybrid_rerank" meaning the same thing in both. The documents go through the real pipeline (SQLite,
FTS5, Chroma) with a fake embedder; the cross-encoder is replaced by a scorer that needs no model.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from geekbot.config import Settings
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.retrieval import Retriever, SearchMode
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.factory import build_retriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.lexical import LexicalRetriever
from geekbot.retrieval.rerank import Reranker

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

TOPICS = ("zebra", "quartz", "bread", "river")
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=len(TOPICS),
    with_header=True,
)
CORPUS = {
    "zebras.md": "Zebras have black and white stripes.",
    "quartz.txt": "Quartz is a hard mineral.",
    "bread.md": "Bread needs flour and sourdough bread uses a starter.",
    "river.md": "The river Brahmaputra flows south.",
}


class TopicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(sum(w.startswith(topic) for w in words)) for topic in TOPICS]
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm if norm else 0.01 for c in counts]


class FakeScorer:
    """Stands in for the cross-encoder: remembers how it was built and what it was asked to score."""

    built: ClassVar[list["FakeScorer"]] = []

    def __init__(
        self, model_name: str, *, cache_dir: Path | None = None, threads: int | None = None
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.asked: list[int] = []
        FakeScorer.built.append(self)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.asked.append(len(passages))
        return [float(len(passage)) for passage in passages]  # longer passage, higher score


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Workspace]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    files, _ = collect_files([docs])
    with Workspace.open(settings, embeddings=prefixed, spec=SPEC) as opened:
        assert not Ingestor(opened.store, opened.dense, opened.embedder).ingest(files).has_failures
        yield opened


@pytest.fixture
def fake_scorer(monkeypatch: pytest.MonkeyPatch) -> type[FakeScorer]:
    FakeScorer.built = []
    monkeypatch.setattr("geekbot.retrieval.rerank.FastembedScorer", FakeScorer)
    return FakeScorer


def titles(retriever: Retriever, query: str, k: int = 4) -> list[str]:
    return [scored.chunk.title for scored in retriever.search(query, k).chunks]


def test_each_mode_name_gives_the_matching_retriever(
    workspace: Workspace, fake_scorer: type[FakeScorer], tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    assert isinstance(build_retriever(SearchMode.LEXICAL, workspace, settings), LexicalRetriever)
    assert isinstance(build_retriever(SearchMode.DENSE, workspace, settings), DenseRetriever)
    assert isinstance(build_retriever(SearchMode.HYBRID, workspace, settings), HybridRetriever)
    assert isinstance(build_retriever(SearchMode.HYBRID_RERANK, workspace, settings), Reranker)


def test_the_retrievers_find_what_they_are_asked_for(
    workspace: Workspace, fake_scorer: type[FakeScorer], tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    for mode in (SearchMode.DENSE, SearchMode.LEXICAL, SearchMode.HYBRID):
        assert titles(build_retriever(mode, workspace, settings), "bread")[0] == "bread.md", mode


def test_only_the_reranking_mode_builds_a_scorer_and_it_gets_the_settings(
    workspace: Workspace, fake_scorer: type[FakeScorer], tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        rerank_model="some/model",
        rerank_threads=3,
        rerank_depth=2,
    )
    for mode in (SearchMode.DENSE, SearchMode.LEXICAL, SearchMode.HYBRID):
        build_retriever(mode, workspace, settings)
    assert fake_scorer.built == []  # nothing was loaded for a mode that never reranks
    reranker = build_retriever(SearchMode.HYBRID_RERANK, workspace, settings)
    (scorer,) = fake_scorer.built
    assert scorer.model_name == "some/model"
    assert scorer.threads == 3
    assert scorer.cache_dir == tmp_path / "data" / "models"
    reranker.search("bread", 1)
    assert scorer.asked == [2]  # the rerank depth: it scores two candidates, not all four


def test_the_candidates_option_reaches_the_hybrid_search(
    workspace: Workspace, fake_scorer: type[FakeScorer], tmp_path: Path
) -> None:
    """A value the hybrid retriever refuses proves it was passed on."""
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    with pytest.raises(ValueError, match="candidates"):
        build_retriever(SearchMode.HYBRID, workspace, settings, candidates=0)
    with pytest.raises(ValueError, match="candidates"):
        build_retriever(SearchMode.HYBRID_RERANK, workspace, settings, candidates=0)
