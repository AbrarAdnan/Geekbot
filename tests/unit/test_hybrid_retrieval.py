"""Tests for step S2.2: lexical retrieval, hybrid retrieval and ``geekbot search --mode`` (no model).

The dense side uses a fake embedder whose vectors count topic words, the lexical side is real FTS5.
Layout: the lexical retriever, then the hybrid wiring with stand-in retrievers, then both sides
real, then the command.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.config import Settings
from geekbot.domain import Chunk, ScoredChunk
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.sqlite import Store
from geekbot.index.workspace import Workspace
from geekbot.ingest.ids import doc_id_of
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.retrieval import RetrievalResult
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.hybrid import DEFAULT_CANDIDATES, HybridRetriever
from geekbot.retrieval.lexical import LexicalRetriever

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


class TopicEmbeddings:
    """Vectors made of topic-word counts, so a question about bread points the way bread chunks do."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(sum(w.startswith(topic) for w in words)) for topic in TOPICS]
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm if norm else 0.01 for c in counts]


def settings_for(root: Path) -> Settings:
    return Settings(_env_file=None, data_dir=root / "data")


# One short chunk per document, so a document is one chunk and "top 2 chunks" means "top 2 documents".
CORPUS = {
    "zebras.md": "Zebras have black and white stripes.",
    "quartz.txt": "Quartz is a hard mineral.",
    "bread.md": "Bread needs flour and sourdough bread uses a starter.",
    "river.md": "The river Brahmaputra flows south. Ticket Qx9812 is about it.",
}


def write_corpus(folder: Path) -> list[Path]:
    folder.mkdir(exist_ok=True)
    for name, text in CORPUS.items():
        (folder / name).write_text(text, encoding="utf-8")
    files, _ = collect_files([folder])
    return files


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")
    with Workspace.open(settings_for(tmp_path), embeddings=prefixed, spec=SPEC) as opened:
        yield opened


@pytest.fixture
def loaded(ws: Workspace, tmp_path: Path) -> Workspace:
    report = Ingestor(ws.store, ws.dense, ws.embedder, chunk_tokens=40, overlap_tokens=2).ingest(
        write_corpus(tmp_path / "docs")
    )
    assert not report.has_failures
    return ws


def titles(result: RetrievalResult) -> list[str]:
    return [s.chunk.title for s in result.chunks]


# --- the lexical retriever -------------------------------------------------------------------


def test_lexical_finds_the_exact_rare_token(loaded: Workspace) -> None:
    result = LexicalRetriever(loaded.store).search("Qx9812", 5)
    assert titles(result) == ["river.md"]
    assert result.chunks[0].score > 0  # a positive BM25 score, higher is better


def test_lexical_ranks_the_chunk_with_more_of_the_words_first(loaded: Workspace) -> None:
    result = LexicalRetriever(loaded.store).search("bread starter flour", 5)
    assert titles(result)[0] == "bread.md"
    scores = [s.score for s in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_lexical_cannot_match_a_synonym_which_is_what_dense_is_for(loaded: Workspace) -> None:
    assert LexicalRetriever(loaded.store).search("loaf baking", 5).chunks == ()


def test_lexical_blank_query_and_bad_k(loaded: Workspace) -> None:
    retriever = LexicalRetriever(loaded.store)
    assert retriever.search("   ", 3).chunks == ()
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("bread", 0)


def test_lexical_limits_to_k(loaded: Workspace) -> None:
    assert len(LexicalRetriever(loaded.store).search("the", 1).chunks) <= 1


def test_lexical_returns_only_ready_documents(loaded: Workspace, tmp_path: Path) -> None:
    loaded.store.set_document_state(doc_id_of(tmp_path / "docs" / "bread.md"), "embedding")
    assert LexicalRetriever(loaded.store).search("bread", 5).chunks == ()


def test_lexical_reports_its_time() -> None:
    ticks = iter([0.0, 0.004])
    with_clock = LexicalRetriever(_EmptyStore(), clock=lambda: next(ticks))  # type: ignore[arg-type]
    result = with_clock.search("anything", 3)
    assert result.timings_ms == pytest.approx({"bm25": 4.0, "total": 4.0})


class _EmptyStore:
    def search_bm25(self, text: str, limit: int = 10) -> list[ScoredChunk]:
        return []


# --- hybrid wiring, with stand-in retrievers -------------------------------------------------


def stand_in_chunks(*ids: str) -> tuple[ScoredChunk, ...]:
    """Chunks whose text is their id, so each is distinct and easy to read in an assertion."""
    return tuple(
        ScoredChunk(
            Chunk(
                chunk_id=i,
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
                text=i,
            ),
            1.0,
        )
        for n, i in enumerate(ids)
    )


class Fixed:
    """A retriever that returns a prepared list and remembers what it was asked."""

    def __init__(self, chunks: tuple[ScoredChunk, ...]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 7.0})


def test_hybrid_asks_each_side_for_the_candidate_depth_not_for_k() -> None:
    dense, lexical = Fixed(stand_in_chunks("a", "b")), Fixed(stand_in_chunks("b", "a"))
    HybridRetriever(dense, lexical, candidates=50).search("q", 3)
    assert dense.calls == [("q", 50)]
    assert lexical.calls == [("q", 50)]


def test_hybrid_asks_for_at_least_k_when_k_is_larger_than_the_depth() -> None:
    dense, lexical = Fixed(stand_in_chunks("a")), Fixed(stand_in_chunks("a"))
    HybridRetriever(dense, lexical, candidates=5).search("q", 80)
    assert dense.calls == [("q", 80)]


def test_hybrid_fuses_by_rank_and_cuts_to_k() -> None:
    both = stand_in_chunks("a", "b", "c")
    dense, lexical = Fixed(both), Fixed((both[1], both[0], both[2]))
    result = HybridRetriever(dense, lexical).search("q", 2)
    assert [s.chunk.text for s in result.chunks] == ["a", "b"]  # a and b tie; list 1 (dense) wins
    assert len(result.chunks) == 2
    assert result.chunks[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_reports_a_time_for_each_side_the_fusion_and_the_total() -> None:
    dense, lexical = Fixed(stand_in_chunks("a")), Fixed(stand_in_chunks("a"))
    timings = HybridRetriever(dense, lexical).search("q", 1).timings_ms
    assert list(timings) == ["dense", "lexical", "fusion", "total"]
    assert timings["dense"] == timings["lexical"] == 7.0  # each side's own total
    assert all(ms >= 0 for ms in timings.values())


def test_hybrid_validates_its_arguments() -> None:
    fixed = Fixed(())
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        HybridRetriever(fixed, fixed, candidates=0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        HybridRetriever(fixed, fixed).search("q", 0)
    assert DEFAULT_CANDIDATES == 50


def test_hybrid_of_two_empty_sides_is_empty() -> None:
    fixed = Fixed(())
    assert HybridRetriever(fixed, fixed).search("q", 3).chunks == ()


# --- both sides real -------------------------------------------------------------------------


def hybrid(workspace: Workspace) -> HybridRetriever:
    dense = DenseRetriever(workspace.store, workspace.dense, workspace.embedder)
    return HybridRetriever(dense, LexicalRetriever(workspace.store))


def test_hybrid_finds_what_each_side_alone_would_miss(loaded: Workspace) -> None:
    # "bread" is a meaning the embedder knows; "Qx9812" is only a token. Dense puts the bread
    # chunk first and has nothing to say about the ticket; BM25 matches both words.
    query = "bread Qx9812"
    dense = DenseRetriever(loaded.store, loaded.dense, loaded.embedder).search(query, 1)
    assert titles(dense) == ["bread.md"]
    assert set(titles(LexicalRetriever(loaded.store).search(query, 4))) == {"bread.md", "river.md"}
    assert set(titles(hybrid(loaded).search(query, 2))) == {"bread.md", "river.md"}


def test_hybrid_returns_nothing_for_a_blank_query(loaded: Workspace) -> None:
    assert hybrid(loaded).search("   ", 3).chunks == ()


def test_hybrid_only_returns_ready_documents(loaded: Workspace, tmp_path: Path) -> None:
    loaded.store.set_document_state(doc_id_of(tmp_path / "docs" / "bread.md"), "embedding")
    assert "bread.md" not in titles(hybrid(loaded).search("bread Qx9812", 10))


def test_hybrid_scores_are_rrf_scores(loaded: Workspace) -> None:
    result = hybrid(loaded).search("bread Qx9812", 2)
    # a chunk found by both sides scores between one and two terms of 1/(60 + rank)
    assert all(1 / 64 < s.score <= 2 / 61 for s in result.chunks)


def test_the_stores_need_no_changes_for_hybrid(loaded: Workspace) -> None:
    before = loaded.dense.count()
    hybrid(loaded).search("bread", 3)
    assert loaded.dense.count() == before  # searching is read-only


# --- the command -----------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    state: dict[str, Any] = {"opens": 0, "forbid": False}
    real_open = Workspace.open.__func__  # type: ignore[attr-defined]
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")

    def fake_open(
        cls: type[Workspace], settings: Settings, *, reset_dense: bool = False, **_: Any
    ) -> Workspace:
        assert not state["forbid"], "this mode must not open the workspace (it needs no Ollama)"
        state["opens"] += 1
        return real_open(cls, settings, reset_dense=reset_dense, embeddings=prefixed, spec=SPEC)

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    docs = tmp_path / "docs"
    write_corpus(docs)
    assert runner.invoke(app, ["ingest", str(docs)]).exit_code == 0
    return state


def test_lexical_mode_needs_neither_ollama_nor_the_vector_index(cli: dict[str, Any]) -> None:
    cli["forbid"] = True
    result = runner.invoke(app, ["search", "Qx9812", "--mode", "lexical"])
    assert result.exit_code == 0, result.output
    assert "river.md" in result.output
    assert "bm25 " in result.output  # the timing line names its stage


def test_hybrid_mode_prints_the_fused_chunks_and_the_stage_timings(cli: dict[str, Any]) -> None:
    result = runner.invoke(app, ["search", "bread Qx9812", "--mode", "hybrid", "--top-k", "2"])
    assert result.exit_code == 0, result.output
    assert "bread.md" in result.output
    assert "river.md" in result.output
    assert "dense " in result.output
    assert "lexical " in result.output
    assert "fusion " in result.output


def test_hybrid_candidates_must_be_positive(cli: dict[str, Any]) -> None:
    result = runner.invoke(app, ["search", "x", "--mode", "hybrid", "--candidates", "0"])
    assert result.exit_code == 2


def test_lexical_mode_on_an_empty_knowledge_base_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "empty"))
    result = runner.invoke(app, ["search", "anything", "--mode", "lexical"])
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output


def test_lexical_mode_reports_a_held_lock(cli: dict[str, Any], tmp_path: Path) -> None:
    cli["forbid"] = True
    with Store(tmp_path / "data"):
        result = runner.invoke(app, ["search", "bread", "--mode", "lexical"])
    assert result.exit_code == 1
    assert "error:" in result.output
