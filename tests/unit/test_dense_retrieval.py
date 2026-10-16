"""Tests for step S2.1: dense retrieval and ``geekbot search`` (no model, no network).

The embedder is a fake whose vectors have a real geometry: each text becomes a count of a few topic
words, so a question about zebras points the same way as a chunk about zebras. That lets the tests
check *ranking* (which a hash-based fake cannot), while SQLite and Chroma are real.

Layout: ``resolve_hits`` (this step's ✍️ piece, written by Claude in prototyping mode, D-047), then
the retriever, then ready-only behaviour, then the command.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.config import Settings
from geekbot.domain import NewDocument
from geekbot.index.dense import DenseEntry, DenseHit
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.workspace import Workspace
from geekbot.ingest.chunking import ChunkDraft
from geekbot.ingest.ids import IdentifiedChunk, doc_id_of, identify_chunks
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.llm.models import OllamaError
from geekbot.retrieval import RetrievalResult, format_result
from geekbot.retrieval.dense import DenseRetriever, resolve_hits

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

TOPICS = ("zebra", "quartz", "bread", "river")
DIM = len(TOPICS)
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=DIM,
    with_header=True,
)


class TopicEmbeddings:
    """Counts of topic words (so the vector of a text about zebras points along the first axis).

    Records what it was asked, so a test can see which prefix reached it.
    """

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(sum(w.startswith(topic) for w in words)) for topic in TOPICS]
        norm = math.sqrt(sum(c * c for c in counts))
        # a tiny constant keeps a text with no topic word from being the zero vector
        return [c / norm if norm else 0.01 for c in counts]


def settings_for(root: Path) -> Settings:
    return Settings(_env_file=None, data_dir=root / "data")


@pytest.fixture
def embeddings() -> TopicEmbeddings:
    return TopicEmbeddings()


@pytest.fixture
def ws(tmp_path: Path, embeddings: TopicEmbeddings) -> Iterator[Workspace]:
    prefixed = PrefixedEmbeddings(embeddings, document_prefix="doc: ", query_prefix="q: ")
    with Workspace.open(settings_for(tmp_path), embeddings=prefixed, spec=SPEC) as opened:
        yield opened


def write_corpus(folder: Path) -> list[Path]:
    folder.mkdir(exist_ok=True)
    texts = {
        "zebras.md": "Zebras live on the savanna.\n\nA zebra has black and white stripes.",
        "quartz.txt": "Quartz is a hard mineral.\n\nQuartz crystals are found in granite.",
        "bread.md": "Bread needs flour and water.\n\nSourdough bread uses a starter.",
    }
    for name, text in texts.items():
        (folder / name).write_text(text, encoding="utf-8")
    files, _ = collect_files([folder])
    return files


def ingest(ws: Workspace, files: list[Path]) -> None:
    report = Ingestor(ws.store, ws.dense, ws.embedder, chunk_tokens=20, overlap_tokens=2).ingest(
        files
    )
    assert not report.has_failures


def retriever(ws: Workspace) -> DenseRetriever:
    return DenseRetriever(ws.store, ws.dense, ws.embedder)


# --- resolve_hits ----------------------------------------------------------------------------

PATH = "/data/a.md"


def chunks_for(texts: list[str], path: str = PATH) -> list[IdentifiedChunk]:
    drafts = [
        ChunkDraft(
            text=t,
            char_start=i * 50,
            char_end=i * 50 + len(t),
            page=None,
            section_path=(),
            header="a.md",
            ordinal=i,
        )
        for i, t in enumerate(texts)
    ]
    return identify_chunks(path, drafts)


def store_document(ws: Workspace, texts: list[str], path: str, *, ready: bool) -> list[str]:
    doc_id = doc_id_of(path)
    ws.store.upsert_document(NewDocument(doc_id, path, Path(path).name, "markdown", "f" * 64, "p1"))
    items = chunks_for(texts, path)
    ws.store.sync_chunks(doc_id, items)
    if ready:
        ws.store.set_document_state(doc_id, "ready")
    return [item.chunk_id for item in items]


def test_hits_become_chunks_in_the_same_order(ws: Workspace) -> None:
    ids = store_document(ws, ["one", "two", "three"], PATH, ready=True)
    hits = [DenseHit(ids[2], 0.9), DenseHit(ids[0], 0.8), DenseHit(ids[1], 0.1)]
    resolved = resolve_hits(ws.store, hits)
    assert [s.chunk.text for s in resolved] == ["three", "one", "two"]
    assert [s.score for s in resolved] == [0.9, 0.8, 0.1]  # Chroma's similarity, unchanged


def test_chunks_of_a_document_that_is_not_ready_are_dropped(ws: Workspace) -> None:
    ready = store_document(ws, ["visible"], "/data/ready.md", ready=True)
    hidden = store_document(ws, ["hidden"], "/data/pending.md", ready=False)
    hits = [DenseHit(hidden[0], 0.99), DenseHit(ready[0], 0.5)]
    assert [s.chunk.text for s in resolve_hits(ws.store, hits)] == ["visible"]


def test_a_vector_with_no_stored_chunk_is_dropped_without_an_error(ws: Workspace) -> None:
    ids = store_document(ws, ["real"], PATH, ready=True)
    hits = [DenseHit("orphan-vector", 0.99), DenseHit(ids[0], 0.5)]
    assert [s.chunk.text for s in resolve_hits(ws.store, hits)] == ["real"]


def test_no_hits_give_no_chunks(ws: Workspace) -> None:
    assert resolve_hits(ws.store, []) == []
    assert resolve_hits(ws.store, [DenseHit("nothing", 1.0)]) == []


def test_a_chunk_carries_its_document_so_a_citation_needs_no_second_query(ws: Workspace) -> None:
    ids = store_document(ws, ["text"], "/data/notes.md", ready=True)
    chunk = resolve_hits(ws.store, [DenseHit(ids[0], 1.0)])[0].chunk
    assert (chunk.path, chunk.title) == ("/data/notes.md", "notes.md")


# --- the retriever ---------------------------------------------------------------------------


def test_the_nearest_chunk_is_about_the_question(ws: Workspace, tmp_path: Path) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    result = retriever(ws).search("zebra stripes", 3)
    top = result.chunks[0]
    assert top.chunk.title == "zebras.md"
    assert "zebra" in top.chunk.text.lower()
    assert top.score > 0.5
    scores = [s.score for s in result.chunks]
    assert scores == sorted(scores, reverse=True)  # best first


@pytest.mark.parametrize(
    ("question", "title"),
    [("zebra", "zebras.md"), ("quartz crystals", "quartz.txt"), ("bread starter", "bread.md")],
)
def test_each_question_finds_its_own_document(
    ws: Workspace, tmp_path: Path, question: str, title: str
) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    assert retriever(ws).search(question, 1).chunks[0].chunk.title == title


def test_k_limits_the_results_and_asking_for_more_than_there_is_returns_all(
    ws: Workspace, tmp_path: Path
) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    total = ws.dense.count()
    assert len(retriever(ws).search("zebra", 2).chunks) == 2
    assert len(retriever(ws).search("zebra", 1000).chunks) == total


def test_a_blank_query_and_an_empty_index_give_nothing(
    ws: Workspace, embeddings: TopicEmbeddings, tmp_path: Path
) -> None:
    assert retriever(ws).search("zebra", 3).chunks == ()  # nothing indexed yet
    ingest(ws, write_corpus(tmp_path / "docs"))
    embeddings.queries.clear()
    result = retriever(ws).search("   ", 3)
    assert result == RetrievalResult(chunks=(), timings_ms={})
    assert embeddings.queries == []  # no embedder call for a query with no words


def test_k_must_be_positive(ws: Workspace) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever(ws).search("zebra", 0)


def test_the_query_and_the_documents_are_embedded_in_their_own_modes(
    ws: Workspace, embeddings: TopicEmbeddings, tmp_path: Path
) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    retriever(ws).search("zebra stripes", 1)
    assert embeddings.queries == ["q: zebra stripes"]
    assert embeddings.documents
    assert all(text.startswith("doc: ") for text in embeddings.documents)


def test_timings_name_every_stage(ws: Workspace, tmp_path: Path) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    timings = retriever(ws).search("zebra", 3).timings_ms
    assert list(timings) == ["embed_query", "vector_search", "resolve", "total"]
    assert all(ms >= 0 for ms in timings.values())
    assert timings["total"] >= timings["embed_query"]


def test_timings_come_from_the_injected_clock(ws: Workspace, tmp_path: Path) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    ticks = iter(i * 0.001 for i in range(100))  # every call to the clock is 1 ms later
    timings = (
        DenseRetriever(ws.store, ws.dense, ws.embedder, clock=lambda: next(ticks))
        .search("zebra", 2)
        .timings_ms
    )
    assert timings["embed_query"] == pytest.approx(1.0)
    assert timings["vector_search"] == pytest.approx(1.0)
    assert timings["resolve"] == pytest.approx(1.0)
    assert timings["total"] == pytest.approx(5.0)  # five clock calls, 1 ms apart


# --- ready only ------------------------------------------------------------------------------


def test_a_document_that_is_being_reingested_is_not_returned(ws: Workspace, tmp_path: Path) -> None:
    files = write_corpus(tmp_path / "docs")
    ingest(ws, files)
    zebra = doc_id_of(tmp_path / "docs" / "zebras.md")
    ws.store.set_document_state(zebra, "embedding")  # as if an edit were half processed
    titles = {s.chunk.title for s in retriever(ws).search("zebra", 10).chunks}
    assert "zebras.md" not in titles
    assert titles  # the others are still found


def test_hidden_hits_do_not_make_the_answer_short(ws: Workspace, tmp_path: Path) -> None:
    # Eight zebra chunks that match the question perfectly but are not ready, and three quartz
    # chunks that are. Taking the top 3 and then dropping the hidden ones would return nothing.
    quartz = store_document(
        ws, ["quartz one", "quartz two", "quartz three"], "/data/q.md", ready=True
    )
    hidden = store_document(ws, [f"zebra {i}" for i in range(8)], "/data/z.md", ready=False)
    vectors = TopicEmbeddings()
    ws.dense.upsert(
        [
            DenseEntry(chunk_id, doc_id_of("/data/q.md"), vectors._vector("quartz"))
            for chunk_id in quartz
        ]
        + [
            DenseEntry(chunk_id, doc_id_of("/data/z.md"), vectors._vector("zebra"))
            for chunk_id in hidden
        ]
    )
    result = retriever(ws).search("zebra", 3)
    assert {s.chunk.text for s in result.chunks} == {"quartz one", "quartz two", "quartz three"}


def test_a_vector_without_a_chunk_never_reaches_the_result(ws: Workspace, tmp_path: Path) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    ws.dense.upsert([DenseEntry("orphan", "no-doc", TopicEmbeddings()._vector("zebra"))])
    chunk_ids = {s.chunk.chunk_id for s in retriever(ws).search("zebra", 50).chunks}
    assert "orphan" not in chunk_ids


# --- formatting ------------------------------------------------------------------------------


def test_the_result_reads_as_numbered_entries_with_a_timing_line(
    ws: Workspace, tmp_path: Path
) -> None:
    ingest(ws, write_corpus(tmp_path / "docs"))
    text = format_result(retriever(ws).search("zebra", 2))
    lines = text.splitlines()
    assert lines[0].startswith("1. ")
    assert "zebras.md" in lines[0]
    assert lines[1].startswith("   ")
    assert lines[-1].startswith("embed_query ")
    assert "ms" in lines[-1]


def test_an_empty_result_says_so() -> None:
    assert format_result(RetrievalResult(chunks=())) == "no results"


def test_a_long_chunk_is_shortened_to_an_excerpt(ws: Workspace) -> None:
    ids = store_document(ws, ["word " * 200], PATH, ready=True)
    scored = resolve_hits(ws.store, [DenseHit(ids[0], 0.5)])
    text = format_result(RetrievalResult(chunks=tuple(scored)), excerpt_chars=30)
    excerpt = text.splitlines()[1].strip()
    assert len(excerpt) == 30
    assert excerpt.endswith("…")


# --- the command -----------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embeddings: TopicEmbeddings
) -> dict[str, Any]:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    state: dict[str, Any] = {"error": None}
    real_open = Workspace.open.__func__  # type: ignore[attr-defined]
    prefixed = PrefixedEmbeddings(embeddings, document_prefix="doc: ", query_prefix="q: ")

    def fake_open(
        cls: type[Workspace], settings: Settings, *, reset_dense: bool = False, **_: Any
    ) -> Workspace:
        if state["error"] is not None:
            raise state["error"]
        return real_open(cls, settings, reset_dense=reset_dense, embeddings=prefixed, spec=SPEC)

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    return state


def test_search_prints_the_nearest_chunks(cli: dict[str, Any], tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    write_corpus(docs)
    assert runner.invoke(app, ["ingest", str(docs)]).exit_code == 0

    result = runner.invoke(app, ["search", "zebra stripes", "--top-k", "2"])
    assert result.exit_code == 0, result.output
    assert "1. " in result.output
    assert "zebras.md" in result.output
    assert "embed_query" in result.output


def test_search_on_an_empty_knowledge_base_says_what_to_do(cli: dict[str, Any]) -> None:
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output


def test_search_reports_ollama_being_down_in_one_line(cli: dict[str, Any]) -> None:
    cli["error"] = OllamaError("Ollama GET /api/tags failed: connection refused")
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 1
    assert "connection refused" in result.output
    assert "Traceback" not in result.output


def test_search_needs_a_question() -> None:
    assert runner.invoke(app, ["search"]).exit_code == 2


def test_search_top_k_must_be_positive(cli: dict[str, Any]) -> None:
    assert runner.invoke(app, ["search", "x", "--top-k", "0"]).exit_code == 2
