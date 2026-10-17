"""Tests for step S1.6: the ingest pipeline, the repair and the workspace (no model, no network).

The embedder is a deterministic fake; SQLite and Chroma are real, in ``tmp_path``. Layout: finding
files, then a first ingest and a second one (Gate A: "re-ingesting unchanged data costs 0
embeddings"), then edits, then failures, then crashes and ``doctor --repair``, then the workspace,
and a hypothesis property over random edits at the end.
"""

import hashlib
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.config import Settings
from geekbot.index.consistency import check_consistency, repair
from geekbot.index.dense import DenseEntry, FingerprintMismatchError
from geekbot.index.embed_cache import CachedEmbedder, EmbeddingCache
from geekbot.index.embedder import EmbedderSpec
from geekbot.index.sqlite import Store
from geekbot.index.workspace import Workspace
from geekbot.ingest.chunking import chunk_document
from geekbot.ingest.ids import doc_id_of
from geekbot.ingest.jobs import ParsedFile, parse_files
from geekbot.ingest.loaders import LoadedDocument, Segment
from geekbot.ingest.pipeline import (
    FileOutcome,
    Ingestor,
    collect_files,
    format_outcome,
    pipeline_version,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

DIM = 8
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=DIM,
    with_header=True,
)
CHUNK_TOKENS = 40  # small, so a few paragraphs make several chunks
OVERLAP_TOKENS = 5


class Crash(BaseException):
    """A stand-in for the process dying: it is not an ``Exception``, so nothing catches it."""


class FakeEmbeddings:
    """Deterministic vectors, a record of every text embedded, and switchable failures."""

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.explode_on: str | None = None  # raise RuntimeError for a text containing this
        self.crash_on: str | None = None  # raise Crash for a text containing this

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if self.crash_on and self.crash_on in text:
                raise Crash
            if self.explode_on and self.explode_on in text:
                raise RuntimeError("embedder exploded")
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:DIM]]


def prose(topic: str, paragraphs: int = 6) -> str:
    """Several short paragraphs about one topic: with the small budget, about one chunk each."""
    filler = "lorem ipsum dolor sit amet consectetur " * 3
    return "\n\n".join(f"{topic} paragraph {i} {filler}" for i in range(paragraphs))


def make_settings(root: Path) -> Settings:
    return Settings(_env_file=None, data_dir=root / "data")


@pytest.fixture
def fake() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def ws(tmp_path: Path, fake: FakeEmbeddings) -> Iterator[Workspace]:
    with Workspace.open(make_settings(tmp_path), embeddings=fake, spec=SPEC) as opened:
        yield opened


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "zebra.md").write_text(prose("zebra"), encoding="utf-8")
    (folder / "quartz.txt").write_text(prose("quartz"), encoding="utf-8")
    return folder


def ingestor(ws: Workspace, *, workers: int = 1, chunk_tokens: int = CHUNK_TOKENS) -> Ingestor:
    return Ingestor(
        ws.store,
        ws.dense,
        ws.embedder,
        chunk_tokens=chunk_tokens,
        overlap_tokens=OVERLAP_TOKENS,
        workers=workers,
    )


def ingest_folder(ws: Workspace, folder: Path, **kwargs: int) -> list[FileOutcome]:
    files, _ = collect_files([folder])
    return list(ingestor(ws, **kwargs).ingest(files).outcomes)


def assert_in_step(ws: Workspace) -> None:
    """The invariant of the whole step: SQLite's chunks and Chroma's vectors are the same set."""
    owned = set().union(*ws.store.chunk_ids_by_document().values())
    assert ws.dense.ids() == owned


# --- finding the files -----------------------------------------------------------------------


def test_collect_files_walks_folders_and_skips_what_it_should(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    for name in (
        "a.md",
        "sub/b.txt",
        "sub/deeper/c.py",
        "sub/notes.xyz",
        ".git/hidden.md",
        "venv/lib.md",
        "node_modules/pkg.md",
        "__pycache__/x.txt",
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    files, unsupported = collect_files([root])
    assert [p.relative_to(root).as_posix() for p in files] == [
        "a.md",
        "sub/b.txt",
        "sub/deeper/c.py",
    ]
    assert [p.name for p in unsupported] == ["notes.xyz"]


def test_collect_files_keeps_an_explicit_file_and_never_repeats(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "a.md").write_text("x", encoding="utf-8")
    (folder / "b.xyz").write_text("x", encoding="utf-8")
    files, unsupported = collect_files([folder / "b.xyz", folder, folder / "a.md", folder])
    assert [p.name for p in files] == ["b.xyz", "a.md"]  # an explicit file is kept as given
    assert unsupported == []  # (b.xyz was already taken as a file, so it is not "left out")


def test_collect_files_returns_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    files, _ = collect_files([Path("a.md")])
    assert files[0].is_absolute()
    assert files[0].samefile(tmp_path / "a.md")


def test_a_missing_path_is_an_error_not_an_empty_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope"):
        collect_files([tmp_path / "nope"])


# --- the first ingest and the second ---------------------------------------------------------


def test_a_first_ingest_makes_documents_ready_searchable_and_embedded(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    outcomes = ingest_folder(ws, docs)
    assert [o.status for o in outcomes] == ["ingested", "ingested"]
    assert all(o.chunks > 1 for o in outcomes)
    assert all(d.state == "ready" for d in ws.store.list_documents())
    assert [h.chunk.title for h in ws.store.search_bm25("zebra", 1)] == ["zebra.md"]
    assert [h.chunk.title for h in ws.store.search_bm25("quartz", 1)] == ["quartz.txt"]
    total = sum(o.chunks for o in outcomes)
    assert ws.dense.count() == total
    assert len(fake.embedded) == total  # every chunk embedded once
    assert_in_step(ws)


def test_the_embedded_text_has_the_header_in_front(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    ingest_folder(ws, docs)
    assert all(text.startswith(("zebra.md", "quartz.txt")) for text in fake.embedded)
    assert all("\n\n" in text for text in fake.embedded)


def test_a_second_ingest_of_unchanged_files_embeds_nothing(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    first = ingest_folder(ws, docs)
    calls = len(fake.embedded)
    count = ws.dense.count()

    second = ingest_folder(ws, docs)
    assert [o.status for o in second] == ["unchanged", "unchanged"]
    assert [o.chunks for o in second] == [o.chunks for o in first]
    assert sum(o.embedded for o in second) == 0
    assert len(fake.embedded) == calls  # the embedder was not even asked
    assert ws.dense.count() == count


def test_an_unchanged_file_costs_no_chunk_text_reads_and_still_reports_its_chunk_count(
    ws: Workspace, docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ingest_folder(ws, docs)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an unchanged file must not load its chunks just to count them")

    monkeypatch.setattr(ws.store, "document_chunks", forbidden)
    second = ingest_folder(ws, docs)
    assert [o.status for o in second] == ["unchanged", "unchanged"]
    assert [o.chunks for o in second] == [o.chunks for o in first]


def test_the_report_adds_up(ws: Workspace, docs: Path) -> None:
    files, _ = collect_files([docs])
    report = ingestor(ws).ingest(files)
    assert report.count("ingested") == 2
    assert report.embedded == sum(o.chunks for o in report.outcomes)
    assert report.cache_hits == 0
    assert not report.has_failures
    assert report.summary().startswith("2 ingested, 0 unchanged, 0 failed, 0 skipped;")
    again = ingestor(ws).ingest(files)
    assert again.summary() == (
        "0 ingested, 2 unchanged, 0 failed, 0 skipped; 0 embeddings computed, 0 from the cache"
    )


def test_progress_is_reported_per_file_in_order(ws: Workspace, docs: Path) -> None:
    files, _ = collect_files([docs])
    seen: list[str] = []
    ingestor(ws).ingest(files, on_file=lambda o: seen.append(o.path.name))
    assert seen == ["quartz.txt", "zebra.md"]


def test_a_changed_pipeline_version_reprocesses_unchanged_files(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    assert pipeline_version(CHUNK_TOKENS, OVERLAP_TOKENS) != pipeline_version(
        CHUNK_TOKENS * 2, OVERLAP_TOKENS
    )
    files, _ = collect_files([docs])
    report = ingestor(ws, chunk_tokens=CHUNK_TOKENS * 2).ingest(files)
    assert report.count("ingested") == 2  # same bytes, different chunking: not skipped
    assert_in_step(ws)


# --- edits -----------------------------------------------------------------------------------


def test_an_appended_paragraph_embeds_only_what_is_new(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    target = docs / "zebra.md"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\nzebra extra stripes " * 3, encoding="utf-8"
    )

    outcomes = {o.path.name: o for o in ingest_folder(ws, docs)}
    assert outcomes["quartz.txt"].status == "unchanged"
    changed = outcomes["zebra.md"]
    assert changed.status == "ingested"
    assert 0 < changed.embedded < changed.chunks  # most chunks came from the cache
    assert changed.cache_hits == changed.chunks - changed.embedded
    assert_in_step(ws)


def test_a_removed_paragraph_removes_its_vector(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    before = ws.dense.count()
    target = docs / "zebra.md"
    paragraphs = target.read_text(encoding="utf-8").split("\n\n")
    target.write_text("\n\n".join(paragraphs[:2] + paragraphs[3:]), encoding="utf-8")

    changed = next(o for o in ingest_folder(ws, docs) if o.path.name == "zebra.md")
    assert changed.status == "ingested"
    assert changed.removed_vectors >= 1
    assert ws.dense.count() < before
    assert_in_step(ws)


def test_an_edited_document_is_invisible_to_search_until_it_is_ready_again(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    ingest_folder(ws, docs)
    target = docs / "zebra.md"
    target.write_text(prose("zebra", 7), encoding="utf-8")
    fake.explode_on = "paragraph 6"  # the new paragraph cannot be embedded
    outcome = next(o for o in ingest_folder(ws, docs) if o.path.name == "zebra.md")
    assert outcome.status == "failed"
    assert ws.store.search_bm25("zebra") == []  # the old text is not served as if current
    assert ws.store.search_bm25("quartz") != []  # others are untouched


def test_remove_forgets_a_file_in_both_stores(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    target = docs / "zebra.md"
    assert ingestor(ws).remove(target) is True
    assert ws.store.get_document(doc_id_of(target)) is None
    assert ws.dense.ids_of_document(doc_id_of(target)) == set()
    assert ws.store.search_bm25("zebra") == []
    assert ws.store.search_bm25("quartz") != []
    assert ingestor(ws).remove(target) is False
    assert_in_step(ws)


# --- failures --------------------------------------------------------------------------------


def test_a_bad_file_is_reported_and_the_rest_still_ingest(ws: Workspace, docs: Path) -> None:
    (docs / "broken.pdf").write_bytes(b"this is not a pdf")
    outcomes = {o.path.name: o for o in ingest_folder(ws, docs)}
    assert outcomes["broken.pdf"].status == "failed"
    assert outcomes["broken.pdf"].error
    assert outcomes["zebra.md"].status == "ingested"
    assert outcomes["quartz.txt"].status == "ingested"
    assert ws.store.get_document(doc_id_of(docs / "broken.pdf")) is None  # nothing to hide


def test_an_unsupported_file_is_skipped_not_failed(ws: Workspace, tmp_path: Path) -> None:
    odd = tmp_path / "data.xyz"
    odd.write_text("x", encoding="utf-8")
    report = ingestor(ws).ingest([odd])
    assert report.count("skipped") == 1
    assert not report.has_failures


def test_a_file_that_goes_missing_is_a_failure(ws: Workspace, tmp_path: Path) -> None:
    report = ingestor(ws).ingest([tmp_path / "ghost.md"])
    assert report.has_failures
    assert "could not be read" in (report.outcomes[0].error or "")


def test_a_known_file_that_stops_loading_is_hidden_not_served_stale(
    ws: Workspace, docs: Path
) -> None:
    ingest_folder(ws, docs)
    (docs / "zebra.md").write_text("", encoding="utf-8")  # an empty file is a load error
    outcome = next(o for o in ingest_folder(ws, docs) if o.path.name == "zebra.md")
    assert outcome.status == "failed"
    document = ws.store.get_document(doc_id_of(docs / "zebra.md"))
    assert document is not None
    assert document.state == "failed"
    assert document.error
    assert ws.store.search_bm25("zebra") == []


def test_an_embedder_failure_fails_one_document_and_the_next_run_recovers(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    fake.explode_on = "zebra"
    outcomes = {o.path.name: o for o in ingest_folder(ws, docs)}
    assert outcomes["zebra.md"].status == "failed"
    assert "embedder exploded" in (outcomes["zebra.md"].error or "")
    assert outcomes["quartz.txt"].status == "ingested"
    document = ws.store.get_document(doc_id_of(docs / "zebra.md"))
    assert document is not None
    assert (document.state, "RuntimeError" in (document.error or "")) == ("failed", True)

    fake.explode_on = None
    recovered = {o.path.name: o for o in ingest_folder(ws, docs)}
    assert recovered["zebra.md"].status == "ingested"  # a failed document is never skipped
    assert recovered["quartz.txt"].status == "unchanged"
    assert ws.store.search_bm25("zebra", 1) != []
    assert_in_step(ws)


def test_a_text_that_cannot_be_hashed_is_a_failure_not_a_crash(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A lone surrogate (S1.4's known gap) hashes fine but cannot be written to SQLite. The loaders
    # may never produce one, so the parsed file is handed to the pipeline directly. The gap closes
    # here: the file is reported with the reason, and its document is recorded as failed.
    odd = tmp_path / "odd.txt"
    odd.write_text("placeholder", encoding="utf-8")
    text = "fine text " + chr(0xD800) + " more text"
    document = LoadedDocument(odd, "text", "odd.txt", (Segment(text),))
    parsed = ParsedFile(odd, document, tuple(chunk_document(document)))
    monkeypatch.setattr("geekbot.ingest.pipeline.parse_files", lambda *_a, **_k: iter([parsed]))

    report = ingestor(ws).ingest([odd])
    assert report.has_failures
    assert "UnicodeEncodeError" in (report.outcomes[0].error or "")
    document = ws.store.get_document(doc_id_of(odd))
    assert document is not None
    assert document.state == "failed"
    assert "surrogates not allowed" in (document.error or "")
    assert ws.store.document_chunks(document.doc_id) == []  # the insert rolled back
    assert ws.store.search_bm25("fine") == []


# --- crashes and doctor --repair -------------------------------------------------------------


def test_a_crash_while_embedding_is_finished_by_the_repair(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    files, _ = collect_files([docs])
    fake.crash_on = "zebra"  # the process "dies" while embedding zebra.md (after quartz.txt)
    with pytest.raises(Crash):
        ingestor(ws).ingest(files)
    zebra = ws.store.get_document(doc_id_of(docs / "zebra.md"))
    assert zebra is not None
    assert zebra.state == "embedding"  # exactly where it had got to
    assert ws.store.search_bm25("zebra") == []  # and nobody can retrieve it

    report = check_consistency(ws.store, ws.dense)
    assert report.unfinished == (zebra.doc_id,)
    assert not report.ok

    fake.crash_on = None
    repaired = repair(ws.store, ws.dense, ws.embedder)
    assert repaired.finished == (zebra.doc_id,)
    assert repaired.embedded > 0
    assert ws.store.search_bm25("zebra", 1) != []
    assert check_consistency(ws.store, ws.dense).ok
    assert_in_step(ws)


def test_a_crash_before_the_rows_are_written_is_marked_failed_by_the_repair(
    ws: Workspace, docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def die(*_args: object, **_kwargs: object) -> None:
        raise Crash

    files, _ = collect_files([docs / "zebra.md"])
    monkeypatch.setattr(ws.store, "sync_chunks", die)
    with pytest.raises(Crash):
        ingestor(ws).ingest(files)
    monkeypatch.undo()
    doc_id = doc_id_of(docs / "zebra.md")
    stuck = ws.store.get_document(doc_id)
    assert stuck is not None
    assert stuck.state == "pending"

    repaired = repair(ws.store, ws.dense, ws.embedder)
    assert repaired.marked_failed == (doc_id,)
    failed = ws.store.get_document(doc_id)
    assert failed is not None
    assert failed.state == "failed"
    assert "ingest it again" in (failed.error or "")

    again = next(iter(ingestor(ws).ingest(files).outcomes))
    assert again.status == "ingested"  # not skipped as "unchanged": it never was ready
    assert ws.store.search_bm25("zebra", 1) != []


def test_wiped_vectors_are_rebuilt_from_the_cache_without_embedding(
    ws: Workspace, docs: Path, fake: FakeEmbeddings
) -> None:
    ingest_folder(ws, docs)
    calls = len(fake.embedded)
    wiped = ws.dense.ids()
    ws.dense.delete(sorted(wiped))  # Chroma's folder was lost, say

    report = check_consistency(ws.store, ws.dense)
    assert len(report.missing_vectors) == 2
    assert ws.store.search_bm25("zebra", 1) != []  # BM25 still works meanwhile

    repaired = repair(ws.store, ws.dense, ws.embedder)
    assert len(repaired.re_embedded) == 2
    assert repaired.embedded == 0  # the cache had every vector
    assert repaired.cache_hits == len(wiped)
    assert len(fake.embedded) == calls
    assert ws.dense.ids() == wiped
    assert all(d.state == "ready" for d in ws.store.list_documents())
    assert check_consistency(ws.store, ws.dense).ok


def test_orphan_vectors_are_removed(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    ws.dense.upsert([DenseEntry("ghost", "no-such-doc", [0.1] * DIM)])
    assert check_consistency(ws.store, ws.dense).orphan_vectors == ("ghost",)
    repaired = repair(ws.store, ws.dense, ws.embedder)
    assert repaired.orphans_removed == 1
    assert "ghost" not in ws.dense.ids()
    assert_in_step(ws)


def test_repair_twice_finds_nothing_the_second_time(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    ws.dense.delete(sorted(ws.dense.ids()))
    repair(ws.store, ws.dense, ws.embedder)
    second = repair(ws.store, ws.dense, ws.embedder)
    assert second.summary().startswith("0 unfinished documents finished, 0 rebuilt, 0 interrupted")
    assert second.embedded == second.cache_hits == second.orphans_removed == 0


def test_a_repair_that_cannot_embed_fails_the_document_and_goes_on(
    ws: Workspace, docs: Path, fake: FakeEmbeddings, tmp_path: Path
) -> None:
    ingest_folder(ws, docs)
    ws.dense.delete(sorted(ws.dense.ids()))
    fake.explode_on = "zebra"
    with EmbeddingCache(tmp_path / "empty-cache.db") as empty:  # nothing cached: every text misses
        repaired = repair(ws.store, ws.dense, CachedEmbedder(fake, empty, SPEC))
    assert len(repaired.failed) == 1
    assert len(repaired.re_embedded) == 1
    states = {Path(d.path).name: d.state for d in ws.store.list_documents()}
    assert states == {"zebra.md": "failed", "quartz.txt": "ready"}


def test_the_repair_can_rebuild_the_full_text_index(ws: Workspace, docs: Path) -> None:
    ingest_folder(ws, docs)
    repaired = repair(ws.store, ws.dense, ws.embedder, rebuild_fts=True)
    assert repaired.orphans_removed == 0
    assert ws.store.search_bm25("zebra", 1) != []


def test_a_fresh_store_is_consistent(ws: Workspace) -> None:
    assert check_consistency(ws.store, ws.dense).ok
    assert check_consistency(ws.store, ws.dense).summary() == "stores are consistent"


# --- re-index --------------------------------------------------------------------------------


def test_a_new_embedder_is_refused_until_a_reindex_rebuilds_every_vector(
    tmp_path: Path, docs: Path
) -> None:
    settings = make_settings(tmp_path)
    first = FakeEmbeddings()
    with Workspace.open(settings, embeddings=first, spec=SPEC) as ws:
        ingest_folder(ws, docs)
        chunks = ws.dense.count()

    other = EmbedderSpec(
        model="fake",
        digest="d2",  # re-pulled
        document_prefix="doc: ",
        query_prefix="q: ",
        dimension=DIM,
        with_header=True,
    )
    second = FakeEmbeddings()
    with pytest.raises(FingerprintMismatchError):
        Workspace.open(settings, embeddings=second, spec=other)

    with Workspace.open(settings, reset_dense=True, embeddings=second, spec=other) as ws:
        assert ws.dense.count() == 0
        repaired = repair(ws.store, ws.dense, ws.embedder)
        assert len(repaired.re_embedded) == 2
        assert repaired.embedded == chunks  # new fingerprint: the old cache entries do not apply
        assert ws.dense.count() == chunks
        assert_in_step(ws)


# --- the workspace ---------------------------------------------------------------------------


def test_the_workspace_releases_the_lock_when_it_closes(
    tmp_path: Path, fake: FakeEmbeddings
) -> None:
    settings = make_settings(tmp_path)
    Workspace.open(settings, embeddings=fake, spec=SPEC).close()
    with Workspace.open(settings, embeddings=fake, spec=SPEC):
        pass


def test_a_failed_open_releases_everything_it_took(tmp_path: Path, fake: FakeEmbeddings) -> None:
    settings = make_settings(tmp_path)
    with Workspace.open(settings, embeddings=fake, spec=SPEC):
        pass
    changed = EmbedderSpec(
        model="x", digest="x", document_prefix="", query_prefix="", dimension=DIM, with_header=False
    )
    with pytest.raises(FingerprintMismatchError):
        Workspace.open(settings, embeddings=fake, spec=changed)
    with Workspace.open(settings, embeddings=fake, spec=SPEC):  # the lock was released
        pass


def test_the_workspace_wants_both_or_neither_of_embeddings_and_spec(
    tmp_path: Path, fake: FakeEmbeddings
) -> None:
    with pytest.raises(ValueError, match="both"):
        Workspace.open(make_settings(tmp_path), embeddings=fake)


def test_a_second_workspace_on_the_same_folder_is_refused(
    tmp_path: Path, fake: FakeEmbeddings
) -> None:
    from geekbot.index.lock import StorageLockedError

    settings = make_settings(tmp_path)
    with Workspace.open(settings, embeddings=fake, spec=SPEC), pytest.raises(StorageLockedError):
        Workspace.open(settings, embeddings=fake, spec=SPEC)


# --- parallel parsing ------------------------------------------------------------------------


def test_parsing_in_a_process_pool_gives_the_same_result_in_the_same_order(docs: Path) -> None:
    files, _ = collect_files([docs])
    serial = list(
        parse_files(files, workers=1, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    )
    pooled = list(
        parse_files(files, workers=2, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    )
    assert [p.path for p in pooled] == files
    assert pooled == serial


def test_parse_files_rejects_zero_workers(docs: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        list(parse_files([docs / "zebra.md"], workers=0))


def test_an_ingest_with_workers_matches_one_without(tmp_path: Path, docs: Path) -> None:
    results = []
    for workers, name in ((1, "one"), (2, "two")):
        with Workspace.open(
            make_settings(tmp_path / name), embeddings=FakeEmbeddings(), spec=SPEC
        ) as ws:
            outcomes = ingest_folder(ws, docs, workers=workers)
            results.append([(o.path.name, o.status, o.chunks, o.embedded) for o in outcomes])
            assert_in_step(ws)
    assert results[0] == results[1]


def test_workers_must_be_positive(ws: Workspace) -> None:
    with pytest.raises(ValueError, match="workers"):
        Ingestor(ws.store, ws.dense, ws.embedder, workers=0)


# --- formatting ------------------------------------------------------------------------------


def test_outcomes_format_to_one_line_each() -> None:
    path = Path("a.md")
    assert format_outcome(FileOutcome(path, "ingested", chunks=3, embedded=2, cache_hits=1)) == (
        "ingested   a.md (3 chunks; 2 embedded, 1 cached)"
    )
    assert format_outcome(FileOutcome(path, "unchanged", chunks=3)) == "unchanged  a.md (3 chunks)"
    assert format_outcome(FileOutcome(path, "failed", error="boom")) == "FAILED     a.md: boom"
    assert (
        format_outcome(FileOutcome(path, "skipped", error="no loader"))
        == "skipped    a.md: no loader"
    )


# --- a property ------------------------------------------------------------------------------

_BODIES = st.lists(
    st.sampled_from(["alpha beta gamma", "delta epsilon", "zeta eta theta iota", "kappa lambda"]),
    min_size=1,
    max_size=5,
)


@settings(max_examples=12, deadline=None)
@given(first=_BODIES, second=_BODIES)
def test_after_any_edit_the_two_stores_hold_the_same_chunks(
    first: list[str], second: list[str]
) -> None:
    # Whatever a file is edited into, once the ingest is done SQLite and Chroma own the same
    # chunk IDs, and ingesting again without a change embeds nothing.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        root = Path(folder)
        target = root / "docs" / "note.md"
        target.parent.mkdir()
        with Workspace.open(make_settings(root), embeddings=FakeEmbeddings(), spec=SPEC) as ws:
            for body in (first, second):
                target.write_text("\n\n".join(body), encoding="utf-8")
                ingest_folder(ws, target.parent)
                assert_in_step(ws)
                assert next(iter(ingest_folder(ws, target.parent))).embedded == 0


def test_a_store_alone_holds_no_vectors(tmp_path: Path) -> None:
    # Sanity for the helper above: a bare store has chunk IDs by document and nothing else.
    with Store(tmp_path) as store:
        assert store.chunk_ids_by_document() == {}
