"""Tests for step S1.4: the SQLite store, its migrations and its BM25 search (PLAN.md §6.2-§6.4).

Layout: schema and migrations first, then documents, then ``sync_chunks`` (re-ingest as a set
difference), then search (this step's ✍️ piece: ``fts_match_query`` and ``search_bm25``, written by
Claude in prototyping mode, D-047), then threads and shutdown. Two hypothesis properties close the
file: any text makes a query FTS5 accepts, and after any two syncs the search results equal a
simple model of what is stored.

Every store lives in ``tmp_path``. The storage lock (``test_storage_lock.py``) means two stores
cannot share a directory, which is also why no test shares one.
"""

import sqlite3
import threading
import unicodedata
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.domain import Document, NewDocument
from geekbot.index.sqlite import (
    DB_FILE_NAME,
    MAX_QUERY_TERMS,
    MIGRATIONS,
    ChunkSync,
    Migration,
    Store,
    StoreClosedError,
    StoreError,
    UnknownDocumentError,
    fts_match_query,
    migrate,
)
from geekbot.ingest.chunking import ChunkDraft
from geekbot.ingest.ids import IdentifiedChunk, doc_id_of, identify_chunks

PATH = "/data/notes.md"
DOC_ID = doc_id_of(PATH)
FILE_HASH = "f" * 64
# Built with chr() so the file stays plain ASCII and no tool can turn an escape into a glyph.
E_ACUTE = chr(0xE9)
COMBINING_ACUTE = chr(0x301)

CORPUS = [
    "The refund policy allows returns within thirty days of purchase.",
    "Shipping takes five business days for domestic orders.",
    "Our refund team reviews every refund request personally.",
    "Warranty claims require the original receipt.",
    "Passwords must be rotated every ninety days.",
    "The cafeteria opens at eight in the morning.",
    "Visitors sign in at the front desk.",
    "Expense reports are due on the last Friday.",
]


# --- helpers ---------------------------------------------------------------------------------


def new_document(path: str = PATH, title: str = "notes.md") -> NewDocument:
    return NewDocument(
        doc_id=doc_id_of(path),
        path=path,
        title=title,
        kind="markdown",
        file_hash=FILE_HASH,
        pipeline_version="p1",
    )


def add_document(store: Store, path: str = PATH, *, ready: bool = True) -> str:
    document = new_document(path)
    store.upsert_document(document)
    if ready:
        store.set_document_state(document.doc_id, "ready")
    return document.doc_id


def identified(
    texts: Sequence[str],
    path: str = PATH,
    *,
    page: int | None = None,
    header: str = "notes.md",
    section: tuple[str, ...] = ("Intro",),
) -> list[IdentifiedChunk]:
    """Chunks with made-up but consistent positions: each one starts after the one before."""
    drafts: list[ChunkDraft] = []
    position = 0
    for ordinal, text in enumerate(texts):
        drafts.append(
            ChunkDraft(
                text=text,
                char_start=position,
                char_end=position + len(text),
                page=page,
                section_path=section,
                header=header,
                ordinal=ordinal,
            )
        )
        position += len(text) + 2
    return identify_chunks(path, drafts)


def texts_of(store: Store, query: str, limit: int = 10) -> list[str]:
    return [hit.chunk.text for hit in store.search_bm25(query, limit)]


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path) as opened:
        yield opened


@pytest.fixture
def corpus(store: Store) -> list[IdentifiedChunk]:
    """One ready document holding the eight sentences above."""
    add_document(store)
    chunks = identified(CORPUS)
    store.sync_chunks(DOC_ID, chunks)
    return chunks


# --- schema and migrations -------------------------------------------------------------------


def test_a_new_store_is_at_the_latest_schema_version(store: Store) -> None:
    version = store.read(lambda conn: conn.execute("PRAGMA user_version").fetchone()[0])
    assert version == len(MIGRATIONS)


def test_the_database_uses_wal_and_enforces_foreign_keys(store: Store) -> None:
    mode = store.read(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0])
    foreign_keys = store.read(lambda conn: conn.execute("PRAGMA foreign_keys").fetchone()[0])
    assert mode == "wal"
    assert foreign_keys == 1


def test_migration_one_creates_the_tables_and_the_full_text_index(store: Store) -> None:
    names = store.read(
        lambda conn: {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    )
    assert {"documents", "chunks", "chunks_fts"} <= names


def test_a_pooled_reader_cannot_write(store: Store) -> None:
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store.read(lambda conn: conn.execute("DELETE FROM documents"))


def test_a_chunk_cannot_exist_without_its_document(store: Store) -> None:
    def work(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, content_hash, occurrence, ordinal, char_start, "
            "char_end, page, section_path, header, text) "
            "VALUES (?, ?, ?, 0, 0, 0, 1, NULL, '[]', 'h', 't')",
            ("a" * 24, "b" * 16, "c" * 64),
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.write(work)


def raw_connection() -> sqlite3.Connection:
    return sqlite3.connect(":memory:", isolation_level=None)


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_migrate_applies_each_migration_exactly_once() -> None:
    conn = raw_connection()
    first = Migration(1, "table a", "CREATE TABLE a (x INTEGER);")
    second = Migration(2, "table b", "CREATE TABLE b (x INTEGER);")
    assert migrate(conn, (first,)) == 1
    assert table_names(conn) == {"a"}
    assert migrate(conn, (first, second)) == 2
    assert migrate(conn, (first, second)) == 2  # nothing left to do, and no "already exists"
    assert table_names(conn) == {"a", "b"}
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_a_failing_migration_leaves_the_database_as_it_was() -> None:
    conn = raw_connection()
    good = Migration(1, "good", "CREATE TABLE good (x INTEGER);")
    bad = Migration(2, "bad", "CREATE TABLE half (x INTEGER); CREATE TABLE half (x INTEGER);")
    migrate(conn, (good,))
    with pytest.raises(StoreError, match="migration 2"):
        migrate(conn, (good, bad))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "half" not in table_names(conn)
    assert not conn.in_transaction


def test_migrations_must_be_numbered_from_one_without_gaps() -> None:
    with pytest.raises(ValueError, match="numbered"):
        migrate(raw_connection(), (Migration(2, "starts at two", "SELECT 1;"),))


def test_a_database_from_a_newer_geekbot_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path):
        pass
    future = sqlite3.connect(tmp_path / DB_FILE_NAME)
    future.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    future.close()
    with pytest.raises(StoreError, match="newer"):
        Store(tmp_path)


def test_a_later_migration_is_applied_to_an_existing_database(tmp_path: Path) -> None:
    with Store(tmp_path) as first:
        add_document(first)
    extra = Migration(
        len(MIGRATIONS) + 1, "a note column", "ALTER TABLE documents ADD COLUMN n TEXT;"
    )
    with Store(tmp_path, migrations=(*MIGRATIONS, extra)) as second:
        assert second.get_document(DOC_ID) is not None  # the data survived the upgrade
        columns = second.read(
            lambda conn: [row[1] for row in conn.execute("PRAGMA table_info(documents)")]
        )
    assert "n" in columns


# --- documents -------------------------------------------------------------------------------


def test_upsert_stores_a_pending_document(tmp_path: Path) -> None:
    with Store(tmp_path, clock=FakeClock()) as store:
        store.upsert_document(new_document())
        stamp = "2026-10-16T12:00:00+00:00"
        assert store.get_document(DOC_ID) == Document(
            doc_id=DOC_ID,
            path=PATH,
            title="notes.md",
            kind="markdown",
            file_hash=FILE_HASH,
            pipeline_version="p1",
            state="pending",
            error=None,
            created_at=stamp,
            updated_at=stamp,
        )


def test_upserting_again_restarts_the_outbox_and_keeps_the_creation_time(tmp_path: Path) -> None:
    clock = FakeClock()
    with Store(tmp_path, clock=clock) as store:
        store.upsert_document(new_document())
        clock.advance(60)
        store.set_document_state(DOC_ID, "failed", error="embedder down")
        failed = store.get_document(DOC_ID)
        assert failed is not None
        assert (failed.state, failed.error) == ("failed", "embedder down")
        assert failed.updated_at == "2026-10-16T12:01:00+00:00"

        clock.advance(60)
        store.upsert_document(replace(new_document(), title="renamed", file_hash="e" * 64))
        again = store.get_document(DOC_ID)
        assert again is not None
        assert (again.state, again.error) == ("pending", None)
        assert (again.title, again.file_hash) == ("renamed", "e" * 64)
        assert again.created_at == "2026-10-16T12:00:00+00:00"
        assert again.updated_at == "2026-10-16T12:02:00+00:00"


def test_leaving_the_failed_state_clears_the_error(store: Store) -> None:
    add_document(store, ready=False)
    store.set_document_state(DOC_ID, "failed", error="boom")
    store.set_document_state(DOC_ID, "embedding")
    document = store.get_document(DOC_ID)
    assert document is not None
    assert (document.state, document.error) == ("embedding", None)


def test_set_document_state_rejects_bad_input(store: Store) -> None:
    add_document(store)
    with pytest.raises(ValueError, match="unknown document state"):
        store.set_document_state(DOC_ID, "done")
    with pytest.raises(ValueError, match="only goes with"):
        store.set_document_state(DOC_ID, "ready", error="not an error")
    with pytest.raises(UnknownDocumentError):
        store.set_document_state("0" * 16, "ready")


def test_a_document_id_of_the_wrong_length_is_refused(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_document(replace(new_document(), doc_id="short"))


def test_an_unknown_document_is_none(store: Store) -> None:
    assert store.get_document("0" * 16) is None


def test_deleting_a_document_removes_its_chunks_and_search_hits(
    store: Store, corpus: list[IdentifiedChunk]
) -> None:
    assert store.delete_document(DOC_ID) is True
    assert store.get_document(DOC_ID) is None
    assert store.get_chunks([chunk.chunk_id for chunk in corpus], ready_only=False) == []
    assert store.search_bm25("refund") == []
    assert store.delete_document(DOC_ID) is False


# --- sync_chunks: re-ingest as a set difference ----------------------------------------------


def test_the_first_sync_adds_every_chunk(store: Store) -> None:
    doc_id = add_document(store)
    chunks = identified(["alpha one", "bravo two", "charlie three"], page=3)
    ids = tuple(chunk.chunk_id for chunk in chunks)
    result = store.sync_chunks(doc_id, chunks)
    assert result == ChunkSync(added=ids, removed=(), updated=(), unchanged=())

    stored = store.get_chunks(ids)
    assert [chunk.text for chunk in stored] == ["alpha one", "bravo two", "charlie three"]
    first = stored[0]
    assert (first.doc_id, first.path, first.title) == (doc_id, PATH, "notes.md")
    assert (first.ordinal, first.char_start, first.char_end, first.page) == (0, 0, 9, 3)
    assert first.section_path == ("Intro",)
    assert first.header == "notes.md"
    assert first.content_hash == chunks[0].content_hash
    assert first.occurrence == 0


def test_syncing_the_same_chunks_again_changes_nothing(store: Store) -> None:
    doc_id = add_document(store)
    chunks = identified(["alpha one", "bravo two"])
    ids = tuple(chunk.chunk_id for chunk in chunks)
    store.sync_chunks(doc_id, chunks)
    assert store.sync_chunks(doc_id, chunks) == ChunkSync(
        added=(), removed=(), updated=(), unchanged=ids
    )


def test_an_inserted_chunk_adds_one_id_and_moves_the_others_in_place(store: Store) -> None:
    doc_id = add_document(store)
    old = identified(["alpha one", "bravo two", "charlie three"])
    old_ids = tuple(chunk.chunk_id for chunk in old)
    store.sync_chunks(doc_id, old)

    new = identified(["zulu zero", "alpha one", "bravo two", "charlie three"])
    result = store.sync_chunks(doc_id, new)

    assert result.added == (new[0].chunk_id,)
    assert result.removed == ()
    assert result.updated == old_ids  # same IDs, new ordinal and offsets
    assert result.unchanged == ()
    assert [chunk.ordinal for chunk in store.get_chunks(old_ids)] == [1, 2, 3]
    for word in ("alpha", "bravo", "charlie", "zulu"):
        assert len(store.search_bm25(word)) == 1  # a move leaves the full-text index alone


def test_a_removed_chunk_removes_exactly_its_id(store: Store) -> None:
    doc_id = add_document(store)
    old = identified(["alpha one", "bravo two", "charlie three"])
    store.sync_chunks(doc_id, old)

    result = store.sync_chunks(doc_id, identified(["alpha one", "charlie three"]))

    assert result.removed == (old[1].chunk_id,)
    assert result.added == ()
    assert result.unchanged == (old[0].chunk_id,)
    assert result.updated == (old[2].chunk_id,)  # it moved up one place
    assert store.get_chunks([old[1].chunk_id]) == []
    assert store.search_bm25("bravo") == []


def test_repeated_text_is_stored_once_per_copy_and_removed_one_copy_at_a_time(
    store: Store,
) -> None:
    doc_id = add_document(store)
    chunks = identified(["footer text", "alpha one", "footer text"])
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == 3  # the rev-2 collision, D-021: two copies must not share a row
    store.sync_chunks(doc_id, chunks)
    assert len(store.get_chunks(ids)) == 3

    result = store.sync_chunks(doc_id, identified(["footer text", "alpha one"]))

    assert result.removed == (ids[2],)  # the second copy, occurrence 1
    assert result.added == ()
    assert result.unchanged == (ids[0], ids[1])


def test_a_change_the_id_ignores_still_updates_the_stored_text(store: Store) -> None:
    doc_id = add_document(store)
    first = identified(["alpha beta"])
    store.sync_chunks(doc_id, first)

    second = identified(["alpha beta  "])  # trailing spaces: same normalized text, same ID
    assert second[0].chunk_id == first[0].chunk_id
    result = store.sync_chunks(doc_id, second)

    assert result.updated == (first[0].chunk_id,)
    assert store.get_chunks([first[0].chunk_id])[0].text == "alpha beta  "
    assert len(store.search_bm25("alpha")) == 1
    assert len(store.search_bm25("beta")) == 1


def test_documents_do_not_disturb_each_other(store: Store) -> None:
    a_path, b_path = "/data/a.md", "/data/b.md"
    a_id = add_document(store, a_path)
    b_id = add_document(store, b_path)
    a_chunks = identified(["shared words", "alpha one"], a_path)
    b_chunks = identified(["shared words", "bravo two"], b_path)
    assert a_chunks[0].chunk_id != b_chunks[0].chunk_id  # same text, different document
    store.sync_chunks(a_id, a_chunks)
    store.sync_chunks(b_id, b_chunks)
    assert len(store.search_bm25("shared")) == 2

    result = store.sync_chunks(b_id, [])  # an empty list empties the document

    assert len(result.removed) == 2
    assert store.search_bm25("bravo") == []
    assert len(store.search_bm25("shared")) == 1
    assert len(store.search_bm25("alpha")) == 1


def test_sync_for_a_document_the_store_does_not_have_is_an_error(store: Store) -> None:
    with pytest.raises(UnknownDocumentError):
        store.sync_chunks(DOC_ID, identified(["alpha"]))


def test_sync_rejects_chunks_of_another_document(store: Store) -> None:
    add_document(store)
    with pytest.raises(ValueError, match="not part of document"):
        store.sync_chunks(DOC_ID, identified(["alpha"], "/data/other.md"))


def test_sync_rejects_two_chunks_with_one_id(store: Store) -> None:
    add_document(store)
    chunks = identified(["alpha"])
    with pytest.raises(ValueError, match="share a chunk_id"):
        store.sync_chunks(DOC_ID, [*chunks, *chunks])


def test_a_failing_sync_changes_nothing(store: Store) -> None:
    doc_id = add_document(store)
    old = identified(["alpha one", "bravo two"])
    store.sync_chunks(doc_id, old)
    new = identified(["alpha one", "charlie three"])
    # A negative ordinal breaks the table's CHECK (ordinal >= 0), so the insert fails last.
    broken = replace(new[1], chunk=replace(new[1].chunk, ordinal=-1))

    with pytest.raises(sqlite3.IntegrityError):
        store.sync_chunks(doc_id, [new[0], broken])

    kept = store.get_chunks([chunk.chunk_id for chunk in old])
    assert [chunk.text for chunk in kept] == ["alpha one", "bravo two"]  # bravo's delete undone
    assert store.get_chunks([new[1].chunk_id]) == []
    assert len(store.search_bm25("bravo")) == 1


def test_get_chunks_keeps_the_order_asked_for_and_skips_unknown_ids(store: Store) -> None:
    doc_id = add_document(store)
    chunks = identified(["alpha one", "bravo two", "charlie three"])
    store.sync_chunks(doc_id, chunks)
    ids = [chunk.chunk_id for chunk in chunks]

    found = store.get_chunks([ids[2], "0" * 24, ids[0]])

    assert [chunk.chunk_id for chunk in found] == [ids[2], ids[0]]
    assert store.get_chunks([]) == []


def test_get_chunks_hides_chunks_that_are_not_ready_unless_asked(store: Store) -> None:
    doc_id = add_document(store, ready=False)
    chunks = identified(["alpha one"])
    store.sync_chunks(doc_id, chunks)
    ids = [chunks[0].chunk_id]
    assert store.get_chunks(ids) == []
    assert len(store.get_chunks(ids, ready_only=False)) == 1
    store.set_document_state(doc_id, "ready")
    assert len(store.get_chunks(ids)) == 1


@pytest.mark.usefixtures("corpus")
def test_rebuild_fts_repairs_an_emptied_index(store: Store) -> None:
    def empty_the_index(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('delete-all')")

    store.write(empty_the_index)
    assert store.search_bm25("refund") == []
    store.rebuild_fts()
    assert len(store.search_bm25("refund")) == 2


# --- search: fts_match_query -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("refund policy", '"refund" OR "policy"'),
        ("  Refund,  REFUND! policy ", '"Refund" OR "policy"'),  # repeats dropped, first kept
        ('say "hello" AND (world', '"say" OR "hello" OR "AND" OR "world"'),  # AND is a word
        ("snake_case_name", '"snake" OR "case" OR "name"'),
        ("co-op", '"co" OR "op"'),
        ("2026 plan", '"2026" OR "plan"'),
        ("", None),
        ("   ", None),
        ("!!! ??? ...", None),
        ("___", None),
    ],
)
def test_fts_match_query_quotes_words_and_ors_them(text: str, expected: str | None) -> None:
    assert fts_match_query(text) == expected


def test_fts_match_query_composes_an_accent_typed_as_a_separate_mark() -> None:
    assert fts_match_query("cafe" + COMBINING_ACUTE) == '"caf' + E_ACUTE + '"'


def test_fts_match_query_keeps_only_the_first_terms() -> None:
    words = [f"word{number}" for number in range(MAX_QUERY_TERMS + 20)]
    query = fts_match_query(" ".join(words))
    assert query is not None
    assert query.count(" OR ") + 1 == MAX_QUERY_TERMS
    assert query.startswith('"word0" OR "word1"')
    assert f'"word{MAX_QUERY_TERMS - 1}"' in query
    assert f'"word{MAX_QUERY_TERMS}"' not in query


# --- search: search_bm25 ---------------------------------------------------------------------


@pytest.mark.usefixtures("corpus")
def test_search_ranks_the_chunk_with_more_of_the_word_first(store: Store) -> None:
    assert texts_of(store, "refund") == [CORPUS[2], CORPUS[0]]


@pytest.mark.usefixtures("corpus")
def test_search_scores_are_positive_and_never_increase(store: Store) -> None:
    hits = store.search_bm25("refund days policy")
    scores = [hit.score for hit in hits]
    assert len(hits) >= 3
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


@pytest.mark.usefixtures("corpus")
def test_a_rare_word_outweighs_a_common_one(store: Store) -> None:
    hits = texts_of(store, "the refund")  # "the" is in most chunks, "refund" in two
    assert set(hits[:2]) == {CORPUS[0], CORPUS[2]}


def test_search_returns_the_whole_chunk_with_its_document(
    store: Store, corpus: list[IdentifiedChunk]
) -> None:
    (hit,) = store.search_bm25("warranty")
    assert hit.chunk.text == CORPUS[3]
    assert hit.chunk.chunk_id == corpus[3].chunk_id
    assert (hit.chunk.path, hit.chunk.title, hit.chunk.ordinal) == (PATH, "notes.md", 3)


@pytest.mark.usefixtures("corpus")
def test_search_stems_words(store: Store) -> None:
    assert texts_of(store, "returning") == [CORPUS[0]]  # "returns" in the chunk
    assert texts_of(store, "shipped") == [CORPUS[1]]  # "Shipping" in the chunk


def test_search_ignores_case_and_accents(store: Store) -> None:
    add_document(store)
    cafe = "caf" + E_ACUTE
    store.sync_chunks(DOC_ID, identified([f"The {cafe} opens at eight."]))
    for query in ("cafe", cafe, "CAFE", cafe.upper()):
        assert len(store.search_bm25(query)) == 1


def test_only_the_text_is_indexed_not_the_header_or_the_section_path(store: Store) -> None:
    # The header is the embedder's business (ablation rung R5); BM25 must not see it.
    add_document(store)
    chunks = identified(["plain body words"], header="notes.md > Zebrafish", section=("Narwhal",))
    store.sync_chunks(DOC_ID, chunks)
    assert store.search_bm25("zebrafish") == []
    assert store.search_bm25("narwhal") == []
    assert len(store.search_bm25("body")) == 1


def test_only_ready_documents_are_searched(store: Store) -> None:
    doc_id = add_document(store, ready=False)
    store.sync_chunks(doc_id, identified(["a quokka sighting"]))
    assert store.search_bm25("quokka") == []  # pending

    store.set_document_state(doc_id, "embedding")
    assert store.search_bm25("quokka") == []
    store.set_document_state(doc_id, "failed", error="embedder down")
    assert store.search_bm25("quokka") == []

    store.set_document_state(doc_id, "ready")
    assert len(store.search_bm25("quokka")) == 1

    store.upsert_document(new_document())  # edited again: back to pending, invisible again
    assert store.search_bm25("quokka") == []


def test_the_limit_counts_ready_chunks_only(store: Store) -> None:
    pending_path = "/data/pending.md"
    pending = add_document(store, pending_path, ready=False)
    store.sync_chunks(pending, identified(["quokka quokka quokka"] * 3, pending_path))
    ready = add_document(store)
    store.sync_chunks(ready, identified(["one quokka among many other words here"]))

    # One hit, not zero: the three pending chunks do not use up the limit of two.
    assert len(store.search_bm25("quokka", limit=2)) == 1


def test_the_limit_cuts_the_results(store: Store, corpus: list[IdentifiedChunk]) -> None:
    assert len(store.search_bm25("days", limit=1)) == 1
    assert len(store.search_bm25("days", limit=10)) == 3
    with pytest.raises(ValueError, match="limit"):
        store.search_bm25("days", limit=0)


def test_identical_chunks_tie_and_come_back_in_chunk_id_order(store: Store) -> None:
    doc_id = add_document(store)
    chunks = identified(["same words here", "same words here"])
    store.sync_chunks(doc_id, chunks)

    hits = store.search_bm25("words")

    assert [hit.chunk.chunk_id for hit in hits] == sorted(chunk.chunk_id for chunk in chunks)
    assert hits[0].score == hits[1].score


@pytest.mark.usefixtures("corpus")
@pytest.mark.parametrize("query", ["", "   ", "!!!", "()", '""', "___", "- - -"])
def test_a_query_without_words_finds_nothing(store: Store, query: str) -> None:
    assert store.search_bm25(query) == []


@pytest.mark.parametrize(
    "query",
    [
        'refund AND (policy OR "unbalanced',
        "NOT refund",
        "NEAR(refund policy, 2)",
        "refund*",
        "^refund",
        "text:refund",
        "{text}: refund",
        "refund -policy",
        '"',
        "'",
        "(",
        "\\",
    ],
)
@pytest.mark.usefixtures("corpus")
def test_fts5_syntax_in_the_query_is_just_words(store: Store, query: str) -> None:
    hits = store.search_bm25(query)
    assert isinstance(hits, list)  # and, above all, nothing raised "fts5: syntax error"


@pytest.mark.usefixtures("corpus")
def test_operators_are_not_interpreted(store: Store) -> None:
    # As FTS5 syntax this would need both words in one chunk, which only CORPUS[0] has. As plain
    # words it is an OR, so CORPUS[2] (which has "refund" only) matches too.
    assert set(texts_of(store, "refund AND policy")) == {CORPUS[0], CORPUS[2]}


# --- threads, snapshots, shutdown ------------------------------------------------------------


def test_many_threads_can_write_and_read_at_once(store: Store) -> None:
    def ingest(index: int) -> int:
        path = f"/data/doc{index}.md"
        doc_id = add_document(store, path)
        store.sync_chunks(doc_id, identified([f"shared words and unique{index}"], path))
        return len(store.search_bm25("shared", limit=100))

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(ingest, range(16)))

    assert all(count >= 1 for count in counts)  # each thread at least sees its own chunk
    assert len(store.search_bm25("shared", limit=100)) == 16


def test_a_reader_is_not_blocked_by_a_write_in_progress(store: Store) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_write(conn: sqlite3.Connection) -> None:
        document = new_document()
        conn.execute(
            "INSERT INTO documents (doc_id, path, title, kind, file_hash, pipeline_version, "
            "state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', 'now', 'now')",
            (
                document.doc_id,
                document.path,
                document.title,
                document.kind,
                document.file_hash,
                document.pipeline_version,
            ),
        )
        started.set()
        release.wait(timeout=30)

    writer = threading.Thread(target=store.write, args=(slow_write,))
    writer.start()
    try:
        assert started.wait(timeout=30)
        # WAL: the reader sees the last commit, not the half-finished write, and does not wait.
        assert store.get_document(DOC_ID) is None
    finally:
        release.set()
        writer.join(timeout=30)
    assert store.get_document(DOC_ID) is not None


def test_a_failing_write_is_rolled_back_and_the_writer_keeps_working(store: Store) -> None:
    def work(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO documents (doc_id, path, title, kind, file_hash, pipeline_version, "
            "state, created_at, updated_at) "
            "VALUES (?, 'p', 't', 'k', 'h', 'v', 'pending', 'n', 'n')",
            (DOC_ID,),
        )
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        store.write(work)
    assert store.get_document(DOC_ID) is None
    add_document(store)
    assert store.get_document(DOC_ID) is not None


def test_a_write_cannot_start_a_write(store: Store) -> None:
    with pytest.raises(StoreError, match="wait for itself"):
        store.write(lambda _conn: store.write(lambda _inner: None))


def test_data_survives_closing_and_reopening(tmp_path: Path) -> None:
    with Store(tmp_path) as first:
        add_document(first)
        first.sync_chunks(DOC_ID, identified(CORPUS))
    with Store(tmp_path) as second:
        assert second.get_document(DOC_ID) is not None
        assert texts_of(second, "refund") == [CORPUS[2], CORPUS[0]]


def test_a_closed_store_refuses_every_call_and_closing_twice_is_fine(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.close()
    store.close()
    with pytest.raises(StoreClosedError):
        store.get_document(DOC_ID)
    with pytest.raises(StoreClosedError):
        store.upsert_document(new_document())
    with pytest.raises(StoreClosedError):
        store.search_bm25("anything")


def test_a_read_pool_of_zero_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read_pool_size"):
        Store(tmp_path, read_pool_size=0)


# --- properties ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe() -> Iterator[sqlite3.Connection]:
    """A tiny FTS5 table with the store's tokenizer, to run generated queries against."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE probe USING fts5(text, "
        "tokenize = 'porter unicode61 remove_diacritics 2')"
    )
    conn.execute("INSERT INTO probe (text) VALUES ('alpha bravo charlie')")
    yield conn
    conn.close()


@given(text=st.text())
@settings(max_examples=300, deadline=None)
def test_any_text_makes_a_query_that_fts5_accepts(probe: sqlite3.Connection, text: str) -> None:
    query = fts_match_query(text)
    has_a_word = any(char.isalnum() for char in unicodedata.normalize("NFC", text))
    assert (query is not None) == has_a_word
    if query is not None:
        assert query.count(" OR ") + 1 <= MAX_QUERY_TERMS
        probe.execute("SELECT rowid FROM probe WHERE probe MATCH ?", (query,)).fetchall()


WORDS = ("alpha", "bravo", "charlie", "delta", "echo")
chunk_lists = st.lists(
    st.lists(st.sampled_from(WORDS), min_size=1, max_size=4).map(" ".join), max_size=6
)


def test_search_always_agrees_with_a_model_of_what_is_stored(tmp_path: Path) -> None:
    """After any two syncs: the diff is the set difference, and search finds what the text holds.

    The model is a few lines of Python: a chunk matches a word if the word is in its text. The
    store (inserts, deletes, in-place rewrites, triggers, stemming, the ready filter) must agree
    with it for every pair of chunk lists, repeats included.
    """
    with Store(tmp_path) as store:

        @given(first=chunk_lists, second=chunk_lists)
        @settings(max_examples=40, deadline=None)
        def check(first: list[str], second: list[str]) -> None:
            store.delete_document(DOC_ID)
            store.upsert_document(new_document())
            store.set_document_state(DOC_ID, "ready")
            before = {chunk.chunk_id for chunk in identified(first)}
            store.sync_chunks(DOC_ID, identified(first))

            wanted = identified(second)
            after = {chunk.chunk_id for chunk in wanted}
            result = store.sync_chunks(DOC_ID, wanted)

            assert set(result.added) == after - before
            assert set(result.removed) == before - after
            assert set(result.added) | set(result.updated) | set(result.unchanged) == after
            for word in WORDS:
                expected = {item.chunk_id for item in wanted if word in item.chunk.text.split()}
                found = {hit.chunk.chunk_id for hit in store.search_bm25(word, limit=100)}
                assert found == expected

        check()
