"""Tests for step S1.6: the outbox state machine and the store methods it stands on.

The state machine is this step's ✍️ piece (written by Claude in prototyping mode, D-047): every one
of the 16 (from, to) pairs is checked against the table in ``ingest/outbox.py``, then a real store is
driven through the legal and the illegal moves.
"""

import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from geekbot.domain import DOC_STATES, DocState, NewDocument
from geekbot.index.sqlite import StateConflictError, Store, UnknownDocumentError
from geekbot.ingest.chunking import ChunkDraft
from geekbot.ingest.ids import IdentifiedChunk, doc_id_of, identify_chunks
from geekbot.ingest.outbox import ALLOWED_MOVES, InvalidTransitionError, Outbox, check_move

# The edges of the diagram in ingest/outbox.py, written out a second time on purpose: a change to
# the table must be a change to this list too.
LEGAL = {
    ("pending", "embedding"),
    ("pending", "failed"),
    ("embedding", "ready"),
    ("embedding", "failed"),
    ("embedding", "pending"),
    ("ready", "pending"),
    ("failed", "pending"),
}


# --- the pure rule ---------------------------------------------------------------------------


@pytest.mark.parametrize(("current", "new"), list(itertools.product(DOC_STATES, DOC_STATES)))
def test_every_pair_of_states_is_decided(current: DocState, new: DocState) -> None:
    if (current, new) in LEGAL:
        check_move(current, new)  # does not raise
    else:
        with pytest.raises(InvalidTransitionError) as caught:
            check_move(current, new)
        assert (caught.value.current, caught.value.new) == (current, new)


def test_the_table_covers_every_state_and_names_only_real_ones() -> None:
    assert set(ALLOWED_MOVES) == set(DOC_STATES)
    assert all(targets <= set(DOC_STATES) for targets in ALLOWED_MOVES.values())


def test_a_document_never_skips_embedding_on_its_way_to_ready() -> None:
    with pytest.raises(InvalidTransitionError, match="pending cannot become ready"):
        check_move("pending", "ready")


def test_staying_where_you_are_is_a_bug_in_the_caller() -> None:
    for state in DOC_STATES:
        with pytest.raises(InvalidTransitionError):
            check_move(state, state)


def test_an_unknown_state_is_a_value_error_not_a_transition_error() -> None:
    with pytest.raises(ValueError, match="unknown document state"):
        check_move("pending", "done")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown document state"):
        check_move("done", "pending")  # type: ignore[arg-type]


def test_the_error_names_where_the_document_can_go() -> None:
    with pytest.raises(InvalidTransitionError, match="it can go to: embedding, failed"):
        check_move("pending", "ready")


# --- a real store ----------------------------------------------------------------------------

PATH = "/data/notes.md"
DOC_ID = doc_id_of(PATH)


def new_document(path: str = PATH) -> NewDocument:
    return NewDocument(
        doc_id=doc_id_of(path),
        path=path,
        title="notes.md",
        kind="markdown",
        file_hash="f" * 64,
        pipeline_version="p1",
    )


def identified(texts: Sequence[str], path: str = PATH) -> list[IdentifiedChunk]:
    drafts = [
        ChunkDraft(
            text=text,
            char_start=i * 100,
            char_end=i * 100 + len(text),
            page=None,
            section_path=("Intro",),
            header="notes.md > Intro",
            ordinal=i,
        )
        for i, text in enumerate(texts)
    ]
    return identify_chunks(path, drafts)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path) as opened:
        yield opened


def state_of(store: Store, doc_id: str = DOC_ID) -> DocState:
    document = store.get_document(doc_id)
    assert document is not None
    return document.state


def test_the_whole_route_to_ready(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    assert state_of(store) == "pending"
    assert outbox.move(DOC_ID, "embedding") == "pending"  # the move returns where it was
    assert outbox.move(DOC_ID, "ready") == "embedding"
    assert state_of(store) == "ready"


def test_an_illegal_move_changes_nothing(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    with pytest.raises(InvalidTransitionError) as caught:
        outbox.move(DOC_ID, "ready")
    assert (caught.value.current, caught.value.new) == ("pending", "ready")
    assert state_of(store) == "pending"


@pytest.mark.parametrize(("current", "new"), [p for p in itertools.product(DOC_STATES, DOC_STATES)])
def test_the_store_agrees_with_the_pure_rule(
    store: Store, current: DocState, new: DocState
) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    store.set_document_state(DOC_ID, current)  # put it anywhere, bypassing the machine
    if (current, new) in LEGAL:
        assert outbox.move(DOC_ID, new) == current
        assert state_of(store) == new
    else:
        with pytest.raises(InvalidTransitionError):
            outbox.move(DOC_ID, new)
        assert state_of(store) == current


def test_begin_restarts_a_ready_or_failed_document(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    outbox.move(DOC_ID, "embedding")
    outbox.move(DOC_ID, "ready")
    outbox.begin(new_document())  # an edited file: back to the start
    assert state_of(store) == "pending"
    outbox.fail(DOC_ID, "boom")
    outbox.begin(new_document())  # a retry
    document = store.get_document(DOC_ID)
    assert document is not None
    assert (document.state, document.error) == ("pending", None)


def test_fail_records_a_trimmed_one_line_reason(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    outbox.fail(DOC_ID, "line one\n   line two\t" + "x" * 1000)
    document = store.get_document(DOC_ID)
    assert document is not None
    assert document.state == "failed"
    assert document.error is not None
    assert document.error.startswith("line one line two xxx")
    assert len(document.error) == 500
    assert "\n" not in document.error


def test_fail_with_an_empty_reason_still_says_something(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    outbox.fail(DOC_ID, "   ")
    document = store.get_document(DOC_ID)
    assert document is not None
    assert document.error == "failed for an unknown reason"


def test_a_ready_document_cannot_simply_be_failed(store: Store) -> None:
    outbox = Outbox(store)
    outbox.begin(new_document())
    outbox.move(DOC_ID, "embedding")
    outbox.move(DOC_ID, "ready")
    with pytest.raises(InvalidTransitionError):
        outbox.fail(DOC_ID, "late")
    assert state_of(store) == "ready"


def test_an_unknown_document_is_the_stores_error(store: Store) -> None:
    with pytest.raises(UnknownDocumentError):
        Outbox(store).move("nope", "embedding")


# --- Store.transition_document ---------------------------------------------------------------


def test_transition_is_a_compare_and_set(store: Store) -> None:
    store.upsert_document(new_document())
    assert store.transition_document(DOC_ID, {"pending"}, "embedding") == "pending"
    with pytest.raises(StateConflictError) as caught:
        store.transition_document(DOC_ID, {"pending"}, "failed")
    assert caught.value.actual == "embedding"
    assert state_of(store) == "embedding"


def test_transition_validates_its_arguments(store: Store) -> None:
    store.upsert_document(new_document())
    with pytest.raises(ValueError, match="unknown document state"):
        store.transition_document(DOC_ID, {"pending"}, "done")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only goes with the state 'failed'"):
        store.transition_document(DOC_ID, {"pending"}, "embedding", error="x")


# --- Store.list_documents / document_chunks / chunk_ids_by_document -------------------------


def test_list_documents_is_ordered_and_can_filter(store: Store) -> None:
    for path in ("/data/b.md", "/data/a.md", "/data/c.md"):
        store.upsert_document(new_document(path))
    store.set_document_state(doc_id_of("/data/b.md"), "ready")
    assert [d.path for d in store.list_documents()] == ["/data/a.md", "/data/b.md", "/data/c.md"]
    assert [d.path for d in store.list_documents("ready")] == ["/data/b.md"]
    assert store.list_documents("failed") == []
    with pytest.raises(ValueError, match="unknown document state"):
        store.list_documents("done")  # type: ignore[arg-type]


def test_document_chunks_come_in_document_order_whatever_the_state(store: Store) -> None:
    store.upsert_document(new_document())  # pending: get_chunks would hide these
    store.sync_chunks(DOC_ID, identified(["first", "second", "third"]))
    chunks = store.document_chunks(DOC_ID)
    assert [c.text for c in chunks] == ["first", "second", "third"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert chunks[0].header == "notes.md > Intro"
    assert store.get_chunks([c.chunk_id for c in chunks]) == []  # still not ready
    assert store.document_chunks("nope") == []


def test_chunk_ids_by_document_groups_without_reading_text(store: Store) -> None:
    other = "/data/other.md"
    store.upsert_document(new_document())
    store.upsert_document(new_document(other))
    store.upsert_document(new_document("/data/empty.md"))
    mine = identified(["a", "b"])
    theirs = identified(["c"], other)
    store.sync_chunks(DOC_ID, mine)
    store.sync_chunks(doc_id_of(other), theirs)
    grouped = store.chunk_ids_by_document()
    assert grouped == {
        DOC_ID: {c.chunk_id for c in mine},
        doc_id_of(other): {c.chunk_id for c in theirs},
    }  # the document with no chunks is absent
