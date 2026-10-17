"""Are the two stores in step, and if not, make them so (step S1.6, PLAN.md §6.4; ``doctor --repair``).

SQLite holds the rows, Chroma the vectors, and a crash can land between the two writes. Nothing
here is surprising once the outbox (``ingest/outbox.py``) is clear, because the state of a document
says exactly how far its ingest got:

=============  =====================================================  ==============================
state found    what it means                                          the repair
=============  =====================================================  ==============================
``pending``    recorded, rows not (all) written: the file's hash and  marks it ``failed`` ("ingest
               version are already the new ones but the chunks may    it again"): the repair has not
               still be the old file's, so the chunks cannot be       read the file, and a document
               trusted                                                must never be ``ready`` with
                                                                      content of another version
``embedding``  rows final, vectors being written                      finishes the job: embeds what
                                                                      is missing, then ``ready``
``ready``      rows and vectors should both exist                     if vectors are missing (Chroma
                                                                      was wiped or a write was lost),
                                                                      hides the document, re-embeds,
                                                                      ``ready`` again
``failed``     a recorded error                                       left alone (ingest retries it)
=============  =====================================================  ==============================

Vectors in Chroma that no chunk row owns (a document was deleted, a crash came between the two
deletes) are removed: they can never be returned, because search goes through SQLite, but they take
space and would be confusing in a count.

Re-embedding goes through the cache, so a repair after a wiped vector store costs no embeddings if
the cache is intact, and a re-index under a new embedder (``reset`` the dense index, then repair)
embeds each chunk once.
"""

from dataclasses import dataclass

from geekbot.domain import Document
from geekbot.index.dense import DenseIndex
from geekbot.index.embed_cache import CachedEmbedder
from geekbot.index.sqlite import Store
from geekbot.index.vectors import write_document_vectors
from geekbot.ingest.outbox import Outbox

__all__ = ["ConsistencyReport", "RepairReport", "check_consistency", "repair"]

_INTERRUPTED = "interrupted before its chunks were written; ingest it again"


@dataclass(frozen=True)
class ConsistencyReport:
    """What is out of step between the stores. All fields are document or chunk IDs, sorted."""

    interrupted: tuple[str, ...]  # documents stuck in ``pending``
    unfinished: tuple[str, ...]  # documents stuck in ``embedding``
    missing_vectors: tuple[str, ...]  # ``ready`` documents with chunks that have no vector
    orphan_vectors: tuple[str, ...]  # vectors that no chunk row owns

    @property
    def ok(self) -> bool:
        """True when there is nothing to repair."""
        return not (
            self.interrupted or self.unfinished or self.missing_vectors or self.orphan_vectors
        )

    def summary(self) -> str:
        """One line a person can read."""
        if self.ok:
            return "stores are consistent"
        return (
            f"{len(self.interrupted)} interrupted, {len(self.unfinished)} unfinished, "
            f"{len(self.missing_vectors)} documents missing vectors, "
            f"{len(self.orphan_vectors)} orphan vectors"
        )


@dataclass(frozen=True)
class RepairReport:
    """What a repair did, in documents and counts."""

    marked_failed: tuple[str, ...]  # doc IDs that were ``pending`` and are now ``failed``
    finished: tuple[str, ...]  # doc IDs that were ``embedding`` and are now ``ready``
    re_embedded: tuple[str, ...]  # ``ready`` doc IDs whose vectors were rebuilt
    failed: tuple[str, ...]  # doc IDs the repair could not finish (now ``failed``)
    orphans_removed: int
    embedded: int
    cache_hits: int

    def summary(self) -> str:
        """One line a person can read."""
        return (
            f"{len(self.finished)} unfinished documents finished, {len(self.re_embedded)} rebuilt, "
            f"{len(self.marked_failed)} interrupted ones marked failed, {len(self.failed)} could "
            f"not be repaired, {self.orphans_removed} orphan vectors removed; "
            f"{self.embedded} embeddings computed, {self.cache_hits} from the cache"
        )


def check_consistency(store: Store, dense: DenseIndex) -> ConsistencyReport:
    """Compare the two stores without changing either."""
    present = dense.ids()
    by_document = store.chunk_ids_by_document()
    owned = set().union(*by_document.values()) if by_document else set()
    interrupted: list[str] = []
    unfinished: list[str] = []
    missing: list[str] = []
    for document in store.list_documents():
        if document.state == "pending":
            interrupted.append(document.doc_id)
        elif document.state == "embedding":
            unfinished.append(document.doc_id)
        elif document.state == "ready" and by_document.get(document.doc_id, set()) - present:
            missing.append(document.doc_id)
    return ConsistencyReport(
        interrupted=tuple(sorted(interrupted)),
        unfinished=tuple(sorted(unfinished)),
        missing_vectors=tuple(sorted(missing)),
        orphan_vectors=tuple(sorted(present - owned)),
    )


def repair(
    store: Store,
    dense: DenseIndex,
    embedder: CachedEmbedder,
    *,
    rebuild_fts: bool = False,
) -> RepairReport:
    """Bring the stores back in step (see the table in the module docstring).

    ``rebuild_fts`` also rebuilds the full-text index from the chunk rows, the repair for a
    full-text index that has drifted (it is derived data, so this is always safe, only slow).
    Safe to run twice: the second run finds nothing to do.
    """
    outbox = Outbox(store)
    by_document = store.chunk_ids_by_document()
    owned = set().union(*by_document.values()) if by_document else set()
    orphans = sorted(dense.ids() - owned)
    dense.delete(orphans)
    present = dense.ids()

    marked_failed: list[str] = []
    finished: list[str] = []
    rebuilt: list[str] = []
    failed: list[str] = []
    embedded = cache_hits = 0

    def rebuild(document: Document) -> None:
        nonlocal embedded, cache_hits
        chunks = store.document_chunks(document.doc_id)
        items = [
            (chunk.chunk_id, chunk.embedding_text(with_header=embedder.spec.with_header))
            for chunk in chunks
        ]
        written = write_document_vectors(dense, embedder, document.doc_id, items)
        embedded += written.embedded
        cache_hits += written.cache_hits

    for document in store.list_documents():
        doc_id = document.doc_id
        if document.state == "pending":
            outbox.fail(doc_id, _INTERRUPTED)
            marked_failed.append(doc_id)
            continue
        if document.state == "embedding":
            bucket = finished
        elif document.state == "ready" and by_document.get(doc_id, set()) - present:
            # Hidden from search while its vectors are rebuilt: pending, then embedding.
            outbox.move(doc_id, "pending")
            outbox.move(doc_id, "embedding")
            bucket = rebuilt
        else:
            continue
        try:
            rebuild(document)
            outbox.move(doc_id, "ready")
            bucket.append(doc_id)
        except Exception as error:  # noqa: BLE001 - any failure fails this document, not the repair
            outbox.fail(doc_id, f"repair failed: {type(error).__name__}: {error}")
            failed.append(doc_id)

    if rebuild_fts:
        store.rebuild_fts()
    return RepairReport(
        marked_failed=tuple(marked_failed),
        finished=tuple(finished),
        re_embedded=tuple(rebuilt),
        failed=tuple(failed),
        orphans_removed=len(orphans),
        embedded=embedded,
        cache_hits=cache_hits,
    )
