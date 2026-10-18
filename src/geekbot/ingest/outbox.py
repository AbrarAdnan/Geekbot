"""The outbox: the legal moves of a document through ingest (step S1.6, PLAN.md §6.4).

A document lives in two stores, SQLite (the rows) and Chroma (the vectors), and no operation can
change both at once. The **outbox pattern** makes that safe by writing the order down. A document
is in one of four states and only moves along these edges::

    pending ──▶ embedding ──▶ ready
       │            │  ▲        │
       ▼            ▼  │        │ (re-ingest or repair: back to pending)
     failed ◀───────┘  └── failed ──▶ pending (retry)

- ``pending``: the document is recorded and its chunk rows are being written.
- ``embedding``: the rows are final (one SQLite transaction), and the vectors are being written.
- ``ready``: rows and vectors both exist. Only now is the document searchable.
- ``failed``: something went wrong; ``Document.error`` says what.

Search only ever sees ``ready`` documents (S1.4), so a crash between the two stores leaves a
document that nobody can retrieve and that ``doctor --repair`` (``index/consistency.py``) can
finish, instead of one that is half-visible. Because the order is fixed, a repair can read the
state and know exactly what had and had not been done.

The table is written here and checked here, in one place. ``Store.transition_document`` makes each
move atomic (read the state and change it in one transaction); this module decides which moves
are allowed.
"""

from collections.abc import Mapping
from typing import Final

from geekbot.domain import DOC_STATES, DocState, NewDocument
from geekbot.index.sqlite import StateConflictError, Store

__all__ = ["ALLOWED_MOVES", "InvalidTransitionError", "Outbox", "check_move"]

ALLOWED_MOVES: Final[Mapping[DocState, frozenset[DocState]]] = {
    "pending": frozenset({"embedding", "failed"}),
    "embedding": frozenset({"ready", "failed", "pending"}),
    "ready": frozenset({"pending"}),
    "failed": frozenset({"pending"}),
}

_MAX_ERROR_CHARS: Final = 500


class InvalidTransitionError(ValueError):
    """A document was asked to make a move the outbox does not allow."""

    def __init__(self, current: DocState, new: DocState) -> None:
        allowed = ", ".join(sorted(ALLOWED_MOVES[current])) or "nowhere"
        super().__init__(
            f"a document that is {current} cannot become {new} (it can go to: {allowed})"
        )
        self.current = current
        self.new = new


def check_move(current: DocState, new: DocState) -> None:
    """Raise ``InvalidTransitionError`` unless a document may go from ``current`` to ``new``.

    ✍️ Written by Claude (prototyping mode, D-047). The legal moves are the edges in this module's
    docstring and nothing else: in particular a document never skips ``embedding`` on its way to
    ``ready`` (that would claim vectors exist that nobody wrote), never leaves ``ready`` except
    back to ``pending`` (a re-ingest or a repair starts the whole route again), and never stays
    where it is (a move to the same state is a bug in the caller, so it raises too).
    ``ValueError`` for a state that is not one of the four.
    """
    for state in (current, new):
        if state not in DOC_STATES:
            raise ValueError(f"unknown document state {state!r}; it must be one of {DOC_STATES}")
    if new not in ALLOWED_MOVES[current]:
        raise InvalidTransitionError(current, new)


def _sources_of(new: DocState) -> frozenset[DocState]:
    """The states a document may be in right before it moves to ``new``."""
    return frozenset(state for state, targets in ALLOWED_MOVES.items() if new in targets)


class Outbox:
    """Drives one store's documents through the state machine, one atomic move at a time."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def begin(self, document: NewDocument) -> None:
        """Record a document for (re)processing: it becomes ``pending`` from whatever it was.

        This is the one move that is allowed from every state: a new file, an edited file, a
        retry after ``failed`` and a re-index of a ``ready`` document all start the route over.
        """
        self._store.upsert_document(document)

    def move(self, doc_id: str, new: DocState, *, error: str | None = None) -> DocState:
        """Move a document to ``new`` if the table allows it; return the state it was in.

        ``InvalidTransitionError`` if the move is not allowed from the state the document is in
        right now. The check and the change are one transaction, so two workers cannot both
        make the same move.
        """
        try:
            return self._store.transition_document(doc_id, _sources_of(new), new, error=error)
        except StateConflictError as conflict:
            raise InvalidTransitionError(conflict.actual, new) from conflict

    def fail(self, doc_id: str, reason: str) -> DocState:
        """Move a document to ``failed`` with a reason that fits the report (trimmed if long)."""
        text = " ".join(reason.split())[:_MAX_ERROR_CHARS] or "failed for an unknown reason"
        return self.move(doc_id, "failed", error=text)
