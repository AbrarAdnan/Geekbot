"""Tests for step C5's ✍️ piece: the registry of loaded CSV sessions (``csv_mode.sessions``).

The registry knows nothing about CSV files. It is given a function that turns a path into a session
(here: a small object that has a size and counts how often it was closed), so every rule in its
docstring can be checked with plain objects: one session per thread, least recently used out first,
both bounds, a session in use never closed, a table too big refused, a bad file leaving the old
session alone. The last test is a property test: whatever sequence of opens, acquires, releases and
drops comes in, no session is closed twice, none is closed while in use, none is leaked.
"""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.csv_mode.sessions import CsvSessions, Lease, SessionTooLargeError


class Table:
    """A loaded table: how big it is, and how many times it was closed."""

    def __init__(self, path: Path, size: int) -> None:
        self.path = path
        self.size = size
        self.closed = 0

    @property
    def memory_bytes(self) -> int:
        return self.size

    def close(self) -> None:
        self.closed += 1


class Opener:
    """Opens a ``Table`` for a path. The size comes from ``sizes`` by file name, else from a name of
    the form ``<anything>-<size>.csv``, else it is 10. A name starting with ``bad`` fails."""

    def __init__(self, **sizes: int) -> None:
        self.sizes = sizes
        self.opened: list[Table] = []

    def __call__(self, path: Path) -> Table:
        if path.name.startswith("bad"):
            raise ValueError(f"cannot load {path.name}")
        size = self.sizes.get(path.stem)
        if size is None:
            tail = path.stem.rpartition("-")[2]
            size = int(tail) if tail.isdigit() else 10
        table = Table(path, size)
        self.opened.append(table)
        return table


def make(
    max_sessions: int = 3, max_bytes: int = 1000, **sizes: int
) -> tuple[CsvSessions[Table], Opener]:
    opener = Opener(**sizes)
    return CsvSessions(opener, max_sessions=max_sessions, max_bytes=max_bytes), opener


def csv(name: str) -> Path:
    return Path(f"{name}.csv")


# --- one session per thread ------------------------------------------------------------------------


def test_an_opened_session_is_registered_with_its_file_and_size() -> None:
    sessions, opener = make()
    table = sessions.open("t-a", csv("a"))
    assert table is opener.opened[0]
    assert sessions.has("t-a") and not sessions.has("t-b")
    assert sessions.path_of("t-a") == csv("a") and sessions.path_of("t-b") is None
    assert (sessions.count, sessions.total_bytes) == (1, 10)


def test_a_new_upload_replaces_the_threads_session_and_closes_the_old_one() -> None:
    sessions, opener = make()
    first = sessions.open("t-a", csv("first"))
    second = sessions.open("t-a", csv("second"))
    assert sessions.count == 1 and sessions.path_of("t-a") == csv("second")
    assert (first.closed, second.closed) == (1, 0)
    assert opener.opened == [first, second]


def test_acquire_reuses_the_loaded_session_for_the_same_file() -> None:
    sessions, opener = make()
    loaded = sessions.open("t-a", csv("a"))
    lease = sessions.acquire("t-a", csv("a"))
    assert lease.session is loaded and len(opener.opened) == 1  # nothing was loaded again
    sessions.release(lease)


def test_acquire_loads_the_file_when_the_thread_has_no_session() -> None:
    """The restart case, and the evicted case: the file is the thread's, the table is gone."""
    sessions, opener = make()
    lease = sessions.acquire("t-a", csv("a"))
    assert len(opener.opened) == 1 and lease.session is opener.opened[0]
    assert sessions.has("t-a")
    sessions.release(lease)


def test_acquire_for_another_file_replaces_the_session() -> None:
    sessions, opener = make()
    old = sessions.open("t-a", csv("old"))
    lease = sessions.acquire("t-a", csv("new"))
    assert lease.session is opener.opened[1] and old.closed == 1
    sessions.release(lease)


# --- the two bounds, least recently used first -------------------------------------------------------


def test_the_session_count_is_bounded_and_the_oldest_goes_first() -> None:
    sessions, opener = make(max_sessions=2)
    a, b, c = (sessions.open(thread, csv(thread)) for thread in ("a", "b", "c"))
    assert sessions.thread_ids == ["b", "c"]
    assert (a.closed, b.closed, c.closed) == (1, 0, 0)
    assert len(opener.opened) == 3


def test_using_a_session_makes_it_the_newest() -> None:
    sessions, _ = make(max_sessions=2)
    sessions.open("a", csv("a"))
    b = sessions.open("b", csv("b"))
    sessions.release(sessions.acquire("a", csv("a")))  # a is used: b is now the oldest
    sessions.open("c", csv("c"))
    assert sessions.thread_ids == ["a", "c"] and b.closed == 1


def test_the_memory_is_bounded_too() -> None:
    sessions, _ = make(max_sessions=10, max_bytes=100, a=60, b=60, c=30)
    a = sessions.open("a", csv("a"))
    b = sessions.open("b", csv("b"))  # 120 bytes: a goes
    assert (a.closed, b.closed, sessions.total_bytes) == (1, 0, 60)
    sessions.open("c", csv("c"))  # 90 bytes: fits next to b
    assert sessions.thread_ids == ["b", "c"] and sessions.total_bytes == 90


def test_a_table_too_big_for_the_whole_budget_is_refused_and_closed() -> None:
    sessions, opener = make(max_bytes=100, huge=101)
    with pytest.raises(SessionTooLargeError, match="GEEKBOT_CSV_SESSION_BUDGET_BYTES"):
        sessions.open("t-a", csv("huge"))
    assert opener.opened[0].closed == 1  # it was loaded to measure it, and let go again
    assert sessions.count == 0


def test_a_refused_replacement_leaves_the_old_session_alone() -> None:
    sessions, opener = make(max_bytes=100, huge=101)
    good = sessions.open("t-a", csv("a"))
    with pytest.raises(SessionTooLargeError):
        sessions.open("t-a", csv("huge"))
    assert sessions.path_of("t-a") == csv("a") and good.closed == 0
    assert opener.opened[1].closed == 1


def test_a_file_that_cannot_be_loaded_leaves_the_old_session_alone() -> None:
    sessions, _ = make()
    good = sessions.open("t-a", csv("a"))
    with pytest.raises(ValueError, match="cannot load bad"):
        sessions.open("t-a", csv("bad"))
    assert sessions.path_of("t-a") == csv("a") and good.closed == 0
    with pytest.raises(ValueError, match="cannot load bad"):
        sessions.acquire("t-b", csv("bad"))
    assert not sessions.has("t-b")


# --- a session in use is never closed under its user --------------------------------------------------


def test_a_pinned_session_is_not_evicted_and_the_bounds_may_be_exceeded_meanwhile() -> None:
    sessions, _ = make(max_sessions=1)
    lease = sessions.acquire("a", csv("a"))
    b = sessions.open("b", csv("b"))  # would evict a, which is in use
    assert lease.session.closed == 0 and b.closed == 0
    assert sessions.thread_ids == ["a", "b"]  # two sessions with room for one: for now
    sessions.release(lease)
    sessions.open("c", csv("c"))  # the next installation brings the bounds back
    assert sessions.thread_ids == ["c"]
    assert lease.session.closed == 1 and b.closed == 1


def test_a_session_replaced_while_in_use_is_closed_when_the_question_ends() -> None:
    sessions, _ = make()
    lease = sessions.acquire("a", csv("old"))
    new = sessions.open("a", csv("new"))
    assert lease.session.closed == 0  # the question that holds it is still asking
    sessions.release(lease)
    assert lease.session.closed == 1 and new.closed == 0


def test_a_session_dropped_while_in_use_is_closed_when_the_question_ends() -> None:
    sessions, _ = make()
    lease = sessions.acquire("a", csv("a"))
    assert sessions.drop("a") is True
    assert not sessions.has("a") and lease.session.closed == 0
    sessions.release(lease)
    assert lease.session.closed == 1


def test_two_questions_can_hold_one_session_and_it_closes_after_the_last() -> None:
    sessions, _ = make()
    first = sessions.acquire("a", csv("a"))
    second = sessions.acquire("a", csv("a"))
    assert first.session is second.session
    sessions.drop("a")
    sessions.release(first)
    assert first.session.closed == 0
    sessions.release(second)
    assert first.session.closed == 1


def test_releasing_a_lease_twice_changes_nothing() -> None:
    sessions, _ = make()
    lease = sessions.acquire("a", csv("a"))
    other = sessions.acquire("a", csv("a"))
    sessions.release(lease)
    sessions.release(lease)  # a second give-back must not free the other holder's pin
    sessions.drop("a")
    assert lease.session.closed == 0
    sessions.release(other)
    assert lease.session.closed == 1


# --- dropping and closing everything -----------------------------------------------------------------


def test_drop_forgets_the_session_and_says_whether_there_was_one() -> None:
    sessions, _ = make()
    table = sessions.open("a", csv("a"))
    assert sessions.drop("a") is True and table.closed == 1
    assert sessions.drop("a") is False
    assert sessions.count == 0


def test_close_all_closes_every_session_but_leaves_the_pinned_to_their_release() -> None:
    sessions, _ = make()
    free = sessions.open("a", csv("a"))
    lease = sessions.acquire("b", csv("b"))
    sessions.close_all()
    assert sessions.count == 0 and free.closed == 1 and lease.session.closed == 0
    sessions.release(lease)
    assert lease.session.closed == 1


def test_a_session_that_fails_to_close_does_not_stop_the_installation() -> None:
    class Stubborn(Table):
        def close(self) -> None:
            raise OSError("cannot close")

    def open_session(path: Path) -> Table:
        return (Stubborn if path.stem == "a" else Table)(path, 10)

    sessions = CsvSessions(open_session, max_sessions=1, max_bytes=1000)
    sessions.open("a", csv("a"))
    sessions.open("b", csv("b"))  # evicting a fails to close it; b is installed all the same
    assert sessions.thread_ids == ["b"]


def test_a_registry_needs_room_for_something() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CsvSessions(Opener(), max_sessions=0, max_bytes=10)
    with pytest.raises(ValueError, match="at least one"):
        CsvSessions(Opener(), max_sessions=1, max_bytes=0)


# --- the promises, over any sequence -------------------------------------------------------------------

OPERATION = st.tuples(
    st.sampled_from(["open", "acquire", "release", "drop"]),
    st.sampled_from(["a", "b", "c", "d"]),
    st.integers(min_value=1, max_value=60),
)


@settings(max_examples=200, deadline=None)
@given(st.lists(OPERATION, max_size=40))
def test_whatever_happens_nothing_is_closed_twice_or_under_its_user_or_leaked(
    operations: list[tuple[str, str, int]],
) -> None:
    opener = Opener()
    sessions: CsvSessions[Table] = CsvSessions(opener, max_sessions=2, max_bytes=100)
    held: list[Lease[Table]] = []
    for kind, thread, size in operations:
        path = Path(f"{thread}-{size}.csv")  # the size is part of the name, see ``Opener``
        if kind == "open":
            sessions.open(thread, path)
        elif kind == "acquire":
            held.append(sessions.acquire(thread, path))
        elif kind == "release" and held:
            sessions.release(held.pop(0))
        elif kind == "drop":
            sessions.drop(thread)
        assert all(table.closed <= 1 for table in opener.opened)  # never closed twice
        assert all(lease.session.closed == 0 for lease in held)  # never closed while in use
    for lease in held:
        sessions.release(lease)
    sessions.open("z", Path("z-1.csv"))  # one more installation: the bounds hold again
    assert sessions.count <= 2 and sessions.total_bytes <= 100
    assert all(table.closed <= 1 for table in opener.opened)
    still_open = [table for table in opener.opened if table.closed == 0]
    assert len(still_open) == sessions.count  # nothing is open that the registry does not hold
