"""Tests for step S1.4: the storage lock (PLAN.md §6.2) and the store that holds it.

The invariant under test: exactly one process owns the stores, and a second one is refused at
once with an error, never made to wait and never allowed to carry on. The last test starts a real
second process, because a lock that only works inside one process proves little.
"""

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from geekbot.index.lock import LOCK_FILE_NAME, StorageLock, StorageLockedError
from geekbot.index.sqlite import DB_FILE_NAME, Store

# What the second process runs: try to take the lock, and say through the exit code what happened.
CHILD = """
import sys
from pathlib import Path

from geekbot.index.lock import StorageLock, StorageLockedError

try:
    StorageLock(Path(sys.argv[1])).acquire()
except StorageLockedError:
    sys.exit(3)
"""


def test_acquire_creates_the_data_directory_and_the_lock_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "not" / "there" / "yet"
    lock = StorageLock(data_dir)
    lock.acquire()
    try:
        assert lock.is_held
        assert lock.path == data_dir / LOCK_FILE_NAME
        assert lock.path.exists()
    finally:
        lock.release()
    assert not lock.is_held


def test_a_second_lock_on_the_same_directory_is_refused_at_once(tmp_path: Path) -> None:
    with StorageLock(tmp_path):
        started = time.monotonic()
        with pytest.raises(StorageLockedError, match="already owns the storage"):
            StorageLock(tmp_path).acquire()
        assert time.monotonic() - started < 5  # refused, not made to wait


def test_the_lock_can_be_taken_again_after_it_is_released(tmp_path: Path) -> None:
    first = StorageLock(tmp_path)
    first.acquire()
    first.release()
    with StorageLock(tmp_path):
        pass


def test_the_context_manager_releases_the_lock_when_the_body_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), StorageLock(tmp_path):
        raise RuntimeError("boom")
    with StorageLock(tmp_path):
        pass


def test_one_lock_object_cannot_be_taken_twice(tmp_path: Path) -> None:
    with StorageLock(tmp_path) as lock, pytest.raises(StorageLockedError, match="already holds"):
        lock.acquire()


def test_locks_on_different_directories_do_not_interfere(tmp_path: Path) -> None:
    with StorageLock(tmp_path / "one"), StorageLock(tmp_path / "two"):
        pass


def test_releasing_a_lock_that_is_not_held_does_nothing(tmp_path: Path) -> None:
    StorageLock(tmp_path).release()


def test_a_second_process_is_refused(tmp_path: Path) -> None:
    with StorageLock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", CHILD, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert result.returncode == 3, result.stderr
    # With the lock released, the same child gets it.
    free = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert free.returncode == 0, free.stderr


def test_a_store_holds_the_lock_until_it_is_closed(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(StorageLockedError):
        StorageLock(tmp_path).acquire()
    store.close()
    with StorageLock(tmp_path):
        pass


def test_a_second_store_on_the_same_directory_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path) as first:
        with pytest.raises(StorageLockedError):
            Store(tmp_path)
        assert first.get_document("0" * 16) is None  # the first store is unharmed


def test_a_store_that_fails_to_open_releases_the_lock(tmp_path: Path) -> None:
    (tmp_path / DB_FILE_NAME).write_bytes(b"this is not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        Store(tmp_path)
    with StorageLock(tmp_path):
        pass
