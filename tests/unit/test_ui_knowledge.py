"""Tests for step S4.4: what the knowledge page does (``geekbot.ui.views.knowledge``).

The page is thin; its work is in ``parse_paths``, ``add_uploads``, ``add_paths`` and
``remove_document``, which take a client and a ``wait`` function and return ``Outcome`` rows. They run
here against the real server (the same app the client tests use), so "uploading the same file again
costs nothing" is checked end to end, not assumed.

The page module imports Streamlit, so ``conftest.py`` leaves this file out when Streamlit is not
installed (an environment that has not been updated for S4.4) instead of breaking the whole run.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from serverkit import make_app, write_note

from geekbot.server.security import LOOPBACK_HOSTS
from geekbot.ui.client import GeekbotClient, wait_for_job
from geekbot.ui.views.knowledge import (
    Outcome,
    add_paths,
    add_uploads,
    parse_paths,
    remove_document,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

KEY = "ui-knowledge-test-key-" + "k" * 12
LOCAL = "http://127.0.0.1:8000"
NOTE = b"Zebras live on the savanna.\n\nTheir stripes are unique.\n"


@contextmanager
def connected(tmp_path: Path) -> Iterator[GeekbotClient]:
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS)
    with TestClient(app, base_url=LOCAL) as http:
        yield GeekbotClient(LOCAL, KEY, http=http)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[GeekbotClient]:
    with connected(tmp_path) as opened:
        yield opened


def waiter(client: GeekbotClient) -> Callable[[str], dict[str, Any]]:
    def wait(job_id: str) -> dict[str, Any]:
        return wait_for_job(client, job_id, poll_seconds=0.02, timeout_seconds=30)

    return wait


# --- the parser ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "paths"),
    [
        ("", []),
        ("   \n\n  ", []),
        ("C:\\notes", ["C:\\notes"]),
        ('"C:\\My Docs\\a.pdf"', ["C:\\My Docs\\a.pdf"]),  # Windows' "Copy as path"
        ("'/home/me/notes'", ["/home/me/notes"]),
        ("  C:\\a  \n C:\\b\r\nC:\\c", ["C:\\a", "C:\\b", "C:\\c"]),
        ("C:\\a\nC:\\b\nC:\\a", ["C:\\a", "C:\\b"]),  # a repeat is dropped, the order kept
        ('""\n"  "\n', []),  # quotes around nothing are nothing
    ],
)
def test_parse_paths(raw: str, paths: list[str]) -> None:
    assert parse_paths(raw) == paths


def test_parse_paths_changes_nothing_but_quotes_and_white_space() -> None:
    """Whether a path is allowed is the server's decision; a relative or odd path goes through."""
    assert parse_paths("..\\secrets\nnot a path at all") == ["..\\secrets", "not a path at all"]


# --- adding uploads --------------------------------------------------------------------------


def test_an_uploaded_file_is_added_and_its_outcome_says_what_happened(
    client: GeekbotClient,
) -> None:
    (outcome,) = add_uploads(client, [("zebras.txt", NOTE)], wait=waiter(client))
    assert outcome.ok is True
    assert outcome.label == "zebras.txt"
    assert "1 ingested" in outcome.message
    assert len(client.documents()) == 1


def test_the_same_file_again_is_unchanged_and_says_the_server_had_it(client: GeekbotClient) -> None:
    """A repeated click must not cost an embedding (the lesson of v1 bug #1)."""
    wait = waiter(client)
    add_uploads(client, [("zebras.txt", NOTE)], wait=wait)
    (again,) = add_uploads(client, [("zebras.txt", NOTE)], wait=wait)
    assert again.ok is True
    assert "1 unchanged" in again.message
    assert "0 ingested" in again.message
    assert "already had this exact file" in again.message
    assert len(client.documents()) == 1


def test_one_bad_file_does_not_stop_the_others(client: GeekbotClient) -> None:
    outcomes = add_uploads(
        client,
        [("evil.exe", b"MZ"), ("zebras.txt", NOTE), ("empty.txt", b"")],
        wait=waiter(client),
    )
    assert [outcome.label for outcome in outcomes] == ["evil.exe", "zebras.txt", "empty.txt"]
    assert [outcome.ok for outcome in outcomes] == [False, True, False]
    assert "not accepted" in outcomes[0].message
    assert "empty" in outcomes[2].message
    assert len(client.documents()) == 1


def test_no_files_no_outcomes(client: GeekbotClient) -> None:
    assert add_uploads(client, [], wait=waiter(client)) == []


def test_a_file_the_loader_cannot_read_is_reported_as_failed_with_its_reason(
    client: GeekbotClient,
) -> None:
    (outcome,) = add_uploads(client, [("broken.pdf", b"this is not a pdf")], wait=waiter(client))
    assert outcome.ok is False
    assert "broken.pdf" in outcome.message
    assert len(client.documents()) == 0 or client.documents()[0]["state"] == "failed"


# --- adding by path --------------------------------------------------------------------------


def test_paths_inside_an_allowed_folder_are_added(client: GeekbotClient, tmp_path: Path) -> None:
    note = write_note(tmp_path)
    (outcome,) = add_paths(client, f'"{note}"\n', wait=waiter(client))  # pasted with quotes
    assert outcome.ok is True
    assert outcome.label == str(note)
    assert "1 ingested" in outcome.message


def test_a_path_outside_the_allowed_folders_is_refused_with_the_servers_reason(
    client: GeekbotClient, tmp_path: Path
) -> None:
    outside = Path(tmp_path.anchor) / "elsewhere"
    (outcome,) = add_paths(client, str(outside), wait=waiter(client))
    assert outcome.ok is False
    assert outcome.message  # the server's own explanation, whatever its words
    assert client.documents() == []


def test_an_empty_box_asks_for_a_path_and_calls_nothing(client: GeekbotClient) -> None:
    def must_not_wait(job_id: str) -> dict[str, Any]:
        raise AssertionError("no job should have been made")

    (outcome,) = add_paths(client, "  \n ", wait=must_not_wait)
    assert outcome == Outcome("", ok=False, message="type at least one full path")


# --- removing --------------------------------------------------------------------------------


def test_a_document_is_removed_from_the_index_and_its_file_is_not_touched(
    client: GeekbotClient, tmp_path: Path
) -> None:
    note = write_note(tmp_path)
    add_paths(client, str(note), wait=waiter(client))
    (document,) = client.documents()
    outcome = remove_document(client, document["doc_id"], "Zebras", wait=waiter(client))
    assert outcome == Outcome("Zebras", ok=True, message="removed from the index")
    assert client.documents() == []
    assert note.exists()


def test_removing_a_document_that_is_not_there_is_a_failed_outcome_not_a_crash(
    client: GeekbotClient,
) -> None:
    outcome = remove_document(client, "no-such-id", "Ghost", wait=waiter(client))
    assert outcome.ok is False
    assert "no document" in outcome.message
