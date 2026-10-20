"""Tests for step S4.3: receiving a file (``server/uploads.py`` and the ``/v1/kb/upload`` route).

First the name (a table and two properties), then ``save_upload`` on its own with an in-memory body
(size, empty, all-or-nothing, cancellation), then the route through the test client.
"""

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from serverkit import make_app, wait_for

from geekbot.server.uploads import (
    MAX_STEM_CHARS,
    SHA_CHARS,
    UploadError,
    safe_upload_name,
    save_upload,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

SUFFIXES = frozenset({".txt", ".pdf", ".md"})
E_ACUTE = chr(0xE9)
COMBINING_ACUTE = chr(0x301)


# --- the name --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given_name", "stored"),
    [
        ("notes.txt", "notes.txt"),
        ("REPORT.PDF", "REPORT.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\evil.md", "evil.md"),
        ("C:\\Users\\me\\notes.md", "notes.md"),
        ("/abs/path/notes.md", "notes.md"),
        ("my notes (final).txt", "my_notes_final.txt"),
        ("a.exe.txt", "a.exe.txt"),
        ("***.txt", "file.txt"),
        ("-._.txt", "file.txt"),
        ("a\x00b.txt", "a_b.txt"),
        ("caf" + E_ACUTE + ".txt", "caf" + E_ACUTE + ".txt"),
        ("cafe" + COMBINING_ACUTE + ".txt", "caf" + E_ACUTE + ".txt"),  # NFC
        ("x" * 500 + ".txt", "x" * MAX_STEM_CHARS + ".txt"),
    ],
)
def test_a_name_becomes_a_plain_file_name(given_name: str, stored: str) -> None:
    assert safe_upload_name(given_name, SUFFIXES) == stored


@pytest.mark.parametrize(
    "given_name",
    ["", "noextension", ".txt", "..txt", "a.exe", "a.txt.exe", "dir/", "a.txt/"],
    ids=[
        "empty",
        "no-extension",
        "dotfile",
        "dots-only",
        "exe",
        "double-exe",
        "folder",
        "folder-txt",
    ],
)
def test_a_type_that_is_not_listed_is_415(given_name: str) -> None:
    with pytest.raises(UploadError) as raised:
        safe_upload_name(given_name, SUFFIXES)
    assert raised.value.status == 415
    assert ".txt" in str(raised.value)  # it says what is accepted


@settings(max_examples=300, deadline=None)
@given(name=st.one_of(st.text(), st.text().map(lambda text: text + ".pdf")))
def test_whatever_the_name_the_result_is_a_bare_file_name(name: str) -> None:
    try:
        stored = safe_upload_name(name, SUFFIXES)
    except UploadError:
        return
    assert stored == Path(stored).name  # no directory part
    assert not set(stored) & set('/\\:\x00*?"<>|')
    assert not stored.startswith(".")
    assert stored.endswith(tuple(SUFFIXES))
    assert len(stored) <= MAX_STEM_CHARS + len(".pdf")
    assert safe_upload_name(stored, SUFFIXES) == stored  # cleaning twice changes nothing


# --- save_upload -----------------------------------------------------------------------------


async def body(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def save(directory: Path, name: str, *parts: bytes, max_bytes: int = 1000) -> Any:
    return asyncio.run(
        save_upload(
            body(*parts), filename=name, directory=directory, max_bytes=max_bytes, suffixes=SUFFIXES
        )
    )


def left_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def test_a_file_is_stored_as_its_digest_and_its_name(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    stored = save(folder, "../notes.txt", b"hello ", b"world")
    digest = hashlib.sha256(b"hello world").hexdigest()
    assert stored.sha256 == digest
    assert stored.size == 11
    assert stored.existed is False
    assert stored.path == folder / f"{digest[:SHA_CHARS]}-notes.txt"
    assert stored.path.read_bytes() == b"hello world"
    assert left_in(folder) == [stored.path.name]  # no temporary file left


def test_the_same_bytes_under_the_same_name_are_one_file(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    first = save(folder, "notes.txt", b"same")
    again = save(folder, "notes.txt", b"same")
    assert (again.path, again.existed) == (first.path, True)
    assert left_in(folder) == [first.path.name]


def test_different_bytes_never_overwrite_each_other_even_under_one_name(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    one = save(folder, "notes.txt", b"one")
    two = save(folder, "notes.txt", b"two")
    assert one.path != two.path
    assert (one.path.read_bytes(), two.path.read_bytes()) == (b"one", b"two")


def test_a_body_exactly_at_the_cap_is_kept_and_one_byte_more_is_413(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    assert save(folder, "a.txt", b"x" * 10, max_bytes=10).size == 10
    with pytest.raises(UploadError) as raised:
        save(folder, "b.txt", b"x" * 6, b"x" * 5, max_bytes=10)  # the cap is crossed mid-stream
    assert raised.value.status == 413
    assert len(left_in(folder)) == 1  # only a.txt: the failed upload left nothing


def test_an_empty_body_is_422_and_leaves_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    with pytest.raises(UploadError) as raised:
        save(folder, "a.txt")
    assert raised.value.status == 422
    assert left_in(folder) == []


def test_a_type_that_is_not_accepted_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    with pytest.raises(UploadError) as raised:
        save(folder, "program.exe", b"MZ")
    assert raised.value.status == 415
    assert not folder.exists()


def test_a_body_that_fails_half_way_leaves_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"

    async def broken() -> AsyncIterator[bytes]:
        yield b"first half"
        raise ConnectionResetError("the client went away")

    with pytest.raises(ConnectionResetError):
        asyncio.run(
            save_upload(
                broken(), filename="a.txt", directory=folder, max_bytes=1000, suffixes=SUFFIXES
            )
        )
    assert left_in(folder) == []


def test_a_cancelled_upload_still_removes_its_temporary_file(tmp_path: Path) -> None:
    """The client disconnects: the request task is cancelled while the body is still arriving."""
    folder = tmp_path / "uploads"

    async def never_finishes() -> AsyncIterator[bytes]:
        yield b"the start of a file"
        await asyncio.Event().wait()

    async def scenario() -> None:
        upload = save_upload(
            never_finishes(), filename="a.txt", directory=folder, max_bytes=1000, suffixes=SUFFIXES
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(upload, timeout=0.3)

    asyncio.run(scenario())
    assert left_in(folder) == []


# --- the route -------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(make_app(tmp_path)) as opened:
        yield opened


def upload(client: TestClient, name: str, data: bytes) -> Any:
    return client.post("/v1/kb/upload", params={"filename": name}, content=data)


def test_an_uploaded_file_is_saved_ingested_and_searchable_by_its_job(
    client: TestClient, tmp_path: Path
) -> None:
    data = b"Zebras live on the savanna.\n\nTheir stripes are unique.\n"
    response = upload(client, "zebras.txt", data)
    assert response.status_code == 202
    answer = response.json()
    digest = hashlib.sha256(data).hexdigest()
    assert answer["stored_as"] == f"{digest[:SHA_CHARS]}-zebras.txt"
    assert re.fullmatch(r"[0-9a-f]{16}-zebras\.txt", answer["stored_as"])
    assert (answer["sha256"], answer["size_bytes"], answer["existed"]) == (digest, len(data), False)
    job = wait_for(client, answer["job"]["id"])
    assert job["state"] == "done", job
    saved = tmp_path / "data" / "uploads" / answer["stored_as"]
    assert saved.read_bytes() == data
    (document,) = client.get("/v1/kb/documents").json()
    assert document["state"] == "ready"
    assert document["path"] == str(saved.resolve())


def test_uploading_the_same_file_again_changes_nothing(client: TestClient) -> None:
    first = upload(client, "zebras.txt", b"Zebras live on the savanna.\n").json()
    wait_for(client, first["job"]["id"])
    second = upload(client, "zebras.txt", b"Zebras live on the savanna.\n").json()
    assert second["existed"] is True
    job = wait_for(client, second["job"]["id"])
    assert job["result"]["files"][0]["status"] == "unchanged"
    assert "0 embeddings computed" in job["result"]["summary"]


def test_a_name_with_directories_is_stored_inside_the_uploads_folder(
    client: TestClient, tmp_path: Path
) -> None:
    answer = upload(client, "../../outside.txt", b"not outside at all\n").json()
    wait_for(client, answer["job"]["id"])
    assert answer["stored_as"].endswith("-outside.txt")
    assert "/" not in answer["stored_as"] and "\\" not in answer["stored_as"]
    assert (tmp_path / "data" / "uploads" / answer["stored_as"]).is_file()
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path.parent / "outside.txt").exists()


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [("program.exe", b"MZ", 415), ("empty.txt", b"", 422), ("noext", b"x", 415)],
)
def test_a_bad_upload_is_refused_and_queues_nothing(
    client: TestClient, tmp_path: Path, name: str, data: bytes, status: int
) -> None:
    assert upload(client, name, data).status_code == status
    assert client.get("/v1/jobs").json() == []
    uploads = tmp_path / "data" / "uploads"
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_the_file_name_is_required(client: TestClient) -> None:
    assert client.post("/v1/kb/upload", content=b"x").status_code == 422
    assert client.post("/v1/kb/upload", params={"filename": ""}, content=b"x").status_code == 422


def test_an_uploaded_file_can_also_be_ingested_by_its_path(
    client: TestClient, tmp_path: Path
) -> None:
    answer = upload(client, "zebras.txt", b"Zebras live on the savanna.\n").json()
    wait_for(client, answer["job"]["id"])
    saved = tmp_path / "data" / "uploads" / answer["stored_as"]
    again = client.post("/v1/kb/ingest", json={"paths": [str(saved)]})
    assert again.status_code == 202
    assert wait_for(client, again.json()["id"])["result"]["files"][0]["status"] == "unchanged"
