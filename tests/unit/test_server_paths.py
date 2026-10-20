"""Tests for step S4.3: the path allowlist as the ingest route enforces it.

``test_path_policy.py`` tests the policy on its own; these tests send real requests: what status each
kind of refusal gets, that a refused request queues nothing, and that a link inside an allowed folder
cannot be used to ingest a file outside it.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from serverkit import make_app, wait_for, write_note

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """An app whose one allowed folder is ``tmp_path`` (its data folder, inside, is forbidden)."""
    with TestClient(make_app(tmp_path)) as opened:
        yield opened


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder with a file in it that is not under the allowed folder."""
    folder = tmp_path_factory.mktemp("outside")
    (folder / "secret.txt").write_text("the secret", encoding="utf-8")
    return folder


def ingest(client: TestClient, *paths: str) -> tuple[int, str]:
    response = client.post("/v1/kb/ingest", json={"paths": list(paths)})
    detail = response.json().get("detail", "") if response.status_code >= 400 else ""
    return response.status_code, str(detail)


def test_a_file_outside_the_allowed_folders_is_403_and_queues_nothing(
    client: TestClient, outside: Path
) -> None:
    status, detail = ingest(client, str(outside / "secret.txt"))
    assert status == 403
    assert "GEEKBOT_ALLOWED_ROOTS" in detail
    assert client.get("/v1/jobs").json() == []
    assert client.get("/v1/kb/documents").json() == []


def test_one_forbidden_path_refuses_the_whole_request(
    client: TestClient, tmp_path: Path, outside: Path
) -> None:
    note = write_note(tmp_path)
    status, _ = ingest(client, str(note), str(outside / "secret.txt"))
    assert status == 403
    assert client.get("/v1/jobs").json() == []


def test_outside_the_folders_a_missing_path_is_refused_like_an_existing_one(
    client: TestClient, outside: Path
) -> None:
    """The answer must not tell a caller what exists on the disk outside the allowed folders."""
    assert ingest(client, str(outside / "secret.txt"))[0] == 403
    assert ingest(client, str(outside / "nothing-here.txt"))[0] == 403


def test_dot_dot_out_of_the_allowed_folder_is_403(
    client: TestClient, tmp_path: Path, outside: Path
) -> None:
    sneaky = str(tmp_path / ".." / outside.name / "secret.txt")
    assert ingest(client, sneaky)[0] == 403


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "relative/path.txt",
        "notes.txt",
        "\\\\server\\share\\x.txt",
        "//server/share/x.txt",
        "/\\server\\share\\x.txt",
    ],
    ids=["empty", "blank", "relative", "bare-name", "unc", "unc-slashes", "unc-mixed"],
)
def test_text_that_is_not_a_full_local_path_is_422(client: TestClient, raw: str) -> None:
    assert ingest(client, raw)[0] == 422
    assert client.get("/v1/jobs").json() == []


def test_the_data_folder_cannot_be_ingested_even_though_it_is_inside_the_allowed_one(
    client: TestClient, tmp_path: Path
) -> None:
    status, detail = ingest(client, str(tmp_path / "data"))
    assert status == 403
    assert "data folder" in detail


def test_a_missing_path_inside_the_allowed_folder_is_422_and_names_it(
    client: TestClient, tmp_path: Path
) -> None:
    status, detail = ingest(client, str(tmp_path / "nope.txt"))
    assert status == 422
    assert "nope.txt" in detail


def test_a_normal_folder_ingest_refuses_nothing(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    write_note(folder)
    job = wait_for(client, client.post("/v1/kb/ingest", json={"paths": [str(folder)]}).json()["id"])
    assert job["state"] == "done"
    assert (job["result"]["refused"], job["result"]["refused_count"]) == ([], 0)


def test_a_link_in_an_allowed_folder_to_a_file_outside_it_is_not_ingested(
    client: TestClient, tmp_path: Path, outside: Path
) -> None:
    """The folder passes the check; the link inside it must not smuggle a file out of the sandbox."""
    folder = tmp_path / "docs"
    folder.mkdir()
    write_note(folder)
    try:
        (folder / "innocent.txt").symlink_to(outside / "secret.txt")
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks (Windows: needs Developer Mode)")
    job = wait_for(client, client.post("/v1/kb/ingest", json={"paths": [str(folder)]}).json()["id"])
    assert job["state"] == "done", job
    assert job["result"]["refused_count"] == 1
    assert [Path(path).name for path in job["result"]["refused"]] == ["innocent.txt"]
    paths = [document["path"] for document in client.get("/v1/kb/documents").json()]
    assert [Path(path).name for path in paths] == ["zebras.txt"]
    assert not any("secret" in path for path in paths)
