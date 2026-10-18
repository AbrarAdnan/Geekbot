"""Tests for step S1.6: ``geekbot ingest`` and ``geekbot doctor --repair``, with a fake embedder.

The command opens a ``Workspace``, which would call Ollama. Here ``Workspace.open`` is patched to
hand in a deterministic fake embedder instead; everything else (SQLite, Chroma, the loaders, the
pipeline, the exit codes and the messages) is real.
"""

import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.index.embedder import EmbedderSpec
from geekbot.index.sqlite import Store
from geekbot.index.workspace import Workspace
from geekbot.llm.models import OllamaError

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

runner = CliRunner()
DIM = 8
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=DIM,
    with_header=True,
)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:DIM]]


class Holder:
    """What the patched ``Workspace.open`` hands in; a test can change the spec between runs."""

    def __init__(self) -> None:
        self.spec = SPEC
        self.fake = FakeEmbeddings()
        self.opens = 0
        self.error: Exception | None = None


@pytest.fixture
def holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Holder:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    state = Holder()
    real_open: Callable[..., Workspace] = Workspace.open.__func__  # type: ignore[attr-defined]

    def fake_open(
        cls: type[Workspace], settings: Any, *, reset_dense: bool = False, **_: Any
    ) -> Workspace:
        state.opens += 1
        if state.error is not None:
            raise state.error
        return real_open(
            cls, settings, reset_dense=reset_dense, embeddings=state.fake, spec=state.spec
        )

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    return state


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "zebra.md").write_text(
        "# Zebras\n\nZebras have stripes.\n\nThey live in herds.\n", encoding="utf-8"
    )
    (folder / "quartz.txt").write_text(
        "Quartz is a mineral.\n\nIt is very hard.\n", encoding="utf-8"
    )
    return folder


def run(*args: str) -> Any:
    return runner.invoke(app, list(args))


# --- ingest ----------------------------------------------------------------------------------


def test_ingest_without_a_path_is_an_error_not_an_index_of_the_current_folder(
    holder: Holder,
) -> None:
    result = run("ingest")
    assert result.exit_code == 1
    assert "give at least one file or folder" in result.output
    assert holder.opens == 0  # nothing was opened, nothing was touched


def test_a_path_that_does_not_exist_is_a_usage_error(holder: Holder, tmp_path: Path) -> None:
    result = run("ingest", str(tmp_path / "nope"))
    assert result.exit_code == 2
    assert holder.opens == 0


def test_a_first_ingest_lists_each_file_and_a_summary(holder: Holder, docs: Path) -> None:
    result = run("ingest", str(docs))
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output
    assert "quartz.txt" in result.output
    assert "zebra.md" in result.output
    assert "2 ingested, 0 unchanged, 0 failed, 0 skipped" in result.output


def test_a_second_run_computes_no_embeddings(holder: Holder, docs: Path) -> None:
    assert run("ingest", str(docs)).exit_code == 0
    result = run("ingest", str(docs))
    assert result.exit_code == 0
    assert "0 ingested, 2 unchanged, 0 failed, 0 skipped; 0 embeddings computed" in result.output


def test_a_bad_file_exits_1_but_the_others_are_ingested(holder: Holder, docs: Path) -> None:
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    result = run("ingest", str(docs))
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "broken.pdf" in result.output
    assert "2 ingested, 0 unchanged, 1 failed" in result.output


def test_unsupported_files_in_a_folder_are_counted_not_failed(holder: Holder, docs: Path) -> None:
    (docs / "photo.xyz").write_text("x", encoding="utf-8")
    result = run("ingest", str(docs))
    assert result.exit_code == 0
    assert "1 files of unsupported types were left out" in result.output


def test_workers_runs_the_same_ingest_in_processes(holder: Holder, docs: Path) -> None:
    result = run("ingest", str(docs), "--workers", "2")
    assert result.exit_code == 0, result.output
    assert "2 ingested" in result.output


def test_the_storage_lock_is_reported_not_a_traceback(
    holder: Holder, docs: Path, tmp_path: Path
) -> None:
    with Store(tmp_path / "data"):  # another process owns the storage
        result = run("ingest", str(docs))
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "Traceback" not in result.output


def test_ollama_being_down_is_one_clear_line(holder: Holder, docs: Path) -> None:
    holder.error = OllamaError("Ollama GET /api/tags failed: connection refused")
    result = run("ingest", str(docs))
    assert result.exit_code == 1
    assert "connection refused" in result.output


# --- re-index --------------------------------------------------------------------------------


def test_a_changed_embedder_needs_reindex_and_reindex_rebuilds(holder: Holder, docs: Path) -> None:
    assert run("ingest", str(docs)).exit_code == 0
    holder.spec = dataclasses.replace(SPEC, digest="re-pulled")

    refused = run("ingest", str(docs))
    assert refused.exit_code == 1
    assert "geekbot ingest --reindex" in refused.output

    rebuilt = run("ingest", "--reindex")  # no paths: only rebuild the vectors
    assert rebuilt.exit_code == 0, rebuilt.output
    assert "re-indexing" in rebuilt.output
    assert "2 rebuilt" in rebuilt.output

    after = run("ingest", str(docs))
    assert after.exit_code == 0
    assert "0 ingested, 2 unchanged" in after.output


# --- doctor --repair -------------------------------------------------------------------------


def test_doctor_repair_rebuilds_wiped_vectors(holder: Holder, docs: Path, tmp_path: Path) -> None:
    assert run("ingest", str(docs)).exit_code == 0
    with Workspace.open(_settings(tmp_path)) as workspace:  # lose the vectors
        workspace.dense.delete(sorted(workspace.dense.ids()))

    result = run("doctor", "--repair")
    assert result.exit_code == 0, result.output
    assert "before: 0 interrupted, 0 unfinished, 2 documents missing vectors" in result.output
    assert "2 rebuilt" in result.output
    assert "after: stores are consistent" in result.output


def test_doctor_repair_on_a_healthy_store_changes_nothing(holder: Holder, docs: Path) -> None:
    assert run("ingest", str(docs)).exit_code == 0
    result = run("doctor", "--repair")
    assert result.exit_code == 0
    assert "before: stores are consistent" in result.output
    assert "after: stores are consistent" in result.output


def test_doctor_repair_reports_a_held_lock(holder: Holder, tmp_path: Path) -> None:
    with Store(tmp_path / "data"):
        result = run("doctor", "--repair")
    assert result.exit_code == 1
    assert "error:" in result.output


def _settings(tmp_path: Path) -> Any:
    from geekbot.config import Settings

    return Settings(_env_file=None, data_dir=tmp_path / "data")
