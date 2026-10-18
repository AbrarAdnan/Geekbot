"""Tests for step S2.4: the ``geekbot eval`` commands (import, validate, generate, fetch).

The corpus is ingested for real (SQLite and Chroma) with a fake embedder, as in the ingest tests.
Ollama is replaced by a fake client for ``generate``, and the network by a fake ``fetch_beir``.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import httpx
import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.evals.beir import BeirError, FetchResult
from geekbot.evals.datasets import GoldenItem, Source, write_golden
from geekbot.evals.draft import DRAFT_SCHEMA, candidate_chunks
from geekbot.index.embedder import EmbedderSpec
from geekbot.index.sqlite import Store
from geekbot.index.workspace import Workspace
from geekbot.llm.models import OllamaError, SwapResult

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
SHEET = "Keep this sheet with the manual."


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:DIM]]


def long_text(topic: str) -> str:
    """One paragraph of about 400 characters: long enough to be worth asking a question about."""
    return (
        f"The {topic} unit resets to factory settings when you hold the recessed button on its "
        "back for ten seconds while it is powered on. The status light turns amber, then blinks "
        "green once the reset has finished. A reset removes the saved name and password, so write "
        "them down first. If the light stays red, unplug the unit for a minute and try again "
        f"before calling support. {SHEET}\n"
    )


def run(*args: str) -> Any:
    return runner.invoke(app, list(args))


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def ingested(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A corpus folder that has been ingested; returns the folder."""
    real_open: Callable[..., Workspace] = Workspace.open.__func__  # type: ignore[attr-defined]

    def fake_open(cls: type[Workspace], settings: Any, **_: Any) -> Workspace:
        return real_open(cls, settings, embeddings=FakeEmbeddings(), spec=SPEC)

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    corpus = tmp_path / "corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "alpha.md").write_text(long_text("alpha"), encoding="utf-8")
    (corpus / "beta.md").write_text(long_text("beta"), encoding="utf-8")
    (corpus / "sub" / "gamma.txt").write_text(long_text("gamma"), encoding="utf-8")
    (corpus / "short.txt").write_text("Too short to ask about.\n", encoding="utf-8")
    result = run("ingest", str(corpus))
    assert result.exit_code == 0, result.output
    return corpus


QUESTIONS = """\
### S-001
- **Question:** How long do I hold the button?
- **Where the answer is:** alpha.md (Reset)
- **Must include:** ten seconds

### M-001
- **Question:** What do the two units have in common?
- **Where the answer is:** alpha.md (Reset); sub/gamma.txt
- **Must include:** ten seconds; amber

### U-001
- **Question:** Does it work on Linux?
- **Why it can't be answered:** nothing about operating systems
"""


def write_questions(folder: Path, text: str = QUESTIONS) -> Path:
    path = folder / "questions.md"
    path.write_text(text, encoding="utf-8")
    return path


# --- import ----------------------------------------------------------------------------------


def test_import_writes_the_jsonl_and_prints_the_split(tmp_path: Path, ingested: Path) -> None:
    out = tmp_path / "out" / "golden.jsonl"
    result = run(
        "eval",
        "import",
        "--source",
        str(write_questions(tmp_path)),
        "--out",
        str(out),
        "--corpus",
        str(ingested),
    )
    assert result.exit_code == 0, result.output
    assert "read 3 items" in result.output
    assert "single_hop" in result.output
    assert f"wrote {out}" in result.output
    items = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [item["id"] for item in items] == ["S-001", "M-001", "U-001"]
    assert items[1]["sources"] == [
        {"path": "alpha.md", "locator": "Reset"},
        {"path": "sub/gamma.txt", "locator": ""},
    ]


def test_import_twice_gives_the_same_file(tmp_path: Path, ingested: Path) -> None:
    source, out = write_questions(tmp_path), tmp_path / "golden.jsonl"
    args = ["eval", "import", "--source", str(source), "--out", str(out), "--corpus", str(ingested)]
    assert run(*args).exit_code == 0
    first = out.read_bytes()
    assert run(*args).exit_code == 0
    assert out.read_bytes() == first


def test_import_writes_nothing_while_there_are_errors(tmp_path: Path, ingested: Path) -> None:
    bad = QUESTIONS.replace("alpha.md (Reset)", "missing.md (Reset)", 1)
    out = tmp_path / "golden.jsonl"
    out.write_text("keep me\n", encoding="utf-8")
    result = run(
        "eval",
        "import",
        "--source",
        str(write_questions(tmp_path, bad)),
        "--out",
        str(out),
        "--corpus",
        str(ingested),
    )
    assert result.exit_code == 1
    assert "source 'missing.md' is not a file in the corpus folder" in result.output
    assert "was not written" in result.output
    assert out.read_text(encoding="utf-8") == "keep me\n"  # the old file is untouched


def test_import_reports_unreadable_blocks_as_errors(tmp_path: Path) -> None:
    text = QUESTIONS + "\n### DRAFT-001 (single-hop)\n- **Question:** q?\n"
    result = run(
        "eval",
        "import",
        "--source",
        str(write_questions(tmp_path, text)),
        "--out",
        str(tmp_path / "golden.jsonl"),
        "--corpus",
        str(tmp_path / "no-corpus"),
    )
    assert result.exit_code == 1
    assert "DRAFT-001" in result.output
    assert not (tmp_path / "golden.jsonl").exists()


def test_import_without_a_corpus_folder_skips_the_file_check_and_says_so(tmp_path: Path) -> None:
    result = run(
        "eval",
        "import",
        "--source",
        str(write_questions(tmp_path)),
        "--out",
        str(tmp_path / "golden.jsonl"),
        "--corpus",
        str(tmp_path / "no-corpus"),
    )
    assert result.exit_code == 0, result.output
    assert "is not a folder; source file names are not checked" in result.output


def test_import_of_a_missing_source_is_an_error(tmp_path: Path) -> None:
    result = run("eval", "import", "--source", str(tmp_path / "nope.md"))
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_import_of_a_blank_template_finds_nothing_to_write(tmp_path: Path) -> None:
    out = tmp_path / "golden.jsonl"
    blank = (
        "### S-001\n- **Question:**\n- **Where the answer is:**\n- **Must include:**\n"
        "- **Notes:**\n\n### U-001\n- **Question:**\n- **Why it can't be answered:**\n"
        "- **Notes:**\n"
    )
    source = write_questions(tmp_path, blank)
    result = run("eval", "import", "--source", str(source), "--out", str(out))
    assert result.exit_code == 1
    assert "read 0 items" in result.output
    assert "2 template blocks left blank" in result.output
    assert "the golden set is empty" in result.output
    assert not out.exists()


# --- validate --------------------------------------------------------------------------------


@pytest.fixture
def golden(tmp_path: Path, ingested: Path) -> Path:
    out = tmp_path / "golden.jsonl"
    result = run(
        "eval",
        "import",
        "--source",
        str(write_questions(tmp_path)),
        "--out",
        str(out),
        "--corpus",
        str(ingested),
    )
    assert result.exit_code == 0, result.output
    return out


def test_validate_accepts_a_good_file_and_shows_the_split(golden: Path, ingested: Path) -> None:
    result = run("eval", "validate", "--golden", str(golden), "--corpus", str(ingested))
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("ok")
    assert "total" in result.output


def test_validate_names_a_line_that_is_not_golden_data(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "S-001"\n', encoding="utf-8")
    result = run("eval", "validate", "--golden", str(path))
    assert result.exit_code == 1
    assert "line 1: not valid JSON" in result.output


def test_validate_of_a_missing_file_says_to_run_import(tmp_path: Path) -> None:
    result = run("eval", "validate", "--golden", str(tmp_path / "nope.jsonl"))
    assert result.exit_code == 1
    assert "geekbot eval import" in result.output


def test_validate_checks_the_sources_against_the_corpus_folder(
    golden: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-corpus"
    empty.mkdir()
    result = run("eval", "validate", "--golden", str(golden), "--corpus", str(empty))
    assert result.exit_code == 1
    assert "is not a file in the corpus folder" in result.output


def test_validate_check_index_passes_when_every_source_is_a_ready_document(
    golden: Path, ingested: Path
) -> None:
    result = run(
        "eval", "validate", "--golden", str(golden), "--corpus", str(ingested), "--check-index"
    )
    assert result.exit_code == 0, result.output


def test_validate_check_index_names_a_source_that_was_never_ingested(
    tmp_path: Path, ingested: Path
) -> None:
    (ingested / "late.md").write_text(long_text("late"), encoding="utf-8")  # added after the ingest
    item = GoldenItem(
        id="S-001",
        type="single_hop",
        question="Q?",
        sources=(Source("late.md", "x"),),
        must_include=("p",),
    )
    path = tmp_path / "late.jsonl"
    write_golden(path, [item])
    plain = run("eval", "validate", "--golden", str(path), "--corpus", str(ingested))
    assert plain.exit_code == 0, plain.output  # the file exists, so the plain check is happy
    result = run(
        "eval", "validate", "--golden", str(path), "--corpus", str(ingested), "--check-index"
    )
    assert result.exit_code == 1
    assert "S-001: late.md is not a ready document in the index" in result.output


def test_validate_check_index_without_an_index_says_to_ingest(
    golden: Path, ingested: Path, data_dir: Path
) -> None:
    for leftover in data_dir.glob("geekbot.db*"):
        leftover.unlink()
    result = run(
        "eval", "validate", "--golden", str(golden), "--corpus", str(ingested), "--check-index"
    )
    assert result.exit_code == 1
    assert "nothing is indexed yet" in result.output


# --- generate --------------------------------------------------------------------------------


class FakeOllama:
    """Records what the command asked of Ollama and answers from a list (or a default)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.swaps: list[str] = []
        self.replies: list[str] = []
        self.error: Exception | None = None

    def next_reply(self) -> str:
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        number = len(self.calls)
        return json.dumps({"question": f"Draft question {number}?", "must_include": [SHEET[:-1]]})


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    state = FakeOllama()

    class Client:
        def __init__(self, host: str) -> None:
            state.calls.append({"host": host})

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def generate(self, model: str, prompt: str, **options: Any) -> str:
            state.calls.append({"model": model, "prompt": prompt, **options})
            return state.next_reply()

    def swap(_client: object, _settings: object, mode: str, **_: object) -> SwapResult:
        state.swaps.append(mode)
        return SwapResult(loaded="m", unloaded=(), unload_s=0.0, warm_up_s=1.5)

    monkeypatch.setattr("geekbot.cli.OllamaClient", Client)
    monkeypatch.setattr("geekbot.llm.models.swap_to_mode", swap)
    return state


def generated(state: FakeOllama) -> list[dict[str, Any]]:
    return [call for call in state.calls if "prompt" in call]


def generate(corpus: Path, out: Path, *options: str) -> Any:
    return run("eval", "generate", "--corpus", str(corpus), "--out", str(out), *options)


def test_generate_drafts_from_the_stored_passages(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    out = tmp_path / "drafts" / "draft.md"
    result = generate(ingested, out, "--count", "2", "--unanswerable", "1")
    assert result.exit_code == 0, result.output
    assert fake_ollama.swaps == ["chat"]  # the chat model alone on the GPU (R-4)
    calls = generated(fake_ollama)
    assert len(calls) == 3
    assert all(call["model"] == "gemma3:4b" for call in calls)  # the proto profile's chat model
    assert all(call["schema"] == DRAFT_SCHEMA for call in calls)
    assert {(c["num_ctx"], c["keep_alive"], c["seed"]) for c in calls} == {(8192, "30m", 0)}
    assert "does NOT answer" in calls[2]["system"]  # the last one is the unanswerable draft
    assert "does NOT answer" not in calls[0]["system"]
    text = out.read_text(encoding="utf-8")
    assert "### DRAFT-001 (single-hop)" in text
    assert "### DRAFT-003 (unanswerable)" in text
    assert "`gemma3:4b` (profile `proto`)" in text
    assert "wrote 3 drafts" in result.output


def test_generate_asks_about_each_document_and_skips_the_stub(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    result = generate(ingested, tmp_path / "draft.md", "--count", "3", "--unanswerable", "0")
    assert result.exit_code == 0, result.output
    prompts = [call["prompt"] for call in generated(fake_ollama)]
    for topic in ("alpha", "beta", "gamma"):
        assert sum(f"The {topic} unit" in prompt for prompt in prompts) == 1
    assert not any("Too short" in prompt for prompt in prompts)


def test_generate_asks_for_no_more_passages_than_there_are(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    result = generate(ingested, tmp_path / "draft.md", "--count", "10", "--unanswerable", "0")
    assert result.exit_code == 0, result.output
    assert "only 3 usable passages" in result.output
    assert len(generated(fake_ollama)) == 3


def test_generate_counts_unusable_replies(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    fake_ollama.replies = ["not json at all"]
    result = generate(ingested, tmp_path / "draft.md", "--count", "3", "--unanswerable", "0")
    assert result.exit_code == 0, result.output
    assert "wrote 2 drafts" in result.output
    assert "1 replies were unusable" in result.output


def test_generate_does_not_overwrite_a_draft_file_unless_forced(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    out = tmp_path / "draft.md"
    out.write_text("my notes\n", encoding="utf-8")
    one = ("--count", "1", "--unanswerable", "0")
    refused = generate(ingested, out, *one)
    assert refused.exit_code == 1
    assert "exists" in refused.output
    assert out.read_text(encoding="utf-8") == "my notes\n"
    assert generated(fake_ollama) == []  # no model was touched
    forced = generate(ingested, out, *one, "--force")
    assert forced.exit_code == 0, forced.output
    assert "### DRAFT-001" in out.read_text(encoding="utf-8")


def test_generate_with_nothing_to_draft_is_an_error(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    result = generate(ingested, tmp_path / "d.md", "--count", "0", "--unanswerable", "0")
    assert result.exit_code == 1
    assert "nothing to draft" in result.output


def test_generate_without_an_index_says_to_ingest(
    tmp_path: Path, data_dir: Path, fake_ollama: FakeOllama
) -> None:
    result = run("eval", "generate", "--out", str(tmp_path / "draft.md"))
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output
    assert fake_ollama.calls == []


def test_generate_in_a_folder_with_no_ingested_documents_says_so(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    elsewhere = tmp_path / "other-corpus"
    result = run("eval", "generate", "--corpus", str(elsewhere), "--out", str(tmp_path / "d.md"))
    assert result.exit_code == 1
    assert "no usable passages" in result.output
    assert generated(fake_ollama) == []


def test_generate_stops_with_one_line_when_ollama_fails(
    tmp_path: Path, ingested: Path, fake_ollama: FakeOllama
) -> None:
    fake_ollama.error = OllamaError("Ollama POST /api/generate failed: connection refused")
    out = tmp_path / "draft.md"
    result = generate(ingested, out, "--count", "1", "--unanswerable", "0")
    assert result.exit_code == 1
    assert "connection refused" in result.output
    assert not out.exists()  # no half-written file


def test_the_candidate_chunks_are_the_long_ones_in_the_corpus_folder(
    ingested: Path, data_dir: Path
) -> None:
    with Store(data_dir) as store:
        groups = candidate_chunks(store, ingested)
        elsewhere = candidate_chunks(store, ingested.parent / "other-corpus")
    assert [group[0].text.split()[1] for group in groups] == ["alpha", "beta", "gamma"]
    assert all(len(group) == 1 for group in groups)  # the stub is too short to ask about
    assert elsewhere == []


# --- fetch -----------------------------------------------------------------------------------


def write_beir(root: Path, name: str) -> Path:
    folder = root / name
    (folder / "qrels").mkdir(parents=True)
    row = '{"_id": "1", "title": "t", "text": "x"}\n'
    (folder / "corpus.jsonl").write_text(row, encoding="utf-8")
    (folder / "queries.jsonl").write_text('{"_id": "q1", "text": "y"}\n', encoding="utf-8")
    for split in ("train", "dev", "test"):
        (folder / "qrels" / f"{split}.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\t1\t1\n", encoding="utf-8"
        )
    return folder


def test_fetch_prints_what_each_tuning_and_headline_split_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "beir"
    seen: list[tuple[str, Path, bool]] = []

    def fake_fetch(
        name: str, destination: Path, *, client: httpx.Client, force: bool = False
    ) -> FetchResult:
        seen.append((name, destination, force))
        folder = write_beir(destination, name)
        return FetchResult(name, folder, name == "scifact", "ab" * 32, 2_700_000, True)

    monkeypatch.setattr("geekbot.evals.beir.fetch_beir", fake_fetch)
    result = run("eval", "fetch", "--dest", str(root), "--force")
    assert result.exit_code == 0, result.output
    assert seen == [("scifact", root, True), ("nfcorpus", root, True)]
    assert "scifact: downloaded 2.7 MB, sha256 " + "ab" * 32 in result.output
    assert "(first download: recorded as the reference)" in result.output
    assert "nfcorpus: already unpacked in" in result.output
    assert "scifact train: 1 documents, 1 queries with judgements, 1 judgements" in result.output
    assert "scifact test: " in result.output
    assert "nfcorpus dev: " in result.output  # NFCorpus tunes on its dev part
    assert "nfcorpus train" not in result.output


def test_fetch_without_dest_keeps_the_data_under_the_data_folder(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destinations: list[Path] = []

    def fake_fetch(name: str, destination: Path, **_: Any) -> FetchResult:
        destinations.append(destination)
        return FetchResult(name, write_beir(destination, name), False, None, 0, False)

    monkeypatch.setattr("geekbot.evals.beir.fetch_beir", fake_fetch)
    assert run("eval", "fetch", "scifact").exit_code == 0
    assert destinations == [data_dir / "beir"]


def test_fetch_reports_a_dataset_error_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(name: str, destination: Path, **_: Any) -> FetchResult:
        raise BeirError("scifact: the download's SHA-256 differs from the one recorded")

    monkeypatch.setattr("geekbot.evals.beir.fetch_beir", broken)
    result = run("eval", "fetch", "scifact", "--dest", str(tmp_path))
    assert result.exit_code == 1
    assert "error: scifact: the download's SHA-256 differs" in result.output


def test_fetch_refuses_a_dataset_it_does_not_know(tmp_path: Path) -> None:
    result = run("eval", "fetch", "msmarco", "--dest", str(tmp_path))
    assert result.exit_code == 1
    assert "unknown dataset(s) msmarco" in result.output


def test_the_eval_group_lists_its_commands() -> None:
    result = run("eval", "--help")
    assert result.exit_code == 0
    for command in ("import", "validate", "generate", "fetch"):
        assert command in result.output
