"""Tests for golden v0 (D-067): the 15 corpus documents and the 100 questions agree with each other.

Nothing here runs a model. The Word, PowerPoint and Excel files are built again from the script that
writes them, every document is read back through Geekbot's own loaders (what chat mode would index:
text for most, a schema card for tables), and each question is checked against that text:

- every source file exists, and the set is the plan's 50 / 15 / 15 / 20 with no validator problem;
- every must-include phrase occurs in the sources of its question, and every source of a
  multi-hop question contributes one (otherwise it would not be a multi-hop question);
- every document is a source of at least one question;
- every item is flagged as provisional.

These checks do not prove the questions are good. They prove they are consistent with the documents,
which is the mistake a hand-made answer key is most likely to contain.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from geekbot.evals.datasets import GoldenItem, assign_splits, validate_golden
from geekbot.evals.draft import longest_shared_run
from geekbot.evals.golden_md import import_markdown
from geekbot.ingest.loaders import load_file, load_paths

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "evals" / "corpus"
SCRIPT = REPO / "evals" / "corpus_src" / "build_office_files.py"
QUESTIONS = REPO / "evals" / "datasets" / "golden-questions.md"
PROVISIONAL = "v0 (Claude-authored, provisional)"

# The documents that are plain files in the repository (the rest come from the build script).
TEXT_FILES = {
    "analysis/usage-analysis.ipynb",
    "ops/hardware-setup.md",
    "ops/incident-log.csv",
    "ops/incident-runbook.md",
    "ops/log_cleanup.py",
    "ops/security-checklist.md",
    "policy/data-handling-standard.txt",
    "project/meeting-notes.txt",
    "staff/glossary.md",
    "staff/staff-faq.html",
}
EVERY_FORMAT = {".pdf", ".docx", ".pptx", ".html", ".md", ".txt", ".py", ".ipynb", ".csv", ".xlsx"}
COPY_LIMIT_WORDS = 7  # a question sharing this many words in a row with its source was copied
KICKOFF_SLIDES = 8  # a title slide and seven topic slides


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_office_files", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load_script()
EXPECTED = TEXT_FILES | set(BUILD.OFFICE_FILES) | {BUILD.MEMO_PDF_PATH}


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


class Built:
    """The office files built into a temporary folder, and a reader for any corpus document."""

    def __init__(self, corpus: Path, source: Path) -> None:
        self.corpus = corpus
        self.source = source
        self._texts: dict[str, str] = {}

    def path_of(self, relative: str) -> Path:
        if relative == BUILD.MEMO_PDF_PATH:  # the PDF is a hand export of the memo's Word file
            return self.source / BUILD.MEMO_DOCX_NAME
        if relative in BUILD.OFFICE_FILES:
            return self.corpus / relative
        return CORPUS / relative

    def text(self, relative: str) -> str:
        """What chat mode would index for this document, lower-cased with spaces collapsed."""
        if relative not in self._texts:
            document = load_file(self.path_of(relative))
            joined = "\n".join(segment.text for segment in document.segments)
            self._texts[relative] = normalized(joined)
        return self._texts[relative]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("golden-v0")
    corpus, source = root / "corpus", root / "source"
    source.mkdir()
    BUILD.build_all(corpus, source)
    return Built(corpus, source)


@pytest.fixture(scope="module")
def items() -> list[GoldenItem]:
    result = import_markdown(QUESTIONS.read_text(encoding="utf-8"))
    assert result.problems == ()
    return list(result.items)


def on_disk() -> set[str]:
    return {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*") if p.is_file()}


# --- the corpus ------------------------------------------------------------------------------


def test_the_corpus_is_fifteen_documents_in_every_format() -> None:
    assert len(EXPECTED) == 15
    assert {Path(name).suffix for name in EXPECTED} == EVERY_FORMAT


def test_no_stray_file_sits_in_the_corpus_folder() -> None:
    """Only documents go in the folder, because `geekbot ingest` reads everything in it."""
    present = on_disk()
    assert TEXT_FILES <= present, f"missing text documents: {sorted(TEXT_FILES - present)}"
    assert present <= EXPECTED, f"unexpected files: {sorted(present - EXPECTED)}"


def test_the_text_documents_load_without_a_problem() -> None:
    report = load_paths([CORPUS / name for name in sorted(TEXT_FILES)])
    assert report.failed == ()
    assert report.skipped == ()
    assert len(report.loaded) == len(TEXT_FILES)


def test_the_script_builds_every_office_file_and_each_one_loads(built: Built) -> None:
    for relative in BUILD.OFFICE_FILES:
        assert built.path_of(relative).is_file(), relative
        assert len(built.text(relative)) > 200, relative
    assert (built.source / BUILD.MEMO_DOCX_NAME).is_file()


@pytest.mark.parametrize(
    ("relative", "phrase"),
    [
        ("policy/ai-use-policy.docx", "21 march 2025"),
        ("policy/ai-use-policy.docx", "| green |"),  # the table survives conversion
        ("policy/onboarding-letter.docx", "45 minutes"),
        ("project/rollout-kickoff.pptx", "17 march: workstation ordered"),
        ("project/tools-and-costs.xlsx", "sheet budget"),
        ("project/vendor-memo.pdf", "£17,258"),
    ],
)
def test_the_converted_documents_keep_their_structure(
    built: Built, relative: str, phrase: str
) -> None:
    assert phrase in built.text(relative)


def test_the_deck_becomes_one_numbered_segment_per_slide(built: Built) -> None:
    """Citations can say "slide 3" only if the loader still finds MarkItDown's slide markers."""
    document = load_file(built.path_of("project/rollout-kickoff.pptx"))
    assert [segment.page for segment in document.segments] == list(range(1, KICKOFF_SLIDES + 1))


def test_the_office_files_in_the_repo_match_the_script(built: Built) -> None:
    """If the generated files were committed, they must say what the script says now."""
    present = [name for name in BUILD.OFFICE_FILES if (CORPUS / name).is_file()]
    if not present:
        pytest.skip("the office files have not been generated yet (run the build script)")
    for relative in present:
        document = load_file(CORPUS / relative)
        committed = normalized("\n".join(s.text for s in document.segments))
        assert committed == built.text(relative), f"{relative} is out of date: rebuild it"


def test_the_exported_pdf_says_what_the_memo_says() -> None:
    pdf = CORPUS / BUILD.MEMO_PDF_PATH
    if not pdf.is_file():
        pytest.skip("the memo's PDF has not been exported yet (Word: Save as PDF)")
    text = normalized("\n".join(s.text for s in load_file(pdf).segments))
    for phrase in ("17,258", "14,090", "corvid foods"):
        assert phrase in text


# --- the questions ---------------------------------------------------------------------------


def test_the_questions_are_the_plans_hundred_and_have_no_validator_problem(
    items: list[GoldenItem],
) -> None:
    report = validate_golden(items, corpus_files=EXPECTED)
    assert [str(problem) for problem in report.problems] == []
    assert dict(report.counts) == {
        "single_hop": 50,
        "multi_hop": 15,
        "multi_turn": 15,
        "unanswerable": 20,
    }


def test_every_item_is_flagged_as_provisional(items: list[GoldenItem]) -> None:
    unflagged = [item.id for item in items if not item.notes.startswith(PROVISIONAL)]
    assert unflagged == []


def test_every_must_include_phrase_occurs_in_the_sources_of_its_question(
    items: list[GoldenItem], built: Built
) -> None:
    missing: list[str] = []
    for item in items:
        texts = [built.text(source.path) for source in item.sources]
        for phrase in item.must_include:
            if not any(normalized(phrase) in text for text in texts):
                missing.append(f"{item.id}: {phrase!r}")
    assert missing == []


def test_every_source_of_a_multi_hop_question_is_needed(
    items: list[GoldenItem], built: Built
) -> None:
    """A source with none of the question's phrases would be a document the answer never uses."""
    unused: list[str] = []
    for item in items:
        if item.type != "multi_hop":
            continue
        for source in item.sources:
            text = built.text(source.path)
            if not any(normalized(phrase) in text for phrase in item.must_include):
                unused.append(f"{item.id}: {source.path}")
    assert unused == []


def test_every_document_is_the_source_of_at_least_one_question(items: list[GoldenItem]) -> None:
    used = {source.path for item in items for source in item.sources}
    assert EXPECTED - used == set()


def test_single_hop_answers_name_exactly_one_document(items: list[GoldenItem]) -> None:
    assert {len(item.sources) for item in items if item.type == "single_hop"} == {1}


def test_v0_splits_into_the_plans_forty_development_items(items: list[GoldenItem]) -> None:
    splits = assign_splits(items)
    assert sum(1 for side in splits.values() if side == "dev") == 40


def test_the_questions_are_not_copied_from_the_documents(
    items: list[GoldenItem], built: Built
) -> None:
    """Paraphrase check: no question repeats a long run of words from its own source."""
    copied: list[str] = []
    for item in items:
        for source in item.sources:
            raw_text = built.text(source.path)
            run = longest_shared_run(item.question, raw_text)
            if run >= COPY_LIMIT_WORDS:
                copied.append(f"{item.id}: {run} words in a row from {source.path}")
    assert copied == []
