"""Tests for step S1.2: the chunker (loaded text -> citable chunks).

Two kinds of test live here. The examples pin down behaviour a reader can check by eye: where a
heading, a page or a ``def`` cuts, what the overlap is. The property tests at the end throw
arbitrary text at the chunker (headings, code fences, blank lines, CJK, odd Unicode, tiny budgets)
and check what must hold for every input; the first of them, "no text is lost", is the ✍️ piece
of this step (written by Claude in prototyping mode, D-047).
"""

from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.ingest.chunking import (
    ChunkDraft,
    chunk_document,
    contextual_header,
    estimate_tokens,
)
from geekbot.ingest.loaders import load_file
from geekbot.ingest.loaders.model import SEGMENT_SEPARATOR, DocKind, LoadedDocument, Segment

SALES_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "sales_small.csv"


def make_doc(
    kind: DocKind,
    *segments: str | Segment,
    language: str | None = None,
    title: str = "notes.md",
) -> LoadedDocument:
    parts = tuple(Segment(item) if isinstance(item, str) else item for item in segments)
    return LoadedDocument(
        path=Path(title), kind=kind, title=title, segments=parts, language=language
    )


def texts_of(chunks: list[ChunkDraft]) -> list[str]:
    return [chunk.text for chunk in chunks]


# --- the token estimate ----------------------------------------------------------------------


def test_estimate_tokens_counts_words_punctuation_and_cjk() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n") == 0
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("a, b.") == 4
    assert estimate_tokens("internationalization") == 4  # 20 letters, one token per five
    assert estimate_tokens("\u6f22\u5b57\u3002") == 3  # two ideographs and a full stop
    assert estimate_tokens("abc\u6f22") == 2  # a CJK character ends a Latin word


# --- one chunk, or none ----------------------------------------------------------------------


def test_a_short_document_is_one_chunk_with_exact_positions() -> None:
    doc = make_doc("text", "The quick brown fox.")
    (chunk,) = chunk_document(doc)
    assert chunk.text == "The quick brown fox."
    assert (chunk.char_start, chunk.char_end) == (0, 20)
    assert chunk.page is None
    assert chunk.section_path == ()
    assert chunk.header == "notes.md"
    assert chunk.ordinal == 0


@pytest.mark.parametrize("segments", [(), ("  \n\n  ",), ("", "\n")])
def test_a_document_without_text_has_no_chunks(segments: tuple[str, ...]) -> None:
    assert chunk_document(make_doc("text", *segments)) == []


def test_the_indentation_of_the_first_line_is_kept() -> None:
    doc = make_doc("code", "    indented()\n", language="python")
    assert texts_of(chunk_document(doc)) == ["    indented()"]


@pytest.mark.parametrize(
    ("chunk_tokens", "overlap_tokens", "message"),
    [(0, 0, "chunk_tokens"), (10, 10, "overlap_tokens"), (10, -1, "overlap_tokens")],
)
def test_budgets_that_make_no_sense_are_rejected(
    chunk_tokens: int, overlap_tokens: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_document(
            make_doc("text", "some text"), chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        )


# --- headings --------------------------------------------------------------------------------


def test_each_section_is_a_chunk_that_knows_its_heading_path() -> None:
    text = "# Guide\n\nIntro text.\n\n## Install\n\nRun it.\n\n### Windows\n\nUse conda."
    chunks = chunk_document(make_doc("markdown", text))
    assert texts_of(chunks) == [
        "# Guide\n\nIntro text.",
        "## Install\n\nRun it.",
        "### Windows\n\nUse conda.",
    ]
    assert [chunk.section_path for chunk in chunks] == [
        ("Guide",),
        ("Guide", "Install"),
        ("Guide", "Install", "Windows"),
    ]
    assert chunks[2].header == "notes.md > Guide > Install > Windows"


def test_a_heading_with_nothing_under_it_joins_the_section_after_it() -> None:
    chunks = chunk_document(make_doc("markdown", "# Title\n## Sub\ntext"))
    assert texts_of(chunks) == ["# Title\n## Sub\ntext"]
    assert chunks[0].section_path == ("Title", "Sub")


def test_text_before_the_first_heading_has_an_empty_path() -> None:
    chunks = chunk_document(make_doc("markdown", "Intro line.\n\n# A\nbody"))
    assert texts_of(chunks) == ["Intro line.", "# A\nbody"]
    assert [chunk.section_path for chunk in chunks] == [(), ("A",)]


def test_a_hash_line_inside_a_code_fence_is_not_a_heading() -> None:
    text = "# A\n```\n# not a heading\n```\nbody"
    (chunk,) = chunk_document(make_doc("markdown", text))
    assert chunk.section_path == ("A",)
    assert chunk.text == text


@pytest.mark.parametrize(
    ("text", "title"),
    [("## Title ##\nx", "Title"), ("# C#\ntext", "C#"), ("#  spaced   \nx", "spaced")],
)
def test_heading_titles_drop_the_closing_hashes_but_keep_a_trailing_hash_in_the_word(
    text: str, title: str
) -> None:
    (chunk,) = chunk_document(make_doc("markdown", text))
    assert chunk.section_path == (title,)


@pytest.mark.parametrize(
    ("kind", "expected_chunks"),
    [("markdown", 2), ("docx", 2), ("notebook", 2), ("text", 1), ("code", 1), ("pdf", 1)],
)
def test_only_markdown_like_kinds_are_read_for_headings(
    kind: DocKind, expected_chunks: int
) -> None:
    """A "# ..." line in a config file or a script is a comment; in a Word file it is a heading."""
    chunks = chunk_document(make_doc(kind, "# one\nbody\n# two\nbody"))
    assert len(chunks) == expected_chunks


def test_a_long_section_keeps_its_path_in_every_chunk() -> None:
    steps = " ".join(f"Step {i} is done." for i in range(40))
    chunks = chunk_document(
        make_doc("markdown", f"## Install\n\n{steps}"), chunk_tokens=30, overlap_tokens=0
    )
    assert len(chunks) > 1
    assert all(chunk.section_path == ("Install",) for chunk in chunks)
    assert chunks[0].text.startswith("## Install")
    assert all(chunk.text.startswith("Step") for chunk in chunks[1:])


# --- pages -----------------------------------------------------------------------------------


def test_chunks_keep_their_page_and_point_into_the_joined_text() -> None:
    doc = make_doc("pdf", Segment("First page text.", 1), Segment("Second page text.", 2))
    chunks = chunk_document(doc)
    assert texts_of(chunks) == ["First page text.", "Second page text."]
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[1].char_start == len("First page text.") + len(SEGMENT_SEPARATOR)
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)


# --- size and overlap ------------------------------------------------------------------------


def test_a_long_text_is_cut_at_sentences_with_a_sentence_of_overlap() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(60)]  # 8 tokens each
    doc = make_doc("text", " ".join(sentences))
    chunks = chunk_document(doc, chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) > 5
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)
    assert all(estimate_tokens(chunk.text) <= 50 for chunk in chunks)
    for before, after in pairwise(chunks):
        shared = doc.text[after.char_start : before.char_end]
        assert shared  # the chunks overlap...
        assert estimate_tokens(shared) <= 10  # ...by no more than the budget
        assert before.text.endswith(shared)
        assert after.text.startswith(shared)
        assert before.text.endswith(".")  # and every cut falls on a sentence end


def test_without_an_overlap_budget_the_chunks_tile_the_text() -> None:
    words = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_document(make_doc("text", words), chunk_tokens=20, overlap_tokens=0)
    assert len(chunks) > 1
    assert " ".join(texts_of(chunks)) == words


def test_text_without_any_boundary_is_cut_into_runs_of_characters() -> None:
    blob = "a" * 1000  # one 200-token word
    chunks = chunk_document(make_doc("text", blob), chunk_tokens=20, overlap_tokens=0)
    assert all(estimate_tokens(chunk.text) <= 20 for chunk in chunks)
    assert "".join(texts_of(chunks)) == blob


def test_unspaced_cjk_text_counts_one_token_per_character() -> None:
    doc = make_doc("text", "\u6f22" * 100)
    chunks = chunk_document(doc, chunk_tokens=30, overlap_tokens=5)
    assert [len(chunk.text) for chunk in chunks] == [30, 30, 30, 10]


# --- code ------------------------------------------------------------------------------------


def test_python_is_cut_before_a_def_not_in_the_middle_of_one() -> None:
    code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    chunks = chunk_document(
        make_doc("code", code, language="python"), chunk_tokens=10, overlap_tokens=0
    )
    assert texts_of(chunks) == ["def a():\n    return 1", "def b():\n    return 2"]


def test_a_decorator_stays_with_its_function() -> None:
    code = "x = 1\n\n@app.get\ndef a():\n    return 1\n"
    chunks = chunk_document(
        make_doc("code", code, language="python"), chunk_tokens=12, overlap_tokens=0
    )
    assert texts_of(chunks) == ["x = 1", "@app.get\ndef a():\n    return 1"]


def test_a_big_python_class_is_cut_between_its_methods() -> None:
    methods = "".join(f"    def m{i}(self):\n        return {i}\n\n" for i in range(20))
    chunks = chunk_document(
        make_doc("code", f"class C:\n{methods}", language="python"),
        chunk_tokens=40,
        overlap_tokens=0,
    )
    assert len(chunks) > 2
    assert chunks[0].text.startswith("class C:")
    assert all(chunk.text.startswith("    def m") for chunk in chunks[1:])
    for chunk in chunks:
        assert chunk.text.splitlines()[-1].strip().startswith("return")


_SNIPPETS = {
    None: "alpha\nbeta\n\ngamma\ndelta\nepsilon\n",
    "bash": "echo one\necho two\n\nfunction f() {\n  echo three\n}\n",
    "go": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println(1)\n}\n',
    "js": "import x from 'x'\n\nexport function f() {\n  return 1\n}\n\nconst y = 2\n",
    "ruby": "class A\n  def a\n    1\n  end\nend\n\ndef b\n  2\nend\n",
    "rust": 'use std::io;\n\npub fn main() {\n    println!("hi");\n}\n\nstruct S;\n',
    "ts": "interface A { x: number }\n\nexport const f = (a: A) => a.x\n",
}


@pytest.mark.parametrize("language", list(_SNIPPETS))
def test_every_language_rule_chunks_a_snippet_into_exact_slices(language: str | None) -> None:
    doc = make_doc("code", _SNIPPETS[language], language=language)
    chunks = chunk_document(doc, chunk_tokens=4, overlap_tokens=1)  # every snippet is longer
    assert len(chunks) > 1
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)
    assert all(estimate_tokens(chunk.text) <= 4 for chunk in chunks)


# --- the contextual header -------------------------------------------------------------------


def test_the_header_is_the_title_then_the_section_path() -> None:
    assert contextual_header("guide.md", ()) == "guide.md"
    assert contextual_header("guide.md", ("Install", "Windows")) == "guide.md > Install > Windows"


def test_the_embedding_text_adds_the_header_only_when_asked() -> None:
    (chunk,) = chunk_document(make_doc("markdown", "# A\nbody"))
    assert chunk.embedding_text(with_header=False) == "# A\nbody"
    assert chunk.embedding_text(with_header=True) == "notes.md > A\n\n# A\nbody"


# --- with the real loaders -------------------------------------------------------------------


def test_a_loaded_markdown_file_is_chunked_by_its_headings(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    text = "# Guide\n\nHello.\n\n## Install\n\nRun it.\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    doc = load_file(path)
    chunks = chunk_document(doc)
    assert [chunk.section_path for chunk in chunks] == [("Guide",), ("Guide", "Install")]
    assert chunks[1].header == "guide.md > Guide > Install"
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_a_loaded_python_file_is_chunked_as_code(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    path.write_text("import os\n\n\ndef main():\n    return os.name\n", encoding="utf-8")
    doc = load_file(path)
    assert texts_of(chunk_document(doc)) == [doc.text]


def test_a_table_card_is_chunked_under_its_title_heading() -> None:
    chunks = chunk_document(load_file(SALES_CSV))
    assert chunks[0].text.startswith("# Table:")
    assert all(chunk.section_path == ("Table: sales_small.csv",) for chunk in chunks)
    assert chunks[0].header == "sales_small.csv > Table: sales_small.csv"


# --- properties: what must hold for any text -------------------------------------------------

_FRAGMENTS = [
    "alpha",
    "beta",
    "gamma-delta",
    "x",
    "1234567890",
    "a_long_identifier_name",
    "Don't",
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\n\n\n",
    "\r\n",
    ". ",
    "! ",
    "? ",
    ", ",
    "(",
    ")",
    "# Title\n",
    "## Part\n",
    "###### Deep\n",
    "#\n",
    "# C#\n",
    "## Closed ##\n",
    "```\n",
    "```python\n",
    "~~~\n",
    "def f():\n",
    "    return 1\n",
    "@decorator\n",
    "class C:\n",
    "func main() {\n",
    "}\n",
    "\u6f22\u5b57",
    "\u3002",
    "e\u0301",
]
_texts = st.lists(st.one_of(st.sampled_from(_FRAGMENTS), st.text(max_size=6)), max_size=60).map(
    "".join
)
_segments = st.lists(
    st.builds(Segment, text=_texts, page=st.none() | st.integers(min_value=1, max_value=99)),
    min_size=1,
    max_size=3,
)


@st.composite
def cases(draw: st.DrawFn) -> tuple[LoadedDocument, int, int]:
    """A document of any kind and language, and a token budget with its overlap."""
    doc = LoadedDocument(
        path=Path("doc"),
        kind=draw(st.sampled_from(["markdown", "text", "code", "pdf", "docx", "notebook"])),
        title="doc",
        segments=tuple(draw(_segments)),
        language=draw(st.sampled_from([None, "python", "ruby", "js", "go", "rust", "bash"])),
    )
    size = draw(st.integers(min_value=1, max_value=60))
    overlap = draw(st.integers(min_value=0, max_value=size - 1))
    return doc, size, overlap


@settings(max_examples=300, deadline=None)
@given(case=cases())
def test_no_text_is_lost_and_every_chunk_is_an_exact_slice(
    case: tuple[LoadedDocument, int, int],
) -> None:
    """The ✍️ property: whatever the chunker does, nothing a reader could search for goes missing.

    Every chunk is exactly the text at its offsets, and every character that is not whitespace
    sits inside at least one chunk. Only blank lines and trailing whitespace may fall between
    chunks, which is what the chunker trims on purpose.
    """
    doc, size, overlap = case
    chunks = chunk_document(doc, chunk_tokens=size, overlap_tokens=overlap)
    covered = [False] * len(doc.text)
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        for position in range(chunk.char_start, chunk.char_end):
            covered[position] = True
    lost = [(i, ch) for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
    assert lost == []


@settings(max_examples=300, deadline=None)
@given(case=cases())
def test_chunks_fit_the_budget_and_come_in_document_order(
    case: tuple[LoadedDocument, int, int],
) -> None:
    doc, size, overlap = case
    chunks = chunk_document(doc, chunk_tokens=size, overlap_tokens=overlap)
    for index, chunk in enumerate(chunks):
        assert chunk.ordinal == index
        assert chunk.text.strip()
        assert estimate_tokens(chunk.text) <= size
    for before, after in pairwise(chunks):
        assert before.char_start <= after.char_start
        assert before.char_end < after.char_end
        if after.char_start < before.char_end:
            assert estimate_tokens(doc.text[after.char_start : before.char_end]) <= overlap


@settings(max_examples=300, deadline=None)
@given(case=cases())
def test_a_chunk_never_crosses_a_page(case: tuple[LoadedDocument, int, int]) -> None:
    doc, size, overlap = case
    chunks = chunk_document(doc, chunk_tokens=size, overlap_tokens=overlap)
    bounds: list[tuple[int, int, int | None]] = []
    position = 0
    for segment in doc.segments:
        bounds.append((position, position + len(segment.text), segment.page))
        position += len(segment.text) + len(SEGMENT_SEPARATOR)
    for chunk in chunks:
        owners = [
            page
            for start, end, page in bounds
            if start <= chunk.char_start and chunk.char_end <= end
        ]
        assert owners == [chunk.page]


@settings(max_examples=300, deadline=None)
@given(a=st.text(max_size=40), b=st.text(max_size=40))
def test_the_token_estimate_never_grows_when_a_text_is_cut_in_two(a: str, b: str) -> None:
    """The merge step adds up piece counts, which is safe only if this holds."""
    assert estimate_tokens(a + b) <= estimate_tokens(a) + estimate_tokens(b)
    assert estimate_tokens(a) <= len(a)
