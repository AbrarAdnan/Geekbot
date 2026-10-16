"""Tests for step S1.3: IDs and hashes (PLAN.md §6.4).

Two kinds of test live here. The examples pin down the formulas and the edge cases a reader can
check by eye: what the path and text normalizers fold together, what a repeated chunk is called,
what moving a file changes. The property tests at the end throw arbitrary lists of chunk texts at
the numbering and check what must hold for every list: repeats get distinct IDs, an insert adds
one ID and keeps the others. They are the tests of this step's ✍️ piece, ``assign_occurrences``
(written by Claude in prototyping mode, D-047).
"""

import hashlib
import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.ingest.chunking import ChunkDraft, chunk_document
from geekbot.ingest.ids import (
    IdentifiedChunk,
    assign_occurrences,
    chunk_id_of,
    content_hash_of,
    doc_id_of,
    file_hash_of,
    identify_chunks,
    normalize_path,
    normalize_text,
)
from geekbot.ingest.loaders.model import LoadedDocument, Segment

DOC_PATH = "/data/notes.md"
DOC = "0123456789abcdef"
HASH = "a" * 64
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# Built with chr() so the file stays plain ASCII and no tool can turn an escape into a glyph.
E_ACUTE = chr(0xE9)  # "e" with an acute accent, as one character
COMBINING_ACUTE = chr(0x301)  # the accent on its own: "e" + this looks the same
LONE_SURROGATE = chr(0xD800)  # not valid text on its own; a PDF can still hand us one


def draft(text: str, ordinal: int = 0) -> ChunkDraft:
    """A chunk with made-up positions: identity must not depend on them."""
    start = ordinal * 1000
    return ChunkDraft(
        text=text,
        char_start=start,
        char_end=start + len(text),
        page=None,
        section_path=(),
        header="doc",
        ordinal=ordinal,
    )


def ids_of(texts: Sequence[str], path: str = DOC_PATH) -> list[str]:
    drafts = [draft(text, index) for index, text in enumerate(texts)]
    return [item.chunk_id for item in identify_chunks(path, drafts)]


def markdown_doc(text: str) -> LoadedDocument:
    return LoadedDocument(
        path=Path("notes.md"), kind="markdown", title="notes.md", segments=(Segment(text),)
    )


def by_heading(items: list[IdentifiedChunk]) -> dict[str, IdentifiedChunk]:
    """Index real chunks by the heading they sit under (one chunk per tiny section)."""
    return {item.chunk.section_path[-1]: item for item in items}


# --- file hash -------------------------------------------------------------------------------


def test_file_hash_matches_the_sha256_test_vectors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    abc = tmp_path / "abc.txt"
    abc.write_bytes(b"abc")
    assert file_hash_of(empty) == EMPTY_SHA256
    assert file_hash_of(abc) == ABC_SHA256
    assert file_hash_of(str(abc)) == ABC_SHA256  # a plain string works as well as a Path


def test_file_hash_follows_the_bytes_not_the_name(tmp_path: Path) -> None:
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    assert file_hash_of(first) == file_hash_of(second)
    second.write_bytes(b"same bytez")  # one byte
    assert file_hash_of(first) != file_hash_of(second)


def test_file_hash_reads_a_file_larger_than_one_block(tmp_path: Path) -> None:
    data = bytes(range(256)) * 2400  # 614,400 bytes, more than hashlib's 256 KiB block
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert file_hash_of(path) == hashlib.sha256(data).hexdigest()


def test_file_hash_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_hash_of(tmp_path / "gone.pdf")


# --- document ID -----------------------------------------------------------------------------


def test_doc_id_is_16_hex_characters_of_sha256_of_the_normalized_path() -> None:
    expected = hashlib.sha256(normalize_path(DOC_PATH).encode("utf-8")).hexdigest()[:16]
    assert doc_id_of(DOC_PATH) == expected
    assert re.fullmatch(r"[0-9a-f]{16}", expected)


def test_different_files_get_different_doc_ids() -> None:
    assert doc_id_of("/data/a.pdf") != doc_id_of("/data/b.pdf")
    assert doc_id_of("/one/a.pdf") != doc_id_of("/two/a.pdf")  # same name, other folder


def test_spellings_of_one_path_share_a_doc_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "docs" / "a.txt"
    spellings = [
        "docs/a.txt",  # relative to the current directory
        "./docs/./a.txt",
        absolute,  # a Path as well as a string
        tmp_path / "docs" / "sub" / ".." / "a.txt",  # ".." collapsed without touching the disk
    ]
    assert len({doc_id_of(spelling) for spelling in spellings}) == 1
    assert doc_id_of(spellings[0]) == doc_id_of(absolute)


def test_a_composed_and_a_decomposed_file_name_are_one_document() -> None:
    composed = "/data/caf" + E_ACUTE + ".txt"
    decomposed = "/data/cafe" + COMBINING_ACUTE + ".txt"  # macOS stores names this way
    assert composed != decomposed
    assert doc_id_of(composed) == doc_id_of(decomposed)


@pytest.mark.skipif(os.name != "nt", reason="only Windows ignores case and accepts both slashes")
def test_windows_paths_ignore_case_and_slash_direction() -> None:
    assert doc_id_of("C:\\Docs\\Report.PDF") == doc_id_of("c:/docs/report.pdf")


@pytest.mark.skipif(os.name == "nt", reason="Windows paths are case-insensitive")
def test_posix_paths_keep_their_case() -> None:
    assert doc_id_of("/data/Report.pdf") != doc_id_of("/data/report.pdf")


def test_normalize_path_is_idempotent_and_absolute(tmp_path: Path) -> None:
    once = normalize_path(tmp_path / "x" / ".." / "a.txt")
    assert normalize_path(once) == once
    assert Path(once).is_absolute()


# --- content hash ----------------------------------------------------------------------------


def test_content_hash_of_plain_text_is_its_sha256() -> None:
    assert content_hash_of("abc") == ABC_SHA256
    assert content_hash_of("") == EMPTY_SHA256


def test_content_hash_ignores_line_endings_composition_and_trailing_whitespace() -> None:
    assert content_hash_of("a\r\nb") == content_hash_of("a\nb") == content_hash_of("a\rb")
    assert content_hash_of("caf" + E_ACUTE) == content_hash_of("cafe" + COMBINING_ACUTE)
    assert content_hash_of("a  \n\n") == content_hash_of("a")
    assert content_hash_of("a \n b \t\n") == content_hash_of("a \n b")  # only the end is trimmed


def test_content_hash_keeps_differences_that_carry_meaning() -> None:
    assert content_hash_of("    return x") != content_hash_of("return x")  # code indentation
    assert content_hash_of("a  b") != content_hash_of("a b")
    assert content_hash_of("Alpha") != content_hash_of("alpha")
    assert content_hash_of("a\nb") != content_hash_of("a b")
    assert content_hash_of("a\n\nb") != content_hash_of("a\nb")  # a paragraph break


def test_normalize_text_examples() -> None:
    assert normalize_text("x\r\ny\rz  \n") == "x\ny\nz"
    assert normalize_text("  indented\t") == "  indented"  # leading spaces are kept
    assert normalize_text("") == ""
    assert normalize_text(" \n\t ") == ""


def test_text_with_a_lone_surrogate_still_has_a_hash() -> None:
    # A PDF can hand us one. It must not crash an ingest, and the hash must be stable.
    odd = "a" + LONE_SURROGATE + "b"
    assert content_hash_of(odd) == content_hash_of(odd)
    assert content_hash_of(odd) != content_hash_of("ab")


# --- chunk ID --------------------------------------------------------------------------------


def test_chunk_id_is_24_hex_characters_of_sha256_of_the_three_parts() -> None:
    expected = hashlib.sha256(f"{DOC}:{HASH}:2".encode()).hexdigest()[:24]
    assert chunk_id_of(DOC, HASH, 2) == expected
    assert re.fullmatch(r"[0-9a-f]{24}", expected)


def test_every_part_of_the_identity_changes_the_chunk_id() -> None:
    base = chunk_id_of(DOC, HASH, 0)
    assert chunk_id_of("fedcba9876543210", HASH, 0) != base  # another document
    assert chunk_id_of(DOC, "b" * 64, 0) != base  # other content
    assert chunk_id_of(DOC, HASH, 1) != base  # the second copy


@pytest.mark.parametrize(
    ("doc_id", "content_hash", "occurrence"),
    [
        ("short", HASH, 0),  # not 16 characters
        (DOC.upper(), HASH, 0),  # IDs are lowercase hex
        (DOC, "xyz", 0),  # a chunk's text where its hash belongs
        (DOC, HASH[:-1], 0),  # one character short
        (DOC, HASH, -1),
    ],
)
def test_chunk_id_rejects_arguments_of_the_wrong_shape(
    doc_id: str, content_hash: str, occurrence: int
) -> None:
    with pytest.raises(ValueError, match=r"must be|counts from 0"):
        chunk_id_of(doc_id, content_hash, occurrence)


# --- occurrence numbering --------------------------------------------------------------------


def test_assign_occurrences_numbers_each_repeat_from_zero() -> None:
    assert assign_occurrences([]) == []
    assert assign_occurrences(["a"]) == [0]
    assert assign_occurrences(["a", "b", "a", "a", "b"]) == [0, 0, 1, 2, 1]
    assert assign_occurrences(["x", "x", "x"]) == [0, 1, 2]
    assert assign_occurrences(["a", "b", "c"]) == [0, 0, 0]  # no repeats, no numbering


def test_a_new_text_does_not_renumber_the_others() -> None:
    before = assign_occurrences(["a", "b", "a"])
    after = assign_occurrences(["a", "NEW", "b", "a"])
    assert before == [0, 0, 1]
    assert after == [0, 0, 0, 1]  # a, b, a keep 0, 0, 1; the new text starts at 0


# --- putting it together on drafts -----------------------------------------------------------


def test_identify_chunks_gives_repeated_text_distinct_ids() -> None:
    items = identify_chunks(DOC_PATH, [draft("Confidential", i) for i in range(3)])
    assert len({item.content_hash for item in items}) == 1
    assert [item.occurrence for item in items] == [0, 1, 2]
    assert len({item.chunk_id for item in items}) == 3
    assert {item.doc_id for item in items} == {doc_id_of(DOC_PATH)}
    assert [item.chunk.ordinal for item in items] == [0, 1, 2]  # the draft is carried along


def test_identify_chunks_of_nothing_is_nothing() -> None:
    assert identify_chunks(DOC_PATH, []) == []


def test_position_metadata_is_not_part_of_the_identity() -> None:
    drafts = [draft("alpha", 0), draft("beta", 1), draft("alpha", 2)]
    moved = [
        replace(
            item,
            char_start=item.char_start + 999,
            char_end=item.char_end + 999,
            page=7,
            ordinal=item.ordinal + 5,
        )
        for item in drafts
    ]
    original_ids = [item.chunk_id for item in identify_chunks(DOC_PATH, drafts)]
    moved_ids = [item.chunk_id for item in identify_chunks(DOC_PATH, moved)]
    assert moved_ids == original_ids


def test_identity_is_the_same_on_every_run() -> None:
    # Guards against anything salted per process (the built-in hash()): two calls agree, and the
    # ID equals what hashlib computes by hand.
    assert ids_of(["alpha", "beta"]) == ids_of(["alpha", "beta"])
    expected = chunk_id_of(doc_id_of(DOC_PATH), content_hash_of("alpha"), 0)
    assert ids_of(["alpha", "beta"])[0] == expected


# --- with the real chunker -------------------------------------------------------------------

THREE_SECTIONS = "## One\n\nAlpha text.\n\n## Two\n\nBeta text.\n\n## Three\n\nGamma text.\n"


def test_two_identical_sections_get_distinct_ids_from_one_content_hash() -> None:
    doc = markdown_doc("## Notes\n\nSame body text.\n\n## Notes\n\nSame body text.\n")
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    first, second = identify_chunks(doc.path, chunks)
    assert first.content_hash == second.content_hash
    assert (first.occurrence, second.occurrence) == (0, 1)
    assert first.chunk_id != second.chunk_id
    assert first.chunk.char_start != second.chunk.char_start  # same text, two places


def sections_of(text: str) -> dict[str, IdentifiedChunk]:
    """Chunk and identify a Markdown text, indexed by heading."""
    return by_heading(identify_chunks(DOC_PATH, chunk_document(markdown_doc(text))))


def test_inserting_a_section_keeps_the_ids_of_the_others() -> None:
    before = sections_of(THREE_SECTIONS)
    new_section = "## One and a half\n\nNew text.\n\n"
    after = sections_of(THREE_SECTIONS.replace("## Two", new_section + "## Two"))

    assert set(before) == {"One", "Two", "Three"}
    assert set(after) == {"One", "One and a half", "Two", "Three"}
    for heading in before:
        assert after[heading].chunk_id == before[heading].chunk_id
    assert (before["Two"].chunk.ordinal, after["Two"].chunk.ordinal) == (1, 2)  # moved, same ID
    old_ids = {item.chunk_id for item in before.values()}
    new_ids = {item.chunk_id for item in after.values()} - old_ids
    assert new_ids == {after["One and a half"].chunk_id}


def test_editing_a_section_changes_only_its_own_id() -> None:
    before = sections_of(THREE_SECTIONS)
    after = sections_of(THREE_SECTIONS.replace("Beta text.", "Beta text, edited."))

    assert after["One"].chunk_id == before["One"].chunk_id
    assert after["Three"].chunk_id == before["Three"].chunk_id
    assert after["Two"].content_hash != before["Two"].content_hash
    assert after["Two"].chunk_id != before["Two"].chunk_id


def test_moving_a_file_changes_every_chunk_id_but_no_content_hash() -> None:
    chunks = chunk_document(markdown_doc(THREE_SECTIONS))
    here = identify_chunks("/data/a/notes.md", chunks)
    there = identify_chunks("/data/b/notes.md", chunks)
    assert [item.content_hash for item in here] == [item.content_hash for item in there]
    assert {item.chunk_id for item in here}.isdisjoint({item.chunk_id for item in there})
    assert here[0].doc_id != there[0].doc_id


# --- properties ------------------------------------------------------------------------------

# A small pool of words, so that repeats are common in the lists below, plus short arbitrary text.
_pool = st.one_of(st.sampled_from(["alpha", "beta", "gamma"]), st.text(max_size=4))
_texts = st.lists(_pool, max_size=25)
_hashes = st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=40)


@settings(max_examples=300, deadline=None)
@given(hashes=_hashes)
def test_occurrences_count_each_hash_from_zero_in_order(hashes: list[str]) -> None:
    occurrences = assign_occurrences(hashes)
    assert len(occurrences) == len(hashes)
    for value in set(hashes):
        mine = [number for item, number in zip(hashes, occurrences, strict=True) if item == value]
        assert mine == list(range(hashes.count(value)))


@settings(max_examples=300, deadline=None)
@given(texts=_texts)
def test_ids_in_one_document_are_distinct_even_for_repeated_text(texts: list[str]) -> None:
    ids = ids_of(texts)
    assert len(set(ids)) == len(ids)


@settings(max_examples=300, deadline=None)
@given(texts=_texts, new=_pool, where=st.integers(min_value=0, max_value=1000))
def test_inserting_a_chunk_adds_exactly_one_id_and_keeps_the_others(
    texts: list[str], new: str, where: int
) -> None:
    index = where % (len(texts) + 1)
    before = set(ids_of(texts))
    after = set(ids_of([*texts[:index], new, *texts[index:]]))
    assert before <= after
    assert len(after - before) == 1


@settings(max_examples=300, deadline=None)
@given(
    texts=st.lists(_pool, min_size=1, max_size=25),
    where=st.integers(min_value=0, max_value=1000),
)
def test_removing_a_chunk_removes_exactly_one_id_and_keeps_the_others(
    texts: list[str], where: int
) -> None:
    index = where % len(texts)
    before = set(ids_of(texts))
    after = set(ids_of([*texts[:index], *texts[index + 1 :]]))
    assert after <= before
    assert len(before - after) == 1


@settings(max_examples=300, deadline=None)
@given(texts=_texts, data=st.data())
def test_the_set_of_ids_does_not_depend_on_the_order_of_the_chunks(
    texts: list[str], data: st.DataObject
) -> None:
    shuffled = data.draw(st.permutations(texts))
    assert set(ids_of(shuffled)) == set(ids_of(texts))


_plain_line = st.text(
    alphabet=st.characters(exclude_characters="\r\n", exclude_categories=("Cs",)), max_size=12
)


@settings(max_examples=300, deadline=None)
@given(lines=st.lists(_plain_line, max_size=6))
def test_line_ending_style_never_changes_the_content_hash(lines: list[str]) -> None:
    unix = content_hash_of("\n".join(lines))
    assert content_hash_of("\r\n".join(lines)) == unix
    assert content_hash_of("\r".join(lines)) == unix


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=40))
def test_normalizing_twice_changes_nothing(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=40))
def test_unicode_composition_never_changes_the_content_hash(text: str) -> None:
    assert content_hash_of(unicodedata.normalize("NFD", text)) == content_hash_of(text)


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=40), tail=st.text(alphabet=" \t\r\n", max_size=6))
def test_trailing_whitespace_never_changes_the_content_hash(text: str, tail: str) -> None:
    assert content_hash_of(text + tail) == content_hash_of(text)
