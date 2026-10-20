"""Tests for step S4.3: size and page limits in the loaders (``ingest/loaders/limits.py``).

The real limits are 100 MB and 2,000 pages, so the tests lower them (the functions read the module's
constants when they are called) or pass a small limit directly. One test builds a miniature zip bomb:
a few kilobytes that unpack into megabytes.
"""

import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from geekbot.ingest.loaders import LoadError, load_file, load_paths
from geekbot.ingest.loaders import limits as limits_module
from geekbot.ingest.loaders.limits import check_file_limits, check_page_limit
from geekbot.ingest.loaders.pdf import load_pdf
from geekbot.server.security import MAX_UPLOAD_BYTES


def make_zip(
    path: Path,
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
    comment: bytes = b"",
) -> Path:
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.comment = comment
    return path


def make_pdf(path: Path, pages: int) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_an_upload_may_be_exactly_as_big_as_a_file_the_loaders_accept() -> None:
    assert MAX_UPLOAD_BYTES == limits_module.MAX_FILE_BYTES


# --- file size -------------------------------------------------------------------------------


def test_a_file_over_the_limit_is_refused_and_one_at_the_limit_is_not(tmp_path: Path) -> None:
    path = tmp_path / "big.pdf"
    path.write_bytes(b"x" * 20)
    check_file_limits(path, max_bytes=20)
    with pytest.raises(LoadError, match="over the .* limit"):
        check_file_limits(path, max_bytes=19)


def test_load_file_applies_the_limit_before_any_parser_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x" * 50, encoding="utf-8")
    monkeypatch.setattr(limits_module, "MAX_FILE_BYTES", 10)
    with pytest.raises(LoadError, match="over the"):
        load_file(path)


def test_a_file_that_is_too_big_is_one_failure_among_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big, small = tmp_path / "big.txt", tmp_path / "small.txt"
    big.write_text("x" * 50, encoding="utf-8")
    small.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(limits_module, "MAX_FILE_BYTES", 10)
    report = load_paths([big, small])
    assert [problem.path for problem in report.failed] == [big]
    assert [document.path for document in report.loaded] == [small]


# --- archives --------------------------------------------------------------------------------


def test_an_office_file_that_unpacks_to_more_than_the_limit_is_refused(tmp_path: Path) -> None:
    path = make_zip(tmp_path / "a.docx", {"one": b"a" * 2000, "two": b"b" * 2000})
    check_file_limits(path, max_unpacked=4000)  # the sum is 4000: at the limit
    with pytest.raises(LoadError, match="unpacks to"):
        check_file_limits(path, max_unpacked=3999)


def test_a_small_zip_that_unpacks_into_megabytes_is_caught(tmp_path: Path) -> None:
    """A miniature zip bomb: it passes the size limit by a wide margin and fails the unpacked one."""
    path = make_zip(
        tmp_path / "bomb.xlsx", {"zeros": b"\x00" * 5_000_000}, compression=zipfile.ZIP_DEFLATED
    )
    assert path.stat().st_size < 50_000
    check_file_limits(
        path
    )  # the real limits let it through; a real bomb is a thousand times bigger
    with pytest.raises(LoadError, match="unpacks to"):
        check_file_limits(path, max_unpacked=1_000_000)


def test_an_archive_with_too_many_parts_is_refused_before_its_directory_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits_module, "MAX_ZIP_MEMBERS", 3)
    fits = make_zip(tmp_path / "three.pptx", {f"p{i}": b"x" for i in range(3)})
    check_file_limits(fits)
    too_many = make_zip(tmp_path / "four.pptx", {f"p{i}": b"x" for i in range(4)})
    with pytest.raises(LoadError, match="4 parts"):
        check_file_limits(too_many)


def test_the_part_count_is_found_even_when_the_archive_has_a_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits_module, "MAX_ZIP_MEMBERS", 3)
    commented = make_zip(
        tmp_path / "four.docx", {f"p{i}": b"x" for i in range(4)}, comment=b"c" * 500
    )
    with pytest.raises(LoadError, match="4 parts"):
        check_file_limits(commented)


def test_a_file_that_is_not_an_archive_is_left_to_its_loader(tmp_path: Path) -> None:
    path = tmp_path / "fake.docx"
    path.write_bytes(b"this is not a zip file")
    check_file_limits(path)  # no error here: the loader will say it cannot read it


def test_only_office_extensions_are_opened_as_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits_module, "MAX_ZIP_MEMBERS", 1)
    path = make_zip(tmp_path / "data.txt", {f"p{i}": b"x" for i in range(5)})
    check_file_limits(path)  # a .txt is not unpacked, whatever its bytes


# --- pdf pages -------------------------------------------------------------------------------


def test_a_pdf_with_too_many_pages_is_refused_before_any_page_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits_module, "MAX_PDF_PAGES", 2)
    with pytest.raises(LoadError, match="3 pages"):
        load_pdf(make_pdf(tmp_path / "three.pdf", 3))
    # Two pages pass the limit; the blank pages then fail for the usual reason, which proves it.
    with pytest.raises(LoadError, match="no text found"):
        load_pdf(make_pdf(tmp_path / "two.pdf", 2))


def test_load_file_reports_the_page_limit_for_a_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits_module, "MAX_PDF_PAGES", 1)
    with pytest.raises(LoadError, match="2 pages"):
        load_file(make_pdf(tmp_path / "two.pdf", 2))


def test_the_page_check_alone() -> None:
    check_page_limit(5, limit=5)
    with pytest.raises(LoadError, match="6 pages"):
        check_page_limit(6, limit=5)
