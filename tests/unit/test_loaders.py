"""Tests for step S1.1: the loaders (file -> text) and the per-file report.

Every input file is built inside the test, in ``tmp_path``, so nothing binary is committed and a
reader can see exactly what each loader was given. The PDF is assembled by hand (a few lines of
PDF syntax with a correct cross-reference table), the Word, PowerPoint and Excel files by the
libraries that write them. The one committed file is the small CSV in ``tests/fixtures``.
"""

import datetime
import json
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pypdf import PdfWriter

from geekbot.ingest.loaders import (
    SUPPORTED_EXTENSIONS,
    LoadError,
    UnsupportedFormatError,
    load_file,
    load_paths,
    tabular,
)
from geekbot.ingest.loaders.common import clean_text, decode_bytes
from geekbot.ingest.loaders.notebook import load_notebook
from geekbot.ingest.loaders.text import load_text

SALES_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "sales_small.csv"


# --- builders --------------------------------------------------------------------------------


def make_pdf(page_texts: list[str]) -> bytes:
    """A valid one-font PDF with one page per string (an empty string = a page with no text).

    Hand-written so the test needs no PDF-writing library: header, numbered objects, a
    cross-reference table with the real byte offsets, and a trailer pointing at the catalog.
    """
    page_count = len(page_texts)
    first_page_id = 4
    page_ids = [first_page_id + 2 * index for index in range(page_count)]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, page_texts, strict=True):
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
            ).encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1") if text else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# --- the shared helpers ----------------------------------------------------------------------


def test_clean_text_normalises_newlines_and_trims_the_ends() -> None:
    assert clean_text("\r\n\r\nfirst\r\nsecond\rthird  \n\n") == "first\nsecond\nthird"


def test_clean_text_keeps_the_indentation_of_the_first_text_line() -> None:
    assert clean_text("\n\n    indented\n") == "    indented"


def test_clean_text_drops_nul_and_byte_order_marks_and_whitespace_only_text() -> None:
    assert clean_text("﻿a\x00b") == "ab"
    assert clean_text(" \n \t \n") == ""


def test_decode_bytes_reads_utf8_with_and_without_a_bom() -> None:
    assert decode_bytes("héllo".encode()) == "héllo"
    assert decode_bytes(b"\xef\xbb\xbfhello") == "hello"


def test_decode_bytes_reads_utf16_with_a_bom() -> None:
    assert decode_bytes("héllo wörld".encode("utf-16")) == "héllo wörld"


def test_decode_bytes_guesses_a_legacy_encoding_for_text_that_is_not_utf8() -> None:
    """Regression for a wrong guess found in the first run: an unrestricted search chose an
    encoding where byte 0xE9 is "ť", so the text came back as "cafť"."""
    legacy = ("Le café était ouvert, le thé était servi et la télé allumée. " * 12).encode("cp1252")
    decoded = decode_bytes(legacy)
    assert "café" in decoded
    assert "télé" in decoded


def test_decode_bytes_refuses_binary_data() -> None:
    with pytest.raises(LoadError, match="binary"):
        decode_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")


# --- text, markdown, code ---------------------------------------------------------------------


def test_a_plain_text_file_is_one_segment_titled_with_its_file_name(tmp_path: Path) -> None:
    path = write(tmp_path / "notes.txt", "First line.\nSecond line.\n")
    doc = load_file(path)
    assert (doc.kind, doc.title, doc.language) == ("text", "notes.txt", None)
    assert [segment.text for segment in doc.segments] == ["First line.\nSecond line."]
    assert doc.segments[0].page is None


def test_markdown_is_kept_as_written(tmp_path: Path) -> None:
    source = "# Title\n\nSome *emphasis*.\n\n## Section\n\n- one\n- two\n"
    doc = load_file(write(tmp_path / "readme.md", source))
    assert doc.kind == "markdown"
    assert doc.text == source.strip()


def test_code_keeps_its_indentation_and_names_its_language(tmp_path: Path) -> None:
    doc = load_file(write(tmp_path / "util.py", "def double(x):\n    return x * 2\n"))
    assert (doc.kind, doc.language) == ("code", "python")
    assert doc.text == "def double(x):\n    return x * 2"


@pytest.mark.parametrize(
    ("name", "language"),
    [
        ("app.ts", "ts"),
        ("main.go", "go"),
        ("lib.cpp", "cpp"),
        ("run.sh", "bash"),
        ("Q.cs", "csharp"),
    ],
)
def test_code_extensions_map_to_languages(tmp_path: Path, name: str, language: str) -> None:
    assert load_file(write(tmp_path / name, "x")).language == language


def test_yaml_is_plain_text_not_code(tmp_path: Path) -> None:
    doc = load_file(write(tmp_path / "config.yaml", "key: value\n"))
    assert (doc.kind, doc.language) == ("text", None)


def test_the_extension_is_matched_case_insensitively(tmp_path: Path) -> None:
    assert load_file(write(tmp_path / "NOTES.TXT", "hello")).kind == "text"


def test_windows_line_endings_and_a_utf8_bom_do_not_reach_the_text(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    assert load_file(path).text == "one\ntwo"


def test_an_empty_or_blank_text_file_is_an_error(tmp_path: Path) -> None:
    for name, content in (("empty.txt", ""), ("blank.md", " \n\n\t\n")):
        with pytest.raises(LoadError, match="empty"):
            load_file(write(tmp_path / name, content))


def test_a_binary_file_with_a_text_extension_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "image.txt"
    path.write_bytes(bytes(range(256)))
    with pytest.raises(LoadError, match="binary"):
        load_file(path)


def test_a_text_file_over_the_size_limit_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path / "big.txt", "x" * 100)
    with pytest.raises(LoadError, match="limit"):
        load_text(path, max_bytes=10)


# --- notebooks --------------------------------------------------------------------------------


def make_notebook(path: Path, cells: list[dict[str, object]], language: str | None = None) -> Path:
    metadata: dict[str, object] = {"language_info": {"name": language}} if language else {}
    path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": cells}),
        encoding="utf-8",
    )
    return path


def test_a_notebook_keeps_markdown_and_fenced_code_and_drops_outputs(tmp_path: Path) -> None:
    path = make_notebook(
        tmp_path / "analysis.ipynb",
        [
            {"cell_type": "markdown", "source": ["# Analysis\n", "We load the data."]},
            {
                "cell_type": "code",
                "source": "import pandas as pd\ndf = pd.read_csv('x.csv')",
                "outputs": [{"output_type": "stream", "text": "SECRET_OUTPUT_TEXT"}],
            },
            {"cell_type": "raw", "source": "raw cell text"},
            {"cell_type": "code", "source": "   \n"},
        ],
    )
    doc = load_file(path)
    assert (doc.kind, doc.language) == ("notebook", "python")
    assert doc.text == (
        "# Analysis\nWe load the data.\n\n"
        "```python\nimport pandas as pd\ndf = pd.read_csv('x.csv')\n```"
    )
    assert "SECRET_OUTPUT_TEXT" not in doc.text
    assert "raw cell text" not in doc.text


def test_a_notebook_uses_the_language_it_declares(tmp_path: Path) -> None:
    path = make_notebook(
        tmp_path / "j.ipynb", [{"cell_type": "code", "source": "println(1)"}], language="Julia"
    )
    doc = load_notebook(path)
    assert doc.language == "julia"
    assert "```julia" in doc.text


def test_a_notebook_with_nothing_to_index_is_an_error(tmp_path: Path) -> None:
    path = make_notebook(tmp_path / "empty.ipynb", [{"cell_type": "raw", "source": "x"}])
    with pytest.raises(LoadError, match="no markdown or code"):
        load_file(path)


def test_invalid_json_and_non_notebooks_are_errors(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not valid JSON"):
        load_file(write(tmp_path / "broken.ipynb", '{"cells": ['))
    with pytest.raises(LoadError, match="not a Jupyter notebook"):
        load_file(write(tmp_path / "other.ipynb", '{"hello": "world"}'))


# --- PDF --------------------------------------------------------------------------------------


def test_a_pdf_gives_one_segment_per_page_numbered_from_one(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(["First page text", "Second page text"]))
    doc = load_file(path)
    assert doc.kind == "pdf"
    assert [(segment.page, segment.text) for segment in doc.segments] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_a_pdf_page_without_text_leaves_a_gap_in_the_numbers(tmp_path: Path) -> None:
    path = tmp_path / "gap.pdf"
    path.write_bytes(make_pdf(["Alpha", "", "Gamma"]))
    assert [segment.page for segment in load_file(path).segments] == [1, 3]


def test_a_pdf_with_no_text_at_all_is_reported_as_a_probable_scan(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf(["", ""]))
    with pytest.raises(LoadError, match="OCR"):
        load_file(path)


def test_a_password_protected_pdf_is_an_error(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret", algorithm="AES-256")
    path = tmp_path / "locked.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(LoadError, match="password"):
        load_file(path)


# --- Word, PowerPoint, HTML -------------------------------------------------------------------


def test_a_word_file_becomes_markdown_with_its_headings(tmp_path: Path) -> None:
    document = Document()
    document.add_heading("Quarterly report", level=1)
    document.add_paragraph("Revenue grew in every region.")
    path = tmp_path / "report.docx"
    document.save(str(path))
    doc = load_file(path)
    assert doc.kind == "docx"
    assert "# Quarterly report" in doc.text
    assert "Revenue grew in every region." in doc.text
    assert doc.segments[0].page is None


def test_a_deck_gives_one_segment_per_slide_and_drops_slides_without_text(tmp_path: Path) -> None:
    presentation = Presentation()
    for title, body in (("Opening slide", "First body text"), ("Closing slide", "Last body text")):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    presentation.slides.add_slide(presentation.slide_layouts[6])  # blank: no text
    path = tmp_path / "deck.pptx"
    presentation.save(str(path))
    doc = load_file(path)
    assert doc.kind == "pptx"
    assert [segment.page for segment in doc.segments] == [1, 2]
    assert "Opening slide" in doc.segments[0].text
    assert "First body text" in doc.segments[0].text
    assert "Closing slide" in doc.segments[1].text
    assert "Slide number" not in doc.text


def test_html_loses_its_scripts_and_keeps_structure_and_title(tmp_path: Path) -> None:
    page = (
        "<html><head><title>My page</title><script>stealCookies()</script></head>"
        "<body><h1>Heading</h1><p>Some <b>bold</b> words.</p></body></html>"
    )
    doc = load_file(write(tmp_path / "page.html", page))
    assert (doc.kind, doc.title) == ("html", "My page")
    assert "# Heading" in doc.text
    assert "**bold**" in doc.text
    assert "stealCookies" not in doc.text


def test_a_truncated_word_file_is_a_load_error_not_a_crash(tmp_path: Path) -> None:
    """A real .docx cut in half: the zip's directory at the end is gone."""
    whole = tmp_path / "whole.docx"
    document = Document()
    document.add_paragraph("Some text that makes the file big enough to cut.")
    document.save(str(whole))
    data = whole.read_bytes()
    path = tmp_path / "cut.docx"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(LoadError, match="could not be read"):
        load_file(path)


# --- tables: schema cards, never rows ---------------------------------------------------------


def test_a_csv_becomes_a_schema_card_with_types_and_examples() -> None:
    doc = load_file(SALES_CSV)
    assert (doc.kind, doc.title, len(doc.segments)) == ("table", "sales_small.csv", 1)
    card = doc.text
    assert card.startswith("# Table: sales_small.csv")
    assert "CSV file with 12 rows and 4 columns" in card
    assert "all 12 rows" in card
    assert "| region | text | 0% | North, South, East |" in card
    assert "| month | date | 0% | 2024-01, 2024-02, 2024-03 |" in card
    assert "| units | integer | 0% | 10, 12, 14 |" in card
    assert "| revenue | integer | 0% | 100, 120, 140 |" in card


def test_a_csv_card_holds_no_row_text_and_points_to_csv_mode() -> None:
    card = load_file(SALES_CSV).text
    assert "North,2024-01,10,100" not in card
    assert "2024-04" not in card  # a value beyond the three examples is nowhere in the card
    assert "CSV mode" in card


def test_column_types_empty_shares_and_zip_like_codes(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mixed.csv",
        "name,score,ratio,active,zip,seen\n"
        "ann,5,0.5,true,00123,2024-01-05\n"
        "bob,,1,false,00456,2024-02-05\n"
        "cy,7,2.25,true,00789,2024-03-05T08:30:00\n"
        "dee,,3,false,00012,\n",
    )
    card = load_file(path).text
    assert "| name | text | 0% |" in card
    assert "| score | integer | 50% | 5, 7 |" in card
    assert "| ratio | decimal | 0% |" in card  # integers mixed with decimals are decimal
    assert "| active | boolean | 0% |" in card
    assert "| zip | text | 0% |" in card  # leading zeros: an identifier, not a number
    assert "| seen | date | 25% |" in card


def test_a_pipe_inside_a_value_cannot_break_the_table(tmp_path: Path) -> None:
    path = write(tmp_path / "pipes.csv", 'label\n"a|b"\n')
    assert "a\\|b" in load_file(path).text


def test_semicolon_and_tab_separated_files_are_split_correctly(tmp_path: Path) -> None:
    rows = "id{s}city{s}pop\n" + "".join(f"{n}{{s}}Town{n}{{s}}{n * 10}\n" for n in range(1, 7))
    semicolons = write(tmp_path / "eu.csv", rows.format(s=";"))
    tabs = write(tmp_path / "data.tsv", rows.format(s="\t"))
    for path, label in ((semicolons, "CSV file"), (tabs, "TSV file")):
        card = load_file(path).text
        assert f"{label} with 6 rows and 3 columns" in card
        assert "| city | text | 0% | Town1, Town2, Town3 |" in card


def test_a_legacy_encoded_csv_is_read(tmp_path: Path) -> None:
    text = "city,note\n" + "Genève,Le café était ouvert et le thé servi\n" * 30
    path = tmp_path / "latin.csv"
    path.write_bytes(text.encode("cp1252"))
    card = load_file(path).text
    assert "30 rows and 2 columns" in card
    assert "Genève" in card


def test_blank_lines_are_not_rows_and_a_header_only_file_has_zero(tmp_path: Path) -> None:
    path = write(tmp_path / "gaps.csv", "a,b\n1,2\n\n   ,\n3,4\n")
    assert "2 rows and 2 columns" in load_file(path).text
    only_header = write(tmp_path / "header.csv", "a,b,c\n")
    card = load_file(only_header).text
    assert "0 rows and 3 columns" in card
    assert "| a | empty | - |" in card


def test_an_empty_csv_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="empty"):
        load_file(write(tmp_path / "nothing.csv", "\n\n"))


def test_only_the_first_sample_rows_describe_types_but_the_count_is_exact(tmp_path: Path) -> None:
    total = tabular.SAMPLE_ROWS + 500
    path = write(tmp_path / "long.csv", "n\n" + "".join(f"{n}\n" for n in range(total)))
    card = load_file(path).text
    assert f"{total:,} rows" in card
    assert f"the first {tabular.SAMPLE_ROWS:,} rows" in card


def test_a_very_wide_table_lists_only_the_first_columns(tmp_path: Path) -> None:
    width = tabular.MAX_COLUMNS_SHOWN + 5
    header = ",".join(f"c{n}" for n in range(width))
    values = ",".join("1" for _ in range(width))
    card = load_file(write(tmp_path / "wide.csv", f"{header}\n{values}\n")).text
    assert "| c0 |" in card
    assert f"| c{tabular.MAX_COLUMNS_SHOWN} |" not in card
    assert "5 more columns are not described here." in card


def make_workbook(path: Path) -> Path:
    workbook = Workbook()
    sales = workbook.active
    sales.title = "Sales"
    sales.append(["date", "amount", "note"])
    sales.append([datetime.datetime(2024, 1, 5), 12.5, "ok"])
    sales.append([datetime.datetime(2024, 2, 5), 7, None])
    secret = workbook.create_sheet("Secret")
    secret.append(["hidden_column"])
    secret.append(["hidden value"])
    secret.sheet_state = "hidden"
    workbook.create_sheet("Blank")  # visible but with no data
    workbook.save(path)
    return path


def test_an_excel_workbook_gets_one_card_per_visible_sheet_with_data(tmp_path: Path) -> None:
    doc = load_file(make_workbook(tmp_path / "book.xlsx"))
    assert (doc.kind, len(doc.segments)) == ("table", 1)
    card = doc.text
    assert card.startswith("# Table: book.xlsx, sheet Sales")
    assert "Excel sheet with 2 rows and 3 columns" in card
    assert "| date | date | 0% |" in card
    assert "| amount | decimal | 0% | 12.5, 7 |" in card
    assert "| note | text | 50% | ok |" in card
    assert "hidden" not in card


def test_a_workbook_without_visible_data_is_an_error(tmp_path: Path) -> None:
    workbook = Workbook()  # starts with one empty sheet
    workbook.create_sheet("Second")
    path = tmp_path / "blank.xlsx"
    workbook.save(path)
    with pytest.raises(LoadError, match="no visible sheet"):
        load_file(path)


# --- dispatch and the per-file report ---------------------------------------------------------


def test_the_supported_extensions() -> None:
    assert {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".py", ".ipynb"} <= (
        SUPPORTED_EXTENSIONS
    )
    assert {".csv", ".tsv", ".xlsx"} <= SUPPORTED_EXTENSIONS
    # Legacy binary Office formats are not read: say so in the report instead of guessing.
    assert not {".doc", ".ppt", ".xls"} & SUPPORTED_EXTENSIONS


def test_an_unknown_extension_is_unsupported_not_a_failure(tmp_path: Path) -> None:
    path = write(tmp_path / "archive.xyz", "data")
    with pytest.raises(UnsupportedFormatError, match=r"\.xyz"):
        load_file(path)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_file(tmp_path / "ghost.txt")


def test_one_bad_file_does_not_stop_the_batch(tmp_path: Path) -> None:
    good = write(tmp_path / "good.txt", "hello")
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"this is not a pdf")
    unknown = write(tmp_path / "data.xyz", "?")
    missing = tmp_path / "missing.txt"
    also_good = write(tmp_path / "also.md", "# Hi")

    report = load_paths([good, broken_pdf, unknown, missing, also_good])

    assert [doc.path for doc in report.loaded] == [good, also_good]
    assert [problem.path for problem in report.failed] == [broken_pdf, missing]
    assert [problem.path for problem in report.skipped] == [unknown]
    assert "could not be read" in report.failed[0].reason
    assert "not found" in report.failed[1].reason
    assert ".xyz" in report.skipped[0].reason
    assert report.summary() == "2 loaded, 2 failed, 1 skipped"


def test_a_parser_crash_keeps_its_cause(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(LoadError) as caught:
        load_file(path)
    assert caught.value.__cause__ is not None
