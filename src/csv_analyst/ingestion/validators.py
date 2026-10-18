"""CSV ingestion guards (PLAN.md Phase 1). Every check raises a typed
IngestionError with a machine-readable `code` — never a bare traceback.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_path

ALLOWED_EXTENSIONS = (".csv", ".tsv")

# Case-insensitive, whitespace-stripped. Extended per CLAUDE.md example list.
NA_SENTINELS = frozenset({"", "na", "n/a", "null", "none", "nan", "-", "?"})

# Binary file signatures that must never be accepted despite a .csv extension.
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "a zip-based format (e.g. .xlsx/.docx)"),
    (b"%PDF", "a PDF"),
    (b"\x89PNG", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"\xd0\xcf\x11\xe0", "a legacy Office binary format (e.g. .xls)"),
)

_ENCODING_FALLBACK_ORDER = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class IngestionError(Exception):
    """Raised for any CSV loading failure. `code` is machine-readable and
    intended for API/CLI responses; `message` is human-readable."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class LoadWarning:
    code: str
    message: str


def check_file_exists(path: Path) -> None:
    if not path.exists():
        raise IngestionError("file_not_found", f"File does not exist: {path}")
    if not path.is_file():
        raise IngestionError("not_a_file", f"Path is not a file: {path}")


def check_extension(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            "bad_extension",
            f"Unsupported file extension {path.suffix!r}; expected one of {ALLOWED_EXTENSIONS}",
        )


def check_file_size(path: Path, max_mb: int) -> int:
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise IngestionError("empty_file", f"File is empty: {path}")
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionError(
            "file_too_large",
            f"File is {size_bytes / 1024 / 1024:.2f} MB, exceeds the {max_mb} MB limit",
        )
    return size_bytes


def check_binary_signature(path: Path) -> None:
    """Reject a renamed non-CSV binary file even if the extension is .csv."""
    with open(path, "rb") as f:
        head = f.read(8)
    for signature, kind in _BINARY_SIGNATURES:
        if head.startswith(signature):
            raise IngestionError("not_csv", f"File content looks like {kind}, not CSV text")


def detect_encoding(path: Path) -> str:
    """Try utf-8-sig -> utf-8 -> charset-normalizer detection -> cp1252 -> latin-1.
    latin-1 accepts any byte sequence, so this only fails if the file is unreadable.

    utf-8-sig is only reported when a real BOM is present (it is checked first
    against the raw bytes, not just "does this codec decode the file" — the
    utf-8-sig codec happily decodes plain utf-8 too, which would otherwise
    mislabel every ordinary utf-8 file as BOM-prefixed).
    """
    raw = path.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            raw.decode("utf-8-sig")
            return "utf-8-sig"
        except UnicodeDecodeError:
            pass

    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        best = from_path(str(path)).best()
        if best is not None and best.encoding:
            raw.decode(best.encoding)
            return best.encoding
    except (UnicodeDecodeError, LookupError, Exception):
        pass

    for enc in ("cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue

    raise IngestionError("encoding_error", "Could not determine a working text encoding for this file")


def sniff_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def is_na_sentinel(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in NA_SENTINELS
