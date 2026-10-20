"""Helpers the loaders share: reading text files safely and tidying extracted text (S1.1)."""

import codecs
from pathlib import Path
from typing import Final

from charset_normalizer import from_bytes

from geekbot.ingest.loaders.model import LoadError

# A text, code or notebook file bigger than this is refused instead of read whole into memory.
# Nothing in the plan's corpus comes near it. The limits for every other kind of file, and the
# 100 MB ceiling over all of them, are in ``limits.py`` (S4.3).
MAX_TEXT_BYTES: Final = 10_000_000

_BINARY_SNIFF_BYTES: Final = 8192
_UTF16_BOMS: Final = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)

# The legacy encodings a guess may choose from: the Windows code pages and the East-Asian
# encodings a Windows user is likely to have. Found at S1.1: left to search every encoding,
# charset-normalizer picked one where byte 0xE9 is "ť" (probably a Mac code page) for French
# text written in cp1252, and the index would have held "cafť". A short list trades a rare
# unreadable Mac or DOS file (it fails loudly) for fewer silent wrong guesses.
_LEGACY_ENCODINGS: Final = [
    "cp1252",
    "cp1250",
    "cp1251",
    "cp1253",
    "cp1254",
    "cp1255",
    "cp1256",
    "cp1257",
    "gb18030",
    "big5",
    "shift_jis",
    "euc_jp",
    "euc_kr",
]


def clean_text(text: str) -> str:
    """Normalise newlines to ``\\n``, drop NUL and byte-order-mark characters, trim the ends.

    Leading blank lines and trailing whitespace go; leading spaces on the first text line stay,
    because in code they are meaning. A whitespace-only text comes back as ``""``, which callers
    treat as "no text found".
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\N{ZERO WIDTH NO-BREAK SPACE}", "")  # a stray BOM
    return text.lstrip("\n").rstrip()


def detect_encoding(head: bytes) -> str:
    """The encoding to read a text file with, from its first bytes, or ``LoadError``.

    ``head`` may be only the start of a file (a character cut off at its end is tolerated).
    Order: UTF-16 if it starts with a byte-order mark (UTF-16 contains NUL bytes by nature, so
    it must be tested before the binary check); a NUL byte near the start means a binary file;
    then UTF-8 (with or without a BOM); then ``charset-normalizer``'s best guess among
    ``_LEGACY_ENCODINGS`` (cp1252 and its relatives, East-Asian encodings). A guess that finds
    nothing plausible is an error, not a lossy decode: a half-garbled chunk in the index is
    worse than a file in the failure report.
    """
    if head.startswith(_UTF16_BOMS):
        return "utf-16"
    if b"\x00" in head[:_BINARY_SNIFF_BYTES]:
        raise LoadError("the file looks binary (it contains NUL bytes), not text")
    try:
        codecs.getincrementaldecoder("utf-8-sig")().decode(head, final=False)
    except UnicodeDecodeError:
        return _guess_legacy_encoding(head)
    return "utf-8-sig"


def _guess_legacy_encoding(head: bytes) -> str:
    """The most likely legacy encoding of bytes that are not UTF-8, or ``LoadError``.

    The Windows code pages differ on only a few bytes (cp1252 reads byte 0xE8 as "è", cp1250 as
    "č"), so a short Western text cannot tell them apart by itself. Found at S1.1: the library's
    own ranking chose cp1250 for French text written in cp1252. cp1252 is by far the most common
    legacy encoding on Windows, so when the library finds it plausible it wins; the ranking
    decides only between the others. The price: Central-European text that also happens to look
    plausible as cp1252 is read as cp1252. UTF-8 is the format Geekbot supports without caveats.
    """
    matches = from_bytes(head, cp_isolation=_LEGACY_ENCODINGS)
    for match in matches:
        if "cp1252" in match.could_be_from_charset:
            return "cp1252"
    best = matches.best()
    if best is None:
        raise LoadError("the text encoding could not be determined")
    return str(best.encoding)


def decode_bytes(data: bytes) -> str:
    """Decode a whole text file's bytes, or raise ``LoadError`` if they are not text."""
    encoding = detect_encoding(data)
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as error:
        raise LoadError(f"the file is not valid {encoding} text: {error.reason}") from error


def read_text_file(path: Path, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    """The decoded content of a text file; ``LoadError`` if it is too big or not text."""
    size = path.stat().st_size
    if size > max_bytes:
        raise LoadError(
            f"the file is {size / 1_000_000:.1f} MB, over the {max_bytes / 1_000_000:.0f} MB "
            "limit for text files"
        )
    return decode_bytes(path.read_bytes())
