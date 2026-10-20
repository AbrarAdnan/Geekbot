"""Size and page limits for the files the loaders open (step S4.3, PLAN.md section 8.1).

Until the server existed the only reader was the user, feeding in their own files from the command
line. A server takes files (and paths to files) from a client, so a file that is far bigger than any
real document, or that is built to unpack into gigabytes, must be turned away before a parser reads
it. A limit here is a clean ``LoadError`` for that one file ("too big", reported with the others),
never a crash and never a stalled machine.

**Every number below is a starting point, not a measurement**: generous for real documents, small
enough to stop a mistake. Text, code and notebook files keep their own, lower caps (``common.py``:
10 MB, 50 MB).

Office files (``.docx``, ``.pptx``, ``.xlsx``) are zip archives. A "zip bomb" is a few kilobytes that
unpack into gigabytes, or a few megabytes of empty entries, so two things are limited: the number of
entries, read from the end-of-archive record *before* the archive's directory is parsed (parsing a
million entries is itself the attack), and the size of the unpacked contents as that directory
declares them. Python's ``zipfile`` never reads past a member's declared size, which is what makes the
declared size a real bound.
"""

import os
import struct
import zipfile
from pathlib import Path
from typing import Final

from geekbot.ingest.loaders.model import LoadError

__all__ = [
    "MAX_FILE_BYTES",
    "MAX_PDF_PAGES",
    "MAX_UNPACKED_BYTES",
    "MAX_ZIP_MEMBERS",
    "ZIP_EXTENSIONS",
    "check_file_limits",
    "check_page_limit",
]

MAX_FILE_BYTES: Final = 100_000_000  # any file, as stored; the same as an upload may be
MAX_UNPACKED_BYTES: Final = 500_000_000  # all members of a .docx / .pptx / .xlsx, unpacked
MAX_ZIP_MEMBERS: Final = 20_000  # a real Office file has dozens, a long deck some thousands
MAX_PDF_PAGES: Final = 2_000
ZIP_EXTENSIONS: Final = frozenset({".docx", ".pptx", ".xlsx"})

# The end-of-central-directory record that ends every zip: signature, disk number, disk with the
# directory, entries on this disk, entries in total, directory size, directory offset, comment length.
_END_RECORD: Final = struct.Struct("<4s4H2LH")
_END_SIGNATURE: Final = b"PK\x05\x06"
_MAX_COMMENT: Final = 65_535  # the comment after the record is at most this long


def check_file_limits(
    path: Path, *, max_bytes: int | None = None, max_unpacked: int | None = None
) -> None:
    """Raise ``LoadError`` if the file is over a limit; otherwise return.

    The arguments are for the tests (a limit of a few bytes instead of a file of 100 MB); production
    callers pass nothing and get the module's constants, read at call time.
    """
    limit = MAX_FILE_BYTES if max_bytes is None else max_bytes
    size = path.stat().st_size
    if size > limit:
        raise LoadError(
            f"the file is {size / 1_000_000:.1f} MB, over the {limit / 1_000_000:g} MB limit"
        )
    if path.suffix.lower() in ZIP_EXTENSIONS:
        _check_archive(path, MAX_UNPACKED_BYTES if max_unpacked is None else max_unpacked)


def _declared_member_count(path: Path) -> int | None:
    """How many entries the zip's end record says it has, or ``None`` if it has no such record.

    A zip64 archive with more than 65,534 entries writes 65,535 here, which is over any limit.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _END_RECORD.size - _MAX_COMMENT))
        tail = handle.read()
    at = tail.rfind(_END_SIGNATURE)
    if at < 0 or len(tail) - at < _END_RECORD.size:
        return None
    _signature, _disk, _directory_disk, _here, total, *_ = _END_RECORD.unpack_from(tail, at)
    return int(total)


def _check_archive(path: Path, unpacked_limit: int) -> None:
    declared = _declared_member_count(path)
    if declared is not None and declared > MAX_ZIP_MEMBERS:
        raise LoadError(f"the file holds {declared} parts, over the {MAX_ZIP_MEMBERS} limit")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile:
        return  # not an archive after all: the loader reports a damaged file in its own words
    unpacked = sum(member.file_size for member in members)
    if unpacked > unpacked_limit:
        raise LoadError(
            f"the file unpacks to {unpacked / 1_000_000:.0f} MB, over the "
            f"{unpacked_limit / 1_000_000:g} MB limit"
        )


def check_page_limit(pages: int, *, limit: int | None = None) -> None:
    """Raise ``LoadError`` if a PDF has more pages than ``limit`` (default ``MAX_PDF_PAGES``)."""
    cap = MAX_PDF_PAGES if limit is None else limit
    if pages > cap:
        raise LoadError(f"the PDF has {pages} pages, over the {cap}-page limit")
