"""Receiving a file safely: a plain name, a size cap, a save that is all or nothing (step S4.3).

An uploaded file is the one thing a client can put on this machine's disk, so the three questions are
answered here, in a function that knows nothing about HTTP (the route in ``routes.py`` only hands it
the request body; C5's CSV upload will do the same):

- **Name.** Whatever the client calls the file is untrusted text. ``safe_upload_name`` keeps the last
  part only (no directories, with either slash), keeps letters, digits, ``.``, ``_`` and ``-``,
  lower-cases the extension and refuses an extension the caller did not list. The file is then stored
  as ``<first 16 hex of its SHA-256>-<safe name>`` (PLAN.md section 8.1), so two different files can
  never overwrite each other, the same file uploaded twice is one file (and the ingest of the second
  is "unchanged"), and a name can never be ``..`` or a Windows device name such as ``CON``.
- **Size.** The bytes are counted as they arrive and the upload stops, with HTTP 413, at ``max_bytes``.
  The middleware has already refused a body whose ``Content-Length`` is over the cap; this is the
  count that does not trust that header.
- **All or nothing.** The bytes go to a hidden temporary file in the same folder and are moved to
  their final name only when the whole body has arrived. A client that disconnects, a body over the
  cap or a full disk leaves no half file under a real name: the temporary file is removed in every
  case, even when the request is cancelled (the clean-up is shielded from the cancellation).

All file work runs in worker threads (``anyio``), never on the event loop (PLAN.md section 6.3).

C5's CSV upload (``csvroutes.py``) uses the same ``save_upload`` with the CSV suffixes, and the
two helpers at the end of this module read a stored name back: ``display_name`` gives the name the
user knows, ``stored_name_ok`` says whether a name has the shape ``save_upload`` produces.
"""

import hashlib
import posixpath
import re
import unicodedata
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import anyio
import anyio.to_thread

from geekbot.fsutil import replace_file

__all__ = [
    "MAX_STEM_CHARS",
    "SHA_CHARS",
    "StoredUpload",
    "UploadError",
    "display_name",
    "safe_upload_name",
    "save_upload",
    "stored_name_ok",
]

MAX_STEM_CHARS: Final = 80  # of the name, without its extension
SHA_CHARS: Final = 16  # 64 bits of the digest in the stored name: plenty to tell files apart
_UNSAFE: Final = re.compile(r"[^\w.-]+")  # \w is every letter and digit of every script, and "_"
_SEPARATORS: Final = re.compile(r"[\\/]")
_TEMP_PREFIX: Final = ".part-"
_STORED: Final = re.compile(rf"[0-9a-f]{{{SHA_CHARS}}}-(?P<name>[^\\/:]+)")


class UploadError(ValueError):
    """An upload that cannot be accepted. ``status`` is the HTTP status that says why."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class StoredUpload:
    """A saved upload: where, its SHA-256, its size in bytes, and whether it was already there."""

    path: Path
    sha256: str
    size: int
    existed: bool


def safe_upload_name(filename: str, suffixes: frozenset[str]) -> str:
    """A name that is safe to put in a path, or ``UploadError`` (415) for a type not in ``suffixes``.

    ``suffixes`` are lower-case with the dot (``".pdf"``). The result is a bare file name: the stem
    is cut to ``MAX_STEM_CHARS`` characters, anything that is not a letter, digit, ``.``, ``_`` or ``-``
    becomes ``_``, and dots, underscores and hyphens at either end of the stem are dropped (so no name
    starts with a dot or is empty: an empty stem becomes ``file``).
    """
    base = unicodedata.normalize("NFC", _SEPARATORS.split(filename)[-1]).strip()
    stem, suffix = posixpath.splitext(base)
    suffix = suffix.lower()
    if suffix not in suffixes:
        raise UploadError(
            f"files of type {suffix or '(no extension)'} are not accepted "
            f"(accepted: {', '.join(sorted(suffixes))})",
            status=415,
        )
    cleaned = _UNSAFE.sub("_", stem).strip("._-")[:MAX_STEM_CHARS].strip("._-")
    return f"{cleaned or 'file'}{suffix}"


async def save_upload(
    chunks: AsyncIterable[bytes],
    *,
    filename: str,
    directory: Path,
    max_bytes: int,
    suffixes: frozenset[str],
) -> StoredUpload:
    """Write the body to ``directory/<sha16>-<safe name>`` and say where it went.

    ``UploadError`` with status 415 (a type that is not accepted), 413 (over ``max_bytes``) or 422
    (an empty body). Nothing is left behind in any failure. If the same bytes were saved under the
    same name before, the file is not written again and ``existed`` is true.
    """
    name = safe_upload_name(filename, suffixes)
    await anyio.to_thread.run_sync(partial(directory.mkdir, parents=True, exist_ok=True))
    temporary = directory / f"{_TEMP_PREFIX}{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    size = 0
    try:
        async with await anyio.open_file(temporary, "xb") as handle:
            async for chunk in chunks:
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(f"the file is larger than {max_bytes:,} bytes", status=413)
                digest.update(chunk)
                await handle.write(chunk)
        if size == 0:
            raise UploadError("the file is empty", status=422)
        sha256 = digest.hexdigest()
        final = directory / f"{sha256[:SHA_CHARS]}-{name}"
        existed = await anyio.to_thread.run_sync(final.exists)
        if not existed:
            await anyio.to_thread.run_sync(replace_file, temporary, final)
        return StoredUpload(path=final, sha256=sha256, size=size, existed=existed)
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(partial(temporary.unlink, missing_ok=True))


def stored_name_ok(stored_as: str) -> bool:
    """Whether ``stored_as`` has the shape ``save_upload`` gives a file: ``<16 hex>-<name>``.

    Nothing else may be turned into a path under the uploads folder. The name part has no slash, no
    backslash and no colon (``safe_upload_name`` never makes one), and the whole is not ``.`` or ``..``
    by construction (it starts with hex digits). A conversation's file name is read back from the
    thread database by this check before it is opened, so a corrupt or foreign value cannot name a
    file outside the folder.
    """
    return _STORED.fullmatch(stored_as) is not None


def display_name(stored_as: str) -> str:
    """The name the user gave a file, from its stored name; a name of another shape is returned as is.

    ``<sha16>-sales.csv`` becomes ``sales.csv``. A conversation started from the command line keeps
    the plain name it was given, and that has no prefix to remove.
    """
    match = _STORED.fullmatch(stored_as)
    return match["name"] if match else stored_as
