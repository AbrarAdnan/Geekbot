"""Loaders: file in, text out (step S1.1, PLAN.md §6.6).

``load_file`` turns one file into a ``LoadedDocument`` or raises ``LoadError``; ``load_paths``
does it for many files and never stops at a bad one, returning a ``LoadReport`` of what loaded,
what failed and what was skipped. That second part answers v1's finding #4 (PLAN.md §5): one
unreadable file used to abort a whole folder.

The extension picks the loader. Content sniffing is MarkItDown's own business inside the office
loader; everywhere else a wrong extension is the file's problem, reported as a failure.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geekbot.ingest.loaders.model import DocKind, LoadedDocument, LoadError, Segment
from geekbot.ingest.loaders.notebook import load_notebook
from geekbot.ingest.loaders.office import KINDS as OFFICE_KINDS
from geekbot.ingest.loaders.office import load_office
from geekbot.ingest.loaders.pdf import load_pdf
from geekbot.ingest.loaders.tabular import TABULAR_EXTENSIONS, load_table
from geekbot.ingest.loaders.text import TEXT_LIKE_EXTENSIONS, load_text

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocKind",
    "FileProblem",
    "LoadError",
    "LoadReport",
    "LoadedDocument",
    "Segment",
    "UnsupportedFormatError",
    "load_file",
    "load_paths",
]

_MAX_REASON_CHARS: Final = 300

_LOADERS: Final[dict[str, Callable[[Path], LoadedDocument]]] = {
    ".pdf": load_pdf,
    ".ipynb": load_notebook,
    **dict.fromkeys(OFFICE_KINDS, load_office),
    **dict.fromkeys(TABULAR_EXTENSIONS, load_table),
    **dict.fromkeys(TEXT_LIKE_EXTENSIONS, load_text),
}

SUPPORTED_EXTENSIONS: Final = frozenset(_LOADERS)


class UnsupportedFormatError(LoadError):
    """No loader exists for this file type. Reported as *skipped*, not as a failure."""


@dataclass(frozen=True)
class FileProblem:
    """A file that did not load, and the reason in plain words."""

    path: Path
    reason: str


@dataclass(frozen=True)
class LoadReport:
    """The outcome of loading many files, in the order they were given."""

    loaded: tuple[LoadedDocument, ...]
    failed: tuple[FileProblem, ...]  # a supported type that could not be read
    skipped: tuple[FileProblem, ...]  # a type with no loader

    def summary(self) -> str:
        """``3 loaded, 1 failed, 2 skipped``."""
        return f"{len(self.loaded)} loaded, {len(self.failed)} failed, {len(self.skipped)} skipped"


def load_file(path: Path) -> LoadedDocument:
    """Load one file. Raises ``LoadError`` (``UnsupportedFormatError`` for an unknown type).

    Third-party parsers can raise almost anything on a damaged file (``KeyError`` from a broken
    zip, ``ValueError`` from a bad date). All of it is wrapped in ``LoadError`` with the original
    as ``__cause__``, so the callers have one exception to handle and the report one kind of
    reason to show.
    """
    if not path.is_file():
        raise LoadError("not found, or not a regular file")
    extension = path.suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise UnsupportedFormatError(f"no loader for {extension or 'files without an extension'}")
    try:
        return loader(path)
    except LoadError:
        raise
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"[:_MAX_REASON_CHARS]
        raise LoadError(f"could not be read ({detail})") from error


def load_paths(paths: Iterable[Path]) -> LoadReport:
    """Load every file; collect failures instead of raising on the first."""
    loaded: list[LoadedDocument] = []
    failed: list[FileProblem] = []
    skipped: list[FileProblem] = []
    for path in paths:
        try:
            loaded.append(load_file(path))
        except UnsupportedFormatError as error:
            skipped.append(FileProblem(path, str(error)))
        except LoadError as error:
            failed.append(FileProblem(path, str(error)))
    return LoadReport(tuple(loaded), tuple(failed), tuple(skipped))
