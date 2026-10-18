"""Plain text, Markdown and source-code loader (step S1.1).

The simplest loader, and the one that decides what "a text file" means for the whole pipeline:
which extensions count, which are code (S1.2 splits code on its own syntax), and what happens to
a file that is not text after all.
"""

from pathlib import Path
from typing import Final

from geekbot.ingest.loaders.common import MAX_TEXT_BYTES, clean_text, read_text_file
from geekbot.ingest.loaders.model import DocKind, LoadedDocument, LoadError, Segment

MARKDOWN_EXTENSIONS: Final = frozenset({".md", ".markdown"})

# Data and prose formats that are read as plain text: no language-aware splitting for them.
TEXT_EXTENSIONS: Final = frozenset(
    {".txt", ".text", ".rst", ".log", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
)

# Source code, mapped to a language name. The names follow LangChain's ``Language`` values where
# it has one (``js``, ``ts``, ``cpp``, ``csharp``, ...), so S1.2's code splitter can use them
# directly; the rest (``bash``, ``sql``) fall back to generic splitting there.
CODE_LANGUAGES: Final[dict[str, str]] = {
    ".py": "python",
    ".js": "js",
    ".mjs": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
}

TEXT_LIKE_EXTENSIONS: Final = MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS | frozenset(CODE_LANGUAGES)


def load_text(path: Path, *, max_bytes: int = MAX_TEXT_BYTES) -> LoadedDocument:
    """Read a text, Markdown or source-code file as one segment.

    ✍️ Written by Claude in prototyping mode (D-047, step S1.1); on the ✍️ backlog in STATUS.md.

    What it does, in order:

    1. Pick the kind from the extension: ``markdown``, ``code`` (with its language, from
       ``CODE_LANGUAGES``) or plain ``text``. The extension decides, not the content: guessing a
       language from the text would make the same file index differently after an edit.
    2. Read and decode the file (``read_text_file``: UTF-8 first, then a guessed legacy
       encoding; a binary file or one over ``max_bytes`` is a ``LoadError``).
    3. Tidy the text (``clean_text``: newlines, NUL, BOM, outer blank lines). Markdown is left
       otherwise untouched, because S1.2 splits it at its ``#`` headers.
    4. A file with nothing left is a ``LoadError("the file is empty")``, never an empty document.

    The title is the file name. Code keeps its indentation.
    """
    extension = path.suffix.lower()
    kind: DocKind
    language: str | None = None
    if extension in MARKDOWN_EXTENSIONS:
        kind = "markdown"
    elif extension in CODE_LANGUAGES:
        kind = "code"
        language = CODE_LANGUAGES[extension]
    else:
        kind = "text"
    text = clean_text(read_text_file(path, max_bytes=max_bytes))
    if not text:
        raise LoadError("the file is empty")
    return LoadedDocument(
        path=path, kind=kind, title=path.name, segments=(Segment(text),), language=language
    )
