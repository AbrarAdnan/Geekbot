"""Jupyter notebook loader (step S1.1).

A notebook is JSON. What a reader wants from it is the prose in the markdown cells and the code in
the code cells, in order; the rest is dropped on purpose:

- **Outputs** are not indexed. They are often huge (base64 images, dataframes), they change every
  time the notebook runs, and they are text nobody typed, which makes them poor evidence.
- **Raw cells** and cell metadata carry no content worth searching.
"""

import json
from pathlib import Path
from typing import Final

from geekbot.ingest.loaders.common import read_text_file
from geekbot.ingest.loaders.model import LoadedDocument, LoadError, Segment

# Notebooks with embedded outputs are much bigger than their text, so they get a higher cap than
# a plain text file.
MAX_NOTEBOOK_BYTES: Final = 50_000_000

_DEFAULT_LANGUAGE: Final = "python"


def load_notebook(path: Path) -> LoadedDocument:
    """Markdown cells as written, code cells as fenced blocks, in notebook order.

    The code fence carries the notebook's language (from its metadata, else ``python``), so the
    chunker and the model both see which language a block is. The language is also set on the
    document. Only notebook format 4 and later is read (it has a top-level ``cells`` list).
    """
    try:
        notebook = json.loads(read_text_file(path, max_bytes=MAX_NOTEBOOK_BYTES))
    except json.JSONDecodeError as error:
        raise LoadError(f"not valid JSON: {error.msg} (line {error.lineno})") from error
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise LoadError("not a Jupyter notebook (no 'cells' list; format 4 or later is needed)")
    language = _language(notebook)
    parts: list[str] = []
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            continue
        source = _source(cell.get("source"))
        if not source.strip():
            continue
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            parts.append(source.strip())
        elif cell_type == "code":
            parts.append(f"```{language}\n{source.rstrip()}\n```")
    if not parts:
        raise LoadError("the notebook has no markdown or code cells with content")
    return LoadedDocument(
        path=path,
        kind="notebook",
        title=path.name,
        segments=(Segment("\n\n".join(parts)),),
        language=language,
    )


def _language(notebook: dict[str, object]) -> str:
    """The code language the notebook declares (``metadata.language_info.name``), else Python."""
    metadata = notebook.get("metadata")
    if isinstance(metadata, dict):
        info = metadata.get("language_info")
        if isinstance(info, dict) and isinstance(info.get("name"), str) and info["name"]:
            return str(info["name"]).lower()
    return _DEFAULT_LANGUAGE


def _source(value: object) -> str:
    """A cell's ``source``: a string, or a list of lines that already end in newlines."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(line for line in value if isinstance(line, str))
    return ""
