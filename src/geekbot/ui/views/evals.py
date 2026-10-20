"""The evals page (step S4.4): read the result files the evaluation commands wrote. Read-only.

``geekbot eval ...`` writes its reports and result tables under ``evals/results/`` (S2.6, S3.5). This
page lists them and shows one. It does not run anything: an evaluation loads models onto the GPU and
takes minutes to hours, which a click in a browser should not start (GPU etiquette, PLAN.md section 2).
The labelling view for the answer judge (S5.2) is not here; it comes with that step.

**These are files, not storage.** The UI does not import the storage packages, and reading a Markdown
report off the disk is not what that rule is about. The folder is ``evals/results`` relative to where
``geekbot up`` was started (the repository folder, as for ``geekbot ingest evals/corpus``).

**What is shown as what.** A ``.md`` report is written by ``geekbot eval report`` from numbers and is
shown as Markdown, so its tables render. A ``.json`` file is shown as data (``st.json``): result
files can hold questions and model answers, which are text from elsewhere and must not be rendered as
Markdown. A file is only read if its real path is inside the results folder (a link in there that
points elsewhere is refused) and it is under ``MAX_SHOWN_BYTES``.
"""

import json
from pathlib import Path
from typing import Final

import streamlit as st

from geekbot.ui.client import GeekbotClient
from geekbot.ui.display import human_bytes

__all__ = ["list_results", "read_result", "render"]

RESULTS_DIR: Final = Path("evals/results")
SHOWN_SUFFIXES: Final = frozenset({".md", ".json"})
MAX_FILES: Final = 200
MAX_SHOWN_BYTES: Final = 2_000_000


def list_results(root: Path, limit: int = MAX_FILES) -> list[Path]:
    """The ``.md`` and ``.json`` files under ``root``, as paths relative to it, sorted by name."""
    if not root.is_dir():
        return []
    found = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.suffix.lower() in SHOWN_SUFFIXES and path.is_file()
    )
    return found[:limit]


def read_result(root: Path, relative: Path) -> str:
    """The text of one result file, or ``ValueError`` saying why it will not be shown.

    The real path (links resolved) must be inside ``root``, and the file must not be larger than
    ``MAX_SHOWN_BYTES``.
    """
    real_root = root.resolve()
    real = (root / relative).resolve()
    if not real.is_relative_to(real_root):
        raise ValueError(f"{relative.as_posix()} is outside the results folder")
    size = real.stat().st_size
    if size > MAX_SHOWN_BYTES:
        name = relative.as_posix()
        raise ValueError(f"{name} is {human_bytes(size)}: too large to show, open the file itself")
    return real.read_text(encoding="utf-8", errors="replace")


def render(client: GeekbotClient) -> None:
    del client  # this page reads files, not the server
    st.title("Evaluation results")
    st.caption(
        "Read-only: results written by `geekbot eval ...`. "
        "Run those in a terminal, with the GPU free."
    )
    files = list_results(RESULTS_DIR)
    if not files:
        st.info(
            f"No result files in `{RESULTS_DIR.resolve()}`. Start `geekbot up` from the repository "
            "folder, and run an evaluation (see the README) to make some."
        )
        return
    chosen = st.selectbox("Result file", files, format_func=lambda path: path.as_posix())
    try:
        text = read_result(RESULTS_DIR, chosen)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    if chosen.suffix.lower() == ".md":
        st.markdown(text)
        return
    try:
        st.json(json.loads(text))
    except json.JSONDecodeError:
        st.text(text)
