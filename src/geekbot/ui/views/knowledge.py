"""The knowledge page (step S4.4): what is indexed, add files, remove a document.

✍️ Written by Claude (prototyping mode, D-047, step S4.4); on the ✍️ backlog in STATUS.md.

**Why this page is the lesson of the step.** v1's whole indexing ran in the Streamlit script: the file
uploader's loop was at the top level, so *every* interaction (a chat message, a button, a sidebar
toggle) re-ran it, saved each uploaded file again and embedded each one again (v1 bug #1,
PLAN.md section 5). Streamlit re-runs the script from the top on every interaction; that is its model,
and code that costs something must not sit where the model runs it unasked. Here:

1. Nothing is sent to the server except inside ``if st.button(...)``. A plain rerun reads the document
   list and draws it, which is free.
2. After an upload the uploader is *replaced* (its ``key`` changes), so the files it held cannot be
   sent a second time by a later click on some other button.
3. The server is the one that decides whether a file changed: sending the same bytes again is cheap
   (the stored file has the same hash and the ingest reports it "unchanged", 0 embeddings, S1.6), so
   even a repeated click cannot cost an embedding.

The work is in three plain functions (``add_uploads``, ``add_paths``, ``remove_document``) that take
the client and a ``wait`` function and return ``Outcome`` rows, and one parser (``parse_paths``).
The page only calls them and draws what they return, so the tests call them without Streamlit.

**Untrusted text.** File names, titles and error messages come from the user's files. They are shown
with ``st.text`` and in a table (plain text), never as Markdown.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import streamlit as st

from geekbot.ui.client import ClientError, GeekbotClient, wait_for_job
from geekbot.ui.display import document_rows, job_outcome, state_counts

__all__ = ["Outcome", "add_paths", "add_uploads", "parse_paths", "remove_document", "render"]

RESULTS_KEY: Final = "geekbot_kb_results"
GENERATION_KEY: Final = "geekbot_kb_uploader"
QUOTES: Final = "\"'"

type Waiter = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class Outcome:
    """One line of what an action did: what it was about, whether it worked, and why not."""

    label: str
    ok: bool
    message: str


def parse_paths(raw: str) -> list[str]:
    """The paths typed in the box: one per line, quotes and blanks dropped, repeats dropped.

    Windows' "Copy as path" puts the path in double quotes, so a pasted path arrives quoted; the
    server wants it bare. Nothing else is changed: whether a path is allowed is the server's decision
    (``PathPolicy``), and its refusal says why.
    """
    paths: list[str] = []
    for line in raw.splitlines():
        path = line.strip().strip(QUOTES).strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def add_uploads(
    client: GeekbotClient, files: Sequence[tuple[str, bytes]], *, wait: Waiter
) -> list[Outcome]:
    """Send each file to the server, one after the other, and wait for its ingest to finish.

    One at a time because the server runs jobs one at a time anyway, and so the outcome of each file
    is its own. A file that fails does not stop the others.
    """
    outcomes: list[Outcome] = []
    for name, data in files:
        try:
            accepted = client.upload(name, data)
            job = wait(accepted["job"]["id"])
        except ClientError as error:
            outcomes.append(Outcome(name, ok=False, message=str(error)))
            continue
        ok, message = job_outcome(job)
        if accepted.get("existed"):
            message += " (the server already had this exact file)"
        outcomes.append(Outcome(name, ok=ok, message=message))
    return outcomes


def add_paths(client: GeekbotClient, raw: str, *, wait: Waiter) -> list[Outcome]:
    """Ask the server to index the typed paths (files or folders it is allowed to read)."""
    paths = parse_paths(raw)
    label = ", ".join(paths)
    if not paths:
        return [Outcome("", ok=False, message="type at least one full path")]
    try:
        accepted = client.ingest(paths)
        job = wait(accepted["id"])
    except ClientError as error:
        return [Outcome(label, ok=False, message=str(error))]
    ok, message = job_outcome(job)
    return [Outcome(label, ok=ok, message=message)]


def remove_document(client: GeekbotClient, doc_id: str, label: str, *, wait: Waiter) -> Outcome:
    """Remove a document from the index. The file itself, wherever it is, is not touched."""
    try:
        accepted = client.delete_document(doc_id)
        job = wait(accepted["id"])
    except ClientError as error:
        return Outcome(label, ok=False, message=str(error))
    ok, message = job_outcome(job)
    return Outcome(label, ok=ok, message="removed from the index" if ok else message)


# --- the page --------------------------------------------------------------------------------


def _progress_line(job: dict[str, Any]) -> str:
    return f"{job['kind']} job {job['state']}: {len(job.get('progress', []))} file(s) done"


def _waiter(client: GeekbotClient, progress: Any) -> Waiter:
    """A ``wait`` that polls the server and keeps one progress line up to date in ``progress``."""

    def show(job: dict[str, Any]) -> None:
        progress.text(_progress_line(job))

    def wait(job_id: str) -> dict[str, Any]:
        return wait_for_job(client, job_id, on_update=show)

    return wait


def _show_outcomes(outcomes: Sequence[Outcome]) -> None:
    for outcome in outcomes:
        mark = "OK  " if outcome.ok else "FAIL"
        st.text(f"{mark} {outcome.label}: {outcome.message}")


def render(client: GeekbotClient) -> None:
    st.title("Knowledge base")
    try:
        documents = client.documents()
        formats = client.formats()
    except ClientError as error:
        st.error(str(error))
        return
    st.caption(f"{len(documents)} document(s): {state_counts(documents)}")
    last = st.session_state.pop(RESULTS_KEY, None)
    if last:
        _show_outcomes(last)
    _upload_section(client, formats)
    _path_section(client)
    _documents_section(client, documents)


def _upload_section(client: GeekbotClient, formats: list[str]) -> None:
    st.subheader("Add files")
    generation = st.session_state.setdefault(GENERATION_KEY, 0)
    files = st.file_uploader(
        "Choose files to add",
        accept_multiple_files=True,
        type=sorted(extension.lstrip(".") for extension in formats),
        key=f"uploader-{generation}",
    )
    if not st.button("Add to the knowledge base", disabled=not files):
        return
    with st.status("Adding files...", expanded=True) as status:
        wait = _waiter(client, st.empty())
        outcomes = add_uploads(client, [(f.name, f.getvalue()) for f in files], wait=wait)
        failed = sum(1 for outcome in outcomes if not outcome.ok)
        status.update(
            label="Done" if not failed else f"Done, {failed} file(s) failed",
            state="complete" if not failed else "error",
        )
    st.session_state[RESULTS_KEY] = outcomes
    st.session_state[GENERATION_KEY] = generation + 1  # a new uploader: the old files are gone
    st.rerun()


def _path_section(client: GeekbotClient) -> None:
    with st.expander("Add files or folders by path"):
        st.caption(
            "Full paths, one per line, inside the folders the server may read "
            "(GEEKBOT_ALLOWED_ROOTS in .env). Uploading above needs no folder."
        )
        raw = st.text_area("Paths", key="kb-paths", label_visibility="collapsed")
        if not st.button("Add these paths", disabled=not raw.strip()):
            return
        with st.status("Adding...", expanded=True) as status:
            wait = _waiter(client, st.empty())
            outcomes = add_paths(client, raw, wait=wait)
            status.update(label="Done", state="complete" if outcomes[0].ok else "error")
        st.session_state[RESULTS_KEY] = outcomes
        st.rerun()


def _documents_section(client: GeekbotClient, documents: list[dict[str, Any]]) -> None:
    st.subheader("Documents")
    st.button("Refresh")
    if not documents:
        st.caption("Nothing is indexed yet.")
        return
    st.dataframe(document_rows(documents), hide_index=True)
    by_id = {document["doc_id"]: document for document in documents}
    chosen = st.selectbox(
        "Remove a document from the index",
        options=list(by_id),
        format_func=lambda doc_id: f"{by_id[doc_id]['title']}  ({by_id[doc_id]['path']})",
    )
    st.caption("Only the index entry goes; the file itself is not deleted.")
    if chosen is not None and st.button("Remove"):
        label = str(by_id[chosen]["title"])
        with st.status("Removing...", expanded=False) as status:
            wait = _waiter(client, st.empty())
            outcome = remove_document(client, chosen, label, wait=wait)
            status.update(label="Done", state="complete" if outcome.ok else "error")
        st.session_state[RESULTS_KEY] = [outcome]
        st.rerun()
