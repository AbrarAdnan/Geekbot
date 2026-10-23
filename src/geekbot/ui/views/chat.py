"""The chat page (step S4.4): ask a question, watch the answer arrive, see where it came from.

**The thread lives in the address.** The current conversation is the ``?thread=<id>`` of the page's
URL (and, so that the Knowledge or Settings page and back does not forget it, in the session). A
browser reload therefore comes back to the same conversation, and the messages, their sources and the
cited marks are read from the server, which saved them (S4.2): the UI keeps no history of its own.

**What a rerun does.** Reading the thread and the thread list is all that happens on a plain rerun.
A question is sent only from the chat input's own branch, once per submitted question. While an answer
streams, any other click re-runs the script and abandons the stream; the server then sees the client
go away and forgets that turn. That is the cost of Streamlit's model; the page says so rather than
hiding it.

**CSV conversations (C5).** A blank conversation starts with a choice: the documents, or one CSV file.
For a CSV file the page shows a file chooser and a button, and the file is sent **only from inside
``if st.button(...)``** and the chooser is replaced (a new ``key``) once the file is loaded: the same
two rules as the Knowledge page, for the same reason (v1 bug #1: a rerun must never repeat work). The
server loads the table and the page shows what it found; the first question then makes the thread a
CSV conversation for good. A saved CSV conversation is continued like a chat one. The reply is the
package's own rendering, Markdown with the SQL that was run and a preview of its result: the server's
output filter (G3) has removed images and outside links from it, as from every answer. What the
*file* says about itself (its name, its columns, the loader's notes) is untrusted text and is shown
with ``st.text``.

**What is shown as what.** The answer is Markdown (the server's output filter has already removed
images and links to places the sources do not name). Everything that came from a document, a source's
title, path and passage, and what the user typed, is shown with ``st.text``: plain text, never
interpreted (``server/schemas.py`` says why).
"""

from typing import Any, Final

import streamlit as st

from geekbot.ui.client import ApiError, ClientError, GeekbotClient
from geekbot.ui.display import (
    TurnView,
    assistant_message_id,
    csv_upload_lines,
    passages_key,
    source_header,
    thread_label,
    turn_details,
)

__all__ = ["render"]

THREAD_PARAM: Final = "thread"  # the query parameter that carries the thread id
THREAD_KEY: Final = "geekbot_thread"  # the same id in the session, for coming back from other pages
THINK_KEY: Final = "geekbot_think"
PASSAGES_KEY: Final = "geekbot_passages"  # {thread/message id: {n: passage}} for this session
START_KEY: Final = "geekbot_start_with"  # the radio of a blank conversation
CSV_LOADED_KEY: Final = "geekbot_csv_loaded"  # {thread id: the upload summary} until it is saved
CSV_CHOOSER_KEY: Final = "geekbot_csv_chooser"  # counts uploads, to give the chooser a new key
DOCUMENTS: Final = "My documents"
CSV_FILE: Final = "A CSV file"
CSV_TYPES: Final = ["csv", "tsv"]
CURSOR: Final = " ▌"
BUSY: Final = 409

# What ``st.empty()`` gives back: a slot on the page that can be written to again and again.
# Streamlit does not promise a stable public name for its type, so this alias says only what the
# page uses it for.
type Placeholder = Any


def render(client: GeekbotClient) -> None:
    st.title("Chat")
    try:
        thread_id = _current_thread()
        _sidebar(client, thread_id)
        thread = client.thread(thread_id) if thread_id else None
        loaded = _csv_loaded(thread_id) if thread is None else None
        csv_file = _csv_file_of(thread, loaded)
        if thread_id is None and _start_choice() == CSV_FILE:
            _csv_start(client)  # nothing to ask until a file is loaded
            return
        if csv_file is None:
            _banner(client)
        _history(thread_id, thread, loaded)
    except ClientError as error:
        st.error(str(error))
        return
    prompt = f"Ask about {csv_file}" if csv_file is not None else "Ask about your documents"
    question = st.chat_input(prompt)
    if question:
        _ask(client, thread_id, question, csv=csv_file is not None)


# --- CSV conversations (C5) --------------------------------------------------------------------


def _csv_loaded(thread_id: str | None) -> dict[str, Any] | None:
    """The upload summary of a CSV file loaded in this session into a thread not saved yet."""
    if thread_id is None:
        return None
    summary = st.session_state.get(CSV_LOADED_KEY, {}).get(thread_id)
    return summary if isinstance(summary, dict) else None


def _csv_file_of(thread: dict[str, Any] | None, loaded: dict[str, Any] | None) -> str | None:
    """The file name if the conversation is about a CSV file, else ``None`` (a chat conversation).

    A saved thread says so itself. A thread that has a file but no answered turn is not saved on
    the server yet, so this session's own record of the upload tells.
    """
    if thread is not None:
        if thread.get("mode") != "csv":
            return None
        return str(thread.get("csv_file") or "the CSV file")
    if loaded is not None:
        return str(loaded.get("file") or "the CSV file")
    return None


def _start_choice() -> str:
    """What a blank conversation will be about: the knowledge base, or one CSV file."""
    return str(
        st.radio(
            "This conversation is about",
            [DOCUMENTS, CSV_FILE],
            key=START_KEY,
            horizontal=True,
            help="A CSV conversation answers numeric questions from one table and shows the SQL it "
            "ran. It uses another model, so the first question swaps models and takes a few "
            "seconds. The choice cannot be changed once the conversation has begun.",
        )
    )


def _csv_start(client: GeekbotClient) -> None:
    """The file chooser and its button. Nothing is sent unless the button was pressed this run."""
    chooser = st.session_state.get(CSV_CHOOSER_KEY, 0)
    chosen = st.file_uploader(
        "A CSV or TSV file",
        type=CSV_TYPES,
        accept_multiple_files=False,
        key=f"geekbot-csv-{chooser}",
        help="The file is kept on the server, in its uploads folder.",
    )
    if st.button("Load the file", disabled=chosen is None) and chosen is not None:
        _load_csv(client, chosen.name, chosen.getvalue())


def _load_csv(client: GeekbotClient, name: str, data: bytes) -> None:
    """Reserve a thread, send the file, remember what the server found, and show the thread."""
    with st.spinner(f"Loading {name} ..."):
        try:
            thread_id = client.new_thread("csv")
            summary = client.upload_csv(thread_id, name, data)
        except ClientError as error:
            st.error(str(error))
            return
    st.session_state.setdefault(CSV_LOADED_KEY, {})[thread_id] = summary
    st.session_state[CSV_CHOOSER_KEY] = st.session_state.get(CSV_CHOOSER_KEY, 0) + 1
    _select(thread_id)
    st.rerun()


# --- which conversation ----------------------------------------------------------------------


def _current_thread() -> str | None:
    """The thread id from the address, else the one this session last used, else none."""
    from_address = st.query_params.get(THREAD_PARAM)
    if from_address:
        st.session_state[THREAD_KEY] = from_address
        return from_address
    remembered = st.session_state.get(THREAD_KEY)
    if remembered:
        st.query_params[THREAD_PARAM] = remembered
        return str(remembered)
    return None


def _select(thread_id: str | None) -> None:
    """Make ``thread_id`` the current conversation (``None``: a blank one). A button's callback."""
    if thread_id:
        st.session_state[THREAD_KEY] = thread_id
        st.query_params[THREAD_PARAM] = thread_id
    else:
        st.session_state.pop(THREAD_KEY, None)
        st.query_params.clear()


def _sidebar(client: GeekbotClient, current: str | None) -> None:
    with st.sidebar:
        st.button("New chat", on_click=_select, args=(None,))
        st.toggle(
            "Think harder",
            key=THINK_KEY,
            help="Let the model reason before it answers (document conversations only). Slower, "
            "and only a model with a thinking mode can do it (the target model gemma4:12b can; "
            "gemma3:4b cannot).",
        )
        st.subheader("Conversations")
        threads = client.threads()
        if not threads:
            st.caption("None yet.")
        for summary in threads:
            st.button(
                thread_label(summary),
                key=f"thread-{summary['thread_id']}",
                on_click=_select,
                args=(summary["thread_id"],),
                disabled=summary["thread_id"] == current,
            )
        if current and any(summary["thread_id"] == current for summary in threads):
            _delete_button(client, current)


def _delete_button(client: GeekbotClient, thread_id: str) -> None:
    """Delete the current conversation, after a tick in the box: it cannot be undone."""
    sure = st.checkbox("Really delete this conversation", key=f"sure-{thread_id}")
    if not st.button("Delete", disabled=not sure, key=f"delete-{thread_id}"):
        return
    try:
        client.delete_thread(thread_id)
    except ApiError as error:
        st.error(error.detail)
        return
    st.session_state.get(CSV_LOADED_KEY, {}).pop(thread_id, None)
    _select(None)
    st.rerun()


# --- what is on the page ---------------------------------------------------------------------


def _banner(client: GeekbotClient) -> None:
    """One line about the state of the server when it matters: an empty library, a cold model."""
    health = client.health()
    ready = int(health.get("documents", {}).get("ready", 0))
    if ready == 0:
        st.info("The knowledge base has no ready documents. Add some on the Knowledge page first.")
    if not health.get("chat_ready", True):
        st.caption(
            f"The chat model {health.get('chat_model', '')} is not loaded yet: "
            "the first question loads it and waits."
        )


def _history(
    thread_id: str | None, thread: dict[str, Any] | None, loaded: dict[str, Any] | None = None
) -> None:
    if loaded is not None:  # a file was just loaded and nothing has been asked about it yet
        with st.chat_message("assistant"):
            st.text("\n".join(csv_upload_lines(loaded)))
    elif thread is not None and thread.get("mode") == "csv":
        st.caption(f"A conversation about {thread.get('csv_file') or 'a CSV file'}.")
    for message in (thread or {}).get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["content"])
            else:
                st.markdown(message["content"])
                _saved_sources(thread_id or "", message)


def _saved_sources(thread_id: str, message: dict[str, Any]) -> None:
    """The sources of a remembered answer: references, and passages if this session has them."""
    references = message.get("sources") or []
    if not references:
        return
    cited = set(message.get("cited") or [])
    kept = st.session_state.get(PASSAGES_KEY, {})
    passages = kept.get(passages_key(thread_id, message["id"]), {})
    with st.expander(f"Sources ({len(references)})"):
        for reference in references:
            lines = [source_header(reference, cited=reference["n"] in cited), reference["path"]]
            passage = passages.get(reference["n"])
            if passage:
                lines += ["", passage]
            st.text("\n".join(lines))


# --- asking ----------------------------------------------------------------------------------


def _ask(client: GeekbotClient, thread_id: str | None, question: str, *, csv: bool = False) -> None:
    try:
        if thread_id is None:
            thread_id = client.new_thread()
            _select(thread_id)
    except ClientError as error:
        st.error(str(error))
        return
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        # The CSV model is not asked to reason, whatever the toggle says.
        think = bool(st.session_state.get(THINK_KEY)) and not csv
        view = _stream(client, thread_id, question, think=think, csv=csv)
        if view is not None and view.done is not None:
            _fresh_sources(thread_id, view.done)
            with st.expander("Details"):
                st.text("\n".join(turn_details(view.done)))


def _stream(
    client: GeekbotClient, thread_id: str, question: str, *, think: bool, csv: bool = False
) -> TurnView | None:
    """Send the question and paint the answer as it arrives. ``None`` if no answer came.

    A CSV question says so (``mode="csv"``): if the server has lost the conversation's file it then
    answers 409 and says to upload it again, and does not answer from the documents instead. A
    document question asks for step events (``steps=True``), so the line under the question says
    "Searching ...", "Ranking ...", "Writing ..." before the first word arrives (S3.6).
    """
    view = TurnView()
    status = st.empty()
    reasoning = _reasoning_box() if think else None
    answer = st.empty()
    try:
        for event in client.ask(
            thread_id, question, think=think, mode="csv" if csv else None, steps=not csv
        ):
            view.feed(event)
            _paint(view, status, answer, reasoning)
    except ApiError as error:
        st.error(_refusal(error))
        return None
    except ClientError as error:
        status.empty()
        answer.empty()
        st.error(str(error))
        return None
    status.empty()
    if view.error is not None:
        answer.empty()  # the failed turn is not remembered by the server; nothing of it stays here
        st.error(view.error)
        return None
    answer.markdown(view.answer)
    return view


def _reasoning_box() -> Placeholder:
    with st.expander("The model's reasoning"):
        return st.empty()


def _paint(
    view: TurnView, status: Placeholder, answer: Placeholder, reasoning: Placeholder | None
) -> None:
    if view.status:
        status.caption(view.status)
    else:
        status.empty()
    if reasoning is not None and view.thinking:
        reasoning.text(view.thinking)
    if view.text:
        answer.markdown(view.text + CURSOR)


def _refusal(error: ApiError) -> str:
    if error.status == BUSY:
        return f"{error.detail}. Another question is still running; wait for its answer."
    return error.detail


def _fresh_sources(thread_id: str, done: dict[str, Any]) -> None:
    """The sources of the answer that just arrived, with their passages; kept for later reruns."""
    sources = done.get("sources") or []
    if not sources:
        return
    cited = set(done.get("cited") or [])
    key = passages_key(thread_id, assistant_message_id(int(done["turn"])))
    st.session_state.setdefault(PASSAGES_KEY, {})[key] = {s["n"]: s["text"] for s in sources}
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            header = source_header(source, cited=source["n"] in cited)
            st.text("\n".join([header, source["path"], "", source["text"]]))
