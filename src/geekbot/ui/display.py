"""Text and folding helpers for the pages, with no Streamlit in them (step S4.4).

The pages are thin on purpose: Streamlit code is awkward to test, so anything that is a decision
(which line to show for a job, what a stream of events adds up to, how a source is labelled) lives
here as a plain function over the dicts the server sends, and a test calls it directly.

**What is untrusted.** The server marks it (``server/schemas.py``): a source's ``text`` and its
``title`` and ``path`` come from the user's documents, which may say anything. This module never
adds markup to them; the pages show them with ``st.text``, which renders no Markdown or HTML. The
answer itself has been through the server's output filter (G3: no images, no links the sources do
not name), so a page may show it as Markdown.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from geekbot.sse import SseEvent

__all__ = [
    "TurnView",
    "assistant_message_id",
    "document_rows",
    "health_rows",
    "human_bytes",
    "job_outcome",
    "one_line",
    "passages_key",
    "source_header",
    "state_counts",
    "thread_label",
    "turn_details",
]

_KILO: Final = 1024
_UNITS: Final = ("B", "KB", "MB", "GB")
DEFAULT_LINE_CHARS: Final = 100
THREAD_LABEL_CHARS: Final = 48
SENTENCE_CHARS: Final = 120
LOADING_MODEL: Final = "loading_model"


def human_bytes(size: int) -> str:
    """``1536`` -> ``1.5 KB``; bytes are shown whole, every larger unit with one decimal."""
    value = float(size)
    unit = _UNITS[0]
    for unit in _UNITS:
        if value < _KILO or unit == _UNITS[-1]:
            break
        value /= _KILO
    return f"{value:.0f} {unit}" if unit == _UNITS[0] else f"{value:.1f} {unit}"


def one_line(text: str, limit: int = DEFAULT_LINE_CHARS) -> str:
    """The text on one line (white space runs become one space), cut with ``...`` past ``limit``."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: max(limit - 3, 0)].rstrip() + "..."


# --- the knowledge base ----------------------------------------------------------------------


def document_rows(documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The rows of the documents table: title, type, state, file name, last change, problem.

    Plain strings only, so the table shows them as they are. The folder is left out on purpose (a
    long path says little); the full path is in the remove list's labels.
    """
    return [
        {
            "Title": str(document["title"]),
            "Type": str(document["kind"]),
            "State": str(document["state"]),
            "File": Path(str(document["path"])).name,
            "Updated": str(document["updated_at"]),
            "Problem": one_line(str(document.get("error") or "")),
        }
        for document in documents
    ]


def state_counts(documents: list[dict[str, Any]]) -> str:
    """``3 ready, 1 failed`` (in the order the states come in), or ``no documents``."""
    counts: dict[str, int] = {}
    for document in documents:
        state = str(document["state"])
        counts[state] = counts.get(state, 0) + 1
    if not counts:
        return "no documents"
    return ", ".join(f"{count} {state}" for state, count in counts.items())


def health_rows(health: dict[str, Any]) -> list[dict[str, str]]:
    """The settings page's table of what the server reports at ``/v1/health``."""
    documents = health.get("documents") or {}
    held = ", ".join(f"{count} {state}" for state, count in documents.items()) or "none"
    loaded = (
        "loaded" if health.get("chat_ready") else "not loaded yet (the first question loads it)"
    )
    rows = [
        ("Status", str(health.get("status", "?"))),
        ("Chat model", f"{health.get('chat_model', '?')}: {loaded}"),
        ("Answering a question now", "yes" if health.get("answering") else "no"),
        ("Embedder", str(health.get("embedder", "?"))),
        ("Documents", held),
        ("Chunks", str(health.get("chunks", "?"))),
        ("Vectors", str(health.get("vectors", "?"))),
        ("Jobs running", str(health.get("jobs_active", "?"))),
    ]
    return [{"Setting": name, "Value": value} for name, value in rows]


def job_outcome(job: dict[str, Any]) -> tuple[bool, str]:
    """Whether a finished job succeeded, and one line that says what happened.

    An ingest job's result has a ``summary`` (``2 ingested, 1 unchanged, 0 failed, ...``), the files
    that failed with their reasons, and the files the server refused to read. A job that raised has
    only an ``error``. A delete job has neither summary nor files and is simply done.
    """
    if job.get("state") == "failed":
        return False, f"failed: {job.get('error') or 'the server gave no reason'}"
    result = job.get("result") or {}
    parts = [str(result.get("summary") or "done")]
    for outcome in result.get("files", []):
        if outcome.get("error"):
            name = Path(str(outcome.get("path", ""))).name
            parts.append(f"{name}: {one_line(str(outcome['error']))}")
    refused = int(result.get("refused_count", 0))
    if refused:
        parts.append(f"{refused} file(s) refused: outside the folders the server may read")
    return not result.get("failed", False), "; ".join(parts)


# --- threads and sources ---------------------------------------------------------------------


def thread_label(summary: dict[str, Any]) -> str:
    """The sidebar label of a thread: how its first question starts, else its id."""
    preview = one_line(str(summary.get("preview") or ""), THREAD_LABEL_CHARS)
    label = preview or str(summary["thread_id"])
    return f"{label} [csv]" if summary.get("mode") == "csv" else label


def source_header(source: dict[str, Any], *, cited: bool = False) -> str:
    """``[2] Handbook · Retention · page 3``, with ``(cited)`` when the answer used the mark."""
    parts = [f"[{source['n']}] {source['title']}"]
    if source.get("section"):
        parts.append(str(source["section"]))
    if source.get("page") is not None:
        parts.append(f"page {source['page']}")
    header = " · ".join(parts)
    return f"{header} (cited)" if cited else header


def assistant_message_id(turn: int) -> str:
    """The id the server gives the assistant message of a turn (``t1-assistant``)."""
    return f"t{turn}-assistant"


def passages_key(thread_id: str, message_id: str) -> str:
    """Where the page keeps the passages of one answer for as long as this browser session lives.

    A saved thread keeps only the references of an answer's sources, not their text (S4.2), so the
    passages that arrived with the stream are remembered here and shown again on a rerun.
    """
    return f"{thread_id}/{message_id}"


def turn_details(done: dict[str, Any]) -> list[str]:
    """The lines of the "details" box under a fresh answer: model, timings, what was changed."""
    lines = [f"model: {done.get('model', '?')}"]
    generation = done.get("generation") or {}
    timing = []
    if "first_token_seconds" in generation:
        timing.append(f"first token after {generation['first_token_seconds']:.2f} s")
    if "seconds" in generation:
        timing.append(f"{generation['seconds']:.2f} s in the model")
    retrieval = (done.get("retrieval_ms") or {}).get("total")
    if retrieval is not None:
        timing.append(f"search {retrieval:.0f} ms")
    if timing:
        lines.append("time: " + ", ".join(timing))
    if done.get("searched_as"):
        lines.append(f"searched for: {done['searched_as']}")
    if done.get("rewrite_error"):
        lines.append(f"the follow-up could not be rewritten: {done['rewrite_error']}")
    if done.get("dropped_citations"):
        marks = " ".join(f"[{n}]" for n in done["dropped_citations"])
        lines.append(f"removed citation marks that named no source: {marks}")
    for check in done.get("citation_checks") or []:
        if check.get("weak"):
            sentence = one_line(str(check.get("sentence", "")), SENTENCE_CHARS)
            lines.append(f"citation [{check['n']}] may not support: {sentence}")
    if done.get("removed_markup"):
        lines.append("removed from the answer: " + ", ".join(map(str, done["removed_markup"])))
    labels = " + ".join(map(str, done.get("labels") or []))
    if labels:
        outside = " (includes text from outside the chat)" if done.get("tainted") else ""
        lines.append(f"built from: {labels}{outside}")
    return lines


# --- the stream ------------------------------------------------------------------------------


def _status_line(data: dict[str, Any]) -> str:
    if data.get("stage") == LOADING_MODEL:
        model = data.get("model", "the model")
        return f"Loading {model} (the first question after a start is slow)"
    return "Working"


@dataclass
class TurnView:
    """What the answer being streamed adds up to, so far.

    Feed it every event of the stream, in order. The page shows ``text`` (or the status while there
    is no text yet), then, when the stream ends, ``answer``: the server's checked answer from the
    ``done`` event, which can differ from the pieces (a citation mark that named no source is
    removed). On an ``error`` event the pieces are discarded, because the server does not remember
    a failed turn (the S4.2 contract), so the page must not either.
    """

    text: str = ""
    thinking: str = ""
    status: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    done: dict[str, Any] | None = None
    error: str | None = None

    def feed(self, event: SseEvent) -> None:
        """Take one event in. An event this client does not know, or a malformed one, is ignored."""
        data = event.data if isinstance(event.data, dict) else {}
        match event.event:
            case "meta":
                self.meta = data
            case "status":
                self.status = _status_line(data)
            case "text":
                self.text += str(data.get("text", ""))
                self.status = ""
            case "thinking":
                self.thinking += str(data.get("text", ""))
                self.status = ""
            case "done":
                self.done = data
                self.status = ""
            case "error":
                self.error = str(data.get("message") or "the answer failed")
                self.text = ""
                self.thinking = ""
                self.status = ""
            case _:
                pass

    @property
    def finished(self) -> bool:
        return self.done is not None or self.error is not None

    @property
    def answer(self) -> str:
        """The finished answer if there is one, else what has streamed in."""
        if self.done is not None:
            return str(self.done.get("answer", ""))
        return self.text
