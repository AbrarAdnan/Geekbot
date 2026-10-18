"""Which branch of the graph a turn takes: chat or CSV (step C4, PLAN.md §6.7, §7).

A thread has one **mode**, picked when the conversation starts and never changed after: ``chat`` answers
from the ingested documents, ``csv`` answers from one CSV file through the copied ``csv_analyst``
package. Fixing the mode per thread keeps three things simple that would otherwise interact: which
model is on the GPU (the two never share it, and a swap costs seconds), what the memory contains (a
table's numbers next to a document's sentences), and what the taint means (a CSV's cells are untrusted
text from a different source; PLAN.md §6.7: "CSV data never reaches the web tool").

**Deterministic routing, not model routing.** The router is a plain function of the thread's state:
no model call, no prompt, nothing a document or a CSV cell could talk it out of. A model that chose the
branch would spend a call on every turn and could be steered by the very text the branch handles.

This module is the pure part, the part a person can read and test without a graph: ``choose_mode``
(what the mode of this turn is) and the two routing functions the graph uses as conditional edges.
The nodes that do the work are in ``agent.nodes``.
"""

from typing import Final, Literal

from geekbot.agent.state import AgentState

__all__ = ["MODES", "ModeError", "choose_mode", "route_after_begin", "route_after_condense"]

Mode = Literal["chat", "csv"]
MODES: Final[tuple[Mode, ...]] = ("chat", "csv")


class ModeError(ValueError):
    """The mode asked for is unknown, or contradicts the one the thread already has."""


def _known(name: str, *, what: str) -> Mode:
    if name == "chat":
        return "chat"
    if name == "csv":
        return "csv"
    raise ModeError(f"unknown {what} {name!r}: the modes are {', '.join(MODES)}")


def choose_mode(saved: str | None, requested: str | None) -> Mode:
    """The mode of this turn, from the mode the thread already has and the one the caller asks for.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. The rules:

    1. A thread that has a mode keeps it. If the caller asks for the same one (or for none), that is
       the answer.
    2. A thread with no mode yet (its first turn) takes the one asked for.
    3. With neither, the mode is ``chat``: the default, and what every thread was before C4.
    4. Asking for a different mode than the thread has is an error, never a switch: ``ModeError``
       names both. A thread cannot become a CSV conversation half way, and a CSV conversation cannot
       start answering from documents, because its memory, its model and its taint were built under
       the other mode.
    5. A name that is not a mode is an error too, whichever argument carries it (a typo must not
       silently mean ``chat``). ``None`` and the empty string both mean "not given".
    """
    have = saved or None
    want = requested or None
    kept = _known(have, what="saved mode") if have is not None else None
    asked = _known(want, what="mode") if want is not None else None
    if kept is not None and asked is not None and kept != asked:
        raise ModeError(
            f"this thread is a {kept} conversation and cannot become a {asked} one: "
            "a thread keeps the mode it started with (use a new thread name)"
        )
    return kept or asked or "chat"


def route_after_begin(state: AgentState) -> Literal["condense", "retrieve", "csv_ask"]:
    """The first conditional edge, after the turn has started and its mode is known.

    A thread with history is on a follow-up, which is rewritten first (``condense``, in either mode);
    a first turn goes straight to its branch: ``retrieve`` for chat, ``csv_ask`` for CSV. A plain
    function of the state, so the first question of a conversation makes no model call before the
    branch does its own work.
    """
    if state.get("history"):
        return "condense"
    return "csv_ask" if state.get("mode") == "csv" else "retrieve"


def route_after_condense(state: AgentState) -> Literal["retrieve", "csv_ask"]:
    """The second conditional edge: once a follow-up has been rewritten, on to the thread's branch."""
    return "csv_ask" if state.get("mode") == "csv" else "retrieve"
