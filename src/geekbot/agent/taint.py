"""Taint: who wrote a piece of text, and what that allows (step S5.3, PLAN.md §8.1 G2 and §8.2).

Phase 3 kept the *books*: every message in a thread carries labels (``domain.Label``), a model call's
output carries the union of the labels of everything in its prompt (``packing.pack_prompt`` is the only
place that decides them), and ``thread_taint`` only grows. This module is where the books are *used*.
Three pure functions, no I/O, no model:

- ``egress_decision`` is the rule of **G2**: no network call whose arguments were built from text the
  user did not write, unless the user approved those exact arguments. It is a plain function of its
  inputs so that it can be tested against every combination and so that the one object that calls
  tools (``tools.egress.EgressGate``) has nothing to decide for itself.
- ``query_labels`` says what a search query was built from. The user's own words carry ``USER`` and
  nothing else; a query that a model wrote after reading the conversation carries the labels of the
  whole thread (a superset of what was in the model's prompt, which errs on the side of asking).
- ``import_client_history`` is the one door for a conversation that arrives from outside: an assistant
  message a client sends back cannot be shown to come from this program, so it is ``UNKNOWN``, which is
  tainted. No route needs it today (the stateless OpenAI-compatible endpoint it was written for is
  parked, D-096); it is here, tested, for the day a route accepts a client-written history, and so
  that "client-supplied assistant messages are ``UNKNOWN``" is a rule in code and not in a plan.

**No declassification.** Nothing in this module removes a label. A tainted thread stays tainted when
its tainted messages scroll out of the history window; a label that cannot be read becomes ``UNKNOWN``
(``domain.label_of``), never ``USER``; an empty set of labels on arguments is read as ``UNKNOWN`` too,
because "no information about where this came from" is the case the rule exists for.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from geekbot.agent.state import HistoryMessage
from geekbot.domain import Label, is_tainted, label_of

__all__ = [
    "EgressAction",
    "EgressDecision",
    "egress_decision",
    "import_client_history",
    "query_labels",
]


class EgressAction(StrEnum):
    """What the gate does with a call: let it through, ask the user first, or refuse it."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class EgressDecision:
    """The gate's answer for one call, with the reason in words (it is shown to the user and logged).

    ``tainted`` is whether the arguments were built from text the user did not write; it is kept even
    when the call is allowed (after an approval) so a report can count how many calls were tainted.
    """

    action: EgressAction
    reason: str
    tainted: bool


def egress_decision(
    *,
    enabled: bool,
    labels: Iterable[Label],
    arguments: str,
    approved: str | None = None,
    denied: bool = False,
) -> EgressDecision:
    """Decide one network call (G2). ✍️ Written by Claude (prototyping mode, D-047).

    In order, the first rule that applies wins:

    1. Network tools are **switched off** (``enabled`` false): ``DENY``. Nothing, not even an approval,
       turns a switched-off tool on: "web off" has to mean no egress.
    2. The user **denied** this call: ``DENY``.
    3. The arguments are **untainted** (``labels`` hold only ``USER``): ``ALLOW``. Text the user typed
       is theirs to send.
    4. The arguments are tainted and the user **approved exactly these arguments**: ``ALLOW``.
       ``approved`` is the text that was shown to the user, compared character for character with
       ``arguments``: an edit, a trimmed space or one more word is a different call.
    5. Otherwise ``ASK``.

    ``labels`` empty counts as ``UNKNOWN``: the caller knows nothing about where the arguments came
    from, and that is a tainted call, not a clean one.
    """
    label_set = frozenset(labels) or frozenset({Label.UNKNOWN})
    tainted = is_tainted(label_set)
    if not enabled:
        return EgressDecision(EgressAction.DENY, "network tools are switched off", tainted)
    if denied:
        return EgressDecision(EgressAction.DENY, "the user denied this call", tainted)
    if not tainted:
        return EgressDecision(EgressAction.ALLOW, "the arguments are the user's own words", False)
    if approved is not None and approved == arguments:
        return EgressDecision(EgressAction.ALLOW, "the user approved these exact arguments", True)
    return EgressDecision(
        EgressAction.ASK, "the arguments were built from text the user did not write", True
    )


def query_labels(question: str, query: str, thread_taint: Iterable[str]) -> frozenset[Label]:
    """The labels of a search ``query`` for ``question``, in a thread that has carried ``thread_taint``.

    A query that is the user's own question (same text apart from surrounding space) was written by
    the user: ``USER``. Any other query is the output of the rewrite step, a model that had the
    conversation in its prompt, so it carries every label the thread has ever carried. That is more
    than the labels of the messages that actually fitted in the rewrite's prompt, on purpose: the
    thread's taint is monotonic and exact bookkeeping per message would be a second place to get wrong.
    """
    labels = {Label.USER}
    if query.strip() != question.strip():
        labels.update(label_of(name) for name in thread_taint)
    return frozenset(labels)


_CLIENT_ROLES: Final = {"user": Label.USER, "assistant": Label.UNKNOWN}


def import_client_history(messages: Sequence[Mapping[str, object]]) -> list[HistoryMessage]:
    """A conversation that arrived from outside, as history this program can trust the labels of.

    Every label is assigned here, none is read from the client: ``user`` messages are ``USER`` and
    ``assistant`` messages are ``UNKNOWN`` (the program cannot show that it wrote them, and a model
    repeats what it was told to repeat). A ``labels`` field in the input is ignored, so a client
    cannot claim its assistant message is clean. ``ValueError`` for any other role, ``TypeError`` for
    a content that is not text. Ids are made here (``c<n>-<role>``) so that they cannot collide with a saved turn's.
    """
    imported: list[HistoryMessage] = []
    for number, message in enumerate(messages, start=1):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in _CLIENT_ROLES:
            raise ValueError(f"message {number}: role must be 'user' or 'assistant', got {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"message {number}: content must be text")
        imported.append(
            HistoryMessage(
                id=f"c{number}-{role}",
                role="user" if role == "user" else "assistant",
                content=content,
                labels=[_CLIENT_ROLES[role].value],
            )
        )
    return imported
