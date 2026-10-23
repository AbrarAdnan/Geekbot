"""Which failures are worth trying again, and how often (step S3.6, PLAN.md §7 "node ``RetryPolicy``").

A local Ollama fails in two different ways, and only one of them goes away by itself. *Transient*
failures: the connection was refused or reset while the server restarts, a request timed out because
the GPU was busy with another program (this PC also runs the user's other projects), the server
answered 503 "busy" or 500 after a model runner stopped. Asking again a second later often works.
*Permanent* failures: the model is not pulled (404), the request is wrong (400), the reply is empty or
unparsable, the thread is in the other mode. Asking again returns the same error, after costing the
user the seconds it took to fail the first time. So a retry policy is not "retry on error": it is a
decision about *which* errors, and this module is that decision.

The default of LangGraph's ``RetryPolicy`` (``default_retry_on``) is not suitable here. It never
retries ``RuntimeError``, ``OSError`` or ``ValueError`` and everything this program raises on purpose
(``ChatError``, ``OllamaError``, ``EmbeddingError``) is a ``RuntimeError``. It would retry nothing
this program raises, so ``is_transient`` looks at the *cause* each of them was raised from.

**One more rule, the one that matters for a streaming answer: a call that has already shown part of
its reply is never retried.** ``generate`` sends each piece to the user as it arrives. If the stream
breaks half way and the node ran again, the second reply would be written after the first half that
is already on the screen. ``ChatError.partial`` carries that fact out of the node and
``is_transient`` refuses on it. A failure before the first piece (the usual transient one: the
connection did not open) is retried freely.

The numbers are starting points, not measurements: three attempts and waits of about 0.5 s and 1 s
(with jitter, which spreads out clients that fail together; here there is one client, so it hardly
matters) add at most about 1.5 s to a turn that is going to fail anyway.
"""

from typing import Final

import httpx
from langgraph.types import RetryPolicy

__all__ = [
    "RETRY_ATTEMPTS",
    "RETRY_FIRST_WAIT_S",
    "TRANSIENT_STATUS",
    "is_transient",
    "transient_retry",
]

RETRY_ATTEMPTS: Final = 3  # the first try included
RETRY_FIRST_WAIT_S: Final = 0.5
# 429 (asked to slow down), 500 (a model runner stopped), 502 to 504 (a proxy or a busy server).
# 400 and 404 are never here: the request or the model name is wrong, and asking again will not fix
# it.
TRANSIENT_STATUS: Final = frozenset({429, 500, 502, 503, 504})

_CHAIN_LIMIT: Final = 8  # links of a cause chain that are followed (so a cycle cannot loop)


def _transient_link(error: BaseException) -> bool:
    """Whether this one exception, taken alone, is a failure that may pass."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in TRANSIENT_STATUS
    if isinstance(error, httpx.TransportError | ConnectionError | TimeoutError):
        return True
    # The ``ollama`` client's own ``ResponseError`` carries the status of the reply; it is read by
    # name so that this module does not import the client (slow to load, and not always needed).
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status in TRANSIENT_STATUS


def is_transient(error: BaseException) -> bool:
    """Whether running the step again may succeed. The ``retry_on`` of every step that has a retry.

    Walks the chain of causes (``raise X from Y``) because the step's own exception is a sentence
    for the user and the reason is underneath it: a ``ChatError`` raised from a refused connection is
    transient, one raised from a 404 is not, one raised from nothing (an empty reply) is not. An
    error that says it had already sent part of its reply (``partial``) is never transient.
    """
    if getattr(error, "partial", False):
        return False
    current: BaseException | None = error
    for _ in range(_CHAIN_LIMIT):
        if current is None:
            return False
        if _transient_link(current):
            return True
        current = current.__cause__
    return False


def transient_retry(
    *,
    max_attempts: int = RETRY_ATTEMPTS,
    initial_interval: float = RETRY_FIRST_WAIT_S,
    jitter: bool = True,
) -> RetryPolicy:
    """A ``RetryPolicy`` that retries only what ``is_transient`` accepts, with a doubling wait.

    The arguments exist for the tests, which cannot wait seconds: they pass ``initial_interval=0``
    and ``jitter=False``.
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        max_interval=8.0,
        jitter=jitter,
        retry_on=is_transient,
    )
