"""``geekbot mcp-stdio``: the MCP server of a running Geekbot, for a client that speaks over stdin and
stdout (step S6.1).

Some MCP clients start a program and talk to it over its standard input and output instead of calling an
HTTP address. This is that program: it reads one JSON-RPC message per line, passes each to the server's
``POST /mcp`` (with the API key it finds the way the UI does) and writes the server's answer as one line.
It holds no storage and no model, it adds no tool, and it needs the server to be running (``geekbot up`` or
``geekbot serve``): the one process that owns the storage stays the one process (PLAN.md §6.2).

**Standard output is the protocol.** Nothing but JSON-RPC messages, one per line, is ever written to it;
every note for a person goes to standard error. A request the server cannot be asked (it is not running, it
refused the key) is answered with a JSON-RPC error carrying the reason, so the client is told instead of
waiting; a notification (no ``id``) that cannot be delivered is dropped, as the protocol allows.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx

__all__ = ["Reply", "relay", "send_over_http"]

UNREACHABLE = -32000  # JSON-RPC's range for implementation-defined server errors


@dataclass(frozen=True)
class Reply:
    """What the server said: the HTTP status and the body (empty for ``202``)."""

    status: int
    body: str


def send_over_http(client: httpx.Client, url: str, api_key: str) -> Callable[[str], Reply]:
    """A ``send`` function for ``relay``: POST the line to ``url`` with the key, return the reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def send(line: str) -> Reply:
        response = client.post(url, content=line.encode("utf-8"), headers=headers)
        return Reply(response.status_code, response.text)

    return send


def _request_id(line: str) -> object | None:
    """The ``id`` of a JSON-RPC request in ``line``, or ``None`` for a notification or non-JSON."""
    try:
        message = json.loads(line)
    except ValueError:
        return None
    if isinstance(message, dict) and "id" in message and isinstance(message.get("method"), str):
        request_id: object = message["id"]
        return request_id
    return None


def relay(
    lines: Iterable[str],
    send: Callable[[str], Reply],
    write: Callable[[str], None],
    note: Callable[[str], None],
) -> None:
    """Pass every non-empty line to ``send`` and write each answer as one line.

    ``write`` is standard output (messages only), ``note`` standard error. A transport failure
    (``httpx.HTTPError``) on a request becomes a JSON-RPC error with the request's own ``id``; on a
    notification it is noted and dropped. A body that is not JSON is never written to ``write``.
    """
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            reply = send(line)
        except httpx.HTTPError as error:
            reason = (
                f"cannot reach the Geekbot server ({type(error).__name__}); is it running? "
                "(`geekbot up` or `geekbot serve`)"
            )
            note(reason)
            request_id = _request_id(line)
            if request_id is not None:
                error_message = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": UNREACHABLE, "message": reason},
                }
                write(json.dumps(error_message))
            continue
        if reply.status == httpx.codes.ACCEPTED or not reply.body.strip():
            continue
        try:
            message = json.loads(reply.body)
        except ValueError:
            note(f"the server answered HTTP {reply.status} with something that is not JSON")
            continue
        if reply.status in {httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN}:
            note(f"the server refused the request (HTTP {reply.status})")
        write(json.dumps(message, ensure_ascii=False))
