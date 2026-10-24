"""Tests for step S6.1: ``geekbot mcp-stdio``, the relay between a client's stdin/stdout and ``POST /mcp``.

The relay's one rule is that standard output carries nothing but JSON-RPC messages; the rest is how a
request that cannot be delivered is answered (so the client is told) and how a notification is dropped."""

import json
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.mcp_stdio import Reply, relay, send_over_http

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

PING = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
NOTE = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})


def run(lines: list[str], send: Any) -> tuple[list[str], list[str]]:
    out: list[str] = []
    notes: list[str] = []
    relay(lines, send, out.append, notes.append)
    return out, notes


def test_a_reply_is_written_as_one_line_and_a_202_and_blank_lines_write_nothing() -> None:
    sent: list[str] = []

    def send(line: str) -> Reply:
        sent.append(line)
        return (
            Reply(202, "")
            if "notifications" in line
            else Reply(200, '{"jsonrpc": "2.0",\n "id": 7, "result": {}}')
        )

    out, notes = run(["", PING + "\n", "  ", NOTE], send)
    assert sent == [PING, NOTE]
    assert out == ['{"jsonrpc": "2.0", "id": 7, "result": {}}'] and notes == []


def test_a_server_that_cannot_be_reached_is_answered_with_an_error_carrying_the_requests_id() -> (
    None
):
    def send(line: str) -> Reply:
        raise httpx.ConnectError("refused")

    out, notes = run([PING, NOTE], send)
    (reply,) = [json.loads(line) for line in out]  # the notification got nothing
    assert reply["id"] == 7 and reply["error"]["code"] == -32000
    assert "geekbot up" in reply["error"]["message"] and len(notes) == 2


def test_a_refusal_is_forwarded_and_noted_and_something_that_is_not_json_is_only_noted() -> None:
    refusal = '{"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "no"}}'
    out, notes = run([PING], lambda line: Reply(401, refusal))
    assert out == [refusal] and "HTTP 401" in notes[0]
    out, notes = run([PING], lambda line: Reply(502, "<html>bad gateway</html>"))
    assert out == [] and "not JSON" in notes[0]


@given(st.lists(st.text(max_size=40), max_size=8))
def test_nothing_that_is_not_json_ever_reaches_standard_output(lines: list[str]) -> None:
    def send(line: str) -> Reply:
        return Reply(200, "garbage " + line)

    out, _ = run(lines, send)
    assert out == []


def test_the_http_sender_posts_the_line_with_the_key_and_the_right_headers() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 7, "result": {}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        reply = send_over_http(client, "http://127.0.0.1:8000/mcp", "secret-key")(PING)
    (request,) = seen
    assert request.url == "http://127.0.0.1:8000/mcp" and request.content.decode() == PING
    assert request.headers["authorization"] == "Bearer secret-key"
    assert "application/json" in request.headers["accept"]
    assert reply.status == 200


# --- the command -------------------------------------------------------------------------------

runner = CliRunner()


def test_the_command_says_so_when_there_is_no_key_yet(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "empty"))
    result = runner.invoke(app, ["mcp-stdio"])
    assert result.exit_code == 1 and "start the server first" in result.output


def test_the_command_relays_lines_and_keeps_standard_output_clean(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEEKBOT_API_KEY", "k" * 32)
    monkeypatch.setattr(
        "geekbot.mcp_stdio.send_over_http",
        lambda client, url, key: (
            lambda line: Reply(200, '{"jsonrpc": "2.0", "id": 7, "result": {}}')
        ),
    )
    result = runner.invoke(app, ["mcp-stdio"], input=PING + "\n")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.strip()) == {"jsonrpc": "2.0", "id": 7, "result": {}}
