"""Step S6.1: does a real MCP client accept our server? The official SDK's client against a real HTTP server.

``test_mcp.py`` checks what the server says; this checks that a client written by someone else can talk to
it: the handshake, the tool list, a call and a malformed call, over a real loopback socket with the API key,
in both of the SDK's negotiation modes ("legacy": the initialize handshake; "auto": probe ``server/discover``
first and fall back to the handshake). The SDK is a development dependency used only here; the program
itself never imports it. Skipped when it is not installed.
"""

import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from serverkit import ingest_note, make_app

pytest.importorskip("mcp")
httpx2 = pytest.importorskip("httpx2")

from fastapi.testclient import TestClient
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

KEY = "k" * 32


@pytest.fixture
def url(tmp_path: Path) -> Iterator[str]:
    """A real server on a free loopback port, with one note ingested, stopped afterwards."""
    app = make_app(tmp_path, api_key=KEY)
    # The note is ingested through the API first (its own lifespan, which ends and frees the storage lock);
    # the same app is then served for real and opens the same data folder again.
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as seeded:
        ingest_note(seeded, tmp_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "the test server did not start"
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=20)


async def _conversation(url: str, mode: str, key: str) -> dict[str, object]:
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=30)
    async with Client(streamable_http_client(url, http_client=http), mode=mode) as client:  # type: ignore[arg-type]
        tools = await client.list_tools()
        found = await client.call_tool("search_knowledge_base", {"query": "zebras savanna", "k": 2})
        listing = await client.call_tool("list_documents", {})
        try:
            await client.call_tool("search_knowledge_base", {"query": ""})
            refused = False
        except Exception:  # noqa: BLE001  # the SDK raises its own error type for a JSON-RPC error
            refused = True
        return {
            "tools": [tool.name for tool in tools.tools],
            "readonly": [
                bool(tool.annotations and tool.annotations.read_only_hint) for tool in tools.tools
            ],
            "found": found.content[0].text,  # type: ignore[union-attr]
            "found_is_error": found.is_error,
            "listing": listing.content[0].text,  # type: ignore[union-attr]
            "refused": refused,
        }


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_the_official_client_can_use_the_server(url: str, mode: str) -> None:
    seen = asyncio.run(_conversation(url, mode, KEY))
    assert seen["tools"] == ["search_knowledge_base", "list_documents"]
    assert seen["readonly"] == [True, True]
    assert seen["found_is_error"] is False
    assert "Zebras live on the savanna." in str(seen["found"])
    assert "zebras.txt" in str(seen["listing"])
    assert seen["refused"] is True  # a malformed call is a protocol error the client raises


def test_the_official_client_without_the_key_is_refused(url: str) -> None:
    """The SDK reports a 401 as an exception group with no useful text, so the test only asks that it
    fails; that the server answered 401 (and not something else) is ``test_mcp.py``'s check."""
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(_conversation(url, "legacy", "x" * 32))
