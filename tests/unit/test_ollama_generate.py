"""Tests for step S2.4: ``OllamaClient.generate``, against a fake HTTP server."""

import json

import httpx
import pytest

from geekbot.llm.models import OllamaClient, OllamaError


def client_for(handler: httpx.MockTransport) -> OllamaClient:
    return OllamaClient("http://ollama.test", transport=handler)


def test_generate_sends_a_json_constrained_non_streaming_request() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"response": '{"a": 1}'})

    schema: dict[str, object] = {"type": "object"}
    with client_for(httpx.MockTransport(handle)) as client:
        text = client.generate(
            "gemma3:4b",
            "hello",
            system="be brief",
            schema=schema,
            num_ctx=8192,
            keep_alive="30m",
            temperature=0.1,
            seed=0,
        )
    assert text == '{"a": 1}'
    assert seen[0].url.path == "/api/generate"
    assert json.loads(seen[0].content) == {
        "model": "gemma3:4b",
        "prompt": "hello",
        "system": "be brief",
        "stream": False,
        "format": {"type": "object"},
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_ctx": 8192, "seed": 0},  # 0 is a real seed
    }


def test_without_schema_system_or_seed_those_parts_are_left_out() -> None:
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "{}"})

    with client_for(httpx.MockTransport(handle)) as client:
        client.generate("m", "p", num_ctx=4096, keep_alive="5m")
    body = bodies[0]
    assert body["format"] == "json"  # plain JSON mode when there is no schema
    assert "system" not in body
    assert body["options"] == {"temperature": 0.3, "num_ctx": 4096}  # no "seed" key


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"response": "", "thinking": '{"q": 1}'}, '{"q": 1}'),
        ({"response": ""}, ""),
        ({"response": "", "thinking": 5}, ""),
    ],
)
def test_an_empty_response_falls_back_to_the_thinking_text(
    payload: dict[str, object], expected: str
) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    with client_for(transport) as client:
        assert client.generate("m", "p", num_ctx=1, keep_alive="1m") == expected


def test_a_reply_without_response_text_is_an_error() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"done": True}))
    with client_for(transport) as client, pytest.raises(OllamaError, match="no response text"):
        client.generate("m", "p", num_ctx=1, keep_alive="1m")


def test_a_server_error_is_an_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(500, text="boom"))
    with client_for(transport) as client, pytest.raises(OllamaError, match="/api/generate"):
        client.generate("m", "p", num_ctx=1, keep_alive="1m")
