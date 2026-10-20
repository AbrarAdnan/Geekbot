"""Tests for step S0.3: the Ollama client, against a fake HTTP server (httpx.MockTransport)."""

import json

import httpx
import pytest

from geekbot.config import Settings
from geekbot.llm.models import OllamaClient, OllamaError, find_chat_problem, model_is_pulled


def _client(handler: httpx.MockTransport) -> OllamaClient:
    return OllamaClient("http://ollama.test", transport=handler)


def test_version_and_model_list() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.2"})
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}, {"size": 1}]})

    with _client(httpx.MockTransport(handle)) as client:
        assert client.version() == "0.34.2"
        assert client.list_models() == ["gemma3:4b"]  # the entry without a name is skipped


def test_warm_up_sends_context_and_keep_alive_and_no_prompt() -> None:
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    with _client(httpx.MockTransport(handle)) as client:
        seconds = client.warm_up("gemma3:4b", num_ctx=8192, keep_alive="30m")
    assert seconds >= 0
    assert seen == [
        {
            "model": "gemma3:4b",
            "prompt": "",
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_ctx": 8192},
        }
    ]


def test_unload_sends_keep_alive_zero() -> None:
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    with _client(httpx.MockTransport(handle)) as client:
        client.unload("gemma3:4b")
    assert seen[0]["keep_alive"] == 0


def test_server_error_becomes_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with _client(transport) as client, pytest.raises(OllamaError):
        client.version()


def test_unreachable_server_becomes_ollama_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(httpx.MockTransport(refuse)) as client, pytest.raises(OllamaError):
        client.list_models()


def test_model_is_pulled_treats_a_bare_name_as_latest() -> None:
    available = ["gemma3:4b", "nomic-embed-text:latest"]
    assert model_is_pulled("gemma3:4b", available)
    assert model_is_pulled("nomic-embed-text", available)
    assert not model_is_pulled("gemma4:12b", available)
    assert model_is_pulled("nomic-embed-text:latest", ["nomic-embed-text"])


def test_loaded_models_asks_api_ps_and_skips_entries_without_a_name() -> None:
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:3b"}, {"size": 1}]})

    with _client(httpx.MockTransport(handle)) as client:
        assert client.loaded_models() == ["qwen2.5-coder:3b"]
    assert paths == ["/api/ps"]


def test_nothing_loaded_is_an_empty_list() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []}))
    with _client(transport) as client:
        assert client.loaded_models() == []


def test_loaded_models_without_a_list_is_an_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"models": "none"}))
    with _client(transport) as client, pytest.raises(OllamaError, match="/api/ps"):
        client.loaded_models()


# --- find_chat_problem (step S4.2: `geekbot serve` asks before it starts) -----------------------


def _ollama_with_models(*names: str) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _: httpx.Response(200, json={"models": [{"name": name} for name in names]})
    )


def test_no_chat_problem_when_ollama_has_the_chat_model() -> None:
    settings = Settings(_env_file=None)
    transport = _ollama_with_models(settings.resolved_chat_model, "nomic-embed-text:latest")
    assert find_chat_problem(settings, transport=transport) is None


def test_the_chat_problem_names_the_model_and_the_command_that_fixes_it() -> None:
    settings = Settings(_env_file=None)
    problem = find_chat_problem(settings, transport=_ollama_with_models("qwen2.5-coder:3b"))
    assert problem == (
        f"the chat model {settings.resolved_chat_model} is not pulled: "
        f"run `ollama pull {settings.resolved_chat_model}`"
    )


def test_the_chat_problem_follows_the_profile_and_the_override() -> None:
    target = Settings(_env_file=None, profile="target")
    assert target.resolved_chat_model == "gemma4:12b"
    problem = find_chat_problem(target, transport=_ollama_with_models("gemma3:4b"))
    assert problem is not None and "gemma4:12b" in problem
    override = Settings(_env_file=None, chat_model="custom:7b")
    assert find_chat_problem(override, transport=_ollama_with_models("custom:7b")) is None


def test_the_chat_problem_says_when_ollama_cannot_be_reached() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    problem = find_chat_problem(Settings(_env_file=None), transport=httpx.MockTransport(refuse))
    assert problem is not None
    assert "cannot reach Ollama at http://127.0.0.1:11434" in problem
