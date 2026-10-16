"""Tests for step C2: the CSV analyst's Ollama adapter, against a fake HTTP server.

Same idea as the S0.3 client tests: ``httpx.MockTransport`` plays the server, so a test can check
the exact request that left the adapter and make the "server" answer with success, an error, or
garbage. Most tests here fail until you write ``OllamaCsvProvider.generate_json`` (your piece).
"""

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import pytest

from csv_analyst.llm.base import LLMProvider, LLMProviderError, PromptTooLongError
from csv_analyst.llm.ollama_provider import CHARS_PER_TOKEN, OUTPUT_RESERVE_TOKENS
from geekbot.csv_mode.provider import OllamaCsvProvider

# Small on purpose, so the "too long" tests need short strings.
NUM_CTX = 1024
BUDGET = NUM_CTX - OUTPUT_RESERVE_TOKENS  # tokens a prompt may use; the rest is for the reply

Handler = Callable[[httpx.Request], httpx.Response]


class Server:
    """A fake Ollama: records every request and answers each one with the same reply."""

    def __init__(self, reply: Mapping[str, object] | None = None) -> None:
        self.reply = dict(reply) if reply is not None else {"response": '{"ok": true}'}
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.reply)


def _provider(
    handler: Handler, *, on_response: Callable[[Mapping[str, object]], None] | None = None
) -> OllamaCsvProvider:
    return OllamaCsvProvider(
        host="http://ollama.test",
        model="qwen-test:1b",
        num_ctx=NUM_CTX,
        keep_alive="30m",
        timeout_s=5.0,
        transport=httpx.MockTransport(handler),
        on_response=on_response,
    )


def _status(code: int) -> Handler:
    """A server that answers every request with an empty reply and this HTTP status."""
    return lambda _: httpx.Response(code)


def _reply(**kwargs: Any) -> Handler:
    """A server that answers every request with ``httpx.Response(200, **kwargs)``."""
    return lambda _: httpx.Response(200, **kwargs)


def _text_of_tokens(tokens: int) -> str:
    """A string the package's token estimate counts as exactly ``tokens`` tokens."""
    return "x" * int(tokens * CHARS_PER_TOKEN)


# --- the request that leaves the adapter ---------------------------------------------------


def test_posts_to_api_generate_in_plain_json_mode() -> None:
    server = Server()
    with _provider(server) as provider:
        text = provider.generate_json("What is 2+2?", system="Be brief.", temperature=0.0)
    assert text == '{"ok": true}'
    (request,) = server.requests
    assert request.method == "POST"
    assert request.url.path == "/api/generate"
    assert server.bodies == [
        {
            "model": "qwen-test:1b",
            "prompt": "What is 2+2?",
            "system": "Be brief.",
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": NUM_CTX},
        }
    ]


def test_a_schema_replaces_the_plain_json_format() -> None:
    schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    server = Server()
    with _provider(server) as provider:
        provider.generate_json("x", schema=schema)
    assert server.bodies[0]["format"] == schema


def test_default_temperature_is_the_protocols_default() -> None:
    server = Server()
    with _provider(server) as provider:
        provider.generate_json("x")
    assert server.bodies[0]["options"]["temperature"] == 0.2


def test_seed_is_sent_only_when_given() -> None:
    server = Server()
    with _provider(server) as provider:
        provider.generate_json("no seed")
        provider.generate_json("seed seven", seed=7)
        provider.generate_json("seed zero", seed=0)  # 0 is a real seed, not "missing"
    assert "seed" not in server.bodies[0]["options"]
    assert server.bodies[1]["options"]["seed"] == 7
    assert server.bodies[2]["options"]["seed"] == 0


# --- the text that comes back --------------------------------------------------------------


def test_returns_the_reply_untouched_even_when_it_is_not_valid_json() -> None:
    broken = '{"code": "SELECT 1",}  \n'
    with _provider(Server({"response": broken})) as provider:
        assert provider.generate_json("x") == broken


def test_empty_response_falls_back_to_thinking() -> None:
    with _provider(Server({"response": "", "thinking": '{"answer": 1}'})) as provider:
        assert provider.generate_json("x") == '{"answer": 1}'


def test_empty_response_and_no_thinking_gives_an_empty_string() -> None:
    with _provider(Server({"response": ""})) as provider:
        assert provider.generate_json("x") == ""


@pytest.mark.parametrize(
    "reply",
    [{"done": True}, {"response": 5}, {"response": ["a"]}],
    ids=["missing", "number", "list"],
)
def test_a_reply_without_response_text_is_a_provider_error(reply: dict[str, object]) -> None:
    with _provider(Server(reply)) as provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_on_response_gets_the_parsed_response() -> None:
    reply = {"response": "{}", "load_duration": 123456789, "eval_count": 7}
    seen: list[Mapping[str, object]] = []
    with _provider(Server(reply), on_response=seen.append) as provider:
        provider.generate_json("x")
    assert seen == [reply]


# --- when the server misbehaves ------------------------------------------------------------


def test_server_error_becomes_provider_error() -> None:
    with _provider(_status(500)) as provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_unreachable_server_becomes_provider_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _provider(refuse) as provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_timeout_becomes_provider_error() -> None:
    def too_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with _provider(too_slow) as provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_a_body_that_is_not_json_becomes_provider_error() -> None:
    provider = _provider(_reply(content=b"<html>oops</html>"))
    with provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_a_json_body_that_is_not_an_object_becomes_provider_error() -> None:
    provider = _provider(_reply(json=["a"]))
    with provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")


def test_the_error_names_the_model_and_keeps_the_original_cause() -> None:
    expected = pytest.raises(LLMProviderError, match="qwen-test:1b")
    with _provider(_status(500)) as provider, expected as info:
        provider.generate_json("x")
    assert info.value.__cause__ is not None


def test_on_response_is_not_called_when_the_call_fails() -> None:
    seen: list[Mapping[str, object]] = []
    provider = _provider(_status(500), on_response=seen.append)
    with provider, pytest.raises(LLMProviderError):
        provider.generate_json("x")
    assert seen == []


# --- the prompt-too-long guard -------------------------------------------------------------


def test_a_prompt_that_just_fits_is_sent() -> None:
    server = Server()
    with _provider(server) as provider:
        provider.generate_json(_text_of_tokens(BUDGET))
    assert len(server.requests) == 1


def test_a_prompt_one_token_over_is_refused_before_any_request() -> None:
    server = Server()
    with _provider(server) as provider, pytest.raises(PromptTooLongError):
        provider.generate_json(_text_of_tokens(BUDGET + 1))
    assert server.requests == []  # nothing was sent


def test_the_system_prompt_counts_toward_the_limit() -> None:
    half = BUDGET // 2
    server = Server()
    with _provider(server) as provider, pytest.raises(PromptTooLongError):
        provider.generate_json(_text_of_tokens(half), system=_text_of_tokens(BUDGET - half + 1))
    assert server.requests == []


# --- the shape the package expects ---------------------------------------------------------


def test_generate_json_has_the_protocols_signature() -> None:
    """The package calls ``generate_json`` by these names, so ours must match them exactly."""
    ours = inspect.signature(OllamaCsvProvider.generate_json).parameters
    theirs = inspect.signature(LLMProvider.generate_json).parameters
    assert list(ours) == list(theirs)
    for name, parameter in theirs.items():
        assert ours[name].kind == parameter.kind
        assert ours[name].default == parameter.default
