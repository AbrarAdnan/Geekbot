"""PLAN.md Phase 6B.2: the Ollama adapter must forward a JSON Schema object
as `format` when given one, and fall back to the plain "json" string mode
when it isn't -- no other request shape should change."""

from __future__ import annotations

from unittest.mock import patch

from csv_analyst.llm.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_generate_json_without_schema_sends_plain_json_mode():
    provider = OllamaProvider(model="qwen2.5-coder:3b")
    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse({"response": '{"ok": true}'})
        result = provider.generate_json("prompt text", temperature=0.2, seed=1)

    assert result == '{"ok": true}'
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["format"] == "json"


def test_generate_json_falls_back_to_thinking_field_when_response_empty():
    """A 'thinking' model (e.g. qwen3.5) can put its whole structured answer
    in `thinking` and leave `response` empty -- Ollama splits a model's own
    <think> tags into that field. Must not be treated as a malformed/empty
    response when `thinking` actually has the content."""
    provider = OllamaProvider(model="qwen3.5:9b")
    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse({"response": "", "thinking": '{"code": "result = 1"}'})
        result = provider.generate_json("prompt text", temperature=0.2, seed=1)

    assert result == '{"code": "result = 1"}'


def test_generate_json_prefers_response_over_thinking_when_both_present():
    provider = OllamaProvider(model="qwen2.5-coder:3b")
    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse({"response": '{"code": "real"}', "thinking": "scratch notes"})
        result = provider.generate_json("prompt text", temperature=0.2, seed=1)

    assert result == '{"code": "real"}'


def test_generate_json_with_schema_forwards_schema_object_as_format():
    provider = OllamaProvider(model="qwen2.5-coder:3b")
    schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse({"response": '{"code": "result = 1"}'})
        result = provider.generate_json("prompt text", temperature=0.2, seed=1, schema=schema)

    assert result == '{"code": "result = 1"}'
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["format"] == schema


def test_on_response_hook_receives_prompt_and_raw_response():
    """Card H-S timing: the optional hook sees Ollama's response JSON (token
    counts, durations); without a hook nothing changes."""
    seen = []
    provider = OllamaProvider(model="qwen2.5-coder:3b", on_response=lambda p, d: seen.append((p, d)))
    payload = {"response": '{"ok": true}', "eval_count": 12, "eval_duration": 100}
    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse(payload)
        result = provider.generate_json("prompt text", temperature=0.2, seed=1)

    assert result == '{"ok": true}'
    assert seen == [("prompt text", payload)]


def test_prompt_that_would_not_fit_is_refused_before_sending():
    """An over-long prompt is not sent: Ollama would keep only its end
    (losing the schema) and the model would answer from what was left."""
    import pytest

    from csv_analyst.llm.base import PromptTooLongError

    provider = OllamaProvider(model="qwen2.5-coder:7b", num_ctx=2048)
    with patch("csv_analyst.llm.ollama_provider.requests.post") as mock_post:
        mock_post.return_value = _FakeResponse({"response": '{"ok": true}'})
        with pytest.raises(PromptTooLongError, match="prompt too long"):
            provider.generate_json("x" * 5000, temperature=0.2, seed=1)
        assert not mock_post.called
        assert provider.generate_json("x" * 3000, temperature=0.2, seed=1) == '{"ok": true}'
