"""Tests for step S3.1: the chat model is built from the same settings the warm-up reads.

``ChatOllama`` is created but never called, so nothing here needs Ollama. What is checked is the rule
behind the factory (llm/chat.py): a query must ask Ollama for exactly the options the model was loaded
with, or Ollama reloads it during the query (R-2).
"""

import pytest
from pydantic import ValidationError

from geekbot.config import Settings
from geekbot.llm.chat import make_chat_model

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_the_chat_model_repeats_the_values_the_warm_up_uses() -> None:
    config = settings(chat_num_ctx=4096, keep_alive="10m")
    chat = make_chat_model(config)
    assert chat.num_ctx == config.chat_num_ctx == 4096
    assert chat.keep_alive == config.keep_alive == "10m"


def test_it_points_at_the_configured_ollama_and_the_profiles_chat_model() -> None:
    chat = make_chat_model(settings(ollama_host="http://ollama.test:11434"))
    assert chat.model == "gemma3:4b"  # the proto profile
    assert chat.base_url == "http://ollama.test:11434"


def test_the_target_profile_and_an_explicit_model_are_respected() -> None:
    assert make_chat_model(settings(profile="target")).model == "gemma4:12b"
    assert make_chat_model(settings(chat_model="mine:1b")).model == "mine:1b"
    assert make_chat_model(settings(), model="other:2b").model == "other:2b"


def test_the_reply_is_capped_sampling_is_cool_and_thinking_is_off_by_default() -> None:
    chat = make_chat_model(settings())
    assert chat.num_predict == 512
    assert chat.temperature == 0.2
    assert chat.reasoning is False


def test_the_limits_come_from_the_settings() -> None:
    chat = make_chat_model(settings(chat_num_predict=64, chat_temperature=0.0))
    assert chat.num_predict == 64
    assert chat.temperature == 0.0


@pytest.mark.parametrize(
    "bad",
    [{"chat_num_predict": 0}, {"chat_temperature": 3.0}, {"context_budget_tokens": 0}],
)
def test_nonsense_limits_are_refused_at_start_up(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**bad)
