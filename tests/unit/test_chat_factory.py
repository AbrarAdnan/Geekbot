"""Tests for step S3.1: the chat model is built from the same settings the warm-up reads.

``ChatOllama`` is created but never called, so nothing here needs Ollama. What is checked is the rule
behind the factory (llm/chat.py): a query must ask Ollama for exactly the options the model was loaded
with, or Ollama reloads it during the query (R-2).
"""

import pytest
from pydantic import ValidationError

from geekbot.agent.condense import CondenseOutput
from geekbot.config import Settings
from geekbot.llm.chat import make_chat_model, make_structured_model

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


def test_thinking_on_asks_for_reasoning_and_gets_a_bigger_reply_budget() -> None:
    config = settings(chat_num_predict=64, chat_think_num_predict=640)
    thinking = make_chat_model(config, think=True)
    assert thinking.reasoning is True
    assert thinking.num_predict == 640  # the reasoning is generated tokens too
    plain = make_chat_model(config)
    assert plain.reasoning is False
    assert plain.num_predict == 64


def test_thinking_changes_nothing_that_would_make_ollama_reload_the_model() -> None:
    config = settings(chat_num_ctx=4096, keep_alive="10m")
    thinking = make_chat_model(config, think=True)
    assert thinking.num_ctx == config.chat_num_ctx
    assert thinking.keep_alive == config.keep_alive


def test_the_rewrite_model_is_the_same_model_with_a_small_cap_and_the_schema_as_its_format() -> (
    None
):
    """Step S3.4: a follow-up's rewrite must not make Ollama reload the model (R-2)."""
    config = settings(chat_num_ctx=4096, keep_alive="10m")
    structured = make_structured_model(config, CondenseOutput, num_predict=96)
    chat = structured.first.bound  # type: ignore[attr-defined]
    assert chat.model == "gemma3:4b"
    assert chat.num_ctx == 4096  # the warm-up's value
    assert chat.keep_alive == "10m"
    assert chat.num_predict == 96
    assert chat.temperature == 0.0
    assert chat.reasoning is False
    assert structured.first.kwargs["format"] == CondenseOutput.model_json_schema()  # type: ignore[attr-defined]


def test_the_rewrite_models_name_can_be_overridden() -> None:
    structured = make_structured_model(settings(), CondenseOutput, num_predict=32, model="other:2b")
    assert structured.first.bound.model == "other:2b"  # type: ignore[attr-defined]


# --- the batch size (step S5.1): a load-time option, so every call must send it -----------------


def request_options(chat: object, **overrides: object) -> dict[str, object]:
    """The ``options`` of the HTTP request the chat model would send (nothing is sent)."""
    from langchain_core.messages import HumanMessage

    params = chat._chat_params([HumanMessage("hi")], **overrides)  # type: ignore[attr-defined]
    options: dict[str, object] = params["options"]
    return options


def test_no_batch_size_is_sent_unless_one_is_set() -> None:
    options = request_options(make_chat_model(settings()))
    assert "num_batch" not in options
    assert options["num_ctx"] == 8192


def test_a_batch_size_goes_into_every_request_next_to_the_context_size() -> None:
    config = settings(chat_num_batch=256, chat_num_ctx=4096)
    options = request_options(make_chat_model(config))
    assert options["num_batch"] == 256
    assert options["num_ctx"] == 4096  # the other load-time option is still there
    thinking = request_options(make_chat_model(config, think=True))
    assert thinking["num_batch"] == 256


def test_the_rewrite_model_gets_the_batch_size_too_but_another_models_does_not() -> None:
    """A follow-up's rewrite uses the chat model, so it must not make Ollama reload it (R-2); the
    CSV model is loaded without a batch size and must not be given one."""
    config = settings(chat_num_batch=256)
    chat = make_structured_model(config, CondenseOutput, num_predict=96).first.bound  # type: ignore[attr-defined]
    assert request_options(chat)["num_batch"] == 256
    other = make_structured_model(config, CondenseOutput, num_predict=96, model="other:2b")
    assert "num_batch" not in request_options(other.first.bound)  # type: ignore[attr-defined]


def test_an_explicit_options_argument_is_not_overwritten() -> None:
    """LangChain lets a call pass its own ``options`` (then none of the field values are used)."""
    chat = make_chat_model(settings(chat_num_batch=256))
    options = request_options(chat, options={"num_ctx": 1234})
    assert options == {"num_ctx": 1234, "num_batch": 256}


@pytest.mark.parametrize("bad", [0, -1])
def test_a_batch_size_of_zero_or_less_is_refused(bad: int) -> None:
    with pytest.raises(ValidationError):
        settings(chat_num_batch=bad)


def test_the_conversation_settings_have_starting_values_and_refuse_nonsense() -> None:
    config = settings()
    assert (config.history_budget_tokens, config.history_keep_messages) == (1000, 40)
    assert config.condense_num_predict == 96  # PLAN.md section 6.5
    for bad in (
        {"history_budget_tokens": 0},
        {"history_keep_messages": 1},
        {"condense_num_predict": 0},
    ):
        with pytest.raises(ValidationError):
            settings(**bad)


def test_the_default_thinking_budget_leaves_room_for_the_reasoning() -> None:
    config = settings()
    assert config.chat_think_num_predict > config.chat_num_predict


@pytest.mark.parametrize(
    "bad",
    [
        {"chat_num_predict": 0},
        {"chat_think_num_predict": 0},
        {"chat_temperature": 3.0},
        {"context_budget_tokens": 0},
    ],
)
def test_nonsense_limits_are_refused_at_start_up(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**bad)
