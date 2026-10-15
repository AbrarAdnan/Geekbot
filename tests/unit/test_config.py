"""Tests for step S0.2: how settings are read and which models a profile resolves to.

``_env_file=None`` keeps a developer's real ``.env`` out of the tests, so they behave the same
on every machine.
"""

import os

import pytest
from pydantic import ValidationError

from geekbot.config import PROFILES, SLOS, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A GEEKBOT_* variable set in the shell running pytest must not change the results.
    for name in list(os.environ):
        if name.startswith("GEEKBOT_"):
            monkeypatch.delenv(name)


def test_default_profile_is_proto_with_light_models() -> None:
    settings = Settings(_env_file=None)
    assert settings.profile == "proto"
    assert settings.resolved_chat_model == "gemma3:4b"
    assert settings.resolved_csv_model == "qwen2.5-coder:3b"


def test_target_profile_resolves_to_the_plan_models() -> None:
    settings = Settings(_env_file=None, profile="target")
    assert settings.resolved_chat_model == "gemma4:12b"
    assert settings.resolved_csv_model == "qwen2.5-coder:7b"


def test_no_profile_uses_a_model_over_12b() -> None:
    # CLAUDE.md rule 4. A crude guard: every profile's tags must name a size of at most 12b.
    for model_set in PROFILES.values():
        for tag in (model_set.chat, model_set.csv):
            size = float(tag.rsplit(":", 1)[1].removesuffix("b"))
            assert size <= 12


def test_model_override_beats_the_profile() -> None:
    settings = Settings(_env_file=None, profile="target", chat_model="gemma3:4b")
    assert settings.resolved_chat_model == "gemma3:4b"
    assert settings.resolved_csv_model == "qwen2.5-coder:7b"  # not overridden


def test_values_are_read_from_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEEKBOT_PROFILE", "target")
    monkeypatch.setenv("GEEKBOT_LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.profile == "target"
    assert settings.log_level == "DEBUG"


def test_unprefixed_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE", "target")
    assert Settings(_env_file=None).profile == "proto"


def test_a_bad_profile_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEEKBOT_PROFILE", "turbo")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_are_frozen() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.profile = "target"  # type: ignore[misc]


def test_secret_is_hidden_in_repr() -> None:
    settings = Settings(_env_file=None, api_key="s3cret")  # type: ignore[arg-type]
    assert "s3cret" not in repr(settings)


def test_slos_match_the_plan() -> None:
    # PLAN.md §6.5. If the plan's targets change, this test and config.py change together.
    assert SLOS.retrieval_rerank_p95_ms == 400
    assert (SLOS.first_turn_ttft_p50_s, SLOS.first_turn_ttft_p95_s) == (2.5, 4.0)
    assert SLOS.followup_ttft_p95_s == 5.0
    assert SLOS.end_to_end_p95_s == 15.0
    assert SLOS.model_loads_during_queries == 0
