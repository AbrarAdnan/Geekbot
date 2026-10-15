"""Tests for the ✍️ piece of step S0.2: ``mask_settings`` and ``geekbot config show``.

These fail with ``NotImplementedError`` until you write ``mask_settings`` in
``src/geekbot/config.py``. They are the spec: when they pass, you're done.
"""

import json
import os

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.config import Settings, mask_settings

SECRET = "s3cret-value-123"  # a test value, not a real secret


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("GEEKBOT_"):
            monkeypatch.delenv(name)


def test_keys_and_order() -> None:
    masked = mask_settings(Settings(_env_file=None))
    assert list(masked) == [
        "profile",
        "chat_model",
        "csv_model",
        "ollama_host",
        "data_dir",
        "log_level",
        "api_key",
    ]


def test_models_are_the_resolved_ones() -> None:
    masked = mask_settings(Settings(_env_file=None, profile="target", csv_model="my-csv:1b"))
    assert masked["profile"] == "target"
    assert masked["chat_model"] == "gemma4:12b"  # from the profile
    assert masked["csv_model"] == "my-csv:1b"  # the override wins


def test_unset_secret_is_none() -> None:
    assert mask_settings(Settings(_env_file=None))["api_key"] is None


def test_set_secret_is_masked_and_never_leaks() -> None:
    masked = mask_settings(Settings(_env_file=None, api_key=SECRET))  # type: ignore[arg-type]
    assert masked["api_key"] == "********"
    assert SECRET not in json.dumps(masked)


def test_data_dir_is_a_plain_string() -> None:
    masked = mask_settings(Settings(_env_file=None))
    assert isinstance(masked["data_dir"], str)
    json.dumps(masked)  # everything in it must be JSON-serialisable


def test_config_show_prints_masked_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEEKBOT_API_KEY", SECRET)
    monkeypatch.setenv("GEEKBOT_PROFILE", "target")
    result = CliRunner().invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert SECRET not in result.output
    printed = json.loads(result.output)
    assert printed["api_key"] == "********"
    assert printed["chat_model"] == "gemma4:12b"
