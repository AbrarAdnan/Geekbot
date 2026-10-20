"""Tests for step S0.2: the Typer CLI. Stub commands must say which step builds them."""

import re
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from geekbot import __version__
from geekbot.cli import NOT_BUILT_EXIT_CODE, _not_built, app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"geekbot {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "config" in result.output
    assert "doctor" in result.output


def test_not_built_exits_2_and_names_its_step() -> None:
    """No command is a stub right now (`serve` was the last, S4.1); the helper stays for the next."""
    with pytest.raises(typer.Exit) as raised:
        _not_built("future", "S9.9")
    assert raised.value.exit_code == NOT_BUILT_EXIT_CODE == 2


def test_serve_has_no_host_or_workers_option() -> None:
    """PLAN §6.2: one process, this machine only; the command offers no way around that."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    # CI turns colours on, and the colour codes split an option name into styled pieces.
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--port" in text
    assert "--workers" not in text
    assert "--host" not in text


def test_serve_offers_a_way_to_skip_the_warm_up() -> None:
    text = re.sub(r"\x1b\[[0-9;]*m", "", runner.invoke(app, ["serve", "--help"]).output)
    assert "--no-warmup" in text


@pytest.mark.usefixtures("clean_geekbot_env")
def test_serve_stops_with_one_line_when_the_chat_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4.2: the problem is told before the web server starts, not as a traceback from inside it."""
    problem = "the chat model gemma3:4b is not pulled: run `ollama pull gemma3:4b`"
    monkeypatch.setattr("geekbot.cli.find_chat_problem", lambda settings, **kwargs: problem)

    def must_not_start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the server must not start")

    monkeypatch.setattr("uvicorn.run", must_not_start)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert f"error: {problem}" in result.output


@pytest.mark.usefixtures("clean_geekbot_env")
@pytest.mark.parametrize(
    ("options", "warm_up"), [([], True), (["--no-warmup"], False)], ids=["default", "no-warmup"]
)
def test_serve_hands_the_warm_up_choice_the_port_and_a_perimeter_to_the_app(
    monkeypatch: pytest.MonkeyPatch, options: list[str], warm_up: bool
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("geekbot.cli.find_chat_problem", lambda settings, **kwargs: None)

    def fake_create_app(settings: Any, **kwargs: Any) -> str:
        seen["create_app"] = kwargs
        return "the app"

    def fake_run(application: Any, **kwargs: Any) -> None:
        seen["run"] = (application, kwargs)

    monkeypatch.setattr("geekbot.server.app.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["serve", "--port", "9123", *options])
    assert result.exit_code == 0, result.output
    assert set(seen["create_app"]) == {"warm_up", "perimeter"}
    assert seen["create_app"]["warm_up"] is warm_up
    perimeter = seen["create_app"]["perimeter"]
    assert perimeter.api_key  # S4.3: a key is always required
    assert perimeter.hosts == ("127.0.0.1", "localhost")
    application, kwargs = seen["run"]
    assert application == "the app"
    assert kwargs == {"host": "127.0.0.1", "port": 9123, "workers": 1, "log_level": "warning"}


def stub_the_server(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Let ``serve`` run to the point of starting uvicorn, which is recorded instead of started."""
    started: list[str] = []
    monkeypatch.setattr("geekbot.cli.find_chat_problem", lambda settings, **kwargs: None)
    monkeypatch.setattr("geekbot.server.app.create_app", lambda settings, **kwargs: "the app")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: started.append("started"))
    return started


@pytest.mark.usefixtures("clean_geekbot_env")
def test_serve_makes_a_key_the_first_time_and_never_prints_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub_the_server(monkeypatch)
    first = runner.invoke(app, ["serve"])
    assert first.exit_code == 0, first.output
    key = (tmp_path / "data" / "api-key").read_text(encoding="utf-8").strip()
    assert key
    assert key not in first.output
    assert "API key from" in first.output
    assert "(new)" in first.output
    second = runner.invoke(app, ["serve"])
    assert second.exit_code == 0, second.output
    assert "(new)" not in second.output  # the same key is read back
    assert (tmp_path / "data" / "api-key").read_text(encoding="utf-8").strip() == key


@pytest.mark.usefixtures("clean_geekbot_env")
def test_serve_tells_where_a_configured_key_came_from_not_what_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub_the_server(monkeypatch)
    configured = "c" * 32
    monkeypatch.setenv("GEEKBOT_API_KEY", configured)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert "API key from GEEKBOT_API_KEY" in result.output
    assert configured not in result.output
    assert not (tmp_path / "data" / "api-key").exists()  # nothing is written when it is configured


@pytest.mark.usefixtures("clean_geekbot_env")
@pytest.mark.parametrize(
    ("variable", "value", "complaint"),
    [
        ("GEEKBOT_API_KEY", "short", "at least 24 characters"),
        ("GEEKBOT_ALLOWED_ROOTS", "relative/folder", "not a full path"),
    ],
    ids=["weak-key", "relative-root"],
)
def test_serve_stops_with_one_line_on_a_bad_perimeter_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str, value: str, complaint: str
) -> None:
    started = stub_the_server(monkeypatch)
    monkeypatch.setenv(variable, value)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert complaint in result.output
    assert started == []  # the server never started
    assert not (tmp_path / "data" / "api-key").exists()  # and no key file was left behind
