"""Tests for step S0.2: the Typer CLI. Stub commands must say which step builds them."""

import re
import time
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
def test_serve_stops_with_one_line_when_another_process_owns_the_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gate A: a second `serve` used to die inside uvicorn with a traceback and exit code 3."""
    from geekbot.index.lock import StorageLock

    started = stub_the_server(monkeypatch)
    with StorageLock(tmp_path / "data"):  # the first server, as far as the lock can tell
        result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "error: another Geekbot process already owns the storage" in result.output
    assert "Traceback" not in result.output
    assert started == []  # the server never started


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


# --- `geekbot up` (step S4.4) ----------------------------------------------------------------


class FakeChild:
    """Stands in for the Streamlit child process."""


class Launched:
    """What the fake launcher recorded: the child started, what it was given, what was stopped."""

    def __init__(self) -> None:
        self.child = FakeChild()
        self.started: list[tuple[list[str], dict[str, str]]] = []
        self.stopped: list[object] = []
        self.browser: list[tuple[str, str]] = []
        self.server: list[str] = []


def stub_up(monkeypatch: pytest.MonkeyPatch, *, server_fails: bool = False) -> Launched:
    """Let ``up`` run to the point of starting uvicorn and the child, both recorded, not started."""
    launched = Launched()

    def start(command: list[str], environment: dict[str, str]) -> FakeChild:
        launched.started.append((command, environment))
        return launched.child

    def run_server(*args: Any, **kwargs: Any) -> None:
        launched.server.append("started")
        if server_fails:
            raise RuntimeError("the storage is locked")

    monkeypatch.setattr("geekbot.cli.find_chat_problem", lambda settings, **kwargs: None)
    monkeypatch.setattr("geekbot.server.app.create_app", lambda settings, **kwargs: "the app")
    monkeypatch.setattr("uvicorn.run", run_server)
    monkeypatch.setattr("geekbot.ui.launch.streamlit_available", lambda: True)
    monkeypatch.setattr("geekbot.ui.launch.start_streamlit", start)
    monkeypatch.setattr("geekbot.ui.launch.stop_child", launched.stopped.append)
    monkeypatch.setattr(
        "geekbot.ui.launch.open_browser_when_ready",
        lambda health, page, **kwargs: launched.browser.append((health, page)),
    )
    return launched


def option_after(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_up_offers_ports_and_no_way_to_widen_the_bind() -> None:
    text = re.sub(r"\x1b\[[0-9;]*m", "", runner.invoke(app, ["up", "--help"]).output)
    for wanted in ("--port", "--ui-port", "--no-warmup", "--no-browser"):
        assert wanted in text
    assert "--host" not in text
    assert "--workers" not in text


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_starts_the_ui_on_its_port_tells_it_the_servers_and_stops_it_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = stub_up(monkeypatch)
    result = runner.invoke(app, ["up", "--port", "9123", "--ui-port", "9124", "--no-browser"])
    assert result.exit_code == 0, result.output
    ((command, environment),) = launched.started
    assert option_after(command, "--server.port") == "9124"
    assert option_after(command, "--server.address") == "127.0.0.1"
    assert environment["GEEKBOT_SERVER_PORT"] == "9123"  # the UI is told which port the server has
    assert launched.server == ["started"]
    assert launched.stopped == [launched.child]  # the child does not outlive the server
    assert "Geekbot UI on http://127.0.0.1:9124" in result.output
    assert launched.browser == []  # --no-browser


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_opens_the_browser_when_the_server_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = stub_up(monkeypatch)
    result = runner.invoke(app, ["up"])
    assert result.exit_code == 0, result.output
    for _ in range(200):  # the browser is opened from a background thread
        if launched.browser:
            break
        time.sleep(0.01)
    assert launched.browser == [("http://127.0.0.1:8000/healthz", "http://127.0.0.1:8501")]


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_stops_the_ui_even_when_the_server_fails_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = stub_up(monkeypatch, server_fails=True)
    result = runner.invoke(app, ["up", "--no-browser"])
    assert result.exit_code != 0  # the failure is not swallowed
    assert launched.stopped == [launched.child]


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_stops_with_one_line_when_streamlit_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = stub_up(monkeypatch)
    monkeypatch.setattr("geekbot.ui.launch.streamlit_available", lambda: False)
    result = runner.invoke(app, ["up"])
    assert result.exit_code == 1
    assert "streamlit" in result.output and "pip install" in result.output
    assert launched.started == [] and launched.server == []


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_refuses_to_give_the_server_and_the_ui_one_port(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = stub_up(monkeypatch)
    result = runner.invoke(app, ["up", "--port", "8501"])  # 8501 is the UI's default port
    assert result.exit_code == 1
    assert "cannot both use port 8501" in result.output
    assert launched.started == [] and launched.server == []


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_checks_the_model_and_the_key_before_it_starts_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched = stub_up(monkeypatch)
    monkeypatch.setattr("geekbot.cli.find_chat_problem", lambda settings, **kwargs: "model missing")
    result = runner.invoke(app, ["up"])
    assert result.exit_code == 1
    assert "error: model missing" in result.output
    assert launched.started == [] and launched.server == []
    assert not (tmp_path / "data" / "api-key").exists()


@pytest.mark.usefixtures("clean_geekbot_env")
def test_up_stops_before_the_ui_when_another_process_owns_the_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from geekbot.index.lock import StorageLock

    launched = stub_up(monkeypatch)
    with StorageLock(tmp_path / "data"):
        result = runner.invoke(app, ["up", "--no-browser"])
    assert result.exit_code == 1
    assert "error: another Geekbot process already owns the storage" in result.output
    assert launched.started == [] and launched.server == []  # no UI child, no server
