"""Tests for step S4.4: how ``geekbot up`` starts the UI (``geekbot.ui.launch``).

Nothing is started: the command line and the environment are plain data, and the child process, the
probe, the browser and the clock are replaced by fakes.
"""

import itertools
import subprocess
import sys
from typing import Any

import pytest

from geekbot.config import Settings
from geekbot.server.security import MAX_UPLOAD_BYTES
from geekbot.ui import APP_SCRIPT
from geekbot.ui.launch import (
    MAX_UPLOAD_MB,
    health_url,
    open_browser_when_ready,
    stop_child,
    streamlit_available,
    streamlit_command,
    streamlit_env,
    ui_url,
)


def option(command: list[str], name: str) -> str:
    """The value that follows ``name`` on the command line."""
    return command[command.index(name) + 1]


# --- the command line ------------------------------------------------------------------------


def test_the_entry_script_exists_and_is_what_streamlit_runs() -> None:
    assert APP_SCRIPT.name == "app.py"
    assert APP_SCRIPT.is_file()
    command = streamlit_command(8501)
    assert command[1:5] == ["-m", "streamlit", "run", str(APP_SCRIPT)]


def test_the_ui_listens_on_the_loopback_address_only_and_sends_no_telemetry() -> None:
    command = streamlit_command(9123)
    assert option(command, "--server.address") == "127.0.0.1"
    assert option(command, "--server.port") == "9123"
    assert option(command, "--server.headless") == "true"
    assert option(command, "--browser.gatherUsageStats") == "false"
    assert command.count("--server.address") == 1  # one address, and it is the one above


def test_the_python_is_the_one_running_geekbot_unless_told_otherwise() -> None:
    assert streamlit_command(8501)[0] == sys.executable
    assert streamlit_command(8501, python="/venv/bin/python")[0] == "/venv/bin/python"


def test_the_ui_never_accepts_a_file_the_server_would_refuse() -> None:
    """Streamlit counts megabytes of 1,048,576 bytes; the server's cap is 100,000,000 bytes."""
    assert MAX_UPLOAD_MB * 1024 * 1024 <= MAX_UPLOAD_BYTES
    assert option(streamlit_command(8501), "--server.maxUploadSize") == str(MAX_UPLOAD_MB)


def test_the_urls() -> None:
    assert ui_url(8501) == "http://127.0.0.1:8501"
    assert health_url(8000) == "http://127.0.0.1:8000/healthz"


def test_the_default_ui_port_is_streamlits_own_and_differs_from_the_servers() -> None:
    settings = Settings(_env_file=None)
    assert settings.ui_port == 8501
    assert settings.ui_port != settings.server_port


# --- the environment -------------------------------------------------------------------------


def test_the_child_is_told_the_port_the_server_really_uses() -> None:
    environment = streamlit_env(9000, {"PATH": "/bin", "GEEKBOT_SERVER_PORT": "8000"})
    assert environment == {"PATH": "/bin", "GEEKBOT_SERVER_PORT": "9000"}


def test_the_childs_environment_is_the_parents_by_default_and_a_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEEKBOT_TEST_MARKER", "yes")
    environment = streamlit_env(8000)
    assert environment["GEEKBOT_TEST_MARKER"] == "yes"
    environment["GEEKBOT_TEST_MARKER"] = "changed"
    assert streamlit_env(8000)["GEEKBOT_TEST_MARKER"] == "yes"  # the parent's was not touched


def test_streamlit_is_found_in_this_environment() -> None:
    assert streamlit_available() is True  # it is a dependency: if this fails, the env is stale


# --- starting and stopping the child ---------------------------------------------------------


class FakeChild:
    """What ``stop_child`` needs of a ``subprocess.Popen``, recording what it was asked."""

    def __init__(self, *, running: bool = True, stubborn: bool = False) -> None:
        self.running = running
        self.stubborn = stubborn
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.calls.append("terminate")
        self.running = self.stubborn

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if self.running:
            raise subprocess.TimeoutExpired("streamlit", timeout or 0)
        return 0

    def kill(self) -> None:
        self.calls.append("kill")
        self.running = False


def test_a_child_that_already_exited_is_left_alone() -> None:
    child = FakeChild(running=False)
    stop_child(child)  # type: ignore[arg-type]
    assert child.calls == []


def test_a_child_that_stops_when_asked_is_not_killed() -> None:
    child = FakeChild()
    stop_child(child)  # type: ignore[arg-type]
    assert child.calls == ["terminate", "wait"]


def test_a_child_that_ignores_the_request_is_killed_after_the_grace_period() -> None:
    child = FakeChild(stubborn=True)
    stop_child(child, grace_seconds=0.01)  # type: ignore[arg-type]
    assert child.calls == ["terminate", "wait", "kill", "wait"]
    assert child.running is False


# --- opening the browser ---------------------------------------------------------------------


def test_the_browser_opens_once_the_server_answers_and_not_before() -> None:
    answers = iter([False, False, True])
    probed: list[str] = []
    opened: list[str] = []
    slept: list[float] = []

    def probe(url: str) -> bool:
        probed.append(url)
        return next(answers)

    result = open_browser_when_ready(
        "http://127.0.0.1:8000/healthz",
        "http://127.0.0.1:8501",
        probe=probe,
        opener=opened.append,
        sleep=slept.append,
        clock=itertools.count().__next__,
    )
    assert result is True
    assert probed == ["http://127.0.0.1:8000/healthz"] * 3
    assert opened == ["http://127.0.0.1:8501"]  # opened once, at the end
    assert len(slept) == 2


def test_the_browser_is_never_opened_on_a_server_that_never_answers() -> None:
    opened: list[str] = []
    ticks = itertools.count()
    result = open_browser_when_ready(
        "http://127.0.0.1:8000/healthz",
        "http://127.0.0.1:8501",
        probe=lambda url: False,
        opener=opened.append,
        sleep=lambda seconds: None,
        clock=lambda: float(next(ticks)),
        timeout_seconds=5,
    )
    assert result is False
    assert opened == []


def test_the_default_probe_treats_a_refused_connection_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no server on the port, ``open_browser_when_ready`` must wait, not raise."""
    import httpx

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", refuse)
    ticks = itertools.count()
    assert (
        open_browser_when_ready(
            "http://127.0.0.1:1/healthz",
            "http://127.0.0.1:2",
            opener=lambda url: pytest.fail("must not open"),
            sleep=lambda seconds: None,
            clock=lambda: float(next(ticks)),
            timeout_seconds=3,
        )
        is False
    )
