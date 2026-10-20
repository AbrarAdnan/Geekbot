"""How ``geekbot up`` starts the UI next to the server (step S4.4).

``geekbot up`` runs two processes: the API server (this process, the one that owns the storage) and
Streamlit (a child process). This module builds the child's command line and environment, starts and
stops it, and opens the browser once the server answers. None of it imports Streamlit or the server,
so the command can be assembled, and tested, without either.

**What the command line fixes, and why**

- ``--server.address 127.0.0.1``: Streamlit listens on every network interface unless told
  otherwise. The UI holds the API key in its memory and has no login of its own, so a UI reachable
  from the local network would hand the whole assistant to anyone on it.
- ``--server.headless true``: no "enter your email" prompt, and no browser opened by Streamlit
  (``geekbot up`` opens it itself, when the server is ready, not before).
- ``--browser.gatherUsageStats false``: Streamlit's usage telemetry is on by default. Geekbot is a
  local-first tool and sends nothing anywhere.
- ``--server.maxUploadSize``: Streamlit's own cap on a chosen file, a little under the server's
  ``MAX_UPLOAD_BYTES`` (100,000,000), so the UI never accepts a file the server will refuse.
"""

import importlib.util
import os
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable, Mapping
from typing import Final

import httpx

from geekbot.ui import APP_SCRIPT
from geekbot.ui.client import LOCAL_HOST

__all__ = [
    "MAX_UPLOAD_MB",
    "health_url",
    "open_browser_when_ready",
    "start_streamlit",
    "stop_child",
    "streamlit_available",
    "streamlit_command",
    "streamlit_env",
    "ui_url",
]

# Streamlit counts megabytes of 1,048,576 bytes; 95 of them are 99,614,720, under the server's cap.
MAX_UPLOAD_MB: Final = 95
BROWSER_WAIT_S: Final = 300.0  # a cold start loads the chat model first; that can take a while
BROWSER_POLL_S: Final = 0.5
PROBE_TIMEOUT_S: Final = 1.0
STOP_GRACE_S: Final = 5.0


def streamlit_available() -> bool:
    """Whether the ``streamlit`` package can be imported (without importing it)."""
    return importlib.util.find_spec("streamlit") is not None


def ui_url(ui_port: int) -> str:
    return f"http://{LOCAL_HOST}:{ui_port}"


def health_url(server_port: int) -> str:
    """The server's ``/healthz``, which answers once the server listens (and needs no key)."""
    return f"http://{LOCAL_HOST}:{server_port}/healthz"


def streamlit_command(ui_port: int, *, python: str | None = None) -> list[str]:
    """``python -m streamlit run <entry script>`` and the options the module docstring explains."""
    return [
        python or sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
        "--server.address",
        LOCAL_HOST,
        "--server.port",
        str(ui_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.maxUploadSize",
        str(MAX_UPLOAD_MB),
    ]


def streamlit_env(server_port: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: the parent's, plus the port the server really listens on.

    ``geekbot up --port 9000`` overrides ``GEEKBOT_SERVER_PORT`` for this run only; the UI reads the
    same setting the server does, so it is handed the value that is in force.
    """
    environment = dict(os.environ if base is None else base)
    environment["GEEKBOT_SERVER_PORT"] = str(server_port)
    return environment


def start_streamlit(command: list[str], environment: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start the UI as a child process that shares this terminal."""
    return subprocess.Popen(command, env=environment)


def stop_child(child: subprocess.Popen[bytes], grace_seconds: float = STOP_GRACE_S) -> None:
    """Ask the child to stop, and kill it if it has not by ``grace_seconds``. Safe if it is gone."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _healthz_answers(url: str) -> bool:
    try:
        return httpx.get(url, timeout=PROBE_TIMEOUT_S).is_success
    except httpx.TransportError:
        return False


def open_browser_when_ready(
    health_url: str,
    page_url: str,
    *,
    probe: Callable[[str], bool] = _healthz_answers,
    opener: Callable[[str], object] = webbrowser.open,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    timeout_seconds: float = BROWSER_WAIT_S,
) -> bool:
    """Open ``page_url`` once ``health_url`` answers; ``False`` if it does not within the wait.

    ``geekbot up`` runs this in a background thread while the server loads the chat model, so the
    page is not opened on a server that is still starting (the UI would only say "not reachable").
    The probe, the opener and the clock are parameters so a test needs no network, browser or wait.
    """
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        if probe(health_url):
            opener(page_url)
            return True
        sleep(BROWSER_POLL_S)
    return False
