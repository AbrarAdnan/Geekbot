"""Fixtures shared by every test in ``tests/unit``."""

import importlib.util
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from geekbot.agent import graph as graph_module
from geekbot.agent.resilience import transient_retry
from geekbot.log import LOGGER_NAME

# The two test files that import the UI's pages need Streamlit at collection time. In an environment
# that has not installed it yet (S4.4 added the dependency) they are left out, instead of one import
# error stopping the whole run; ``test_ui_launch.py::test_streamlit_is_found_in_this_environment``
# is the loud one that says what is missing.
collect_ignore: list[str] = (
    [] if importlib.util.find_spec("streamlit") else ["test_ui_knowledge.py", "test_ui_pages.py"]
)


@pytest.fixture
def clean_geekbot_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make ``Settings()`` see the defaults: no ``GEEKBOT_*`` variable and no ``.env`` file.

    Request it with ``pytestmark = pytest.mark.usefixtures("clean_geekbot_env")``. A variable set
    in the shell that runs pytest, or a ``.env`` in the working directory, must not change a
    result (the CLI builds its own ``Settings()``, which reads both).
    """
    for name in list(os.environ):
        if name.startswith("GEEKBOT_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _retries_do_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the answer graph's default retry policy retry at once (S3.6).

    The real policy waits about 0.5 s, then 1 s, plus jitter, before it runs a step again. Many tests
    script a refused connection to see what a failing model does; with the real waits each of them
    would sit for seconds. The policy is the same otherwise (same attempts, same ``retry_on``), and
    ``test_resilience.py`` checks the real numbers by calling ``transient_retry()`` itself, which this
    does not touch.
    """
    monkeypatch.setattr(
        graph_module, "transient_retry", lambda: transient_retry(initial_interval=0.0, jitter=False)
    )


@pytest.fixture(autouse=True)
def _restore_geekbot_logger() -> Iterator[None]:
    """Give every test the ``geekbot`` logger as it found it.

    ``setup_logging`` changes global state: it adds a handler, sets the level and turns
    propagation off, and Python keeps that for the rest of the test run. Any test that runs a CLI
    command calls it, so without this the next test to look at the logger (``test_log.py``) sees
    handlers it did not create. Found at C2: the new CSV CLI tests made ``test_log`` fail with
    "assert 3 == 1" when the whole suite ran, although it passes alone.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
