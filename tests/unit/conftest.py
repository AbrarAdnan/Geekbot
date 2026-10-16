"""Fixtures shared by every test in ``tests/unit``."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from geekbot.log import LOGGER_NAME


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
