"""Fixtures shared by every test in ``tests/unit``."""

import logging
from collections.abc import Iterator

import pytest

from geekbot.log import LOGGER_NAME


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
