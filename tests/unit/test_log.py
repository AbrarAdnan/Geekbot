"""Tests for step S0.2: logging setup."""

import logging

from geekbot.log import LOGGER_NAME, setup_logging


def test_setup_is_idempotent_and_changes_the_level() -> None:
    logger = setup_logging("INFO")
    handlers_after_first = len(logger.handlers)
    setup_logging("DEBUG")
    assert len(logger.handlers) == handlers_after_first == 1
    assert logger.level == logging.DEBUG


def test_child_loggers_inherit_from_the_geekbot_logger() -> None:
    setup_logging("WARNING")
    child = logging.getLogger(f"{LOGGER_NAME}.ingest")
    assert child.getEffectiveLevel() == logging.WARNING
