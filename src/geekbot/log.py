"""Logging setup (lesson: project/learn/02-config-cli.md §5).

Named ``log.py``, not ``logging.py``, so nobody mistakes it for the standard-library module.
"""

import logging

LOGGER_NAME = "geekbot"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the ``geekbot`` logger and return it.

    We configure only our own logger, never the root logger: other libraries (httpx, chromadb)
    keep their own levels and don't flood the output at DEBUG. Safe to call twice: the handler
    is added once, and a later call only changes the level (tests and the CLI both call it).
    Modules get their logger with ``logging.getLogger(__name__)``, which is a child of this one.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()  # stderr, so stdout stays clean for command output
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False  # no duplicate lines if the root logger is configured too
    return logger
