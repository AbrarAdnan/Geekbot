"""The ``geekbot`` command.

A placeholder from step S0.1: it gives the ``[project.scripts]`` entry in pyproject.toml real
code to point at, so ``geekbot --version`` proves the install works end to end. Step S0.2
replaces it with the Typer CLI.
"""

import argparse
from collections.abc import Sequence

from geekbot import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return its exit code (0 = success).

    The launcher pip generates for ``[project.scripts]`` calls ``sys.exit(main())``, so the
    returned int becomes the process exit code. ``argv=None`` makes argparse read
    ``sys.argv``; tests pass their own list instead. Like any argparse CLI, ``--version``
    and ``--help`` print and exit straight away (``SystemExit(0)``).
    """
    parser = argparse.ArgumentParser(
        prog="geekbot",
        description="Geekbot v2: a local RAG engine (work in progress).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
