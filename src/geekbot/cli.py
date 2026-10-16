"""The ``geekbot`` command, built with Typer (lesson: project/learn/02-config-cli.md §6).

Typer turns typed Python functions into CLI commands. Commands whose step hasn't been built
yet are stubs: they say which step builds them and exit with code 2, so a script or test can't
mistake "not built" for success.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from geekbot import __version__
from geekbot.config import Settings, mask_settings
from geekbot.doctor import format_checks, has_failure, run_doctor
from geekbot.llm.models import OllamaClient
from geekbot.llm.vram import (
    NVIDIA_SMI_MEMORY_COMMAND,
    OLLAMA_PS_COMMAND,
    parse_nvidia_smi_memory,
    parse_ollama_ps,
    run_command,
)
from geekbot.log import setup_logging

# show_locals off: a crash traceback must never print a local variable that holds a secret.
app = typer.Typer(
    name="geekbot",
    help="Geekbot v2: a local RAG engine (work in progress).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Inspect the settings.", no_args_is_help=True)
csv_app = typer.Typer(help="CSV analyst mode.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(csv_app, name="csv")

# Stubs accept any arguments, so `geekbot ask "hi"` reaches the "not built yet" message
# instead of a usage error.
_STUB = {"allow_extra_args": True, "ignore_unknown_options": True}
NOT_BUILT_EXIT_CODE = 2


def _not_built(command: str, step: str) -> None:
    typer.echo(f"`geekbot {command}` is not built yet (step {step}).", err=True)
    raise typer.Exit(NOT_BUILT_EXIT_CODE)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"geekbot {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Runs before every command (``--version`` is handled eagerly, before this body)."""
    del version  # handled by the callback


@config_app.command("show")
def config_show() -> None:
    """Print the effective settings (secrets masked)."""
    settings = Settings()
    setup_logging(settings.log_level)
    logging.getLogger(__name__).debug("config show")
    typer.echo(json.dumps(mask_settings(settings), indent=2))


@app.command()
def doctor(
    warmup: Annotated[
        bool,
        typer.Option(
            "--warmup/--no-warmup",
            help="Load the chat model onto the GPU and check it (uses the GPU for a moment).",
        ),
    ] = True,
) -> None:
    """Check Ollama, the models and the GPU. Exits 1 if a plan rule is broken."""
    settings = Settings()
    setup_logging(settings.log_level)
    with OllamaClient(settings.ollama_host) as client:
        checks = run_doctor(
            settings,
            client,
            warm_up=warmup,
            read_gpu=lambda: parse_nvidia_smi_memory(run_command(NVIDIA_SMI_MEMORY_COMMAND)),
            read_ps=lambda: parse_ollama_ps(run_command(OLLAMA_PS_COMMAND)),
        )
    typer.echo(format_checks(checks))
    if has_failure(checks):
        raise typer.Exit(1)


@app.command(context_settings=_STUB)
def ingest() -> None:
    """Index documents."""
    _not_built("ingest", "S1.6")


@app.command(context_settings=_STUB)
def search() -> None:
    """Search the knowledge base."""
    _not_built("search", "S2.1")


@app.command(context_settings=_STUB)
def ask() -> None:
    """Ask a question."""
    _not_built("ask", "S3.1")


@app.command(context_settings=_STUB)
def serve() -> None:
    """Start the API server."""
    _not_built("serve", "S4.1")


CsvFile = Annotated[
    Path,
    typer.Argument(help="The CSV file.", exists=True, dir_okay=False, readable=True),
]


def _fail(message: str) -> NoReturn:
    """One clear line on stderr and exit code 1 (a usage mistake exits with 2, via Typer)."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


@csv_app.command("describe")
def csv_describe(file: CsvFile) -> None:
    """Show what was loaded from a CSV file: rows, columns and warnings. Uses no model."""
    # Imported here, not at the top: the CSV package pulls in pandas and pyarrow, and
    # `geekbot --version` or `geekbot doctor` should not wait for them.
    from geekbot.csv_mode.adapter import CsvAnalyst, CsvLoadError

    settings = Settings()
    setup_logging(settings.log_level)
    try:
        with CsvAnalyst.open(file, settings) as analyst:
            typer.echo(analyst.describe())
    except CsvLoadError as error:
        _fail(str(error))


@csv_app.command("ask")
def csv_ask(
    file: CsvFile,
    question: Annotated[str, typer.Argument(help="Your question about the data.")],
) -> None:
    """Ask a question about a CSV file (uses the CSV model, so it needs a free GPU).

    The reply goes to stdout. A footer on stderr names the model, profile, status, time and the
    time spent loading the model, so a pasted result always says which model produced it
    (D-043). Exit code 0 means the tool worked (even if the data could not answer); 1 means the
    question failed or Ollama, the model or the file was not usable.
    """
    from geekbot.csv_mode.adapter import CsvAnalyst, CsvLoadError, find_problem

    settings = Settings()
    setup_logging(settings.log_level)
    problem = find_problem(settings)
    if problem is not None:
        _fail(problem)
    try:
        with CsvAnalyst.open(file, settings) as analyst:
            answer = analyst.ask(question)
    except CsvLoadError as error:
        _fail(str(error))
    typer.echo(answer.markdown)
    typer.echo(
        f"[csv] model={answer.model} profile={settings.profile} status={answer.status} "
        f"time={answer.seconds:.1f}s model_load={answer.model_load_ms}ms",
        err=True,
    )
    if answer.failed:
        raise typer.Exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point named in pyproject.toml's ``[project.scripts]``.

    Typer exits the process itself (``SystemExit``), so this returns nothing. ``argv=None``
    reads ``sys.argv``; tests pass their own list.
    """
    app(args=None if argv is None else list(argv), prog_name="geekbot")
