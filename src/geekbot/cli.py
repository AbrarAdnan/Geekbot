"""The ``geekbot`` command, built with Typer (lesson: project/learn/02-config-cli.md §6).

Typer turns typed Python functions into CLI commands. Commands whose step hasn't been built
yet are stubs: they say which step builds them and exit with code 2, so a script or test can't
mistake "not built" for success.
"""

import json
import logging
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from geekbot import __version__
from geekbot.config import Mode, Settings, mask_settings
from geekbot.doctor import format_checks, has_failure, run_doctor
from geekbot.evals import DEFAULT_RUNS, DEFAULT_SOURCE, DEFAULT_TIME_LIMIT_S, SUITE_NAMES
from geekbot.llm.models import OllamaClient, OllamaError, model_is_pulled
from geekbot.llm.vram import (
    NVIDIA_SMI_MEMORY_COMMAND,
    OLLAMA_PS_COMMAND,
    GpuMemory,
    LoadedModel,
    parse_nvidia_smi_memory,
    parse_ollama_ps,
    run_command,
)
from geekbot.log import setup_logging
from geekbot.swapcheck import format_swap_report, run_swap_cycles

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


class ModeChoice(StrEnum):
    """The values of ``--mode``; Typer shows them as the allowed choices."""

    CHAT = "chat"
    CSV = "csv"


def _read_gpu() -> GpuMemory:
    return parse_nvidia_smi_memory(run_command(NVIDIA_SMI_MEMORY_COMMAND))


def _read_ps() -> list[LoadedModel]:
    return parse_ollama_ps(run_command(OLLAMA_PS_COMMAND))


@app.command()
def doctor(
    warmup: Annotated[
        bool,
        typer.Option(
            "--warmup/--no-warmup",
            help="Load the mode's model onto the GPU and check it (uses the GPU for a moment).",
        ),
    ] = True,
    mode: Annotated[
        ModeChoice,
        typer.Option(
            "--mode",
            help="Whose model to check: chat (default) or csv. The warm-up is a swap: the other "
            "mode's model is unloaded first (R-4).",
        ),
    ] = ModeChoice.CHAT,
    repair_stores: Annotated[
        bool,
        typer.Option(
            "--repair",
            help="Instead of the checks: bring SQLite and the vector index back in step (finish "
            "or fail interrupted ingests, rebuild missing vectors, remove orphan vectors). Uses "
            "the CPU embedder, only for text the cache does not have.",
        ),
    ] = False,
) -> None:
    """Check Ollama, the models and the GPU. Exits 1 if a plan rule is broken."""
    settings = Settings()
    setup_logging(settings.log_level)
    if repair_stores:
        _repair_stores(settings)
        return
    chosen: Mode = "csv" if mode is ModeChoice.CSV else "chat"
    with OllamaClient(settings.ollama_host) as client:
        checks = run_doctor(
            settings, client, warm_up=warmup, read_gpu=_read_gpu, read_ps=_read_ps, mode=chosen
        )
    typer.echo(format_checks(checks))
    if has_failure(checks):
        raise typer.Exit(1)


def _repair_stores(settings: Settings) -> None:
    """``doctor --repair``: report, repair, report again; exit 1 if anything is still wrong."""
    from geekbot.index.consistency import check_consistency, repair
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.workspace import Workspace

    try:
        with Workspace.open(settings) as workspace:
            typer.echo(f"before: {check_consistency(workspace.store, workspace.dense).summary()}")
            typer.echo(
                f"repair: {repair(workspace.store, workspace.dense, workspace.embedder).summary()}"
            )
            after = check_consistency(workspace.store, workspace.dense)
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except OllamaError as error:
        _fail(str(error))
    typer.echo(f"after: {after.summary()}")
    if not after.ok:
        raise typer.Exit(1)


@app.command()
def ingest(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Files or folders to index. A folder is walked recursively; there is no default "
            "folder, so nothing is indexed by accident (v1 finding #13).",
            exists=True,
            readable=True,
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Processes that parse files in parallel."),
    ] = 1,
    reindex: Annotated[
        bool,
        typer.Option(
            "--reindex",
            help="Throw the stored vectors away and rebuild them from the stored chunks (needed "
            "after the embedder, its prefixes or the header setting changed). Then ingest the "
            "paths, if any.",
        ),
    ] = False,
) -> None:
    """Index documents: parse, chunk, embed on the CPU, store. Needs Ollama for the embedder.

    Files that did not change since the last run are skipped without being read again, and text
    that was embedded before is taken from the cache, so a second run over the same folder
    computes 0 embeddings. One bad file never stops the run: it is reported, and the exit code is
    1 if any file failed.
    """
    # Imported here: chromadb and the loaders are slow to import, and `geekbot --version` or
    # `geekbot doctor` should not wait for them.
    from geekbot.index.consistency import repair
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.workspace import Workspace
    from geekbot.ingest.pipeline import Ingestor, collect_files, format_outcome

    settings = Settings()
    setup_logging(settings.log_level)
    if not paths and not reindex:
        _fail("give at least one file or folder to ingest (or --reindex)")
    try:
        files, unsupported = collect_files(paths or [])
    except FileNotFoundError as error:
        _fail(str(error))
    if unsupported:
        typer.echo(f"{len(unsupported)} files of unsupported types were left out.", err=True)
    try:
        with Workspace.open(settings, reset_dense=reindex) as workspace:
            if reindex:
                typer.echo("re-indexing: rebuilding the vectors from the stored chunks", err=True)
                typer.echo(
                    repair(workspace.store, workspace.dense, workspace.embedder).summary(), err=True
                )
            report = Ingestor(
                workspace.store, workspace.dense, workspace.embedder, workers=workers
            ).ingest(files, on_file=lambda outcome: typer.echo(format_outcome(outcome)))
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except OllamaError as error:
        _fail(str(error))
    typer.echo(report.summary())
    if report.has_failures:
        raise typer.Exit(1)


class SearchMode(StrEnum):
    """The values of ``--mode`` for ``search``. ``hybrid_rerank`` arrives in S2.3."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for, in your own words.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="How many chunks to show.")] = 5,
    mode: Annotated[
        SearchMode,
        typer.Option(
            "--mode",
            help="dense (by meaning), lexical (BM25, by words; needs no Ollama) or hybrid (both, "
            "fused by rank with RRF). hybrid_rerank arrives in S2.3.",
        ),
    ] = SearchMode.DENSE,
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="hybrid only: how many chunks each side contributes before fusing.",
        ),
    ] = 50,
) -> None:
    """Search the knowledge base and print the best chunks with the time each stage took.

    Only ready documents are searched. The dense and hybrid modes need Ollama for the embedder
    (CPU). Every mode takes the storage lock, so none can run while another Geekbot process owns
    the data folder.
    """
    if mode is SearchMode.HYBRID_RERANK:
        _not_built("search --mode hybrid_rerank", "S2.3")
    # Imported here: chromadb is slow to import and `geekbot --version` should not wait for it.
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import Store
    from geekbot.index.workspace import Workspace
    from geekbot.retrieval import RetrievalResult, format_result
    from geekbot.retrieval.dense import DenseRetriever
    from geekbot.retrieval.hybrid import HybridRetriever
    from geekbot.retrieval.lexical import LexicalRetriever

    settings = Settings()
    setup_logging(settings.log_level)
    nothing = "nothing is indexed yet; run `geekbot ingest <folder>` first"
    result: RetrievalResult
    try:
        if mode is SearchMode.LEXICAL:
            # BM25 needs only SQLite: no embedder, no Ollama, no vector index.
            with Store(settings.data_dir) as store:
                if not store.list_documents("ready"):
                    _fail(nothing)
                result = LexicalRetriever(store).search(query, top_k)
        else:
            with Workspace.open(settings) as workspace:
                if workspace.dense.count() == 0:
                    _fail(nothing)
                dense = DenseRetriever(workspace.store, workspace.dense, workspace.embedder)
                if mode is SearchMode.DENSE:
                    result = dense.search(query, top_k)
                else:
                    hybrid = HybridRetriever(
                        dense, LexicalRetriever(workspace.store), candidates=candidates
                    )
                    result = hybrid.search(query, top_k)
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except OllamaError as error:
        _fail(str(error))
    typer.echo(format_result(result))


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


@csv_app.command("swap")
def csv_swap(
    cycles: Annotated[
        int, typer.Option("--cycles", min=1, help="How many chat->csv->chat round trips to time.")
    ] = 3,
) -> None:
    """Time the swap between the chat model and the CSV model, in both directions (uses the GPU).

    Loads and unloads both models on purpose, so run it when nothing else needs the GPU. After
    each swap it checks that the new model is 100% on the GPU and the old one is gone (R-1, R-4).
    Exit code 1 if one of those checks fails.
    """
    settings = Settings()
    setup_logging(settings.log_level)
    wanted = [settings.resolved_chat_model, settings.resolved_csv_model]
    try:
        with OllamaClient(settings.ollama_host) as client:
            available = client.list_models()
            missing = [m for m in wanted if not model_is_pulled(m, available)]
            if missing:
                _fail("not pulled: " + ", ".join(f"`ollama pull {m}`" for m in missing))
            records = run_swap_cycles(
                settings, client, cycles=cycles, read_gpu=_read_gpu, read_ps=_read_ps
            )
    except OllamaError as error:
        _fail(str(error))
    typer.echo(format_swap_report(settings, records))
    if not all(record.ok for record in records):
        raise typer.Exit(1)


class SuiteChoice(StrEnum):
    """The values of ``--suite``."""

    BOTH = "both"
    HOSPITAL = "hospital"
    SALES = "sales"


@csv_app.command("repro")
def csv_repro(
    source: Annotated[
        Path,
        typer.Option(
            "--source", help="The Gigalogy folder. It is only read, never written or copied."
        ),
    ] = DEFAULT_SOURCE,
    suite: Annotated[SuiteChoice, typer.Option("--suite", help="Which dataset(s).")] = (
        SuiteChoice.BOTH
    ),
    runs: Annotated[
        int, typer.Option("--runs", min=1, help="Repeats of every question, with new seeds.")
    ] = DEFAULT_RUNS,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Only the first N questions per suite (smoke test)."),
    ] = None,
    time_limit_s: Annotated[
        int,
        typer.Option(
            "--time-limit-s",
            min=1,
            help="Per-question time cap for the run. Default: the package's own 600 s, so the "
            "run measures the copy, not Geekbot's cap; the report counts the questions that "
            "took longer than Geekbot's.",
        ),
    ] = DEFAULT_TIME_LIMIT_S,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to save the JSON result (default: evals/csv/results/)."),
    ] = None,
) -> None:
    """Re-run Gigalogy's 40-question CSV benchmark through Geekbot's adapter (uses the GPU).

    A full run (both datasets, 3 runs) takes about half an hour on the 7b model, with the GPU to
    itself. Only a run on the target model and an idle GPU is flagged official; any other run is
    a plumbing check (D-043). The result file holds ids and numbers, not the questions.
    """
    # Imported here for the same reason as in `csv describe`: pandas is slow to import.
    from geekbot.csv_mode.adapter import find_problem
    from geekbot.evals.csv_repro import (
        ReproError,
        default_output_path,
        format_report,
        load_suite,
        run_repro,
    )

    settings = Settings()
    setup_logging(settings.log_level)
    problem = find_problem(settings)
    if problem is not None:
        _fail(problem)
    names = SUITE_NAMES if suite is SuiteChoice.BOTH else (suite.value,)
    try:
        suites = [load_suite(source, name, limit=limit) for name in names]
    except ReproError as error:
        _fail(str(error))
    out_path = out or default_output_path(settings)
    typer.echo(
        f"profile={settings.profile} csv model={settings.resolved_csv_model}; "
        f"result file: {out_path}",
        err=True,
    )
    if settings.profile != "target":
        typer.echo(
            "note: this is not the target profile, so the run is a plumbing check only "
            "(set GEEKBOT_PROFILE=target for numbers you want to report).",
            err=True,
        )
    try:
        with OllamaClient(settings.ollama_host) as client:
            result = run_repro(
                settings,
                suites,
                runs=runs,
                limit=limit,
                time_limit_s=time_limit_s,
                source=source,
                client=client,
                read_gpu=_read_gpu,
                read_ps=_read_ps,
                report=typer.echo,
                out=out_path,
            )
    except OllamaError as error:
        _fail(str(error))
    typer.echo("")
    typer.echo(format_report(result))


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point named in pyproject.toml's ``[project.scripts]``.

    Typer exits the process itself (``SystemExit``), so this returns nothing. ``argv=None``
    reads ``sys.argv``; tests pass their own list.
    """
    app(args=None if argv is None else list(argv), prog_name="geekbot")
