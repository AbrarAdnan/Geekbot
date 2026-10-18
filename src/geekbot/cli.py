"""The ``geekbot`` command, built with Typer (lesson: project/learn/02-config-cli.md §6).

Typer turns typed Python functions into CLI commands. Commands whose step hasn't been built
yet are stubs: they say which step builds them and exit with code 2, so a script or test can't
mistake "not built" for success.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from geekbot import __version__
from geekbot.config import Mode, Settings, mask_settings
from geekbot.doctor import format_checks, has_failure, run_doctor
from geekbot.evals import (
    DEFAULT_CORPUS,
    DEFAULT_GOLDEN_DRAFT,
    DEFAULT_GOLDEN_JSONL,
    DEFAULT_GOLDEN_MARKDOWN,
    DEFAULT_RUNS,
    DEFAULT_SOURCE,
    DEFAULT_TIME_LIMIT_S,
    SUITE_NAMES,
)
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
from geekbot.retrieval import SearchMode
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
eval_app = typer.Typer(help="Evaluation data: golden set, BEIR sets.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(csv_app, name="csv")
app.add_typer(eval_app, name="eval")

THREAD_SHOW_WIDTH = 160  # how much of a message `geekbot thread show` prints on its line

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


def _probe_embedder(settings: Settings) -> None:
    """Embed one short text on the CPU, so `doctor` can see where the embedder ends up."""
    from geekbot.index.embedder import OllamaEmbeddingsClient

    with OllamaEmbeddingsClient(
        settings.ollama_host, settings.embed_model, num_gpu=0, keep_alive=settings.keep_alive
    ) as embedder:
        embedder.embed_query("doctor probe")


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
            settings,
            client,
            warm_up=warmup,
            read_gpu=_read_gpu,
            read_ps=_read_ps,
            mode=chosen,
            probe_embedder=lambda: _probe_embedder(settings),
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


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for, in your own words.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="How many chunks to show.")] = 5,
    mode: Annotated[
        SearchMode,
        typer.Option(
            "--mode",
            help="dense (by meaning), lexical (BM25, by words; needs no Ollama), hybrid (both, "
            "fused by rank with RRF) or hybrid_rerank (hybrid, then a cross-encoder re-scores "
            "the best candidates on the CPU; the first use downloads its model).",
        ),
    ] = SearchMode.DENSE,
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="hybrid and hybrid_rerank: how many chunks each side contributes before fusing.",
        ),
    ] = 50,
) -> None:
    """Search the knowledge base and print the best chunks with the time each stage took.

    Only ready documents are searched. The dense, hybrid and hybrid_rerank modes need Ollama for
    the embedder (CPU). Every mode takes the storage lock, so none can run while another Geekbot
    process owns the data folder.
    """
    # Imported here: chromadb is slow to import and `geekbot --version` should not wait for it.
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME, Store
    from geekbot.index.workspace import Workspace
    from geekbot.retrieval import RetrievalResult, format_result
    from geekbot.retrieval.factory import build_retriever
    from geekbot.retrieval.lexical import LexicalRetriever
    from geekbot.retrieval.rerank import RerankError

    settings = Settings()
    setup_logging(settings.log_level)
    nothing = "nothing is indexed yet; run `geekbot ingest <folder>` first"
    # Opening the store would create an empty database, so a mistyped GEEKBOT_DATA_DIR would
    # silently become a new, empty knowledge base. A search never creates one.
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _fail(nothing)
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
                retriever = build_retriever(mode, workspace, settings, candidates=candidates)
                result = retriever.search(query, top_k)
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except (OllamaError, RerankError) as error:
        _fail(str(error))
    typer.echo(format_result(result))


@app.command("rerank-timing")
def rerank_timing(
    query: Annotated[
        list[str],
        typer.Option("--query", help="A question to time with (repeat the option for several)."),
    ],
    depths: Annotated[
        str,
        typer.Option("--depths", help="Comma-separated depths to compare, e.g. 10,20,30,50."),
    ] = "10,20,30,50",
    repeats: Annotated[
        int, typer.Option("--repeats", min=1, help="Timed calls per query and depth.")
    ] = 5,
) -> None:
    """Time the reranker at several depths and print p50 / p95 (uses the CPU, no Ollama).

    The passages are the first stored chunks of ready documents, so there must be at least as many
    chunks as the largest depth. The first call loads the model (and downloads it the first time);
    it is not counted. Run it when the CPU is otherwise idle: the numbers are wall-clock times on
    this machine now (PLAN.md section 6.5 gives retrieval plus reranking 400 ms at p95).
    """
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME, Store
    from geekbot.retrieval.rerank import FastembedScorer, RerankError
    from geekbot.retrieval.rerank_timing import format_depth_timings, measure_rerank_depths

    settings = Settings()
    setup_logging(settings.log_level)
    try:
        chosen = [int(part) for part in depths.split(",") if part.strip()]
    except ValueError:
        _fail(f"--depths must be comma-separated whole numbers, got {depths!r}")
    if not chosen:
        _fail("--depths needs at least one number")
    needed = max(chosen)
    # Opening the store would create an empty database (see `search`); a timing run never does.
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _fail("nothing is indexed yet; run `geekbot ingest <folder>` first")
    try:
        with Store(settings.data_dir) as store:
            passages: list[str] = []
            for document in store.list_documents("ready"):
                passages.extend(chunk.text for chunk in store.document_chunks(document.doc_id))
                if len(passages) >= needed:
                    break
    except StorageLockedError as error:
        _fail(str(error))
    if len(passages) < needed:
        _fail(
            f"only {len(passages)} chunks are stored but the largest depth is {needed}; "
            "ingest more documents or lower --depths"
        )
    scorer = FastembedScorer(
        settings.rerank_model,
        cache_dir=settings.data_dir / "models",
        threads=settings.rerank_threads,
    )
    try:
        timings = measure_rerank_depths(scorer, query, passages, chosen, repeats=repeats)
    except (RerankError, ValueError) as error:
        _fail(str(error))
    typer.echo(format_depth_timings(settings.rerank_model, timings))


# --- eval: the golden set and the public benchmarks (step S2.4) ------------------------------

CorpusOption = Annotated[
    Path,
    typer.Option(
        "--corpus",
        help="The folder the corpus was ingested from. Document ids come from file paths, so "
        "it must be the same folder (and spelling) that was given to `geekbot ingest`.",
    ),
]


def _echo_problems(problems: Sequence[object]) -> None:
    for problem in problems:
        typer.echo(str(problem), err=True)


@eval_app.command("import")
def eval_import(
    source: Annotated[
        Path, typer.Option("--source", help="The filled-in question template.")
    ] = DEFAULT_GOLDEN_MARKDOWN,
    out: Annotated[
        Path, typer.Option("--out", help="The JSON Lines file to write.")
    ] = DEFAULT_GOLDEN_JSONL,
    corpus: CorpusOption = DEFAULT_CORPUS,
) -> None:
    """Turn the question template into golden.jsonl. Nothing is written while it has errors.

    Edit the markdown, never the JSONL: the JSONL is overwritten every time. Uses no model.
    """
    from geekbot.evals.datasets import (
        assign_splits,
        describe_set,
        list_corpus_files,
        validate_golden,
        write_golden,
    )
    from geekbot.evals.golden_md import import_markdown

    if not source.is_file():
        _fail(f"{source} does not exist")
    result = import_markdown(source.read_text(encoding="utf-8"))
    corpus_files = list_corpus_files(corpus) if corpus.is_dir() else None
    if corpus_files is None:
        typer.echo(f"note: {corpus} is not a folder; source file names are not checked", err=True)
    report = validate_golden(result.items, corpus_files=corpus_files)
    problems = [*result.problems, *report.problems]
    typer.echo(
        f"read {len(result.items)} items from {source} ({len(result.blank)} template blocks left "
        "blank)"
    )
    _echo_problems(problems)
    errors = sum(1 for problem in problems if problem.severity == "error")
    if errors:
        typer.echo(f"{errors} errors: {out} was not written", err=True)
        raise typer.Exit(1)
    write_golden(out, result.items)
    typer.echo(describe_set(result.items, assign_splits(result.items)))
    typer.echo(f"wrote {out}")


@eval_app.command("validate")
def eval_validate(
    golden: Annotated[
        Path, typer.Option("--golden", help="The JSON Lines file to check.")
    ] = DEFAULT_GOLDEN_JSONL,
    corpus: CorpusOption = DEFAULT_CORPUS,
    check_index: Annotated[
        bool,
        typer.Option(
            "--check-index",
            help="Also check that every source is a ready document in the knowledge base. "
            "Needs `geekbot ingest` to have run on the corpus folder; takes the storage lock.",
        ),
    ] = False,
) -> None:
    """Check golden.jsonl and show how it splits into development and test items. No model."""
    from geekbot.evals.datasets import (
        GoldenFormatError,
        assign_splits,
        describe_set,
        list_corpus_files,
        load_golden,
        unindexed_sources,
        validate_golden,
    )
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME, Store

    if not golden.is_file():
        _fail(f"{golden} does not exist; run `geekbot eval import` first")
    try:
        items = load_golden(golden)
    except GoldenFormatError as error:
        _fail(f"{golden}: {error}")
    corpus_files = list_corpus_files(corpus) if corpus.is_dir() else None
    report = validate_golden(items, corpus_files=corpus_files)
    _echo_problems(report.problems)
    failed = not report.ok
    if report.ok:
        typer.echo(describe_set(items, assign_splits(items)))
    if check_index and report.ok:
        settings = Settings()
        setup_logging(settings.log_level)
        if not (settings.data_dir / DB_FILE_NAME).is_file():
            _fail("nothing is indexed yet; run `geekbot ingest <folder>` first")
        try:
            with Store(settings.data_dir) as store:
                ready = {document.doc_id for document in store.list_documents("ready")}
        except StorageLockedError as error:
            _fail(str(error))
        missing = unindexed_sources(items, corpus, ready)
        for item_id, path in missing:
            typer.echo(f"error: {item_id}: {path} is not a ready document in the index", err=True)
        failed = failed or bool(missing)
    if failed:
        raise typer.Exit(1)
    typer.echo("ok")


@eval_app.command("generate")
def eval_generate(
    count: Annotated[
        int, typer.Option("--count", min=0, help="How many answerable questions to draft.")
    ] = 40,
    unanswerable: Annotated[
        int, typer.Option("--unanswerable", min=0, help="How many unanswerable ones to draft.")
    ] = 10,
    corpus: CorpusOption = DEFAULT_CORPUS,
    out: Annotated[
        Path, typer.Option("--out", help="The markdown file of drafts to curate from.")
    ] = DEFAULT_GOLDEN_DRAFT,
    seed: Annotated[int, typer.Option("--seed", help="Which passages, and the model's seed.")] = 0,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite the draft file if it exists.")
    ] = False,
) -> None:
    """Draft questions from stored passages with the chat model (uses the GPU).

    The output is a markdown file to copy from, never data: you keep, reword or drop every draft.
    Needs the corpus ingested (`geekbot ingest`), but not the embedder. Multi-hop and multi-turn
    questions are not drafted; write those yourself.
    """
    from geekbot.evals.draft import (
        DRAFT_SCHEMA,
        candidate_chunks,
        make_drafts,
        render_drafts,
        spread_picks,
    )
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME, Store
    from geekbot.llm.models import swap_to_mode

    settings = Settings()
    setup_logging(settings.log_level)
    if count + unanswerable == 0:
        _fail("nothing to draft: --count and --unanswerable are both 0")
    if out.exists() and not force:
        _fail(f"{out} exists; copy what you want from it, then move it away or use --force")
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _fail("nothing is indexed yet; run `geekbot ingest <folder>` first")
    try:
        with Store(settings.data_dir) as store:
            groups = candidate_chunks(store, corpus)
    except StorageLockedError as error:
        _fail(str(error))
    picks = spread_picks(groups, count + unanswerable, f"draft-{seed}")
    if not picks:
        _fail(f"no usable passages (300 to 3000 characters) in the ready documents under {corpus}")
    if len(picks) < count + unanswerable:
        typer.echo(f"note: only {len(picks)} usable passages, fewer than asked for", err=True)
    model = settings.resolved_chat_model
    try:
        with OllamaClient(settings.ollama_host) as client:
            swap = swap_to_mode(client, settings, "chat")
            typer.echo(f"{model} loaded in {swap.warm_up_s:.1f} s; drafting {len(picks)} questions")

            def generate(system: str, prompt: str) -> str:
                return client.generate(
                    model,
                    prompt,
                    system=system,
                    schema=DRAFT_SCHEMA,
                    num_ctx=settings.chat_num_ctx,
                    keep_alive=settings.keep_alive,
                    seed=seed,
                )

            answerable = make_drafts(picks[:count], "single_hop", generate, corpus)
            near_misses = picks[count : count + unanswerable]
            unanswered = make_drafts(near_misses, "unanswerable", generate, corpus)
    except OllamaError as error:
        _fail(str(error))
    drafts = [*answerable.drafts, *unanswered.drafts]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_drafts(drafts, model=model, profile=settings.profile), encoding="utf-8", newline="\n"
    )
    failed = answerable.failed + unanswered.failed
    typer.echo(f"wrote {len(drafts)} drafts to {out} ({failed} replies were unusable)")


@eval_app.command("fetch")
def eval_fetch(
    datasets: Annotated[
        list[str] | None,
        typer.Argument(help="scifact and/or nfcorpus. Default: both."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Download again even if already unpacked.")
    ] = False,
    destination: Annotated[
        Path | None,
        typer.Option("--dest", help="Where to keep them. Default: <data dir>/beir."),
    ] = None,
) -> None:
    """Download the BEIR datasets once and print what each split holds (network, no model).

    The first download of a dataset records its SHA-256 beside the data (trust on first use);
    every later download must match it.
    """
    import httpx

    from geekbot.evals.beir import (
        DATASET_NAMES,
        HEADLINE_SPLIT,
        TUNING_SPLIT,
        BeirError,
        fetch_beir,
        load_beir,
    )

    settings = Settings()
    setup_logging(settings.log_level)
    names = datasets or list(DATASET_NAMES)
    unknown = [name for name in names if name not in DATASET_NAMES]
    if unknown:
        _fail(f"unknown dataset(s) {', '.join(unknown)}; choose from {', '.join(DATASET_NAMES)}")
    root = destination or settings.data_dir / "beir"
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            for name in names:
                fetched = fetch_beir(name, root, client=client, force=force)
                if not fetched.downloaded:
                    typer.echo(f"{name}: already unpacked in {fetched.folder}")
                else:
                    note = " (first download: recorded as the reference)"
                    typer.echo(
                        f"{name}: downloaded {fetched.size_bytes / 1e6:.1f} MB, "
                        f"sha256 {fetched.sha256}{note if fetched.new_checksum else ''}"
                    )
                for split in (TUNING_SPLIT[name], HEADLINE_SPLIT):
                    typer.echo(f"  {load_beir(fetched.folder, split).summary()}")
    except BeirError as error:
        _fail(str(error))


DEFAULT_ABLATION_CONFIG = Path("evals/configs/ablation.toml")


@eval_app.command("retrieval")
def eval_retrieval(
    config: Annotated[
        Path, typer.Option("--config", help="The run's TOML file (PLAN.md §9).")
    ] = DEFAULT_ABLATION_CONFIG,
    phase: Annotated[
        list[str] | None,
        typer.Option(
            "--phase",
            help="rewrite, ablation, tuning, bakeoff, abstain or all (repeatable). Default: all. "
            "`rewrite` loads the chat model on the GPU; check the GPU is free first.",
        ),
    ] = None,
    rung: Annotated[
        list[str] | None,
        typer.Option("--rung", help="Only these rungs (R0..R7, 'R7 - BM25', ...)."),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", help="Only these datasets (scifact, nfcorpus, golden)."),
    ] = None,
    fresh: Annotated[
        bool, typer.Option("--fresh", help="Ignore earlier results and measure everything again.")
    ] = False,
) -> None:
    """Run the retrieval evaluation: the ablation, sweeps, bake-offs and the abstain threshold.

    CPU-heavy and long. Results go to `<results_dir>/<name>/*.json` and are rewritten after each
    finished piece, so an interrupted run resumes where it stopped. Indexes are built in the scratch
    folder of the configuration; the knowledge base in the data folder is never opened.
    """
    from geekbot.evals.ablation import load_config
    from geekbot.evals.orchestrate import Phase, run_phases
    from geekbot.evals.phases import default_backends

    settings = Settings()
    setup_logging(settings.log_level)
    try:
        loaded = load_config(config)
        names = phase or ["all"]
        every = [p.value for p in Phase]
        unknown = [name for name in names if name != "all" and name not in every]
        if unknown:
            _fail(f"unknown phase(s) {', '.join(unknown)}; choose from {', '.join(every)}, all")
        chosen = list(Phase) if "all" in names else [Phase(name) for name in names]
        written = run_phases(
            loaded,
            settings,
            default_backends(settings),
            chosen,
            rungs=rung,
            datasets=dataset,
            resume=not fresh,
            log=typer.echo,
        )
    except (ValueError, OllamaError) as error:
        _fail(str(error))
    for path in written:
        typer.echo(f"wrote {path}")


@eval_app.command("report")
def eval_report(
    config: Annotated[
        Path, typer.Option("--config", help="The run's TOML file (it names the results folder).")
    ] = DEFAULT_ABLATION_CONFIG,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write. Default: <results folder>/report.md"),
    ] = None,
) -> None:
    """Write the Markdown report from the result files of `geekbot eval retrieval`. No model."""
    from geekbot.evals.ablation import load_config
    from geekbot.evals.report import load_results, render_report

    try:
        loaded = load_config(config)
        results = load_results(loaded.results_path)
        text = render_report(
            results, name=loaded.name, resamples=loaded.resamples, seed=loaded.seed
        )
    except ValueError as error:
        _fail(str(error))
    target = out or loaded.results_path / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(f"wrote {target}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in your own words.")],
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, help="How many chunks to find and offer the model.")
    ] = 5,
    mode: Annotated[
        SearchMode,
        typer.Option(
            "--mode",
            help="How to find the chunks, as in `search`. The default adds the cross-encoder "
            "(its first use downloads its model).",
        ),
    ] = SearchMode.HYBRID_RERANK,
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="hybrid and hybrid_rerank: how many chunks each side contributes before fusing.",
        ),
    ] = 50,
    show_prompt: Annotated[
        bool,
        typer.Option("--show-prompt", help="Also print the exact messages the model received."),
    ] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Print the answer as the model writes it. The readings then show the time to "
            "the first token. A citation the check removes cannot be taken back from the "
            "screen: the corrected answer is printed after the stream.",
        ),
    ] = False,
    think: Annotated[
        bool,
        typer.Option(
            "--think",
            help="Let the model reason before it answers: slower, and only for a model with a "
            "thinking mode. Off by default (PLAN.md section 6.5).",
        ),
    ] = False,
    show_thinking: Annotated[
        bool,
        typer.Option("--show-thinking", help="With --think: print the model's reasoning too."),
    ] = False,
    thread: Annotated[
        str | None,
        typer.Option(
            "--thread",
            help="Keep a conversation under this name: the next `ask --thread` with the same name "
            "remembers this question and answer. A follow-up such as 'and by month?' is rewritten "
            "into a full question first (one extra model call). Without it every question stands "
            "alone. See `geekbot thread show` and `thread delete`.",
        ),
    ] = None,
) -> None:
    """Answer a question from the knowledge base with the chat model (uses the GPU).

    The chat model is loaded explicitly first (the other mode's model is unloaded: they never share
    the GPU), then the question runs through a fixed graph: find the best chunks, then ask the model
    to answer from them. Citations that name no source are removed from the answer. Below the
    answer come the sources it cites, the sources the model read without citing them, and the
    readings of the run: where the time went, the time to the first token, and whether the model had
    to be loaded. With --thread the conversation is saved in the data folder and later questions can
    refer back to it. Only ready documents are searched. The storage lock is held for the whole
    command.
    """
    # Imported here: LangGraph, LangChain and Chroma are slow to import, and `geekbot --version`
    # or `geekbot doctor` should not wait for them.
    from geekbot.agent.condense import CondenseOutput
    from geekbot.agent.mode import ModeError, choose_mode
    from geekbot.agent.packing import QuestionError, check_question, pack_prompt
    from geekbot.agent.render import (
        StreamCollector,
        StreamedAnswer,
        format_after_stream,
        format_answer,
        format_prompt,
    )
    from geekbot.agent.runner import (
        THREADS_DB,
        ThreadNameError,
        check_thread_name,
        read_thread,
        run_turn,
    )
    from geekbot.agent.runtime import AgentContext
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME
    from geekbot.index.workspace import Workspace
    from geekbot.llm.chat import ChatError, make_chat_model, make_structured_model
    from geekbot.llm.models import swap_to_mode
    from geekbot.retrieval.factory import build_retriever
    from geekbot.retrieval.rerank import RerankError

    try:
        check_question(question)  # before any model is loaded: a mistake here should cost nothing
        if thread is not None:
            check_thread_name(thread)
    except (QuestionError, ThreadNameError) as error:
        _fail(str(error))
    if show_thinking and not think:
        _fail("--show-thinking needs --think: without it the model does not reason")
    settings = Settings()
    setup_logging(settings.log_level)
    nothing = "nothing is indexed yet; run `geekbot ingest <folder>` first"
    # Opening the store would create an empty database (see `search`); a question never does.
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _fail(nothing)
    if thread is not None:
        # A thread keeps the mode it started with: say so before any model is loaded (C4).
        saved = asyncio.run(read_thread(settings.data_dir / THREADS_DB, thread))
        try:
            choose_mode(saved.get("mode") if saved else None, "chat")
        except ModeError as error:
            _fail(str(error))
    model = settings.resolved_chat_model
    answered: StreamedAnswer
    try:
        with Workspace.open(settings) as workspace:
            if workspace.dense.count() == 0:
                _fail(nothing)
            retriever = build_retriever(mode, workspace, settings, candidates=candidates)
            with OllamaClient(settings.ollama_host) as client:
                swap = swap_to_mode(client, settings, "chat")
            released = f", unloaded {', '.join(swap.unloaded)}" if swap.unloaded else ""
            typer.echo(f"{model} ready ({swap.warm_up_s:.1f} s to load{released})", err=True)
            context = AgentContext(
                retriever=retriever,
                llm=make_chat_model(settings, think=think),
                model=model,
                top_k=top_k,
                context_budget_tokens=settings.context_budget_tokens,
                # Only a conversation has a follow-up to rewrite; a lone question has no history.
                condenser=make_structured_model(
                    settings, CondenseOutput, num_predict=settings.condense_num_predict
                )
                if thread is not None
                else None,
                history_budget_tokens=settings.history_budget_tokens,
                history_keep_messages=settings.history_keep_messages,
            )
            typer.echo(f"asking {model}{' with thinking on' if think else ''}...", err=True)
            collector = (
                StreamCollector(
                    write=lambda piece: typer.echo(piece, nl=False),
                    note=lambda piece: typer.echo(piece, nl=False, err=True),
                    show_thinking=show_thinking,
                )
                if stream
                else None
            )
            answered = asyncio.run(
                run_turn(
                    question,
                    context=context,
                    thread=thread,
                    threads_db=settings.data_dir / THREADS_DB,
                    collector=collector,
                    mode="chat",
                )
            )
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except (OllamaError, RerankError, ChatError, ModeError) as error:
        _fail(_with_thinking_hint(str(error), think=think))
    final = answered.final
    if stream:
        if answered.streamed.strip():
            typer.echo("\n")  # end the streamed line (it has no newline) and leave a blank line
        typer.echo(format_after_stream(answered, model=model, thread=thread))
    else:
        typer.echo(format_answer(final, model=model, show_thinking=show_thinking, thread=thread))
    if show_prompt:
        # `remember` has already added this turn's two messages; the prompt held the ones before.
        packed = pack_prompt(
            question,
            final.get("sources", []),
            budget_tokens=settings.context_budget_tokens,
            history=final.get("history", [])[:-2],
            history_budget_tokens=settings.history_budget_tokens,
        )
        typer.echo(f"\n{format_prompt(packed)}")


thread_app = typer.Typer(help="Saved conversations (see `ask --thread`).", no_args_is_help=True)
app.add_typer(thread_app, name="thread")


@thread_app.command("show")
def thread_show(
    name: Annotated[str, typer.Argument(help="The name given to `ask --thread`.")],
) -> None:
    """Print what a saved conversation remembers: the messages, their labels and the taint."""
    from geekbot.agent.runner import THREADS_DB, ThreadNameError, read_thread

    settings = Settings()
    try:
        saved = asyncio.run(read_thread(settings.data_dir / THREADS_DB, name))
    except ThreadNameError as error:
        _fail(str(error))
    if saved is None:
        _fail(f"no thread named {name!r} (it is created by `geekbot ask --thread {name} ...`)")
    history = saved.get("history", [])
    mode, csv_file = saved.get("mode"), saved.get("csv_file")
    kind = (f" ({mode}, {csv_file})" if csv_file else f" ({mode})") if mode else ""
    typer.echo(
        f"thread {name}{kind}: {saved.get('turn', 0)} turns, {len(history)} messages remembered"
    )
    for message in history:
        labels = ", ".join(message["labels"]) or "no labels"
        text = " ".join(message["content"].split())  # a CSV answer is a whole table: one line each
        shown = text if len(text) <= THREAD_SHOW_WIDTH else text[:THREAD_SHOW_WIDTH] + "..."
        typer.echo(f"  {message['id']} [{labels}] {shown}")
    taint = ", ".join(saved.get("thread_taint", [])) or "none"
    typer.echo(f"taint: {taint}")


@thread_app.command("delete")
def thread_delete(
    name: Annotated[str, typer.Argument(help="The name given to `ask --thread`.")],
) -> None:
    """Forget a saved conversation completely (its memory and its taint). It cannot be undone."""
    from geekbot.agent.runner import THREADS_DB, ThreadNameError, delete_thread

    settings = Settings()
    try:
        existed = asyncio.run(delete_thread(settings.data_dir / THREADS_DB, name))
    except ThreadNameError as error:
        _fail(str(error))
    if not existed:
        _fail(f"no thread named {name!r}")
    typer.echo(f"thread {name} deleted")


def _with_thinking_hint(message: str, *, think: bool) -> str:
    """Add what to do when ``--think`` met a model that cannot think (Ollama says so itself)."""
    if think and "support thinking" in message:
        return f"{message}\nDrop --think, or choose a chat model with a thinking mode."
    return message


@app.command("ask-timing")
def ask_timing(
    question: Annotated[
        list[str],
        typer.Option("--question", help="A question to time (repeat the option for several)."),
    ],
    cold_rounds: Annotated[
        int,
        typer.Option(
            "--cold-rounds",
            min=0,
            help="How many times to unload the chat model, load it again and time the first "
            "question (0 skips the cold rounds).",
        ),
    ] = 3,
    warm_runs: Annotated[
        int,
        typer.Option(
            "--warm-runs",
            min=1,
            help="How many times to ask every question once the model is warm.",
        ),
    ] = 5,
    top_k: Annotated[
        int, typer.Option("--top-k", min=1, help="How many chunks to find and offer the model.")
    ] = 5,
    mode: Annotated[
        SearchMode, typer.Option("--mode", help="How to find the chunks, as in `ask`.")
    ] = SearchMode.HYBRID_RERANK,
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="hybrid and hybrid_rerank: how many chunks each side contributes before fusing.",
        ),
    ] = 50,
    think: Annotated[
        bool, typer.Option("--think", help="Time the answers with thinking on.")
    ] = False,
) -> None:
    """Time answers properly: the first question after a fresh load, then steady state (GPU).

    Loads the chat model onto the GPU, so run it when nothing else uses the GPU. First one retrieval
    is made and timed alone (a new process loads the reranker on its first use; that cost is
    reported apart, not mixed into any run). Then each cold round unloads the chat model, loads it
    again and times the first question, which is how the server starts; then every question is asked
    in turn ``--warm-runs`` times with the model warm. The report gives p50, p95 and the maximum of
    the time to the first token, the whole answer, retrieval and tokens per second, how many runs
    had the model loaded again during a question (the plan wants none), and Ollama's prompt token
    count per run. The numbers are wall-clock readings for the named model, on this machine, now.
    """
    from geekbot.agent.graph import build_graph
    from geekbot.agent.latency import (
        LatencyError,
        LatencyRun,
        format_latency_report,
        measure_latency,
    )
    from geekbot.agent.packing import QuestionError, check_question
    from geekbot.agent.runtime import AgentContext
    from geekbot.domain import RunStats
    from geekbot.index.dense import FingerprintMismatchError
    from geekbot.index.lock import StorageLockedError
    from geekbot.index.sqlite import DB_FILE_NAME
    from geekbot.index.workspace import Workspace
    from geekbot.llm.chat import ChatError, make_chat_model
    from geekbot.llm.models import swap_to_mode, unload_and_wait
    from geekbot.retrieval.factory import build_retriever
    from geekbot.retrieval.rerank import RerankError

    try:
        for text in question:
            check_question(text)
    except QuestionError as error:
        _fail(str(error))
    settings = Settings()
    setup_logging(settings.log_level)
    nothing = "nothing is indexed yet; run `geekbot ingest <folder>` first"
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _fail(nothing)
    model = settings.resolved_chat_model
    try:
        with Workspace.open(settings) as workspace, OllamaClient(settings.ollama_host) as client:
            if workspace.dense.count() == 0:
                _fail(nothing)
            retriever = build_retriever(mode, workspace, settings, candidates=candidates)
            swap = swap_to_mode(client, settings, "chat")
            typer.echo(f"{model} ready ({swap.warm_up_s:.1f} s to load)", err=True)
            context = AgentContext(
                retriever=retriever,
                llm=make_chat_model(settings, think=think),
                model=model,
                top_k=top_k,
                context_budget_tokens=settings.context_budget_tokens,
            )
            graph = build_graph()
            typer.echo("one retrieval first, timed apart (it loads the reranker)...", err=True)
            began = time.perf_counter()
            retriever.search(question[0], top_k)
            first_retrieval_ms = (time.perf_counter() - began) * 1000

            def run_once(text: str) -> LatencyRun:
                typer.echo(f"  asking: {text[:60]}", err=True)
                final = graph.invoke({"question": text}, context=context)
                generation = final.get("generation")
                if not generation:
                    raise LatencyError(
                        f"no source was found for {text!r}, so the model was never called; "
                        "time a question the documents can answer"
                    )
                timings = final.get("retrieval_ms", {})
                return LatencyRun(
                    question=text,
                    stats=RunStats.from_mapping(generation),
                    retrieval_ms=timings.get("total"),
                    stages_ms={name: ms for name, ms in timings.items() if name != "total"},
                )

            def cold_start() -> None:
                typer.echo("unloading the chat model and loading it again...", err=True)
                unload_and_wait(client, [model])
                swap_to_mode(client, settings, "chat")

            report = measure_latency(
                run_once,
                cold_start,
                question,
                model=model,
                cold_rounds=cold_rounds,
                warm_runs=warm_runs,
                first_retrieval_ms=first_retrieval_ms,
            )
    except StorageLockedError as error:
        _fail(str(error))
    except FingerprintMismatchError as error:
        _fail(f"{error}\nRun `geekbot ingest --reindex` to rebuild the vectors.")
    except (OllamaError, RerankError, ChatError, LatencyError) as error:
        _fail(_with_thinking_hint(str(error), think=think))
    setup = (
        f"chat model {model} ({settings.profile} profile) · retrieval {mode.value}, top-k {top_k}, "
        f"thinking {'on' if think else 'off'} · {cold_rounds} cold rounds, {warm_runs} warm passes"
    )
    typer.echo(format_latency_report(report, setup=setup))


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
    thread: Annotated[
        str | None,
        typer.Option(
            "--thread",
            help="Keep a conversation about this file under this name (a CSV thread: its mode is "
            "fixed). A follow-up such as 'and by month?' is rewritten into a full question first, "
            "because the analysis package has no memory. Pass the same file each time. Without it "
            "every question stands alone.",
        ),
    ] = None,
) -> None:
    """Ask a question about a CSV file (uses the CSV model, so it needs a free GPU).

    The reply goes to stdout. A footer on stderr names the model, profile, status, time and the
    time spent loading the model, so a pasted result always says which model produced it
    (D-043). Exit code 0 means the tool worked (even if the data could not answer); 1 means the
    question failed or Ollama, the model or the file was not usable. With --thread the question
    runs through the answer graph in CSV mode (the CSV model is loaded explicitly first, the chat
    model is unloaded) and the conversation is saved like a chat thread (`geekbot thread show`).
    """
    from geekbot.csv_mode.adapter import CsvAnalyst, CsvLoadError, find_problem

    settings = Settings()
    setup_logging(settings.log_level)
    problem = find_problem(settings)
    if problem is not None:
        _fail(problem)
    if thread is not None:
        _csv_ask_in_thread(file, question, thread, settings)
        return
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


def _csv_ask_in_thread(file: Path, question: str, thread: str, settings: Settings) -> None:
    """One turn of a CSV conversation: the answer graph in CSV mode, saved under ``thread`` (C4)."""
    from geekbot.agent.condense import CondenseOutput
    from geekbot.agent.mode import ModeError, choose_mode
    from geekbot.agent.packing import QuestionError, check_question
    from geekbot.agent.render import format_answer
    from geekbot.agent.runner import (
        THREADS_DB,
        ThreadNameError,
        check_thread_name,
        read_thread,
        run_turn,
    )
    from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
    from geekbot.csv_mode.adapter import CsvAnalyst, CsvLoadError
    from geekbot.index.lock import StorageLock, StorageLockedError
    from geekbot.llm.chat import ChatError, make_structured_model
    from geekbot.llm.models import swap_to_mode

    model = settings.resolved_csv_model
    threads_db = settings.data_dir / THREADS_DB
    try:
        check_question(question)  # before any model is loaded: a mistake here should cost nothing
        check_thread_name(thread)
        saved = asyncio.run(read_thread(threads_db, thread))
        choose_mode(
            saved.get("mode") if saved else None, "csv"
        )  # a chat thread stays a chat thread
    except (QuestionError, ThreadNameError, ModeError) as error:
        _fail(str(error))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with StorageLock(settings.data_dir), CsvAnalyst.open(file, settings) as analyst:
            with OllamaClient(settings.ollama_host) as client:
                swap = swap_to_mode(client, settings, "csv")
            released = f", unloaded {', '.join(swap.unloaded)}" if swap.unloaded else ""
            typer.echo(f"{model} ready ({swap.warm_up_s:.1f} s to load{released})", err=True)
            context = AgentContext(
                retriever=UnavailableRetriever(),
                llm=UnavailableChat(),
                model=model,
                top_k=1,
                context_budget_tokens=settings.context_budget_tokens,
                # The rewrite uses the CSV model, the one on the GPU, with the context size it was
                # warmed up with (R-2): the package keeps no memory, so a follow-up needs it.
                condenser=make_structured_model(
                    settings,
                    CondenseOutput,
                    num_predict=settings.condense_num_predict,
                    model=model,
                    num_ctx=settings.csv_num_ctx,
                ),
                history_budget_tokens=settings.history_budget_tokens,
                history_keep_messages=settings.history_keep_messages,
                csv=analyst,
            )
            typer.echo(f"asking {model} about {file.name}...", err=True)
            answered = asyncio.run(
                run_turn(
                    question,
                    context=context,
                    thread=thread,
                    threads_db=threads_db,
                    mode="csv",
                    csv_file=file.name,
                )
            )
    except StorageLockedError as error:
        _fail(str(error))
    except CsvLoadError as error:
        _fail(str(error))
    except (OllamaError, ChatError, ModeError) as error:
        _fail(str(error))
    final = answered.final
    typer.echo(format_answer(final, model=model, thread=thread))
    if final.get("csv_status") == "failed":
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
