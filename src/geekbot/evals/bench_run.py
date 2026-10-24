"""Running the bench: servers, loads, questions, files (step S5.1, PLAN.md Â§6.5 and Â§9).

``bench.py`` holds the rules; this file does the work and writes what it saw. It is built in three
layers so that each can be tested at the right size:

1. **Orchestration with no machine in it** (``run_point``, ``run_sweep``, ``run_cold_rounds`` ...): they
   drive a ``ChatSession`` (load a model for a setting, say how much of it is on the GPU, ask a
   question) and a ``VramSampler``, so the tests give them fakes and check the order of the work,
   what is shared and what a failure does.
2. **The real session** (``GraphSession``): the answer graph of the product, the real retriever and the
   real chat model, on a private Ollama server (``ollama_instance``) so the KV-cache type can be a
   setting of the sweep without touching the user's own server.
3. **``run_bench``**: the whole run, in the order of PLAN.md Â§9: the guard, the baseline VRAM and
   ingest speed, the sweep, the extras at the shipped setting (cold rounds, follow-ups, thinking),
   the CSV rows, then the files.

The sweep never changes a default: it measures, ``choose_default`` says which setting the numbers
favour, and the report names the difference from the shipped one. Changing the product is the
user's decision (D-097, D-082), made from the report.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Protocol, Self

from geekbot.config import Settings
from geekbot.evals.bench import (
    BenchConfig,
    ChatRun,
    ConfigResult,
    GpuReading,
    GuardVerdict,
    SweepPoint,
    check_gpu_idle,
    choose_default,
    expand_sweep,
    format_bench_report,
    must_include_recall,
    parse_gpu_reading,
    pick_questions,
    summarize_runs,
    wait_until_gpu_idle,
)
from geekbot.evals.datasets import GoldenItem, load_golden
from geekbot.llm.models import OllamaClient, OllamaError, same_model

__all__ = [
    "ChatSession",
    "GraphSession",
    "VramSampler",
    "run_bench",
    "run_cold_rounds",
    "run_point",
    "run_sweep",
    "select_items",
]

Log = Callable[[str], None]


# --- sampling the video memory while questions run -------------------------------------------


class VramSampler:
    """Reads the GPU memory in use every ``interval_s`` on a thread and keeps the highest reading.

    A model's peak is not its resting size: loading and the first long prompt allocate buffers that
    are freed again, and the card must have room for the peak, not for the average. The sampler
    reads once on entry and once on exit as well, so even a very short block has two readings.
    A reader that returns ``None`` (``nvidia-smi`` failed) adds nothing; ``peak_mib`` is ``None``
    when no reading ever came.
    """

    def __init__(self, read_used_mib: Callable[[], int | None], *, interval_s: float = 0.4) -> None:
        self._read = read_used_mib
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._peak: int | None = None

    @property
    def peak_mib(self) -> int | None:
        with self._lock:
            return self._peak

    def _sample(self) -> None:
        value = self._read()
        if value is None:
            return
        with self._lock:
            if self._peak is None or value > self._peak:
                self._peak = value

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> Self:
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample()


# --- the session the sweep drives ------------------------------------------------------------


class ChatSession(Protocol):
    """What the sweep needs from "the chat model on a server" (the tests give it a fake)."""

    def load(self, point: SweepPoint) -> float:
        """Warm the chat model up with the point's context size and batch size; seconds taken."""
        ...

    def unload(self) -> None:
        """Unload the chat model and wait until the server no longer lists it."""
        ...

    def residency(self) -> tuple[int | None, int | None]:
        """``(share of the model on the GPU in percent, context it was loaded with)``."""
        ...

    def ask(self, item: GoldenItem, point: SweepPoint, *, think: bool = False) -> ChatRun:
        """One question through the answer graph on a fresh thread."""
        ...

    def ask_followup(self, item: GoldenItem, point: SweepPoint) -> ChatRun:
        """A multi-turn item: its earlier turns, then the follow-up; the follow-up is measured."""
        ...


SessionOpener = Callable[[str], AbstractContextManager[ChatSession]]
SamplerFactory = Callable[[], AbstractContextManager[VramSampler]]


def select_items(
    items: Sequence[GoldenItem], config: BenchConfig
) -> tuple[list[GoldenItem], list[GoldenItem]]:
    """The questions of the sweep and the follow-ups of the extras, chosen the same way every run.

    The sweep asks answerable single-hop and multi-hop items (they have must-include phrases, which
    is what quality is measured on); unanswerable ones carry no phrases and multi-turn ones need a
    conversation (they are the follow-up rows).
    """
    answerable = [item for item in items if item.type in ("single_hop", "multi_hop")]
    follow_ups = [item for item in items if item.type == "multi_turn"]
    return (
        pick_questions(answerable, config.chat.questions),
        pick_questions(follow_ups, config.chat.followups),
    )


def run_point(
    session: ChatSession,
    point: SweepPoint,
    items: Sequence[GoldenItem],
    *,
    passes: int,
    load: bool,
    sampler: SamplerFactory,
    read_used_mib: Callable[[], int | None],
    cpu_idle: Callable[[], bool],
    log: Log,
) -> ConfigResult:
    """Measure one sweep point: (load if needed,) read residency, ask every question ``passes`` times.

    One question is asked first and thrown away: the first call after a load pays for things that
    are not the setting's fault (the cross-encoder's first use, prompt-cache warm-up; S3.1 saw 5.9 s
    once). The VRAM sampler runs across the load and the questions so the peak includes the load.
    A failure while loading or asking (Ollama refusing the context, out of memory) becomes a result
    with a note and no runs, so one bad corner of the grid does not end a long sweep.
    """
    idle_cpu = cpu_idle()
    idle_mib = read_used_mib()
    load_s = 0.0
    runs: list[ChatRun] = []
    gpu: int | None = None
    context: int | None = None
    notes: list[str] = []
    with sampler() as sample:
        try:
            if load:
                load_s = session.load(point)
            gpu, context = session.residency()
            if items:
                session.ask(items[0], point)  # discarded, see the docstring
            for _ in range(passes):
                for item in items:
                    runs.append(session.ask(item, point))
        except Exception as error:  # noqa: BLE001 - a failed corner of the grid is a result, not the end
            notes.append(f"failed: {type(error).__name__}: {error}")
            log(f"  {point.key}: {notes[-1]}")
    summary = summarize_runs(runs)
    result = ConfigResult(
        point=point,
        load_s=load_s,
        gpu_percent=gpu,
        loaded_context=context,
        vram_idle_mib=idle_mib,
        vram_peak_mib=sample.peak_mib,
        summary=summary,
        cpu_idle=idle_cpu,
        notes=tuple(notes),
    )
    first = summary.to_first_token
    log(
        f"  {point.key}: GPU {gpu}%, peak {result.vram_peak_mib} MiB, "
        f"question to first token p50 {first.p50 if first else float('nan'):.2f} s, "
        f"quality {summary.quality if summary.quality is not None else float('nan'):.3f}, "
        f"loads {summary.loads}, n={summary.n}"
    )
    return result


def run_sweep(
    points: Sequence[SweepPoint],
    open_session: SessionOpener,
    items: Sequence[GoldenItem],
    *,
    passes: int,
    sampler: SamplerFactory,
    read_used_mib: Callable[[], int | None],
    cpu_idle: Callable[[], bool],
    log: Log,
) -> list[ConfigResult]:
    """Every point, with one server per KV-cache type and one model load per (context, batch) pair.

    ``points`` must come from ``expand_sweep`` (or be ordered like it): a server is opened for each
    run of equal ``kv_cache_type`` and the model is loaded when ``(context, batch)`` changes. The
    server is closed when its group is done or fails (the context manager does it), so no private
    Ollama is left holding the GPU. A server that cannot start gives every point of its group a
    result with the reason.
    """
    results: list[ConfigResult] = []
    for kv, group_iter in groupby(points, key=lambda p: p.kv_cache_type):
        group = list(group_iter)
        log(f"server with KV cache {kv}: {len(group)} points")
        try:
            with open_session(kv) as session:
                previous: tuple[str, int, int] | None = None
                load_failure: str | None = None
                for point in group:
                    needs_load = point.load_key != previous
                    if needs_load:
                        load_failure = None
                    if load_failure is not None:
                        # The load these points share failed: asking now would make Ollama load the
                        # model with whatever options it likes, and the numbers would be another setting's.
                        skipped = f"skipped: shares a load that failed ({load_failure})"
                        results.append(
                            ConfigResult(
                                point,
                                0.0,
                                None,
                                None,
                                None,
                                None,
                                summarize_runs([]),
                                True,
                                (skipped,),
                            )
                        )
                        previous = point.load_key
                        continue
                    result = run_point(
                        session,
                        point,
                        items,
                        passes=passes,
                        load=needs_load,
                        sampler=sampler,
                        read_used_mib=read_used_mib,
                        cpu_idle=cpu_idle,
                        log=log,
                    )
                    results.append(result)
                    if needs_load and result.notes and result.gpu_percent is None:
                        load_failure = result.notes[0]  # the load (or the residency read) failed
                    previous = point.load_key
        except Exception as error:  # noqa: BLE001 - the server itself failed (start, workspace, stop)
            reason = f"failed: server for KV cache {kv}: {type(error).__name__}: {error}"
            log(f"  {reason}")
            done = {r.point for r in results}
            for point in group:
                if point not in done:
                    results.append(
                        ConfigResult(
                            point, 0.0, None, None, None, None, summarize_runs([]), True, (reason,)
                        )
                    )
    return results


def run_cold_rounds(
    session: ChatSession, point: SweepPoint, item: GoldenItem, *, rounds: int
) -> list[ChatRun]:
    """Unload the chat model, load it again, ask one question: ``rounds`` times (the server's start).

    This is "the first question after a fresh load", the situation S3.1 once saw take 5.9 s. A
    question that follows a load the user did not wait for is the one that matters, so every round
    records it alone.
    """
    runs: list[ChatRun] = []
    for _ in range(rounds):
        session.unload()
        session.load(point)
        runs.append(session.ask(item, point))
    return runs


# --- the real session ------------------------------------------------------------------------


class GraphSession:
    """The product's answer graph over one Ollama server (the real thing; live-tested only)."""

    def __init__(
        self,
        settings: Settings,
        client: OllamaClient,
        retriever: Any,
        follow_up_graph: Any,
        graph: Any,
        condenser: Any,
        scorer: Any,
    ) -> None:
        self._settings = settings
        self._client = client
        self._retriever = retriever
        self._graph = graph
        self._follow_up_graph = follow_up_graph
        self._condenser = condenser
        self._scorer = scorer
        self._threads = 0

    def settings_for(self, point: SweepPoint) -> Settings:
        """The settings with this point's load-time options (the warm-up and every query agree, R-2)."""
        return self._settings.model_copy(
            update={"chat_num_ctx": point.num_ctx, "chat_num_batch": point.num_batch}
        )

    def load(self, point: SweepPoint) -> float:
        from geekbot.llm.models import swap_to_mode

        return swap_to_mode(self._client, self.settings_for(point), "chat").warm_up_s

    def unload(self) -> None:
        from geekbot.llm.models import unload_and_wait

        unload_and_wait(self._client, [self._settings.resolved_chat_model])

    def residency(self) -> tuple[int | None, int | None]:
        model = self._settings.resolved_chat_model
        for loaded in self._client.running():
            if same_model(loaded.name, model):
                return loaded.gpu_percent, loaded.context
        return None, None

    def _context(self, point: SweepPoint, *, think: bool, with_history: bool) -> Any:
        from geekbot.agent.runtime import AgentContext
        from geekbot.llm.chat import make_chat_model

        settings = self.settings_for(point)
        return AgentContext(
            retriever=self._retriever,
            llm=make_chat_model(settings, think=think),
            model=settings.resolved_chat_model,
            top_k=point.top_k,
            context_budget_tokens=settings.context_budget_tokens,
            condenser=self._condenser if with_history else None,
            history_budget_tokens=settings.history_budget_tokens,
            history_keep_messages=settings.history_keep_messages,
            citation_scorer=self._scorer,
            citation_weak_below=settings.cite_weak_below,
        )

    def _to_run(self, item: GoldenItem, final: dict[str, Any], *, think: bool) -> ChatRun:
        from geekbot.domain import RunStats

        generation = final.get("generation")
        if not generation:
            raise RuntimeError(f"the model was never called for {item.id}: no source was found")
        stats = RunStats.from_mapping(generation)
        timings = final.get("retrieval_ms", {})
        return ChatRun(
            question_id=item.id,
            first_token_s=stats.first_token_s,
            whole_s=stats.seconds,
            tokens_per_s=stats.tokens_per_second,
            load_s=stats.load_s or 0.0,
            retrieval_ms=float(timings.get("total", 0.0)),
            recall=must_include_recall(str(final.get("answer", "")), item.must_include),
            abstained=bool(final.get("abstained", False)),
            output_tokens=stats.output_tokens,
            thinking=think,
            rewrite_s=float(final.get("condense_seconds", 0.0)),
        )

    def ask(self, item: GoldenItem, point: SweepPoint, *, think: bool = False) -> ChatRun:
        context = self._context(point, think=think, with_history=False)
        final = self._graph.invoke({"question": item.question}, context=context)
        return self._to_run(item, final, think=think)

    def ask_followup(self, item: GoldenItem, point: SweepPoint) -> ChatRun:
        self._threads += 1
        config = {"configurable": {"thread_id": f"bench-{self._threads}"}}
        context = self._context(point, think=False, with_history=True)
        final: dict[str, Any] = {}
        for turn in (*item.earlier_turns, item.question):
            final = self._follow_up_graph.invoke({"question": turn}, config, context=context)
        return self._to_run(item, final, think=False)


# --- machine readers -------------------------------------------------------------------------


def read_gpu_reading() -> GpuReading | None:
    """The GPU now (memory, utilisation, compute processes), or ``None`` if ``nvidia-smi`` fails."""
    from geekbot.llm.vram import CommandError, run_command

    try:
        memory = run_command(
            (
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            )
        )
        try:
            apps = run_command(
                (
                    "nvidia-smi",
                    "--query-compute-apps=process_name,pid",
                    "--format=csv,noheader,nounits",
                )
            )
        except CommandError:
            apps = ""
        return parse_gpu_reading(memory, apps)
    except (CommandError, ValueError):
        return None


def read_used_mib() -> int | None:
    reading = read_gpu_reading()
    return None if reading is None else reading.used_mib


# --- the whole run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Machine:
    """What the report says about the environment, gathered once."""

    chat_model: str
    csv_model: str
    profile: str
    embedder: str
    reranker: str
    ollama: str


def _ingest_corpus(settings: Settings, corpus: Path, log: Log) -> dict[str, Any]:
    """Ingest the golden corpus into ``settings.data_dir`` and say how fast it went.

    The embedder runs on the CPU, so the speed is chunks per second of the PC's processor and the
    embedding model, not of the GPU.
    """
    from geekbot.index.workspace import Workspace
    from geekbot.ingest.pipeline import Ingestor, collect_files

    files, _ = collect_files([corpus])
    with Workspace.open(settings) as workspace:
        began = time.perf_counter()
        report = Ingestor(workspace.store, workspace.dense, workspace.embedder).ingest(files)
        seconds = time.perf_counter() - began
        chunks = workspace.dense.count()
    log(f"ingest: {report.summary()}")
    return {
        "documents": len(files),
        "chunks": chunks,
        "seconds": round(seconds, 1),
        "chunks_per_s": round(chunks / seconds, 1) if seconds > 0 else None,
    }


def _prepare_corpus(settings: Settings, config: BenchConfig, log: Log) -> dict[str, Any]:
    """Measure the ingest speed in a fresh throwaway folder and make sure the sweep's own folder holds
    the corpus (ingested once, kept: the sweep needs the same index every run).

    The speed is measured every run, in ``<scratch>/ingest-timing``, which is deleted first: a folder
    that already holds the documents ingests in no time and would say nothing.
    """
    import shutil

    from geekbot.index.sqlite import DB_FILE_NAME

    timing_dir = config.scratch_dir / "ingest-timing"
    if timing_dir.exists():
        shutil.rmtree(timing_dir)
    measured = _ingest_corpus(
        settings.model_copy(update={"data_dir": timing_dir}), config.chat.corpus, log
    )
    shutil.rmtree(timing_dir, ignore_errors=True)
    if not (settings.data_dir / DB_FILE_NAME).is_file():
        _ingest_corpus(settings, config.chat.corpus, log)
    return measured


def _baseline_vram(settings: Settings, client: OllamaClient, log: Log) -> dict[str, Any]:
    """GPU memory with nothing of ours loaded, with the chat model warm, with the CSV model warm.

    Uses the product's own server and settings (the numbers PLAN.md Â§6.5's budget estimated). Ends
    with our models unloaded.
    """
    from geekbot.llm.models import swap_to_mode, unload_and_wait

    unload_and_wait(client, [settings.resolved_chat_model, settings.resolved_csv_model])
    time.sleep(1.0)
    idle = read_gpu_reading()
    swap_to_mode(client, settings, "chat")
    time.sleep(1.0)
    chat = read_gpu_reading()
    swap_to_mode(client, settings, "csv")
    time.sleep(1.0)
    csv = read_gpu_reading()
    unload_and_wait(client, [settings.resolved_chat_model, settings.resolved_csv_model])
    known = next((reading for reading in (idle, chat, csv) if reading is not None), None)
    total = None if known is None else known.total_mib
    result = {
        "total_mib": total,
        "idle_mib": None if idle is None else idle.used_mib,
        "chat_mib": None if chat is None else chat.used_mib,
        "csv_mib": None if csv is None else csv.used_mib,
    }
    log(
        f"VRAM: idle {result['idle_mib']}, chat warm {result['chat_mib']}, csv warm {result['csv_mib']} MiB"
    )
    return result


def run_csv_section(
    config: BenchConfig, settings: Settings, client: OllamaClient, log: Log
) -> dict[str, Any]:
    """Seconds per question, load events and swap times of CSV mode (the product's own server)."""
    from geekbot.evals import DEFAULT_TIME_LIMIT_S, SUITE_NAMES
    from geekbot.evals.csv_repro import load_suite, run_repro
    from geekbot.llm.vram import (
        NVIDIA_SMI_MEMORY_COMMAND,
        OLLAMA_PS_COMMAND,
        GpuMemory,
        LoadedModel,
        parse_nvidia_smi_memory,
        parse_ollama_ps,
        run_command,
    )
    from geekbot.swapcheck import run_swap_cycles

    def read_gpu() -> GpuMemory:
        return parse_nvidia_smi_memory(run_command(NVIDIA_SMI_MEMORY_COMMAND))

    def read_ps() -> list[LoadedModel]:
        return parse_ollama_ps(run_command(OLLAMA_PS_COMMAND))

    suites = [load_suite(config.csv.source, name, limit=config.csv.limit) for name in SUITE_NAMES]
    repro = run_repro(
        settings,
        suites,
        runs=config.csv.runs,
        limit=config.csv.limit,
        time_limit_s=DEFAULT_TIME_LIMIT_S,
        source=config.csv.source,
        client=client,
        read_gpu=read_gpu,
        read_ps=read_ps,
        report=log,
    )
    records = run_swap_cycles(
        settings, client, cycles=config.csv.swap_cycles, read_gpu=read_gpu, read_ps=read_ps
    )
    totals = {
        label: [r.result.total_s for r in records if r.label == label]
        for label in ("chat->csv", "csv->chat")
    }

    def median(values: list[float]) -> float | None:
        ordered = sorted(values)
        return round(ordered[len(ordered) // 2], 2) if ordered else None

    summary = repro.summary
    return {
        "model": repro.context.model,
        "question_runs": summary.question_runs,
        "suites": ", ".join(repro.context.suites),
        "p50_s": round(summary.latency_p50_s, 2),
        "p95_s": round(summary.latency_p95_s, 2),
        "max_s": round(summary.latency_max_s, 2),
        "load_events": summary.load_events,
        "solved": summary.solved,
        "questions": summary.questions,
        "residency": repro.context.preflight.residency,
        "official_notes": repro.context.official_notes,
        "swap": {
            "cycles": config.csv.swap_cycles,
            "to_csv_median_s": median(totals["chat->csv"]),
            "to_csv_max_s": round(max(totals["chat->csv"], default=0.0), 2),
            "to_chat_median_s": median(totals["csv->chat"]),
            "to_chat_max_s": round(max(totals["csv->chat"], default=0.0), 2),
            "all_ok": all(r.ok for r in records),
        },
    }


def run_bench(
    config: BenchConfig,
    settings: Settings,
    *,
    phases: Sequence[str],
    only: str | None,
    quick: bool,
    log: Log,
) -> dict[str, Any]:
    """The whole bench, in the order of PLAN.md Â§9. Returns what ``bench.json`` holds.

    ``phases`` is any of ``sweep``, ``extras`` (cold rounds, follow-ups, thinking) and ``csv``;
    ``only`` limits the sweep to the one point whose key it equals; ``quick`` shrinks the question
    count and passes for a plumbing check and marks the result not official. The result is written
    to ``<results>/<name>/bench.json`` and ``report.md`` after every phase, so a crash keeps what was
    measured.
    """
    from geekbot.agent.condense import CondenseOutput
    from geekbot.agent.graph import build_graph
    from geekbot.evals.cpugate import read_snapshot, wait_until_idle
    from geekbot.evals.ollama_instance import InstanceSpec, PrivateOllama
    from geekbot.index.workspace import Workspace
    from geekbot.llm.chat import make_structured_model
    from geekbot.llm.models import unload_and_wait
    from geekbot.retrieval import SearchMode
    from geekbot.retrieval.factory import build_retriever, citation_scorer_of

    out_dir = config.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_golden(config.chat.golden)
    questions, follow_ups = select_items(items, config)
    passes = 1 if quick else config.chat.warm_passes
    if quick:
        questions = questions[:3]
        follow_ups = follow_ups[:1]
    bench_settings = settings.model_copy(update={"data_dir": config.scratch_dir / "data"})
    bench_settings.data_dir.mkdir(parents=True, exist_ok=True)
    models = Path("data") / "models"
    target_models = bench_settings.data_dir / "models"
    if models.is_dir() and not target_models.is_dir():
        import shutil

        shutil.copytree(models, target_models)  # the reranker's weights, so nothing is downloaded

    result: dict[str, Any] = {
        "name": config.name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "official": False,
        "official_notes": [],
        "machine": {},
        "guard": {},
        "sweep": [],
    }

    def save() -> None:
        (out_dir / "bench.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out_dir / "report.md").write_text(format_bench_report(result), encoding="utf-8")

    with OllamaClient(settings.ollama_host) as main:
        version = main.version()
        result["machine"] = {
            "chat_model": settings.resolved_chat_model,
            "csv_model": settings.resolved_csv_model,
            "profile": settings.profile,
            "embedder": settings.embed_model,
            "reranker": settings.rerank_model,
            "ollama": version,
        }
        # 1. the guard: with our models unloaded, is the GPU free of anyone else's work?
        unload_and_wait(main, [settings.resolved_chat_model, settings.resolved_csv_model])
        time.sleep(2.0)
        ours = [settings.resolved_chat_model, settings.resolved_csv_model, settings.embed_model]

        def check() -> GuardVerdict:
            others = [
                m.name
                for m in main.running()
                if m.vram_bytes > 0 and not any(same_model(m.name, o) for o in ours)
            ]
            return check_gpu_idle(
                read_gpu_reading(),
                other_gpu_models=others,
                max_used_mib=config.guard.max_gpu_used_mib,
                max_util_percent=config.guard.max_gpu_util_percent,
            )

        verdict = wait_until_gpu_idle(
            check, attempts=config.guard.attempts, pause_s=config.guard.pause_s, sleep=time.sleep
        )
        result["guard"] = verdict.to_json()
        cpu = wait_until_idle(
            idle_percent=config.guard.max_cpu_load_percent,
            attempts=config.guard.attempts,
            pause_s=config.guard.pause_s,
        )
        # Only the load: the gate's snapshot also names the busiest programs, which says what is
        # installed and running on the PC, and the result files are committed to a public repository.
        result["cpu"] = {"load_percent": cpu.before.load_percent}
        log(
            f"GPU idle: {verdict.idle} {list(verdict.reasons)}; CPU load {cpu.before.load_percent}%"
        )
        notes: list[str] = []
        if settings.profile != "target":
            notes.append(
                f"profile is {settings.profile}, not target: the models are not the target ones"
            )
        if not verdict.idle:
            notes.append("the GPU was not idle: " + "; ".join(verdict.to_json()["reasons"]))
        if not cpu.idle:
            notes.append(f"the CPU was busy before the first section ({cpu.before.load_percent}%)")
        if quick or only is not None:
            notes.append("partial run (quick, or a single sweep point)")
        result["official_notes"] = notes
        save()

        # 2. what the product's own settings cost: ingest speed and the VRAM table
        result["ingest"] = _prepare_corpus(bench_settings, config, log)
        result["vram"] = _baseline_vram(settings, main, log)
        save()

        shipped = SweepPoint(
            config.sweep.default_kv_cache_type,
            config.sweep.default_num_ctx,
            config.sweep.default_num_batch,
            config.sweep.default_top_k,
        )
        result["shipped_default"] = shipped.key
        points = expand_sweep(config.sweep)
        if only is not None:
            points = [p for p in points if p.key == only]
            if not points:
                raise ValueError(f"no sweep point called {only!r}")
        axes = config.sweep

        def sample() -> AbstractContextManager[VramSampler]:
            return VramSampler(read_used_mib)

        def cpu_idle() -> bool:
            gate = wait_until_idle(
                idle_percent=config.guard.max_cpu_load_percent,
                attempts=2,
                pause_s=10.0,
                read=read_snapshot,
            )
            return gate.idle

        @contextmanager
        def open_session(kv: str) -> Iterator[ChatSession]:
            with PrivateOllama(InstanceSpec(port=axes.port, kv_cache_type=kv)) as host:
                private = bench_settings.model_copy(update={"ollama_host": host})
                with OllamaClient(host) as client, Workspace.open(private) as workspace:
                    retriever = build_retriever(SearchMode.HYBRID_RERANK, workspace, private)
                    condenser = make_structured_model(
                        private, CondenseOutput, num_predict=private.condense_num_predict
                    )
                    yield GraphSession(
                        private,
                        client,
                        retriever,
                        build_graph(checkpointer=_memory_saver()),
                        build_graph(),
                        condenser,
                        citation_scorer_of(retriever),
                    )

        if "sweep" in phases:
            log(f"sweep: {len(points)} points, {len(questions)} questions x {passes} passes")
            sweep = run_sweep(
                points,
                open_session,
                questions,
                passes=passes,
                sampler=sample,
                read_used_mib=read_used_mib,
                cpu_idle=cpu_idle,
                log=log,
            )
            result["sweep"] = [r.to_json() for r in sweep]
            choice = choose_default(sweep)
            result["default_choice"] = {
                "key": None if choice.point is None else choice.point.key,
                "reason": choice.reason,
                "eligible": list(choice.eligible),
            }
            save()
        if "extras" in phases and not quick:
            with open_session(shipped.kv_cache_type) as session:
                session.load(shipped)
                result["cold_rounds"] = _cold(session, shipped, questions, config)
                result["follow_up"] = _follow_ups(session, shipped, follow_ups, passes=2)
                if config.thinking.enabled:
                    result["thinking"] = _thinking(session, shipped, questions, config)
            save()
        unload_and_wait(main, [settings.resolved_chat_model, settings.resolved_csv_model])
        if "csv" in phases and config.csv.enabled:
            try:
                result["csv"] = run_csv_section(config, settings, main, log)
            except Exception as error:  # noqa: BLE001 - the CSV rows are one section; the chat results stay
                result["csv"] = {
                    "model": settings.resolved_csv_model,
                    "residency": f"failed: {error}",
                }
            unload_and_wait(main, [settings.resolved_chat_model, settings.resolved_csv_model])
            save()
        result["official"] = not result["official_notes"] and bool(result["sweep"])
        if result["sweep"] and any(not row.get("cpu_idle", True) for row in result["sweep"]):
            result["official_notes"].append("the CPU gate was not idle before some sweep points")
            result["official"] = False
        save()
    return result


def _memory_saver() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def _cold(
    session: ChatSession, point: SweepPoint, questions: Sequence[GoldenItem], config: BenchConfig
) -> dict[str, Any]:
    if not questions or config.chat.cold_rounds == 0:
        return {}
    runs = run_cold_rounds(session, point, questions[0], rounds=config.chat.cold_rounds)
    summary = summarize_runs(runs)
    return {"setting": point.key, "summary": summary.to_json()}


def _follow_ups(
    session: ChatSession, point: SweepPoint, items: Sequence[GoldenItem], *, passes: int
) -> dict[str, Any]:
    runs: list[ChatRun] = []
    notes: list[str] = []
    for _ in range(passes):
        for item in items:
            try:
                runs.append(session.ask_followup(item, point))
            except Exception as error:  # noqa: BLE001 - one follow-up that fails is a note, not the run
                notes.append(f"{item.id}: {type(error).__name__}: {error}")
    return {
        "setting": point.key,
        "summary": summarize_runs(runs).to_json(),
        "note": "; ".join(notes[:3]),
    }


def _thinking(
    session: ChatSession, point: SweepPoint, questions: Sequence[GoldenItem], config: BenchConfig
) -> dict[str, Any]:
    runs: list[ChatRun] = []
    note = ""
    try:
        for item in list(questions)[: config.thinking.runs]:
            runs.append(session.ask(item, point, think=True))
    except (OllamaError, RuntimeError) as error:
        note = f"thinking could not be measured: {error}"
    return {"setting": point.key, "summary": summarize_runs(runs).to_json(), "note": note}
