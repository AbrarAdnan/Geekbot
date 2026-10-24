"""The bench and the tuning sweep: configuration, the idle-GPU guard, the choice of default, the report
(step S5.1, PLAN.md §6.5 and §9).

``geekbot bench --config evals/configs/bench.toml`` answers three questions, in this order of trust:

1. **Is the machine quiet enough for a number to mean anything?** A latency measured while another
   program uses the GPU is a number about that program. The *idle-GPU guard* (``check_gpu_idle``, the
   ✍️ piece of this step) looks at the card before anything is timed and says why it is not idle; the
   run then still goes on, but its numbers are marked *not official* in the file and in the report
   (PLAN.md §2, §9: "official numbers are only recorded when the GPU is idle").
2. **What does each setting cost, and what does it buy?** The sweep of PLAN.md §9 runs the real answer
   graph under every combination of context size, KV-cache type, batch size and top-k, and records
   for each: residency (is the model 100% on the GPU?), peak VRAM, time to first token, tokens per
   second, model loads during questions, and answer quality (``must_include_recall``: the share of
   the golden question's must-include phrases the answer contains).
3. **Which setting should be the default?** ``choose_default``: *the best quality that meets every
   SLO at 100% residency*, where ties go to the faster first token and then to the smaller VRAM peak.

This file is the part that needs no machine: the parsing, the rules and the formatting are plain
functions, tested with made-up numbers. ``bench_run.py`` starts the servers, loads the models and asks
the questions. What the report prints is a measurement of this PC at that moment for the named models
(rule 5: a target is a target until it is measured, and a measurement is of one sitting).
"""

import itertools
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from geekbot.agent.latency import Spread, spread

__all__ = [
    "QUALITY_TOLERANCE",
    "SLO_TARGETS",
    "BenchConfig",
    "Choice",
    "ConfigResult",
    "GpuReading",
    "GuardVerdict",
    "SloMiss",
    "SweepPoint",
    "check_gpu_idle",
    "choose_default",
    "config_result_from_json",
    "expand_sweep",
    "format_bench_report",
    "load_bench_config",
    "must_include_recall",
    "parse_gpu_reading",
    "pick_questions",
    "rechoose",
    "slo_misses",
    "summarize_runs",
    "wait_until_gpu_idle",
]

# PLAN.md §6.5 "Latency SLOs" (targets until measured; warm server, local KB, no web).
SLO_TARGETS: Final = {
    "retrieval_p95_ms": 400.0,
    "first_token_p50_s": 2.5,
    "first_token_p95_s": 4.0,
    "whole_turn_p95_s": 15.0,
}
# How the guard's reason about listed processes begins (the saved form replaces what follows it).
PROCESS_REASON: Final = "the driver lists these processes:"
# Qualities this close are treated as equal when choosing the default: one missed phrase on one of
# eight questions moves the mean by about 0.04, so a smaller difference says nothing (``choose_default``).
# A guess from the question count, not a measured noise level.
QUALITY_TOLERANCE: Final = 0.05
# Ollama's own load time for a call: a few hundredths of a second when the model is warm, seconds
# when it had to be loaded again (the same threshold as ``agent.latency.LOAD_NOTICE_SECONDS``).
LOAD_NOTICE_S: Final = 0.5


# --- configuration ---------------------------------------------------------------------------


class _Section(BaseModel):
    """A config section: unknown keys are errors, so a misspelled option fails before a long run."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GuardConfig(_Section):
    """When the machine counts as idle (PLAN.md §2 GPU etiquette; D-078's CPU gate)."""

    # GPU memory in use with none of our models loaded. The desktop alone uses about 1,000 to 1,300
    # MiB on this PC (PLAN.md §1), so 2,048 leaves room for it and not for another model.
    max_gpu_used_mib: int = Field(default=2048, gt=0)
    max_gpu_util_percent: float = Field(default=15.0, ge=0)
    max_cpu_load_percent: float = Field(default=15.0, ge=0)
    attempts: int = Field(default=4, ge=1)
    pause_s: float = Field(default=20.0, ge=0)


class ChatConfig(_Section):
    """The chat questions asked under every sweep point."""

    golden: Path = Path("evals/datasets/golden.jsonl")
    corpus: Path = Path("evals/corpus")
    questions: int = Field(default=8, gt=0)  # answerable single- and multi-hop items, spread evenly
    warm_passes: int = Field(default=3, gt=0)  # every question is asked this many times
    followups: int = Field(default=3, ge=0)  # multi-turn items timed at the default setting
    cold_rounds: int = Field(
        default=2, ge=0
    )  # unload, load, first question; at the default setting


class SweepAxes(_Section):
    """PLAN.md §9: ``num_ctx`` {4k, 8k} x KV {f16, q8_0} x ``num_batch`` {256, 512} x top-k {3, 5}."""

    num_ctx: tuple[int, ...] = (4096, 8192)
    kv_cache_type: tuple[str, ...] = ("f16", "q8_0")
    num_batch: tuple[int, ...] = (256, 512)
    top_k: tuple[int, ...] = (3, 5)
    # The setting the product ships: the thinking and follow-up rows are measured at it, and the
    # report says how the chosen default differs from it.
    default_num_ctx: int = 8192
    default_kv_cache_type: str = "f16"
    default_num_batch: int = 512
    default_top_k: int = 5
    port: int = Field(default=11435, ge=1, le=65535)  # the private Ollama (never 11434)


class ThinkingConfig(_Section):
    """Thinking on, as information only (PLAN.md §9): PLAN.md §6.5 keeps it off by default."""

    enabled: bool = True
    runs: int = Field(default=3, gt=0)


class CsvConfig(_Section):
    """The CSV-mode rows (PLAN.md §9): seconds per question, swap time, residency."""

    enabled: bool = True
    source: Path = Path("scratch/gigalogy-latest")  # Gigalogy's folder, read only (C3)
    runs: int = Field(default=1, gt=0)
    limit: int | None = Field(default=None, gt=0)  # questions per suite; None = all 40
    swap_cycles: int = Field(default=3, gt=0)


class BenchConfig(_Section):
    """Everything ``geekbot bench`` needs. Paths are relative to where it is run."""

    name: str = "bench"
    results_dir: Path = Path("evals/results")
    scratch_dir: Path = Path("scratch/bench")
    guard: GuardConfig = GuardConfig()
    chat: ChatConfig = ChatConfig()
    sweep: SweepAxes = SweepAxes()
    thinking: ThinkingConfig = ThinkingConfig()
    csv: CsvConfig = CsvConfig()

    @property
    def results_path(self) -> Path:
        return self.results_dir / self.name


def load_bench_config(path: Path) -> BenchConfig:
    """Read and validate a bench TOML file. Raises ``ValueError`` with the file name on a problem."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return BenchConfig.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    except ValueError as error:  # pydantic's ValidationError is one
        raise ValueError(f"{path}: {error}") from error


@dataclass(frozen=True, order=True)
class SweepPoint:
    """One combination of the swept settings."""

    kv_cache_type: str
    num_ctx: int
    num_batch: int
    top_k: int

    @property
    def key(self) -> str:
        return f"ctx{self.num_ctx}-kv{self.kv_cache_type}-batch{self.num_batch}-k{self.top_k}"

    @property
    def load_key(self) -> tuple[str, int, int]:
        """What makes Ollama load the model again: the server's KV type and the two load options."""
        return (self.kv_cache_type, self.num_ctx, self.num_batch)


def expand_sweep(axes: SweepAxes) -> list[SweepPoint]:
    """Every combination, ordered so that work sharing a server or a model load is adjacent.

    The KV-cache type needs another server (``ollama_instance``), then ``num_ctx`` and ``num_batch``
    need another load of the model; ``top_k`` changes only the prompt. Sorting by that nesting means
    one server start per KV type and one model load per (context, batch) pair. A KV type is tried
    ``f16`` first, then the others, in the order given.
    """
    kv_order = {kind: index for index, kind in enumerate(axes.kv_cache_type)}
    points = [
        SweepPoint(kv, ctx, batch, top_k)
        for kv, ctx, batch, top_k in itertools.product(
            axes.kv_cache_type, axes.num_ctx, axes.num_batch, axes.top_k
        )
    ]
    return sorted(
        points, key=lambda p: (kv_order[p.kv_cache_type], p.num_ctx, p.num_batch, p.top_k)
    )


# --- choosing the questions ------------------------------------------------------------------


def pick_questions[T](items: Sequence[T], count: int) -> list[T]:
    """``count`` items spread evenly over ``items`` (all of them when there are fewer).

    Deterministic and independent of any random seed, so two runs of the bench ask the same
    questions and their numbers can be compared. The first and last are always in.
    """
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


# --- answer quality --------------------------------------------------------------------------

_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalized(text: str) -> str:
    """Lower case, one kind of dash and quote, no markdown emphasis, single spaces."""
    flat = text.translate(_DASHES).translate(_QUOTES).replace("*", "").replace("_", " ")
    return re.sub(r"\s+", " ", flat).strip().lower()


def must_include_recall(answer: str, phrases: Sequence[str]) -> float | None:
    """The share of ``phrases`` that appear in ``answer`` (case, dashes, quotes and spacing ignored).

    ``None`` when there are no phrases (an unanswerable item has none), never 0: a question that
    asks for nothing cannot be answered badly. This is the first of the deterministic answer
    metrics of PLAN.md §9 ("must-include recall"); S5.2 builds the rest on it. It is a floor on
    quality, not a judgement: an answer can contain the phrase and still be wrong around it.
    """
    wanted = [_normalized(phrase) for phrase in phrases if phrase.strip()]
    if not wanted:
        return None
    text = _normalized(answer)
    return sum(phrase in text for phrase in wanted) / len(wanted)


# --- the idle-GPU guard (the ✍️ piece of step S5.1) -------------------------------------------


@dataclass(frozen=True)
class GpuReading:
    """What ``nvidia-smi`` says about the card now: memory, how busy it is and who is computing."""

    used_mib: int
    total_mib: int
    utilization_percent: float | None  # None when the driver does not report it
    compute_apps: tuple[str, ...] = ()  # "name (pid)" of each compute process it lists

    def to_json(self) -> dict[str, Any]:
        """The saved form. The names of the listed processes are **not** saved, only how many: they
        are a list of what is installed and running on someone's PC (the first live run listed a
        chat client, a browser, a game launcher and the user's folder), and result files are
        committed to a public repository. The terminal shows the names while the run goes on."""
        return {
            "used_mib": self.used_mib,
            "total_mib": self.total_mib,
            "utilization_percent": self.utilization_percent,
            "compute_apps_count": len(self.compute_apps),
        }


def parse_gpu_reading(memory_text: str, apps_text: str = "") -> GpuReading:
    """Parse ``nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu`` (csv, no units)
    and, optionally, ``--query-compute-apps=process_name,pid`` (csv, no header).

    With several GPUs the first one. A utilization of ``[N/A]`` (or a missing third column) is
    ``None``. Raises ``ValueError`` for text with no GPU line. Windows drivers in WDDM mode often
    list no compute processes at all, so an empty ``apps_text`` proves nothing; the memory and the
    utilization are the evidence that always exists.
    """
    for line in memory_text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        try:
            used, total = int(parts[0]), int(parts[1])
        except (ValueError, IndexError) as error:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}") from error
        utilization: float | None = None
        if len(parts) > 2:
            try:
                utilization = float(parts[2])
            except ValueError:
                utilization = None  # "[N/A]", "[Not Supported]"
        apps: list[str] = []
        for app_line in apps_text.splitlines():
            cells = [cell.strip() for cell in app_line.split(",")]
            if len(cells) >= 2 and cells[0] and not cells[0].startswith("["):
                apps.append(f"{cells[0]} ({cells[1]})")
        return GpuReading(used, total, utilization, tuple(apps))
    raise ValueError("nvidia-smi printed no GPU line")


@dataclass(frozen=True)
class GuardVerdict:
    """Whether the GPU is idle, and every reason it is not (empty when it is)."""

    idle: bool
    reasons: tuple[str, ...]
    reading: GpuReading | None
    waited_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """The saved form: the reasons as they are, except that a list of process names becomes a
        count (see ``GpuReading.to_json``)."""
        count = 0 if self.reading is None else len(self.reading.compute_apps)
        reasons = [
            f"{PROCESS_REASON} {count} of them (the names are not saved)"
            if reason.startswith(PROCESS_REASON)
            else reason
            for reason in self.reasons
        ]
        return {
            "idle": self.idle,
            "reasons": reasons,
            "reading": None if self.reading is None else self.reading.to_json(),
            "waited_s": round(self.waited_s, 1),
        }


def check_gpu_idle(
    reading: GpuReading | None,
    *,
    other_gpu_models: Sequence[str],
    max_used_mib: int,
    max_util_percent: float,
) -> GuardVerdict:
    """Is the GPU free of other work? ``reading`` is taken **with our own models unloaded**.

    ✍️ Written by Claude in prototyping mode (D-047, step S5.1); on the backlog in STATUS.md.

    Every reason that applies is reported, not only the first, because the person reading wants to
    know what to close:

    1. **No reading** (``None``: ``nvidia-smi`` missing or unreadable): not idle. An unverified
       GPU is never assumed idle.
    2. **Memory**: more than ``max_used_mib`` in use with none of our models loaded means another
       program holds video memory (the desktop alone stays under the default).
    3. **Utilisation**: a busy card (above ``max_util_percent``) even with little memory in use,
       for example another program's inference or a video encode. ``None`` is not held against it.
    4. **Other models in Ollama**: ``other_gpu_models`` are models some Ollama server holds on the
       GPU that the bench does not own (the CSV project's model, say). Our own models are the
       caller's to unload before the reading and to leave out of this list; the embedder runs on
       the CPU and never belongs in it.

    The processes the driver lists are **evidence, not a rule**: on Windows the driver lists every
    program that has a window (the shell, a browser, a chat client: 21 of them on this PC with the
    GPU doing nothing, found in the first live run), so "any listed process" would call no desktop
    idle. They are added to the reasons that rule 2 or 3 already produced, to say who may be using
    the card, and never make a quiet card busy. Another program's inference shows up as memory and
    utilisation anyway.
    """
    if reading is None:
        return GuardVerdict(
            False, ("the GPU could not be read, so it cannot be called idle",), None
        )
    reasons: list[str] = []
    if reading.used_mib > max_used_mib:
        reasons.append(
            f"{reading.used_mib} MiB of video memory is in use with our models unloaded "
            f"(limit {max_used_mib} MiB): another program holds it"
        )
    util = reading.utilization_percent
    if util is not None and util > max_util_percent:
        reasons.append(f"the GPU is {util:.0f}% busy (limit {max_util_percent:.0f}%)")
    if reasons and reading.compute_apps:
        reasons.append(f"{PROCESS_REASON} " + _short_apps(reading.compute_apps))
    if other_gpu_models:
        reasons.append("other models are loaded on the GPU: " + ", ".join(other_gpu_models))
    return GuardVerdict(not reasons, tuple(reasons), reading)


def _short_apps(apps: Sequence[str], *, limit: int = 6) -> str:
    """``name (pid)`` of the first ``limit`` processes, without their folders, and how many more."""
    names = [app.replace("/", "\\").rsplit("\\", 1)[-1] for app in apps]
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def wait_until_gpu_idle(
    check: Callable[[], GuardVerdict],
    *,
    attempts: int,
    pause_s: float,
    sleep: Callable[[float], None],
) -> GuardVerdict:
    """Ask ``check`` up to ``attempts`` times, ``pause_s`` apart, and return the last verdict.

    The verdict carries how long was waited. A card that never goes quiet is a verdict of "not
    idle" with its reasons, not an error: the caller decides whether to go on (the bench does, and
    marks everything unofficial).
    """
    waited = 0.0
    verdict = check()
    for _ in range(attempts - 1):
        if verdict.idle:
            break
        sleep(pause_s)
        waited += pause_s
        verdict = check()
    return GuardVerdict(verdict.idle, verdict.reasons, verdict.reading, waited)


# --- one run, one configuration --------------------------------------------------------------


@dataclass(frozen=True)
class ChatRun:
    """One question through the answer graph: the model's readings, retrieval and quality."""

    question_id: str
    first_token_s: float | None
    whole_s: float
    tokens_per_s: float | None
    load_s: float
    retrieval_ms: float
    recall: float | None  # must-include recall; None for an item with no phrases
    abstained: bool = False
    output_tokens: int | None = None
    thinking: bool = False
    rewrite_s: float = 0.0  # a follow-up's rewrite into a standalone question (0 on a first turn)

    @property
    def question_to_first_token_s(self) -> float | None:
        """What the user waits for before the first word: the rewrite of a follow-up (if any),
        retrieval, then the model's first token."""
        if self.first_token_s is None:
            return None
        return self.rewrite_s + self.retrieval_ms / 1000.0 + self.first_token_s


@dataclass(frozen=True)
class RunSummary:
    """The readings of a list of runs: spreads, loads, quality. The run count is always kept."""

    n: int
    first_token: Spread | None  # the model's own time to first token (s)
    to_first_token: Spread | None  # question to first token: retrieval plus the model's (s)
    whole: Spread | None  # the whole turn (s)
    tokens_per_s: Spread | None
    retrieval_ms: Spread | None
    loads: int  # runs in which Ollama spent half a second or more loading the model
    quality: float | None  # mean must-include recall over the runs that have phrases
    quality_n: int

    def to_json(self) -> dict[str, Any]:
        def one(value: Spread | None) -> dict[str, float] | None:
            if value is None:
                return None
            return {"n": value.n, "p50": value.p50, "p95": value.p95, "max": value.maximum}

        return {
            "n": self.n,
            "first_token_s": one(self.first_token),
            "to_first_token_s": one(self.to_first_token),
            "whole_s": one(self.whole),
            "tokens_per_s": one(self.tokens_per_s),
            "retrieval_ms": one(self.retrieval_ms),
            "loads": self.loads,
            "quality": self.quality,
            "quality_n": self.quality_n,
        }


def summarize_runs(runs: Sequence[ChatRun]) -> RunSummary:
    """Spreads of every reading over ``runs`` (a reading a run lacks is left out, never zero)."""
    recalls = [run.recall for run in runs if run.recall is not None]
    return RunSummary(
        n=len(runs),
        first_token=spread([r.first_token_s for r in runs if r.first_token_s is not None]),
        to_first_token=spread(
            [t for t in (r.question_to_first_token_s for r in runs) if t is not None]
        ),
        whole=spread([r.whole_s for r in runs]),
        tokens_per_s=spread([r.tokens_per_s for r in runs if r.tokens_per_s is not None]),
        retrieval_ms=spread([r.retrieval_ms for r in runs]),
        loads=sum(1 for r in runs if r.load_s >= LOAD_NOTICE_S),
        quality=(sum(recalls) / len(recalls)) if recalls else None,
        quality_n=len(recalls),
    )


@dataclass(frozen=True)
class SloMiss:
    """One target a configuration missed: which, what was seen, what was wanted."""

    name: str
    seen: float
    target: float
    unit: str

    def __str__(self) -> str:
        return f"{self.name} {self.seen:.2f} {self.unit} > {self.target:g} {self.unit}"


def slo_misses(summary: RunSummary, *, include_retrieval: bool = True) -> list[SloMiss]:
    """The targets of PLAN.md §6.5 this summary misses (empty = all met).

    ``include_retrieval=False`` leaves out the retrieval p95: retrieval runs on the CPU and does not
    depend on any swept setting, so it cannot help choose between them (``choose_default``); the
    report still shows it. A reading that does not exist (no run reported a first token) is not a
    miss and not a pass: the caller sees ``n`` and decides.
    """
    misses: list[SloMiss] = []

    def check(name: str, value: float | None, target: float, unit: str) -> None:
        if value is not None and value > target:
            misses.append(SloMiss(name, value, target, unit))

    if include_retrieval and summary.retrieval_ms is not None:
        check("retrieval p95", summary.retrieval_ms.p95, SLO_TARGETS["retrieval_p95_ms"], "ms")
    if summary.to_first_token is not None:
        check("question to first token p50", summary.to_first_token.p50, 2.5, "s")
        check("question to first token p95", summary.to_first_token.p95, 4.0, "s")
    if summary.whole is not None:
        check("whole turn p95", summary.whole.p95, SLO_TARGETS["whole_turn_p95_s"], "s")
    return misses


@dataclass(frozen=True)
class ConfigResult:
    """Everything measured for one sweep point."""

    point: SweepPoint
    load_s: float  # warming the model up for this point (0 when the load was shared)
    gpu_percent: int | None  # the model's share in video memory, from Ollama (None: unreadable)
    loaded_context: int | None
    vram_idle_mib: int | None  # video memory in use before the model was loaded
    vram_peak_mib: int | None  # the most in use while the questions ran
    summary: RunSummary
    cpu_idle: bool = True  # the CPU gate before the section (D-078)
    notes: tuple[str, ...] = ()

    @property
    def resident(self) -> bool:
        return self.gpu_percent == 100

    def misses(self, *, include_retrieval: bool = False) -> list[SloMiss]:
        return slo_misses(self.summary, include_retrieval=include_retrieval)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.point.key,
            "point": {
                "num_ctx": self.point.num_ctx,
                "kv_cache_type": self.point.kv_cache_type,
                "num_batch": self.point.num_batch,
                "top_k": self.point.top_k,
            },
            "load_s": round(self.load_s, 2),
            "gpu_percent": self.gpu_percent,
            "loaded_context": self.loaded_context,
            "vram_idle_mib": self.vram_idle_mib,
            "vram_peak_mib": self.vram_peak_mib,
            "cpu_idle": self.cpu_idle,
            "resident": self.resident,
            "slo_misses": [str(m) for m in self.misses()],
            "summary": self.summary.to_json(),
            "notes": list(self.notes),
        }


def config_result_from_json(row: Mapping[str, Any]) -> ConfigResult:
    """Read back a row of ``bench.json`` (what ``ConfigResult.to_json`` wrote).

    Lets ``geekbot bench --report-only`` apply the *current* choice rule to a saved run instead of
    repeating the sentence the old rule wrote: the saved numbers are data, the choice is a function
    of them (PLAN.md §9: every number regenerable).
    """

    def spread_of(value: Mapping[str, Any] | None) -> Spread | None:
        if not value:
            return None
        return Spread(
            int(value["n"]), float(value["p50"]), float(value["p95"]), float(value["max"])
        )

    point = row["point"]
    summary = row["summary"]
    return ConfigResult(
        point=SweepPoint(
            point["kv_cache_type"],
            int(point["num_ctx"]),
            int(point["num_batch"]),
            int(point["top_k"]),
        ),
        load_s=float(row.get("load_s", 0.0)),
        gpu_percent=row.get("gpu_percent"),
        loaded_context=row.get("loaded_context"),
        vram_idle_mib=row.get("vram_idle_mib"),
        vram_peak_mib=row.get("vram_peak_mib"),
        summary=RunSummary(
            n=int(summary.get("n", 0)),
            first_token=spread_of(summary.get("first_token_s")),
            to_first_token=spread_of(summary.get("to_first_token_s")),
            whole=spread_of(summary.get("whole_s")),
            tokens_per_s=spread_of(summary.get("tokens_per_s")),
            retrieval_ms=spread_of(summary.get("retrieval_ms")),
            loads=int(summary.get("loads", 0)),
            quality=summary.get("quality"),
            quality_n=int(summary.get("quality_n", 0)),
        ),
        cpu_idle=bool(row.get("cpu_idle", True)),
        notes=tuple(row.get("notes", ())),
    )


def rechoose(result: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a saved bench result whose ``default_choice`` follows today's rule.

    Results with no sweep come back unchanged. Used by the report so it never shows a sentence
    that an older version of the rule wrote.
    """
    rows = result.get("sweep") or []
    updated: dict[str, Any] = dict(result)
    if not rows:
        return updated
    choice = choose_default(config_result_from_json(row) for row in rows)
    updated["default_choice"] = {
        "key": None if choice.point is None else choice.point.key,
        "reason": choice.reason,
        "eligible": list(choice.eligible),
    }
    return updated


# --- choosing the default --------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """The outcome of ``choose_default``: the winner (or none) and why, in words."""

    point: SweepPoint | None
    reason: str
    eligible: tuple[str, ...] = field(default=())  # keys of the points that met every rule


def choose_default(
    results: Iterable[ConfigResult], *, quality_tolerance: float = QUALITY_TOLERANCE
) -> Choice:
    """The best quality that meets every SLO at 100% residency (PLAN.md §9).

    A configuration is **eligible** when its model is 100% on the GPU (R-1), no run of it loaded the
    model during a question (R-2), and it misses none of the SLOs that depend on the swept settings
    (the retrieval target is the CPU's and is the same everywhere).

    Among the eligible ones, the quality is compared with a **tolerance**: every configuration whose
    mean must-include recall is within ``quality_tolerance`` of the best is treated as equally good.
    The first full bench (S5.1) asked 8 questions three times each; one missed phrase on one question
    moves the mean by about 0.04, so differences of that size are noise, and ranking on them would
    pick a winner from nothing. Among the equally good, the winner has the lowest median
    question-to-first-token time (rounded to the hundredth of a second), then the lowest VRAM peak,
    then the smaller context size and batch (cheaper to run, and deterministic). With
    ``quality_tolerance=0`` the best quality decides. ``Choice.point`` is ``None`` with the reasons
    when nothing is eligible: the report then says which rule excluded each point.
    """
    all_results = list(results)
    eligible: list[ConfigResult] = []
    excluded: list[str] = []
    for result in all_results:
        why: list[str] = []
        if not result.resident:
            share = "unknown" if result.gpu_percent is None else f"{result.gpu_percent}%"
            why.append(f"residency {share}")
        if result.summary.loads:
            why.append(f"{result.summary.loads} model loads during questions")
        why.extend(str(miss) for miss in result.misses())
        if result.summary.n == 0:
            why.append("no runs")
        if why:
            excluded.append(f"{result.point.key}: " + "; ".join(why))
        else:
            eligible.append(result)
    if not eligible:
        detail = " | ".join(excluded) if excluded else "no configuration was measured"
        return Choice(None, f"no configuration meets every rule ({detail})")

    def quality_of(result: ConfigResult) -> float:
        value = result.summary.quality
        return 0.0 if value is None else value  # no evidence ranks below any measured quality

    def rank(result: ConfigResult) -> tuple[float, int, int, int]:
        first = result.summary.to_first_token
        peak = result.vram_peak_mib
        return (
            round(first.p50 if first is not None else float("inf"), 2),
            peak if peak is not None else 10**9,
            result.point.num_ctx,
            result.point.num_batch,
        )

    top_quality = max(quality_of(result) for result in eligible)
    equally_good = [
        result
        for result in eligible
        if quality_of(result) >= top_quality - quality_tolerance - 1e-9
    ]
    best = min(equally_good, key=rank)
    first = best.summary.to_first_token
    reason = (
        f"{best.point.key}: {len(eligible)} of {len(all_results)} configurations meet every rule; "
        f"{len(equally_good)} of them are within {quality_tolerance:g} of the best must-include "
        f"recall ({top_quality:.3f}), which is as fine as this question set can tell apart, so the "
        f"fastest median first token decides "
        f"({'n/a' if first is None else f'{first.p50:.2f} s'}), then the smaller VRAM peak"
    )
    if excluded:
        reason += ". Excluded: " + " | ".join(excluded)
    return Choice(best.point, reason, tuple(r.point.key for r in eligible))


# --- the report ------------------------------------------------------------------------------


def _seconds(value: Spread | None, attribute: str = "p50") -> str:
    return "n/a" if value is None else f"{getattr(value, attribute):.2f}"


def _millis(value: Spread | None, attribute: str = "p50") -> str:
    return "n/a" if value is None else f"{getattr(value, attribute):.0f}"


def format_bench_report(result: Mapping[str, Any]) -> str:
    """The Markdown report of a bench result (the dictionary ``bench.json`` holds).

    Reads only what the file holds, so ``geekbot bench report`` can regenerate it from the results
    folder (PLAN.md §9: every number is regenerable). Every table names its model and says whether
    the run was official; nothing here is a claim about a different machine or model.
    """
    result = rechoose(result)  # the choice is a function of the saved rows, not a saved sentence
    lines: list[str] = []
    add = lines.append
    machine = result.get("machine", {})
    add(f"# Bench report: {result.get('name', '?')}")
    add("")
    add(
        f"Chat model `{machine.get('chat_model', '?')}`, CSV model `{machine.get('csv_model', '?')}`, "
        f"profile `{machine.get('profile', '?')}`, embedder `{machine.get('embedder', '?')}` and "
        f"reranker `{machine.get('reranker', '?')}` on the CPU; Ollama {machine.get('ollama', '?')}; "
        f"run {result.get('created', '?')}."
    )
    add("")
    official = bool(result.get("official"))
    add(f"**Official: {'yes' if official else 'NO (a plumbing check, not a result)'}.**")
    for note in result.get("official_notes", []):
        add(f"- {note}")
    guard = result.get("guard") or {}
    reading = guard.get("reading") or {}
    add("")
    add("## The machine at the start")
    add(
        f"GPU idle: **{'yes' if guard.get('idle') else 'no'}** "
        f"({reading.get('used_mib', '?')} of {reading.get('total_mib', '?')} MiB used with our "
        f"models unloaded, {reading.get('utilization_percent', '?')}% busy; waited "
        f"{guard.get('waited_s', 0)} s). "
        + (
            "Reasons it was not: " + "; ".join(guard.get("reasons", []))
            if guard.get("reasons")
            else ""
        )
    )
    cpu = result.get("cpu") or {}
    if cpu:
        add(f"CPU load before the first section: {cpu.get('load_percent', '?')}%.")

    ingest = result.get("ingest")
    if ingest:
        add("")
        add("## Ingest")
        add(
            f"{ingest.get('documents', '?')} documents, {ingest.get('chunks', '?')} chunks in "
            f"{ingest.get('seconds', '?')} s: **{ingest.get('chunks_per_s', '?')} chunks per second** "
            f"(embedder on the CPU)."
        )

    vram = result.get("vram")
    if vram:
        add("")
        add("## VRAM at the product defaults (PLAN.md §6.5 budget, measured)")
        add("| state | GPU memory in use (MiB) | free (MiB) |")
        add("|---|---|---|")
        total = vram.get("total_mib")
        for label, key in (
            ("nothing of ours loaded", "idle_mib"),
            ("chat model warm", "chat_mib"),
            ("CSV model warm", "csv_mib"),
        ):
            used = vram.get(key)
            free = "n/a" if used is None or total is None else str(total - used)
            add(f"| {label} | {'n/a' if used is None else used} | {free} |")

    sweep = result.get("sweep") or []
    if sweep:
        add("")
        add("## The sweep (every row: the real answer graph, retrieval on the CPU)")
        add(
            "Quality is the mean must-include recall over the answerable golden questions asked "
            "(Claude-authored, provisional). The model's first token and the question-to-first-token "
            "time are in seconds; p95 over few runs is close to the maximum."
        )
        add("")
        add(
            "| ctx | KV | batch | top-k | GPU % | load s | peak MiB | quality | 1st token p50 | "
            "Q to 1st p50 | Q to 1st p95 | tok/s p50 | whole p95 | retrieval p95 ms | loads | "
            "misses |"
        )
        add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in sweep:
            point, summary = row["point"], row["summary"]
            ftk, q2f, whole, tps, retr = (
                summary.get("first_token_s"),
                summary.get("to_first_token_s"),
                summary.get("whole_s"),
                summary.get("tokens_per_s"),
                summary.get("retrieval_ms"),
            )

            def value(spread_json: Mapping[str, Any] | None, key: str, digits: int = 2) -> str:
                return "n/a" if not spread_json else f"{spread_json[key]:.{digits}f}"

            quality = summary.get("quality")
            add(
                f"| {point['num_ctx']} | {point['kv_cache_type']} | {point['num_batch']} | "
                f"{point['top_k']} | {row.get('gpu_percent', 'n/a')} | {row.get('load_s', 'n/a')} | "
                f"{row.get('vram_peak_mib') or 'n/a'} | "
                f"{'n/a' if quality is None else f'{quality:.3f}'} | {value(ftk, 'p50')} | "
                f"{value(q2f, 'p50')} | {value(q2f, 'p95')} | {value(tps, 'p50', 1)} | "
                f"{value(whole, 'p95')} | {value(retr, 'p95', 0)} | {summary.get('loads', 'n/a')} | "
                f"{'; '.join(row.get('slo_misses', [])) or '-'} |"
            )
        add("")
        add(
            "The retrieval target (p95 400 ms) is the CPU's and is the same under every setting, so "
            "it takes no part in choosing; the column shows how this sitting did against it."
        )
        choice = result.get("default_choice") or {}
        add("")
        add(f"**Chosen default:** {choice.get('reason', 'none')}")
        shipped = result.get("shipped_default")
        if shipped:
            add(f"The product ships `{shipped}`.")

    for section, title in (
        ("follow_up", "Follow-up questions (condense, then retrieval, then the model)"),
        ("thinking", "Thinking on, as information (thinking is off by default)"),
    ):
        data = result.get(section)
        if not data:
            continue
        summary = data.get("summary", {})
        q2f, whole = summary.get("to_first_token_s"), summary.get("whole_s")
        add("")
        add(f"## {title}")
        add(
            f"{summary.get('n', 0)} runs at `{data.get('setting', '?')}`: question to first token "
            f"p50 {'n/a' if not q2f else f'{q2f["p50"]:.2f}'} s / p95 "
            f"{'n/a' if not q2f else f'{q2f["p95"]:.2f}'} s; whole turn p50 "
            f"{'n/a' if not whole else f'{whole["p50"]:.2f}'} s / p95 "
            f"{'n/a' if not whole else f'{whole["p95"]:.2f}'} s."
            + (f" {data['note']}" if data.get("note") else "")
        )

    csv = result.get("csv")
    if csv:
        add("")
        add("## CSV mode")
        add(
            f"Model `{csv.get('model', '?')}`; {csv.get('question_runs', '?')} question-runs "
            f"({csv.get('suites', '?')}): seconds per question **p50 {csv.get('p50_s', 'n/a')} / "
            f"p95 {csv.get('p95_s', 'n/a')} / max {csv.get('max_s', 'n/a')}**; model loads during "
            f"questions: {csv.get('load_events', 'n/a')}; solved {csv.get('solved', 'n/a')} of "
            f"{csv.get('questions', 'n/a')} (a check on the copy, not a new claim)."
        )
        swap = csv.get("swap") or {}
        if swap:
            add(
                f"Swaps ({swap.get('cycles', '?')} round trips): chat to csv median "
                f"{swap.get('to_csv_median_s', 'n/a')} s (max {swap.get('to_csv_max_s', 'n/a')}), "
                f"csv to chat median {swap.get('to_chat_median_s', 'n/a')} s (max "
                f"{swap.get('to_chat_max_s', 'n/a')}); every residency check passed: "
                f"{swap.get('all_ok', 'n/a')}."
            )
        add(f"Residency: {csv.get('residency', 'n/a')}.")
        add(f"CSV official notes: {'; '.join(csv.get('official_notes', [])) or 'none'}.")

    add("")
    add("## Targets (PLAN.md §6.5), for reading the tables above")
    add(
        "Retrieval plus rerank p95 400 ms; question to first token p50 2.5 s and p95 4 s; whole turn "
        "p95 15 s; model loads during questions 0; CSV: p50 3 s, p95 10 s, swap 6 s. A target is a "
        "target until a sitting on an idle PC measures it."
    )
    return "\n".join(lines) + "\n"
