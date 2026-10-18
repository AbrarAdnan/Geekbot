"""``geekbot doctor``: is this machine ready to run the active model the way the plan needs?

Each check is a small pure function (data in, ``Check`` out) so it can be tested with made-up
inputs. ``run_doctor`` only collects the data and calls them. Statuses:

- PASS: as required.
- WARN: works, but numbers or speed may suffer.
- FAIL: a plan rule is broken (the command exits with code 1).
- SKIP: not checked, and the detail says why (the thing is not built yet, or the run was told
  not to load anything, or the other mode does not need it).

Step C3 adds ``--mode``: the doctor checks the *active mode's* model (chat or CSV). Getting that
model onto the GPU is an explicit swap (unload the other mode's model, wait, warm up), so a
mode check also proves the swap leaves the new model 100% on the GPU and the old one gone.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from geekbot.config import Mode, Settings
from geekbot.index.lock import StorageLock, StorageLockedError
from geekbot.llm.models import OllamaError, SwapResult, same_model, swap_to_mode
from geekbot.llm.vram import CommandError, GpuMemory, LoadedModel

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]

# A heuristic, not a measurement: PLAN §1 saw ~1,300 MiB used by the desktop alone. Above this,
# something else is probably using the GPU and benchmark numbers would be invalid (PLAN §2).
BUSY_GPU_USED_MIB = 2048


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


class OllamaLike(Protocol):
    """What ``run_doctor`` needs from the Ollama client (so tests can pass a fake)."""

    def version(self) -> str: ...
    def list_models(self) -> list[str]: ...
    def loaded_models(self) -> list[str]: ...
    def unload(self, model: str) -> None: ...
    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float: ...


def check_models_pulled(
    chat: str, csv: str, available: list[str], *, mode: Mode = "chat"
) -> list[Check]:
    """The active mode's model must exist (FAIL); the other mode's model only warns."""
    wanted: list[tuple[str, str, Status]] = [
        ("chat", chat, "FAIL" if mode == "chat" else "WARN"),
        ("csv", csv, "FAIL" if mode == "csv" else "WARN"),
    ]
    checks: list[Check] = []
    for label, model, status_if_missing in wanted:
        if any(same_model(model, name) for name in available):
            checks.append(Check(f"{label} model pulled", "PASS", model))
        else:
            detail = f"{model}: run `ollama pull {model}`"
            checks.append(Check(f"{label} model pulled", status_if_missing, detail))
    return checks


def check_embedder_pulled(model: str, available: list[str], *, mode: Mode = "chat") -> Check:
    """The embedder (``nomic-embed-text`` by default) must exist in chat mode (FAIL); CSV mode
    does not use it, so there it only warns."""
    if any(same_model(model, name) for name in available):
        return Check("embedder model pulled", "PASS", model)
    status: Status = "FAIL" if mode == "chat" else "WARN"
    return Check("embedder model pulled", status, f"{model}: run `ollama pull {model}`")


def check_embedder_on_cpu(model: str, loaded: list[LoadedModel]) -> Check:
    """PLAN §6.5: the embedder runs on the CPU, so the chat model keeps the whole GPU.

    Looked at after a probe embedding, in ``ollama ps``: the embedder must be listed and must be
    100% CPU (it is asked for with ``num_gpu = 0``). Any share on the GPU takes VRAM from the chat
    model, which breaks rule R-1, so that is a FAIL.
    """
    entry = next((m for m in loaded if same_model(m.name, model)), None)
    if entry is None:
        return Check(
            "embedder on CPU",
            "WARN",
            f"{model} is not listed by `ollama ps` after the probe, so its placement is unknown",
        )
    if entry.gpu_percent == 0:
        return Check("embedder on CPU", "PASS", f"{model}: 100% CPU")
    return Check(
        "embedder on CPU",
        "FAIL",
        f"{model}: {entry.gpu_percent}% on the GPU (it must be 100% CPU so the chat model keeps "
        "the whole GPU)",
    )


def check_storage_lock(data_dir: Path) -> Check:
    """Is the exclusive storage lock free? (PLAN §6.2: exactly one process owns the storage.)

    Takes the lock and lets it go at once. A held lock is a WARN, not a FAIL: it normally means
    a running Geekbot server owns the data folder, which is how it should be. A folder that does
    not exist yet is nobody's, so it is free (and the check does not create it).
    """
    if not data_dir.exists():
        return Check("storage lock free", "PASS", f"no data folder yet at {data_dir}")
    lock = StorageLock(data_dir)
    try:
        lock.acquire()
    except StorageLockedError:
        return Check(
            "storage lock free",
            "WARN",
            f"another Geekbot process owns {data_dir} (a running server?)",
        )
    lock.release()
    return Check("storage lock free", "PASS", f"{data_dir} is free")


def check_gpu_memory(
    gpu: GpuMemory, *, label: str, model_loaded: bool = False, before: GpuMemory | None = None
) -> Check:
    """Report GPU memory. Before any warm-up, a high figure warns that something else is using
    the GPU. Once our model is loaded (``model_loaded``) a high figure is expected, so we only
    report it, with the change since ``before`` when we have that reading.
    """
    detail = f"{gpu.used_mib} MiB used, {gpu.free_mib} MiB free of {gpu.total_mib} MiB"
    if model_loaded:
        if before is not None:
            detail += f" ({gpu.used_mib - before.used_mib:+d} MiB)"
        return Check(label, "PASS", detail)
    if gpu.used_mib > BUSY_GPU_USED_MIB:
        return Check(label, "WARN", f"{detail}; other programs may be using the GPU")
    return Check(label, "PASS", detail)


def check_residency(model: str, loaded: list[LoadedModel], *, expected_context: int) -> Check:
    """Rule R-1: the active model is 100% on the GPU. Also catches a context mismatch."""
    entry = next((m for m in loaded if same_model(m.name, model)), None)
    if entry is None:
        return Check("residency (R-1)", "FAIL", f"{model} is not loaded")
    if entry.gpu_percent != 100:
        return Check(
            "residency (R-1)",
            "FAIL",
            f"{model}: {entry.cpu_percent}% CPU / {entry.gpu_percent}% GPU (needs 100% GPU)",
        )
    if entry.context is not None and entry.context != expected_context:
        return Check(
            "residency (R-1)",
            "WARN",
            f"{model} is 100% GPU but loaded with context {entry.context}; queries use "
            f"{expected_context}, so Ollama would reload it",
        )
    return Check("residency (R-1)", "PASS", f"{model}: 100% GPU, context {entry.context}")


def check_exclusive(active: str, other: str, loaded: list[LoadedModel]) -> Check:
    """Rules R-1/R-4: the two chat models are never on the GPU together.

    The embedder may be listed too (it runs on the CPU); only the other mode's model matters.
    """
    if same_model(active, other):
        return Check("no co-residency (R-4)", "PASS", "both modes use the same model")
    if any(same_model(m.name, other) for m in loaded):
        return Check(
            "no co-residency (R-4)",
            "FAIL",
            f"{other} is still loaded next to {active}; the two never share the GPU",
        )
    return Check("no co-residency (R-4)", "PASS", f"{other} is not loaded")


def describe_swap(result: SwapResult) -> str:
    """One line for the warm-up check: what was unloaded, what was loaded, and the times."""
    loaded = f"{result.loaded} loaded in {result.warm_up_s:.1f} s"
    if not result.unloaded:
        return loaded
    return f"{loaded}; unloaded {', '.join(result.unloaded)} first ({result.unload_s:.1f} s)"


def _gpu_check(
    read_gpu: Callable[[], GpuMemory],
    label: str,
    *,
    model_loaded: bool = False,
    before: GpuMemory | None = None,
) -> tuple[Check, GpuMemory | None]:
    """Read the GPU; return the check and the reading (None if it could not be read)."""
    try:
        gpu = read_gpu()
    except (CommandError, ValueError) as error:
        return Check(label, "WARN", f"cannot read the GPU: {error}"), None
    return check_gpu_memory(gpu, label=label, model_loaded=model_loaded, before=before), gpu


def run_doctor(
    settings: Settings,
    client: OllamaLike,
    *,
    warm_up: bool,
    read_gpu: Callable[[], GpuMemory],
    read_ps: Callable[[], list[LoadedModel]],
    mode: Mode = "chat",
    probe_embedder: Callable[[], None] | None = None,
) -> list[Check]:
    """Collect the facts and run every check. With ``warm_up`` the mode's model is loaded.

    ``read_gpu`` and ``read_ps`` return parsed data (the CLI builds them from the real commands
    plus the parsers), so the doctor itself never depends on the text format of any tool.
    The warm-up is a swap: the other mode's model is unloaded first (R-4).

    ``probe_embedder`` embeds one short text with the real embedder on the CPU (the CLI passes it;
    it raises ``OllamaError`` if that fails). It is only called in chat mode with ``warm_up``,
    because it loads a model, and the "embedder on CPU" check reads ``ollama ps`` right after.
    """
    try:
        version = client.version()
        available = client.list_models()
    except OllamaError as error:
        return [Check("ollama", "FAIL", f"{error} (is Ollama running at {settings.ollama_host}?)")]

    chat, csv = settings.resolved_chat_model, settings.resolved_csv_model
    if mode == "csv":
        active, other, num_ctx = csv, chat, settings.csv_num_ctx
    else:
        active, other, num_ctx = chat, csv, settings.chat_num_ctx
    checks = [
        Check("ollama", "PASS", f"version {version} at {settings.ollama_host}"),
        Check("profile", "PASS", f"{settings.profile}: chat {chat}, csv {csv}; checking {mode}"),
        *check_models_pulled(chat, csv, available, mode=mode),
        check_embedder_pulled(settings.embed_model, available, mode=mode),
    ]
    # If our model is already loaded (keep_alive keeps it for a while), the memory in use
    # includes it, so a high "before" figure is not another program's fault.
    try:
        already_loaded = any(same_model(m.name, active) for m in read_ps())
    except (CommandError, ValueError):
        already_loaded = False
    before_check, gpu_before = _gpu_check(
        read_gpu, "GPU memory before", model_loaded=already_loaded
    )
    if already_loaded:
        before_check = Check(
            before_check.name,
            before_check.status,
            f"{before_check.detail}; {active} already loaded",
        )
    checks.append(before_check)

    if warm_up:
        try:
            swap = swap_to_mode(client, settings, mode)
        except OllamaError as error:
            checks.append(Check("warm-up", "FAIL", str(error)))
            return checks
        checks.append(Check("warm-up", "PASS", describe_swap(swap)))

    try:
        loaded = read_ps()
    except (CommandError, ValueError) as error:
        checks.append(Check("residency (R-1)", "WARN", f"cannot read `ollama ps`: {error}"))
    else:
        if warm_up or any(same_model(m.name, active) for m in loaded):
            checks.append(check_residency(active, loaded, expected_context=num_ctx))
            checks.append(check_exclusive(active, other, loaded))
        else:
            checks.append(Check("residency (R-1)", "SKIP", f"{active} not loaded (--no-warmup)"))
    if warm_up:
        after_check, _ = _gpu_check(
            read_gpu, "GPU memory after", model_loaded=True, before=gpu_before
        )
        checks.append(after_check)

    checks.append(
        _embedder_cpu_check(
            settings.embed_model,
            available,
            mode=mode,
            warm_up=warm_up,
            probe_embedder=probe_embedder,
            read_ps=read_ps,
        )
    )
    checks.append(check_storage_lock(settings.data_dir))
    if mode == "chat":
        checks.append(
            Check("CSV model residency after a swap", "SKIP", "run `geekbot doctor --mode csv`")
        )
    return checks


def _embedder_cpu_check(
    model: str,
    available: list[str],
    *,
    mode: Mode,
    warm_up: bool,
    probe_embedder: Callable[[], None] | None,
    read_ps: Callable[[], list[LoadedModel]],
) -> Check:
    """Probe the embedder and read where it ended up, or say why that was not done."""
    name = "embedder on CPU"
    if mode != "chat":
        return Check(name, "SKIP", "CSV mode does not use the embedder")
    if not any(same_model(model, pulled) for pulled in available):
        return Check(name, "SKIP", f"{model} is not pulled")
    if not warm_up:
        return Check(name, "SKIP", "not probed (--no-warmup loads nothing)")
    if probe_embedder is None:
        return Check(name, "SKIP", "no embedder probe was given")
    try:
        probe_embedder()
    except OllamaError as error:
        return Check(name, "FAIL", f"the probe embedding failed: {error}")
    try:
        return check_embedder_on_cpu(model, read_ps())
    except (CommandError, ValueError) as error:
        return Check(name, "WARN", f"cannot read `ollama ps`: {error}")


def format_checks(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"[{c.status:<4}] {c.name:<{width}}  {c.detail}" for c in checks)


def has_failure(checks: list[Check]) -> bool:
    return any(c.status == "FAIL" for c in checks)
