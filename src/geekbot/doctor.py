"""``geekbot doctor``: is this machine ready to run the active model the way the plan needs?

Each check is a small pure function (data in, ``Check`` out) so it can be tested with made-up
inputs. ``run_doctor`` only collects the data and calls them. Statuses:

- PASS: as required.
- WARN: works, but numbers or speed may suffer.
- FAIL: a plan rule is broken (the command exits with code 1).
- SKIP: the thing doesn't exist yet; the detail names the step that builds it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from geekbot.config import Settings
from geekbot.llm.models import OllamaError, model_is_pulled
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
    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float: ...


def check_models_pulled(chat: str, csv: str, available: list[str]) -> list[Check]:
    """The chat model must exist (FAIL); the CSV model is only needed from step C2 (WARN)."""
    wanted: list[tuple[str, str, Status]] = [("chat", chat, "FAIL"), ("csv", csv, "WARN")]
    checks: list[Check] = []
    for label, model, status_if_missing in wanted:
        if model_is_pulled(model, available):
            checks.append(Check(f"{label} model pulled", "PASS", model))
        else:
            detail = f"{model}: run `ollama pull {model}`"
            checks.append(Check(f"{label} model pulled", status_if_missing, detail))
    return checks


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
    entry = next((m for m in loaded if m.name == model), None)
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
) -> list[Check]:
    """Collect the facts and run every check. With ``warm_up`` the chat model is loaded.

    ``read_gpu`` and ``read_ps`` return parsed data (the CLI builds them from the real commands
    plus the parsers), so the doctor itself never depends on the text format of any tool.
    """
    try:
        version = client.version()
        available = client.list_models()
    except OllamaError as error:
        return [Check("ollama", "FAIL", f"{error} (is Ollama running at {settings.ollama_host}?)")]

    chat, csv = settings.resolved_chat_model, settings.resolved_csv_model
    checks = [
        Check("ollama", "PASS", f"version {version} at {settings.ollama_host}"),
        Check("profile", "PASS", f"{settings.profile}: chat {chat}, csv {csv}"),
        *check_models_pulled(chat, csv, available),
    ]
    # If our model is already loaded (keep_alive keeps it for a while), the memory in use
    # includes it, so a high "before" figure is not another program's fault.
    try:
        already_loaded = any(m.name == chat for m in read_ps())
    except (CommandError, ValueError):
        already_loaded = False
    before_check, gpu_before = _gpu_check(
        read_gpu, "GPU memory before", model_loaded=already_loaded
    )
    if already_loaded:
        before_check = Check(
            before_check.name, before_check.status, f"{before_check.detail}; {chat} already loaded"
        )
    checks.append(before_check)

    if warm_up:
        try:
            seconds = client.warm_up(
                chat, num_ctx=settings.chat_num_ctx, keep_alive=settings.keep_alive
            )
        except OllamaError as error:
            checks.append(Check("warm-up", "FAIL", str(error)))
            return checks
        checks.append(Check("warm-up", "PASS", f"{chat} loaded in {seconds:.1f} s"))

    try:
        loaded = read_ps()
    except (CommandError, ValueError) as error:
        checks.append(Check("residency (R-1)", "WARN", f"cannot read `ollama ps`: {error}"))
    else:
        if warm_up or any(m.name == chat for m in loaded):
            checks.append(check_residency(chat, loaded, expected_context=settings.chat_num_ctx))
        else:
            checks.append(Check("residency (R-1)", "SKIP", f"{chat} not loaded (--no-warmup)"))
    if warm_up:
        after_check, _ = _gpu_check(
            read_gpu, "GPU memory after", model_loaded=True, before=gpu_before
        )
        checks.append(after_check)

    checks += [
        Check("embedder on CPU", "SKIP", "not built yet (step S1.5)"),
        Check("storage lock free", "SKIP", "not built yet (step S1.4)"),
        Check("CSV model residency after a swap", "SKIP", "not built yet (step C3)"),
    ]
    return checks


def format_checks(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"[{c.status:<4}] {c.name:<{width}}  {c.detail}" for c in checks)


def has_failure(checks: list[Check]) -> bool:
    return any(c.status == "FAIL" for c in checks)
