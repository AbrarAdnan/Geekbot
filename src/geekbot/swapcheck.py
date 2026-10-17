"""``geekbot csv swap``: measure the swap between the chat model and the CSV model (step C3).

PLAN.md §6.5 R-4 says swaps are explicit; §6.7 says C3 measures the swap time in both
directions. This module runs the swaps (``llm.models.swap_to_mode``), checks after each one that
the new model is 100% on the GPU and the old one is gone, and formats the timings.

The first swap ("setup") only puts the machine in a known state (chat model loaded). Its time
depends on what was loaded before, so it is shown but never counted as a direction's timing.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from geekbot.config import Mode, Settings
from geekbot.doctor import Check, check_exclusive, check_residency
from geekbot.llm.models import Patience, SwapClient, SwapResult, swap_to_mode
from geekbot.llm.vram import GpuMemory, LoadedModel, try_read

# Plain ASCII on purpose: the report is pasted from a Windows terminal, and a redirected
# Windows console (cp1252) cannot print an arrow.
SETUP = "setup (to chat)"
TO_CSV = "chat->csv"
TO_CHAT = "csv->chat"


@dataclass(frozen=True)
class SwapRecord:
    """One swap and what was seen right after it."""

    label: str
    result: SwapResult
    residency: Check | None  # R-1 for the new model; None when `ollama ps` could not be read
    exclusive: Check | None  # R-4: the other model is gone; None when `ollama ps` failed
    gpu_used_mib: int | None  # None when nvidia-smi could not be read

    @property
    def ok(self) -> bool:
        checks = (self.residency, self.exclusive)
        return all(c is None or c.status != "FAIL" for c in checks)


def run_swap_cycles(
    settings: Settings,
    client: SwapClient,
    *,
    cycles: int,
    read_gpu: Callable[[], GpuMemory],
    read_ps: Callable[[], list[LoadedModel]],
    patience: Patience | None = None,
) -> list[SwapRecord]:
    """Set up with the chat model loaded, then swap chat→csv→chat ``cycles`` times."""
    steps: list[tuple[str, Mode]] = [(SETUP, "chat")]
    for _ in range(cycles):
        steps += [(TO_CSV, "csv"), (TO_CHAT, "chat")]

    records: list[SwapRecord] = []
    for label, mode in steps:
        result = swap_to_mode(client, settings, mode, patience=patience)
        if mode == "csv":
            active, other = settings.resolved_csv_model, settings.resolved_chat_model
            num_ctx = settings.csv_num_ctx
        else:
            active, other = settings.resolved_chat_model, settings.resolved_csv_model
            num_ctx = settings.chat_num_ctx
        loaded = try_read(read_ps)
        gpu = try_read(read_gpu)
        records.append(
            SwapRecord(
                label,
                result,
                residency=None
                if loaded is None
                else check_residency(active, loaded, expected_context=num_ctx),
                exclusive=None if loaded is None else check_exclusive(active, other, loaded),
                gpu_used_mib=None if gpu is None else gpu.used_mib,
            )
        )
    return records


def _worst(record: SwapRecord) -> str:
    """The status to show for a swap: FAIL beats WARN beats PASS; ``?`` when unreadable."""
    checks = [c for c in (record.residency, record.exclusive) if c is not None]
    if not checks:
        return "?"
    for status in ("FAIL", "WARN"):
        if any(c.status == status for c in checks):
            return status
    return "PASS"


def format_swap_report(settings: Settings, records: list[SwapRecord]) -> str:
    """A table of every swap, min/median/max per direction, and every check that is not PASS."""
    lines = [
        (
            f"profile={settings.profile}: chat {settings.resolved_chat_model} "
            f"<-> csv {settings.resolved_csv_model} (keep_alive {settings.keep_alive}, "
            f"csv num_ctx {settings.csv_num_ctx})"
        ),
        "",
        f"{'step':<17}{'unload':>9}{'warm-up':>10}{'total':>9}  {'checks':<7}{'GPU used':>10}",
    ]
    for record in records:
        gpu = f"{record.gpu_used_mib} MiB" if record.gpu_used_mib is not None else "n/a"
        lines.append(
            f"{record.label:<17}{record.result.unload_s:>7.1f} s{record.result.warm_up_s:>8.1f} s"
            f"{record.result.total_s:>7.1f} s  {_worst(record):<7}{gpu:>10}"
        )
    lines.append(f"({SETUP} is not counted: its time depends on what was loaded before.)")

    for direction in (TO_CSV, TO_CHAT):
        totals = [r.result.total_s for r in records if r.label == direction]
        if totals:
            lines.append(
                f"{direction}: {len(totals)} swap(s), total min {min(totals):.1f} s / "
                f"median {statistics.median(totals):.1f} s / max {max(totals):.1f} s"
            )
    for record in records:
        for check in (record.residency, record.exclusive):
            if check is not None and check.status != "PASS":
                lines.append(f"[{check.status}] after {record.label}: {check.name}: {check.detail}")
    return "\n".join(lines)
