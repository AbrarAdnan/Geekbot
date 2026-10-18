"""The CPU gate: is the PC idle enough for a timing to mean anything? (D-078, PLAN.md §6.5)

A latency number from a busy PC is a number about the PC. Two timing sessions in run 2 were ruined
that way (a depth-10 rerank slower than depth 30, a p95 of eight seconds), so every timed section
of the ablation is bracketed by a snapshot of the machine's load and its busiest processes, and the
snapshots are stored next to the numbers. Quality numbers do not depend on the load; latency
numbers do.

The snapshot asks Windows (``Get-CimInstance Win32_Processor``, the counter the user and the rules
name), through PowerShell. On any other system, or if PowerShell fails, the load is ``None`` and the
section is flagged as unchecked rather than assumed idle. The function that runs the command is a
parameter, so the tests need no PowerShell.
"""

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "DEFAULT_IDLE_LOAD_PERCENT",
    "CpuSnapshot",
    "GateReading",
    "read_snapshot",
    "wait_until_idle",
]

DEFAULT_IDLE_LOAD_PERCENT: Final = 15
# Prints {"load": <percent>, "top": [{"name": ..., "cpu": ...}, ...]} as JSON. The process counter
# needs two samples a moment apart; it is the only way to see *current* use per process.
_SCRIPT: Final = (
    "$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average)."
    "Average; "
    "$top = (Get-Counter '\\Process(*)\\% Processor Time' -ErrorAction SilentlyContinue)"
    ".CounterSamples | Where-Object { $_.InstanceName -notin @('_total','idle') } | "
    "Sort-Object CookedValue -Descending | Select-Object -First 5 | "
    "ForEach-Object { [pscustomobject]@{ name = $_.InstanceName; cpu = [math]::Round("
    "$_.CookedValue, 1) } }; "
    "[pscustomobject]@{ load = $load; top = @($top) } | ConvertTo-Json -Compress -Depth 4"
)


@dataclass(frozen=True)
class CpuSnapshot:
    """The load (percent, ``None`` if it could not be read) and the five busiest processes."""

    load_percent: float | None
    top: tuple[tuple[str, float], ...]

    def to_json(self) -> dict[str, Any]:
        return {"load_percent": self.load_percent, "top": [list(item) for item in self.top]}


@dataclass(frozen=True)
class GateReading:
    """A timed section's gate: the snapshots around it and whether it counts as idle."""

    before: CpuSnapshot
    after: CpuSnapshot | None
    idle: bool
    waited_s: float

    def to_json(self) -> dict[str, Any]:
        return {
            "before": self.before.to_json(),
            "after": None if self.after is None else self.after.to_json(),
            "idle": self.idle,
            "waited_s": round(self.waited_s, 1),
        }


def _run_powershell(script: str) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout


def read_snapshot(run: Callable[[str], str] = _run_powershell) -> CpuSnapshot:
    """Read the load and the busiest processes now. Never raises: a failure is ``load_percent=None``."""
    try:
        data = json.loads(run(_SCRIPT))
        load = data.get("load")
        top = tuple(
            (str(item["name"]), float(item["cpu"])) for item in data.get("top") or [] if item
        )
        return CpuSnapshot(None if load is None else float(load), top)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError):
        return CpuSnapshot(None, ())


def wait_until_idle(
    *,
    idle_percent: float = DEFAULT_IDLE_LOAD_PERCENT,
    attempts: int = 6,
    pause_s: float = 30.0,
    read: Callable[[], CpuSnapshot] = read_snapshot,
    sleep: Callable[[float], None] = time.sleep,
) -> GateReading:
    """Take a ``before`` snapshot, waiting up to ``attempts - 1`` pauses for the PC to go quiet.

    ``idle`` is true only if a reading was obtained and was at or below ``idle_percent``. A section
    that never got a quiet reading still runs (the caller decides), but carries ``idle=False``
    into the results and the report, which says so beside its numbers.
    """
    waited = 0.0
    snapshot = read()
    for _ in range(attempts - 1):
        if snapshot.load_percent is not None and snapshot.load_percent <= idle_percent:
            break
        sleep(pause_s)
        waited += pause_s
        snapshot = read()
    idle = snapshot.load_percent is not None and snapshot.load_percent <= idle_percent
    return GateReading(before=snapshot, after=None, idle=idle, waited_s=waited)
