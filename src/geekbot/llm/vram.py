"""Reading the GPU and Ollama's view of it (lesson: project/learn/03-ollama-gpu.md).

The parsers are pure functions of text, so they are tested on recorded real output
(``tests/fixtures/``) with no GPU or Ollama needed. Running the commands is separate
(``run_command``) and is the only part that touches the machine.
"""

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Output: "<used MiB>, <total MiB>" per GPU, e.g. "654, 10240".
NVIDIA_SMI_MEMORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
OLLAMA_PS_COMMAND = ("ollama", "ps")


class CommandError(RuntimeError):
    """A system command was missing, failed or timed out."""


@dataclass(frozen=True)
class GpuMemory:
    """GPU memory in MiB, as ``nvidia-smi`` reports it (all processes, not just Ollama)."""

    used_mib: int
    total_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


@dataclass(frozen=True)
class LoadedModel:
    """One row of ``ollama ps``: a model Ollama currently holds in memory."""

    name: str
    gpu_percent: int
    cpu_percent: int
    # None when this Ollama version prints no CONTEXT column.
    context: int | None
    # Ollama's own wording, e.g. "4 minutes from now" or "Forever".
    until: str


def run_command(command: Sequence[str], *, timeout_s: float = 10.0) -> str:
    """Run a command and return its standard output, or raise ``CommandError``."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CommandError(f"`{' '.join(command)}` failed: {error}") from error
    return completed.stdout


def try_read[T](read: Callable[[], T]) -> T | None:
    """Run a reader of the machine's state (one of the commands below plus its parser).

    Returns ``None`` when the command is missing, fails or prints something unreadable, so a
    report can say "unknown" instead of crashing (step C3).
    """
    try:
        return read()
    except (CommandError, ValueError):
        return None


def parse_nvidia_smi_memory(text: str) -> GpuMemory:
    """Parse the output of ``NVIDIA_SMI_MEMORY_COMMAND``. With several GPUs, the first one."""
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            used, total = (int(part) for part in line.split(","))
        except ValueError as error:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}") from error
        return GpuMemory(used_mib=used, total_mib=total)
    raise ValueError("nvidia-smi printed no GPU line")


def parse_ollama_ps(text: str) -> list[LoadedModel]:
    """Parse the text printed by ``ollama ps`` into the models it lists.

    ✍️ YOUR PIECE (step S0.3). Real output from Ollama 0.34.2 (see
    ``tests/fixtures/ollama_ps_one_model.txt``)::

        NAME         ID              SIZE      PROCESSOR    CONTEXT    UNTIL
        gemma3:4b    68ceab7e468a    3.0 GB    100% GPU     131072     4 minutes from now

    Rules:

    - The first line is a header. Header only, or empty text, means nothing is loaded: ``[]``.
    - Columns are separated by two or more spaces (a single space can occur inside a value,
      as in ``3.0 GB``). Find each value by its header name, not by a fixed position: other
      Ollama versions print no CONTEXT column (then ``context`` is ``None``).
    - PROCESSOR forms: ``100% GPU`` (gpu 100, cpu 0), ``100% CPU`` (gpu 0, cpu 100) and
      ``48%/52% CPU/GPU``: the numbers follow the order of the label, so cpu 48, gpu 52.
    - ``until`` is the UNTIL text as printed.
    """
    models: list[LoadedModel] = []
    lines = text.splitlines()
    if not lines:
        return []
    cells = re.split(r"\s{2,}", lines[0])

    for line in lines[1:]:
        values = re.split(r"\s{2,}", line)
        row = dict(zip(cells, values))
        processor = row["PROCESSOR"].split()[-1].split("/")
        # print(processor)
        usage = [int(n) for n in re.findall(r"\d+", row["PROCESSOR"])]
        # print(usage)
        shares = dict(zip(processor, usage))
        # print(shares)
        raw_context = row.get("CONTEXT")
        model = LoadedModel(
            name=row["NAME"],
            gpu_percent=shares.get("GPU", 0),
            cpu_percent=shares.get("CPU", 0),
            context=int(raw_context) if raw_context is not None else None,
            until=row["UNTIL"],
        )
        models.append(model)
    return models
