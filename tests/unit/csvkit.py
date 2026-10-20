"""Fakes for the CSV conversation tests (step C5): a table that loads from a real file but is scripted.

``FakeCsv`` stands in for ``csv_mode.adapter.CsvAnalyst``: it reads the uploaded file (so a header-only
file fails to load the way the real loader fails, and a file's size shows in ``memory_bytes``), counts
what it is asked and answers from a script. Nothing here imports pandas, the vendored package or
Ollama. ``CsvFakes`` is the shared record of everything the app opened, which a test inspects.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from geekbot.csv_mode.errors import CsvLoadError

CSV_MODEL = "fake-csv:1b"
SALES = b"region,revenue\nNorth,520\nSouth,380\n"
REPLY = (
    "North leads with 520.\n\n**How this was computed** (2 rows analyzed):\n```sql\nSELECT 1\n```"
)


@dataclass
class CsvFakes:
    """What the fake CSV side was asked, and how it behaves.

    ``reply`` is the markdown every question gets (or ``replies`` in order, the last repeating),
    ``status`` the package's verdict, ``ask_delay_s`` and ``open_delay_s`` block the thread they run
    in with ``time.sleep``, like the real package does. ``memory_factor`` multiplies the file's size
    to give ``memory_bytes``. ``error`` makes ``ask`` raise.
    """

    reply: str = REPLY
    replies: list[str] = field(default_factory=list)
    status: str = "answered"
    ask_delay_s: float = 0.0
    open_delay_s: float = 0.0
    memory_factor: int = 1
    error: Exception | None = None
    opened: list["FakeCsv"] = field(default_factory=list)

    def open(self, path: Path) -> "FakeCsv":
        """The opener the registry is given: load the file, or raise like the real loader."""
        if self.open_delay_s:
            time.sleep(self.open_delay_s)
        session = FakeCsv(path, self)
        self.opened.append(session)
        return session


class FakeCsv:
    """One loaded table. ``questions`` is what it was asked, ``closed`` whether it was closed."""

    def __init__(self, path: Path, fakes: CsvFakes) -> None:
        text = path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise CsvLoadError("CSV has zero data rows (header only)")
        self.path = path
        self._fakes = fakes
        self._columns = lines[0].split(",")
        self._rows = len(lines) - 1
        self._size = len(text.encode("utf-8"))
        self.questions: list[str] = []
        self.closed = False

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    @property
    def load_notes(self) -> list[str]:
        return []

    @property
    def memory_bytes(self) -> int:
        return self._size * self._fakes.memory_factor

    def ask(self, question: str, /) -> SimpleNamespace:
        self.questions.append(question)
        if self._fakes.ask_delay_s:
            time.sleep(self._fakes.ask_delay_s)
        if self._fakes.error is not None:
            raise self._fakes.error
        replies = self._fakes.replies
        markdown = (replies.pop(0) if len(replies) > 1 else replies[0]) if replies else None
        return SimpleNamespace(
            markdown=markdown if markdown is not None else self._fakes.reply,
            status=self._fakes.status,
            model=CSV_MODEL,
            seconds=0.5,
            model_load_ms=0,
        )

    def close(self) -> None:
        self.closed = True
