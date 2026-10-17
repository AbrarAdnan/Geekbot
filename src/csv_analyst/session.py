"""Chat entry point: one CSV per conversation, one grounded answer per message.

Made for a host chatbot's "CSV analyst" mode (see INTEGRATION.md):

    from csv_analyst import CsvAnalystSession, to_markdown

    session = CsvAnalystSession.from_bytes(upload_bytes, filename="sales.csv")  # when the CSV arrives
    record = session.ask("What is the average revenue by region?")              # each message in CSV mode
    reply = to_markdown(record)                                                 # chat-ready text + evidence

Each question is answered on its own: the session keeps the data, not the conversation. `ask` blocks
for seconds (local 7b: about 3 s typically, at most the per-question time limit), so an async host
should run it in a worker thread. A bad CSV raises `IngestionError` at load time; a question never
raises for being hard -- unanswerable, ambiguous and failed questions come back as a `RunRecord` with
`status` ABSTAINED / NEEDS_CLARIFICATION / FAILED and a `reason`. An unreachable model backend comes
back as FAILED with a service-error reason, not as a dataset limitation.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from csv_analyst.config import settings
from csv_analyst.contracts import ResultKind, RunRecord, Status
from csv_analyst.ingestion.loader import LoadResult, load_csv, load_csv_bytes
from csv_analyst.llm.base import LLMProvider
from csv_analyst.llm.ollama_provider import OllamaProvider
from csv_analyst.pipeline import run_pipeline
from csv_analyst.profiling.profiler import profile_dataset
from csv_analyst.profiling.schema import SchemaContext

# GEEKBOT EDIT (C1): the "anthropic" and "openai_compatible" providers are removed. They would
# send the CSV's data to a hosted service on a path Geekbot's egress gate does not control (D-041).
PROVIDERS = ("ollama",)
PREVIEW_ROWS = 10


def make_provider(provider: str = "ollama", model: Optional[str] = None) -> LLMProvider:
    """The model backend. Only "ollama" (local) exists in Geekbot's copy.
    (GEEKBOT EDIT C1: the api_key / api_base_url parameters went with the hosted providers.)"""
    model = model or settings.llm_model_main
    if provider == "ollama":
        return OllamaProvider(model=model)
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


class CsvAnalystSession:
    """A loaded, profiled CSV plus the model that answers questions about it."""

    def __init__(self, load_result: LoadResult, *, name: str = "data.csv", model: Optional[str] = None,
                 provider: str = "ollama", llm: Optional[LLMProvider] = None):
        self.name = name
        self.dataset_id = str(uuid.uuid4())
        self.load_result = load_result
        self.df = load_result.df
        self.schema: SchemaContext = profile_dataset(load_result, dataset_id=self.dataset_id)
        self.model = model or settings.llm_model_main
        self.llm = llm or make_provider(provider, self.model)

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> "CsvAnalystSession":
        return cls(load_csv(path), name=Path(path).name, **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, filename: str = "upload.csv", **kwargs) -> "CsvAnalystSession":
        return cls(load_csv_bytes(data, filename), name=Path(filename or "upload.csv").name, **kwargs)

    # GEEKBOT EDIT (C2): `time_limit_s` is passed through to run_pipeline so the host can set its own
    # per-question cap. Before, ask() always used the package's limit (10 minutes for the main model).
    # GEEKBOT EDIT (C3): `seed_offset` is passed through to run_pipeline (which already has it) so a
    # reproduction run can make its 3 runs differ, the way Gigalogy's benchmark does (README: "3 seed
    # sets"). The default 0 is what every other caller gets.
    def ask(self, question: str, *, samples: Optional[int] = None,
            time_limit_s: Optional[float] = None, seed_offset: int = 0) -> RunRecord:
        """Answer one question from the data (the full reliability pipeline)."""
        return run_pipeline(self.llm, self.df, self.schema, question, self.dataset_id,
                            model_name=self.model, samples=samples, time_limit_s=time_limit_s,
                            seed_offset=seed_offset)

    def describe(self) -> str:
        """A short chat message confirming what was loaded."""
        rows, cols = self.df.shape
        names = ", ".join(f"`{c}`" for c in list(self.df.columns)[:20])
        more = f" and {cols - 20} more" if cols > 20 else ""
        text = f"Loaded **{self.name}**: {rows:,} rows, {cols} columns ({names}{more})."
        notes = [w.message for w in self.load_result.warnings]
        if notes:
            text += "\n\n" + "\n".join(f"- {n}" for n in notes)
        return text


def to_markdown(record: RunRecord, *, evidence: bool = True) -> str:
    """A chat reply: the answer (or why there is none), then the query and result it rests on."""
    if record.status != Status.ANSWERED:
        text = record.reason or "I could not reliably answer this question from the provided dataset."
        if record.clarifying_question:
            text += f"\n\n{record.clarifying_question}"
        return text

    parts = [record.answer or ""]
    if record.confidence is not None:
        parts.append(f"*Confidence: {record.confidence.value}*")
    if record.reliability and record.reliability.warnings:
        parts.append("\n".join(f"- {w}" for w in record.reliability.warnings))
    if evidence and record.evidence:
        lang = "sql" if record.meta and record.meta.engine == "sql" else "python"
        parts.append(f"**How this was computed** ({record.evidence.rows_analyzed:,} rows analyzed):\n"
                     f"```{lang}\n{record.evidence.code}\n```")
        result = record.result
        if result and result.kind == ResultKind.DICT:  # preview rows are [key, value]
            parts.append(_table(["key", "value"], result.preview, result.shape))
        elif result and result.columns:
            parts.append(_table(result.columns, result.preview, result.shape))
        elif result and result.preview:
            parts.append(f"Result: `{result.preview[0][0]}`")
    return "\n\n".join(p for p in parts if p)


def _table(columns: list[str], rows: list[list], shape: list[int]) -> str:
    def cell(v) -> str:
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(c) for c in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows[:PREVIEW_ROWS]]
    total = shape[0] if shape else len(rows)
    if total > PREVIEW_ROWS:
        lines.append(f"\n*First {PREVIEW_ROWS} of {total:,} rows.*")
    return "\n".join(lines)
