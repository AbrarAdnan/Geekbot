"""The frozen response contract (PLAN.md section 4). All later phases code against this.

Changing a field here after Phase 0 is a breaking change to every consumer
(CLI, API, eval harness) — extend, don't mutate, once Phase 1 starts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Status(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    UNSUPPORTED = "unsupported"
    DATA_MISSING = "data_missing"
    AMBIGUOUS = "ambiguous"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResultKind(str, Enum):
    SCALAR = "scalar"
    SERIES = "series"
    DATAFRAME = "dataframe"
    DICT = "dict"


class ConsensusVerdict(str, Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    NONE = "none"


class NumericFidelity(str, Enum):
    PASS = "pass"
    REPAIRED = "repaired"
    FAILED = "failed"


class ResultPayload(BaseModel):
    kind: ResultKind
    shape: list[int]
    columns: list[str] = Field(default_factory=list)
    preview: list[list[Any]] = Field(default_factory=list)
    truncated: bool = False


class Evidence(BaseModel):
    code: str
    plan: list[str]
    rows_analyzed: int
    rows_after_filter: int
    columns_used: list[str]
    # Deprecated (PLAN.md Phase 6A.6): despite the name, this was never rows
    # the generated code actually dropped -- only rows with a null in
    # columns_used, whether or not the code dropped them. Kept for backward
    # compatibility with existing consumers; use the field below instead.
    null_rows_dropped: int = 0
    rows_with_nulls_in_used_columns: int = 0


class TraceEntry(BaseModel):
    attempt: int
    sample: int
    stage: Literal["context", "generate", "validate", "execute", "answer"]
    ok: bool
    ms: int
    error: Optional[str] = None
    # Set on generate entries only when this attempt ran at a raised retry
    # temperature (every sample of the previous attempt failed); None means
    # the run's base temperature in RunMeta.
    temperature: Optional[float] = None
    # "context" entries only (card H-R fact bank): the block added to the
    # planner prompt, kept for evidence.
    detail: Optional[str] = None


class Reliability(BaseModel):
    samples: int
    samples_executed: int
    consensus: ConsensusVerdict
    codes_identical: bool
    retries: int
    numeric_fidelity: NumericFidelity
    warnings: list[str] = Field(default_factory=list)


class RunMeta(BaseModel):
    model: str
    prompt_version: str
    temperature: float
    seeds: list[int]
    elapsed_ms: int
    timestamp: datetime
    # "pandas" or "sql" -- which query engine produced the result (PLAN_v2.md B4).
    engine: str = "pandas"


class RunRecord(BaseModel):
    """The audit trail, the API response, the UI trace panel, and the eval
    harness input — one object, four uses."""

    run_id: str
    question: str
    dataset_id: str
    status: Status
    answerability: Answerability

    # null unless status == "answered" (enforced by construction, not validators —
    # keep the pipeline the single writer of this object).
    answer: Optional[str] = None
    confidence: Optional[Confidence] = None
    result: Optional[ResultPayload] = None
    evidence: Optional[Evidence] = None

    reliability: Optional[Reliability] = None
    trace: list[TraceEntry] = Field(default_factory=list)
    meta: Optional[RunMeta] = None

    reason: Optional[str] = None  # required when answerability != answerable
    clarifying_question: Optional[str] = None  # required when answerability == ambiguous
