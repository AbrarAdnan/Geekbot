"""The ablation ladder and its configuration (step S2.6, PLAN.md §9).

An ablation answers "which change made retrieval better?" by changing **one thing per step** and
measuring after each. The rungs are cumulative and in a fixed order (PLAN.md §9):

====  ==========================================================================================
R0    legacy v1 replica (800-character chunks, no prefixes, L2, five template queries, re-sort)
R1    L2 becomes cosine
R2    the template multi-query goes (and with it the re-sort that merged its results)
R3    + task prefixes
R4    + the new chunker (512 tokens, section-aware)
R5    + the contextual header (BEIR: the title)
R6    + BM25 fused by RRF
R7    + the cross-encoder rerank
====  ==========================================================================================

Plus three leave-one-out variants of R7, each removing exactly one part: no BM25, no rerank (that is
R6, so it is not run twice), and no header. A ladder says what each part adds *given what came
before*; leave-one-out says what it adds *given everything else*. They can disagree when two parts
overlap, which is why the plan asks for both.

The embedder is held at ``nomic-embed-text`` throughout; the embedder and reranker choices are
compared separately (the bake-offs). The configuration is a TOML file (``evals/configs/``), parsed
and validated here, so a typo in it fails before a three-hour run starts and not after.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from geekbot.evals.cpugate import DEFAULT_IDLE_LOAD_PERCENT
from geekbot.evals.indexing import EmbedderChoice, IndexVariant

__all__ = [
    "LEAVE_ONE_OUT",
    "RUNGS",
    "V2_HEADER",
    "V2_PLAIN",
    "AblationConfig",
    "LeaveOneOut",
    "Rung",
    "Strategy",
    "load_config",
    "rung_by_name",
]

Strategy = Literal["legacy_multiquery", "dense", "hybrid", "hybrid_rerank", "dense_rerank"]

# The four index variants of the ladder (plus the L2 twin of the first).
LEGACY_RAW_L2: Final = IndexVariant("legacy", prefixes=False, header=False, space="l2")
LEGACY_RAW: Final = IndexVariant("legacy", prefixes=False, header=False, space="cosine")
LEGACY_PREFIXED: Final = IndexVariant("legacy", prefixes=True, header=False, space="cosine")
V2_PLAIN: Final = IndexVariant("v2", prefixes=True, header=False, space="cosine")
V2_HEADER: Final = IndexVariant("v2", prefixes=True, header=True, space="cosine")


@dataclass(frozen=True)
class Rung:
    """One step of the ladder: which index it searches and how it searches it."""

    name: str
    change: str  # the single change this rung makes, in a few words, for the report
    variant: IndexVariant
    strategy: Strategy


RUNGS: Final = (
    Rung("R0", "legacy v1 replica", LEGACY_RAW_L2, "legacy_multiquery"),
    Rung("R1", "L2 to cosine", LEGACY_RAW, "legacy_multiquery"),
    Rung("R2", "remove template multi-query", LEGACY_RAW, "dense"),
    Rung("R3", "+ prefixes", LEGACY_PREFIXED, "dense"),
    Rung("R4", "new chunking", V2_PLAIN, "dense"),
    Rung("R5", "+ contextual header", V2_HEADER, "dense"),
    Rung("R6", "+ BM25 via RRF", V2_HEADER, "hybrid"),
    Rung("R7", "+ rerank", V2_HEADER, "hybrid_rerank"),
)


@dataclass(frozen=True)
class LeaveOneOut:
    """R7 with exactly one part removed. ``same_as`` names a rung that already is that system."""

    name: str
    removes: str
    variant: IndexVariant
    strategy: Strategy
    same_as: str | None = None


LEAVE_ONE_OUT: Final = (
    LeaveOneOut("R7 - BM25", "BM25", V2_HEADER, "dense_rerank"),
    LeaveOneOut("R7 - rerank", "rerank", V2_HEADER, "hybrid", same_as="R6"),
    LeaveOneOut("R7 - header", "header", V2_PLAIN, "hybrid_rerank"),
)


def rung_by_name(name: str) -> Rung:
    """The rung called ``name`` (``R0`` to ``R7``); ``KeyError`` with the valid names otherwise."""
    for rung in RUNGS:
        if rung.name == name:
            return rung
    raise KeyError(f"no rung {name!r}; the rungs are {', '.join(r.name for r in RUNGS)}")


# --- configuration ---------------------------------------------------------------------------


class _Section(BaseModel):
    """A config section: unknown keys are errors (a misspelled option must not be ignored)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbedderConfig(_Section):
    label: str
    model: str
    document_prefix: str = ""
    query_prefix: str = ""

    def choice(self) -> EmbedderChoice:
        return EmbedderChoice(self.label, self.model, self.document_prefix, self.query_prefix)


class RerankerConfig(_Section):
    label: str = "MiniLM-L-6"
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    depth: int = Field(default=10, gt=0)
    # Where the weights live. ``None`` is the product's own model folder (``<data_dir>/models``),
    # which holds the default model already; a candidate that has to be downloaded gets a scratch
    # folder so the evaluation never adds to the user's data folder.
    cache_dir: Path | None = None


class FusionConfig(_Section):
    candidates: int = Field(default=50, gt=0)


class BeirConfig(_Section):
    folder: Path = Path("data/beir")
    datasets: tuple[str, ...] = ("scifact", "nfcorpus")


class GoldenConfig(_Section):
    enabled: bool = True
    jsonl: Path = Path("evals/datasets/golden.jsonl")
    corpus: Path = Path("evals/corpus")


class TuningConfig(_Section):
    """The sweeps run on the tuning splits only (D-065): SciFact train, NFCorpus dev, golden dev."""

    max_queries: int = Field(default=300, gt=0)
    depths: tuple[int, ...] = (10, 20, 30)
    candidates: tuple[int, ...] = (20, 50, 100)
    latency_queries: int = Field(default=100, gt=0)


class BakeoffConfig(_Section):
    embedders: tuple[EmbedderConfig, ...] = ()
    rerankers: tuple[RerankerConfig, ...] = ()


class AblationConfig(_Section):
    """Everything ``geekbot eval retrieval`` needs. Paths are relative to where it is run."""

    name: str = "ablation"
    scratch_dir: Path = Path("scratch/ablation")
    results_dir: Path = Path("evals/results")
    top_k: int = Field(default=10, ge=1)
    seed: int = 0
    resamples: int = Field(default=10_000, ge=100)
    idle_load_percent: float = DEFAULT_IDLE_LOAD_PERCENT
    warmup_queries: int = Field(default=3, ge=0)
    embedder: EmbedderConfig
    # The model that rewrites golden follow-ups (D-076). ``None``: the profile's chat model. It is
    # named in the rewrite cache and the report, because a score on those items is partly about it.
    rewrite_model: str | None = None
    reranker: RerankerConfig = RerankerConfig()
    fusion: FusionConfig = FusionConfig()
    beir: BeirConfig = BeirConfig()
    golden: GoldenConfig = GoldenConfig()
    tuning: TuningConfig = TuningConfig()
    bakeoff: BakeoffConfig = BakeoffConfig()

    @property
    def results_path(self) -> Path:
        """The folder this run's result files go in."""
        return self.results_dir / self.name


def load_config(path: Path) -> AblationConfig:
    """Read and validate a TOML configuration. ``ValueError`` with the reason if it is not valid."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path} is not valid TOML: {error}") from error
    try:
        return AblationConfig.model_validate(raw)
    except ValueError as error:  # pydantic's ValidationError is a ValueError
        raise ValueError(f"{path} is not a valid configuration:\n{error}") from error
