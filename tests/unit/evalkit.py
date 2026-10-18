"""Shared fakes and tiny datasets for the retrieval-evaluation tests (step S2.6).

Nothing here talks to Ollama or loads a model. The embedder hashes words into a small vector, so
texts that share words point the same way and *ranking* can be tested; the scorer counts shared
words, so a reranker has something to do. SQLite and Chroma are real.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from geekbot.config import Settings
from geekbot.evals.ablation import (
    AblationConfig,
    BakeoffConfig,
    BeirConfig,
    EmbedderConfig,
    FusionConfig,
    GoldenConfig,
    RerankerConfig,
    TuningConfig,
)
from geekbot.evals.cpugate import CpuSnapshot, GateReading
from geekbot.evals.datasets import GoldenItem, Source, write_golden
from geekbot.evals.indexing import EmbedderChoice, IndexVariant
from geekbot.evals.phases import Backends
from geekbot.index.embedder import EmbedderSpec, EmbeddingsClient, PrefixedEmbeddings
from geekbot.retrieval.rerank import PassageScorer

DIM = 512
TOPICS = ("zebra", "quartz", "bread", "river", "violin", "falcon", "maple", "copper")
_WORD = re.compile(r"[a-z]+")


class HashEmbeddings:
    """Words hashed into ``DIM`` buckets, unit length: shared words mean a shared direction."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        counts = [0.0] * DIM
        for word in _WORD.findall(text.lower()):
            counts[int(hashlib.sha256(word.encode()).hexdigest()[:6], 16) % DIM] += 1.0
        norm = math.sqrt(sum(c * c for c in counts)) or 1.0
        return [c / norm for c in counts]


class OverlapScorer:
    """The share of the question's words that a passage contains (a stand-in cross-encoder)."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        words = set(_WORD.findall(query.lower()))
        return [
            len(words & set(_WORD.findall(passage.lower()))) / max(len(words), 1)
            for passage in passages
        ]


def fake_backends(
    *,
    opened: list[tuple[str, str]] | None = None,
    scorers: list[str] | None = None,
    idle: bool = True,
) -> Backends:
    """Backends with no model: ``opened`` records (embedder label, variant key) per open."""

    def opener(
        choice: EmbedderChoice, variant: IndexVariant
    ) -> tuple[EmbeddingsClient, EmbedderSpec, Callable[[], None]]:
        if opened is not None:
            opened.append((choice.label, variant.key))
        document = choice.document_prefix if variant.prefixes else ""
        query = choice.query_prefix if variant.prefixes else ""
        spec = EmbedderSpec(
            model=choice.model,
            digest="digest-1",
            document_prefix=document,
            query_prefix=query,
            dimension=DIM,
            with_header=variant.header,
        )
        embeddings = PrefixedEmbeddings(
            HashEmbeddings(), document_prefix=document, query_prefix=query
        )
        return embeddings, spec, lambda: None

    def make_scorer(model: str, cache_dir: Path | None) -> PassageScorer:
        del cache_dir
        if scorers is not None:
            scorers.append(model)
        return OverlapScorer()

    def gate(*, idle_percent: float = 15.0) -> GateReading:
        del idle_percent
        return GateReading(CpuSnapshot(1.0 if idle else 90.0, ()), None, idle, 0.0)

    def snapshot() -> CpuSnapshot:
        return CpuSnapshot(2.0, (("python", 3.0),))

    return Backends(open_embeddings=opener, make_scorer=make_scorer, gate=gate, snapshot=snapshot)


# --- a tiny BEIR dataset ---------------------------------------------------------------------


def write_tiny_beir(folder: Path) -> Path:
    """Sixteen documents (two per topic) and eight questions per split, in BEIR's file layout."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qrels").mkdir(exist_ok=True)
    corpus = []
    for topic in TOPICS:
        for number in (1, 2):
            corpus.append(
                {
                    "_id": f"{topic}-{number}",
                    "title": f"{topic.title()} notes {number}",
                    "text": (
                        f"The {topic} appears in many old stories. A careful reader of {topic} "
                        f"material learns that the {topic} was valued. Note number {number} about "
                        f"the {topic} adds that people kept a {topic} close."
                    ),
                }
            )
    (folder / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row) for row in corpus) + "\n", encoding="utf-8"
    )
    queries = []
    train_rows = ["query-id\tcorpus-id\tscore"]
    test_rows = ["query-id\tcorpus-id\tscore"]
    for topic in TOPICS:
        queries.append({"_id": f"q-test-{topic}", "text": f"what do people say about the {topic}"})
        queries.append({"_id": f"q-train-{topic}", "text": f"tell me the {topic} story"})
        for number in (1, 2):
            test_rows.append(f"q-test-{topic}\t{topic}-{number}\t1")
            train_rows.append(f"q-train-{topic}\t{topic}-{number}\t1")
    (folder / "queries.jsonl").write_text(
        "\n".join(json.dumps(row) for row in queries) + "\n", encoding="utf-8"
    )
    (folder / "qrels" / "test.tsv").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
    (folder / "qrels" / "train.tsv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    return folder


# --- a tiny golden set -----------------------------------------------------------------------

GOLDEN_FILES = ("zebra", "quartz", "bread", "river")


def write_tiny_golden(root: Path) -> tuple[Path, Path, dict[str, str]]:
    """A corpus folder, a golden.jsonl and the cached rewrites its multi-turn items would have."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in GOLDEN_FILES:
        (corpus / f"{name}.txt").write_text(
            f"# {name.title()}\n\nThe {name} is the subject of this file. Everything the studio "
            f"knows about the {name} is written here. The {name} was last reviewed in spring.\n",
            encoding="utf-8",
        )
    items: list[GoldenItem] = []
    rewrites: dict[str, str] = {}
    for number in range(10):
        name = GOLDEN_FILES[number % 4]
        items.append(
            GoldenItem(
                f"S-{number + 1:03d}",
                "single_hop",
                f"when was the {name} last reviewed",
                sources=(Source(f"{name}.txt"),),
                must_include=("spring",),
            )
        )
    for number in range(3):
        first, second = GOLDEN_FILES[number], GOLDEN_FILES[number + 1]
        items.append(
            GoldenItem(
                f"M-{number + 1:03d}",
                "multi_hop",
                f"compare the {first} and the {second}",
                sources=(Source(f"{first}.txt"), Source(f"{second}.txt")),
                must_include=("spring",),
            )
        )
    for number in range(5):
        name = GOLDEN_FILES[number % 4]
        item_id = f"T-{number + 1:03d}"
        items.append(
            GoldenItem(
                item_id,
                "multi_turn",
                "and when was that one reviewed",
                earlier_turns=(f"what is the {name} file about",),
                sources=(Source(f"{name}.txt"),),
                must_include=("spring",),
            )
        )
        rewrites[item_id] = f"when was the {name} last reviewed"
    for number in range(5):
        items.append(
            GoldenItem(
                f"U-{number + 1:03d}",
                "unanswerable",
                f"who won the {1990 + number} football final in paris",
                reason="the documents never mention football",
            )
        )
    path = root / "golden.jsonl"
    write_golden(path, items)
    return corpus, path, rewrites


def tiny_config(root: Path, **overrides: object) -> AblationConfig:
    """A configuration that points at the tiny datasets under ``root`` (created if missing)."""
    beir_root = root / "beir"
    write_tiny_beir(beir_root / "scifact")
    corpus, golden, _ = write_tiny_golden(root)
    values: dict[str, object] = {
        "name": "t",
        "scratch_dir": root / "scratch",
        "results_dir": root / "results",
        "top_k": 5,
        "resamples": 200,
        "warmup_queries": 1,
        "embedder": EmbedderConfig(
            label="fake-embed",
            model="fake",
            document_prefix="search_document: ",
            query_prefix="search_query: ",
        ),
        "reranker": RerankerConfig(label="overlap", model="fake-reranker", depth=5),
        "fusion": FusionConfig(candidates=10),
        "beir": BeirConfig(folder=beir_root, datasets=("scifact",)),
        "golden": GoldenConfig(enabled=True, jsonl=golden, corpus=corpus),
        "tuning": TuningConfig(max_queries=6, depths=(5, 8), candidates=(5, 10), latency_queries=4),
        "bakeoff": BakeoffConfig(
            embedders=(EmbedderConfig(label="other-embed", model="other-fake"),),
            rerankers=(RerankerConfig(label="other-rerank", model="other-fake-reranker", depth=5),),
        ),
        **overrides,
    }
    return AblationConfig.model_validate(values)


def tiny_settings(root: Path) -> Settings:
    return Settings(_env_file=None, data_dir=root / "data")


def write_tiny_toml(root: Path) -> Path:
    """The same configuration as ``tiny_config``, as the TOML file the command line reads."""
    config = tiny_config(root)
    posix = lambda path: Path(path).as_posix()  # TOML wants forward slashes
    text = f"""
name = "t"
scratch_dir = "{posix(config.scratch_dir)}"
results_dir = "{posix(config.results_dir)}"
top_k = 5
resamples = 200
warmup_queries = 1

[embedder]
label = "fake-embed"
model = "fake"
document_prefix = "search_document: "
query_prefix = "search_query: "

[reranker]
label = "overlap"
model = "fake-reranker"
depth = 5

[fusion]
candidates = 10

[beir]
folder = "{posix(config.beir.folder)}"
datasets = ["scifact"]

[golden]
enabled = true
jsonl = "{posix(config.golden.jsonl)}"
corpus = "{posix(config.golden.corpus)}"

[tuning]
max_queries = 6
depths = [5, 8]
candidates = [5, 10]
latency_queries = 4
"""
    path = root / "tiny.toml"
    path.write_text(text, encoding="utf-8")
    return path
