"""Live smoke test for step S3.6: the answer path against the latency targets, and the two search schedules.

Run it with ``pytest -m live -s tests/live/test_slo_smoke_live.py`` when the PC is quiet (Gigalogy not
using the GPU). Deselected by default. It needs Ollama with ``nomic-embed-text`` and the active
profile's chat model pulled (``gemma3:4b`` on the default ``proto`` profile). The reranker's weights are
copied from your own ``data/models`` if that folder exists; otherwise the first run downloads them
once (about 87 MB). The knowledge base is the golden corpus (``evals/corpus``, 15 files, about 65
chunks), ingested into a temporary folder.

**What this is, and what it is not.** It is a *smoke test*: it checks that the whole answer path holds
together on real models and prints the readings next to the targets of PLAN.md §6.5, so a person can
see at a glance whether anything is wildly off. It is **not an SLO measurement**: the models are the
light prototyping ones (D-042), the runs are few (the 95th percentile of 24 runs is close to the
slowest one), the PC is shared, and a target is only met by the *target* models on an idle GPU. So the
latency numbers are printed, never asserted. What is asserted is what must hold on any machine: every
turn finishes, the model is not loaded again during a question (rule R-2), the stages the graph reports
are there, the steps are announced in order, and the two search schedules return the same chunks.

Two tests:

1. ``test_the_answer_path_against_the_latency_targets`` starts the real server (chat model onto the
   GPU, reranker on the CPU), asks eight questions three times each through the HTTP route, and prints
   retrieval (by stage), the model's time to first token, the question-to-first-token sum, the whole
   turn, and how many questions loaded a model, against the targets.
2. ``test_parallel_search_against_sequential_search`` needs no chat model. It runs the same questions
   through the answer graph twice, once with the sequential retriever and once with the staged one
   (dense ‖ lexical), interleaved so neither has the warmer caches, with an instant fake in place of
   the chat model, so the difference between the two is the search schedule and nothing else. It
   prints both, and the saving the parallel schedule could at most make (the shorter of the two
   searches). That is the number S3.6's honest claim rests on.
"""

import shutil
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import pytest
from chatkit import ScriptedChat
from fastapi.testclient import TestClient
from serverkit import wait_for

from geekbot.agent.graph import build_graph
from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.retrieval import Retriever, SearchMode
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.factory import build_retriever
from geekbot.retrieval.lexical import LexicalRetriever
from geekbot.retrieval.rerank import Reranker
from geekbot.retrieval.rerank_timing import percentile
from geekbot.retrieval.staged import StagedSearch
from geekbot.server import create_app
from geekbot.server.security import Perimeter
from geekbot.sse import parse_text

CORPUS: Final = Path("evals") / "corpus"
QUESTIONS: Final = (
    "How much video memory does the graphics card in the Lantern computer have?",
    "How fast does the assistant produce text, counted in tokens?",
    (
        "In the first speed test, how long did the slowest 5 percent of requests wait before the "
        "first word of an answer?"
    ),
    "What context length did the IT contractor set to bring response times down?",
    "For how long is the battery unit rated to keep the computer running at normal load?",
    "How soon must the most serious kind of incident be acknowledged?",
    (
        "What is the deadline for telling a client about a medium-severity incident that involved "
        "their data?"
    ),
    "Who talks to the client when there is an incident?",
)
ROUNDS: Final = 3  # server test: every question this many times; the first round is a warm-up
AB_ROUNDS: Final = 6  # schedule comparison: one warm-up round, then five interleaved ones
MAX_LOAD_SECONDS: Final = 2.0  # R-2: the warm-up loaded the model; a question must not again

# PLAN.md §6.5: targets until measured, for a warm server, a local knowledge base, no web.
TARGET_RETRIEVAL_P95_MS: Final = 400.0
TARGET_FIRST_TOKEN_P50_S: Final = 2.5
TARGET_FIRST_TOKEN_P95_S: Final = 4.0
TARGET_END_TO_END_P95_S: Final = 15.0

_MS: Final = 1000.0


def line(label: str, values: Sequence[float], unit: str) -> str:
    """One row of a table: how many readings, their median, 95th percentile and the slowest."""
    p50, p95, slowest = percentile(values, 50), percentile(values, 95), max(values)
    return (
        f"  {label:<26} n={len(values):<3} "
        f"p50 {p50:>9.1f}  p95 {p95:>9.1f}  max {slowest:>9.1f} {unit}"
    )


def versus(label: str, seen: float, target: float, unit: str) -> str:
    verdict = "within" if seen <= target else "OVER"
    return f"  {label:<34} seen {seen:>8.1f} {unit}  target {target:>6.1f} {unit}  -> {verdict}"


def seed_models(settings: Settings) -> None:
    """Reuse the reranker weights of the user's own data folder, so the test downloads nothing."""
    models = Path("data") / "models"
    if models.is_dir():
        shutil.copytree(models, settings.data_dir / "models", dirs_exist_ok=True)


# --- 1. the answer path, through the server -------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """The real application over a temporary data folder, started, with the golden corpus ingested."""
    corpus = CORPUS.resolve()
    root = tmp_path_factory.mktemp("slo-smoke")
    settings = Settings(data_dir=root / "data", allowed_roots=str(corpus))
    seed_models(settings)
    perimeter = Perimeter.from_settings(settings)
    headers = {"Authorization": f"Bearer {perimeter.api_key}"}
    app = create_app(settings, perimeter=perimeter)
    with TestClient(app, base_url="http://127.0.0.1:8000", headers=headers) as client:
        accepted = client.post("/v1/kb/ingest", json={"paths": [str(corpus)]})
        assert accepted.status_code == 202
        job = wait_for(client, accepted.json()["id"], timeout_s=300)
        assert job["state"] == "done", job
        assert not job["result"]["failed"], job
        yield client


def ask_through_the_route(client: TestClient, question: str, **body: Any) -> tuple[Any, float]:
    """One question on a fresh thread: the ``done`` event and the seconds the whole turn took."""
    thread = client.post("/v1/threads").json()["thread_id"]
    started = time.perf_counter()
    response = client.post(f"/v1/threads/{thread}/messages", json={"question": question, **body})
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    events = parse_text(response.text)
    assert events[-1].event == "done", events[-1].data
    return events, elapsed


@pytest.mark.live
def test_the_answer_path_against_the_latency_targets(server: TestClient) -> None:
    health = server.get("/v1/health").json()
    print(f"\nhealth: {health}")
    assert health["chat_ready"] is True  # the lifespan warmed the models up

    # The steps a person watching would see, in order: the staged search is on by default.
    events, _ = ask_through_the_route(server, QUESTIONS[0], steps=True)
    stages = [e.data["stage"] for e in events if e.event == "status"]
    print(f"steps announced: {stages}")
    assert stages == ["searching", "ranking", "writing"]

    retrieval: dict[str, list[float]] = {}
    first_token: list[float] = []
    to_first_token: list[float] = []
    whole_turn: list[float] = []
    loads = 0
    runs = 0
    for round_number in range(ROUNDS):
        for question in QUESTIONS:
            events, elapsed = ask_through_the_route(server, question)
            done = events[-1].data
            if round_number == 0:
                continue  # the first round warms the embedder client, the reranker and the caches
            runs += 1
            timings = done["retrieval_ms"]
            assert {"dense", "lexical", "fusion", "rerank", "total"} <= set(timings), timings
            for stage, milliseconds in timings.items():
                retrieval.setdefault(stage, []).append(milliseconds)
            generation = done["generation"]
            if "first_token_seconds" in generation:
                first_token.append(generation["first_token_seconds"])
                to_first_token.append(timings["total"] / _MS + generation["first_token_seconds"])
            whole_turn.append(elapsed)
            assert generation.get("load_seconds", 0.0) < MAX_LOAD_SECONDS
            if generation.get("load_seconds", 0.0) >= 0.5:
                loads += 1

    model = health["chat_model"]
    print(f"\nthe answer path on {model} (proto profile: a plumbing check, not an SLO result)")
    print(f"{runs} warm turns over {len(QUESTIONS)} questions; the 95th percentile of so few runs")
    print("is close to the slowest one.\n")
    print("retrieval, by stage:")
    for stage in ("dense", "lexical", "fusion", "rerank", "total"):
        print(line(stage, retrieval[stage], "ms"))
    print("\nthe model and the whole turn:")
    print(line("model first token", first_token, "s"))
    print(line("question to first token", to_first_token, "s"))
    print(line("whole turn (HTTP)", whole_turn, "s"))
    print(f"  model loads during questions: {loads} of {runs} (rule R-2 wants 0)")
    print("\nagainst PLAN.md section 6.5 (targets for the target models; printed, not asserted):")
    total = retrieval["total"]
    print(versus("retrieval p95", percentile(total, 95), TARGET_RETRIEVAL_P95_MS, "ms"))
    seen_p50 = percentile(to_first_token, 50)
    print(versus("question to first token p50", seen_p50, TARGET_FIRST_TOKEN_P50_S, "s"))
    seen_p95 = percentile(to_first_token, 95)
    print(versus("question to first token p95", seen_p95, TARGET_FIRST_TOKEN_P95_S, "s"))
    print(versus("whole turn p95", percentile(whole_turn, 95), TARGET_END_TO_END_P95_S, "s"))
    assert runs == (ROUNDS - 1) * len(QUESTIONS)
    assert first_token, "the model never reported a time to first token"


# --- 2. the two search schedules, side by side ---------------------------------------------------


def context_with(retriever: Retriever, scorer: Any) -> AgentContext:
    return AgentContext(
        retriever=retriever,
        llm=ScriptedChat("The answer is in the sources [1]."),  # instant: only the search is timed
        model="instant-fake",
        top_k=5,
        context_budget_tokens=3000,
        citation_scorer=scorer,
    )


@pytest.mark.live
def test_parallel_search_against_sequential_search(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    seed_models(settings)
    files, _ = collect_files([CORPUS])
    with Workspace.open(settings) as workspace:
        report = Ingestor(workspace.store, workspace.dense, workspace.embedder).ingest(files)
        print(f"\ningest: {report.summary()}")
        sequential = build_retriever(SearchMode.HYBRID_RERANK, workspace, settings)
        assert isinstance(sequential, Reranker)
        scorer = sequential.scorer  # one cross-encoder for both, so neither pays a second load
        staged = StagedSearch(
            DenseRetriever(workspace.store, workspace.dense, workspace.embedder),
            LexicalRetriever(workspace.store),
            workspace.store.get_chunks,
            scorer,
            depth=settings.rerank_depth,
        )
        graph = build_graph()
        contexts = {
            "sequential": context_with(sequential, scorer),
            "parallel": context_with(staged, scorer),
        }
        wall: dict[str, list[float]] = {"sequential": [], "parallel": []}
        retrieval_total: dict[str, list[float]] = {"sequential": [], "parallel": []}
        dense_ms: list[float] = []
        lexical_ms: list[float] = []
        for round_number in range(AB_ROUNDS):
            for question in QUESTIONS:
                # Alternate which schedule goes first, so neither one always has the warmer caches.
                order = ["sequential", "parallel"]
                if round_number % 2:
                    order.reverse()
                sources: dict[str, list[str]] = {}
                for name in order:
                    started = time.perf_counter()
                    final = graph.invoke({"question": question}, context=contexts[name])
                    elapsed = (time.perf_counter() - started) * _MS
                    sources[name] = [source["chunk_id"] for source in final["sources"]]
                    if round_number == 0:
                        continue  # the warm-up round: loads the embedder client and the reranker
                    wall[name].append(elapsed)
                    retrieval_total[name].append(final["retrieval_ms"]["total"])
                    if name == "parallel":
                        dense_ms.append(final["retrieval_ms"]["dense"])
                        lexical_ms.append(final["retrieval_ms"]["lexical"])
                # The schedule must not change what is found.
                assert sources["parallel"] == sources["sequential"], question

    print("\nthe two schedules through the answer graph (an instant fake in place of the model)")
    print("embedder and reranker on the CPU; interleaved runs; the first round is not counted\n")
    for name in ("sequential", "parallel"):
        print(f"{name}:")
        print(line("graph, whole turn", wall[name], "ms"))
        print(line("retrieval, reported", retrieval_total[name], "ms"))
    saving_at_most = [min(d, x) for d, x in zip(dense_ms, lexical_ms, strict=True)]
    print("\nstage times of the parallel runs (the two searches overlap):")
    print(line("dense search", dense_ms, "ms"))
    print(line("lexical search", lexical_ms, "ms"))
    print(line("at most saved (the shorter)", saving_at_most, "ms"))
    seq_p50 = percentile(wall["sequential"], 50)
    par_p50 = percentile(wall["parallel"], 50)
    print(
        f"\nwhole-turn p50: sequential {seq_p50:.1f} ms, parallel {par_p50:.1f} ms, "
        f"difference {seq_p50 - par_p50:+.1f} ms (positive = parallel was faster)."
    )
    print("The difference includes the cost of the extra steps (a thread hop each); a few")
    print("milliseconds either way is inside the run-to-run noise of so few runs.")
    assert len(wall["parallel"]) == (AB_ROUNDS - 1) * len(QUESTIONS)
