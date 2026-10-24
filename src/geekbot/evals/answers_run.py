"""Running the answer evaluation: the models, the files, the phases (step S5.2, PLAN.md §9).

``answers.py`` holds the checks, ``judge.py`` the judge, ``labelling.py`` the labels and κ. This file
runs them in the order the plan fixes: **generate all answers with the chat model, then unload it and
judge all of them with the judge model, then report** (phased by model: two chat models are never on
the GPU together, R-4, and nothing swaps between questions). Each phase writes its file before the
next starts and every phase resumes where it stopped, so an hour of GPU time is never lost to one
crash:

- ``answers.json``: every golden answer with the sources it was shown (generate)
- ``judge-cache.json`` and ``judge.json``: the verdicts and their summary (judge)
- ``answers-report.md``: the deterministic checks, the judge, and κ against the human labels (report)

Nothing here writes a label. The labels are a person's (D-097); the report reads them if the file
exists and says "no verdict yet" if not.
"""

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from geekbot.config import Settings
from geekbot.evals.answers import (
    AnswerRecord,
    format_answers_report,
    generate_answers,
    load_answers,
    save_answers,
    score_answer,
    summarize_scores,
)
from geekbot.evals.datasets import GoldenItem, assign_splits, load_golden
from geekbot.evals.judge import (
    JudgeOutput,
    JudgeVerdict,
    VerdictCache,
    check_judge_family,
    kappa_against_labels,
    run_judge,
    summarize_judge,
)
from geekbot.labelling import LabelStore

__all__ = ["PHASES", "AnswerRun", "GraphAnswerer", "run_answers"]

PHASES: Final = ("generate", "judge", "report")
Log = Callable[[str], None]


@dataclass(frozen=True)
class AnswerRun:
    """Where a run's files are and what it was called."""

    name: str
    results_dir: Path
    labels_file: Path

    @property
    def folder(self) -> Path:
        return self.results_dir / self.name

    @property
    def answers(self) -> Path:
        return self.folder / "answers.json"

    @property
    def judge_cache(self) -> Path:
        return self.folder / "judge-cache.json"

    @property
    def judge_file(self) -> Path:
        return self.folder / "judge.json"

    @property
    def report(self) -> Path:
        return self.folder / "answers-report.md"

    def labels(self) -> LabelStore:
        return LabelStore(self.labels_file, self.name)


class GraphAnswerer:
    """Runs one golden item through the product's answer graph, as a conversation (live-tested)."""

    def __init__(self, settings: Settings, retriever: Any, condenser: Any, scorer: Any) -> None:
        from geekbot.agent.graph import build_graph

        self._settings = settings
        self._retriever = retriever
        self._condenser = condenser
        self._scorer = scorer
        self._graph = build_graph(checkpointer=_memory_saver())

    def ask(self, item: GoldenItem) -> dict[str, Any]:
        """The final state of the item's last turn. Earlier turns of a multi-turn item are real turns."""
        from geekbot.agent.runtime import AgentContext
        from geekbot.llm.chat import make_chat_model

        settings = self._settings
        context = AgentContext(
            retriever=self._retriever,
            llm=make_chat_model(settings),
            model=settings.resolved_chat_model,
            top_k=settings.answer_top_k,
            context_budget_tokens=settings.context_budget_tokens,
            condenser=self._condenser,
            history_budget_tokens=settings.history_budget_tokens,
            history_keep_messages=settings.history_keep_messages,
            citation_scorer=self._scorer,
            citation_weak_below=settings.cite_weak_below,
        )
        config = RunnableConfig(configurable={"thread_id": f"answers-{item.id}"})
        final: dict[str, Any] = {}
        for turn in (*item.earlier_turns, item.question):
            final = dict(self._graph.invoke({"question": turn}, config, context=context))
        return final


def _memory_saver() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def _digest(settings: Settings, model: str) -> str:
    """The digest Ollama reports for ``model`` (part of a cache key: a re-pulled model differs)."""
    from geekbot.index.embedder import OllamaEmbeddingsClient

    client = OllamaEmbeddingsClient(settings.ollama_host, model)
    try:
        return client.model_digest()
    finally:
        client.close()


def _ensure_ingested(settings: Settings, corpus: Path, log: Log) -> None:
    """Ingest the golden corpus into ``settings.data_dir`` if nothing is indexed there yet."""
    from geekbot.index.sqlite import DB_FILE_NAME
    from geekbot.index.workspace import Workspace
    from geekbot.ingest.pipeline import Ingestor, collect_files

    if (settings.data_dir / DB_FILE_NAME).is_file():
        return
    models = Path("data") / "models"
    target = settings.data_dir / "models"
    if models.is_dir() and not target.is_dir():
        shutil.copytree(models, target)  # the reranker's weights, so nothing is downloaded
    files, _ = collect_files([corpus])
    with Workspace.open(settings) as workspace:
        report = Ingestor(workspace.store, workspace.dense, workspace.embedder).ingest(files)
    log(f"ingest: {report.summary()}")


def _generate(
    run: AnswerRun,
    settings: Settings,
    items: Sequence[GoldenItem],
    splits: dict[str, str],
    corpus: Path,
    log: Log,
) -> list[AnswerRecord]:
    from geekbot.agent.condense import CondenseOutput
    from geekbot.index.workspace import Workspace
    from geekbot.llm.chat import make_structured_model
    from geekbot.llm.models import OllamaClient, swap_to_mode
    from geekbot.retrieval import SearchMode
    from geekbot.retrieval.factory import build_retriever, citation_scorer_of

    existing: dict[str, AnswerRecord] = {}
    if run.answers.is_file():
        _, previous = load_answers(run.answers)
        existing = {record.id: record for record in previous}
    _ensure_ingested(settings, corpus, log)
    model = settings.resolved_chat_model
    meta: dict[str, Any] = {
        "name": run.name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "profile": settings.profile,
        "top_k": settings.answer_top_k,
        "digest": _digest(settings, model),
    }
    with OllamaClient(settings.ollama_host) as client, Workspace.open(settings) as workspace:
        swap = swap_to_mode(client, settings, "chat")
        log(f"{model} ready ({swap.warm_up_s:.1f} s to load)")
        retriever = build_retriever(SearchMode.HYBRID_RERANK, workspace, settings)
        condenser = make_structured_model(
            settings, CondenseOutput, num_predict=settings.condense_num_predict
        )
        answerer = GraphAnswerer(settings, retriever, condenser, citation_scorer_of(retriever))
        done: list[AnswerRecord] = list(existing.values())

        def progress(record: AnswerRecord, reused: bool) -> None:
            if not reused:
                done.append(record)
                save_answers(run.answers, meta, _in_item_order(done, items))
            log(
                f"  {record.id:>6} {'(kept) ' if reused else ''}"
                f"{'refused ' if record.abstained else 'answered'} {record.answer[:70]!r}"
            )

        records = generate_answers(
            items,
            splits,
            answerer.ask,
            corpus=corpus,
            clock=time.perf_counter,
            existing=existing,
            on_record=progress,
        )
    save_answers(run.answers, meta, records)
    return records


def _in_item_order(
    records: Sequence[AnswerRecord], items: Sequence[GoldenItem]
) -> list[AnswerRecord]:
    order = {item.id: index for index, item in enumerate(items)}
    unique = {record.id: record for record in records}
    return sorted(unique.values(), key=lambda record: order.get(record.id, len(order)))


def _judge(
    run: AnswerRun,
    settings: Settings,
    judge_model: str,
    meta: dict[str, Any],
    records: list[AnswerRecord],
    log: Log,
) -> dict[str, JudgeVerdict]:
    from geekbot.llm.chat import make_structured_model
    from geekbot.llm.models import OllamaClient, model_is_pulled, swap_to

    generator = str(meta["model"])
    check_judge_family(generator, judge_model)  # before any model is loaded
    with OllamaClient(settings.ollama_host) as client:
        if not model_is_pulled(judge_model, client.list_models()):
            raise ValueError(
                f"the judge {judge_model} is not pulled; this run never pulls a model: name one that "
                "is (`ollama list`), or run `ollama pull` yourself"
            )
        swap = swap_to(
            client,
            judge_model,
            release=[generator],
            num_ctx=settings.chat_num_ctx,
            keep_alive=settings.keep_alive,
        )
        log(f"judge {judge_model} ready ({swap.warm_up_s:.1f} s to load; {generator} unloaded)")
        structured = make_structured_model(
            settings,
            JudgeOutput,
            num_predict=200,
            model=judge_model,
            num_ctx=settings.chat_num_ctx,
        )
        cache = VerdictCache.load(run.judge_cache)
        digest = _digest(settings, judge_model)

        def progress(record: AnswerRecord, verdict: JudgeVerdict, cached: bool) -> None:
            log(
                f"  {record.id:>6} {'(cached) ' if cached else ''}supported={verdict.supported} "
                f"answers={verdict.answers_question} {verdict.reason[:60]!r}"
            )

        verdicts = run_judge(
            records, structured, cache, model=judge_model, digest=digest, on_verdict=progress
        )
    summary = summarize_judge(records, verdicts, model=judge_model, generator=generator)
    body = {
        "name": run.name,
        "judge": summary,
        "verdicts": {
            item_id: {
                "supported": verdict.supported,
                "answers_question": verdict.answers_question,
                "reason": verdict.reason,
            }
            for item_id, verdict in verdicts.items()
        },
    }
    run.judge_file.write_text(
        json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return verdicts


def _report(run: AnswerRun, items: Sequence[GoldenItem], log: Log) -> str:
    meta, records = load_answers(run.answers)
    by_id = {item.id: item for item in items}
    scores = [score_answer(by_id[record.id], record) for record in records if record.id in by_id]
    summary = summarize_scores(scores)
    judge: dict[str, Any] | None = None
    kappa: dict[str, Any] | None = None
    if run.judge_file.is_file():
        saved = json.loads(run.judge_file.read_text(encoding="utf-8"))
        judge = saved["judge"]
        verdicts = {
            item_id: JudgeVerdict(
                item_id, bool(v["supported"]), bool(v["answers_question"]), str(v.get("reason", ""))
            )
            for item_id, v in saved["verdicts"].items()
        }
        kappa = kappa_against_labels(verdicts, run.labels().labels())
    text = format_answers_report(meta, summary, judge, kappa)
    run.report.write_text(text, encoding="utf-8")
    log(f"wrote {run.report}")
    return text


def run_answers(
    run: AnswerRun,
    settings: Settings,
    *,
    phases: Sequence[str],
    golden: Path,
    corpus: Path,
    judge_model: str,
    limit: int,
    log: Log,
) -> str | None:
    """Run the chosen phases in the plan's order; return the report text if the report phase ran."""
    items = load_golden(golden)
    splits: dict[str, str] = dict(assign_splits(items))
    chosen = items if limit == 0 else items[:limit]
    run.folder.mkdir(parents=True, exist_ok=True)
    records: list[AnswerRecord] | None = None
    meta: dict[str, Any] = {}
    if "generate" in phases:
        records = _generate(run, settings, chosen, splits, corpus, log)
    if "judge" in phases:
        if records is None:
            meta, records = load_answers(run.answers)
        else:
            meta, _ = load_answers(run.answers)
        _judge(run, settings, judge_model, meta, records, log)
    if "report" in phases:
        return _report(run, chosen, log)
    return None
