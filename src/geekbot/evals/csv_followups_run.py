"""Running the CSV follow-up evaluation: the real graph in CSV mode, the real package, the real model
(step C6, PLAN.md §9). The rules and the report are in ``csv_followups``; the expected results are in
``csv_followups_gold``.

For every item and seed set two things are done with one loaded table:

1. **The conversation.** A fresh thread of the answer graph in CSV mode (the product's own graph, the
   product's own ``condense`` with the CSV model, a recording wrapper around the package so that the
   result of the last question can be graded): the first question, then the follow-up. What the package
   is finally asked is the rewrite.
2. **The oracle.** The hand-written standalone question, asked straight to the package.

Both answers are graded by Gigalogy's grader against the result pandas computes from the CSV. The seed
set shifts the package's sampling seeds by 10,000 per set (as the C3 reproduction does), so the three
runs of an item are three different draws of a model that varies from run to run.

Only the CSV model runs (on the GPU); the datasets are Gigalogy's, read from the folder given and never
copied (the C3 rule). The result file holds ids, verdicts and the rewrites, not a row of a dataset.
"""

import json
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from geekbot.config import Settings
from geekbot.csv_mode.adapter import CsvAnalyst, CsvAnswer
from geekbot.csv_mode.grading import grade
from geekbot.evals.csv_followups import (
    DATASETS,
    FollowUpItem,
    FollowUpRun,
    format_followups_report,
    note_category,
    run_followups,
    summarize_followups,
)

__all__ = ["RecordingCsv", "run_csv_followups"]

SEED_OFFSET_STRIDE: Final = 10_000  # as the C3 reproduction (Gigalogy's harness)
_CSV_PATH: Final = {"sales": "data/sales/sales.csv", "hospital": "data/hospital/hospital.csv"}
Log = Callable[[str], None]


class RecordingCsv:
    """The graph's CSV tool: asks the package and keeps the last answer so it can be graded.

    ``seed_offset`` moves the package's sampling seeds (the seed set of the run).
    """

    def __init__(self, analyst: CsvAnalyst, seed_offset: int) -> None:
        self._analyst = analyst
        self._seed_offset = seed_offset
        self.last: CsvAnswer | None = None

    def ask(self, question: str) -> CsvAnswer:
        answer = self._analyst.ask(question, seed_offset=self._seed_offset)
        self.last = answer
        return answer


def _csv_graph() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    from geekbot.agent.graph import build_graph

    return build_graph(checkpointer=InMemorySaver())


def run_csv_followups(
    settings: Settings,
    items: Sequence[FollowUpItem],
    *,
    source: Path,
    name: str,
    out_dir: Path,
    runs: int,
    log: Log,
) -> str:
    """Run every item ``runs`` times, write ``csv-followups.json`` and the report; return the report."""
    import pandas as pd  # type: ignore[import-untyped, unused-ignore]

    from geekbot.agent.condense import CondenseOutput
    from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
    from geekbot.evals.csv_followups_gold import RECIPES, gold_entry
    from geekbot.llm.chat import make_structured_model
    from geekbot.llm.models import OllamaClient, swap_to_mode

    model = settings.resolved_csv_model
    paths = {name_: source / _CSV_PATH[name_] for name_ in DATASETS}
    for dataset, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{path} not found; point --source at the Gigalogy folder ({dataset})")
    frames = {dataset: pd.read_csv(path) for dataset, path in paths.items()}
    gold = {item.id: gold_entry(RECIPES[item.gold], frames[item.dataset]) for item in items}

    graph = _csv_graph()
    condenser = make_structured_model(
        settings,
        CondenseOutput,
        num_predict=settings.condense_num_predict,
        model=model,
        num_ctx=settings.csv_num_ctx,
    )
    with OllamaClient(settings.ollama_host) as client:
        swap = swap_to_mode(client, settings, "csv")
    log(f"{model} ready ({swap.warm_up_s:.1f} s to load)")

    with ExitStack() as stack:
        analysts = {
            dataset: stack.enter_context(CsvAnalyst.open(path, settings))
            for dataset, path in paths.items()
        }

        def run_item(item: FollowUpItem, seed_set: int) -> FollowUpRun:
            started = time.perf_counter()
            analyst = analysts[item.dataset]
            offset = seed_set * SEED_OFFSET_STRIDE
            recorder = RecordingCsv(analyst, offset)
            context = AgentContext(
                retriever=UnavailableRetriever(),
                llm=UnavailableChat(),
                model=model,
                top_k=1,
                context_budget_tokens=settings.context_budget_tokens,
                condenser=condenser,
                history_budget_tokens=settings.history_budget_tokens,
                csv=recorder,
            )
            config = RunnableConfig(configurable={"thread_id": f"{item.id}-{seed_set}"})
            final: dict[str, Any] = {}
            for number, turn in enumerate(item.turns):
                turn_input: dict[str, Any] = {"question": turn}
                if number == 0:
                    turn_input.update(
                        requested_mode="csv", csv_file=Path(_CSV_PATH[item.dataset]).name
                    )
                final = dict(graph.invoke(turn_input, config, context=context))
            last = recorder.last
            if last is None:
                raise RuntimeError(f"{item.id}: the package was never asked")
            conversation_ok, conversation_note = grade(last, gold[item.id])
            oracle = analyst.ask(item.standalone, seed_offset=offset)
            oracle_ok, oracle_note = grade(oracle, gold[item.id])
            asked = str(final.get("standalone_question", item.follow_up))
            return FollowUpRun(
                item_id=item.id,
                dataset=item.dataset,
                seed_set=seed_set,
                conversation_correct=conversation_ok,
                conversation_note=note_category(conversation_note),
                oracle_correct=oracle_ok,
                oracle_note=note_category(oracle_note),
                rewrite=asked,
                rewrite_unchanged=asked.strip() == item.follow_up.strip(),
                rewrite_error=str(final.get("condense_error", "")),
                seconds=time.perf_counter() - started,
            )

        def show(result: FollowUpRun) -> None:
            log(
                f"  {result.item_id} seed {result.seed_set}: conversation "
                f"{'ok' if result.conversation_correct else 'WRONG'}, oracle "
                f"{'ok' if result.oracle_correct else 'WRONG'}  asked: {result.rewrite[:70]!r}"
            )

        results = run_followups(items, run_item, runs=runs, on_run=show)

    summary = summarize_followups(items, results, runs_per_item=runs)
    meta = {
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "profile": settings.profile,
        "runs_per_item": runs,
        "items": len(items),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        **meta,
        "runs": [r.to_json() for r in results],
        "summary": {
            "solved_conversation": summary.solved_conversation,
            "solved_oracle": summary.solved_oracle,
            "rewrite_loss": list(summary.rewrite_loss),
            "package_failures": list(summary.package_failures),
            "gained": list(summary.gained),
            "unstable": list(summary.unstable),
            "rewrites_unchanged": summary.rewrites_unchanged,
            "rewrite_errors": summary.rewrite_errors,
        },
    }
    (out_dir / "csv-followups.json").write_text(
        json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = format_followups_report(meta, items, results, summary)
    (out_dir / "csv-followups.md").write_text(report, encoding="utf-8")
    return report
