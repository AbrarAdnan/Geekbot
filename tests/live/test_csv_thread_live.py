"""Live test for step C4: a CSV conversation through the answer graph, with the real CSV model.

Needs Ollama running with the active profile's chat and CSV models pulled (``gemma3:4b`` and
``qwen2.5-coder:3b`` on the default ``proto`` profile). It swaps the models on the **GPU** (the chat
model is unloaded, the CSV model loaded), so run it when Gigalogy is not using the GPU:
``pytest -m live -s tests/live/test_csv_thread_live.py``. Deselected by default.

Two turns of one thread on the small sales file the unit tests use: a question, then a fragment that
only makes sense with the first. **It is a plumbing check, not a quality result:** it asserts that the
pieces fit together (the swap, the package answering through the graph, the real model rewriting a
follow-up in the shape the schema asks for, the thread being saved with the CSV taint) and only shows
whether the answers are right.
"""

import asyncio
from pathlib import Path

import pytest

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.render import format_answer
from geekbot.agent.runner import read_thread, run_turn
from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.config import Settings
from geekbot.csv_mode.adapter import CsvAnalyst
from geekbot.llm.chat import make_structured_model
from geekbot.llm.models import OllamaClient, swap_to_mode

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sales_small.csv"
FIRST = "Which region sells the most?"
FOLLOW_UP = "And what is its total revenue?"
MAX_LOAD_SECONDS = 2.0  # R-2: the swap loaded the model; a question should not load it again


@pytest.mark.live
def test_a_csv_conversation_swaps_models_answers_and_rewrites_a_follow_up(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    model = settings.resolved_csv_model
    threads_db = settings.data_dir / "threads.db"
    with OllamaClient(settings.ollama_host) as client:
        swap = swap_to_mode(client, settings, "csv")
    print(f"\n{model} warm in {swap.warm_up_s:.1f} s (unloaded: {list(swap.unloaded)})")

    with CsvAnalyst.open(FIXTURE, settings) as analyst:
        context = AgentContext(
            retriever=UnavailableRetriever(),
            llm=UnavailableChat(),
            model=model,
            top_k=1,
            context_budget_tokens=settings.context_budget_tokens,
            condenser=make_structured_model(
                settings,
                CondenseOutput,
                num_predict=settings.condense_num_predict,
                model=model,
                num_ctx=settings.csv_num_ctx,
            ),
            csv=analyst,
        )
        first = asyncio.run(
            run_turn(
                FIRST,
                context=context,
                thread="csv-live",
                threads_db=threads_db,
                mode="csv",
                csv_file=FIXTURE.name,
            )
        ).final
        second = asyncio.run(
            run_turn(
                FOLLOW_UP,
                context=context,
                thread="csv-live",
                threads_db=threads_db,
                mode="csv",
                csv_file=FIXTURE.name,
            )
        ).final

    print(f"\n--- turn 1\n{format_answer(first, model=model, thread='csv-live')}")
    print(f"\n--- turn 2\n{format_answer(second, model=model, thread='csv-live')}")
    print(f"\nrewritten as: {second['standalone_question']!r}")

    assert first["mode"] == "csv"
    assert first["answer"].strip()
    assert first["csv_status"] in {"answered", "abstained", "needs_clarification", "failed"}
    assert first["condense_seconds"] == 0.0  # the first turn makes no rewrite call
    assert first["generation"]["load_seconds"] < MAX_LOAD_SECONDS
    assert second["condense_seconds"] > 0  # the follow-up was rewritten by the real CSV model
    assert second["condense_error"] == ""  # in the shape of the schema
    assert second["standalone_question"].strip()
    assert second["generation"]["load_seconds"] < MAX_LOAD_SECONDS  # the rewrite loaded nothing
    assert second["thread_taint"] == ["TOOL", "USER"]
    saved = asyncio.run(read_thread(threads_db, "csv-live"))
    assert saved is not None
    assert (saved["mode"], saved["csv_file"], saved["turn"]) == ("csv", FIXTURE.name, 2)
    # Shown, not asserted: is the rewrite the full question, and is the answer right?
    print(f"mentions region: {'region' in second['standalone_question'].lower()}")
