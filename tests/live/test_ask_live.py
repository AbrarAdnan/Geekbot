"""Live test for step S3.1: ask a question and get a grounded answer from the real chat model.

Needs Ollama running with ``nomic-embed-text`` and the active profile's chat model pulled
(``gemma3:4b`` on the default ``proto`` profile). It loads the chat model onto the **GPU**, so run it
when Gigalogy is not using the GPU: ``pytest -m live -s tests/live/test_ask_live.py``. Deselected by
default.

It builds its own three-document knowledge base in a temporary folder (the embedder runs on the CPU,
the reranker is not used, so nothing is downloaded), then runs one question through the whole chain:
swap to the chat model, retrieve, pack, generate. It prints what came back and the readings of the
run. **It is a plumbing check, not a quality result:** one question on three one-sentence documents says
whether the pieces fit together (does Ollama accept ``reasoning=False`` on this model, does the model
load once and stay loaded, is the answer grounded in the right document), not how good the answers are.
"""

from pathlib import Path

import pytest

from geekbot.agent.graph import build_graph
from geekbot.agent.render import format_answer
from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.llm.chat import make_chat_model
from geekbot.llm.models import OllamaClient, swap_to_mode
from geekbot.retrieval import SearchMode
from geekbot.retrieval.factory import build_retriever

CORPUS = {
    "sourdough.md": (
        "# Sourdough\n\nA sourdough starter is flour and water left to ferment until it bubbles. "
        "Feed it every day and keep it warm."
    ),
    "rivers.md": (
        "# Rivers of Bangladesh\n\nThe Ganges and the Brahmaputra meet in Bangladesh and flow "
        "together into the Bay of Bengal."
    ),
    "passwords.md": (
        "# Passwords\n\nRotate every password at least every ninety days and never reuse one "
        "across services."
    ),
}
MAX_LOAD_SECONDS = 2.0  # R-2: the warm-up loaded the model; a query should not load it again


@pytest.mark.live
def test_a_real_question_gets_a_grounded_answer_from_the_right_document(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data")
    files, _ = collect_files([docs])
    model = settings.resolved_chat_model

    with Workspace.open(settings) as workspace:
        ingestor = Ingestor(workspace.store, workspace.dense, workspace.embedder)
        assert not ingestor.ingest(files).has_failures
        with OllamaClient(settings.ollama_host) as client:
            swap = swap_to_mode(client, settings, "chat")
        print(f"\n{model} warm in {swap.warm_up_s:.1f} s (unloaded: {list(swap.unloaded)})")

        context = AgentContext(
            retriever=build_retriever(SearchMode.HYBRID, workspace, settings),
            llm=make_chat_model(settings),
            model=model,
            top_k=3,
            context_budget_tokens=settings.context_budget_tokens,
        )
        graph = build_graph()
        question = "How often should I rotate my passwords?"
        answered = graph.invoke({"question": question}, context=context)
        print(format_answer(answered, model=model))

        assert answered["sources"][0]["title"] == "passwords.md"
        assert answered["used"] >= 1
        text = answered["answer"].lower()
        assert "ninety" in text or "90" in text
        assert answered["generation"].get("load_seconds", 0.0) < MAX_LOAD_SECONDS

        # Not asserted, only shown: a question the documents cannot answer. What the model does
        # here is S3.5's concern (abstain); S3.1's prompt only asks it to say it could not find it.
        unanswerable = graph.invoke({"question": "What is the capital of France?"}, context=context)
        print(f"\nunanswerable question:\n{format_answer(unanswerable, model=model)}")
