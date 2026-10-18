"""Live test for steps S1.6 to S2.2: ingest a tiny corpus and search it with the real embedder.

Needs Ollama running with ``nomic-embed-text`` pulled. The embedder runs on the **CPU**
(``num_gpu=0``), so the GPU is not used, but it does use CPU time: run it when the machine is
quiet, with ``pytest -m live -s tests/live/test_search_live.py``. Deselected by default.

It is the first end-to-end check of the whole Phase 1 chain on real vectors, and it prints the
numbers (which stage took how long) so they can be quoted, with the model named.
"""

from pathlib import Path

import pytest

from geekbot.config import Settings
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import Ingestor, collect_files
from geekbot.retrieval import format_result
from geekbot.retrieval.dense import DenseRetriever
from geekbot.retrieval.hybrid import HybridRetriever
from geekbot.retrieval.lexical import LexicalRetriever

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


@pytest.mark.live
def test_a_real_ingest_and_search_find_the_right_document(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in CORPUS.items():
        (folder / name).write_text(text, encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data")
    files, _ = collect_files([folder])

    with Workspace.open(settings) as workspace:
        ingestor = Ingestor(workspace.store, workspace.dense, workspace.embedder)
        first = ingestor.ingest(files)
        second = ingestor.ingest(files)
        print(f"\nembedder {workspace.embedder.spec.model} on the CPU")
        print(f"first ingest : {first.summary()}")
        print(f"second ingest: {second.summary()}")
        assert second.embedded == 0  # the Gate A check, on a real embedder

        retriever = DenseRetriever(workspace.store, workspace.dense, workspace.embedder)
        hybrid = HybridRetriever(retriever, LexicalRetriever(workspace.store))
        for question, expected in [
            ("how do I keep a sourdough starter alive?", "sourdough.md"),
            ("which rivers flow into the Bay of Bengal?", "rivers.md"),
            ("how often should I change my password?", "passwords.md"),
        ]:
            result = retriever.search(question, 3)
            print(f"\n{question}\ndense:\n{format_result(result)}")
            assert result.chunks[0].chunk.title == expected
            fused = hybrid.search(question, 3)
            print(f"hybrid:\n{format_result(fused)}")
            assert fused.chunks[0].chunk.title == expected
