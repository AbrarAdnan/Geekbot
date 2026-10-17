"""Live test for step S2.3: the real cross-encoder on the CPU (needs network once, for the weights).

Run on purpose with ``pytest -m live -s tests/live/test_rerank_live.py``. The first run downloads
the model (about 80 MB for the default MiniLM) from Hugging Face into a temporary folder, so
it needs network access and some patience; later runs of the real command reuse
``<data_dir>/models``. It uses the CPU only. Deselected by default.

It answers two questions the unit tests cannot, and prints the numbers so they can be quoted with
the model named: does the real model put the passage that answers the question first, and how long
does one scoring call take at a few depths on this machine, now.
"""

from pathlib import Path

import pytest

from geekbot.config import Settings
from geekbot.retrieval.rerank import FastembedScorer
from geekbot.retrieval.rerank_timing import format_depth_timings, measure_rerank_depths

QUESTION = "how long should I keep a sourdough starter warm before baking?"
PASSAGES = [
    "The Ganges and the Brahmaputra meet in Bangladesh and flow into the Bay of Bengal.",
    "Rotate every password at least every ninety days and never reuse one across services.",
    "Feed a sourdough starter daily and keep it warm; it is ready to bake with once it doubles.",
    "Zebras have black and white stripes and live in herds on the savanna.",
    "Quartz is a hard mineral found in granite and used in watches.",
]


@pytest.mark.live
def test_the_real_reranker_puts_the_answering_passage_first(tmp_path: Path) -> None:
    settings = Settings()
    scorer = FastembedScorer(settings.rerank_model, cache_dir=tmp_path / "models")
    scores = scorer.score(QUESTION, PASSAGES)
    print(f"\nreranker {settings.rerank_model} on the CPU")
    for score, passage in sorted(zip(scores, PASSAGES, strict=True), reverse=True):
        print(f"{score:8.3f}  {passage[:70]}")
    assert scores.index(max(scores)) == 2

    timings = measure_rerank_depths(scorer, [QUESTION], PASSAGES * 6, [5, 10, 20, 30], repeats=5)
    print(format_depth_timings(settings.rerank_model, timings))
    assert all(t.p50_ms > 0 for t in timings)
