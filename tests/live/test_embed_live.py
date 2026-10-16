"""Live test for step S1.5: the real embedder on the CPU (needs Ollama and ``nomic-embed-text``).

Run on purpose with ``pytest -m live -s tests/live/test_embed_live.py``. It loads the embedding
model on the **CPU** (``num_gpu=0``), so it does not touch the GPU, but it does use CPU time, so run
it when the machine is quiet. Deselected by default.

It answers three questions the plan left open, and prints the numbers so they can be quoted:

1. Are the vectors unit length? (Sets the R1 expectation: if they are, L2 and cosine rank alike.)
2. Does the prefixed embedder rank a related passage above an unrelated one?
3. Does ``estimate_tokens`` (the chunker's cheap count) stay at or above the model's real count?
   S1.2 left this comparison for S1.5.
"""

import math

import pytest

from geekbot.config import Settings
from geekbot.index.embedder import open_ollama_embedder
from geekbot.ingest.chunking import estimate_tokens

PASSAGES = {
    "bread": "Sourdough needs a starter: flour and water fermented for days until it bubbles.",
    "rivers": "The Brahmaputra and the Ganges meet in Bangladesh and flow together into the Bay of Bengal.",
}
SAMPLES = [
    "Short text.",
    PASSAGES["bread"],
    PASSAGES["rivers"] * 4,
    "def add(a: int, b: int) -> int:\n    return a + b\n" * 10,
]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.live
def test_the_real_embedder_on_the_cpu() -> None:
    prefixed, spec, client = open_ollama_embedder(Settings())
    with client:
        document_vectors = prefixed.embed_documents(list(PASSAGES.values()))
        query_vector = prefixed.embed_query("how do I keep a sourdough starter alive?")

        norms = [math.sqrt(_dot(v, v)) for v in (*document_vectors, query_vector)]
        print(f"\nembedder {spec.model} digest {spec.digest} dimension {spec.dimension}")
        print(f"vector norms: min {min(norms):.4f} max {max(norms):.4f}")
        assert all(math.isfinite(n) and n > 0 for n in norms)

        bread, rivers = (
            _dot(query_vector, v) / (math.sqrt(_dot(query_vector, query_vector)) * n)
            for v, n in zip(document_vectors, norms[:2], strict=True)
        )
        print(f"cosine to the query: bread {bread:.3f}, rivers {rivers:.3f}")
        assert bread > rivers

        for text in SAMPLES:
            real = client.prompt_tokens(text)
            estimate = estimate_tokens(text)
            print(f"tokens: real {real}, estimate {estimate}, ratio {estimate / real:.2f}")
            # estimate_tokens claims to err on the high side (S1.2); a lower count would mean
            # chunks can overshoot their budget.
            assert estimate >= real
