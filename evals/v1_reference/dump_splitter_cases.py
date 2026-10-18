"""Step 1 of the splitter check (CP2): make the texts and the chunks v2's rung R0 cuts from them.

Rung R0 cuts text with ``langchain-text-splitters`` 1.x (the only version that installs beside v2's
``langchain-core`` 1.x); v1 used 0.3.11. Run this in the ``geekbot`` environment, from the
repository root. It writes a JSON file of texts (every segment of the golden corpus as v2's loaders
read it, plus the first abstracts of SciFact and NFCorpus) and the chunks ``geekbot.evals.legacy``
makes of each. ``compare_splitter.py`` then cuts the same texts with v1's real splitter in the
``geekbot-v1`` environment and compares, character by character.

    python evals/v1_reference/dump_splitter_cases.py scratch/splitter-cases.json
"""

import json
import sys
from pathlib import Path

from geekbot.evals.beir import load_beir
from geekbot.evals.legacy import split_legacy_text
from geekbot.ingest.loaders import load_file
from geekbot.ingest.pipeline import collect_files

BEIR_ABSTRACTS = 300  # per dataset: long, varied text, the case the splitter actually meets


def main(out: Path) -> int:
    texts: list[tuple[str, str]] = []
    files, _ = collect_files([Path("evals/corpus")])
    for path in files:
        for number, segment in enumerate(load_file(path).segments):
            texts.append((f"golden:{path.name}:{number}", segment.text))
    for name in ("scifact", "nfcorpus"):
        dataset = load_beir(Path("data/beir") / name, "test")
        for doc_id, record in list(dataset.corpus.items())[:BEIR_ABSTRACTS]:
            texts.append((f"{name}:{doc_id}", record.text))
    cases = [{"id": i, "text": t, "chunks": split_legacy_text(t)} for i, t in texts if t.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} texts, {sum(len(c['chunks']) for c in cases)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
