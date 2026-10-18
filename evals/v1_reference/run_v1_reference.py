"""The one-time v1 reference run (CP2, PLAN.md §9, D-034).

What this is: the *real* v1 (the files of git tag ``v1.0``, checked out as a separate ``git worktree``
so step S4.4's removal of v1 changes nothing here) run on the golden corpus and the golden questions,
in its own conda environment (``geekbot-v1``: Python 3.10 and v1's ``requirements.txt``, because v1
needs LangChain 0.3). It records, per question, how long v1 took and which documents it offered the
model, and from that the document-level Recall@5, so that

- the speed and quality of the old system are *measured*, not remembered, and
- ablation rung R0 (v1 rebuilt inside v2, ``geekbot.evals.legacy``) can be checked against the real
  thing: if R0 is a faithful replica its Recall@5 on the same questions should be close to v1's.

Rules of the run, so the comparison is fair and v1's files stay untouched:

- The model is set at *run time* (``rag_pipeline.DEFAULT_MODEL = ...``), the same ``gemma4:12b`` that
  v2 uses, so differences in speed come from the pipeline and not from the model. Web search is off
  (``use_web=False``), matching v2's default.
- v1's loaders are used as v1 had them (``app.py``'s ``SUPPORTED_LOADERS``; ``app.py`` itself is a
  Streamlit app and cannot be imported, so the table is copied here). v1 has **no per-file error
  handling**: the first file it cannot load aborts a whole indexing walk (PLAN.md §5 finding 4). This
  script catches the error *per file* and records it instead, so the run can go on; that is a
  deviation from v1's behaviour, made on purpose and reported. A file that v1 cannot load is simply
  absent from v1's index, and every question whose answer lives there is a miss for v1.
- Indexing and querying are v1's own functions (``storage_utils.split_documents`` and
  ``add_to_chroma``; ``rag_pipeline.query_rag``). The index goes in a scratch folder.
- The time of the model call is measured by wrapping v1's ``Ollama`` class at run time; everything
  else in a question's time is retrieval and prompt building. v1 does not stream, so there is no
  time-to-first-token for v1: the speed number is the whole question.

Run from the root of the v1 worktree, in the ``geekbot-v1`` environment:

    python <path to this file> --corpus <geekbot>/evals/corpus --golden <geekbot>/evals/datasets/golden.jsonl \\
        --out <geekbot>/evals/results/v1-reference.json --workdir <scratch folder>

The script depends on nothing from geekbot v2 (only the standard library and v1's packages).
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

RETRIEVAL_TYPES = ("single_hop", "multi_hop")


def load_loaders() -> dict[str, Any]:
    """v1's extension -> loader table (copied from app.py, which cannot be imported)."""
    from langchain_community.document_loaders import (
        CSVLoader,
        JSONLoader,
        NotebookLoader,
        PyPDFLoader,
        PythonLoader,
        TextLoader,
        UnstructuredExcelLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
    )

    return {
        ".py": PythonLoader,
        ".pdf": PyPDFLoader,
        ".txt": lambda f: TextLoader(f, encoding="utf-8"),
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".html": UnstructuredHTMLLoader,
        ".ipynb": NotebookLoader,
        ".json": JSONLoader,
    }


def load_corpus(corpus: Path, loaders: dict[str, Any]) -> tuple[list[Any], dict[str, str]]:
    """Load every file with v1's loader; a file that raises is recorded, not fatal (see the docstring)."""
    from storage_utils import METADATA_SOURCE_KEY, normalize_source_path

    documents: list[Any] = []
    outcome: dict[str, str] = {}
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        relative = path.relative_to(corpus).as_posix()
        loader_class = loaders.get(path.suffix.lower())
        if loader_class is None:
            outcome[relative] = "no loader in v1"
            continue
        try:
            file_docs = loader_class(str(path)).load()
        except BaseException as error:  # noqa: BLE001 - v1 would have aborted; we record and go on
            outcome[relative] = f"{type(error).__name__}: {str(error)[:160]}"
            continue
        for doc in file_docs:
            doc.metadata.update(
                {
                    "file_name": path.name,
                    "file_type": path.suffix.lower(),
                    METADATA_SOURCE_KEY: normalize_source_path(str(path)),
                    "page": doc.metadata.get("page", 0),
                }
            )
        documents.extend(file_docs)
        outcome[relative] = f"ok ({len(file_docs)} documents)"
    return documents, outcome


def recall_at_5(offered: list[str], relevant: set[str]) -> float | None:
    """Document-level Recall@5 over file names; ``None`` when nothing is relevant (unanswerable)."""
    if not relevant:
        return None
    top = list(dict.fromkeys(offered))[:5]
    return len(relevant.intersection(top)) / len(relevant)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--workdir", type=Path, required=True, help="scratch folder for v1's Chroma index"
    )
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (a trial)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path.cwd()))  # the v1 worktree: rag_pipeline, storage_utils, ...
    import rag_pipeline
    from storage_utils import add_to_chroma, split_documents

    items = [
        json.loads(line) for line in args.golden.read_text("utf-8").splitlines() if line.strip()
    ]
    if args.limit:
        items = items[: args.limit]

    # the model is set at run time; v1's files are not edited
    rag_pipeline.DEFAULT_MODEL = args.model
    llm_seconds: list[float] = []
    original_invoke = rag_pipeline.Ollama.invoke

    def timed_invoke(self: Any, *a: Any, **k: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_invoke(self, *a, **k)
        finally:
            llm_seconds.append(time.perf_counter() - started)

    rag_pipeline.Ollama.invoke = timed_invoke

    documents, outcome = load_corpus(args.corpus, load_loaders())
    chunks = split_documents(documents)
    chroma = args.workdir / "chroma"
    started = time.perf_counter()
    added = add_to_chroma(chunks, chroma)
    index_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for item in items:
        llm_seconds.clear()
        started = time.perf_counter()
        answer, info = rag_pipeline.query_rag(item["question"], str(chroma), use_web=False)
        total = time.perf_counter() - started
        offered = [source["file"] for source in info["local_sources"]]
        relevant = {Path(s["path"]).name for s in item.get("sources", [])}
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "seconds": total,
                "model_seconds": sum(llm_seconds),
                "offered": offered,
                "recall_at_5": recall_at_5(offered, relevant)
                if item["type"] in RETRIEVAL_TYPES
                else None,
                "answer_chars": len(answer),
            }
        )
        print(
            f"{item['id']}: {total:.1f} s (model {sum(llm_seconds):.1f} s), offered {len(set(offered))} files"
        )

    scored = [r["recall_at_5"] for r in rows if r["recall_at_5"] is not None]
    result = {
        "what": "v1 (git tag v1.0) run once on the golden corpus; see run_v1_reference.py",
        "model": args.model,
        "embedder": "nomic-embed-text (v1's get_embedding_function)",
        "python": platform.python_version(),
        "web_search": False,
        "files": outcome,
        "chunks_indexed": added,
        "chunks_made": len(chunks),
        "index_seconds": index_seconds,
        "questions": rows,
        "recall_at_5_mean": sum(scored) / len(scored) if scored else None,
        "recall_at_5_questions": len(scored),
        "note": "multi-turn questions are asked as the follow-up alone: v1 has no rewrite step",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
