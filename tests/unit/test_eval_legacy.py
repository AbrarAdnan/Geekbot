"""Tests for rung R0, v1's retrieval rebuilt inside v2 (step S2.6): its chunks and its query."""

from pathlib import Path

import pytest
from evalkit import DIM, HashEmbeddings

from geekbot.evals.ablation import LEGACY_RAW_L2
from geekbot.evals.indexing import EmbedderChoice, build_index
from geekbot.evals.legacy import (
    LEGACY_CHUNK_CHARS,
    LEGACY_OVERLAP_CHARS,
    LegacyMultiQueryRetriever,
    legacy_chunks,
    legacy_queries,
    split_legacy_text,
)
from geekbot.index.embedder import EmbedderSpec
from geekbot.ingest.loaders.model import SEGMENT_SEPARATOR, LoadedDocument, Segment

SPEC = EmbedderSpec(
    model="fake",
    digest="d",
    document_prefix="",
    query_prefix="",
    dimension=DIM,
    with_header=False,
)


def document(*pages: str, name: str = "doc.txt") -> LoadedDocument:
    return LoadedDocument(
        path=Path("/legacy") / name,
        kind="text",
        title=name,
        segments=tuple(Segment(text=page, page=number) for number, page in enumerate(pages, 1)),
    )


def test_v1s_five_template_queries_are_reproduced_word_for_word() -> None:
    assert legacy_queries("how long") == [
        "how long",
        "Explain: how long",
        "Details about: how long",
        "What is the background of: how long?",
        "Summarize how long in simple terms.",
    ]


def test_the_splitter_uses_v1s_sizes() -> None:
    text = "\n\n".join(f"Paragraph {n} " + "word " * 40 for n in range(12))
    pieces = split_legacy_text(text)
    assert len(pieces) > 3
    assert all(len(piece) <= LEGACY_CHUNK_CHARS for piece in pieces)
    assert max(len(piece) for piece in pieces) > LEGACY_CHUNK_CHARS // 2  # they are not tiny
    # neighbouring pieces share text: v1 overlapped its chunks
    shared = [pieces[n][-LEGACY_OVERLAP_CHARS:].split()[-1] in pieces[n + 1] for n in range(3)]
    assert all(shared)


def test_a_short_text_is_one_chunk_and_an_empty_one_is_none() -> None:
    assert split_legacy_text("a short note") == ["a short note"]
    assert split_legacy_text("") == []


def test_legacy_chunks_keep_pages_order_and_exact_offsets() -> None:
    first = "\n\n".join(f"First page paragraph {n}. " + "alpha " * 60 for n in range(6))
    second = "Second page, short."
    doc = document(first, second)
    drafts = legacy_chunks(doc)
    assert [draft.ordinal for draft in drafts] == list(range(len(drafts)))
    assert {draft.page for draft in drafts} == {1, 2}
    assert drafts[-1].page == 2
    assert all(draft.header == "" and draft.section_path == () for draft in drafts)
    for draft in drafts:
        assert doc.text[draft.char_start : draft.char_end] == draft.text
    assert drafts[-1].char_start == len(first) + len(SEGMENT_SEPARATOR)


def test_a_piece_that_cannot_be_found_is_placed_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geekbot.evals.legacy.split_legacy_text", lambda text: ["not in the text"])
    drafts = legacy_chunks(document("something else entirely"))
    assert drafts[0].char_start == 0
    assert drafts[0].char_end == len("not in the text")


# --- the retriever ---------------------------------------------------------------------------


def build(tmp_path: Path, embeddings: HashEmbeddings) -> object:
    docs = [
        document("The zebra runs across the plain at dawn. " * 3, name="zebra.txt"),
        document("Quartz is a hard mineral found in granite. " * 3, name="quartz.txt"),
        document("Bread rises when the yeast is warm enough. " * 3, name="bread.txt"),
        document("A river carries water from the hills. " * 3, name="river.txt"),
    ]
    return build_index(
        docs,
        LEGACY_RAW_L2,
        EmbedderChoice("fake", "fake"),
        tmp_path / "index",
        tmp_path / "cache.db",
        embeddings=embeddings,
        spec=SPEC,
    )


def test_the_retriever_searches_five_queries_dedupes_and_resorts_by_cosine(tmp_path: Path) -> None:
    embeddings = HashEmbeddings()
    built = build(tmp_path, embeddings)
    with built as index:  # type: ignore[attr-defined]
        embeddings.calls = 0
        retriever = LegacyMultiQueryRetriever(index.store, index.dense, embeddings)
        result = retriever.search("where does the zebra run", 10)
        chunk_ids = [scored.chunk.chunk_id for scored in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # duplicates dropped
        assert result.chunks[0].chunk.title == "zebra.txt"
        scores = [scored.score for scored in result.chunks]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= score <= 1.0 for score in scores)  # cosines, not L2 distances
        # 5 template queries + the question again + one embedding per surviving candidate
        assert embeddings.calls == 5 + 1 + len(chunk_ids)
        assert set(result.timings_ms) == {"embed_queries", "vector_search", "resort", "total"}


def test_k_cuts_the_resorted_list_and_bad_input_is_refused(tmp_path: Path) -> None:
    embeddings = HashEmbeddings()
    with build(tmp_path, embeddings) as index:  # type: ignore[attr-defined]
        retriever = LegacyMultiQueryRetriever(index.store, index.dense, embeddings, per_query=2)
        assert len(retriever.search("river water", 2).chunks) == 2
        assert retriever.search("   ", 5).chunks == ()
        with pytest.raises(ValueError, match="k must"):
            retriever.search("river", 0)
        with pytest.raises(ValueError, match="per_query"):
            LegacyMultiQueryRetriever(index.store, index.dense, embeddings, per_query=0)
