"""Tests for building the ablation's indexes (step S2.6): variants, reuse, shared cache, BEIR input."""

from pathlib import Path

import pytest
from evalkit import DIM, HashEmbeddings, write_tiny_beir

from geekbot.evals.ablation import LEGACY_RAW, LEGACY_RAW_L2, V2_HEADER, V2_PLAIN
from geekbot.evals.beir import load_beir
from geekbot.evals.indexing import (
    BuildStats,
    EmbedderChoice,
    IndexVariant,
    beir_doc_path,
    beir_documents,
    build_index,
)
from geekbot.index.embedder import EmbedderSpec
from geekbot.ingest.ids import doc_id_of
from geekbot.ingest.loaders.model import LoadedDocument, Segment

CHOICE = EmbedderChoice("fake", "fake")


class Recording(HashEmbeddings):
    """Remembers every text it embedded, so a test can see what the header mode sent."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed_documents(texts)


def spec_for(variant: IndexVariant, *, digest: str = "d") -> EmbedderSpec:
    return EmbedderSpec(
        model="fake",
        digest=digest,
        document_prefix="",
        query_prefix="",
        dimension=DIM,
        with_header=variant.header,
    )


def docs(count: int = 5) -> list[LoadedDocument]:
    return [
        LoadedDocument(
            path=Path("/ix") / f"d{n}.txt",
            kind="text",
            title=f"Title {n}",
            segments=(Segment(text=f"Body of document {n} mentions topic{n} twice: topic{n}."),),
        )
        for n in range(count)
    ]


def test_the_variant_key_names_every_dimension() -> None:
    assert LEGACY_RAW_L2.key == "legacy-raw-nohdr-l2"
    assert LEGACY_RAW.key == "legacy-raw-nohdr-cosine"
    assert V2_HEADER.key == "v2-pfx-hdr-cosine"
    assert len({v.key for v in (LEGACY_RAW_L2, LEGACY_RAW, V2_PLAIN, V2_HEADER)}) == 4


def test_build_stores_every_chunk_and_every_vector(tmp_path: Path) -> None:
    with build_index(
        docs(),
        V2_PLAIN,
        CHOICE,
        tmp_path / "i",
        tmp_path / "c.db",
        embeddings=HashEmbeddings(),
        spec=spec_for(V2_PLAIN),
    ) as built:
        stats = built.stats
        assert (stats.documents, stats.chunks, stats.embedded, stats.cache_hits) == (5, 5, 5, 0)
        assert stats.reused is False
        assert stats.chunks_per_s is None or stats.chunks_per_s > 0
        assert built.dense.count() == 5
        ready = built.store.list_documents("ready")
        assert len(ready) == 5
        assert {d.doc_id for d in ready} == {doc_id_of(Path("/ix") / f"d{n}.txt") for n in range(5)}


def test_a_finished_build_is_reused_without_embedding_again(tmp_path: Path) -> None:
    recorder = Recording()
    spec = spec_for(V2_PLAIN)
    args = (docs(), V2_PLAIN, CHOICE, tmp_path / "i", tmp_path / "c.db")
    build_index(*args, embeddings=recorder, spec=spec).close()
    seen = len(recorder.texts)
    with build_index(*args, embeddings=recorder, spec=spec) as again:
        assert again.stats.reused is True
        assert again.stats.chunks == 5
        assert again.dense.count() == 5
    assert len(recorder.texts) == seen


def test_another_embedder_or_another_corpus_rebuilds(tmp_path: Path) -> None:
    args = (V2_PLAIN, CHOICE, tmp_path / "i", tmp_path / "c.db")
    build_index(docs(), *args, embeddings=HashEmbeddings(), spec=spec_for(V2_PLAIN)).close()
    new_digest = spec_for(V2_PLAIN, digest="re-pulled")
    with build_index(docs(), *args, embeddings=HashEmbeddings(), spec=new_digest) as rebuilt:
        assert rebuilt.stats.reused is False  # the fingerprint changed
        assert rebuilt.dense.count() == 5
    with build_index(docs(3), *args, embeddings=HashEmbeddings(), spec=new_digest) as smaller:
        assert smaller.stats.reused is False  # a different number of documents
        assert smaller.dense.count() == 3


def test_variants_share_the_embedding_cache_so_r0_and_r1_embed_once(tmp_path: Path) -> None:
    cache = tmp_path / "shared.db"
    spec = spec_for(LEGACY_RAW)
    with build_index(
        docs(),
        LEGACY_RAW_L2,
        CHOICE,
        tmp_path / "l2",
        cache,
        embeddings=HashEmbeddings(),
        spec=spec,
    ) as first:
        assert first.stats.embedded > 0
    with build_index(
        docs(), LEGACY_RAW, CHOICE, tmp_path / "cos", cache, embeddings=HashEmbeddings(), spec=spec
    ) as second:
        assert second.stats.embedded == 0
        assert second.stats.cache_hits > 0
        assert second.stats.chunks_per_s is None  # nothing was embedded, so no rate


def test_the_header_variant_embeds_header_and_text_and_the_plain_one_only_text(
    tmp_path: Path,
) -> None:
    plain, headed = Recording(), Recording()
    build_index(
        docs(2),
        V2_PLAIN,
        CHOICE,
        tmp_path / "p",
        tmp_path / "pc.db",
        embeddings=plain,
        spec=spec_for(V2_PLAIN),
    ).close()
    build_index(
        docs(2),
        V2_HEADER,
        CHOICE,
        tmp_path / "h",
        tmp_path / "hc.db",
        embeddings=headed,
        spec=spec_for(V2_HEADER),
    ).close()
    assert all(not text.startswith("Title") for text in plain.texts)
    assert all(text.startswith("Title") for text in headed.texts)


def test_a_legacy_variant_cuts_with_v1s_chunker_and_a_v2_variant_with_ours(tmp_path: Path) -> None:
    long_text = "\n\n".join(f"Paragraph {n}. " + "filler " * 80 for n in range(10))
    document = LoadedDocument(Path("/ix/long.txt"), "text", "Long", (Segment(text=long_text),))
    doc_id = doc_id_of(document.path)
    with build_index(
        [document],
        LEGACY_RAW,
        CHOICE,
        tmp_path / "a",
        tmp_path / "a.db",
        embeddings=HashEmbeddings(),
        spec=spec_for(LEGACY_RAW),
    ) as legacy:
        legacy_sizes = [len(c.text) for c in legacy.store.document_chunks(doc_id)]
    with build_index(
        [document],
        V2_PLAIN,
        CHOICE,
        tmp_path / "b",
        tmp_path / "b.db",
        embeddings=HashEmbeddings(),
        spec=spec_for(V2_PLAIN),
    ) as ours:
        our_sizes = [len(c.text) for c in ours.store.document_chunks(doc_id)]
    assert max(legacy_sizes) <= 800
    assert max(our_sizes) > 800  # 512 tokens hold more than 800 characters
    assert len(our_sizes) < len(legacy_sizes)


def test_progress_is_reported_per_batch(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    build_index(
        docs(3),
        V2_PLAIN,
        CHOICE,
        tmp_path / "i",
        tmp_path / "c.db",
        embeddings=HashEmbeddings(),
        spec=spec_for(V2_PLAIN),
        on_progress=lambda done, total: seen.append((done, total)),
    ).close()
    assert seen == [(3, 3)]


def test_build_stats_rate_needs_embedded_chunks_and_time() -> None:
    assert BuildStats(2, 4, 4, 0, 2.0, 3.0).chunks_per_s == 2.0
    assert BuildStats(2, 4, 0, 4, 0.0, 1.0).chunks_per_s is None


def test_a_failed_build_closes_what_it_opened(tmp_path: Path) -> None:
    class Failing(HashEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedder down")

    closed: list[bool] = []
    with pytest.raises(RuntimeError, match="embedder down"):
        build_index(
            docs(),
            V2_PLAIN,
            CHOICE,
            tmp_path / "i",
            tmp_path / "c.db",
            embeddings=Failing(),
            spec=spec_for(V2_PLAIN),
            closer=lambda: closed.append(True),
        )
    assert closed == [True]
    # the folder can be built again afterwards: nothing is still locked
    with build_index(
        docs(),
        V2_PLAIN,
        CHOICE,
        tmp_path / "i",
        tmp_path / "c.db",
        embeddings=HashEmbeddings(),
        spec=spec_for(V2_PLAIN),
    ) as again:
        assert again.dense.count() == 5


# --- BEIR input ------------------------------------------------------------------------------


def test_beir_documents_keep_the_title_apart_and_translate_ids_back(tmp_path: Path) -> None:
    folder = write_tiny_beir(tmp_path / "scifact")
    dataset = load_beir(folder, "test")
    documents, names = beir_documents(dataset)
    assert len(documents) == 16
    first = documents[0]
    assert first.title == "Zebra notes 1"
    assert "Zebra notes" not in first.text  # the title is a header, not part of the text
    assert names[doc_id_of(first.path)] == "zebra-1"
    assert first.path == beir_doc_path("scifact", "zebra-1")


def test_a_beir_document_without_text_cannot_be_indexed_but_keeps_its_name(tmp_path: Path) -> None:
    folder = write_tiny_beir(tmp_path / "scifact")
    with (folder / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"_id": "empty-1", "title": "Nothing", "text": "  "}\n')
    documents, names = beir_documents(load_beir(folder, "test"))
    assert len(documents) == 16  # the empty one is left out
    assert "empty-1" in names.values()
