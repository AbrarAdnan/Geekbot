"""Tests for step S1.5: the Chroma dense index, with hand-made vectors (no embedder involved).

Chroma runs for real against a temporary folder: it works offline when it is given vectors, and a
fake would not catch the things this class exists for (cosine scoring, persistence, the guard).
"""

import dataclasses
import math
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geekbot.index.dense import (
    CHROMA_DIR_NAME,
    DenseEntry,
    DenseIndex,
    DenseIndexError,
    FingerprintMismatchError,
)
from geekbot.index.embedder import EmbedderSpec

SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=3,
    with_header=True,
)

A = DenseEntry("a", "doc1", [1.0, 0.0, 0.0])
B = DenseEntry("b", "doc1", [0.0, 1.0, 0.0])
C = DenseEntry("c", "doc2", [0.7, 0.7, 0.0])


@pytest.fixture
def index(tmp_path: Path) -> Iterator[DenseIndex]:
    with DenseIndex(tmp_path, SPEC) as opened:
        yield opened


# --- search ----------------------------------------------------------------------------------


def test_neighbours_are_ranked_by_cosine_similarity(index: DenseIndex) -> None:
    index.upsert([A, B, C])
    hits = index.query([1.0, 0.0, 0.0], 3)
    assert [h.chunk_id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(1 / math.sqrt(2), abs=1e-3)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)


def test_the_score_is_cosine_not_length(index: DenseIndex) -> None:
    # L2 would rank the long vector far from a unit query; cosine sees the same direction.
    index.upsert(
        [DenseEntry("long", "d", [10.0, 0.0, 0.0]), DenseEntry("near", "d", [0.9, 0.3, 0.0])]
    )
    hits = index.query([1.0, 0.0, 0.0], 2)
    assert [h.chunk_id for h in hits] == ["long", "near"]
    assert hits[0].score == pytest.approx(1.0)


def test_asking_for_more_than_there_is_returns_what_there_is(index: DenseIndex) -> None:
    index.upsert([A, B])
    assert len(index.query([1.0, 0.0, 0.0], 50)) == 2


def test_k_limits_the_hits(index: DenseIndex) -> None:
    index.upsert([A, B, C])
    assert [h.chunk_id for h in index.query([1.0, 0.0, 0.0], 1)] == ["a"]


def test_an_empty_index_and_a_zero_k_return_nothing(index: DenseIndex) -> None:
    assert index.query([1.0, 0.0, 0.0], 5) == []
    index.upsert([A])
    assert index.query([1.0, 0.0, 0.0], 0) == []


# --- writing ---------------------------------------------------------------------------------


def test_upsert_is_idempotent_and_overwrites(index: DenseIndex) -> None:
    index.upsert([A, B])
    index.upsert([A, B])
    assert index.count() == 2
    index.upsert([DenseEntry("a", "doc1", [0.0, 0.0, 1.0])])  # a moved
    assert index.count() == 2
    assert index.query([0.0, 0.0, 1.0], 1)[0].chunk_id == "a"


def test_an_empty_upsert_does_nothing(index: DenseIndex) -> None:
    index.upsert([])
    assert index.count() == 0


def test_a_vector_of_the_wrong_size_is_refused_and_nothing_is_stored(index: DenseIndex) -> None:
    with pytest.raises(DenseIndexError, match="2 numbers"):
        index.upsert([A, DenseEntry("bad", "d", [1.0, 0.0])])
    assert index.count() == 0
    with pytest.raises(DenseIndexError, match="4 numbers"):
        index.query([1.0, 0.0, 0.0, 0.0], 1)


def test_delete_removes_some_and_ignores_missing_ids(index: DenseIndex) -> None:
    index.upsert([A, B, C])
    index.delete(["a", "never-stored"])
    assert index.ids() == {"b", "c"}
    index.delete([])
    assert index.count() == 2


def test_delete_document_removes_all_its_vectors(index: DenseIndex) -> None:
    index.upsert([A, B, C])
    index.delete_document("doc1")
    assert index.ids() == {"c"}
    index.delete_document("no-such-doc")
    assert index.count() == 1


def test_ids_of_a_document(index: DenseIndex) -> None:
    index.upsert([A, B, C])
    assert index.ids_of_document("doc1") == {"a", "b"}
    assert index.ids_of_document("doc2") == {"c"}
    assert index.ids_of_document("other") == set()


def test_ids_pages_through_a_large_index(index: DenseIndex) -> None:
    entries = [
        DenseEntry(f"chunk-{i}", f"doc{i % 7}", [1.0, float(i % 5), 0.5]) for i in range(2100)
    ]
    index.upsert(entries)  # more than two internal batches
    assert index.count() == 2100
    assert index.ids() == {e.chunk_id for e in entries}


def test_stored_vectors_can_be_read_back(index: DenseIndex) -> None:
    index.upsert([A, C])
    found = index.vectors(["a", "c", "missing"])
    assert set(found) == {"a", "c"}
    assert found["a"] == pytest.approx([1.0, 0.0, 0.0])
    assert found["c"] == pytest.approx([0.7, 0.7, 0.0], abs=1e-6)  # float32


# --- persistence and the fingerprint guard ---------------------------------------------------


def test_vectors_survive_a_restart(tmp_path: Path) -> None:
    with DenseIndex(tmp_path, SPEC) as first:
        first.upsert([A, B])
    assert (tmp_path / CHROMA_DIR_NAME).is_dir()
    with DenseIndex(tmp_path, SPEC) as second:
        assert second.ids() == {"a", "b"}
        assert second.query([1.0, 0.0, 0.0], 1)[0].chunk_id == "a"


def test_another_embedder_is_refused(tmp_path: Path) -> None:
    with DenseIndex(tmp_path, SPEC) as first:
        first.upsert([A])
    other = dataclasses.replace(SPEC, document_prefix="passage: ")
    with pytest.raises(FingerprintMismatchError) as caught:
        DenseIndex(tmp_path, other)
    assert caught.value.stored == SPEC.fingerprint()
    assert caught.value.expected == other.fingerprint()
    message = str(caught.value)
    assert "doc: " in message
    assert "passage: " in message  # the message shows what differs
    assert "reset=True" in message


def test_the_guard_notices_a_new_model_digest(tmp_path: Path) -> None:
    DenseIndex(tmp_path, SPEC).close()
    with pytest.raises(FingerprintMismatchError):
        DenseIndex(tmp_path, dataclasses.replace(SPEC, digest="re-pulled"))


def test_the_guard_notices_the_header_mode(tmp_path: Path) -> None:
    DenseIndex(tmp_path, SPEC).close()
    with pytest.raises(FingerprintMismatchError):
        DenseIndex(tmp_path, dataclasses.replace(SPEC, with_header=False))


def test_reset_starts_over_with_the_new_embedder(tmp_path: Path) -> None:
    with DenseIndex(tmp_path, SPEC) as first:
        first.upsert([A, B])
    other = dataclasses.replace(SPEC, dimension=2)
    with DenseIndex(tmp_path, other, reset=True) as fresh:
        assert fresh.count() == 0
        fresh.upsert([DenseEntry("x", "d", [1.0, 0.0])])
    with DenseIndex(tmp_path, other) as again:  # and the new fingerprint is now the stored one
        assert again.ids() == {"x"}


def test_reset_on_a_new_folder_is_fine(tmp_path: Path) -> None:
    with DenseIndex(tmp_path / "fresh", SPEC, reset=True) as opened:
        assert opened.count() == 0


def test_the_spec_is_exposed(index: DenseIndex) -> None:
    assert index.spec is SPEC


# --- a property ------------------------------------------------------------------------------

_VECTORS = st.lists(
    st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=3, max_size=3).filter(
        lambda v: math.sqrt(sum(x * x for x in v)) > 0.2
    ),
    min_size=1,
    max_size=12,
    unique_by=tuple,
)


@settings(max_examples=15, deadline=None)
@given(vectors=_VECTORS)
def test_a_stored_vector_is_its_own_nearest_neighbour(vectors: list[list[float]]) -> None:
    # A property of cosine similarity: nothing is closer to a vector than that vector (score 1).
    # With so few points Chroma's HNSW graph is exact, so this must hold for any vectors.
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder,
        DenseIndex(Path(folder), SPEC) as opened,
    ):
        opened.upsert([DenseEntry(f"v{i}", "d", v) for i, v in enumerate(vectors)])
        for vector in vectors:
            # Another stored vector may point the same way (and then ties at 1.0), so the
            # property is about the score, not about which chunk comes first.
            assert opened.query(vector, 1)[0].score == pytest.approx(1.0, abs=1e-3)
