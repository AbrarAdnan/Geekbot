"""Tests for step S1.5: the embedding cache and the cached embedder (no model, no network)."""

import dataclasses
import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from geekbot.index.embed_cache import (
    CACHE_FILE_NAME,
    CachedEmbedder,
    EmbeddingCache,
    text_key,
    to_float32,
)
from geekbot.index.embedder import EmbedderSpec, EmbeddingError

DIM = 8

SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=DIM,
    with_header=True,
)


def _vector_of(text: str) -> list[float]:
    """A fake, deterministic embedding: the first bytes of the text's SHA-256, scaled."""
    return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:DIM]]


class FakeEmbeddings:
    """An embedder that records every text it was asked to embed."""

    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension
        self.embedded: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [_vector_of(t)[: self.dimension] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return _vector_of(text)[: self.dimension]


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[EmbeddingCache]:
    with EmbeddingCache(tmp_path / CACHE_FILE_NAME) as opened:
        yield opened


# --- keys and rounding -----------------------------------------------------------------------


def test_text_key_is_sha256_hex() -> None:
    assert text_key("abc") == hashlib.sha256(b"abc").hexdigest()
    assert text_key("abc") != text_key("abd")


def test_to_float32_rounds_once_and_is_then_stable() -> None:
    once = to_float32([0.1, 1 / 3, -2.5])
    assert once != [0.1, 1 / 3, -2.5]  # a double is not a float32
    assert to_float32(once) == once


# --- the cache -------------------------------------------------------------------------------


def test_an_empty_lookup_finds_nothing(cache: EmbeddingCache) -> None:
    assert cache.lookup("fp", []) == {}
    assert cache.lookup("fp", ["nope"]) == {}


def test_stored_vectors_come_back_exactly(cache: EmbeddingCache) -> None:
    vector = to_float32([0.1, 0.2, 0.3])
    cache.store("fp", {"h1": vector})
    assert cache.lookup("fp", ["h1", "h2"]) == {"h1": vector}


def test_the_same_text_under_another_fingerprint_is_invisible(cache: EmbeddingCache) -> None:
    cache.store("fp-a", {"h1": [1.0, 2.0]})
    assert cache.lookup("fp-b", ["h1"]) == {}
    assert cache.count("fp-a") == 1
    assert cache.count("fp-b") == 0
    assert cache.count() == 1


def test_storing_again_replaces_the_vector(cache: EmbeddingCache) -> None:
    cache.store("fp", {"h1": [1.0, 2.0]})
    cache.store("fp", {"h1": [3.0, 4.0]})
    assert cache.lookup("fp", ["h1"]) == {"h1": [3.0, 4.0]}
    assert cache.count() == 1


def test_a_large_lookup_is_one_statement(cache: EmbeddingCache) -> None:
    hashes = [f"h{i}" for i in range(40_000)]  # more than SQLite's default variable limit
    cache.store("fp", {h: [float(i)] for i, h in enumerate(hashes[:50])})
    found = cache.lookup("fp", hashes)
    assert len(found) == 50


def test_the_cache_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "nested" / CACHE_FILE_NAME  # the folder does not exist yet
    with EmbeddingCache(path) as first:
        first.store("fp", {"h1": [1.0, 2.0]})
    with EmbeddingCache(path) as second:
        assert second.lookup("fp", ["h1"]) == {"h1": [1.0, 2.0]}


def test_a_file_from_another_version_is_started_over(tmp_path: Path) -> None:
    path = tmp_path / CACHE_FILE_NAME
    with EmbeddingCache(path) as first:
        first.store("fp", {"h1": [1.0]})
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()
    with EmbeddingCache(path) as second:
        assert second.count() == 0  # a cache is only time, so it is dropped, not migrated
        second.store("fp", {"h2": [2.0]})
        assert second.count() == 1


def test_close_twice_is_harmless(tmp_path: Path) -> None:
    opened = EmbeddingCache(tmp_path / CACHE_FILE_NAME)
    opened.close()
    opened.close()


def test_threads_can_share_one_cache(cache: EmbeddingCache) -> None:
    def work(worker: int) -> None:
        for i in range(50):
            key = f"w{worker}-{i}"
            cache.store("fp", {key: [float(i)]})
            assert cache.lookup("fp", [key]) == {key: [float(i)]}

    threads = [threading.Thread(target=work, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert cache.count("fp") == 200


# --- the cached embedder ---------------------------------------------------------------------


def test_the_first_pass_embeds_everything_and_the_second_nothing(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    embedder = CachedEmbedder(fake, cache, SPEC)
    texts = ["alpha", "beta", "gamma"]

    first = embedder.embed_documents(texts)
    assert (first.embedded, first.cache_hits) == (3, 0)
    assert fake.embedded == texts

    second = embedder.embed_documents(texts)
    assert (second.embedded, second.cache_hits) == (0, 3)  # Gate A: unchanged data embeds nothing
    assert fake.embedded == texts  # the embedder was not asked again
    assert second.vectors == first.vectors


def test_only_the_new_texts_are_embedded(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    embedder = CachedEmbedder(fake, cache, SPEC)
    embedder.embed_documents(["alpha", "beta"])
    fake.embedded.clear()

    result = embedder.embed_documents(["beta", "delta", "alpha"])
    assert fake.embedded == ["delta"]
    assert (result.embedded, result.cache_hits) == (1, 2)
    assert result.vectors == [
        to_float32(_vector_of("beta")),
        to_float32(_vector_of("delta")),
        to_float32(_vector_of("alpha")),
    ]


def test_a_repeated_text_is_embedded_once_and_fills_every_position(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    result = CachedEmbedder(fake, cache, SPEC).embed_documents(["x", "y", "x", "x"])
    assert fake.embedded == ["x", "y"]
    assert (result.embedded, result.cache_hits) == (2, 0)
    assert result.vectors[0] == result.vectors[2] == result.vectors[3]
    assert result.vectors[1] != result.vectors[0]


def test_no_texts_is_no_work(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    result = CachedEmbedder(fake, cache, SPEC).embed_documents([])
    assert (result.vectors, result.embedded, result.cache_hits) == ([], 0, 0)
    assert fake.embedded == []


def test_another_fingerprint_never_reuses_a_vector(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    CachedEmbedder(fake, cache, SPEC).embed_documents(["alpha"])
    without_header = dataclasses.replace(SPEC, with_header=False)

    result = CachedEmbedder(fake, cache, without_header).embed_documents(["alpha"])
    assert result.embedded == 1  # same text, different setup: embedded again
    assert cache.count() == 2


def test_the_same_chunk_under_two_headers_is_two_cache_entries(cache: EmbeddingCache) -> None:
    # The key is the exact embedding text, header included (D-058): "Doc A > Intro\n\nbody" and
    # "Doc B > Intro\n\nbody" are different strings and must not share a vector.
    fake = FakeEmbeddings()
    result = CachedEmbedder(fake, cache, SPEC).embed_documents(
        ["Doc A > Intro\n\nbody", "Doc B > Intro\n\nbody"]
    )
    assert result.embedded == 2
    assert result.vectors[0] != result.vectors[1]


def test_a_vector_of_the_wrong_size_is_refused_and_not_cached(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings(dimension=DIM - 1)
    with pytest.raises(EmbeddingError, match="expected 8"):
        CachedEmbedder(fake, cache, SPEC).embed_documents(["alpha"])
    assert cache.count() == 0


def test_the_wrong_number_of_vectors_is_refused(cache: EmbeddingCache) -> None:
    class Short:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * DIM for _ in texts[:-1]]

        def embed_query(self, text: str) -> list[float]:
            return [0.0] * DIM

    with pytest.raises(EmbeddingError, match="asked for 2"):
        CachedEmbedder(Short(), cache, SPEC).embed_documents(["a", "b"])
    assert cache.count() == 0


def test_queries_are_embedded_directly_and_checked(cache: EmbeddingCache) -> None:
    fake = FakeEmbeddings()
    embedder = CachedEmbedder(fake, cache, SPEC)
    assert embedder.embed_query("what?") == to_float32(_vector_of("what?"))
    assert fake.queries == ["what?"]
    assert cache.count() == 0  # queries are not cached

    with pytest.raises(EmbeddingError, match="query"):
        CachedEmbedder(FakeEmbeddings(dimension=3), cache, SPEC).embed_query("what?")


def test_spec_is_exposed(cache: EmbeddingCache) -> None:
    assert CachedEmbedder(FakeEmbeddings(), cache, SPEC).spec is SPEC


# --- properties ------------------------------------------------------------------------------

# A tiny alphabet makes repeated texts common, which is the case the de-duplication must get right.
_TEXTS = st.lists(st.text(alphabet="abc", max_size=3), max_size=12)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(first=_TEXTS, second=_TEXTS)
def test_cached_results_equal_uncached_results(
    cache: EmbeddingCache, first: list[str], second: list[str]
) -> None:
    fake = FakeEmbeddings()
    embedder = CachedEmbedder(fake, cache, SPEC)
    for texts in (first, second, first):
        before = len(fake.embedded)
        result = embedder.embed_documents(texts)
        # Whatever the cache holds, every position gets the vector a direct embedding would give.
        assert result.vectors == [to_float32(_vector_of(t)) for t in texts]
        # Nothing is embedded twice in a call, and a text seen in an earlier call is not embedded
        # again, so the number embedded is the number of distinct texts that are new to the cache.
        assert len(fake.embedded) - before == result.embedded
        assert result.embedded + result.cache_hits == len(set(texts))
    # A third pass over `first` is entirely cached.
    assert embedder.embed_documents(first).embedded == 0
