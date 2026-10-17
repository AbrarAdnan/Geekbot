"""Tests for step S1.5: the Ollama embedder client, the prefixes and the fingerprint.

Ollama is a fake HTTP server (httpx.MockTransport), so nothing here needs a model.
"""

import dataclasses
import json
import re

import httpx
import pytest

from geekbot.config import Settings
from geekbot.domain import Chunk, embedding_text
from geekbot.index.embedder import (
    EmbedderSpec,
    EmbeddingError,
    OllamaEmbeddingsClient,
    PrefixedEmbeddings,
    open_ollama_embedder,
)
from geekbot.ingest.chunking import ChunkDraft
from geekbot.llm.models import OllamaError

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


def _client(
    transport: httpx.MockTransport, *, batch_size: int = 32, model: str = "nomic-embed-text"
) -> OllamaEmbeddingsClient:
    return OllamaEmbeddingsClient(
        "http://ollama.test", model, batch_size=batch_size, transport=transport
    )


def _echo_transport(seen: list[dict[str, object]], dimension: int = 4) -> httpx.MockTransport:
    """A fake /api/embed that answers one vector per input, [i, i, ...] for the i-th input."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        vectors = [[float(i)] * dimension for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"embeddings": vectors, "prompt_eval_count": 7})

    return httpx.MockTransport(handle)


# --- the client ------------------------------------------------------------------------------


def test_request_asks_for_cpu_no_truncation_and_a_keep_alive() -> None:
    seen: list[dict[str, object]] = []
    with _client(_echo_transport(seen)) as client:
        client.embed_documents(["a", "b"])
    assert seen == [
        {
            "model": "nomic-embed-text",
            "input": ["a", "b"],
            "truncate": False,
            "keep_alive": "30m",
            "options": {"num_gpu": 0},
        }
    ]


def test_texts_are_sent_in_batches_and_come_back_in_order() -> None:
    seen: list[dict[str, object]] = []
    with _client(_echo_transport(seen), batch_size=32) as client:
        vectors = client.embed_documents([f"text {i}" for i in range(70)])
    assert [len(body["input"]) for body in seen] == [32, 32, 6]  # type: ignore[arg-type]
    assert len(vectors) == 70
    # The fake numbers vectors within a batch, so the third batch's first vector is [0, ...].
    assert vectors[64] == [0.0] * 4
    assert vectors[69] == [5.0] * 4


def test_no_texts_means_no_request() -> None:
    seen: list[dict[str, object]] = []
    with _client(_echo_transport(seen)) as client:
        assert client.embed_documents([]) == []
    assert seen == []


def test_embed_query_returns_one_vector() -> None:
    with _client(_echo_transport([])) as client:
        assert client.embed_query("hello") == [0.0] * 4


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _client(_echo_transport([]), batch_size=0)


@pytest.mark.parametrize(
    "answer",
    [
        {"embeddings": [[1.0, 2.0]]},  # one vector for two inputs
        {"embeddings": [[1.0, 2.0], []]},  # an empty vector
        {"embeddings": [[1.0, 2.0], [1.0]]},  # sizes differ
        {"embeddings": [[1.0, 2.0], ["x", "y"]]},  # not numbers
        {"embeddings": "nope"},
        {},
    ],
)
def test_a_malformed_answer_is_an_embedding_error(answer: dict[str, object]) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=answer))
    with _client(transport) as client, pytest.raises(EmbeddingError):
        client.embed_documents(["a", "b"])


def test_a_server_error_becomes_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500, text="boom"))
    with _client(transport) as client, pytest.raises(OllamaError, match="/api/embed"):
        client.embed_documents(["a"])


def test_a_non_json_answer_becomes_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="not json"))
    with _client(transport) as client, pytest.raises(OllamaError):
        client.embed_documents(["a"])


def test_an_answer_that_is_not_a_dict_becomes_ollama_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=[1, 2]))
    with _client(transport) as client, pytest.raises(OllamaError, match="not a dict"):
        client.embed_documents(["a"])


def test_prompt_tokens_reads_the_model_count() -> None:
    with _client(_echo_transport([])) as client:
        assert client.prompt_tokens("some text") == 7


def test_prompt_tokens_without_a_count_is_an_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": [[1.0]]}))
    with _client(transport) as client, pytest.raises(OllamaError, match="token count"):
        client.prompt_tokens("x")


# --- the model digest ------------------------------------------------------------------------


def _tags(models: list[dict[str, object]]) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _: httpx.Response(200, json={"models": models}))


def test_digest_of_a_pulled_model() -> None:
    tags = _tags(
        [
            {"name": "gemma3:4b", "digest": "aaa"},
            {"name": "nomic-embed-text:latest", "digest": "bbb"},
        ]
    )
    with _client(tags) as client:
        assert client.model_digest() == "bbb"  # "nomic-embed-text" is "nomic-embed-text:latest"


def test_digest_of_a_model_that_is_not_pulled() -> None:
    tags = _tags([{"name": "gemma3:4b", "digest": "aaa"}])
    with _client(tags) as client, pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        client.model_digest()


def test_digest_ignores_entries_that_are_not_models() -> None:
    with _client(_tags([{"digest": "x"}, {"name": 3}])) as client, pytest.raises(OllamaError):
        client.model_digest()


# --- prefixes --------------------------------------------------------------------------------


class _Recorder:
    """An embedder that records what it was asked and returns its input's length."""

    def __init__(self) -> None:
        self.documents: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.append(list(texts))
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text))]


def test_documents_and_queries_get_their_own_prefix() -> None:
    inner = _Recorder()
    embeddings = PrefixedEmbeddings(
        inner, document_prefix="search_document: ", query_prefix="search_query: "
    )
    embeddings.embed_documents(["a", "b"])
    embeddings.embed_query("q")
    assert inner.documents == [["search_document: a", "search_document: b"]]
    assert inner.queries == ["search_query: q"]


def test_an_empty_prefix_leaves_the_text_alone() -> None:
    inner = _Recorder()
    embeddings = PrefixedEmbeddings(inner, document_prefix="", query_prefix="")
    embeddings.embed_documents(["a"])
    embeddings.embed_query("q")
    assert inner.documents == [["a"]]
    assert inner.queries == ["q"]


# --- the fingerprint -------------------------------------------------------------------------

SPEC = EmbedderSpec(
    model="nomic-embed-text",
    digest="0a109f422b47",
    document_prefix="search_document: ",
    query_prefix="search_query: ",
    dimension=768,
    with_header=True,
)


def test_fingerprint_is_short_hex_and_stable() -> None:
    assert re.fullmatch(r"[0-9a-f]{16}", SPEC.fingerprint())
    assert dataclasses.replace(SPEC).fingerprint() == SPEC.fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"digest": "ffffffffffff"},
        {"document_prefix": "doc: "},
        {"query_prefix": "q: "},
        {"dimension": 384},
        {"with_header": False},
    ],
)
def test_every_field_changes_the_fingerprint(change: dict[str, object]) -> None:
    assert dataclasses.replace(SPEC, **change).fingerprint() != SPEC.fingerprint()  # type: ignore[arg-type]


def test_describe_is_the_spec_as_json() -> None:
    assert json.loads(SPEC.describe()) == dataclasses.asdict(SPEC)


# --- open_ollama_embedder --------------------------------------------------------------------


def _ollama(seen: list[httpx.Request], *, models: list[dict[str, object]]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": models})
        return httpx.Response(200, json={"embeddings": [[0.5] * 6]})

    return httpx.MockTransport(handle)


def test_open_describes_the_embedder_from_the_settings() -> None:
    seen: list[httpx.Request] = []
    transport = _ollama(seen, models=[{"name": "nomic-embed-text:latest", "digest": "d1"}])
    prefixed, spec, client = open_ollama_embedder(Settings(), transport=transport)
    with client:
        assert spec == EmbedderSpec(
            model="nomic-embed-text",
            digest="d1",
            document_prefix="search_document: ",
            query_prefix="search_query: ",
            dimension=6,
            with_header=True,
        )
        assert isinstance(prefixed, PrefixedEmbeddings)
    # the digest, then the probe embedding on the CPU
    assert [r.url.path for r in seen] == ["/api/tags", "/api/embed"]
    assert json.loads(seen[1].content)["options"] == {"num_gpu": 0}


def test_open_fails_when_the_model_is_not_pulled() -> None:
    transport = _ollama([], models=[])
    with pytest.raises(OllamaError, match="not pulled"):
        open_ollama_embedder(Settings(), transport=transport)


def test_open_follows_the_header_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEEKBOT_EMBED_WITH_HEADER", "false")
    transport = _ollama([], models=[{"name": "nomic-embed-text", "digest": "d"}])
    _, spec, client = open_ollama_embedder(Settings(), transport=transport)
    client.close()
    assert spec.with_header is False


# --- the text that is embedded ---------------------------------------------------------------


def test_embedding_text_puts_the_header_in_front_only_when_asked() -> None:
    assert embedding_text("Doc > Intro", "body", with_header=True) == "Doc > Intro\n\nbody"
    assert embedding_text("Doc > Intro", "body", with_header=False) == "body"
    assert embedding_text("", "body", with_header=True) == "body"


def test_chunk_and_draft_embed_the_same_string() -> None:
    draft = ChunkDraft(
        text="body",
        char_start=0,
        char_end=4,
        page=None,
        section_path=("Intro",),
        header="Doc > Intro",
        ordinal=0,
    )
    chunk = Chunk(
        chunk_id="c",
        doc_id="d",
        path="p",
        title="Doc",
        content_hash="h",
        occurrence=0,
        ordinal=0,
        char_start=0,
        char_end=4,
        page=None,
        section_path=("Intro",),
        header="Doc > Intro",
        text="body",
    )
    for with_header in (True, False):
        assert chunk.embedding_text(with_header=with_header) == draft.embedding_text(
            with_header=with_header
        )
