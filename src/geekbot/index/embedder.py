"""The embedder: text in, vectors out (step S1.5, PLAN.md §6.4, §6.5).

An embedding model turns a text into a vector so that texts about the same thing end up close
together. Geekbot embeds on the **CPU** (``num_gpu=0``): the chat model owns the 10 GB card, and a
small embedder on the CPU costs a few milliseconds per chunk instead of a few hundred megabytes of
VRAM (PLAN.md §6.5).

Three layers, each replaceable on its own:

1. ``EmbeddingsClient``: anything with ``embed_documents`` and ``embed_query``, the two method
   names of LangChain's ``Embeddings``. ``OllamaEmbeddingsClient`` is the one used now (plain
   ``httpx``, no new dependency); LangChain's ``OllamaEmbeddings`` fits the same shape if a later
   step wants it.
2. ``PrefixedEmbeddings``: puts the model's task prefix in front of every text. ``nomic-embed-text``
   was trained with ``search_document: `` and ``search_query: ``. Without them documents and
   queries are embedded as if they were the same kind of text, which is the v1 mistake (PLAN.md
   §5, #8) and the ablation's rung R3.
3. ``EmbedderSpec``: everything that decides what a vector means (model, its digest, the prefixes,
   the dimension, the header mode), reduced to one ``fingerprint``. Vectors from two different
   fingerprints are not comparable, so the dense index refuses to mix them (``index/dense.py``) and
   the embedding cache never serves one for the other (``index/embed_cache.py``).
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Final, Protocol, Self

import httpx

from geekbot.config import Settings
from geekbot.llm.models import OllamaError, same_model

__all__ = [
    "EmbedderSpec",
    "EmbeddingError",
    "EmbeddingsClient",
    "OllamaEmbeddingsClient",
    "PrefixedEmbeddings",
    "open_ollama_embedder",
]

# Text used once, to learn the model's vector size and to check that it answers.
_PROBE_TEXT: Final = "dimension probe"


class EmbeddingError(RuntimeError):
    """An embedder answered, but not with what was asked for (wrong count, size or shape)."""


class EmbeddingsClient(Protocol):
    """What the index needs from an embedder. Matches the method names of LangChain's ``Embeddings``."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """The vector of one query."""
        ...


class OllamaEmbeddingsClient:
    """Embeds through Ollama's ``/api/embed``, on the CPU unless told otherwise.

    Synchronous on purpose, like ``OllamaClient``: the CLI is synchronous and the server calls it
    from a worker thread, never on the event loop (PLAN.md §6.3).

    ``truncate`` is sent as ``False``. Ollama's default is to cut an over-long input silently,
    and a vector computed from half a chunk is a quiet retrieval bug; with ``False`` the call
    fails, and the ingest report (S1.6) names the file instead.
    """

    def __init__(
        self,
        host: str,
        model: str,
        *,
        num_gpu: int = 0,
        keep_alive: str = "30m",
        batch_size: int = 32,
        timeout_s: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.model = model
        self._num_gpu = num_gpu
        self._keep_alive = keep_alive
        self._batch_size = batch_size
        # transport lets tests plug in httpx.MockTransport instead of a real server.
        self._client = httpx.Client(base_url=host, transport=transport, timeout=timeout_s)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP connection."""
        self._client.close()

    def _post(self, texts: Sequence[str]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
            "truncate": False,
            "keep_alive": self._keep_alive,
            "options": {"num_gpu": self._num_gpu},
        }
        try:
            response = self._client.post("/api/embed", json=body)
            response.raise_for_status()
            data: object = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OllamaError(f"Ollama POST /api/embed failed: {error}") from error
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama POST /api/embed returned {type(data).__name__}, not a dict")
        return data

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order, sent in batches of ``batch_size``."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(_vectors_of(self._post(batch), expected=len(batch)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """The vector of one text."""
        return self.embed_documents([text])[0]

    def prompt_tokens(self, text: str) -> int:
        """How many tokens the model counted for ``text`` (Ollama's ``prompt_eval_count``).

        Used by the live test that compares the model's real count with ``estimate_tokens``
        (S1.2 left that comparison for this step).
        """
        count = self._post([text]).get("prompt_eval_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise OllamaError(
                f"Ollama /api/embed gave no token count (prompt_eval_count={count!r})"
            )
        return count

    def model_digest(self) -> str:
        """The digest Ollama reports for the model: it changes when the model is re-pulled.

        A new digest under the same name can mean different weights, hence different vectors,
        so the digest is part of the fingerprint. ``name`` and ``name:latest`` are one model.
        """
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data: object = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OllamaError(f"Ollama GET /api/tags failed: {error}") from error
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, list):
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                name, digest = entry.get("name"), entry.get("digest")
                if (
                    isinstance(name, str)
                    and isinstance(digest, str)
                    and same_model(name, self.model)
                ):
                    return digest
        raise OllamaError(f"model {self.model!r} is not pulled; run `ollama pull {self.model}`")


def _vectors_of(data: dict[str, object], *, expected: int) -> list[list[float]]:
    """Check Ollama's answer and turn it into lists of floats."""
    raw = data.get("embeddings")
    if not isinstance(raw, list) or len(raw) != expected:
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise EmbeddingError(f"asked for {expected} embeddings, Ollama returned {got}")
    vectors: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            raise EmbeddingError("Ollama returned an empty or non-list embedding")
        try:
            vectors.append([float(x) for x in item])
        except (TypeError, ValueError) as error:
            raise EmbeddingError(f"Ollama returned a non-numeric embedding: {error}") from error
    if len({len(v) for v in vectors}) > 1:
        raise EmbeddingError("Ollama returned embeddings of different sizes in one batch")
    return vectors


class PrefixedEmbeddings:
    """Adds a model's task prefix to every text before it reaches the embedder.

    Retrieval is asymmetric: a query is short and asks, a chunk is long and answers. Models such as
    ``nomic-embed-text`` learn two "modes" for that and need to be told which one a text is. The
    prefixes live here, in one place, so a document and a query can never be embedded in the same
    mode by mistake, and the fingerprint (``EmbedderSpec``) records them.
    """

    def __init__(self, inner: EmbeddingsClient, *, document_prefix: str, query_prefix: str) -> None:
        self._inner = inner
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` as documents (the prefix goes in front of each)."""
        return self._inner.embed_documents([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed ``text`` as a query."""
        return self._inner.embed_query(self.query_prefix + text)


@dataclass(frozen=True)
class EmbedderSpec:
    """Everything that decides what a stored vector means.

    If any field differs, vectors are not comparable: another model, another weights file under the
    same name (``digest``), another prefix, another size, or another text (the header mode embeds
    ``header + text`` instead of ``text``). ``fingerprint`` is the short identity of the whole set.
    It goes into the dense collection's metadata (a mismatch blocks queries until a re-index) and
    into every embedding-cache key.
    """

    model: str
    digest: str
    document_prefix: str
    query_prefix: str
    dimension: int
    with_header: bool

    def fingerprint(self) -> str:
        """16 hex characters, stable across runs and machines."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """The spec as JSON, kept in the collection's metadata so a mismatch can say what differs."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def open_ollama_embedder(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> tuple[PrefixedEmbeddings, EmbedderSpec, OllamaEmbeddingsClient]:
    """Connect to Ollama and describe the embedder the settings ask for.

    Makes two calls: the model's digest, and one probe embedding to learn the vector size. Returns
    the prefixed embedder, its spec, and the raw client (the caller closes it). Raises
    ``OllamaError`` if Ollama is down or the model is not pulled.
    """
    client = OllamaEmbeddingsClient(
        settings.ollama_host,
        settings.embed_model,
        num_gpu=0,
        keep_alive=settings.keep_alive,
        batch_size=settings.embed_batch_size,
        transport=transport,
    )
    try:
        digest = client.model_digest()
        dimension = len(client.embed_query(_PROBE_TEXT))
    except BaseException:
        client.close()
        raise
    prefixed = PrefixedEmbeddings(
        client,
        document_prefix=settings.embed_document_prefix,
        query_prefix=settings.embed_query_prefix,
    )
    spec = EmbedderSpec(
        model=settings.embed_model,
        digest=digest,
        document_prefix=settings.embed_document_prefix,
        query_prefix=settings.embed_query_prefix,
        dimension=dimension,
        with_header=settings.embed_with_header,
    )
    return prefixed, spec, client
