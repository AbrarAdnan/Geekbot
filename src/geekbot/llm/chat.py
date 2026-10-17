"""The chat model for the answer graph (step S3.1, PLAN.md §6.5, §7).

Everything that talks to Ollama for a *conversation* goes through LangChain's ``ChatOllama``: it
speaks the same message types the graph uses, and later steps use more of it (streaming in S3.3,
structured output in S3.4). The embedder and the model swaps stay on Geekbot's own thin client
(``llm.models``), where the exact HTTP calls matter.

**Why the factory takes its numbers from the settings.** Ollama keeps a model loaded only for the
exact options it was loaded with. The warm-up (``swap_to_mode``) loads the chat model with
``num_ctx = settings.chat_num_ctx`` and ``keep_alive = settings.keep_alive``. If a query then
asks for a different ``num_ctx``, Ollama quietly reloads the model, which breaks rule R-2 (no
model loads during a query) and costs seconds. One place builds the chat model, from the same
settings the warm-up reads, so the two cannot disagree.

``ChatModel`` is the one method the graph needs. The graph is typed against it, not against
``ChatOllama``, so a test passes a fake that has no network and no LangChain base class to fake.
"""

from typing import Protocol

from langchain_core.messages import BaseMessage

from geekbot.config import Settings

__all__ = ["ChatError", "ChatModel", "make_chat_model"]


class ChatError(RuntimeError):
    """The chat model could not answer: Ollama is down, the model is missing, or it errored."""


class ChatModel(Protocol):
    """Anything that turns a list of messages into one reply message."""

    def invoke(self, messages: list[BaseMessage], /) -> BaseMessage:
        """The model's reply. The parameter is positional-only so ``ChatOllama`` fits."""
        ...


def make_chat_model(settings: Settings, *, model: str | None = None) -> ChatModel:
    """A ``ChatOllama`` for the active profile's chat model, with thinking switched off.

    - ``num_ctx`` and ``keep_alive`` repeat the warm-up's values (see the module docstring).
    - ``num_predict`` caps the reply (PLAN.md §6.5: 512), so one runaway answer cannot hold the GPU.
    - ``temperature`` is low (0.2): the answer should follow the sources, not improvise.
    - ``reasoning=False`` turns thinking off (PLAN.md §6.5: off by default). Models without a
      thinking mode ignore it; the first live run checks that Ollama accepts it for ``gemma3:4b``.
    """
    # Imported here: LangChain's Ollama package pulls in the ``ollama`` client and pydantic models,
    # and `geekbot --version` or `geekbot doctor` should not wait for them.
    from langchain_ollama import ChatOllama  # type: ignore[import-untyped, unused-ignore]

    chat: ChatModel = ChatOllama(
        model=model or settings.resolved_chat_model,
        base_url=settings.ollama_host,
        num_ctx=settings.chat_num_ctx,
        num_predict=settings.chat_num_predict,
        keep_alive=settings.keep_alive,
        temperature=settings.chat_temperature,
        reasoning=False,
    )
    return chat
