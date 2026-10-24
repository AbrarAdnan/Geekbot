"""The chat model for the answer graph (steps S3.1 and S3.3, PLAN.md §6.5, §7).

Everything that talks to Ollama for a *conversation* goes through LangChain's ``ChatOllama``: it
speaks the same message types the graph uses, and later steps use more of it (structured output in
S3.4). The embedder and the model swaps stay on Geekbot's own thin client (``llm.models``), where the
exact HTTP calls matter.

**Why the factory takes its numbers from the settings.** Ollama keeps a model loaded only for the
exact options it was loaded with. The warm-up (``swap_to_mode``) loads the chat model with
``num_ctx = settings.chat_num_ctx`` and ``keep_alive = settings.keep_alive``. If a query then
asks for a different ``num_ctx``, Ollama quietly reloads the model, which breaks rule R-2 (no
model loads during a query) and costs seconds. One place builds the chat model, from the same
settings the warm-up reads, so the two cannot disagree.

``ChatModel`` is the one method the graph needs: ``stream``. The graph always streams, even when
nobody watches the pieces, because that is the only way to time the *first* token (S3.3, see
``llm.streaming``). The graph is typed against the Protocol, not against ``ChatOllama``, so a test
passes a fake that has no network and no LangChain base class to fake.
"""

from collections.abc import Iterator
from functools import cache
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from geekbot.config import Settings
from geekbot.llm.models import same_model

__all__ = [
    "ChatError",
    "ChatModel",
    "StructuredChat",
    "make_chat_model",
    "make_structured_model",
]


class ChatError(RuntimeError):
    """The chat model could not answer: Ollama is down, the model is missing, or it errored.

    ``partial`` (S3.6) says that some of the reply had already been sent to the listener when the
    call failed. Such a failure must not be retried: the retry would write the answer again after
    the half that is already on the screen (``agent.resilience``).
    """

    def __init__(self, message: str, *, partial: bool = False) -> None:
        super().__init__(message)
        self.partial = partial


class ChatModel(Protocol):
    """Anything that turns a list of messages into a stream of reply chunks."""

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        """The reply, piece by piece. Positional-only so ``ChatOllama`` fits."""
        ...


class StructuredChat(Protocol):
    """A chat model held to one JSON shape: ``invoke`` returns the parsed object (S3.4)."""

    def invoke(self, messages: list[BaseMessage], /) -> Any:
        """The reply as an instance of the schema it was built with. Raises if it cannot be parsed."""
        ...


@cache
def _ollama_chat_class() -> Any:
    """``ChatOllama`` with one more option, ``num_batch`` (step S5.1).

    LangChain's class has no field for Ollama's batch size, but the batch size is a load-time option
    like ``num_ctx``: Ollama reloads the model when a request carries a different one (R-2). So the
    sweep of PLAN §6.5 (``num_batch`` 256 or 512) needs every call, not only the warm-up, to send it.
    The subclass adds the value to the ``options`` of each request and does nothing when it is
    ``None``, so the default behaviour is exactly ``ChatOllama``'s. Built on first use because the
    LangChain package is slow to import.
    """
    from langchain_ollama import ChatOllama  # type: ignore[import-untyped, unused-ignore]

    class BatchedChatOllama(ChatOllama):  # type: ignore[misc, unused-ignore]
        num_batch: int | None = None

        def _chat_params(
            self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            params: dict[str, Any] = super()._chat_params(messages, stop, **kwargs)
            options = params.get("options")
            if self.num_batch is not None and isinstance(options, dict):
                params["options"] = {**options, "num_batch": self.num_batch}
            return params

    return BatchedChatOllama


def make_structured_model(
    settings: Settings,
    schema: type[BaseModel],
    *,
    num_predict: int,
    model: str | None = None,
    num_ctx: int | None = None,
) -> StructuredChat:
    """A ``ChatOllama`` whose reply Ollama constrains to ``schema`` and LangChain parses into it.

    The same model, ``num_ctx`` and ``keep_alive`` as the answer model (rule R-2: a different
    context size would make Ollama reload the model for this call), so asking for a rewrite never
    loads anything. Differences: ``num_predict`` is the small cap the call needs (PLAN.md §6.5: 96
    for the rewrite of a follow-up), the temperature is 0 (a rewrite should be repeatable), and thinking
    is off. ``method="json_schema"`` sends the schema as Ollama's ``format``, so the output is
    constrained while it is generated, not repaired afterwards. For a CSV conversation (C4) pass the
    CSV model as ``model`` and its context size as ``num_ctx`` (the CSV model is warmed up with
    ``csv_num_ctx``, not ``chat_num_ctx``): the rewrite must use the model that is on the GPU.
    """
    chosen = model or settings.resolved_chat_model
    # Only the chat model is loaded with a batch size; the CSV model is not (see `swap_to_mode`).
    batch = settings.chat_num_batch if same_model(chosen, settings.resolved_chat_model) else None
    chat = _ollama_chat_class()(
        model=chosen,
        base_url=settings.ollama_host,
        num_ctx=num_ctx or settings.chat_num_ctx,
        num_batch=batch,
        num_predict=num_predict,
        keep_alive=settings.keep_alive,
        temperature=0.0,
        reasoning=False,
    )
    structured: StructuredChat = chat.with_structured_output(schema, method="json_schema")
    return structured


def make_chat_model(
    settings: Settings, *, model: str | None = None, think: bool = False
) -> ChatModel:
    """A ``ChatOllama`` for the active profile's chat model; thinking is off unless asked for.

    - ``num_ctx`` and ``keep_alive`` repeat the warm-up's values (see the module docstring).
    - ``num_predict`` caps the reply (PLAN.md §6.5: 512), so one runaway answer cannot hold the GPU.
      With ``think`` the cap is ``chat_think_num_predict`` instead: a model's reasoning is generated
      tokens too, and a reply budget of 512 can be spent before the answer starts.
    - ``temperature`` is low (0.2): the answer should follow the sources, not improvise.
    - ``reasoning`` is ``think``: ``False`` switches thinking off (PLAN.md §6.5: off by default),
      ``True`` asks for it and the reasoning arrives apart from the answer. A model without a
      thinking mode ignores ``False`` and refuses ``True`` (Ollama answers with an error).
    """
    # The class is built on first use (see ``_ollama_chat_class``): LangChain's Ollama package pulls
    # in the ``ollama`` client and pydantic models, and `geekbot --version` or `geekbot doctor`
    # should not wait for them.
    chosen = model or settings.resolved_chat_model
    batch = settings.chat_num_batch if same_model(chosen, settings.resolved_chat_model) else None
    chat: ChatModel = _ollama_chat_class()(
        model=chosen,
        base_url=settings.ollama_host,
        num_ctx=settings.chat_num_ctx,
        num_batch=batch,
        num_predict=settings.chat_think_num_predict if think else settings.chat_num_predict,
        keep_alive=settings.keep_alive,
        temperature=settings.chat_temperature,
        reasoning=think,
    )
    return chat
