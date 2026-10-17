"""Tests for steps S3.1 and S3.2: the answer graph, its nodes, generate and the citation check.

No model, no Ollama, no database: the retriever and the chat model are small fakes that record what
they were asked, and the graph is run for real, so the wiring (which node follows which, what lands in
the state) is what is tested. The fakes satisfy the same two Protocols the real objects do.
"""

import json
from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from geekbot.agent.graph import build_graph
from geekbot.agent.nodes import (
    NO_CONTEXT_ANSWER,
    generation_stats,
    has_sources,
    label_names,
    message_text,
    section_trail,
    source_refs,
    validate_citations,
)
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState
from geekbot.domain import Chunk, Label, ScoredChunk
from geekbot.llm.chat import ChatError
from geekbot.retrieval import RetrievalResult

MODEL = "fake-model:1b"


def scored(
    n: int, text: str, *, title: str = "Handbook", section: tuple[str, ...] = ()
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"chunk{n}",
        doc_id="doc",
        path=f"/docs/{title}.md",
        title=title,
        content_hash="h" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=len(text),
        page=None,
        section_path=section,
        header="",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / n)


class FakeRetriever:
    """Returns canned chunks (at most ``k``) and remembers what it was asked."""

    def __init__(
        self, chunks: Sequence[ScoredChunk], timings: dict[str, float] | None = None
    ) -> None:
        self.chunks = tuple(chunks)
        self.timings = timings if timings is not None else {"total": 12.5}
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        return RetrievalResult(chunks=self.chunks[:k], timings_ms=dict(self.timings))


class FakeChat:
    """Answers with a canned message, or raises, and remembers the messages it was given."""

    def __init__(self, reply: BaseMessage | Exception) -> None:
        self.reply = reply
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], /) -> BaseMessage:
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def run(
    retriever: FakeRetriever,
    chat: FakeChat,
    question: str = "How long are chats kept?",
    *,
    top_k: int = 3,
    budget: int = 3000,
) -> AgentState:
    context = AgentContext(
        retriever=retriever,
        llm=chat,
        model=MODEL,
        top_k=top_k,
        context_budget_tokens=budget,
    )
    final: AgentState = build_graph().invoke({"question": question}, context=context)
    return final


TWO_CHUNKS = [
    scored(1, "Chats are kept for 90 days.", section=("Retention",)),
    scored(2, "Backups: 30 days."),
]


# --- the happy path --------------------------------------------------------------------------


def test_a_question_flows_through_retrieve_and_generate() -> None:
    retriever = FakeRetriever(TWO_CHUNKS, {"dense": 3.0, "total": 12.5})
    chat = FakeChat(AIMessage(content="  Ninety days [1].\n"))
    final = run(retriever, chat)
    assert final["answer"] == "Ninety days [1]."  # stripped
    assert final["used"] == 2
    assert final["retrieval_ms"] == {"dense": 3.0, "total": 12.5}
    assert retriever.calls == [("How long are chats kept?", 3)]  # top_k reached the retriever
    assert len(chat.calls) == 1


def test_the_sources_are_plain_numbered_records() -> None:
    final = run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content="ok")))
    first, second = final["sources"]
    assert (first["n"], second["n"]) == (1, 2)
    assert first["section"] == "Retention"
    assert first["text"] == "Chats are kept for 90 days."
    assert first["title"] == "Handbook"
    assert first["page"] is None
    assert first["chunk_id"] == "chunk1"


def test_the_model_receives_the_system_message_then_the_sources_and_the_question() -> None:
    chat = FakeChat(AIMessage(content="ok"))
    run(FakeRetriever(TWO_CHUNKS), chat, "How long are chats kept?")
    (messages,) = chat.calls
    assert [message.type for message in messages] == ["system", "human"]
    human = str(messages[1].content)
    assert "Chats are kept for 90 days." in human
    assert "Backups: 30 days." in human
    assert human.endswith("Question: How long are chats kept?")


def test_the_whole_final_state_is_plain_data_a_checkpointer_could_save() -> None:
    final = run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content="ok")))
    assert json.loads(json.dumps(final)) == final


def test_the_graph_has_the_four_steps_and_the_two_ends() -> None:
    nodes = set(build_graph().get_graph().nodes)
    assert nodes == {
        "__start__",
        "retrieve",
        "generate",
        "validate_citations",
        "no_context",
        "__end__",
    }


# --- citations (S3.2) ------------------------------------------------------------------------


def test_a_citation_to_a_missing_source_is_removed_and_the_raw_text_is_kept() -> None:
    chat = FakeChat(AIMessage(content="Ninety days [1], but only in Paris [7]."))
    final = run(FakeRetriever(TWO_CHUNKS), chat)
    assert final["answer"] == "Ninety days [1], but only in Paris."
    assert final["raw_answer"] == "Ninety days [1], but only in Paris [7]."
    assert final["cited"] == [1]
    assert final["dropped_citations"] == [7]


def test_the_check_uses_the_number_of_sources_in_the_prompt_not_the_number_retrieved() -> None:
    """Three chunks came back but only two fit: ``[3]`` was never shown, so it goes."""
    three = [*TWO_CHUNKS, scored(3, "word " * 200)]  # too big for the budget below
    chat = FakeChat(AIMessage(content="Seven years [3]."))
    final = run(FakeRetriever(three), chat, top_k=3, budget=100)
    assert final["used"] == 2
    assert final["answer"] == "Seven years."
    assert final["dropped_citations"] == [3]


def test_an_answer_that_cites_nothing_is_passed_on_and_says_so() -> None:
    final = run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content="Ninety days.")))
    assert final["answer"] == "Ninety days."
    assert final["cited"] == []
    assert final["dropped_citations"] == []


def test_a_reply_of_nothing_but_missing_citations_is_an_error_not_a_blank_answer() -> None:
    with pytest.raises(ChatError, match="nothing but citations"):
        run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content=" [9] ")))


def test_the_citation_node_reads_the_state_and_changes_only_its_own_keys() -> None:
    update = validate_citations({"answer": "A [1] B [4].", "used": 2})
    assert update == {
        "answer": "A [1] B.",
        "raw_answer": "A [1] B [4].",
        "cited": [1],
        "dropped_citations": [4],
    }


def test_no_context_cites_nothing_and_skips_the_check() -> None:
    final = run(FakeRetriever([]), FakeChat(AIMessage(content="unused")))
    assert final["cited"] == []
    assert final["dropped_citations"] == []
    assert "raw_answer" not in final  # the citation node did not run


# --- labels (S3.2) ---------------------------------------------------------------------------


def test_an_answer_built_from_documents_carries_the_user_and_kb_labels() -> None:
    final = run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content="ok [1]")))
    assert final["answer_labels"] == ["KB", "USER"]


def test_an_answer_the_application_wrote_itself_carries_no_labels() -> None:
    chat = FakeChat(AIMessage(content="unused"))
    assert run(FakeRetriever([]), chat)["answer_labels"] == []  # nothing retrieved
    assert run(FakeRetriever(TWO_CHUNKS), chat, budget=5)["answer_labels"] == []  # nothing fit


def test_label_names_are_sorted_plain_strings() -> None:
    assert label_names({Label.USER, Label.KB}) == ["KB", "USER"]
    assert label_names([]) == []


# --- the source header (S3.2) ----------------------------------------------------------------


def test_a_first_heading_that_only_repeats_the_title_is_dropped() -> None:
    title = "Lantern staff FAQ"
    assert section_trail(title, (title, "Conversations")) == "Conversations"
    assert section_trail(title, ("lantern  STAFF faq",)) == ""  # case and spacing do not matter
    ref = source_refs([scored(1, "a", title="Handbook", section=("Handbook", "Retention"))])[0]
    assert ref["section"] == "Retention"


def test_a_heading_that_differs_from_the_title_stays() -> None:
    assert section_trail("Handbook", ("Retention", "Handbook")) == "Retention > Handbook"
    assert section_trail("Handbook", ()) == ""


# --- what the model call reports -------------------------------------------------------------


def test_the_readings_of_the_call_are_taken_from_ollamas_metadata() -> None:
    reply = AIMessage(
        content="ok",
        response_metadata={
            "load_duration": 250_000_000,
            "prompt_eval_count": 800,
            "eval_count": 40,
            "eval_duration": 2_000_000_000,
        },
    )
    generation = run(FakeRetriever(TWO_CHUNKS), FakeChat(reply))["generation"]
    assert generation["seconds"] >= 0
    assert generation["load_seconds"] == pytest.approx(0.25)
    assert generation["prompt_tokens"] == 800
    assert generation["output_tokens"] == 40
    assert generation["tokens_per_second"] == pytest.approx(20.0)


def test_a_reply_without_metadata_still_gives_the_wall_clock_time() -> None:
    reply = AIMessage(content="ok")
    assert set(generation_stats(reply, 1.5)) == {"seconds"}
    assert generation_stats(reply, 1.5)["seconds"] == 1.5


def test_odd_metadata_is_ignored_not_trusted() -> None:
    reply = AIMessage(
        content="ok",
        response_metadata={"load_duration": "fast", "eval_count": True, "prompt_eval_count": None},
    )
    assert generation_stats(reply, 1.0) == {"seconds": 1.0}


def test_no_generation_time_means_no_tokens_per_second() -> None:
    reply = AIMessage(content="ok", response_metadata={"eval_count": 10, "eval_duration": 0})
    assert generation_stats(reply, 1.0) == {"seconds": 1.0, "output_tokens": 10.0}


# --- nothing to answer from ------------------------------------------------------------------


def test_with_no_sources_the_model_is_never_called() -> None:
    chat = FakeChat(AIMessage(content="must not be used"))
    final = run(FakeRetriever([]), chat)
    assert chat.calls == []
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["used"] == 0
    assert final["sources"] == []
    assert final["generation"] == {}


def test_when_not_even_one_source_fits_the_budget_the_model_is_not_called_either() -> None:
    chat = FakeChat(AIMessage(content="must not be used"))
    final = run(FakeRetriever(TWO_CHUNKS), chat, budget=5)
    assert chat.calls == []
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["used"] == 0
    assert len(final["sources"]) == 2  # they were found; they just did not fit


def test_the_routing_decision_is_a_plain_function_of_the_state() -> None:
    assert has_sources({"question": "q", "sources": [{"n": 1}]}) == "generate"
    assert has_sources({"question": "q", "sources": []}) == "no_context"
    assert has_sources({"question": "q"}) == "no_context"


# --- failures --------------------------------------------------------------------------------


def test_any_failure_of_the_model_becomes_one_chat_error_with_the_cause_kept() -> None:
    original = ConnectionError("Ollama is not running")
    with pytest.raises(ChatError) as raised:
        run(FakeRetriever(TWO_CHUNKS), FakeChat(original))
    assert MODEL in str(raised.value)
    assert "ConnectionError: Ollama is not running" in str(raised.value)
    assert raised.value.__cause__ is original


@pytest.mark.parametrize("content", ["", "   \n", []])
def test_a_reply_with_no_text_is_an_error_not_a_blank_answer(content: str | list[str]) -> None:
    with pytest.raises(ChatError, match="returned no text"):
        run(FakeRetriever(TWO_CHUNKS), FakeChat(AIMessage(content=content)))


# --- small helpers ---------------------------------------------------------------------------


def test_message_text_reads_a_string_or_a_list_of_content_blocks() -> None:
    assert message_text(AIMessage(content="  plain  ")) == "plain"
    blocks: list[str | dict[str, object]] = [
        {"type": "text", "text": "Hello, "},
        {"type": "image_url", "image_url": "ignored"},
        "world",
    ]
    assert message_text(AIMessage(content=blocks)) == "Hello, world"


def test_source_refs_number_the_chunks_from_one_and_join_the_heading_trail() -> None:
    refs = source_refs([scored(1, "a", section=("A", "B")), scored(2, "b")])
    assert [ref["n"] for ref in refs] == [1, 2]
    assert refs[0]["section"] == "A > B"
    assert refs[1]["section"] == ""
    assert refs[0]["score"] == 1.0
    assert refs[1]["score"] == 0.5
