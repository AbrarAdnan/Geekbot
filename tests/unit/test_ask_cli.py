"""Tests for steps S3.1 to S3.3: ``geekbot ask`` and ``ask-timing``, with a fake chat model and Ollama.

The documents are ingested for real (SQLite, FTS5, Chroma) with a fake embedder, as in the search
tests. The chat model, the model swap and the Ollama client are replaced, so the command's own work
is what runs: opening the workspace, building the retriever, loading the chat model first, running the
graph, and printing the answer with its sources and readings. Questions use ``--mode lexical`` where
the result must be predictable (BM25 finds exactly the documents that contain the words).
"""

import ast
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from typer.testing import CliRunner

from geekbot.agent.nodes import NO_CONTEXT_ANSWER
from geekbot.cli import app
from geekbot.config import Settings
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.workspace import Workspace
from geekbot.llm.models import OllamaError, SwapResult
from geekbot.retrieval import RetrievalResult, SearchMode

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

runner = CliRunner()
TOPICS = ("zebra", "quartz", "bread", "river")
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=len(TOPICS),
    with_header=True,
)
CORPUS = {
    "zebras.md": "Zebras have black and white stripes.",
    "quartz.txt": "Quartz is a hard mineral.",
    "bread.md": "Bread needs flour and sourdough bread uses a starter.",
    "river.md": "The river Brahmaputra flows south. Ticket Qx9812 is about it.",
}
ANSWER = "Use flour and a starter [1]."


class TopicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        counts = [float(sum(w.startswith(topic) for w in words)) for topic in TOPICS]
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm if norm else 0.01 for c in counts]


class FakeChat:
    """Streams a canned reply (as one chunk, or as ``pieces``), or raises; remembers its calls."""

    def __init__(
        self, reply: BaseMessage | Exception, *, pieces: Sequence[AIMessageChunk] | None = None
    ) -> None:
        self.reply = reply
        self.pieces = pieces
        self.calls: list[list[BaseMessage]] = []

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.pieces is not None:
            yield from self.pieces
            return
        yield AIMessageChunk(
            content=self.reply.content,
            response_metadata=self.reply.response_metadata,
            additional_kwargs=self.reply.additional_kwargs,
        )


class FakeLlm:
    """What the command did with the chat model and the swap, and what to make them do."""

    def __init__(self) -> None:
        self.chat = FakeChat(
            AIMessage(
                content=ANSWER,
                response_metadata={
                    "load_duration": 0,
                    "eval_count": 12,
                    "eval_duration": 1_000_000_000,
                },
            )
        )
        self.swaps: list[str] = []
        self.swap_error: Exception | None = None
        self.unloads: list[list[str]] = []
        self.made: list[dict[str, object]] = []  # the keyword arguments make_chat_model got


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLlm:
    state = FakeLlm()

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def swap(_client: object, _settings: object, mode: str, **_: object) -> SwapResult:
        state.swaps.append(mode)
        if state.swap_error is not None:
            raise state.swap_error
        return SwapResult(
            loaded="gemma3:4b", unloaded=("qwen2.5-coder:3b",), unload_s=0.1, warm_up_s=1.5
        )

    def unload(_client: object, names: Sequence[str], **_: object) -> tuple[str, ...]:
        state.unloads.append(list(names))
        return tuple(names)

    def make_chat(_settings: object, **kwargs: object) -> FakeChat:
        state.made.append(dict(kwargs))
        return state.chat

    monkeypatch.setattr("geekbot.cli.OllamaClient", FakeClient)
    monkeypatch.setattr("geekbot.llm.models.swap_to_mode", swap)
    monkeypatch.setattr("geekbot.llm.models.unload_and_wait", unload)
    monkeypatch.setattr("geekbot.llm.chat.make_chat_model", make_chat)
    return state


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "data"))
    real_open = Workspace.open.__func__  # type: ignore[attr-defined]
    prefixed = PrefixedEmbeddings(TopicEmbeddings(), document_prefix="doc: ", query_prefix="q: ")

    def fake_open(
        cls: type[Workspace], settings: Settings, *, reset_dense: bool = False, **_: Any
    ) -> Workspace:
        return real_open(cls, settings, reset_dense=reset_dense, embeddings=prefixed, spec=SPEC)

    monkeypatch.setattr(Workspace, "open", classmethod(fake_open))
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(docs)]).exit_code == 0


def ask(*args: str) -> Any:
    return runner.invoke(app, ["ask", *args])


# --- a question with an answer ---------------------------------------------------------------


def test_the_answer_comes_with_its_sources_and_the_readings_of_the_run(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("how do I bake bread", "--mode", "lexical", "--top-k", "2")
    assert result.exit_code == 0, result.output
    assert ANSWER in result.output
    assert "Sources cited:" in result.output
    assert "[1] bread.md" in result.output
    assert "zebras.md" not in result.output  # only the sources the model was shown
    assert "labels: KB, USER (tainted)" in result.output
    assert "retrieval: total " in result.output
    assert "gemma3:4b: " in result.output  # the model line
    assert "model load 0.00 s" in result.output
    assert "12 tokens out" in result.output
    assert "12.0 tokens/s" in result.output


def test_the_chat_model_is_loaded_first_and_the_user_is_told(indexed: None, llm: FakeLlm) -> None:
    result = ask("bread", "--mode", "lexical")
    assert llm.swaps == ["chat"]
    assert "gemma3:4b ready (1.5 s to load, unloaded qwen2.5-coder:3b)" in result.output
    assert "asking gemma3:4b" in result.output


def test_the_model_is_given_the_sources_and_the_question(indexed: None, llm: FakeLlm) -> None:
    ask("how do I bake bread", "--mode", "lexical")
    (messages,) = llm.chat.calls
    human = str(messages[1].content)
    assert "Bread needs flour and sourdough bread uses a starter." in human
    assert human.endswith("Question: how do I bake bread")


def test_show_prompt_prints_the_exact_messages(indexed: None, llm: FakeLlm) -> None:
    result = ask("bread", "--mode", "lexical", "--show-prompt")
    assert "--- system ---" in result.output
    assert "--- human ---" in result.output
    assert "> Bread needs flour and sourdough bread uses a starter." in result.output  # data-marked
    assert "Question: bread" in result.output


# --- citations -------------------------------------------------------------------------------


def test_a_citation_to_a_source_that_was_not_shown_is_removed_and_named(
    indexed: None, llm: FakeLlm
) -> None:
    llm.chat = FakeChat(AIMessage(content="Use flour [1] and a pinch of salt [4]."))
    result = ask("how do I bake bread", "--mode", "lexical")
    assert result.exit_code == 0, result.output
    assert "Use flour [1] and a pinch of salt." in result.output
    assert "[4]" in result.output  # named in the note below, not in the answer
    assert "removed from the answer (no such source in the prompt): [4]" in result.output


def test_sources_the_answer_does_not_cite_are_listed_apart(indexed: None, llm: FakeLlm) -> None:
    llm.chat = FakeChat(AIMessage(content="Only the first one matters [1]."))
    result = ask("bread river", "--mode", "lexical", "--top-k", "2")
    assert result.exit_code == 0, result.output
    listing = result.output.split("Sources cited:")[1]
    cited, _, not_cited = listing.partition("Read, not cited:")
    # Which of the two documents ranks first does not matter: each is in exactly one group.
    assert ("bread.md" in cited) != ("river.md" in cited)
    assert ("bread.md" in not_cited) != ("river.md" in not_cited)
    assert ("bread.md" in cited) != ("bread.md" in not_cited)


def test_an_answer_that_cites_nothing_lists_what_the_model_read(
    indexed: None, llm: FakeLlm
) -> None:
    llm.chat = FakeChat(AIMessage(content="I could not find it in the documents."))
    result = ask("how do I bake bread", "--mode", "lexical")
    assert "Sources cited:" not in result.output
    assert "Read, but the answer cites none:" in result.output
    assert "[1] bread.md" in result.output


# --- the question ----------------------------------------------------------------------------


def test_an_empty_question_is_refused_before_any_model_is_loaded(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("   ", "--mode", "lexical")
    assert result.exit_code == 1
    assert "error: the question is empty" in result.output
    assert llm.swaps == []


def test_a_question_that_is_too_long_is_refused_before_any_model_is_loaded(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("word " * 5000, "--mode", "lexical")
    assert result.exit_code == 1
    assert "the question is too long" in result.output
    assert llm.swaps == []
    assert llm.chat.calls == []


# --- a question with nothing to answer from --------------------------------------------------


def test_a_question_with_no_matching_chunk_is_answered_without_calling_the_model(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("zzzz qqqq", "--mode", "lexical")
    assert result.exit_code == 0, result.output
    assert NO_CONTEXT_ANSWER in result.output
    assert "Sources" not in result.output
    assert "labels:" not in result.output  # the application wrote that sentence, not a document
    assert llm.chat.calls == []


def test_the_default_way_of_finding_chunks_is_hybrid_with_the_reranker(
    indexed: None, llm: FakeLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[SearchMode, int]] = []

    class NothingFound:
        def search(self, query: str, k: int) -> RetrievalResult:
            return RetrievalResult(chunks=())

    def fake_build(
        mode: SearchMode, workspace: object, settings: object, *, candidates: int
    ) -> Any:
        seen.append((mode, candidates))
        return NothingFound()

    monkeypatch.setattr("geekbot.retrieval.factory.build_retriever", fake_build)
    assert ask("anything").exit_code == 0
    assert seen == [(SearchMode.HYBRID_RERANK, 50)]


# --- failures --------------------------------------------------------------------------------


def test_without_an_index_it_says_what_to_do_and_touches_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLlm
) -> None:
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "empty"))
    result = ask("bread", "--mode", "lexical")
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output
    assert llm.swaps == []


def test_a_failing_chat_model_is_one_line_not_a_traceback(indexed: None, llm: FakeLlm) -> None:
    llm.chat = FakeChat(ConnectionError("Ollama is not running"))
    result = ask("bread", "--mode", "lexical")
    assert result.exit_code == 1
    expected = "error: the chat model 'gemma3:4b' failed (ConnectionError: Ollama is not running)"
    assert expected in result.output
    assert "Traceback" not in result.output


def test_a_failing_swap_stops_before_any_question_is_asked(indexed: None, llm: FakeLlm) -> None:
    llm.swap_error = OllamaError("Ollama POST /api/generate failed: connection refused")
    result = ask("bread", "--mode", "lexical")
    assert result.exit_code == 1
    assert "connection refused" in result.output
    assert llm.chat.calls == []


def test_a_question_is_required() -> None:
    assert runner.invoke(app, ["ask"]).exit_code == 2


# --- streaming, time to first token, thinking (S3.3) ------------------------------------------


def pieces_of(*texts: str) -> list[AIMessageChunk]:
    return [AIMessageChunk(content=piece) for piece in texts]


def test_the_readings_include_the_time_to_the_first_token(indexed: None, llm: FakeLlm) -> None:
    result = ask("how do I bake bread", "--mode", "lexical")
    assert "first token " in result.output  # measured even when nobody watches the stream


def test_stream_prints_the_answer_once_then_the_sources_and_readings(
    indexed: None, llm: FakeLlm
) -> None:
    llm.chat = FakeChat(AIMessage(content=""), pieces=pieces_of("Use flour ", "and a starter [1]."))
    result = ask("how do I bake bread", "--mode", "lexical", "--stream")
    assert result.exit_code == 0, result.output
    assert result.output.count("Use flour and a starter [1].") == 1  # streamed, not printed again
    assert "Sources cited:" in result.output
    assert "[1] bread.md" in result.output
    assert "first token " in result.output
    assert "corrected" not in result.output


def test_stream_prints_the_corrected_answer_when_the_check_removed_a_mark(
    indexed: None, llm: FakeLlm
) -> None:
    llm.chat = FakeChat(AIMessage(content=""), pieces=pieces_of("Use flour [1] ", "and salt [4]."))
    result = ask("how do I bake bread", "--mode", "lexical", "--stream")
    assert result.exit_code == 0, result.output
    assert "Use flour [1] and salt [4]." in result.output  # what was streamed live
    assert "corrected answer (the citation check removed a mark" in result.output
    corrected = result.output.split("corrected answer")[1]
    assert "Use flour [1] and salt." in corrected
    assert "removed from the answer (no such source in the prompt): [4]" in result.output


def test_stream_still_prints_the_applications_own_sentence_when_the_model_is_not_called(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("zzzz qqqq", "--mode", "lexical", "--stream")
    assert result.exit_code == 0, result.output
    assert NO_CONTEXT_ANSWER in result.output
    assert llm.chat.calls == []


def test_without_think_the_model_is_built_with_thinking_off(indexed: None, llm: FakeLlm) -> None:
    result = ask("bread", "--mode", "lexical")
    assert "with thinking on" not in result.output
    assert llm.made == [{"think": False}]


def test_think_builds_a_thinking_model_and_says_so(indexed: None, llm: FakeLlm) -> None:
    result = ask("bread", "--mode", "lexical", "--think")
    assert result.exit_code == 0, result.output
    assert llm.made == [{"think": True}]
    assert "asking gemma3:4b with thinking on..." in result.output


def reasoning_chat() -> FakeChat:
    return FakeChat(
        AIMessage(content=""),
        pieces=[
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Check source 1."}),
            AIMessageChunk(content="Use flour [1]."),
        ],
    )


def test_the_reasoning_is_counted_but_not_printed_unless_asked(indexed: None, llm: FakeLlm) -> None:
    llm.chat = reasoning_chat()
    result = ask("bread", "--mode", "lexical", "--think")
    assert "thinking 15 characters" in result.output  # the readings line
    assert "Check source 1." not in result.output


def test_show_thinking_prints_the_reasoning_under_the_answer(indexed: None, llm: FakeLlm) -> None:
    llm.chat = reasoning_chat()
    result = ask("bread", "--mode", "lexical", "--think", "--show-thinking")
    assert "thinking (15 characters):\n  Check source 1." in result.output


def test_show_thinking_streams_the_reasoning_before_the_answer(indexed: None, llm: FakeLlm) -> None:
    llm.chat = reasoning_chat()
    result = ask("bread", "--mode", "lexical", "--think", "--show-thinking", "--stream")
    assert result.exit_code == 0, result.output
    assert "thinking: Check source 1.\n" in result.output
    assert result.output.index("Check source 1.") < result.output.index("Use flour [1].")


def test_show_thinking_without_think_is_refused_before_any_model_is_loaded(
    indexed: None, llm: FakeLlm
) -> None:
    result = ask("bread", "--mode", "lexical", "--show-thinking")
    assert result.exit_code == 1
    assert "--show-thinking needs --think" in result.output
    assert llm.swaps == []


def test_a_model_that_cannot_think_gets_a_hint_instead_of_a_traceback(
    indexed: None, llm: FakeLlm
) -> None:
    cannot = RuntimeError("registry.ollama.ai/library/gemma3:4b does not support thinking")
    llm.chat = FakeChat(cannot)
    result = ask("bread", "--mode", "lexical", "--think")
    assert result.exit_code == 1
    assert "does not support thinking" in result.output
    assert "Drop --think" in result.output
    assert "Traceback" not in result.output


def test_without_think_that_error_gets_no_hint(indexed: None, llm: FakeLlm) -> None:
    llm.chat = FakeChat(RuntimeError("does not support thinking"))
    result = ask("bread", "--mode", "lexical")
    assert "Drop --think" not in result.output


# --- ask-timing (S3.3) -------------------------------------------------------------------------


def timing(*args: str) -> Any:
    return runner.invoke(app, ["ask-timing", *args])


def test_ask_timing_runs_the_cold_rounds_then_the_warm_passes_and_prints_the_report(
    indexed: None, llm: FakeLlm
) -> None:
    result = timing(
        "--question",
        "how do I bake bread",
        "--mode",
        "lexical",
        "--cold-rounds",
        "2",
        "--warm-runs",
        "3",
    )
    assert result.exit_code == 0, result.output
    assert llm.swaps == ["chat", "chat", "chat"]  # the first load, then one reload per cold round
    assert llm.unloads == [["gemma3:4b"], ["gemma3:4b"]]  # the chat model itself is unloaded
    assert len(llm.chat.calls) == 5  # two cold first questions, three warm passes
    assert "chat model gemma3:4b (proto profile)" in result.output
    assert "first retrieval in this process:" in result.output
    assert "first question after a fresh load of the model (2 rounds):" in result.output
    assert "steady state, model warm (3 runs over 1 question):" in result.output
    assert "time to first token" in result.output
    assert "model loads during questions: 0 of 5 runs" in result.output


def test_ask_timing_without_cold_rounds_skips_the_unloading(indexed: None, llm: FakeLlm) -> None:
    result = timing(
        "--question", "bread", "--mode", "lexical", "--cold-rounds", "0", "--warm-runs", "1"
    )
    assert result.exit_code == 0, result.output
    assert llm.unloads == []
    assert "first question after a fresh load" not in result.output


def test_ask_timing_takes_several_questions_in_turn(indexed: None, llm: FakeLlm) -> None:
    questions = ["--question", "bread", "--question", "river"]
    result = timing(*questions, "--mode", "lexical", "--cold-rounds", "0", "--warm-runs", "2")
    assert result.exit_code == 0, result.output
    asked = [str(call[1].content).rsplit("Question: ", 1)[1] for call in llm.chat.calls]
    assert asked == ["bread", "river", "bread", "river"]
    assert "(4 runs over 2 questions)" in result.output


def test_ask_timing_asks_for_thinking_only_when_told(indexed: None, llm: FakeLlm) -> None:
    timing("--question", "bread", "--mode", "lexical", "--cold-rounds", "0", "--warm-runs", "1")
    timing(
        "--question",
        "bread",
        "--mode",
        "lexical",
        "--cold-rounds",
        "0",
        "--warm-runs",
        "1",
        "--think",
    )
    assert llm.made == [{"think": False}, {"think": True}]


def test_ask_timing_refuses_a_question_the_documents_cannot_answer(
    indexed: None, llm: FakeLlm
) -> None:
    result = timing("--question", "zzzz qqqq", "--mode", "lexical", "--cold-rounds", "0")
    assert result.exit_code == 1
    assert "no source was found for 'zzzz qqqq'" in result.output
    assert llm.chat.calls == []


def test_ask_timing_checks_the_questions_before_it_loads_any_model(
    indexed: None, llm: FakeLlm
) -> None:
    result = timing("--question", "bread", "--question", "   ", "--mode", "lexical")
    assert result.exit_code == 1
    assert "error: the question is empty" in result.output
    assert llm.swaps == []


def test_ask_timing_needs_a_question_and_an_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLlm
) -> None:
    assert timing().exit_code == 2
    monkeypatch.setenv("GEEKBOT_DATA_DIR", str(tmp_path / "empty"))
    result = timing("--question", "bread", "--mode", "lexical")
    assert result.exit_code == 1
    assert "geekbot ingest" in result.output
    assert llm.swaps == []


def test_a_failing_model_during_ask_timing_is_one_line(indexed: None, llm: FakeLlm) -> None:
    llm.chat = FakeChat(ConnectionError("Ollama is not running"))
    result = timing("--question", "bread", "--mode", "lexical", "--cold-rounds", "0")
    assert result.exit_code == 1
    assert (
        "error: the chat model 'gemma3:4b' failed (ConnectionError: Ollama is not running)"
        in result.output
    )
    assert "Traceback" not in result.output


# --- start-up cost ---------------------------------------------------------------------------


def test_the_cli_does_not_import_the_slow_libraries_at_start_up() -> None:
    """`geekbot --version` and `geekbot doctor` must not wait for LangGraph or LangChain."""
    cli = Path(__file__).resolve().parents[2] / "src" / "geekbot" / "cli.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    slow = ("langgraph", "langchain", "geekbot.agent", "geekbot.llm.chat", "chromadb")
    assert not [name for name in top_level if name.startswith(slow)]
