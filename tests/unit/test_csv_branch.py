"""Tests for step C4: the CSV branch of the graph, and a thread's mode being fixed.

The graph is run for real, several turns in a row, with a real ``AsyncSqliteSaver`` on a temporary file
(as in ``test_conversation``) and fakes for everything that would need a model: the loaded CSV, the
rewrite call, and (to prove it is never used) a retriever and chat model that raise if touched.
"""

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.mode import ModeError
from geekbot.agent.runner import read_thread, run_turn
from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.domain import Label, is_tainted
from geekbot.llm.chat import ChatError
from geekbot.retrieval import RetrievalResult

Q1 = "Which region sells the most?"
Q2 = "And by month?"
REWRITE = "Which region sells the most by month?"
TABLE = (
    "North has the highest total revenue, 520.\n\n| region | total |\n|---|---|\n| North | 520 |"
)


@dataclass(frozen=True)
class Reply:
    """What the CSV adapter's ``CsvAnswer`` offers the graph."""

    markdown: str
    status: str = "answered"
    model: str = "qwen-fake:3b"
    seconds: float = 1.5
    model_load_ms: int = 20


class FakeCsv:
    """A loaded CSV: replies in order (an exception is raised), the last one repeats."""

    def __init__(self, *replies: Reply | Exception) -> None:
        self.replies = list(replies) or [Reply(TABLE)]
        self.calls: list[str] = []

    def ask(self, question: str, /) -> Reply:
        self.calls.append(question)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeCondenser:
    def __init__(self, text: str = REWRITE) -> None:
        self.text = text
        self.calls: list[Any] = []

    def invoke(self, messages: Any, /) -> Any:
        self.calls.append(messages)
        return CondenseOutput(standalone_question=self.text)


def csv_context(csv: FakeCsv | None, condenser: FakeCondenser | None = None) -> AgentContext:
    return AgentContext(
        retriever=UnavailableRetriever(),
        llm=UnavailableChat(),
        model="qwen-fake:3b",
        top_k=1,
        context_budget_tokens=3000,
        condenser=condenser or FakeCondenser(),
        csv=csv,
    )


def turn(
    context: AgentContext,
    question: str,
    db: Path,
    *,
    mode: str | None = "csv",
    thread: str = "t1",
    csv_file: str | None = "sales.csv",
) -> dict[str, Any]:
    answered = asyncio.run(
        run_turn(
            question,
            context=context,
            thread=thread,
            threads_db=db,
            mode=mode,
            csv_file=csv_file,
        )
    )
    return answered.final


# --- a CSV turn ------------------------------------------------------------------------------


def test_a_csv_turn_asks_the_loaded_file_and_never_touches_documents_or_the_chat_model(
    tmp_path: Path,
) -> None:
    csv = FakeCsv()
    final = turn(csv_context(csv), Q1, tmp_path / "t.db")  # the Unavailable* stubs raise if used
    assert csv.calls == [Q1]
    assert final["answer"] == TABLE
    assert final["mode"] == "csv"
    assert final["csv_status"] == "answered"
    assert final["csv_file"] == "sales.csv"
    assert final["sources"] == []
    assert (final["used"], final["cited"], final["dropped_citations"]) == (0, [], [])
    assert final["generation"] == {"seconds": 1.5, "load_seconds": 0.02}


def test_the_first_csv_question_makes_no_rewrite_call(tmp_path: Path) -> None:
    condenser = FakeCondenser()
    turn(csv_context(FakeCsv(), condenser), Q1, tmp_path / "t.db")
    assert condenser.calls == []


def test_a_csv_answer_is_labelled_tool_and_user_and_taints_the_thread(tmp_path: Path) -> None:
    final = turn(csv_context(FakeCsv()), Q1, tmp_path / "t.db")
    assert final["answer_labels"] == ["TOOL", "USER"]
    assert final["thread_taint"] == ["TOOL", "USER"]
    assert is_tainted(Label(name) for name in final["thread_taint"])  # a cell can say anything
    assert [m["labels"] for m in final["history"]] == [["USER"], ["TOOL", "USER"]]


def test_a_csv_thread_is_never_marked_as_having_read_documents(tmp_path: Path) -> None:
    final = turn(csv_context(FakeCsv()), Q1, tmp_path / "t.db")
    assert "KB" not in final["thread_taint"]


def test_every_status_the_package_reports_is_an_answer_the_user_reads_and_is_remembered(
    tmp_path: Path,
) -> None:
    for status, text in (
        ("abstained", "The data has no salary column."),
        ("needs_clarification", "Which year do you mean?"),
        ("failed", "No reliable answer: the time limit."),
    ):
        db = tmp_path / f"{status}.db"
        final = turn(csv_context(FakeCsv(Reply(text, status=status))), Q1, db)
        assert final["answer"] == text
        assert final["csv_status"] == status
        assert final["history"][-1]["content"] == text


# --- a follow-up -----------------------------------------------------------------------------


def test_a_csv_follow_up_is_rewritten_with_the_csv_model_and_the_package_gets_the_rewrite(
    tmp_path: Path,
) -> None:
    csv, condenser = FakeCsv(), FakeCondenser()
    context = csv_context(csv, condenser)
    db = tmp_path / "t.db"
    turn(context, Q1, db)
    final = turn(context, Q2, db)
    assert len(condenser.calls) == 1
    human = str(condenser.calls[0][1].content)
    assert f"> User: {Q1}" in human
    assert human.endswith(f"Last message: {Q2}")
    assert csv.calls == [Q1, REWRITE]  # the package has no memory: it is asked a full question
    assert final["question"] == Q2
    assert final["standalone_question"] == REWRITE
    assert final["turn"] == 2
    assert [m["id"] for m in final["history"]] == [
        "t1-user",
        "t1-assistant",
        "t2-user",
        "t2-assistant",
    ]


def test_a_failed_rewrite_asks_the_package_the_users_own_words(tmp_path: Path) -> None:
    class Broken:
        def invoke(self, messages: Any, /) -> Any:
            raise ValueError("not json")

    csv = FakeCsv()
    context = replace(csv_context(csv), condenser=Broken())
    db = tmp_path / "t.db"
    turn(context, Q1, db)
    final = turn(context, Q2, db)
    assert csv.calls == [Q1, Q2]
    assert final["condense_error"] == "ValueError: not json"


def test_the_mode_can_be_left_out_after_the_first_turn(tmp_path: Path) -> None:
    csv = FakeCsv()
    context = csv_context(csv)
    db = tmp_path / "t.db"
    turn(context, Q1, db)
    final = turn(context, Q2, db, mode=None, csv_file=None)
    assert final["mode"] == "csv"
    assert final["csv_file"] == "sales.csv"  # kept from the first turn
    assert len(csv.calls) == 2


def test_the_per_turn_keys_are_cleared_and_the_thread_keys_are_kept(tmp_path: Path) -> None:
    csv = FakeCsv(Reply("first", status="abstained"), Reply("second", status="answered"))
    context = csv_context(csv)
    db = tmp_path / "t.db"
    first = turn(context, Q1, db)
    assert first["csv_status"] == "abstained"
    second = turn(context, Q2, db)
    assert second["csv_status"] == "answered"  # this turn's, not the last turn's
    assert second["requested_mode"] == ""  # an input, cleared once used
    assert second["mode"] == "csv"


# --- a thread keeps its mode -----------------------------------------------------------------


def test_a_csv_thread_cannot_be_asked_as_chat_and_nothing_changes(tmp_path: Path) -> None:
    csv = FakeCsv()
    context = csv_context(csv)
    db = tmp_path / "t.db"
    turn(context, Q1, db)
    with pytest.raises(ModeError, match="csv conversation and cannot become a chat one"):
        turn(context, Q2, db, mode="chat")
    saved = asyncio.run(read_thread(db, "t1"))
    assert saved is not None
    assert (saved["turn"], saved["mode"], len(saved["history"])) == (1, "csv", 2)
    assert csv.calls == [Q1]  # the refused turn did not even reach the package


class FindsNothing:
    """A knowledge base with nothing in it: a chat turn answers 'not found' without a model."""

    def search(self, query: str, k: int) -> RetrievalResult:
        return RetrievalResult(chunks=())


def chat_context() -> AgentContext:
    return AgentContext(
        retriever=FindsNothing(),
        llm=UnavailableChat(),
        model="m",
        top_k=1,
        context_budget_tokens=3000,
    )


def test_a_chat_thread_cannot_be_asked_as_csv(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    csv = FakeCsv()
    started = turn(chat_context(), "anything?", db, mode="chat", csv_file=None)
    assert started["mode"] == "chat"
    with pytest.raises(ModeError, match="chat conversation and cannot become a csv one"):
        turn(csv_context(csv), Q1, db, mode="csv")
    assert csv.calls == []


def test_a_thread_with_no_saved_mode_goes_on_as_chat(tmp_path: Path) -> None:
    """A thread started before C4 has no ``mode`` in its state; asking for nothing means chat."""
    final = turn(chat_context(), "anything?", tmp_path / "t.db", mode=None, csv_file=None)
    assert final["mode"] == "chat"


# --- failures --------------------------------------------------------------------------------


def test_a_package_error_is_one_chat_error_with_the_cause_kept_and_the_turn_is_not_remembered(
    tmp_path: Path,
) -> None:
    original = RuntimeError("the sandbox crashed")
    csv = FakeCsv(Reply(TABLE), original)
    context = csv_context(csv)
    db = tmp_path / "t.db"
    turn(context, Q1, db)
    with pytest.raises(ChatError) as raised:
        turn(context, Q2, db)
    assert "CSV analyst" in str(raised.value)
    assert "RuntimeError: the sandbox crashed" in str(raised.value)
    assert raised.value.__cause__ is original
    saved = asyncio.run(read_thread(db, "t1"))
    assert saved is not None
    assert saved["turn"] == 1


def test_a_csv_thread_without_a_loaded_file_is_an_error_not_a_blank_answer(tmp_path: Path) -> None:
    with pytest.raises(ChatError, match="no CSV file is loaded"):
        turn(csv_context(None), Q1, tmp_path / "t.db")


def test_a_reply_with_no_text_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ChatError, match="returned no text"):
        turn(csv_context(FakeCsv(Reply("  \n"))), Q1, tmp_path / "t.db")
