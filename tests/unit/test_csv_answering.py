"""Tests for step C5: the CSV side of ``Answering`` and reading a stored upload's name back.

``Answering`` decides which model is on the GPU (the chat model or the CSV model, never both) and
builds the graph context of a CSV turn. The name helpers decide what a thread's saved ``csv_file`` may
be turned into: a file in the uploads folder, and nothing else.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from chatkit import ScriptedChat, ScriptedCondenser, ScriptedRetriever
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.config import Settings
from geekbot.server.answering import Answering, CsvSetup
from geekbot.server.csvmode import stored_file
from geekbot.server.uploads import SHA_CHARS, display_name, safe_upload_name, stored_name_ok

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

SHA = "0123456789abcdef"


class Scorer:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [0.0] * len(passages)


def make(loads: list[str], *, csv: bool = True, fail_csv: bool = False) -> Answering:
    context = AgentContext(
        retriever=ScriptedRetriever(),
        llm=ScriptedChat(),
        model="chat:1b",
        top_k=4,
        context_budget_tokens=1234,
        condenser=ScriptedCondenser(),
        history_budget_tokens=77,
        citation_scorer=Scorer(),
    )

    def load_chat() -> float:
        loads.append("chat")
        return 1.5

    def load_csv() -> float:
        if fail_csv:
            raise RuntimeError("Ollama is down")
        loads.append("csv")
        return 2.5

    setup = CsvSetup(model="csv:1b", condenser=ScriptedCondenser(), load=load_csv) if csv else None
    return Answering(context, make_llm=lambda think: ScriptedChat(), load=load_chat, csv=setup)


# --- which model is on the GPU -----------------------------------------------------------------------


def test_the_chat_model_and_the_csv_model_take_turns_and_each_swap_is_paid_once() -> None:
    loads: list[str] = []
    answering = make(loads)
    assert (answering.ready, answering.csv_ready) == (False, False)
    assert answering.ensure_ready() == 1.5
    assert (answering.ready, answering.csv_ready) == (True, False)
    assert answering.ensure_csv_ready() == 2.5  # the swap: the chat model is let go
    assert (answering.ready, answering.csv_ready) == (False, True)
    assert answering.ensure_csv_ready() == 0.0  # already there
    assert answering.ensure_ready() == 1.5  # and back
    assert (answering.ready, answering.csv_ready) == (True, False)
    assert answering.ensure_ready() == 0.0
    assert loads == ["chat", "csv", "chat"]


def test_a_failed_swap_leaves_the_flags_as_they_were() -> None:
    loads: list[str] = []
    answering = make(loads, fail_csv=True)
    answering.ensure_ready()
    with pytest.raises(RuntimeError, match="Ollama is down"):
        answering.ensure_csv_ready()
    assert (answering.ready, answering.csv_ready) == (True, False)  # the chat model is still loaded


def test_a_server_with_no_csv_setup_cannot_swap_to_it_or_build_its_context() -> None:
    answering = make([], csv=False)
    with pytest.raises(RuntimeError, match="no CSV model"):
        answering.ensure_csv_ready()
    with pytest.raises(RuntimeError, match="no CSV model"):
        answering.csv_context(object())  # type: ignore[arg-type]


# --- the context of a CSV turn ------------------------------------------------------------------------


def test_a_csv_turn_has_no_knowledge_base_no_chat_model_and_the_loaded_table() -> None:
    answering = make([])
    table = object()
    context = answering.csv_context(table)  # type: ignore[arg-type]
    assert isinstance(context.retriever, UnavailableRetriever)
    assert isinstance(context.llm, UnavailableChat)
    assert context.csv is table
    assert context.model == "csv:1b"
    assert context.condenser is answering.csv.condenser  # type: ignore[union-attr]
    assert context.citation_scorer is None  # a CSV answer has no sources to check
    # what is shared is the chat context's, so the budgets are not repeated
    assert (context.context_budget_tokens, context.history_budget_tokens) == (1234, 77)
    # and the chat context itself is untouched
    assert answering.context.model == "chat:1b" and answering.context.csv is None
    assert isinstance(answering.context.retriever, ScriptedRetriever)


# --- reading a stored name back ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        f"{SHA}-sales.csv",
        f"{SHA}-a-b_c.d.tsv",
        f"{SHA}-" + "x" * 100 + ".csv",
        f"{SHA}-caf"
        + chr(0xE9)
        + ".csv",  # letters of every script are kept (``safe_upload_name``)
    ],
)
def test_a_name_save_upload_could_have_made_is_ok(name: str) -> None:
    assert stored_name_ok(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        "sales.csv",  # the command line keeps the plain name: no prefix
        f"{SHA}-",
        f"{SHA[:-1]}-sales.csv",  # 15 hex digits
        f"{SHA.upper()}-sales.csv",
        f"{SHA}-../sales.csv",
        f"{SHA}-..\\sales.csv",
        f"{SHA}-c:sales.csv",
        f"x{SHA}-sales.csv",
        "../" + f"{SHA}-sales.csv",
    ],
)
def test_anything_else_is_not_turned_into_a_path(name: str) -> None:
    assert stored_name_ok(name) is False


def test_display_name_removes_the_prefix_and_leaves_other_names_alone() -> None:
    assert display_name(f"{SHA}-sales.csv") == "sales.csv"
    assert display_name("sales.csv") == "sales.csv"
    assert display_name("notes-final.csv") == "notes-final.csv"
    assert display_name("") == ""


@given(
    st.text(min_size=1, max_size=40),
    st.text(alphabet="0123456789abcdef", min_size=SHA_CHARS, max_size=SHA_CHARS),
)
def test_whatever_a_client_calls_a_file_its_stored_name_reads_back(
    given_name: str, sha: str
) -> None:
    safe = safe_upload_name(given_name + ".csv", frozenset({".csv"}))
    stored = f"{sha}-{safe}"
    assert stored_name_ok(stored)
    assert display_name(stored) == safe


def test_stored_file_finds_a_file_in_the_uploads_folder_and_nothing_else(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    folder = tmp_path / "data" / "uploads"
    folder.mkdir(parents=True)
    (folder / f"{SHA}-sales.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "data" / f"{SHA}-outside.csv").write_text("a\n", encoding="utf-8")

    def found(name: str | None) -> Path | None:
        return asyncio.run(stored_file(settings, name))

    assert found(f"{SHA}-sales.csv") == folder / f"{SHA}-sales.csv"
    assert found(f"{SHA}-missing.csv") is None  # well formed, but the file is gone
    assert found(f"{SHA}-../{SHA}-outside.csv") is None  # a name cannot lead out of the folder
    assert found("sales.csv") is None
    assert found("") is None and found(None) is None
