"""The host-app entry point (csv_analyst.session): load once, ask many, chat-ready replies."""

from __future__ import annotations

import pytest

from csv_analyst import CsvAnalystSession, IngestionError, LLMProviderError, Status, make_provider, to_markdown

CSV_BYTES = b"region,revenue\nNorth,100\nSouth,200\nNorth,150\n"


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        return self.responses.pop(0)


def _sql_plan(code):
    return ('{"answerability": "answerable", "plan": ["step"], "columns_used": ["revenue", "region"], '
            f'"code": "{code}"}}')


def test_from_bytes_loads_and_describes():
    session = CsvAnalystSession.from_bytes(CSV_BYTES, filename="sales.csv", llm=FakeLLM([]))

    assert session.df.shape == (3, 2)
    assert "sales.csv" in session.describe() and "3 rows, 2 columns" in session.describe()


def test_from_path_uses_the_file_name(tmp_path):
    path = tmp_path / "regions.csv"
    path.write_bytes(CSV_BYTES)

    assert CsvAnalystSession.from_path(path, llm=FakeLLM([])).name == "regions.csv"


def test_bad_csv_raises_ingestion_error():
    with pytest.raises(IngestionError):
        CsvAnalystSession.from_bytes(b"", filename="empty.csv", llm=FakeLLM([]))


def test_ask_answers_from_the_data_and_renders_evidence():
    code = "SELECT region, SUM(revenue) AS total FROM data GROUP BY region ORDER BY total DESC"
    llm = FakeLLM([_sql_plan(code), _sql_plan(code), '{"answer": "North has the highest total, 250."}'])
    session = CsvAnalystSession.from_bytes(CSV_BYTES, llm=llm)

    record = session.ask("Which region has the highest total revenue?")
    reply = to_markdown(record)

    assert record.status == Status.ANSWERED
    assert record.result.preview[0] == ["North", 250.0]
    assert reply.startswith("North has the highest total, 250.")
    assert "*Confidence: high*" in reply
    assert f"```sql\n{code}\n```" in reply
    assert "| region | total |" in reply and "| North | 250.0 |" in reply


def test_refusal_renders_the_reason_only():
    refusal = ('{"answerability": "data_missing", "reason": "There is no salary column.", '
               '"plan": [], "columns_used": [], "code": ""}')
    session = CsvAnalystSession.from_bytes(CSV_BYTES, llm=FakeLLM([refusal, refusal]))

    record = session.ask("What is the average salary?")

    assert record.status == Status.ABSTAINED
    assert to_markdown(record) == "There is no salary column."


def test_make_provider():
    assert make_provider("ollama", "some-model").model == "some-model"
    # GEEKBOT EDIT (C1): the hosted providers are removed, so "anthropic" is now an unknown provider.
    with pytest.raises(ValueError):
        make_provider("anthropic", "claude-x")
    with pytest.raises(ValueError):
        make_provider("nope")
