"""Sampler tests: N independent samples, distinct seeds, isolated failures."""

from __future__ import annotations

from csv_analyst.generation.sampler import generate_samples
from csv_analyst.ingestion.loader import load_csv
from csv_analyst.llm.base import LLMProviderError
from csv_analyst.profiling.profiler import profile_dataset

VALID_PLAN_JSON = """
{
  "answerability": "answerable",
  "plan": ["step"],
  "columns_used": ["revenue"],
  "code": "result = df['revenue'].mean()"
}
"""


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.seen_seeds: list[int] = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.seen_seeds.append(seed)
        return self.responses.pop(0)


def _schema(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,revenue\n1,100\n2,200\n", encoding="utf-8")
    return profile_dataset(load_csv(path))


def test_generate_samples_returns_n_with_distinct_seeds(tmp_path):
    schema = _schema(tmp_path)
    llm = FakeLLM([VALID_PLAN_JSON] * 3)

    samples = generate_samples(llm, "average revenue", schema, n=3)

    assert len(samples) == 3
    assert len({s.seed for s in samples}) == 3
    assert all(s.ok for s in samples)
    assert llm.seen_seeds == [s.seed for s in samples]


def test_generate_samples_isolates_failures(tmp_path):
    schema = _schema(tmp_path)
    # sample 2 fails both its attempt and its repair attempt; 1 and 3 succeed.
    llm = FakeLLM([VALID_PLAN_JSON, "bad", "still bad", VALID_PLAN_JSON])

    samples = generate_samples(llm, "average revenue", schema, n=3)

    assert len(samples) == 3
    assert samples[0].ok
    assert not samples[1].ok
    assert samples[1].error is not None
    assert samples[2].ok


def test_generate_samples_isolates_transport_failure(tmp_path):
    """A timeout/connection error from the provider must be treated as one
    lost sample, not propagate and crash the whole run (a slow or briefly
    unreachable Ollama server should degrade the answer, never the process)."""
    schema = _schema(tmp_path)

    class FlakyLLM:
        def __init__(self, responses):
            self.responses = list(responses)

        def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
            r = self.responses.pop(0)
            if r == "TIMEOUT":
                raise LLMProviderError("Ollama request failed (test-model): timed out")
            return r

    llm = FlakyLLM([VALID_PLAN_JSON, "TIMEOUT", VALID_PLAN_JSON])

    samples = generate_samples(llm, "average revenue", schema, n=3)

    assert len(samples) == 3
    assert samples[0].ok
    assert not samples[1].ok
    assert "timed out" in samples[1].error
    assert samples[2].ok
