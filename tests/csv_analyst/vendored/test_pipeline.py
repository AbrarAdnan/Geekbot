"""End-to-end pipeline tests with a fake LLM and a real sandbox execution
(PLAN.md Phase 3 exit gate): answered/unanimous, answered/majority with a
warning, abstained on a missing column, ambiguous -> needs_clarification,
and exhausted-retries -> failed. No test ever finds a digit in an abstained
answer, per CLAUDE.md Sec11."""

from __future__ import annotations

import pytest

from csv_analyst.contracts import Answerability, Confidence, ConsensusVerdict, Status
from csv_analyst.ingestion.loader import load_csv
from csv_analyst.pipeline import run_pipeline
from csv_analyst.profiling.profiler import profile_dataset

# Fake LLMs here return pandas code -> pin the deprecated pandas engine.
pytestmark = pytest.mark.usefixtures("pandas_engine")


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.calls.append(seed)
        return self.responses.pop(0)


def _plan_json(code, columns=("revenue",)):
    cols = ", ".join(f'"{c}"' for c in columns)
    return (
        '{"answerability": "answerable", "plan": ["step"], '
        f'"columns_used": [{cols}], "code": "{code}"}}'
    )


def _data_missing_json():
    return (
        '{"answerability": "data_missing", '
        '"reason": "salary column is not present in the dataset", '
        '"plan": [], "columns_used": [], "code": ""}'
    )


def _ambiguous_json():
    return (
        '{"answerability": "ambiguous", '
        '"clarifying_question": "did you mean revenue or cost?", '
        '"plan": [], "columns_used": [], "code": ""}'
    )


def _schema_and_df(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("region,revenue\nNorth,100\nSouth,200\nNorth,150\n", encoding="utf-8")
    load_result = load_csv(path)
    schema = profile_dataset(load_result)
    return load_result.df, schema


def test_unanimous_consensus_answers(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([plan, plan, plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, adaptive_consensus=False)

    assert record.status == Status.ANSWERED
    assert record.answerability == Answerability.ANSWERABLE
    assert record.confidence == Confidence.HIGH
    assert record.reliability.consensus == ConsensusVerdict.UNANIMOUS
    assert record.reliability.samples_executed == 3
    assert record.reliability.retries == 0
    assert record.answer == "The average revenue is 150.0."
    assert record.evidence.code == "result = df['revenue'].mean()"


def test_majority_consensus_answers_with_warning(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    agree = _plan_json("result = df['revenue'].mean()")
    outlier = _plan_json("result = df['revenue'].sum()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([agree, agree, outlier, answer])

    # Fixed mode: adaptive would stop after the two agreeing samples.
    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, adaptive_consensus=False)

    assert record.status == Status.ANSWERED
    assert record.confidence == Confidence.MEDIUM
    assert record.reliability.consensus == ConsensusVerdict.MAJORITY
    assert record.reliability.warnings


def test_data_missing_column_abstains_with_no_digits(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    llm = FakeLLM([_data_missing_json()])

    record = run_pipeline(llm, df, schema, "average employee salary?", "ds1", samples=1)

    assert record.status == Status.ABSTAINED
    assert record.answerability == Answerability.DATA_MISSING
    assert record.answer is None
    assert not any(ch.isdigit() for ch in record.reason)


def test_ambiguous_question_needs_clarification(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    llm = FakeLLM([_ambiguous_json()])

    record = run_pipeline(llm, df, schema, "what about the metric?", "ds1", samples=1)

    assert record.status == Status.NEEDS_CLARIFICATION
    assert record.answer is None
    assert record.clarifying_question == "did you mean revenue or cost?"


def test_persistent_disagreement_fails_after_retries(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    # Every attempt produces 3 mutually-disagreeing scalar results -> consensus
    # NONE every time -> retried until the retry ceiling, then FAILED.
    attempt_plans = [
        _plan_json("result = df['revenue'].mean()"),
        _plan_json("result = df['revenue'].sum()"),
        _plan_json("result = df['revenue'].max()"),
    ]
    llm = FakeLLM(attempt_plans * 2)  # max_retries=1 -> 2 attempts total

    record = run_pipeline(llm, df, schema, "revenue figure?", "ds1", samples=3, max_retries=1)

    assert record.status == Status.FAILED
    assert record.answer is None
    assert record.reliability.retries == 1


def test_invalid_code_exhausts_retries_and_fails(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    bad_plan = _plan_json("result = eval('df.revenue.mean()')")
    llm = FakeLLM([bad_plan, bad_plan])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=1)

    assert record.status == Status.FAILED
    assert record.answer is None


class AlwaysDownLLM:
    """Every call raises a transport failure -- PLAN.md Part A.4: this must
    not be reported as a dataset limitation."""

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        from csv_analyst.llm.base import LLMProviderError
        raise LLMProviderError("404 Client Error: Not Found for url: http://x/api/generate")


def test_backend_unreachable_reports_service_error_not_dataset_limitation(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    llm = AlwaysDownLLM()

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=1)

    assert record.status == Status.FAILED
    assert record.answer is None
    assert "dataset" not in record.reason.lower() or "not a limitation" in record.reason.lower()
    assert "backend" in record.reason.lower() or "service error" in record.reason.lower()


# --- Phase 5 ablation knobs (CLAUDE.md §16 / PLAN.md Phase 5: E0-E3) --------
# These exercise csv_analyst.pipeline.run_pipeline's use_schema / use_ast_gate /
# respect_abstention flags directly -- the evaluation ablation harness
# (evaluation/evaluator/ablation.py) is the only real caller that turns any
# of them off; the production CLI/API never does.


class RecordingLLM:
    """Like FakeLLM but keeps every prompt sent, so tests can assert on what
    the model actually saw (e.g. whether the schema was included)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_use_schema_false_hides_real_columns_from_the_prompt(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = RecordingLLM([plan, answer])

    run_pipeline(llm, df, schema, "average value?", "ds1", samples=1, use_schema=False)

    assert "region" not in llm.prompts[0]
    assert "Rows:" not in llm.prompts[0]


def test_use_ast_gate_false_skips_static_validation_but_sandbox_still_blocks_eval(tmp_path):
    """With the AST gate off, code that validate_code would reject is handed
    straight to the sandbox instead of being rejected up front. The sandbox's
    restricted builtins (no `eval`) still stop it -- it just fails at
    execution time with a NameError instead of a pre-flight rejection, which
    is exactly the E0/E1 "ungated but still sandboxed" arm CLAUDE.md §16
    describes."""
    df, schema = _schema_and_df(tmp_path)
    bad_plan = _plan_json("result = eval('df.revenue.mean()')")
    llm = FakeLLM([bad_plan])

    record = run_pipeline(
        llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=0, use_ast_gate=False,
    )

    assert record.status == Status.FAILED
    assert not any(t.stage == "validate" for t in record.trace)
    execute_entries = [t for t in record.trace if t.stage == "execute"]
    assert execute_entries and not execute_entries[0].ok
    assert "eval" in (execute_entries[0].error or "").lower() or "NameError" in (execute_entries[0].error or "")


def test_respect_abstention_false_attempts_code_despite_non_answerable_tag(tmp_path):
    """A plan the model tags 'unsupported' but still attaches code to (the
    naive-loop failure mode a real abstention feature exists to prevent) gets
    executed instead of cleanly abstained when respect_abstention=False."""
    df, schema = _schema_and_df(tmp_path)
    mistagged = (
        '{"answerability": "unsupported", "reason": "not sure", '
        '"plan": ["step"], "columns_used": ["revenue"], '
        '"code": "result = df[\'revenue\'].mean()"}'
    )
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([mistagged, answer])

    record = run_pipeline(
        llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=0,
        respect_abstention=False,
    )

    # The tag was ignored and the code was actually run -- no dedicated
    # abstained/needs_clarification record was built for it.
    assert record.status == Status.ANSWERED
    assert record.status != Status.ABSTAINED
    assert record.status != Status.NEEDS_CLARIFICATION


def test_abstention_votes_outvote_a_minority_result(tmp_path):
    """PLAN.md Phase 6A.1: two samples abstain, one produces a result -- the
    two abstention votes must outvote the lone result, not be discarded."""
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    llm = FakeLLM([_data_missing_json(), _data_missing_json(), plan])

    record = run_pipeline(llm, df, schema, "average employee salary?", "ds1", samples=3)

    assert record.status == Status.ABSTAINED
    assert record.answerability == Answerability.DATA_MISSING
    assert record.answer is None


def test_single_usable_sample_gets_low_confidence(tmp_path):
    """PLAN.md Phase 6A.2: one surviving sample is not 'unanimous, high
    confidence' -- it's the only vote that succeeded, not agreement."""
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1)

    assert record.status == Status.ANSWERED
    assert record.confidence == Confidence.LOW
    assert any("only 1 of 1" in w for w in record.reliability.warnings)


def test_self_contradicting_sample_executes_and_is_not_an_abstention_vote(tmp_path):
    """PLAN.md Phase 6A.4: the model tagged this 'unsupported' but attached
    working code over a real column -- trust the code, not the tag."""
    df, schema = _schema_and_df(tmp_path)
    contradicting = (
        '{"answerability": "unsupported", "reason": "not sure", '
        '"plan": ["step"], "columns_used": ["revenue"], '
        '"code": "result = df[\'revenue\'].mean()"}'
    )
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([contradicting, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1)

    assert record.status == Status.ANSWERED
    assert record.evidence.code == "result = df['revenue'].mean()"
    assert any("self-contradiction" in (t.error or "") for t in record.trace)


def test_respect_abstention_false_with_no_code_still_fails_cleanly(tmp_path):
    """When the tagged-non-answerable plan has no code at all (the normal
    case once schema grounding is on), there's nothing to execute either way
    -- respect_abstention=False must not crash, it just can't route through
    the dedicated abstention builder."""
    df, schema = _schema_and_df(tmp_path)
    llm = FakeLLM([_data_missing_json()])

    record = run_pipeline(
        llm, df, schema, "average employee salary?", "ds1", samples=1, max_retries=0,
        respect_abstention=False,
    )

    assert record.status == Status.FAILED
    assert record.answer is None


# --- retry budget: identical failing code, retry temperature -----------------


class TemperatureLLM:
    """FakeLLM that also records the temperature of every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.temperatures: list[float] = []

    def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
        self.temperatures.append(temperature)
        return self.responses.pop(0)


def test_identical_failing_code_is_not_executed_again(tmp_path, monkeypatch):
    import csv_analyst.pipeline as pipeline

    df, schema = _schema_and_df(tmp_path)
    bad = _plan_json("result = df['revenue'].no_such_method()")
    executed_codes = []
    real_execute = pipeline.execute_code

    def counting_execute(frame, code, *a, **k):
        executed_codes.append(code)
        return real_execute(frame, code, *a, **k)

    monkeypatch.setattr(pipeline, "execute_code", counting_execute)
    llm = FakeLLM([bad, bad, bad, bad])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=2, max_retries=1)

    assert record.status == Status.FAILED
    assert len(executed_codes) == 1  # 4 samples, one real execution
    repeats = [t for t in record.trace if t.error and t.error.startswith("identical to code")]
    assert len(repeats) == 3


def test_retry_after_every_sample_failed_runs_hotter(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    bad = _plan_json("result = df['revenue'].no_such_method()")
    good = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = TemperatureLLM([bad, good, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=1)

    assert record.status == Status.ANSWERED
    base = record.meta.temperature
    assert llm.temperatures[0] == base
    assert llm.temperatures[1] > base
    retry_generate = [t for t in record.trace if t.stage == "generate" and t.attempt == 2]
    assert retry_generate[0].temperature == llm.temperatures[1]


def test_disagreement_retry_keeps_base_temperature(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    attempt_plans = [
        _plan_json("result = df['revenue'].mean()"),
        _plan_json("result = df['revenue'].sum()"),
        _plan_json("result = df['revenue'].max()"),
    ]
    llm = TemperatureLLM(attempt_plans * 2)

    record = run_pipeline(llm, df, schema, "revenue figure?", "ds1", samples=3, max_retries=1)

    assert record.status == Status.FAILED
    assert set(llm.temperatures) == {record.meta.temperature}


# Adaptive consensus (PLAN_v2.md): samples 1-2 first, sample 3 only on disagreement.

def test_adaptive_consensus_stops_after_two_agreeing_samples(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([plan, plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, adaptive_consensus=True)

    assert record.status == Status.ANSWERED
    assert llm.calls[:2] == [100, 101]  # sample 3 (seed 102) never generated
    assert record.reliability.samples == 2
    assert record.reliability.consensus == ConsensusVerdict.UNANIMOUS
    assert record.confidence == Confidence.HIGH


def test_adaptive_consensus_generates_third_sample_on_disagreement(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    mean_plan = _plan_json("result = df['revenue'].mean()")
    sum_plan = _plan_json("result = df['revenue'].sum()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([mean_plan, sum_plan, mean_plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, adaptive_consensus=True)

    assert record.status == Status.ANSWERED
    assert llm.calls[:3] == [100, 101, 102]
    assert record.reliability.samples == 3
    assert record.reliability.consensus == ConsensusVerdict.MAJORITY
    assert record.result.preview == [[150.0]]


def test_adaptive_consensus_stops_when_first_two_abstain(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    llm = FakeLLM([_data_missing_json(), _data_missing_json()])

    record = run_pipeline(llm, df, schema, "average salary?", "ds1", samples=3, adaptive_consensus=True)

    assert record.status == Status.ABSTAINED
    assert llm.calls == [100, 101]


# SQL engine option (PLAN_v2.md B4): same pipeline, SQL instead of pandas.

def _sql_plan_json(sql, columns=("revenue",)):
    cols = ", ".join(f'"{c}"' for c in columns)
    return (
        '{"answerability": "answerable", "plan": ["step"], '
        f'"columns_used": [{cols}], "code": "{sql}"}}'
    )


def test_sql_engine_answers_with_consensus(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    plan = _sql_plan_json("SELECT AVG(revenue) FROM data")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([plan, plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, engine="sql")

    assert record.status == Status.ANSWERED
    assert record.result.preview == [[150.0]]
    assert record.evidence.code == "SELECT AVG(revenue) FROM data"
    assert record.meta.engine == "sql"
    assert record.meta.prompt_version == "planner-sql-v4-tables+gen"


def test_sql_engine_rejects_a_write_and_uses_the_valid_sample(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    bad = _sql_plan_json("DELETE FROM data")
    good = _sql_plan_json("SELECT SUM(revenue) FROM data")
    answer = '{"answer": "Total revenue is 450.0."}'
    # Adaptive: samples 1-2 are both rejected, so sample 3 is generated and answers.
    llm = FakeLLM([bad, bad, good, answer])

    record = run_pipeline(llm, df, schema, "total revenue?", "ds1", samples=3, engine="sql")

    assert record.status == Status.ANSWERED
    assert record.result.preview == [[450.0]]
    assert record.reliability.samples_executed == 1
    rejected = [t for t in record.trace if t.stage == "validate" and not t.ok]
    assert rejected and "SELECT" in rejected[0].error


def test_engine_follows_setting_and_sql_is_the_shipped_default(tmp_path):
    from csv_analyst.config import Settings

    assert Settings.model_fields["query_engine"].default == "sql"  # PLAN_v2.md D10
    df, schema = _schema_and_df(tmp_path)
    plan = _plan_json("result = df['revenue'].mean()")
    answer = '{"answer": "The average revenue is 150.0."}'
    llm = FakeLLM([plan, plan, answer])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3)

    assert record.meta.engine == "pandas"  # pinned by the pandas_engine fixture


def test_prompt_too_long_stops_at_once_and_says_why(tmp_path):
    from csv_analyst.llm.base import PromptTooLongError

    df, schema = _schema_and_df(tmp_path)

    class OverflowLLM:
        calls = 0

        def generate_json(self, prompt, *, system=None, temperature=0.2, seed=None, schema=None):
            OverflowLLM.calls += 1
            raise PromptTooLongError("prompt too long for the model's context window: about 9000 tokens, 7680 available")

    record = run_pipeline(OverflowLLM(), df, schema, "average revenue?", "ds1", samples=3)

    assert record.status != Status.ANSWERED
    assert "context window" in record.reason
    assert {t.attempt for t in record.trace if t.stage == "generate"} == {1}  # no retries


def test_time_limit_stops_before_any_model_call(tmp_path):
    df, schema = _schema_and_df(tmp_path)
    llm = FakeLLM([])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=3, time_limit_s=0)

    assert record.status == Status.FAILED
    assert "time limit" in record.reason
    assert llm.calls == []


def test_time_limit_is_checked_between_attempts(tmp_path, monkeypatch):
    import csv_analyst.pipeline as pipeline

    df, schema = _schema_and_df(tmp_path)
    now = [0.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: now[0])
    bad_plan = _plan_json("result = eval('df.revenue.mean()')")

    class SlowLLM(FakeLLM):
        def generate_json(self, prompt, **kwargs):
            now[0] += 700.0  # each model call takes "700 s"
            return super().generate_json(prompt, **kwargs)

    llm = SlowLLM([bad_plan])

    record = run_pipeline(llm, df, schema, "average revenue?", "ds1", samples=1, max_retries=2,
                          time_limit_s=600)

    assert record.status == Status.FAILED
    assert "10-minute time limit" in record.reason
    assert len(llm.calls) == 1  # attempt 1 ran; attempt 2 was never started


def test_time_limit_follows_the_model_tier():
    from csv_analyst.config import Settings

    # GEEKBOT EDIT (C1): the heavy tier (and its 900 s limit) is removed.
    s = Settings(llm_model_main="main", llm_model_fast="fast")
    assert s.question_time_limit_s("main") == 600
    assert s.question_time_limit_s("fast") == 300
    assert s.question_time_limit_s("some-hosted-model") == 600
