"""Wires planning -> validation -> execution -> consensus -> answer
generation into one RunRecord (PLAN.md architecture diagram). This is the
only module that constructs a RunRecord -- every other module returns a
smaller typed piece of it, which is what keeps CLAUDE.md Sec4's rule ("the
LLM never states the final number") enforceable in exactly one place.
"""

from __future__ import annotations

import ast
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from csv_analyst.answer.generator import derived_numbers, generate_answer, partial_view_note, template_answer
from csv_analyst.config import settings
from csv_analyst.contracts import (
    Answerability,
    Confidence,
    ConsensusVerdict,
    Evidence,
    NumericFidelity,
    Reliability,
    RunMeta,
    RunRecord,
    Status,
    TraceEntry,
)
from csv_analyst.execution.runner import ExecutionResult, execute_code
from csv_analyst.sql_engine import execute_sql, sql_retry_hint, validate_sql
from csv_analyst.sql_engine.entities import entity_tables
from csv_analyst.sql_engine.prompt import nullable_columns, real_measure_columns
from csv_analyst.sql_engine.rewrite import drop_null_groups
from csv_analyst.generation.planner import BASELINE_PROMPT_VERSION, PROMPT_VERSION, PlanResponse
from csv_analyst.generation.sampler import generate_samples
from csv_analyst.llm.base import LLMProvider
from csv_analyst.profiling.fact_index import facts_block
from csv_analyst.profiling.schema import SchemaContext
from csv_analyst.validation.ast_guard import validate_code
from csv_analyst.validation.consensus import ConsensusOutcome, evaluate_consensus, results_equal
from csv_analyst.validation.fidelity import check_numeric_fidelity
from csv_analyst.validation.result_guard import build_result_payload
from csv_analyst.validation.retry_hints import retry_hint

ABSTENTION_MESSAGE = "I could not reliably answer this question from the provided dataset."
TIME_LIMIT_MESSAGE = (
    "I could not reliably answer this question within the {minutes:g}-minute time limit for this model."
)

# Cap on the transport-error detail appended to a backend-failure reason. The
# reason string is user-facing (CLI, API response, UI abstain card); the full
# untruncated error stays on the TraceEntry.
MAX_BACKEND_ERROR_CHARS = 200

# A retry after an attempt where every sample failed runs this much hotter
# (capped). At the base temperature (0.2) a new seed alone barely changes the
# output: benchmark runs showed the model resubmitting byte-identical failing
# code on all 9 attempts, so the retry budget bought nothing.
RETRY_TEMPERATURE_STEP = 0.3
MAX_RETRY_TEMPERATURE = 0.8


def planner_prompt_version(engine: str, use_schema: bool = True) -> str:
    """The planner prompt a run uses, as recorded in its metadata and /api/health."""
    if not use_schema:
        return BASELINE_PROMPT_VERSION
    if engine == "sql":
        return settings.sql_planner_prompt + ("+gen" if settings.sql_generated_examples else "")
    return PROMPT_VERSION


def _code_key(code: str) -> str:
    """Whitespace-insensitive identity for "this exact code already failed"."""
    return "\n".join(line.strip() for line in code.strip().splitlines() if line.strip())


def _first_two_agree(executed, non_answerable) -> bool:
    """Adaptive consensus stop rule: samples 1-2 both executed to the same
    result, or both declined to answer."""
    if len(non_answerable) == 2:
        return True
    return len(executed) == 2 and results_equal(executed[0][1], executed[1][1])


def _parses(code: str, engine: str = "pandas") -> bool:
    if engine == "sql":
        return validate_sql(code).ok
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_pipeline(
    llm: LLMProvider,
    df: pd.DataFrame,
    schema: SchemaContext,
    question: str,
    dataset_id: str,
    *,
    model_name: str = "unknown",
    samples: Optional[int] = None,
    max_retries: Optional[int] = None,
    use_schema: bool = True,
    use_ast_gate: bool = True,
    respect_abstention: bool = True,
    seed_offset: int = 0,
    adaptive_consensus: Optional[bool] = None,
    engine: Optional[str] = None,
    time_limit_s: Optional[float] = None,
) -> RunRecord:
    """Defaults are the full reliability configuration (schema grounding +
    AST gate + retry + abstention) -- what every real caller (CLI, API) uses.

    The three `use_*`/`respect_*` flags exist only for the Phase 5 ablation
    harness (CLAUDE.md §16, `evaluation/evaluator/ablation.py`), which turns
    them off one at a time to measure what each control actually buys:
    - `use_schema=False`: the planner gets no profiled schema at all and must
      guess plausible column names from the question (the "naive
      LLM-writes-pandas loop" E0 arm measures).
    - `use_ast_gate=False`: generated code skips `validate_code` and goes
      straight to the sandboxed subprocess -- still contained (execution is
      always sandboxed, per CLAUDE.md §8's "prototype-grade, not
      production-hardened" limitation), just not statically vetted first.
    - `respect_abstention=False`: a plan the model marks non-answerable is
      not routed to a clean abstained record; if it still returned code
      despite the tag, that code is attempted anyway (mirrors a system with
      no dedicated abstention handling).

    `seed_offset` shifts every seed this run uses (default 0 = the seeds any
    normal caller gets). It exists only for the Phase 5 consistency@3 metric
    (`evaluation/evaluator/consistency.py`), which runs the whole pipeline 3
    independent times over the same questions -- without a distinct offset
    per run, the seed formula below is a pure function of the attempt number
    and every "independent" run would trivially produce identical seeds
    (and, since Ollama's `seed` makes generation reproducible, identical
    results), making consistency@3 measure nothing.

    `time_limit_s` (default: `settings.question_time_limit_s(model_name)`,
    PLAN_v2.md D5) is checked before each attempt and sample batch; once it is
    used up the question stops as FAILED with a time-limit reason.
    """
    samples_n = samples or settings.consensus_samples
    # Adaptive consensus (PLAN_v2.md): generate + run samples 1-2 first; the
    # third is generated only if they disagree. Same seeds as the fixed
    # mode, so when 1 and 2 agree the 3-sample majority would have picked
    # the same result anyway -- only the third call is saved.
    adaptive = (settings.consensus_adaptive if adaptive_consensus is None else adaptive_consensus) and samples_n >= 3
    max_retries = settings.max_retries if max_retries is None else max_retries
    temperature = settings.llm_temperature
    # Query engine: "sql" (default since PLAN_v2.md D10) runs one generated SQLite
    # SELECT over table `data`; "pandas" (deprecated) runs generated pandas in
    # the subprocess sandbox.
    engine = engine or settings.query_engine
    if engine not in ("pandas", "sql"):
        raise ValueError(f"unknown engine {engine!r}; expected 'pandas' or 'sql'")
    prompt_version = planner_prompt_version(engine, use_schema)

    run_id = str(uuid.uuid4())
    trace: list[TraceEntry] = []
    started = time.monotonic()
    time_limit = settings.question_time_limit_s(model_name) if time_limit_s is None else time_limit_s

    def out_of_time() -> Optional[RunRecord]:
        if time.monotonic() - started < time_limit:
            return None
        return _build_failed_record(
            run_id=run_id, question=question, dataset_id=dataset_id, samples_total=samples_n,
            retries=attempt - 1, trace=trace, model_name=model_name, temperature=temperature,
            seeds=seeds, started=started, prompt_version=prompt_version,
            reason=TIME_LIMIT_MESSAGE.format(minutes=time_limit / 60),
        )
    feedback: Optional[str] = None
    seeds: list[int] = []
    # Across every attempt/sample: True if that generate call failed with a
    # provider (transport) error, False if it produced a plan. Used after the
    # retry loop to tell a backend outage apart from a genuine dataset
    # limitation -- see PLAN.md Part A.4.
    all_provider_failures: list[bool] = []
    # Code that already failed validation/execution in an earlier sample or
    # attempt -> its error. A repeat is not re-run: same code, same data, same
    # error.
    failed_codes: dict[str, str] = {}
    attempt_temperature = temperature

    # Card H-R fact bank (PLAN_v2.md D12): built once per dataset, added under
    # the schema block, shared by every sample and retry. The trace keeps the
    # block as evidence.
    extra_context = ""
    if use_schema and settings.schema_facts != "off":
        t0 = time.monotonic()
        extra_context = facts_block(schema, df, settings.schema_facts)
        trace.append(TraceEntry(
            attempt=0, sample=0, stage="context", ok=True,
            ms=int((time.monotonic() - t0) * 1000), detail=extra_context or None,
        ))

    for attempt in range(1, max_retries + 2):  # up to max_retries + 1 attempts total
        retries_so_far = attempt - 1
        seeds = [seed_offset + 100 * attempt + i for i in range(samples_n)]
        question_for_llm = question
        if feedback:
            question_for_llm = f"{question}\n\n[Previous attempt note: {feedback}]"

        trace_temperature = attempt_temperature if attempt_temperature != temperature else None

        executed: list[tuple[PlanResponse, ExecutionResult]] = []
        non_answerable: list[PlanResponse] = []
        last_failure_code: Optional[str] = None
        last_failure_error: Optional[str] = None

        batches = [seeds[:2], seeds[2:]] if adaptive else [seeds]
        samples_used = 0
        for batch_no, batch_seeds in enumerate(batches):
            if batch_no > 0 and _first_two_agree(executed, non_answerable):
                break
            if (stopped := out_of_time()) is not None:
                return stopped
            gen_samples = generate_samples(
                llm, question_for_llm, schema, n=len(batch_seeds), temperature=attempt_temperature,
                seeds=batch_seeds, include_schema=use_schema, engine=engine,
                extra_context=extra_context,
            )
            for s in gen_samples:
                samples_used += 1
                idx = samples_used
                if not s.ok:
                    trace.append(TraceEntry(
                        attempt=attempt, sample=idx, stage="generate", ok=False,
                        ms=s.gen_ms, error=s.error, temperature=trace_temperature,
                    ))
                    last_failure_error = s.error
                    all_provider_failures.append(s.error_kind in ("provider", "prompt_too_long"))
                    continue
                trace.append(TraceEntry(
                    attempt=attempt, sample=idx, stage="generate", ok=True,
                    ms=s.gen_ms, temperature=trace_temperature,
                ))
                all_provider_failures.append(False)

                plan = s.plan
                if plan.answerability != Answerability.ANSWERABLE:
                    self_contradicts = (
                        bool(plan.code.strip())
                        and bool(plan.columns_used)
                        and all(c in df.columns for c in plan.columns_used)
                        and _parses(plan.code, engine)
                    )
                    if self_contradicts:
                        # PLAN.md Phase 6A.4: the model tagged this non-answerable
                        # but attached working code over real columns anyway --
                        # trust the code, not the tag, and don't spend an
                        # abstention vote on a sample that isn't really abstaining.
                        trace.append(TraceEntry(
                            attempt=attempt, sample=idx, stage="generate", ok=True, ms=0,
                            error=(
                                f"self-contradiction: tagged {plan.answerability.value} "
                                "but returned code over real columns"
                            ),
                        ))
                    elif respect_abstention:
                        non_answerable.append(plan)
                        continue
                    elif not plan.code.strip():
                        # No dedicated abstention handling in this config, but
                        # there is still nothing to run.
                        last_failure_error = (
                            f"model marked answerability={plan.answerability.value!r} "
                            "and returned no code"
                        )
                        continue
                    # else respect_abstention=False: ignore the tag and attempt
                    # the code anyway -- the naive-loop failure mode E0/E1 measure.

                if engine == "sql" and settings.sql_drop_null_groups and plan.code.strip():
                    plan.code = drop_null_groups(plan.code, nullable_columns(schema))

                code_key = _code_key(plan.code)
                if code_key in failed_codes:
                    trace.append(TraceEntry(
                        attempt=attempt, sample=idx, stage="validate", ok=False, ms=0,
                        error=f"identical to code that already failed: {failed_codes[code_key]}",
                    ))
                    last_failure_code = plan.code
                    last_failure_error = failed_codes[code_key]
                    continue

                if use_ast_gate:
                    t0 = time.monotonic()
                    guard = validate_sql(plan.code) if engine == "sql" else validate_code(plan.code)
                    trace.append(TraceEntry(
                        attempt=attempt, sample=idx, stage="validate", ok=guard.ok,
                        ms=int((time.monotonic() - t0) * 1000), error=None if guard.ok else guard.reason,
                    ))
                    if not guard.ok:
                        last_failure_code = plan.code
                        last_failure_error = guard.reason
                        failed_codes[code_key] = guard.reason or "validation failed"
                        continue

                exec_result = (
                    execute_sql(df, plan.code, real_columns=real_measure_columns(schema),
                                tables=entity_tables(schema) if settings.sql_entity_tables else ())
                    if engine == "sql" else execute_code(df, plan.code)
                )
                trace.append(TraceEntry(
                    attempt=attempt, sample=idx, stage="execute", ok=exec_result.ok,
                    ms=exec_result.ms, error=None if exec_result.ok else exec_result.error,
                ))
                if not exec_result.ok:
                    last_failure_code = plan.code
                    last_failure_error = exec_result.error
                    failed_codes[code_key] = exec_result.error or "execution failed"
                    continue

                executed.append((plan, exec_result))

        # PLAN.md Phase 6A.1: abstention votes join the same largest-group-wins
        # count as executed results -- an abstention is a vote, not merely a
        # fallback used when nothing executed, so it can outvote a minority
        # of samples that happened to produce (disagreeing) results.
        if executed or non_answerable:
            outcome = evaluate_consensus([e for _, e in executed], abstain_votes=len(non_answerable))
            if outcome.abstain_won:
                top_plan = _majority_abstention_plan(non_answerable)
                return _build_abstained_record(
                    run_id=run_id, question=question, dataset_id=dataset_id,
                    answerability=top_plan.answerability, reason=top_plan.reason,
                    clarifying_question=top_plan.clarifying_question, samples_total=samples_used,
                    retries=retries_so_far, trace=trace, model_name=model_name,
                    temperature=temperature, seeds=seeds, started=started,
                    prompt_version=prompt_version,
                )
            if outcome.verdict != ConsensusVerdict.NONE:
                return _build_answered_record(
                    llm=llm, run_id=run_id, question=question, dataset_id=dataset_id, df=df,
                    executed=executed, outcome=outcome, samples_total=samples_used,
                    retries=retries_so_far, trace=trace, model_name=model_name,
                    temperature=temperature, seeds=seeds, started=started,
                    prompt_version=prompt_version,
                )

        if executed:
            disagreeing_codes = "\n---\n".join(p.code for p, _ in executed)
            feedback = (
                f"Independent analyses disagreed on the result. Codes tried:\n{disagreeing_codes}\n"
                "Write a single, simpler, more explicit computation that leaves no room for a "
                "different interpretation."
            )
        elif last_failure_code and last_failure_error:
            feedback = (
                f"Your previous code:\n{last_failure_code}\n"
                f"failed with this exact error:\n{last_failure_error}\n"
                "Write different code that avoids this specific error."
            )
            hint_fn = sql_retry_hint if engine == "sql" else retry_hint
            hint = hint_fn(last_failure_error, columns=[str(c) for c in df.columns])
            if hint:
                feedback = f"{feedback}\nHint: {hint}"
        elif last_failure_error:
            feedback = f"Your previous attempt failed: {last_failure_error}. Try a different approach."
        else:
            feedback = "the generated code failed validation or execution"

        # Only when nothing ran and nobody abstained: a disagreement retry
        # stays at the base temperature, since more randomness there would
        # only widen the disagreement.
        if not executed and not non_answerable and last_failure_code:
            attempt_temperature = min(attempt_temperature + RETRY_TEMPERATURE_STEP, MAX_RETRY_TEMPERATURE)

        if attempt > max_retries:
            break
        # The prompt does not fit the context window: the same prompt on the
        # next attempt would not fit either.
        if any(t.stage == "generate" and t.error and t.error.startswith("prompt too long") for t in trace):
            break

    # CLAUDE.md's response contract (csv_analyst/contracts.py) has no enum value for
    # "infrastructure failure" -- it's deliberately frozen (PLAN.md constraint).
    # So when every single generate call across every attempt failed with a
    # provider/transport error (never a usable plan at all), answerability
    # stays UNSUPPORTED but the reason string carries the real cause instead
    # of implying the dataset itself was the problem.
    backend_reason = None
    if all_provider_failures and all(all_provider_failures):
        # One representative error, not one per sample: every sample in an
        # outage fails the same way, and joining all N produced a ~900-char
        # reason (three identical urllib3 messages, object reprs and all) that
        # is rendered verbatim in the CLI and the UI's abstain card. Truncated
        # too -- the full text is still on the trace entries for anyone who
        # needs it.
        transport_errors = [
            s.error for s in gen_samples if s.error_kind in ("provider", "prompt_too_long") and s.error
        ]
        detail = transport_errors[0] if transport_errors else ""
        if len(detail) > MAX_BACKEND_ERROR_CHARS:
            detail = detail[:MAX_BACKEND_ERROR_CHARS].rstrip() + "..."
        if detail.startswith("prompt too long"):
            backend_reason = (
                "This dataset's description is too long for the model's context window, "
                "so no analysis was attempted (the model would have seen only part of "
                "it). Raise LLM_NUM_CTX or load fewer columns. "
                f"Details: {detail}"
            )
        else:
            backend_reason = (
                "The language model backend could not be reached, so no analysis was "
                "attempted. This is a service error, not a limitation of your dataset."
                + (f" Underlying error: {detail}" if detail else "")
            )

    return _build_failed_record(
        run_id=run_id, question=question, dataset_id=dataset_id, samples_total=samples_n,
        retries=max_retries, trace=trace, model_name=model_name, temperature=temperature,
        seeds=seeds, started=started, prompt_version=prompt_version,
        reason=backend_reason if backend_reason else ABSTENTION_MESSAGE,
    )


def _build_answered_record(
    *, llm, run_id, question, dataset_id, df, executed, outcome: ConsensusOutcome,
    samples_total, retries, trace, model_name, temperature, seeds, started, prompt_version,
) -> RunRecord:
    chosen_plan, chosen_exec = executed[outcome.chosen_index]
    result_payload = build_result_payload(chosen_exec)

    columns_used = [c for c in chosen_plan.columns_used if c in df.columns]
    rows_analyzed = int(df.shape[0])
    if columns_used:
        rows_after_filter = int(df.dropna(subset=columns_used).shape[0])
    else:
        rows_after_filter = rows_analyzed
    # PLAN.md Phase 6A.6: this counts rows that *have* a null in columns_used,
    # not rows the generated code actually dropped -- the code may not filter
    # nulls at all, so a warning claiming they "were excluded" would assert
    # something false. `null_rows_dropped` kept for existing consumers.
    rows_with_nulls = rows_analyzed - rows_after_filter

    evidence = Evidence(
        code=chosen_plan.code, plan=chosen_plan.plan, rows_analyzed=rows_analyzed,
        rows_after_filter=rows_after_filter, columns_used=columns_used,
        null_rows_dropped=rows_with_nulls, rows_with_nulls_in_used_columns=rows_with_nulls,
    )

    answer_text, fidelity_final = _generate_grounded_answer(
        llm, question, chosen_plan, result_payload, temperature, seeds[0] if seeds else None
    )

    warnings: list[str] = []
    if outcome.verdict == ConsensusVerdict.MAJORITY:
        n_agree = len(outcome.agreeing_indices)
        warnings.append(
            f"{n_agree} of {len(executed)} independent analyses agreed; "
            f"{len(executed) - n_agree} disagreed and were discarded"
        )
    if rows_with_nulls:
        warnings.append(f"{rows_with_nulls} row(s) have null values in {columns_used}")

    codes = {p.code for p, _ in executed}
    if outcome.verdict == ConsensusVerdict.UNANIMOUS and len(outcome.agreeing_indices) >= 2:
        # PLAN.md Phase 6A.2: HIGH requires actual agreement between
        # independent samples, not just "the one sample that ran didn't
        # contradict itself".
        confidence = Confidence.HIGH
    elif len(executed) == 1:
        confidence = Confidence.LOW
        warnings.append(f"only 1 of {samples_total} independent samples produced a usable result")
    else:
        confidence = Confidence.MEDIUM

    reliability = Reliability(
        samples=samples_total, samples_executed=len(executed), consensus=outcome.verdict,
        codes_identical=len(codes) == 1, retries=retries, numeric_fidelity=fidelity_final,
        warnings=warnings,
    )

    meta = _build_meta(model_name, temperature, seeds, started, prompt_version)

    return RunRecord(
        run_id=run_id, question=question, dataset_id=dataset_id, status=Status.ANSWERED,
        answerability=Answerability.ANSWERABLE, answer=answer_text, confidence=confidence,
        result=result_payload, evidence=evidence, reliability=reliability, trace=trace, meta=meta,
    )


def _generate_grounded_answer(llm, question, plan, result_payload, temperature, seed):
    """Generate the answer, enforce numeric fidelity, regenerate once, then
    fall back to a deterministic template -- never a third free-form attempt
    (CLAUDE.md Sec10's 2-retry ceiling applies here too)."""
    derived = derived_numbers(result_payload)
    answer_text = generate_answer(llm, question, plan.plan, plan.code, result_payload, temperature=temperature, seed=seed)
    check = check_numeric_fidelity(answer_text, question, result_payload, derived, version=settings.fidelity_check)
    if check.verdict == NumericFidelity.PASS:
        return partial_view_note(answer_text, result_payload), NumericFidelity.PASS

    retry_seed = None if seed is None else seed + 1
    answer_text = generate_answer(llm, question, plan.plan, plan.code, result_payload, temperature=temperature, seed=retry_seed)
    check2 = check_numeric_fidelity(answer_text, question, result_payload, derived, version=settings.fidelity_check)
    if check2.verdict == NumericFidelity.PASS:
        return partial_view_note(answer_text, result_payload), NumericFidelity.REPAIRED

    return partial_view_note(template_answer(result_payload), result_payload), NumericFidelity.FAILED


def _majority_abstention_plan(non_answerable: list[PlanResponse]) -> PlanResponse:
    """PLAN.md Phase 6A.1: when abstention wins the consensus vote, report the
    most common answerability/reason among the abstaining samples rather than
    whichever one happened to run first."""
    counts = Counter(p.answerability for p in non_answerable)
    winning_answerability = counts.most_common(1)[0][0]
    return next(p for p in non_answerable if p.answerability == winning_answerability)


def _build_abstained_record(
    *, run_id, question, dataset_id, answerability, reason, clarifying_question,
    samples_total, retries, trace, model_name, temperature, seeds, started, prompt_version,
) -> RunRecord:
    status = Status.NEEDS_CLARIFICATION if answerability == Answerability.AMBIGUOUS else Status.ABSTAINED
    reliability = Reliability(
        samples=samples_total, samples_executed=0, consensus=ConsensusVerdict.NONE,
        codes_identical=False, retries=retries, numeric_fidelity=NumericFidelity.PASS, warnings=[],
    )
    meta = _build_meta(model_name, temperature, seeds, started, prompt_version)

    return RunRecord(
        run_id=run_id, question=question, dataset_id=dataset_id, status=status,
        answerability=answerability, answer=None, reason=reason or ABSTENTION_MESSAGE,
        clarifying_question=clarifying_question, reliability=reliability, trace=trace, meta=meta,
    )


def _build_failed_record(
    *, run_id, question, dataset_id, samples_total, retries, trace, model_name,
    temperature, seeds, started, prompt_version, reason=ABSTENTION_MESSAGE,
) -> RunRecord:
    reliability = Reliability(
        samples=samples_total, samples_executed=0, consensus=ConsensusVerdict.NONE,
        codes_identical=False, retries=retries, numeric_fidelity=NumericFidelity.PASS, warnings=[],
    )
    meta = _build_meta(model_name, temperature, seeds, started, prompt_version)

    return RunRecord(
        run_id=run_id, question=question, dataset_id=dataset_id, status=Status.FAILED,
        answerability=Answerability.UNSUPPORTED, answer=None, reason=reason,
        reliability=reliability, trace=trace, meta=meta,
    )


def _build_meta(model_name, temperature, seeds, started, prompt_version=PROMPT_VERSION) -> RunMeta:
    return RunMeta(
        model=model_name, prompt_version=prompt_version, temperature=temperature, seeds=seeds,
        engine="sql" if prompt_version.startswith("planner-sql") else "pandas",
        elapsed_ms=int((time.monotonic() - started) * 1000), timestamp=datetime.now(timezone.utc),
    )
