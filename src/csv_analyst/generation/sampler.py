"""N independent planner samples (PLAN.md Phase 2 / reliability §2 locked
decision: N-sample consensus). Cross-checking the results is Phase 3
(validation/consensus.py) — this module only produces the independent
samples, isolating one sample's failure from the rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from csv_analyst.config import settings
from csv_analyst.generation.planner import PlanGenerationError, PlanResponse, generate_plan
from csv_analyst.llm.base import LLMProvider, LLMProviderError, PromptTooLongError
from csv_analyst.profiling.schema import SchemaContext

DEFAULT_SEEDS = [11, 22, 33, 44, 55]


@dataclass
class Sample:
    seed: int
    plan: Optional[PlanResponse]
    error: Optional[str] = None
    # "provider" = transport/backend failure (timeout, 404, connection refused);
    # "prompt_too_long" = the prompt would not fit the context window (not sent);
    # "parse" = model responded but produced unparseable JSON. Distinguishes a
    # broken backend from a bad model response -- see PLAN.md Part A.2.
    error_kind: Optional[str] = None
    gen_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.plan is not None


def generate_samples(
    llm: LLMProvider,
    question: str,
    schema: SchemaContext,
    *,
    n: Optional[int] = None,
    temperature: Optional[float] = None,
    seeds: Optional[list[int]] = None,
    include_schema: bool = True,
    engine: str = "pandas",
    extra_context: str = "",
) -> list[Sample]:
    n = n or settings.consensus_samples
    temperature = settings.llm_temperature if temperature is None else temperature
    seeds = (seeds or DEFAULT_SEEDS)[:n]

    samples: list[Sample] = []
    for seed in seeds:
        t0 = time.monotonic()
        try:
            plan = generate_plan(
                llm, question, schema, temperature=temperature, seed=seed, include_schema=include_schema,
                engine=engine, extra_context=extra_context,
            )
            gen_ms = int((time.monotonic() - t0) * 1000)
            samples.append(Sample(seed=seed, plan=plan, gen_ms=gen_ms))
        except PromptTooLongError as exc:
            gen_ms = int((time.monotonic() - t0) * 1000)
            samples.append(Sample(seed=seed, plan=None, error=str(exc), error_kind="prompt_too_long", gen_ms=gen_ms))
        except LLMProviderError as exc:
            gen_ms = int((time.monotonic() - t0) * 1000)
            samples.append(Sample(seed=seed, plan=None, error=str(exc), error_kind="provider", gen_ms=gen_ms))
        except PlanGenerationError as exc:
            gen_ms = int((time.monotonic() - t0) * 1000)
            samples.append(Sample(seed=seed, plan=None, error=str(exc), error_kind="parse", gen_ms=gen_ms))

    return samples
