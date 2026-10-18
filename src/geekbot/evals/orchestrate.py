"""Choose bundles and phases, and run them in order (step S2.6; behind ``geekbot eval retrieval``).

``phases.py`` knows how to run one phase on the bundles it is given. This file decides *which*
bundles each phase gets and in what order the phases run, so the command-line function stays a thin
wrapper and the order can be tested without a terminal:

``rewrite`` (GPU: the chat model rewrites the golden follow-ups, once) comes first because the golden
question sets need its output; then ``ablation``, ``tuning``, ``bakeoff`` and ``abstain``.

The GPU is used by exactly one phase, ``rewrite``. Everything else is CPU, and the reranker and the
embedders run on the CPU by design (PLAN.md §6.5). Checking that the GPU is free before ``rewrite``
is the caller's job (D-071's gate); this module does not look at it.
"""

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from geekbot.agent.condense import CondenseOutput
from geekbot.config import Settings
from geekbot.evals.ablation import AblationConfig
from geekbot.evals.phases import (
    Backends,
    Bundle,
    beir_bundle,
    golden_bundle,
    golden_items,
    only_sets,
    run_ablation,
    run_abstain,
    run_bakeoff,
    run_tuning,
)
from geekbot.evals.rewrites import RewriteCache, cached_rewrites, rewrite_followups
from geekbot.index.embedder import OllamaEmbeddingsClient
from geekbot.llm.chat import make_structured_model

__all__ = ["Phase", "load_rewrites", "rewrite_cache_path", "run_phases", "run_rewrites"]

Log = Callable[[str], None]


class Phase(StrEnum):
    """The phases, in the order ``all`` runs them."""

    REWRITE = "rewrite"
    ABLATION = "ablation"
    TUNING = "tuning"
    BAKEOFF = "bakeoff"
    ABSTAIN = "abstain"


ORDER = tuple(Phase)


def rewrite_cache_path(config: AblationConfig) -> Path:
    """Where the rewrites live: with the results, because they are an input of the golden numbers."""
    return config.results_path / "rewrites.json"


def _rewrite_model(config: AblationConfig, settings: Settings) -> str:
    return config.rewrite_model or settings.resolved_chat_model


def _digest(settings: Settings, model: str) -> str:
    """The digest Ollama reports for ``model`` (part of the cache key: a re-pulled model differs)."""
    client = OllamaEmbeddingsClient(settings.ollama_host, model)
    try:
        return client.model_digest()
    finally:
        client.close()


def run_rewrites(config: AblationConfig, settings: Settings, *, log: Log) -> Mapping[str, str]:
    """Make (or reuse) the standalone rewrite of every golden multi-turn item. Uses the GPU."""
    items, _ = golden_items(config)
    model = _rewrite_model(config, settings)
    digest = _digest(settings, model)
    condenser = make_structured_model(
        settings, CondenseOutput, num_predict=settings.condense_num_predict, model=model
    )
    cache = RewriteCache.load(rewrite_cache_path(config))

    def show(item_id: str, rewrite: str, cached: bool) -> None:
        log(f"  {item_id}{' (cached)' if cached else ''}: {rewrite}")

    log(f"rewrite: {model} ({digest[:12]})")
    return rewrite_followups(
        items,
        condenser,
        cache,
        model=model,
        digest=digest,
        budget_tokens=settings.history_budget_tokens,
        on_item=show,
    )


def load_rewrites(config: AblationConfig, settings: Settings) -> Mapping[str, str]:
    """The rewrites already cached for the configured model; empty if there are none (no GPU use)."""
    path = rewrite_cache_path(config)
    if not path.is_file() or not config.golden.enabled:
        return {}
    items, _ = golden_items(config)
    model = _rewrite_model(config, settings)
    return cached_rewrites(
        items, RewriteCache.load(path), model=model, digest=_digest(settings, model)
    )


def run_phases(
    config: AblationConfig,
    settings: Settings,
    backends: Backends,
    phases: Sequence[Phase],
    *,
    rungs: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    resume: bool = True,
    log: Log,
) -> list[Path]:
    """Run ``phases`` in their fixed order; return the files written. ``ValueError`` on bad names."""
    wanted = [phase for phase in ORDER if phase in phases]
    known = {*config.beir.datasets, *(["golden"] if config.golden.enabled else [])}
    unknown = sorted(set(datasets or ()) - known)
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; configured: {sorted(known)}")

    def chosen(name: str) -> bool:
        return not datasets or name in datasets

    rewrites: Mapping[str, str] = {}
    written: list[Path] = []
    if Phase.REWRITE in wanted and config.golden.enabled:
        rewrites = run_rewrites(config, settings, log=log)
    elif config.golden.enabled and any(p != Phase.REWRITE for p in wanted):
        rewrites = load_rewrites(config, settings)

    def headline() -> list[Bundle]:
        bundles = [
            beir_bundle(config, name, tuning=False) for name in config.beir.datasets if chosen(name)
        ]
        if config.golden.enabled and chosen("golden"):
            bundles.append(golden_bundle(config, rewrites, log=log))
        return bundles

    if Phase.ABLATION in wanted:
        written.append(
            run_ablation(
                config, settings, headline(), backends, rungs=rungs, resume=resume, log=log
            )
        )
    if Phase.TUNING in wanted:
        tuning = [
            beir_bundle(config, name, tuning=True) for name in config.beir.datasets if chosen(name)
        ]
        if config.golden.enabled and chosen("golden"):
            tuning.append(only_sets(golden_bundle(config, rewrites, log=log), ["dev"]))
        written.append(run_tuning(config, settings, tuning, backends, resume=resume, log=log))
    if Phase.BAKEOFF in wanted:
        written.append(run_bakeoff(config, settings, headline(), backends, resume=resume, log=log))
    if Phase.ABSTAIN in wanted:
        if not config.golden.enabled:
            raise ValueError("the abstain calibration needs the golden set (golden.enabled)")
        bundle = golden_bundle(config, rewrites, log=log)
        written.append(run_abstain(config, settings, bundle, backends, rewrites, log=log))
    return written
