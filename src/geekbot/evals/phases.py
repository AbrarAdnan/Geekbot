"""The phases of ``geekbot eval retrieval`` (step S2.6, PLAN.md §9).

====== ===================================================================================
phase  what it does and writes (under ``evals/results/<name>/``)
====== ===================================================================================
rewrite  makes the cached standalone rewrites of the golden multi-turn items (``rewrites.json``)
ablation the ladder R0 to R7 and the leave-one-out variants on SciFact, NFCorpus and golden
tuning  the rerank-depth x fusion-candidates sweep on the *tuning* splits only (D-065)
bakeoff the embedder and reranker bake-offs
abstain the abstain-threshold calibration on golden dev, checked on golden test
====== ===================================================================================

Every phase is resumable: a result file is rewritten after each finished piece, and a piece that is
already in the file is skipped (``resume=True``). A run that dies after three hours loses the piece
it was in, not the three hours. Index folders are resumable too (``indexing.build_index`` reuses a
finished build), and the embedding cache is shared, so even a lost build costs little.

What the phases need from the outside world (Ollama, the reranker model, the CPU load, the clock) is
bundled in ``Backends`` so the tests can run the whole thing with fakes and no model.
"""

import json
import os
import platform
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from geekbot.config import Settings
from geekbot.evals.ablation import (
    LEAVE_ONE_OUT,
    RUNGS,
    V2_HEADER,
    AblationConfig,
    Strategy,
)
from geekbot.evals.beir import HEADLINE_SPLIT, TUNING_SPLIT, load_beir
from geekbot.evals.cpugate import CpuSnapshot, GateReading, read_snapshot, wait_until_idle
from geekbot.evals.datasets import (
    RETRIEVAL_TYPES,
    GoldenItem,
    Split,
    assign_splits,
    build_qrels,
    load_golden,
    unindexed_sources,
)
from geekbot.evals.indexing import (
    BuiltIndex,
    EmbedderChoice,
    IndexVariant,
    beir_doc_path,
    beir_documents,
    build_index,
    open_embedder,
)
from geekbot.evals.runner import (
    MemoScorer,
    QuerySet,
    build_retriever_for,
    record_of,
    run_queries,
)
from geekbot.index.embedder import EmbedderSpec, EmbeddingsClient
from geekbot.ingest.ids import doc_id_of
from geekbot.ingest.loaders import LoadError, load_file
from geekbot.ingest.loaders.model import LoadedDocument
from geekbot.ingest.pipeline import collect_files
from geekbot.retrieval.calibrate import (
    Labelled,
    best_threshold,
    bootstrap_interval,
    confusion,
    sweep,
)
from geekbot.retrieval.rerank import FastembedScorer, PassageScorer
from geekbot.retrieval.rerank_timing import percentile

__all__ = [
    "RESULT_SCHEMA",
    "Backends",
    "Bundle",
    "RunSpec",
    "beir_bundle",
    "default_backends",
    "golden_bundle",
    "ladder_runs",
    "only_sets",
    "run_ablation",
    "run_abstain",
    "run_bakeoff",
    "run_tuning",
]

RESULT_SCHEMA: Final = 1
Log = Callable[[str], None]
OpenEmbeddings = Callable[
    [EmbedderChoice, IndexVariant], tuple[EmbeddingsClient, EmbedderSpec, Callable[[], None]]
]
ScorerFactory = Callable[[str, Path | None], PassageScorer]


def _quiet(_: str) -> None:
    return None


@dataclass(frozen=True)
class Backends:
    """What the phases get from outside: swap these to run without Ollama or the reranker."""

    open_embeddings: OpenEmbeddings
    make_scorer: ScorerFactory
    gate: Callable[..., GateReading] = wait_until_idle
    snapshot: Callable[[], CpuSnapshot] = read_snapshot
    clock: Callable[[], float] = time.perf_counter


def default_backends(settings: Settings) -> Backends:
    """Ollama for embeddings (on the CPU) and fastembed's cross-encoder for reranking."""

    def opener(
        choice: EmbedderChoice, variant: IndexVariant
    ) -> tuple[EmbeddingsClient, EmbedderSpec, Callable[[], None]]:
        return open_embedder(settings, choice, variant)

    def scorer(model: str, cache_dir: Path | None) -> PassageScorer:
        return FastembedScorer(
            model,
            cache_dir=cache_dir or settings.data_dir / "models",
            threads=settings.rerank_threads,
        )

    return Backends(open_embeddings=opener, make_scorer=scorer)


# --- files -----------------------------------------------------------------------------------


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True), "utf-8")
    partial.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == RESULT_SCHEMA else None


def _environment(settings: Settings, config: AblationConfig) -> dict[str, Any]:
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "profile": settings.profile,
        "top_k": config.top_k,
        "embedder": config.embedder.model_dump(mode="json"),
        "reranker": config.reranker.model_dump(mode="json"),
        "fusion_candidates": config.fusion.candidates,
        "seed": config.seed,
    }


def _result_file(
    config: AblationConfig, name: str, settings: Settings, *, resume: bool = True
) -> dict[str, Any]:
    """The existing result file (when resuming), or the header of a new one."""
    existing = _read_json(config.results_path / f"{name}.json") if resume else None
    if existing is not None:
        return existing
    header: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "kind": name,
        "environment": _environment(settings, config),
    }
    return header


# --- datasets --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    """A dataset ready to evaluate: its documents and its named question sets."""

    name: str
    documents: list[LoadedDocument]
    query_sets: dict[str, QuerySet]


def beir_bundle(config: AblationConfig, name: str, *, tuning: bool) -> Bundle:
    """A BEIR dataset: the whole corpus, and either the headline (test) or the tuning questions.

    The tuning split is capped at ``tuning.max_queries`` by a seeded sample, so the sweep's cost is
    fixed whatever the size of the split. Judgements are translated into ``doc_id`` space.
    """
    split = TUNING_SPLIT[name] if tuning else HEADLINE_SPLIT
    dataset = load_beir(config.beir.folder / name, split)
    documents, _ = beir_documents(dataset)
    queries = dict(dataset.queries)
    if tuning and len(queries) > config.tuning.max_queries:
        chosen = random.Random(config.seed).sample(sorted(queries), config.tuning.max_queries)
        queries = {question_id: queries[question_id] for question_id in sorted(chosen)}
    qrels = {
        question_id: {
            doc_id_of(beir_doc_path(name, document)): grade
            for document, grade in dataset.qrels[question_id].items()
        }
        for question_id in queries
    }
    set_name = "tune" if tuning else "test"
    return Bundle(name, documents, {set_name: QuerySet(f"{name}:{split}", queries, qrels)})


def golden_items(config: AblationConfig) -> tuple[list[GoldenItem], dict[str, Split]]:
    """The golden items and their dev/test split (D-065)."""
    items = load_golden(config.golden.jsonl)
    return items, assign_splits(items)


def golden_bundle(
    config: AblationConfig, rewrites: Mapping[str, str], *, log: Log = _quiet
) -> Bundle:
    """The golden corpus (through the real loaders) and its dev and test question sets.

    A multi-turn item is searched as its cached rewrite (D-076); one with no rewrite is left out
    and the log says how many. An unanswerable item has no relevant document and is not in a
    retrieval set (it is scored by ``run_abstain``). ``ValueError`` if a source of an item is not
    among the loaded documents: that is a setup problem and would look like a bad retriever.
    """
    items, splits = golden_items(config)
    files, _ = collect_files([config.golden.corpus])
    documents: list[LoadedDocument] = []
    for path in files:
        try:
            documents.append(load_file(path))
        except LoadError as error:
            log(f"golden: {path} was not loaded ({error})")
    missing = unindexed_sources(
        items, config.golden.corpus, [doc_id_of(document.path) for document in documents]
    )
    if missing:
        raise ValueError(f"golden sources that were not loaded: {missing[:3]}")
    qrels = build_qrels(items, config.golden.corpus)
    sets: dict[str, QuerySet] = {}
    skipped = 0
    for split in ("dev", "test"):
        queries: dict[str, str] = {}
        for item in items:
            if splits[item.id] != split:
                continue
            if item.type in RETRIEVAL_TYPES:
                queries[item.id] = item.question
            elif item.type == "multi_turn":
                if item.id in rewrites:
                    queries[item.id] = rewrites[item.id]
                else:
                    skipped += 1
        sets[split] = QuerySet(
            f"golden:{split}", queries, {question_id: qrels[question_id] for question_id in queries}
        )
    if skipped:
        log(f"golden: {skipped} multi-turn items have no cached rewrite and are left out")
    return Bundle("golden", documents, sets)


def only_sets(bundle: Bundle, names: Sequence[str]) -> Bundle:
    """The same dataset with only the named question sets (the sweep wants golden's ``dev`` alone)."""
    return replace(bundle, query_sets={n: qs for n, qs in bundle.query_sets.items() if n in names})


# --- the ladder ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One system to evaluate: a rung or a leave-one-out variant."""

    name: str
    change: str
    variant: IndexVariant
    strategy: Strategy


def ladder_runs(names: Sequence[str] | None = None) -> list[RunSpec]:
    """The rungs and the leave-one-out variants that need running, optionally only ``names``.

    "R7 - rerank" is R6 and is not run twice; the report reads it from R6's record.
    """
    specs = [RunSpec(r.name, r.change, r.variant, r.strategy) for r in RUNGS]
    specs += [
        RunSpec(loo.name, f"R7 without {loo.removes}", loo.variant, loo.strategy)
        for loo in LEAVE_ONE_OUT
        if loo.same_as is None
    ]
    if names is None:
        return specs
    unknown = sorted(set(names) - {spec.name for spec in specs})
    if unknown:
        raise ValueError(f"unknown rung(s) {unknown}; known: {[spec.name for spec in specs]}")
    return [spec for spec in specs if spec.name in names]


def _open_index(
    config: AblationConfig,
    backends: Backends,
    bundle: Bundle,
    variant: IndexVariant,
    choice: EmbedderChoice,
    log: Log,
) -> BuiltIndex:
    embeddings, spec, closer = backends.open_embeddings(choice, variant)
    directory = config.scratch_dir / bundle.name / choice.label / variant.key
    cache_path = config.scratch_dir / "cache" / choice.label / "embed_cache.db"

    def progress(done: int, total: int) -> None:
        if done == total or done % (10 * 64) == 0:
            log(f"  {bundle.name}/{variant.key}: {done}/{total} documents")

    built = build_index(
        bundle.documents,
        variant,
        choice,
        directory,
        cache_path,
        embeddings=embeddings,
        spec=spec,
        closer=closer,
        clock=backends.clock,
        on_progress=progress,
    )
    stats = built.stats
    how = "reused" if stats.reused else "built"
    rate = "" if stats.chunks_per_s is None else f", {stats.chunks_per_s:.1f} chunks/s embedded"
    log(
        f"  {bundle.name}/{variant.key}: {how}, {stats.documents} documents, "
        f"{stats.chunks} chunks{rate}"
    )
    return built


def _stats_json(built: BuiltIndex) -> dict[str, Any]:
    stats = built.stats
    return {
        "documents": stats.documents,
        "chunks": stats.chunks,
        "embedded": stats.embedded,
        "cache_hits": stats.cache_hits,
        "embed_seconds": round(stats.embed_seconds, 2),
        "total_seconds": round(stats.total_seconds, 2),
        "chunks_per_s": stats.chunks_per_s,
        "reused": stats.reused,
        "fingerprint": built.spec.fingerprint(),
        "embedder": json.loads(built.spec.describe()),
    }


def run_ablation(
    config: AblationConfig,
    settings: Settings,
    bundles: Sequence[Bundle],
    backends: Backends,
    *,
    rungs: Sequence[str] | None = None,
    resume: bool = True,
    log: Log = _quiet,
) -> Path:
    """The ladder and the leave-one-out variants on each bundle's question sets.

    For each dataset, the variants are built one at a time (and closed before the next, so only one
    Chroma and one SQLite file are open), and every rung that searches a variant runs on every
    question set. Returns the path of ``ablation.json``.
    """
    path = config.results_path / "ablation.json"
    results = _result_file(config, "ablation", settings, resume=resume)
    results["aliases"] = {loo.name: loo.same_as for loo in LEAVE_ONE_OUT if loo.same_as}
    datasets: dict[str, Any] = results.setdefault("datasets", {})
    scorer = backends.make_scorer(config.reranker.model, None)
    specs = ladder_runs(rungs)
    for bundle in bundles:
        entry = datasets.setdefault(bundle.name, {"indexes": {}, "runs": {}})
        entry["questions"] = {name: len(qs.queries) for name, qs in bundle.query_sets.items()}
        by_variant: dict[IndexVariant, list[RunSpec]] = {}
        for spec in specs:
            by_variant.setdefault(spec.variant, []).append(spec)
        for variant, wanted in by_variant.items():
            todo = [
                spec
                for spec in wanted
                if not (resume and _done(entry["runs"], spec.name, bundle.query_sets))
            ]
            if not todo:
                continue
            log(f"{bundle.name}: variant {variant.key} for {[s.name for s in todo]}")
            with _open_index(
                config, backends, bundle, variant, config.embedder.choice(), log
            ) as built:
                entry["indexes"][variant.key] = _stats_json(built)
                for spec in todo:
                    retriever = build_retriever_for(
                        spec.strategy,
                        built,
                        scorer,
                        depth=config.reranker.depth,
                        candidates=config.fusion.candidates,
                    )
                    for set_name, query_set in bundle.query_sets.items():
                        output = run_queries(
                            retriever,
                            query_set.queries,
                            k=config.top_k,
                            warmup=config.warmup_queries,
                            idle_percent=config.idle_load_percent,
                            gate=backends.gate,
                            snapshot=backends.snapshot,
                            clock=backends.clock,
                        )
                        record = record_of(
                            output,
                            query_set.qrels,
                            rung=spec.name,
                            change=spec.change,
                            variant=variant.key,
                            strategy=spec.strategy,
                            question_set=query_set.name,
                        )
                        entry["runs"].setdefault(set_name, {})[spec.name] = record
                        _write_json(path, results)
                        log(
                            f"  {bundle.name}:{set_name} {spec.name}: "
                            f"nDCG@10 {_fmt(record['metrics'].get('ndcg@10'))}, "
                            f"Recall@5 {_fmt(record['metrics'].get('recall@5'))}, "
                            f"p50 {_fmt(record['latency_ms'].get('p50'), 0)} ms"
                            f"{'' if record['gate']['idle'] else '  [CPU not idle]'}"
                        )
    _write_json(path, results)
    return path


def _done(runs: Mapping[str, Any], name: str, query_sets: Mapping[str, QuerySet]) -> bool:
    return all(name in runs.get(set_name, {}) for set_name in query_sets)


def _fmt(value: object, digits: int = 4) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{value:.{digits}f}"


# --- tuning ----------------------------------------------------------------------------------


def run_tuning(
    config: AblationConfig,
    settings: Settings,
    bundles: Sequence[Bundle],
    backends: Backends,
    *,
    resume: bool = True,
    log: Log = _quiet,
) -> Path:
    """Sweep rerank depth and fusion candidates on the tuning questions (never the test ones).

    Quality is read from a memoized scorer, so the grid costs one reranker pass per distinct
    (question, passage). Latency is measured separately and honestly: each depth on its own,
    unmemoized, behind the CPU gate. ``bundles`` must hold the *tuning* question sets.
    """
    path = config.results_path / "tuning.json"
    results = _result_file(config, "tuning", settings, resume=resume)
    datasets: dict[str, Any] = results.setdefault("datasets", {})
    raw_scorer = backends.make_scorer(config.reranker.model, None)
    memo = MemoScorer(raw_scorer)
    for bundle in bundles:
        if resume and bundle.name in datasets and datasets[bundle.name].get("complete"):
            continue
        entry: dict[str, Any] = {"grid": [], "hybrid": [], "latency": [], "complete": False}
        datasets[bundle.name] = entry
        log(f"tuning {bundle.name}")
        with _open_index(
            config, backends, bundle, V2_HEADER, config.embedder.choice(), log
        ) as built:
            entry["index"] = _stats_json(built)
            for query_set in bundle.query_sets.values():
                entry["questions"] = len(query_set.queries)
                entry["set"] = query_set.name
                for candidates in config.tuning.candidates:
                    hybrid = build_retriever_for(
                        "hybrid", built, None, depth=1, candidates=candidates
                    )
                    output = run_queries(
                        hybrid,
                        query_set.queries,
                        k=config.top_k,
                        warmup=0,
                        idle_percent=config.idle_load_percent,
                        gate=backends.gate,
                        snapshot=backends.snapshot,
                        clock=backends.clock,
                    )
                    entry["hybrid"].append(_grid_row(output, query_set, {"candidates": candidates}))
                    for depth in config.tuning.depths:
                        retriever = build_retriever_for(
                            "hybrid_rerank", built, memo, depth=depth, candidates=candidates
                        )
                        output = run_queries(
                            retriever,
                            query_set.queries,
                            k=config.top_k,
                            warmup=0,
                            idle_percent=config.idle_load_percent,
                            gate=backends.gate,
                            snapshot=backends.snapshot,
                            clock=backends.clock,
                        )
                        row = _grid_row(
                            output, query_set, {"depth": depth, "candidates": candidates}
                        )
                        entry["grid"].append(row)
                        log(
                            f"  {bundle.name} depth {depth} candidates {candidates}: "
                            f"nDCG@10 {_fmt(row['metrics'].get('ndcg@10'))}, "
                            f"Recall@5 {_fmt(row['metrics'].get('recall@5'))}"
                        )
                        _write_json(path, results)
                sample = dict(list(query_set.queries.items())[: config.tuning.latency_queries])
                for depth in config.tuning.depths:
                    retriever = build_retriever_for(
                        "hybrid_rerank",
                        built,
                        raw_scorer,
                        depth=depth,
                        candidates=config.fusion.candidates,
                    )
                    output = run_queries(
                        retriever,
                        sample,
                        k=config.top_k,
                        warmup=config.warmup_queries,
                        idle_percent=config.idle_load_percent,
                        gate=backends.gate,
                        snapshot=backends.snapshot,
                        clock=backends.clock,
                    )
                    entry["latency"].append(
                        {
                            "depth": depth,
                            "candidates": config.fusion.candidates,
                            "questions": len(sample),
                            "p50": percentile(output.latencies_ms, 50),
                            "p95": percentile(output.latencies_ms, 95),
                            "stage_ms": output.stage_ms,
                            "gate": output.gate.to_json(),
                        }
                    )
                    log(
                        f"  {bundle.name} latency depth {depth}: p50 "
                        f"{entry['latency'][-1]['p50']:.0f} ms, p95 {entry['latency'][-1]['p95']:.0f} ms"
                    )
        entry["complete"] = True
        _write_json(path, results)
    return path


def _grid_row(output: Any, query_set: QuerySet, identity: Mapping[str, Any]) -> dict[str, Any]:
    record = record_of(output, query_set.qrels, **identity)
    return {key: record[key] for key in (*identity, "questions", "metrics", "docs_per_query")}


# --- bake-offs -------------------------------------------------------------------------------


def run_bakeoff(
    config: AblationConfig,
    settings: Settings,
    bundles: Sequence[Bundle],
    backends: Backends,
    *,
    resume: bool = True,
    log: Log = _quiet,
) -> Path:
    """The embedder bake-off and the reranker bake-off, on the headline question sets.

    **Embedders:** each candidate gets its own v2-header index of each dataset (the held embedder's
    one is reused from the ladder) and is searched dense-only and as the full R7 system; recorded
    are quality, the query-embedding time and the ingest rate (chunks the embedder computed per
    second). **Rerankers:** each candidate re-scores the held embedder's hybrid candidates at the
    configured depth; recorded are quality and the rerank time. Candidates the configuration does
    not name are not run.
    """
    path = config.results_path / "bakeoff.json"
    results = _result_file(config, "bakeoff", settings, resume=resume)
    embedders: dict[str, Any] = results.setdefault("embedders", {})
    rerankers: dict[str, Any] = results.setdefault("rerankers", {})
    held = config.embedder.choice()
    candidates = [held] + [e.choice() for e in config.bakeoff.embedders if e.label != held.label]
    default_scorer = backends.make_scorer(config.reranker.model, None)
    for choice in candidates:
        for bundle in bundles:
            key = f"{choice.label}/{bundle.name}"
            if resume and key in embedders:
                continue
            log(f"bake-off embedder {choice.label} on {bundle.name}")
            with _open_index(config, backends, bundle, V2_HEADER, choice, log) as built:
                row: dict[str, Any] = {"index": _stats_json(built), "runs": {}}
                for strategy in ("dense", "hybrid_rerank"):
                    retriever = build_retriever_for(
                        strategy,
                        built,
                        default_scorer,
                        depth=config.reranker.depth,
                        candidates=config.fusion.candidates,
                    )
                    for set_name, query_set in bundle.query_sets.items():
                        output = run_queries(
                            retriever,
                            query_set.queries,
                            k=config.top_k,
                            warmup=config.warmup_queries,
                            idle_percent=config.idle_load_percent,
                            gate=backends.gate,
                            snapshot=backends.snapshot,
                            clock=backends.clock,
                        )
                        row["runs"].setdefault(set_name, {})[strategy] = record_of(
                            output,
                            query_set.qrels,
                            embedder=choice.label,
                            model=choice.model,
                            strategy=strategy,
                            question_set=query_set.name,
                        )
                embedders[key] = row
                _write_json(path, results)
    held_bundles = list(bundles)
    for reranker in [config.reranker, *config.bakeoff.rerankers]:
        for bundle in held_bundles:
            key = f"{reranker.label}/{bundle.name}"
            if key in rerankers and resume:
                continue
            log(f"bake-off reranker {reranker.label} on {bundle.name}")
            cache = reranker.cache_dir
            scorer = backends.make_scorer(reranker.model, cache)
            with _open_index(config, backends, bundle, V2_HEADER, held, log) as built:
                retriever = build_retriever_for(
                    "hybrid_rerank",
                    built,
                    scorer,
                    depth=reranker.depth,
                    candidates=config.fusion.candidates,
                )
                row = {"model": reranker.model, "depth": reranker.depth, "runs": {}}
                for set_name, query_set in bundle.query_sets.items():
                    output = run_queries(
                        retriever,
                        query_set.queries,
                        k=config.top_k,
                        warmup=config.warmup_queries,
                        idle_percent=config.idle_load_percent,
                        gate=backends.gate,
                        snapshot=backends.snapshot,
                        clock=backends.clock,
                    )
                    row["runs"][set_name] = record_of(
                        output,
                        query_set.qrels,
                        reranker=reranker.label,
                        model=reranker.model,
                        depth=reranker.depth,
                        question_set=query_set.name,
                    )
                rerankers[key] = row
                _write_json(path, results)
    return path


# --- abstain ---------------------------------------------------------------------------------


def run_abstain(
    config: AblationConfig,
    settings: Settings,
    bundle: Bundle,
    backends: Backends,
    rewrites: Mapping[str, str],
    *,
    log: Log = _quiet,
) -> Path:
    """Calibrate the abstain threshold on golden dev and check it on golden test (S2.6 ✍️, S3.5).

    Every golden item, answerable or not, is searched by the full R7 system, and its best chunk's
    rerank score is the signal (``retrieval.calibrate``). The threshold is the F1-optimal one on the
    development items only; the test items are used once, to report how it does. A multi-turn item
    with no cached rewrite is left out (and counted). The result names the reranker and depth: the
    threshold means nothing for another reranker.
    """
    items, splits = golden_items(config)
    scorer = backends.make_scorer(config.reranker.model, None)
    queries: dict[str, str] = {}
    left_out = 0
    for item in items:
        if item.type == "multi_turn":
            if item.id in rewrites:
                queries[item.id] = rewrites[item.id]
            else:
                left_out += 1
        else:
            queries[item.id] = item.question
    log(f"abstain: {len(queries)} questions ({left_out} multi-turn left out for lack of a rewrite)")
    with _open_index(config, backends, bundle, V2_HEADER, config.embedder.choice(), log) as built:
        retriever = build_retriever_for(
            "hybrid_rerank",
            built,
            scorer,
            depth=config.reranker.depth,
            candidates=config.fusion.candidates,
        )
        output = run_queries(
            retriever,
            queries,
            k=config.top_k,
            warmup=0,
            idle_percent=config.idle_load_percent,
            gate=backends.gate,
            snapshot=backends.snapshot,
            clock=backends.clock,
        )
    kinds = {item.id: item.type for item in items}
    labelled = {
        split: [
            Labelled(
                question_id, output.top_scores[question_id], kinds[question_id] != "unanswerable"
            )
            for question_id in queries
            if splits[question_id] == split
        ]
        for split in ("dev", "test")
    }
    chosen = best_threshold(labelled["dev"])
    low, high = bootstrap_interval(labelled["dev"], resamples=1000, seed=config.seed)
    on_test = confusion(labelled["test"], chosen.threshold)

    def describe(point: Any, items_: Sequence[Labelled]) -> dict[str, Any]:
        return {
            "questions": len(items_),
            "unanswerable": sum(1 for i in items_ if not i.answerable),
            "tp": point.tp,
            "fp": point.fp,
            "fn": point.fn,
            "tn": point.tn,
            "precision": point.precision,
            "recall": point.recall,
            "f1": point.f1,
            "accuracy": point.accuracy,
        }

    results: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "kind": "abstain",
        "environment": _environment(settings, config),
        "signal": "the best chunk's cross-encoder score after R7 (hybrid + rerank)",
        "rule": "abstain when the best score is below the threshold; no score also abstains",
        "reranker": {"model": config.reranker.model, "depth": config.reranker.depth},
        "threshold": chosen.threshold,
        "threshold_interval_95": [low, high],
        "multi_turn_left_out": left_out,
        "dev": describe(chosen, labelled["dev"]),
        "test": describe(on_test, labelled["test"]),
        "sweep_dev": [
            {"threshold": p.threshold, "f1": p.f1, "precision": p.precision, "recall": p.recall}
            for p in sweep(labelled["dev"])
        ],
        "scores": [
            {
                "id": i.item_id,
                "split": splits[i.item_id],
                "kind": kinds[i.item_id],
                "score": i.score,
            }
            for split in ("dev", "test")
            for i in labelled[split]
        ],
    }
    path = config.results_path / "abstain.json"
    _write_json(path, results)
    log(
        f"abstain: threshold {chosen.threshold:.3f} (95% range {low:.3f} to {high:.3f}); "
        f"dev F1 {chosen.f1:.3f}, test F1 {on_test.f1:.3f}"
    )
    return path
