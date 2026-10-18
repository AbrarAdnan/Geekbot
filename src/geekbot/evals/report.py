"""The report writer: result files in, one Markdown report out (step S2.6, PLAN.md §9).

Everything in the report is computed from the JSON files the phases wrote (``evals/results/<name>/``)
and nothing is typed in by hand, so a number in the report can always be traced to a file and the
whole report can be regenerated with ``geekbot eval report``. The statistics are computed *here*,
from the stored per-question scores, so changing the number of resamples never needs a re-run.

How the plan's rules show up (PLAN.md §9, D-065, D-076):

- **Δ first, then its interval, then p.** Every comparison is printed as "Δ [low, high]". The Holm
  family (R7 against R0 on SciFact nDCG@10, NFCorpus nDCG@10 and golden Recall@5: m = 3) also gets
  the permutation p and the Holm-adjusted p. Every other comparison (rung to rung, leave-one-out)
  is descriptive: Δ and interval only, no p, no verdict, and the words "significant" and "not
  significant" appear nowhere in this file.
- **Each number names its system.** A ladder row names the rung and the single change it made; the
  header names the embedder, the reranker, the chunking of each index variant and the depth.
- **Claims discipline** (CLAUDE.md rule 8): BEIR rows are "standard BEIR datasets with standard
  metrics"; golden rows are "Claude-authored golden v0, provisional" (D-067); latency rows carry the
  CPU gate's reading and are marked when the PC was not idle.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from geekbot.evals.stats import (
    DEFAULT_CONFIDENCE,
    compare_paired,
    format_comparison,
    holm_adjust,
    paired_bootstrap_ci,
)

__all__ = ["FILES", "HOLM_FAMILY", "load_results", "render_report"]

FILES: Final = ("ablation", "tuning", "bakeoff", "abstain")
# (dataset, question set, metric): the three headline comparisons, fixed in advance (D-065).
HOLM_FAMILY: Final = (
    ("scifact", "test", "ndcg@10"),
    ("nfcorpus", "test", "ndcg@10"),
    ("golden", "test", "recall@5"),
)
_LADDER_METRICS: Final = ("ndcg@10", "recall@5", "recall@10", "mrr@10")
_LADDER: Final = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")
_LOO: Final = ("R7 - BM25", "R7 - rerank", "R7 - header")
_NOTE: Final = {
    "scifact": "BEIR SciFact, test queries (standard BEIR dataset, standard metrics)",
    "nfcorpus": "BEIR NFCorpus, test queries (standard BEIR dataset, standard metrics)",
    "golden": "Claude-authored golden v0, provisional (D-067): not a headline claim",
}


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    """The result files that exist in ``directory``, by kind (``ablation``, ``tuning``, ...)."""
    found: dict[str, dict[str, Any]] = {}
    for kind in FILES:
        path = directory / f"{kind}.json"
        if path.is_file():
            found[kind] = json.loads(path.read_text(encoding="utf-8"))
    return found


# --- small helpers ---------------------------------------------------------------------------


def _num(value: object, digits: int = 4) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _paired(
    a: Mapping[str, Any], b: Mapping[str, Any], metric: str
) -> tuple[list[float], list[float]]:
    """The two runs' per-question scores of ``metric``, for the questions both have, in id order."""
    first = a["per_query"].get(metric, {})
    second = b["per_query"].get(metric, {})
    shared = sorted(set(first) & set(second))
    return [first[q] for q in shared], [second[q] for q in shared]


def _delta(
    a: Mapping[str, Any] | None,
    b: Mapping[str, Any] | None,
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> str:
    """``+0.0123 [-0.0050, +0.0301]``: how much ``b`` gained over ``a``, with its 95% interval."""
    if a is None or b is None:
        return "n/a"
    x, y = _paired(a, b, metric)
    if not x:
        return "n/a"
    ci = paired_bootstrap_ci(x, y, resamples=resamples, seed=seed)
    return f"{ci.delta:+.4f} [{ci.low:+.4f}, {ci.high:+.4f}]"


def _run(entry: Mapping[str, Any], set_name: str, rung: str) -> Mapping[str, Any] | None:
    """A rung's record, following the aliases ("R7 - rerank" is R6)."""
    rung = entry.get("aliases", {}).get(rung, rung)
    found: Mapping[str, Any] | None = entry["runs"].get(set_name, {}).get(rung)
    return found


def _idle_mark(run: Mapping[str, Any]) -> str:
    return "" if run["gate"]["idle"] else " (CPU not idle)"


# --- sections --------------------------------------------------------------------------------


def _header(results: Mapping[str, Mapping[str, Any]], name: str) -> list[str]:
    environment = next(iter(results.values()))["environment"]
    embedder, reranker = environment["embedder"], environment["reranker"]
    lines = [
        f"# Retrieval evaluation report: {name}",
        "",
        "_Generated by `geekbot eval report` from the result files next to it. Do not edit by hand;_",
        "_change the code or the configuration and regenerate._",
        "",
        "## What was run",
        "",
        (
            f"- Date: {environment['date']}; Python {environment['python']}; {environment['platform']}; "
            f"{environment['cpu_count']} logical CPUs; profile `{environment['profile']}`."
        ),
        (
            f"- Embedder held constant through the ladder: `{embedder['model']}` "
            f"(prefixes `{embedder['document_prefix']}` / `{embedder['query_prefix']}` from rung R3 on)."
        ),
        (
            f"- Reranker: `{reranker['model']}`, depth {reranker['depth']}; fusion: "
            f"{environment['fusion_candidates']} candidates from each side, RRF k=60."
        ),
        (
            f"- Every system returns its top {environment['top_k']} chunks; a document is ranked at "
            "its best chunk. Per-question seed for the bootstrap and permutation tests: "
            f"{environment['seed']}."
        ),
        "",
        "Read this first:",
        "",
        "- **Golden v0 is Claude-authored and provisional** (D-067). Its rows are not a headline claim.",
        (
            "- **Chunk size changes how many distinct documents ten chunks cover** (`docs/query`), so "
            "rung R4 (new chunking) includes that effect; it is shown, not removed."
        ),
        (
            "- **R0 is a replica**, built from v1's code with the 1.x splitter; CP2's v1 reference run "
            "checks it (PLAN.md §9)."
        ),
        (
            "- **Latency rows** carry the CPU gate's reading; a row marked (CPU not idle) is not a "
            "measurement of the system. The service-level targets stay targets until measured on the "
            "target models."
        ),
        (
            "- Δ is *new minus old*, with its 95% paired-bootstrap interval. Only the three headline "
            "comparisons have a p-value."
        ),
        "",
    ]
    return lines


def _headline(ablation: Mapping[str, Any], *, resamples: int, seed: int) -> list[str]:
    rows: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    pvalues: list[float] = []
    comparisons = []
    for dataset, set_name, metric in HOLM_FAMILY:
        entry = ablation["datasets"].get(dataset)
        if entry is None:
            continue
        base, full = _run(entry, set_name, "R0"), _run(entry, set_name, "R7")
        if base is None or full is None:
            continue
        x, y = _paired(base, full, metric)
        comparison = compare_paired(x, y, resamples=resamples, seed=seed)
        comparisons.append((f"{dataset} {metric} (R7 vs R0)", comparison))
        pvalues.append(comparison.p_value)
        rows.append((dataset, base, full, metric))
    lines = ["## Headline: the full system against the v1 replica", ""]
    if not comparisons:
        return lines + ["Not run: R0 and R7 are needed on SciFact, NFCorpus and golden test.", ""]
    adjusted = holm_adjust(pvalues)
    lines += [
        (
            f"Family of m = {len(HOLM_FAMILY)} comparisons fixed in advance (D-065), Holm-adjusted. "
            f"Intervals are {DEFAULT_CONFIDENCE:.0%}."
        ),
        "",
    ]
    lines += [
        f"- {format_comparison(name, c, adjusted_p=p)}"
        for (name, c), p in zip(comparisons, adjusted, strict=True)
    ]
    lines.append("")
    if len(comparisons) < len(HOLM_FAMILY):
        lines += [f"Only {len(comparisons)} of {len(HOLM_FAMILY)} family members were run.", ""]
    return lines


def _ladder(ablation: Mapping[str, Any], *, resamples: int, seed: int) -> list[str]:
    lines = ["## The ladder, one change per rung", ""]
    for dataset, entry in ablation["datasets"].items():
        for set_name in entry["runs"]:
            first = next(iter(entry["runs"][set_name].values()))
            lines += [
                f"### {dataset}, {set_name} ({first['questions']} questions)",
                "",
                f"_{_NOTE.get(dataset, dataset)}_",
                "",
            ]
            rows: list[list[str]] = []
            previous: Mapping[str, Any] | None = None
            for rung in (*_LADDER, *_LOO):
                run = _run(entry, set_name, rung)
                if run is None:
                    continue
                is_ladder = rung in _LADDER
                latency = run["latency_ms"]
                rows.append(
                    [
                        rung,
                        run["change"]
                        if rung in _LADDER
                        else f"{rung.replace('R7 - ', 'R7 without ')}",
                        *(_num(run["metrics"].get(m)) for m in _LADDER_METRICS),
                        _delta(previous, run, "ndcg@10", resamples=resamples, seed=seed)
                        if is_ladder and previous is not None
                        else "",
                        _num(run["docs_per_query"], 1),
                        f"{_num(latency.get('p50'), 0)} / {_num(latency.get('p95'), 0)}{_idle_mark(run)}",
                    ]
                )
                if is_ladder:
                    previous = run
            lines += _table(
                [
                    "rung",
                    "change",
                    "nDCG@10",
                    "Recall@5",
                    "Recall@10",
                    "MRR@10",
                    "Δ nDCG@10 vs previous rung",
                    "docs/query",
                    "ms p50 / p95",
                ],
                rows,
            )
    return lines


def _leave_one_out(ablation: Mapping[str, Any], *, resamples: int, seed: int) -> list[str]:
    lines = [
        "## Leave-one-out from R7",
        "",
        "Δ is R7 minus the system without that part: what the part adds given everything else.",
        "",
    ]
    rows: list[list[str]] = []
    for dataset, entry in ablation["datasets"].items():
        for set_name in entry["runs"]:
            full = _run(entry, set_name, "R7")
            if full is None:
                continue
            for name in _LOO:
                without = _run(entry, set_name, name)
                if without is None:
                    continue
                rows.append(
                    [
                        f"{dataset}:{set_name}",
                        name.replace("R7 - ", "without "),
                        _delta(without, full, "ndcg@10", resamples=resamples, seed=seed),
                        _delta(without, full, "recall@5", resamples=resamples, seed=seed),
                    ]
                )
    if not rows:
        return lines + ["Not run.", ""]
    return lines + _table(["questions", "removed", "Δ nDCG@10 (R7 − without)", "Δ Recall@5"], rows)


def _indexes(ablation: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            dataset,
            key,
            str(info["documents"]),
            str(info["chunks"]),
            _num(info["chunks_per_s"], 1) if info["chunks_per_s"] else "cached",
        ]
        for dataset, entry in ablation["datasets"].items()
        for key, info in entry["indexes"].items()
    ]
    return ["## Indexes built", ""] + _table(
        ["dataset", "variant", "documents", "chunks", "embedded chunks/s"], rows
    )


def _tuning(tuning: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Tuning sweeps (tuning questions only, D-065)",
        "",
        "Quality uses SciFact's train part, NFCorpus's dev part and golden dev; the headline numbers above use test.",
        "",
    ]
    for dataset, entry in tuning["datasets"].items():
        lines += [
            f"### {dataset}: {entry.get('set', '')} ({entry.get('questions', '?')} questions)",
            "",
        ]
        grid = [
            [
                str(r["depth"]),
                str(r["candidates"]),
                _num(r["metrics"].get("ndcg@10")),
                _num(r["metrics"].get("recall@5")),
                _num(r["docs_per_query"], 1),
            ]
            for r in entry["grid"]
        ]
        lines += _table(["depth", "candidates", "nDCG@10", "Recall@5", "docs/query"], grid)
        base = [
            [
                str(r["candidates"]),
                _num(r["metrics"].get("ndcg@10")),
                _num(r["metrics"].get("recall@5")),
            ]
            for r in entry["hybrid"]
        ]
        lines += ["Hybrid without rerank, for reference:", ""]
        lines += _table(["candidates", "nDCG@10", "Recall@5"], base)
        latency = [
            [
                str(r["depth"]),
                str(r["questions"]),
                _num(r["p50"], 0),
                _num(r["p95"], 0),
                "yes" if r["gate"]["idle"] else "NO",
            ]
            for r in entry["latency"]
        ]
        lines += ["Retrieval latency by depth (unmemoized, CPU gate):", ""]
        lines += _table(["depth", "questions", "ms p50", "ms p95", "CPU idle"], latency)
    return lines


def _bakeoff(bakeoff: Mapping[str, Any], *, resamples: int, seed: int) -> list[str]:
    environment = bakeoff["environment"]
    held_embedder = environment["embedder"]["label"]
    held_reranker = environment["reranker"]["label"]
    lines = [
        "## Bake-offs",
        "",
        (
            "Δ is the candidate minus the held choice (the embedder or reranker of the ladder), "
            "paired by question, with its 95% bootstrap interval. Descriptive: no p-value, no verdict."
        ),
        "",
    ]
    rows: list[list[str]] = []
    for key, row in bakeoff.get("embedders", {}).items():
        label, dataset = key.split("/", 1)
        for set_name, by_strategy in row["runs"].items():
            for strategy, run in by_strategy.items():
                stage = run["stage_ms"].get("embed_query")
                held_row = bakeoff["embedders"].get(f"{held_embedder}/{dataset}")
                held_run = held_row["runs"].get(set_name, {}).get(strategy) if held_row else None
                rows.append(
                    [
                        label + (" (held)" if label == held_embedder else ""),
                        f"{dataset}:{set_name}",
                        strategy,
                        _num(run["metrics"].get("ndcg@10")),
                        ""
                        if label == held_embedder
                        else _delta(held_run, run, "ndcg@10", resamples=resamples, seed=seed),
                        _num(run["metrics"].get("recall@5")),
                        _num(stage, 0),
                        _num(row["index"]["chunks_per_s"], 1)
                        if row["index"]["chunks_per_s"]
                        else "cached",
                    ]
                )
    if rows:
        lines += ["### Embedders (v2 chunks with header; each uses its own prefixes)", ""]
        lines += _table(
            [
                "embedder",
                "questions",
                "system",
                "nDCG@10",
                "Δ nDCG@10 vs held",
                "Recall@5",
                "query-embed ms",
                "ingest chunks/s",
            ],
            rows,
        )
    rows = []
    for key, row in bakeoff.get("rerankers", {}).items():
        label, dataset = key.split("/", 1)
        for set_name, run in row["runs"].items():
            latency = run["latency_ms"]
            held_row = bakeoff["rerankers"].get(f"{held_reranker}/{dataset}")
            held_run = held_row["runs"].get(set_name) if held_row else None
            rows.append(
                [
                    label + (" (held)" if label == held_reranker else ""),
                    f"{dataset}:{set_name}",
                    str(row["depth"]),
                    _num(run["metrics"].get("ndcg@10")),
                    ""
                    if label == held_reranker
                    else _delta(held_run, run, "ndcg@10", resamples=resamples, seed=seed),
                    _num(run["metrics"].get("recall@5")),
                    _num(run["stage_ms"].get("rerank"), 0),
                    f"{_num(latency.get('p50'), 0)} / {_num(latency.get('p95'), 0)}{_idle_mark(run)}",
                ]
            )
    if rows:
        lines += ["### Rerankers (held embedder, hybrid candidates)", ""]
        lines += _table(
            [
                "reranker",
                "questions",
                "depth",
                "nDCG@10",
                "Δ nDCG@10 vs held",
                "Recall@5",
                "rerank ms (mean)",
                "total ms p50 / p95",
            ],
            rows,
        )
    return lines if rows or len(lines) > 2 else lines + ["Not run.", ""]


def _abstain(abstain: Mapping[str, Any]) -> list[str]:
    low, high = abstain["threshold_interval_95"]
    reranker = abstain["reranker"]
    lines = [
        "## Abstain threshold (calibrated on golden dev, checked on golden test)",
        "",
        f"_{_NOTE['golden']}_",
        "",
        (
            f"- Signal: {abstain['signal']} (`{reranker['model']}`, depth {reranker['depth']}). "
            f"Rule: {abstain['rule']}."
        ),
        (
            f"- **Threshold {abstain['threshold']:.3f}**; 95% range of the best threshold over bootstrap "
            f"resamples of the development questions: {low:.3f} to {high:.3f}."
        ),
    ]
    if abstain["multi_turn_left_out"]:
        lines.append(
            f"- {abstain['multi_turn_left_out']} multi-turn questions were left out (no cached rewrite)."
        )
    rows = []
    for split in ("dev", "test"):
        d = abstain[split]
        rows.append(
            [
                split,
                str(d["questions"]),
                str(d["unanswerable"]),
                f"{d['tp']} / {d['fp']} / {d['fn']} / {d['tn']}",
                _num(d["precision"], 3),
                _num(d["recall"], 3),
                _num(d["f1"], 3),
            ]
        )
    lines += [""] + _table(
        ["split", "questions", "unanswerable", "tp / fp / fn / tn", "precision", "recall", "F1"],
        rows,
    )
    return lines + [
        (
            "Positive = should abstain (unanswerable). fp = an answerable question wrongly withheld; "
            "fn = an unanswerable question that would be answered from unrelated text."
        ),
        "",
    ]


def render_report(
    results: Mapping[str, Mapping[str, Any]], *, name: str, resamples: int = 10_000, seed: int = 0
) -> str:
    """The whole report as Markdown. ``ValueError`` if there are no results at all."""
    if not results:
        raise ValueError("no result files found; run `geekbot eval retrieval` first")
    lines = _header(results, name)
    if "ablation" in results:
        ablation = results["ablation"]
        ablation_view = {
            **ablation,
            "datasets": {
                dataset: {**entry, "aliases": ablation.get("aliases", {})}
                for dataset, entry in ablation["datasets"].items()
            },
        }
        lines += _headline(ablation_view, resamples=resamples, seed=seed)
        lines += _ladder(ablation_view, resamples=resamples, seed=seed)
        lines += _leave_one_out(ablation_view, resamples=resamples, seed=seed)
        lines += _indexes(ablation_view)
    if "tuning" in results:
        lines += _tuning(results["tuning"])
    if "bakeoff" in results:
        lines += _bakeoff(results["bakeoff"], resamples=resamples, seed=seed)
    if "abstain" in results:
        lines += _abstain(results["abstain"])
    return "\n".join(lines).rstrip() + "\n"
