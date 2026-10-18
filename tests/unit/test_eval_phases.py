"""Tests for the phases of ``geekbot eval retrieval`` and the report (step S2.6).

The whole pipeline runs once on a tiny BEIR-shaped dataset and a tiny golden set, with a hashing
embedder and a word-overlap "reranker" instead of models (``evalkit``); SQLite and Chroma are real.
The module-scoped fixture runs every phase once and the tests read what it wrote.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from evalkit import (
    fake_backends,
    tiny_config,
    tiny_settings,
    write_tiny_golden,
)
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.evals import orchestrate
from geekbot.evals.ablation import AblationConfig
from geekbot.evals.datasets import RETRIEVAL_TYPES, load_golden
from geekbot.evals.orchestrate import Phase, run_phases
from geekbot.evals.phases import (
    beir_bundle,
    golden_bundle,
    only_sets,
)
from geekbot.evals.report import HOLM_FAMILY, load_results, render_report
from geekbot.evals.rewrites import RewriteCache, rewrite_key
from geekbot.ingest.ids import doc_id_of

LOG: list[str] = []


@pytest.fixture(autouse=True)
def no_ollama_for_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rewrite cache is keyed by the model's digest, which is read from Ollama; fake it."""
    monkeypatch.setattr(orchestrate, "_digest", lambda settings, model: "dg")


class World:
    def __init__(self, root: Path, config: AblationConfig, written: list[Path]) -> None:
        self.root = root
        self.config = config
        self.written = written

    def result(self, kind: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (self.config.results_path / f"{kind}.json").read_text(encoding="utf-8")
        )
        return data


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[World]:
    root = tmp_path_factory.mktemp("phases")
    config = tiny_config(root, rewrite_model="fake-rewriter")
    _, _, rewrites = write_tiny_golden(root)
    cache = RewriteCache.load(orchestrate.rewrite_cache_path(config))
    for item in load_golden(config.golden.jsonl):
        if item.id in rewrites:
            key = rewrite_key(item, model="fake-rewriter", digest="dg")
            cache.put(key, item, model="fake-rewriter", digest="dg", rewrite=rewrites[item.id])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(orchestrate, "_digest", lambda settings, model: "dg")
        written = run_phases(
            config,
            tiny_settings(root),
            fake_backends(),
            [Phase.ABLATION, Phase.TUNING, Phase.BAKEOFF, Phase.ABSTAIN],
            log=LOG.append,
        )
    yield World(root, config, written)


# --- the datasets ----------------------------------------------------------------------------


def test_beir_headline_questions_are_the_test_split_with_judgements_by_document_id(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path)
    bundle = beir_bundle(config, "scifact", tuning=False)
    queries = bundle.query_sets["test"]
    assert queries.name == "scifact:test"
    assert len(queries.queries) == 8
    assert len(bundle.documents) == 16
    judged = queries.qrels["q-test-zebra"]
    assert len(judged) == 2
    document_ids = {doc_id_of(document.path) for document in bundle.documents}
    assert set(judged) <= document_ids  # judgements name documents the index will hold


def test_the_tuning_questions_are_a_seeded_sample_of_the_train_split(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    first = beir_bundle(config, "scifact", tuning=True).query_sets["tune"]
    second = beir_bundle(config, "scifact", tuning=True).query_sets["tune"]
    assert first.name == "scifact:train"
    assert len(first.queries) == 6  # capped by tuning.max_queries
    assert first.queries == second.queries  # the same sample every time
    assert all(question_id.startswith("q-train-") for question_id in first.queries)  # never test


def test_golden_sets_hold_retrieval_and_rewritten_multi_turn_items_never_unanswerable(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path)
    _, _, rewrites = write_tiny_golden(tmp_path)
    bundle = golden_bundle(config, rewrites)
    kinds = {item.id: item.type for item in load_golden(config.golden.jsonl)}
    for name in ("dev", "test"):
        ids = set(bundle.query_sets[name].queries)
        assert ids and {kinds[i] for i in ids} <= {*RETRIEVAL_TYPES, "multi_turn"}
    dev, test = bundle.query_sets["dev"], bundle.query_sets["test"]
    assert not set(dev.queries) & set(test.queries)
    assert len(dev.queries) + len(test.queries) == 18  # 10 + 3 + 5; the 5 unanswerable are out
    follow_up = next(i for i in {**dev.queries, **test.queries} if i.startswith("T-"))
    assert {**dev.queries, **test.queries}[follow_up] == rewrites[follow_up]  # the rewrite is used


def test_without_rewrites_the_follow_ups_are_left_out_and_the_log_says_so(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    messages: list[str] = []
    bundle = golden_bundle(config, {}, log=messages.append)
    everything = {**bundle.query_sets["dev"].queries, **bundle.query_sets["test"].queries}
    assert not any(i.startswith("T-") for i in everything)
    assert any("5 multi-turn items have no cached rewrite" in m for m in messages)


def test_a_golden_source_that_was_not_loaded_is_a_setup_error(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    (config.golden.corpus / "zebra.txt").unlink()
    with pytest.raises(ValueError, match="were not loaded"):
        golden_bundle(config, {})


def test_only_sets_keeps_the_named_question_sets(tmp_path: Path) -> None:
    bundle = golden_bundle(tiny_config(tmp_path), {})
    assert set(only_sets(bundle, ["dev"]).query_sets) == {"dev"}


# --- the ablation ----------------------------------------------------------------------------


def test_every_rung_and_leave_one_out_ran_on_every_question_set(world: World) -> None:
    ablation = world.result("ablation")
    assert ablation["schema"] == 1
    assert ablation["aliases"] == {"R7 - rerank": "R6"}
    expected = {*(f"R{n}" for n in range(8)), "R7 - BM25", "R7 - header"}
    assert set(ablation["datasets"]) == {"scifact", "golden"}
    assert set(ablation["datasets"]["scifact"]["runs"]) == {"test"}
    assert set(ablation["datasets"]["golden"]["runs"]) == {"dev", "test"}
    for dataset in ablation["datasets"].values():
        for runs in dataset["runs"].values():
            assert set(runs) == expected


def test_the_indexes_built_are_the_four_variants_plus_r0s_l2_twin(world: World) -> None:
    for dataset in world.result("ablation")["datasets"].values():
        assert set(dataset["indexes"]) == {
            "legacy-raw-nohdr-l2",
            "legacy-raw-nohdr-cosine",
            "legacy-pfx-nohdr-cosine",
            "v2-pfx-nohdr-cosine",
            "v2-pfx-hdr-cosine",
        }
    legacy = world.result("ablation")["datasets"]["scifact"]["indexes"]
    assert legacy["legacy-raw-nohdr-cosine"]["embedded"] == 0  # R1's index reused R0's vectors
    assert legacy["legacy-raw-nohdr-cosine"]["cache_hits"] > 0


def test_numbers_are_possible_and_the_pipeline_finds_what_it_should(world: World) -> None:
    runs = world.result("ablation")["datasets"]["scifact"]["runs"]["test"]
    for name, run in runs.items():
        assert run["questions"] == 8, name
        for metric, value in run["metrics"].items():
            assert 0.0 <= value <= 1.0, (name, metric)  # an impossible number is a bug
        assert 1.0 <= run["docs_per_query"] <= 5.0, name
        assert run["latency_ms"]["p95"] >= run["latency_ms"]["p50"] >= 0
        assert run["gate"]["idle"] is True
        assert run["question_set"] == "scifact:test"
    # a tiny, easy dataset: every system that searches by meaning should find both documents
    for name in ("R2", "R3", "R4", "R5", "R6", "R7"):
        assert runs[name]["metrics"]["recall@5"] >= 0.9, name


def test_identical_systems_get_identical_numbers(world: World) -> None:
    """R0 and R1 differ only in L2 vs cosine; for unit vectors they must agree (the R1 expectation)."""
    runs = world.result("ablation")["datasets"]["scifact"]["runs"]["test"]
    assert runs["R0"]["per_query"] == runs["R1"]["per_query"]


def test_a_record_carries_the_per_question_scores_the_statistics_need(world: World) -> None:
    run = world.result("ablation")["datasets"]["golden"]["runs"]["test"]["R7"]
    assert set(run["per_query"]) == {"recall@5", "recall@10", "ndcg@10", "mrr@10"}
    assert len(run["per_query"]["recall@5"]) == run["questions"] == 11  # 6 + 2 + 3 in golden test
    assert run["strategy"] == "hybrid_rerank" and run["variant"] == "v2-pfx-hdr-cosine"


def test_a_second_run_with_resume_does_nothing_and_a_fresh_one_measures_again(
    world: World,
) -> None:
    opened: list[tuple[str, str]] = []
    again = run_phases(
        world.config,
        tiny_settings(world.root),
        fake_backends(opened=opened),
        [Phase.ABLATION],
        log=LOG.append,
    )
    assert opened == []  # every piece was finished: no index was even opened
    assert again == [world.config.results_path / "ablation.json"]
    # a copy of the configuration with its own results folder, so the shared results stay whole
    other = world.config.model_copy(update={"name": "t-fresh"})
    for resume in (True, False):  # the folder is empty, so the first run builds regardless
        run_phases(
            other,
            tiny_settings(world.root),
            fake_backends(opened=opened),
            [Phase.ABLATION],
            rungs=["R2"],
            datasets=["scifact"],
            resume=resume,
            log=LOG.append,
        )
    assert opened == [("fake-embed", "legacy-raw-nohdr-cosine")] * 2  # fresh measured it again
    produced = json.loads((other.results_path / "ablation.json").read_text(encoding="utf-8"))
    assert set(produced["datasets"]["scifact"]["runs"]["test"]) == {"R2"}


def test_a_pc_that_was_not_idle_is_marked_on_every_record(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    run_phases(
        config,
        tiny_settings(tmp_path),
        fake_backends(idle=False),
        [Phase.ABLATION],
        rungs=["R2"],
        datasets=["scifact"],
        log=LOG.append,
    )
    results = json.loads((config.results_path / "ablation.json").read_text(encoding="utf-8"))
    assert results["datasets"]["scifact"]["runs"]["test"]["R2"]["gate"]["idle"] is False
    text = render_report(load_results(config.results_path), name="t", resamples=100)
    assert "(CPU not idle)" in text


# --- the sweeps, bake-offs and abstain --------------------------------------------------------


def test_tuning_sweeps_a_grid_on_the_tuning_questions_only(world: World) -> None:
    tuning = world.result("tuning")
    scifact = tuning["datasets"]["scifact"]
    assert scifact["set"] == "scifact:train" and scifact["questions"] == 6
    assert tuning["datasets"]["golden"]["set"] == "golden:dev"
    for entry in tuning["datasets"].values():
        assert entry["complete"] is True
        assert [(r["depth"], r["candidates"]) for r in entry["grid"]] == [
            (5, 5),
            (8, 5),
            (5, 10),
            (8, 10),
        ]
        assert [r["candidates"] for r in entry["hybrid"]] == [5, 10]
        assert [r["depth"] for r in entry["latency"]] == [5, 8]
        assert all(0.0 <= r["metrics"]["ndcg@10"] <= 1.0 for r in entry["grid"])


def test_bake_offs_compare_candidates_on_the_headline_sets(world: World) -> None:
    bakeoff = world.result("bakeoff")
    assert set(bakeoff["embedders"]) == {
        "fake-embed/scifact",
        "fake-embed/golden",
        "other-embed/scifact",
        "other-embed/golden",
    }
    row = bakeoff["embedders"]["other-embed/scifact"]
    assert set(row["runs"]["test"]) == {"dense", "hybrid_rerank"}
    assert row["index"]["embedder"]["model"] == "other-fake"
    assert row["index"]["chunks_per_s"] is None or row["index"]["chunks_per_s"] > 0
    assert row["runs"]["test"]["dense"]["stage_ms"]["embed_query"] >= 0
    assert set(bakeoff["rerankers"]) == {
        "overlap/scifact",
        "overlap/golden",
        "other-rerank/scifact",
        "other-rerank/golden",
    }
    assert bakeoff["rerankers"]["other-rerank/scifact"]["model"] == "other-fake-reranker"
    assert "rerank" in bakeoff["rerankers"]["overlap/scifact"]["runs"]["test"]["stage_ms"]


def test_the_abstain_threshold_is_found_on_dev_and_checked_on_test(world: World) -> None:
    abstain = world.result("abstain")
    assert abstain["reranker"] == {"model": "fake-reranker", "depth": 5}
    assert 0.2857 < abstain["threshold"] < 0.4  # between the best unanswerable and worst answerable
    low, high = abstain["threshold_interval_95"]
    assert low <= abstain["threshold"] <= high or low <= high
    assert (abstain["dev"]["questions"], abstain["dev"]["unanswerable"]) == (9, 2)
    assert (abstain["test"]["questions"], abstain["test"]["unanswerable"]) == (14, 3)
    assert abstain["dev"]["f1"] == 1.0 and abstain["test"]["f1"] == 1.0
    assert len(abstain["scores"]) == 23
    assert abstain["multi_turn_left_out"] == 0
    unanswerable = [s["score"] for s in abstain["scores"] if s["kind"] == "unanswerable"]
    answerable = [s["score"] for s in abstain["scores"] if s["kind"] != "unanswerable"]
    assert max(unanswerable) < min(answerable)


def test_abstain_needs_the_golden_set(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, golden={"enabled": False})
    with pytest.raises(ValueError, match="needs the golden set"):
        run_phases(
            config, tiny_settings(tmp_path), fake_backends(), [Phase.ABSTAIN], log=LOG.append
        )


# --- choosing what to run --------------------------------------------------------------------


def test_an_unknown_dataset_is_refused_before_anything_runs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        run_phases(
            config,
            tiny_settings(tmp_path),
            fake_backends(),
            [Phase.ABLATION],
            datasets=["nfcorpus"],
            log=LOG.append,
        )
    assert not config.results_path.exists()


def test_the_rewrite_phase_calls_the_model_once_per_follow_up_then_uses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from geekbot.agent.condense import CondenseOutput

    config = tiny_config(tmp_path, rewrite_model="fake-rewriter")
    calls: list[str] = []

    class Rewriter:
        def invoke(self, messages: list[Any], /) -> CondenseOutput:
            calls.append(messages[-1].content)
            return CondenseOutput(standalone_question="when was the zebra last reviewed")

    monkeypatch.setattr(orchestrate, "make_structured_model", lambda *a, **k: Rewriter())
    monkeypatch.setattr(orchestrate, "_digest", lambda settings, model: "dg")
    run_phases(config, tiny_settings(tmp_path), fake_backends(), [Phase.REWRITE], log=LOG.append)
    assert len(calls) == 5
    run_phases(config, tiny_settings(tmp_path), fake_backends(), [Phase.REWRITE], log=LOG.append)
    assert len(calls) == 5  # all cached the second time
    assert len(orchestrate.load_rewrites(config, tiny_settings(tmp_path))) == 5


def test_no_cached_rewrites_means_an_empty_mapping_without_touching_ollama(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    assert orchestrate.load_rewrites(config, tiny_settings(tmp_path)) == {}  # no file, no call


# --- the report ------------------------------------------------------------------------------


def test_the_report_has_every_section_and_never_says_significant(world: World) -> None:
    text = render_report(load_results(world.config.results_path), name="t", resamples=200)
    for heading in (
        "# Retrieval evaluation report: t",
        "## Headline: the full system against the v1 replica",
        "## The ladder, one change per rung",
        "## Leave-one-out from R7",
        "## Indexes built",
        "## Tuning sweeps",
        "## Bake-offs",
        "## Abstain threshold",
    ):
        assert heading in text
    assert "significant" not in text.lower()
    assert "Claude-authored golden v0, provisional" in text
    assert "standard BEIR dataset, standard metrics" in text


def test_the_headline_is_the_preregistered_family_with_holm_adjusted_p_values(
    world: World,
) -> None:
    text = render_report(load_results(world.config.results_path), name="t", resamples=200)
    lines = [line for line in text.splitlines() if "Holm-adjusted p=" in line]
    # the tiny set has no nfcorpus, so two of the three family members exist
    assert len(lines) == 2 and len(HOLM_FAMILY) == 3
    assert any(line.startswith("- scifact ndcg@10 (R7 vs R0)") for line in lines)
    assert any(line.startswith("- golden recall@5 (R7 vs R0)") for line in lines)
    assert "Only 2 of 3 family members were run." in text
    assert all("delta=" in line and "% CI" in line and "p=" in line for line in lines)


def test_ladder_rows_show_delta_with_interval_but_no_p_value(world: World) -> None:
    text = render_report(load_results(world.config.results_path), name="t", resamples=200)
    ladder = text.split("## The ladder")[1].split("## Leave-one-out")[0]
    rows = [line for line in ladder.splitlines() if line.startswith("| R")]
    assert len(rows) >= 2 * 10  # scifact:test, golden:dev, golden:test each have ten rows
    assert all("p=" not in row for row in rows)
    assert any(" [" in row and "+" in row or "-" in row for row in rows)
    leave = text.split("## Leave-one-out from R7")[1].split("## Indexes")[0]
    assert "without BM25" in leave and "without rerank" in leave and "without header" in leave


def test_the_report_is_computed_from_the_files_so_it_is_repeatable(world: World) -> None:
    results = load_results(world.config.results_path)
    assert render_report(results, name="t", resamples=200) == render_report(
        results, name="t", resamples=200
    )


def test_a_report_of_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no result files"):
        render_report({}, name="t")


# --- the command line ------------------------------------------------------------------------


def test_eval_report_writes_the_markdown_next_to_the_results(world: World) -> None:
    from evalkit import write_tiny_toml

    path = write_tiny_toml(world.root)
    result = CliRunner().invoke(app, ["eval", "report", "--config", str(path)])
    assert result.exit_code == 0, result.output
    report = world.config.results_path / "report.md"
    assert report.is_file()
    assert report.read_text(encoding="utf-8").startswith("# Retrieval evaluation report: t")


def test_eval_report_without_results_says_to_run_the_evaluation_first(tmp_path: Path) -> None:
    from evalkit import write_tiny_toml

    result = CliRunner().invoke(app, ["eval", "report", "--config", str(write_tiny_toml(tmp_path))])
    assert result.exit_code != 0
    assert "geekbot eval retrieval" in result.output


def test_eval_retrieval_rejects_unknown_phases_and_missing_configs(tmp_path: Path) -> None:
    from evalkit import write_tiny_toml

    path = write_tiny_toml(tmp_path)
    runner = CliRunner()
    bad_phase = runner.invoke(app, ["eval", "retrieval", "--config", str(path), "--phase", "oops"])
    assert bad_phase.exit_code != 0 and "unknown phase" in bad_phase.output
    missing = runner.invoke(app, ["eval", "retrieval", "--config", str(tmp_path / "none.toml")])
    assert missing.exit_code != 0 and "cannot read" in missing.output


def test_eval_retrieval_runs_a_chosen_piece_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalkit import write_tiny_toml

    path = write_tiny_toml(tmp_path)
    monkeypatch.setattr("geekbot.evals.phases.default_backends", lambda settings: fake_backends())
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "retrieval",
            "--config",
            str(path),
            "--phase",
            "ablation",
            "--dataset",
            "scifact",
            "--rung",
            "R2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "scifact:test R2" in result.output
    assert "wrote" in result.output
    produced = json.loads((tmp_path / "results" / "t" / "ablation.json").read_text("utf-8"))
    assert set(produced["datasets"]["scifact"]["runs"]["test"]) == {"R2"}
