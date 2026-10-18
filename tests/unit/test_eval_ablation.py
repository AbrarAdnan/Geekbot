"""Tests for the ablation ladder's definition and its configuration file (step S2.6)."""

from itertools import pairwise
from pathlib import Path

import pytest

from geekbot.evals.ablation import (
    LEAVE_ONE_OUT,
    RUNGS,
    AblationConfig,
    Rung,
    load_config,
    rung_by_name,
)
from geekbot.evals.phases import ladder_runs

ROOT = Path(__file__).resolve().parents[2]


def differences(a: Rung, b: Rung) -> list[str]:
    return [
        name
        for name, left, right in (
            ("chunking", a.variant.chunking, b.variant.chunking),
            ("prefixes", a.variant.prefixes, b.variant.prefixes),
            ("header", a.variant.header, b.variant.header),
            ("space", a.variant.space, b.variant.space),
            ("strategy", a.strategy, b.strategy),
        )
        if left != right
    ]


def test_the_ladder_has_eight_rungs_in_the_plans_order() -> None:
    assert [rung.name for rung in RUNGS] == [f"R{n}" for n in range(8)]
    assert [rung.change for rung in RUNGS] == [
        "legacy v1 replica",
        "L2 to cosine",
        "remove template multi-query",
        "+ prefixes",
        "new chunking",
        "+ contextual header",
        "+ BM25 via RRF",
        "+ rerank",
    ]


def test_each_rung_changes_exactly_one_thing() -> None:
    changes = [differences(a, b) for a, b in pairwise(RUNGS)]
    assert changes == [
        ["space"],
        ["strategy"],
        ["prefixes"],
        ["chunking"],
        ["header"],
        ["strategy"],
        ["strategy"],
    ]


def test_r0_is_defined_by_having_none_of_the_later_additions() -> None:
    r0 = rung_by_name("R0")
    assert (r0.variant.prefixes, r0.variant.header, r0.variant.chunking) == (False, False, "legacy")
    assert (r0.variant.space, r0.strategy) == ("l2", "legacy_multiquery")


def test_each_leave_one_out_removes_exactly_one_part_from_r7() -> None:
    r7 = rung_by_name("R7")
    removed = {}
    for loo in LEAVE_ONE_OUT:
        as_rung = Rung(loo.name, loo.removes, loo.variant, loo.strategy)
        removed[loo.removes] = differences(r7, as_rung)
    assert removed == {"BM25": ["strategy"], "rerank": ["strategy"], "header": ["header"]}


def test_removing_the_rerank_is_rung_r6_and_is_not_run_twice() -> None:
    no_rerank = next(loo for loo in LEAVE_ONE_OUT if loo.removes == "rerank")
    r6 = rung_by_name("R6")
    assert no_rerank.same_as == "R6"
    assert (no_rerank.variant, no_rerank.strategy) == (r6.variant, r6.strategy)
    assert [spec.name for spec in ladder_runs()] == [
        *(rung.name for rung in RUNGS),
        "R7 - BM25",
        "R7 - header",
    ]


def test_ladder_runs_can_be_narrowed_and_rejects_unknown_names() -> None:
    assert [spec.name for spec in ladder_runs(["R3", "R0"])] == ["R0", "R3"]
    with pytest.raises(ValueError, match="unknown rung"):
        ladder_runs(["R9"])
    with pytest.raises(KeyError, match="R0, R1"):
        rung_by_name("nope")


# --- the configuration -----------------------------------------------------------------------


def test_the_committed_configuration_is_valid_and_says_what_the_plan_says() -> None:
    config = load_config(ROOT / "evals" / "configs" / "ablation.toml")
    assert config.embedder.model == "nomic-embed-text"
    assert config.embedder.document_prefix == "search_document: "
    assert (config.reranker.depth, config.fusion.candidates, config.resamples) == (10, 50, 10_000)
    assert config.beir.datasets == ("scifact", "nfcorpus")
    assert [e.label for e in config.bakeoff.embedders] == ["qwen3-embedding-0.6b", "embeddinggemma"]
    assert [r.model for r in config.bakeoff.rerankers] == ["BAAI/bge-reranker-base"]
    assert config.bakeoff.rerankers[0].cache_dir == Path("scratch/models")  # never into data/
    assert config.results_path == Path("evals/results") / config.name
    assert config.rewrite_model == "gemma4:12b"


def test_the_default_reranker_depth_in_the_configuration_matches_the_code() -> None:
    from geekbot.retrieval.rerank import DEFAULT_RERANK_DEPTH

    config = load_config(ROOT / "evals" / "configs" / "ablation.toml")
    assert config.reranker.depth == DEFAULT_RERANK_DEPTH


def test_a_misspelled_option_is_an_error_not_an_ignored_line(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[embedder]\nlabel = "a"\nmodel = "m"\ntop_kk = 3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="top_kk"):
        load_config(path)


def test_missing_broken_and_incomplete_files_are_refused_with_a_reason(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_config(tmp_path / "absent.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("name = ", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_config(broken)
    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text('name = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="embedder"):
        load_config(incomplete)


def test_limits_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('top_k = 0\n[embedder]\nlabel = "a"\nmodel = "m"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="top_k"):
        load_config(path)
    with pytest.raises(ValueError):
        AblationConfig.model_validate({"embedder": {"label": "a", "model": "m"}, "resamples": 5})
