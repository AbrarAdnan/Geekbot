"""Tests for step S5.1: the bench's rules (``geekbot.evals.bench``), with made-up numbers.

Nothing here starts a server, loads a model or reads the GPU: the guard, the choice of default and the
report are plain functions of data, which is why they can be tested to the last case.
"""

import itertools
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.latency import Spread
from geekbot.evals.bench import (
    BenchConfig,
    ChatRun,
    ConfigResult,
    GpuReading,
    GuardVerdict,
    RunSummary,
    SweepAxes,
    SweepPoint,
    check_gpu_idle,
    choose_default,
    config_result_from_json,
    expand_sweep,
    format_bench_report,
    load_bench_config,
    must_include_recall,
    parse_gpu_reading,
    pick_questions,
    rechoose,
    slo_misses,
    summarize_runs,
    wait_until_gpu_idle,
)

# --- configuration ---------------------------------------------------------------------------


def test_the_defaults_are_the_sweep_of_the_plan() -> None:
    config = BenchConfig()
    axes = config.sweep
    assert axes.num_ctx == (4096, 8192)
    assert axes.kv_cache_type == ("f16", "q8_0")
    assert axes.num_batch == (256, 512)
    assert axes.top_k == (3, 5)
    assert axes.port != 11434  # never the user's own Ollama
    assert config.results_path == Path("evals/results/bench")


def test_a_config_file_is_read_and_a_misspelled_option_fails_before_the_run(tmp_path: Path) -> None:
    good = tmp_path / "bench.toml"
    good.write_text('name = "x"\n[sweep]\nnum_ctx = [2048]\n', encoding="utf-8")
    assert load_bench_config(good).sweep.num_ctx == (2048,)
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n[sweep]\nnum_cxt = [2048]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.toml"):
        load_bench_config(bad)
    with pytest.raises(ValueError, match="nope.toml"):
        load_bench_config(tmp_path / "nope.toml")


def test_nonsense_numbers_are_refused() -> None:
    with pytest.raises(ValueError):
        BenchConfig.model_validate({"chat": {"questions": 0}})
    with pytest.raises(ValueError):
        BenchConfig.model_validate({"sweep": {"port": 0}})


def test_the_sweep_has_every_combination_once() -> None:
    points = expand_sweep(SweepAxes())
    assert len(points) == len(set(points)) == 2 * 2 * 2 * 2
    assert {p.kv_cache_type for p in points} == {"f16", "q8_0"}


def test_work_that_shares_a_server_or_a_model_load_is_adjacent() -> None:
    points = expand_sweep(SweepAxes())
    kv_runs = [p.kv_cache_type for p in points]
    assert kv_runs == ["f16"] * 8 + ["q8_0"] * 8  # one server start per KV type, f16 first
    loads = [p.load_key for p in points]
    changes = sum(1 for before, after in itertools.pairwise(loads) if before != after)
    assert changes == len(set(loads)) - 1  # a load key never comes back after it has changed
    first_load = [p.top_k for p in points if p.load_key == loads[0]]
    assert first_load == [3, 5]  # top-k changes only the prompt, not the load


def test_the_order_of_the_kv_types_follows_the_config() -> None:
    axes = SweepAxes(kv_cache_type=("q8_0", "f16"), num_ctx=(8192,), num_batch=(512,), top_k=(5,))
    assert [p.kv_cache_type for p in expand_sweep(axes)] == ["q8_0", "f16"]


def test_a_point_names_itself() -> None:
    assert SweepPoint("q8_0", 4096, 256, 3).key == "ctx4096-kvq8_0-batch256-k3"


# --- choosing the questions ------------------------------------------------------------------


def test_questions_are_spread_evenly_and_the_ends_are_in() -> None:
    items = list(range(10))
    assert pick_questions(items, 4) == [0, 3, 6, 9]
    assert pick_questions(items, 2) == [0, 9]
    assert pick_questions(items, 1) == [0]


def test_asking_for_more_than_there_are_gives_all_and_for_none_gives_none() -> None:
    assert pick_questions([1, 2, 3], 10) == [1, 2, 3]
    assert pick_questions([1, 2, 3], 0) == []
    assert pick_questions([], 5) == []


@given(st.lists(st.integers(), max_size=60), st.integers(min_value=0, max_value=70))
def test_a_pick_is_a_deterministic_ordered_selection(items: list[int], count: int) -> None:
    picked = pick_questions(items, count)
    assert picked == pick_questions(items, count)
    assert len(picked) == min(count, len(items))
    positions = [items.index(x) for x in picked] if len(set(items)) == len(items) else None
    if positions is not None:
        assert positions == sorted(positions)


# --- answer quality --------------------------------------------------------------------------


def test_recall_is_the_share_of_phrases_found() -> None:
    assert must_include_recall("Chat history is kept for 90 days [1].", ["90 days"]) == 1.0
    assert must_include_recall("It is kept for 90 days.", ["90 days", "audit logs"]) == 0.5
    assert must_include_recall("I could not find it in the documents.", ["90 days"]) == 0.0


def test_an_item_with_no_phrases_has_no_recall_not_zero() -> None:
    assert must_include_recall("anything", []) is None
    assert must_include_recall("anything", ["", "  "]) is None


def test_case_dashes_quotes_emphasis_and_spacing_do_not_matter() -> None:
    answer = "The **Lantern‑Guest**  network\nis called “Lantern’s”."
    assert must_include_recall(answer, ["lantern-guest network"]) == 1.0
    assert must_include_recall(answer, ['"lantern\'s"']) == 1.0
    assert must_include_recall("Priya  Nair", ["priya nair"]) == 1.0


@given(st.lists(st.text(min_size=1, max_size=12), max_size=6), st.text(max_size=80))
def test_recall_is_always_a_share_or_nothing(phrases: list[str], answer: str) -> None:
    value = must_include_recall(answer, phrases)
    assert value is None or 0.0 <= value <= 1.0


@given(st.lists(st.from_regex(r"[a-z0-9]{2,8}", fullmatch=True), min_size=1, max_size=5))
def test_an_answer_that_contains_every_phrase_has_full_recall(phrases: list[str]) -> None:
    assert must_include_recall("see " + " and ".join(phrases), phrases) == 1.0


# --- reading the GPU -------------------------------------------------------------------------


def test_the_gpu_line_is_memory_total_and_utilisation() -> None:
    reading = parse_gpu_reading("976, 10240, 15\n")
    assert (reading.used_mib, reading.total_mib, reading.utilization_percent) == (976, 10240, 15.0)
    assert reading.compute_apps == ()


def test_an_unreported_utilisation_is_none_and_the_first_gpu_wins() -> None:
    reading = parse_gpu_reading("976, 10240, [N/A]\n1, 2, 3\n")
    assert reading.utilization_percent is None and reading.used_mib == 976
    assert parse_gpu_reading("976, 10240").utilization_percent is None


def test_compute_processes_are_listed_by_name_and_pid() -> None:
    apps = "python.exe, 1234\nollama.exe, 99\n[N/A], 5\n\n"
    assert parse_gpu_reading("1, 2, 3", apps).compute_apps == (
        "python.exe (1234)",
        "ollama.exe (99)",
    )


@pytest.mark.parametrize("text", ["", "\n\n", "abc", "1"])
def test_text_with_no_gpu_line_is_an_error(text: str) -> None:
    with pytest.raises(ValueError):
        parse_gpu_reading(text)


# --- the idle-GPU guard ----------------------------------------------------------------------

QUIET = GpuReading(used_mib=1000, total_mib=10240, utilization_percent=2.0)


def verdict(reading: GpuReading | None = QUIET, **overrides: object) -> GuardVerdict:
    arguments: dict[str, object] = {
        "other_gpu_models": (),
        "max_used_mib": 2048,
        "max_util_percent": 15.0,
    }
    arguments.update(overrides)
    return check_gpu_idle(reading, **arguments)  # type: ignore[arg-type]


def test_a_quiet_card_is_idle() -> None:
    result = verdict()
    assert result.idle and result.reasons == ()


def test_no_reading_is_never_idle() -> None:
    result = verdict(None)
    assert not result.idle
    assert "could not be read" in result.reasons[0]


def test_memory_over_the_limit_means_another_program_holds_it() -> None:
    result = verdict(GpuReading(5000, 10240, 1.0))
    assert not result.idle and "5000 MiB" in result.reasons[0]
    assert verdict(GpuReading(2048, 10240, 1.0)).idle  # the limit itself is still idle


def test_a_busy_card_is_not_idle_even_with_little_memory_in_use() -> None:
    result = verdict(GpuReading(1000, 10240, 80.0))
    assert not result.idle and "80% busy" in result.reasons[0]
    assert verdict(GpuReading(1000, 10240, 15.0)).idle  # the limit itself is still idle


def test_an_unreported_utilisation_is_not_held_against_the_card() -> None:
    assert verdict(GpuReading(1000, 10240, None)).idle


def test_a_foreign_model_on_the_gpu_is_named() -> None:
    result = verdict(QUIET, other_gpu_models=("qwen3:8b",))
    assert not result.idle and "qwen3:8b" in result.reasons[0]


def test_listed_processes_alone_never_make_a_quiet_card_busy() -> None:
    """On Windows the driver lists every program with a window (found in the first live run: 21 on
    a PC whose GPU was doing nothing), so a process list is evidence, not a rule."""
    desktop = tuple(f"C:\\Windows\\app{n}.exe ({n})" for n in range(21))
    assert verdict(GpuReading(970, 10240, 0.0, compute_apps=desktop)).idle


def test_the_listed_processes_are_named_when_memory_or_utilisation_already_says_busy() -> None:
    apps = ("C:\\Program Files\\Python\\python.exe (7)", "explorer.exe (9)")
    result = verdict(GpuReading(6000, 10240, 1.0, compute_apps=apps))
    assert not result.idle
    assert len(result.reasons) == 2
    assert "python.exe (7), explorer.exe (9)" in result.reasons[1]
    assert "Program Files" not in result.reasons[1]  # names, not folders


def test_a_long_process_list_is_cut_and_counted() -> None:
    apps = tuple(f"app{n}.exe ({n})" for n in range(10))
    result = verdict(GpuReading(6000, 10240, 1.0, compute_apps=apps))
    assert "app0.exe (0)" in result.reasons[1] and "and 4 more" in result.reasons[1]


def test_a_saved_verdict_keeps_the_count_of_listed_processes_not_their_names() -> None:
    """Result files are committed to a public repository: a list of the programs running on someone's
    PC (found in the first live run: a chat client, a browser, a game launcher, a user folder)
    must not be in them, while the terminal still names them."""
    apps = tuple(f"C:\\Users\\someone\\AppData\\Local\\Secret\\app{n}.exe ({n})" for n in range(7))
    result = verdict(GpuReading(6000, 10240, 80.0, compute_apps=apps))
    assert "app0.exe" in " ".join(result.reasons)  # shown on the screen while the run goes on
    saved = json.dumps(result.to_json())
    assert "app0" not in saved and "Secret" not in saved and "someone" not in saved
    assert "7 of them (the names are not saved)" in saved
    assert result.to_json()["reading"]["compute_apps_count"] == 7


def test_every_reason_that_applies_is_reported_not_only_the_first() -> None:
    reading = GpuReading(6000, 10240, 90.0, compute_apps=("a.exe (1)",))
    result = verdict(reading, other_gpu_models=("m:1b",))
    assert len(result.reasons) == 4  # memory, utilisation, who might be using it, the other model


@given(
    st.integers(min_value=0, max_value=10240),
    st.one_of(st.none(), st.floats(min_value=0, max_value=100)),
    st.lists(st.text(min_size=1, max_size=6), max_size=3),
)
def test_idle_means_no_reasons_and_no_reasons_means_idle(
    used: int, util: float | None, others: list[str]
) -> None:
    result = verdict(GpuReading(used, 10240, util), other_gpu_models=tuple(others))
    assert result.idle == (result.reasons == ())


def test_waiting_stops_as_soon_as_the_card_is_idle_and_counts_the_wait() -> None:
    answers = iter([False, False, True])
    slept: list[float] = []

    def check() -> GuardVerdict:
        return GuardVerdict(next(answers), ("busy",), None)

    result = wait_until_gpu_idle(check, attempts=5, pause_s=20.0, sleep=slept.append)
    assert result.idle
    assert slept == [20.0, 20.0] and result.waited_s == 40.0


def test_a_card_that_never_goes_quiet_ends_as_not_idle_with_its_reasons() -> None:
    slept: list[float] = []
    result = wait_until_gpu_idle(
        lambda: GuardVerdict(False, ("busy",), None), attempts=3, pause_s=5.0, sleep=slept.append
    )
    assert not result.idle and result.reasons == ("busy",)
    assert slept == [5.0, 5.0]


def test_one_attempt_means_no_waiting() -> None:
    slept: list[float] = []
    wait_until_gpu_idle(
        lambda: GuardVerdict(False, ("busy",), None), attempts=1, pause_s=5.0, sleep=slept.append
    )
    assert slept == []


# --- summarising runs ------------------------------------------------------------------------


def run(
    first: float | None = 0.4,
    whole: float = 1.5,
    retrieval_ms: float = 300.0,
    load: float = 0.01,
    recall: float | None = 1.0,
    tps: float | None = 60.0,
) -> ChatRun:
    return ChatRun("S-001", first, whole, tps, load, retrieval_ms, recall)


def test_a_run_reports_what_the_user_waits_for_before_the_first_word() -> None:
    assert run(first=0.4, retrieval_ms=300.0).question_to_first_token_s == pytest.approx(0.7)
    assert run(first=None).question_to_first_token_s is None


def test_the_summary_has_spreads_loads_and_quality() -> None:
    runs = [run(recall=1.0), run(recall=0.5, load=2.0), run(recall=None, first=None)]
    summary = summarize_runs(runs)
    assert summary.n == 3
    assert summary.loads == 1  # only the run that spent 2 s loading
    assert summary.quality == pytest.approx(0.75)
    assert summary.quality_n == 2  # the run with no phrases is not a zero
    assert summary.first_token is not None and summary.first_token.n == 2  # one run had none
    assert summary.whole is not None and summary.whole.n == 3


def test_a_summary_of_nothing_is_empty_not_an_error() -> None:
    summary = summarize_runs([])
    assert summary.n == 0 and summary.quality is None and summary.whole is None


# --- the SLO check ---------------------------------------------------------------------------


def summary_with(
    q2f_p50: float = 1.0, q2f_p95: float = 1.5, whole_p95: float = 3.0, retrieval_p95: float = 300.0
) -> RunSummary:
    return RunSummary(
        n=10,
        first_token=Spread(10, 0.5, 0.8, 0.9),
        to_first_token=Spread(10, q2f_p50, q2f_p95, q2f_p95),
        whole=Spread(10, 2.0, whole_p95, whole_p95),
        tokens_per_s=Spread(10, 60.0, 50.0, 40.0),
        retrieval_ms=Spread(10, 250.0, retrieval_p95, retrieval_p95),
        loads=0,
        quality=1.0,
        quality_n=10,
    )


def test_a_summary_inside_every_target_has_no_misses() -> None:
    assert slo_misses(summary_with()) == []


def test_each_target_is_checked_and_named() -> None:
    names = {
        m.name
        for m in slo_misses(
            summary_with(q2f_p50=3.0, q2f_p95=5.0, whole_p95=20.0, retrieval_p95=500.0)
        )
    }
    assert names == {
        "retrieval p95",
        "question to first token p50",
        "question to first token p95",
        "whole turn p95",
    }


def test_the_retrieval_target_can_be_left_out_because_no_setting_changes_it() -> None:
    summary = summary_with(retrieval_p95=500.0)
    assert [m.name for m in slo_misses(summary)] == ["retrieval p95"]
    assert slo_misses(summary, include_retrieval=False) == []


def test_a_target_met_exactly_is_met() -> None:
    assert (
        slo_misses(summary_with(q2f_p50=2.5, q2f_p95=4.0, whole_p95=15.0, retrieval_p95=400.0))
        == []
    )


def test_a_missing_reading_is_neither_a_pass_nor_a_miss() -> None:
    assert slo_misses(summarize_runs([])) == []


# --- choosing the default --------------------------------------------------------------------


def result(
    key: tuple[str, int, int, int] = ("f16", 8192, 512, 5),
    *,
    quality: float | None = 1.0,
    q2f_p50: float = 1.0,
    peak: int | None = 9000,
    gpu: int | None = 100,
    loads: int = 0,
    q2f_p95: float = 1.5,
) -> ConfigResult:
    base = summary_with(q2f_p50=q2f_p50, q2f_p95=q2f_p95)
    summary = RunSummary(
        base.n,
        base.first_token,
        base.to_first_token,
        base.whole,
        base.tokens_per_s,
        base.retrieval_ms,
        loads,
        quality,
        base.quality_n,
    )
    kv, ctx, batch, top_k = key
    return ConfigResult(SweepPoint(kv, ctx, batch, top_k), 6.0, gpu, ctx, 1000, peak, summary)


def test_the_best_quality_wins_among_the_eligible() -> None:
    low = result(("f16", 4096, 256, 3), quality=0.80)
    high = result(("f16", 8192, 512, 5), quality=0.95)
    choice = choose_default([low, high])
    assert choice.point == high.point
    assert set(choice.eligible) == {low.point.key, high.point.key}


def test_a_tie_on_quality_goes_to_the_faster_first_token_then_the_smaller_peak() -> None:
    slow = result(("f16", 8192, 512, 5), q2f_p50=1.50)
    fast = result(("f16", 8192, 256, 5), q2f_p50=1.20)
    assert choose_default([slow, fast]).point == fast.point
    big = result(("f16", 8192, 512, 5), peak=9300)
    small = result(("q8_0", 8192, 512, 5), peak=8900)
    assert choose_default([big, small]).point == small.point


def test_qualities_within_the_tolerance_are_equal_and_speed_decides() -> None:
    """The first full bench asked 8 questions: one phrase moves the mean by 0.04, which is noise."""
    better_but_slow = result(("f16", 8192, 512, 5), quality=0.79, q2f_p50=1.10)
    nearly_as_good_and_fast = result(("q8_0", 8192, 512, 3), quality=0.75, q2f_p50=0.70)
    assert choose_default([better_but_slow, nearly_as_good_and_fast]).point == (
        nearly_as_good_and_fast.point
    )


def test_a_quality_gap_beyond_the_tolerance_is_a_real_difference() -> None:
    good = result(("f16", 8192, 512, 5), quality=0.90, q2f_p50=1.10)
    worse_and_fast = result(("q8_0", 8192, 512, 3), quality=0.80, q2f_p50=0.50)
    choice = choose_default([good, worse_and_fast])
    assert choice.point == good.point
    assert "1 of them are within 0.05" in choice.reason


def test_with_no_tolerance_the_best_quality_decides_even_by_a_hair() -> None:
    a = result(("f16", 8192, 512, 5), quality=0.752, q2f_p50=1.5)
    b = result(("q8_0", 8192, 512, 3), quality=0.750, q2f_p50=0.5)
    assert choose_default([a, b], quality_tolerance=0.0).point == a.point
    assert choose_default([a, b]).point == b.point  # the default tolerance calls them equal


def test_noise_in_the_fourth_decimal_of_quality_does_not_decide() -> None:
    a = result(("f16", 8192, 512, 5), quality=0.9001, q2f_p50=1.4)
    b = result(("f16", 8192, 256, 5), quality=0.9004, q2f_p50=1.1)
    assert choose_default([a, b]).point == b.point  # equal at three places, so speed decides


def test_a_model_that_is_not_fully_on_the_gpu_is_never_chosen() -> None:
    split = result(("f16", 8192, 512, 5), quality=1.0, gpu=88)
    resident = result(("q8_0", 4096, 256, 3), quality=0.5)
    choice = choose_default([split, resident])
    assert choice.point == resident.point
    assert "residency 88%" in choice.reason
    assert choose_default([result(gpu=None)]).point is None  # unreadable is not resident


def test_a_configuration_that_loaded_the_model_during_questions_is_excluded() -> None:
    choice = choose_default([result(loads=2), result(("q8_0", 8192, 512, 5), quality=0.1)])
    assert choice.point is not None and choice.point.kv_cache_type == "q8_0"
    assert "2 model loads" in choice.reason


def test_a_configuration_that_misses_a_target_is_excluded_and_the_miss_is_named() -> None:
    slow = result(("f16", 8192, 512, 5), q2f_p95=5.0)
    ok = result(("f16", 4096, 512, 5), quality=0.4)
    choice = choose_default([slow, ok])
    assert choice.point == ok.point
    assert "question to first token p95" in choice.reason


def test_a_retrieval_miss_alone_excludes_nobody() -> None:
    """Retrieval is on the CPU and the same under every setting, so it cannot choose between them."""
    base = summary_with(retrieval_p95=450.0)
    flawed = ConfigResult(SweepPoint("f16", 8192, 512, 5), 6.0, 100, 8192, 1000, 9000, base)
    assert choose_default([flawed]).point == flawed.point


def test_when_nothing_is_eligible_there_is_no_default_and_the_reasons_are_told() -> None:
    choice = choose_default([result(gpu=50), result(("q8_0", 8192, 512, 5), loads=1)])
    assert choice.point is None and choice.eligible == ()
    assert "residency 50%" in choice.reason and "1 model loads" in choice.reason


def test_no_results_means_no_default() -> None:
    choice = choose_default([])
    assert choice.point is None and "no configuration was measured" in choice.reason


def test_a_missing_quality_ranks_below_a_measured_one() -> None:
    unknown = result(("f16", 8192, 512, 5), quality=None)
    known = result(("f16", 4096, 256, 3), quality=0.2)
    assert choose_default([unknown, known]).point == known.point


@given(
    st.lists(
        st.tuples(
            st.sampled_from([("f16", 4096, 256, 3), ("f16", 8192, 512, 5), ("q8_0", 8192, 256, 3)]),
            st.floats(min_value=0, max_value=1),
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=2),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_the_chosen_configuration_is_always_resident_and_load_free(
    rows: list[tuple[tuple[str, int, int, int], float, int, int]],
) -> None:
    results = [result(key, quality=q, gpu=gpu, loads=loads) for key, q, gpu, loads in rows]
    choice = choose_default(results)
    if choice.point is None:
        assert not any(r.resident and r.summary.loads == 0 for r in results)
    else:  # some measurement of the chosen point was resident and load-free
        assert any(r.point == choice.point and r.resident and r.summary.loads == 0 for r in results)


# --- the report ------------------------------------------------------------------------------


def sample_result() -> dict[str, object]:
    good = result(("f16", 8192, 512, 5), quality=0.9)
    return {
        "name": "t",
        "created": "2026-10-24",
        "official": False,
        "official_notes": ["the GPU was not idle: 5000 MiB in use with our models unloaded"],
        "machine": {
            "chat_model": "gemma4:12b",
            "csv_model": "qwen2.5-coder:7b",
            "profile": "target",
            "embedder": "nomic-embed-text",
            "reranker": "MiniLM",
            "ollama": "0.34.2",
        },
        "guard": GuardVerdict(False, ("busy",), QUIET, 40.0).to_json(),
        "cpu": {"load_percent": 3},
        "ingest": {"documents": 15, "chunks": 65, "seconds": 13.0, "chunks_per_s": 5.0},
        "vram": {"total_mib": 10240, "idle_mib": 900, "chat_mib": 9300, "csv_mib": 6500},
        "sweep": [good.to_json()],
        "default_choice": {"reason": "ctx8192-kvf16-batch512-k5: the best"},
        "shipped_default": "ctx8192-kvf16-batch512-k5",
        "follow_up": {"setting": "x", "summary": summarize_runs([run()]).to_json()},
        "thinking": {"setting": "x", "summary": summarize_runs([run()]).to_json(), "note": "info"},
        "csv": {
            "model": "qwen2.5-coder:7b",
            "question_runs": 40,
            "suites": "hospital, sales",
            "p50_s": 2.3,
            "p95_s": 7.7,
            "max_s": 11.9,
            "load_events": 0,
            "solved": 33,
            "questions": 40,
            "swap": {"cycles": 3, "to_csv_median_s": 2.7, "to_chat_median_s": 5.2, "all_ok": True},
            "residency": "PASS",
            "official_notes": [],
        },
    }


def test_the_report_names_the_models_says_whether_it_is_official_and_shows_every_section() -> None:
    text = format_bench_report(sample_result())
    for wanted in (
        "gemma4:12b",
        "qwen2.5-coder:7b",
        "Official: NO",
        "the GPU was not idle",
        "GPU idle: **no**",
        "5.0 chunks per second",
        "## VRAM at the product defaults",
        "| 8192 | f16 | 512 | 5 |",
        "Chosen default",
        "Follow-up questions",
        "Thinking on",
        "## CSV mode",
        "p50 2.3",
    ):
        assert wanted in text, wanted
    assert "£" not in text  # plain text, pasteable from a Windows terminal


def test_a_saved_row_reads_back_as_the_result_it_was() -> None:
    original = result(("q8_0", 4096, 256, 3), quality=0.75, q2f_p50=0.7, peak=8896)
    again = config_result_from_json(original.to_json())
    assert again.point == original.point
    assert (again.gpu_percent, again.loaded_context, again.vram_peak_mib) == (100, 4096, 8896)
    assert again.summary.quality == 0.75 and again.summary.to_first_token == (
        original.summary.to_first_token
    )
    assert again.summary.loads == original.summary.loads and again.resident


def test_the_report_applies_todays_rule_to_the_saved_rows_not_the_sentence_saved_with_them() -> (
    None
):
    slow = result(("f16", 8192, 512, 5), quality=0.79, q2f_p50=1.10)
    fast = result(("q8_0", 8192, 512, 3), quality=0.75, q2f_p50=0.70)
    saved: dict[str, object] = {
        "sweep": [slow.to_json(), fast.to_json()],
        "default_choice": {"reason": "an older rule chose the slow one"},
    }
    text = format_bench_report(saved)
    assert "older rule" not in text
    assert "**Chosen default:** ctx8192-kvq8_0-batch512-k3" in text
    assert rechoose(saved)["default_choice"]["key"] == fast.point.key  # type: ignore[index]


def test_a_result_with_no_sweep_is_left_as_it_is() -> None:
    assert rechoose({"name": "x"}) == {"name": "x"}


def test_the_report_of_an_empty_result_does_not_crash() -> None:
    text = format_bench_report({})
    assert "Bench report" in text and "Official: NO" in text


def test_the_vram_table_computes_free_memory() -> None:
    text = format_bench_report(sample_result())
    assert "| chat model warm | 9300 | 940 |" in text
