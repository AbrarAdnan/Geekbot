"""Tests for step S5.1: the bench's orchestration, with a fake session and no machine.

What is checked is the order of the work (one server per KV-cache type, one model load per context
and batch pair), what is shared, what a failure does to the rest of a long sweep, and that the VRAM
sampler really keeps the highest reading. The real session (`GraphSession`) and the whole run are
live-tested.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

import pytest

from geekbot.config import Settings
from geekbot.evals import bench_run
from geekbot.evals.bench import BenchConfig, ChatRun, SweepAxes, SweepPoint, expand_sweep
from geekbot.evals.bench_run import (
    GraphSession,
    VramSampler,
    read_gpu_reading,
    run_cold_rounds,
    run_point,
    run_sweep,
    select_items,
)
from geekbot.evals.datasets import GoldenItem
from geekbot.llm.models import RunningModel

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


def item(
    identifier: str, kind: str = "single_hop", phrases: tuple[str, ...] = ("90 days",)
) -> GoldenItem:
    return GoldenItem(
        id=identifier,
        type=kind,  # type: ignore[arg-type]
        question=f"question {identifier}?",
        sources=("a.md",),
        must_include=phrases,
        earlier_turns=("first turn?",) if kind == "multi_turn" else (),
    )


class FakeSession:
    """Plays ``ChatSession`` and writes down every call."""

    def __init__(self, *, fail_on: str | None = None, gpu: int | None = 100) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.gpu = gpu

    def load(self, point: SweepPoint) -> float:
        self.calls.append(f"load {point.num_ctx}/{point.num_batch}")
        if self.fail_on == "load":
            raise RuntimeError("out of memory")
        return 6.0

    def unload(self) -> None:
        self.calls.append("unload")

    def residency(self) -> tuple[int | None, int | None]:
        self.calls.append("residency")
        return self.gpu, 8192

    def ask(self, item: GoldenItem, point: SweepPoint, *, think: bool = False) -> ChatRun:
        self.calls.append(f"ask {item.id} k={point.top_k}{' think' if think else ''}")
        if self.fail_on == item.id:
            raise RuntimeError("model crashed")
        return ChatRun(item.id, 0.4, 1.5, 60.0, 0.01, 300.0, 1.0, thinking=think)

    def ask_followup(self, item: GoldenItem, point: SweepPoint) -> ChatRun:
        self.calls.append(f"followup {item.id}")
        if self.fail_on == "followup":
            raise RuntimeError("no rewrite")
        return ChatRun(item.id, 0.5, 2.0, 55.0, 0.01, 320.0, 0.5, rewrite_s=0.8)


class FakeSampler:
    peak_mib = 9100

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def sampler() -> FakeSampler:
    return FakeSampler()


def lines() -> tuple[list[str], Any]:
    seen: list[str] = []
    return seen, seen.append


POINT = SweepPoint("f16", 8192, 512, 5)


# --- the VRAM sampler ------------------------------------------------------------------------


def test_the_sampler_keeps_the_highest_reading_including_entry_and_exit() -> None:
    readings = iter([1000, 9300, 8000, 9500])  # entry, then the thread, then exit
    last = 9500

    def read() -> int | None:
        return next(readings, last)

    with VramSampler(read, interval_s=0.001) as sample:
        pass
    assert sample.peak_mib is not None and sample.peak_mib >= 1000
    assert sample.peak_mib <= 9500


def test_a_block_has_at_least_two_readings_even_when_it_is_short() -> None:
    reads: list[int] = []

    def read() -> int | None:
        reads.append(len(reads))
        return len(reads)

    with VramSampler(read, interval_s=60.0) as sample:
        pass
    assert len(reads) == 2  # one on entry, one on exit, none from the thread
    assert sample.peak_mib == 2


def test_a_reader_that_fails_adds_nothing_and_no_reading_means_no_peak() -> None:
    with VramSampler(lambda: None, interval_s=0.001) as sample:
        pass
    assert sample.peak_mib is None


def test_the_peak_is_never_lowered_by_a_later_reading() -> None:
    values = iter([5000, 9000, 4000, 4000, 4000])
    with VramSampler(lambda: next(values, 4000), interval_s=0.001) as sample:
        pass
    assert sample.peak_mib in (5000, 9000)  # depends how many the thread got in; never 4000


# --- choosing the items ----------------------------------------------------------------------


def test_the_sweep_asks_answerable_questions_and_the_extras_use_the_multi_turn_items() -> None:
    items = [item(f"S-{n:03d}") for n in range(20)]
    items += [item(f"M-{n:03d}", "multi_hop") for n in range(10)]
    items += [item(f"U-{n:03d}", "unanswerable", ()) for n in range(5)]
    items += [item(f"T-{n:03d}", "multi_turn") for n in range(6)]
    config = BenchConfig.model_validate({"chat": {"questions": 4, "followups": 2}})
    questions, follow_ups = select_items(items, config)
    assert len(questions) == 4 and len(follow_ups) == 2
    assert {q.type for q in questions} <= {"single_hop", "multi_hop"}
    assert all(f.type == "multi_turn" for f in follow_ups)
    assert (questions, follow_ups) == select_items(items, config)  # the same every run


# --- one point -------------------------------------------------------------------------------


def run(
    session: FakeSession, items: list[GoldenItem], *, load: bool = True, passes: int = 2
) -> Any:
    seen, log = lines()
    result = run_point(
        session,
        POINT,
        items,
        passes=passes,
        load=load,
        sampler=sampler,
        read_used_mib=lambda: 950,
        cpu_idle=lambda: True,
        log=log,
    )
    return result, seen


def test_a_point_loads_reads_residency_throws_one_question_away_then_asks_every_pass() -> None:
    session = FakeSession()
    result, seen = run(session, [item("S-001"), item("S-002")], passes=2)
    assert session.calls == [
        "load 8192/512",
        "residency",
        "ask S-001 k=5",  # the discarded first call
        "ask S-001 k=5",
        "ask S-002 k=5",
        "ask S-001 k=5",
        "ask S-002 k=5",
    ]
    assert result.summary.n == 4  # the discarded call is not counted
    assert result.load_s == 6.0 and result.gpu_percent == 100 and result.loaded_context == 8192
    assert result.vram_idle_mib == 950 and result.vram_peak_mib == 9100
    assert result.summary.quality == 1.0 and result.notes == ()
    assert seen and "GPU 100%" in seen[0]


def test_a_point_that_shares_the_models_load_does_not_load_again() -> None:
    session = FakeSession()
    result, _ = run(session, [item("S-001")], load=False, passes=1)
    assert not any(call.startswith("load") for call in session.calls)
    assert result.load_s == 0.0


def test_a_point_that_fails_is_a_result_with_a_note_and_no_runs() -> None:
    session = FakeSession(fail_on="load")
    result, seen = run(session, [item("S-001")])
    assert result.summary.n == 0 and result.gpu_percent is None
    assert "RuntimeError: out of memory" in result.notes[0]
    assert "failed" in seen[0]


def test_a_question_that_crashes_keeps_the_runs_before_it() -> None:
    session = FakeSession(fail_on="S-002")
    result, _ = run(session, [item("S-001"), item("S-002")], passes=1)
    assert result.summary.n == 1  # S-001 was asked before S-002 crashed
    assert "model crashed" in result.notes[0]


def test_the_cpu_gate_reading_is_recorded_with_the_point() -> None:
    _seen, log = lines()
    result = run_point(
        FakeSession(),
        POINT,
        [item("S-001")],
        passes=1,
        load=True,
        sampler=sampler,
        read_used_mib=lambda: None,
        cpu_idle=lambda: False,
        log=log,
    )
    assert result.cpu_idle is False and result.vram_idle_mib is None


# --- the sweep -------------------------------------------------------------------------------


class FakeServers:
    """Opens a FakeSession per KV type and records that each was opened and closed."""

    def __init__(self, *, fail_to_open: str | None = None, fail_on_load: str | None = None) -> None:
        self.events: list[str] = []
        self.sessions: dict[str, FakeSession] = {}
        self.fail_to_open = fail_to_open
        self.fail_on_load = fail_on_load

    @contextmanager
    def open(self, kv: str) -> Iterator[FakeSession]:
        self.events.append(f"open {kv}")
        if kv == self.fail_to_open:
            raise RuntimeError("port in use")
        session = FakeSession(fail_on="load" if kv == self.fail_on_load else None)
        self.sessions[kv] = session
        try:
            yield session
        finally:
            self.events.append(f"close {kv}")


def sweep(servers: FakeServers, points: list[SweepPoint]) -> list[Any]:
    _, log = lines()
    return run_sweep(
        points,
        servers.open,
        [item("S-001")],
        passes=1,
        sampler=sampler,
        read_used_mib=lambda: 1000,
        cpu_idle=lambda: True,
        log=log,
    )


def test_one_server_per_kv_type_and_one_model_load_per_context_and_batch_pair() -> None:
    servers = FakeServers()
    points = expand_sweep(SweepAxes())
    results = sweep(servers, points)
    assert [r.point for r in results] == points  # every point, in order
    assert servers.events == ["open f16", "close f16", "open q8_0", "close q8_0"]
    for kv, session in servers.sessions.items():
        loads = [c for c in session.calls if c.startswith("load")]
        assert loads == [f"load {c}/{b}" for c in (4096, 8192) for b in (256, 512)], kv
    # the second top-k of a pair reused the loaded model
    assert [r.load_s for r in results[:4]] == [6.0, 0.0, 6.0, 0.0]


def test_a_server_that_cannot_start_gives_its_points_the_reason_and_the_others_still_run() -> None:
    servers = FakeServers(fail_to_open="f16")
    results = sweep(servers, expand_sweep(SweepAxes()))
    assert len(results) == 16
    failed = [r for r in results if r.point.kv_cache_type == "f16"]
    assert len(failed) == 8 and all("port in use" in r.notes[0] for r in failed)
    assert all(r.summary.n == 0 for r in failed)
    ran = [r for r in results if r.point.kv_cache_type == "q8_0"]
    assert all(r.summary.n == 1 for r in ran)


def test_a_point_that_fails_to_load_does_not_stop_the_points_after_it() -> None:
    servers = FakeServers(fail_on_load="f16")
    results = sweep(servers, expand_sweep(SweepAxes()))
    assert len(results) == 16
    f16 = [r for r in results if r.point.kv_cache_type == "f16"]
    assert all(r.notes and r.summary.n == 0 for r in f16)
    # the first point of each (context, batch) pair tried the load; the second shares it and is skipped
    assert [r.notes[0].startswith("skipped") for r in f16] == [False, True] * 4
    assert "out of memory" in f16[1].notes[0]
    assert not any(c.startswith("ask") for c in servers.sessions["f16"].calls)
    assert all(not r.notes for r in results if r.point.kv_cache_type == "q8_0")
    assert servers.events == ["open f16", "close f16", "open q8_0", "close q8_0"]  # always closed


def test_an_error_inside_the_server_block_still_closes_it_and_marks_the_unfinished_points() -> None:
    class Exploding(FakeServers):
        @contextmanager
        def open(self, kv: str) -> Iterator[FakeSession]:
            with super().open(kv) as session:
                yield session
                raise RuntimeError("could not stop cleanly")

    servers = Exploding()
    results = sweep(servers, [SweepPoint("f16", 8192, 512, 5), SweepPoint("f16", 8192, 512, 3)])
    assert len(results) == 2  # nothing is duplicated and nothing is lost
    assert "close f16" in servers.events


def test_the_sweep_of_no_points_is_empty() -> None:
    assert sweep(FakeServers(), []) == []


# --- the extras ------------------------------------------------------------------------------


def test_cold_rounds_unload_load_and_ask_one_question_each() -> None:
    session = FakeSession()
    runs = run_cold_rounds(session, POINT, item("S-001"), rounds=2)
    assert len(runs) == 2
    assert session.calls == ["unload", "load 8192/512", "ask S-001 k=5"] * 2


def test_a_follow_up_run_counts_the_rewrite_in_the_wait_before_the_first_word() -> None:
    run_ = FakeSession().ask_followup(item("T-001", "multi_turn"), POINT)
    assert run_.question_to_first_token_s == pytest.approx(0.8 + 0.32 + 0.5)


def test_the_extras_helpers_report_a_failure_as_a_note_not_a_crash() -> None:
    session = FakeSession(fail_on="followup")
    follow = bench_run._follow_ups(session, POINT, [item("T-001", "multi_turn")], passes=2)
    assert follow["summary"]["n"] == 0 and "no rewrite" in follow["note"]
    config = BenchConfig()
    thinking = bench_run._thinking(FakeSession(fail_on="S-001"), POINT, [item("S-001")], config)
    assert thinking["summary"]["n"] == 0 and "could not be measured" in thinking["note"]
    ok = bench_run._thinking(FakeSession(), POINT, [item("S-001"), item("S-002")], config)
    assert ok["summary"]["n"] == 2


# --- the real session's small parts ----------------------------------------------------------


class FakeOllama:
    def __init__(self, loaded: list[RunningModel]) -> None:
        self.loaded = loaded

    def running(self) -> list[RunningModel]:
        return self.loaded


def real_session(loaded: list[RunningModel]) -> GraphSession:
    settings = Settings(_env_file=None, profile="target")
    return GraphSession(settings, FakeOllama(loaded), None, None, None, None, None)  # type: ignore[arg-type]


def test_the_session_gives_the_point_s_load_time_options_to_the_settings() -> None:
    settings = real_session([]).settings_for(SweepPoint("q8_0", 4096, 256, 3))
    assert (settings.chat_num_ctx, settings.chat_num_batch) == (4096, 256)


def test_the_session_reads_the_chat_models_residency_and_nothing_else() -> None:
    session = real_session(
        [
            RunningModel("nomic-embed-text:latest", 376, 0, 2048),
            RunningModel("gemma4:12b", 8_100, 8_100, 4096),
        ]
    )
    assert session.residency() == (100, 4096)
    assert real_session([RunningModel("other:1b", 100, 100, 1)]).residency() == (None, None)


def test_a_graph_result_becomes_a_run_with_recall_the_rewrite_and_the_refusal() -> None:
    session = real_session([])
    final = {
        "generation": {
            "seconds": 1.5,
            "first_token_seconds": 0.5,
            "load_seconds": 0.02,
            "output_tokens": 40,
            "tokens_per_second": 60.0,
        },
        "retrieval_ms": {"total": 330.0, "dense": 25.0},
        "answer": "Chat history is kept for 90 DAYS [1].",
        "abstained": False,
        "condense_seconds": 0.9,
    }
    run_ = session._to_run(item("S-001"), final, think=False)
    assert run_.recall == 1.0 and run_.rewrite_s == 0.9 and run_.retrieval_ms == 330.0
    assert (run_.first_token_s, run_.whole_s, run_.load_s) == (0.5, 1.5, 0.02)
    assert run_.output_tokens == 40 and not run_.abstained and not run_.thinking


def test_a_turn_that_never_reached_the_model_is_an_error_not_a_zero_second_run() -> None:
    with pytest.raises(RuntimeError, match="never called"):
        real_session([])._to_run(item("S-001"), {"answer": "x"}, think=False)


# --- reading the GPU -------------------------------------------------------------------------


def test_the_gpu_reading_combines_memory_utilisation_and_compute_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(command: tuple[str, ...], **_kwargs: object) -> str:
        return (
            "python.exe, 77\n"
            if "--query-compute-apps=process_name,pid" in command
            else "976, 10240, 12\n"
        )

    monkeypatch.setattr("geekbot.llm.vram.run_command", fake)
    reading = read_gpu_reading()
    assert reading is not None
    assert (reading.used_mib, reading.utilization_percent) == (976, 12.0)
    assert reading.compute_apps == ("python.exe (77)",)


def test_a_failing_nvidia_smi_is_no_reading_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    from geekbot.llm.vram import CommandError

    def broken(command: tuple[str, ...], **_kwargs: object) -> str:
        raise CommandError("nvidia-smi is missing")

    monkeypatch.setattr("geekbot.llm.vram.run_command", broken)
    assert read_gpu_reading() is None


def test_a_listing_of_compute_processes_that_fails_does_not_lose_the_memory_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geekbot.llm.vram import CommandError

    def fake(command: tuple[str, ...], **_kwargs: object) -> str:
        if "--query-compute-apps=process_name,pid" in command:
            raise CommandError("not supported")
        return "976, 10240, 12\n"

    monkeypatch.setattr("geekbot.llm.vram.run_command", fake)
    reading = read_gpu_reading()
    assert reading is not None and reading.compute_apps == ()


def test_the_results_folder_of_a_config_is_under_its_name(tmp_path: Path) -> None:
    config = BenchConfig.model_validate({"name": "x", "results_dir": str(tmp_path)})
    assert config.results_path == tmp_path / "x"
    assert json.dumps({"ok": True})  # the result is plain JSON data (see `run_bench`)
