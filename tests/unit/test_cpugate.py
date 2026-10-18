"""Tests for the CPU gate (step S2.6, D-078): reading the load and waiting for a quiet PC."""

import json
import subprocess

from geekbot.evals.cpugate import CpuSnapshot, read_snapshot, wait_until_idle

REPLY = json.dumps(
    {"load": 7, "top": [{"name": "python", "cpu": 41.5}, {"name": "ollama", "cpu": 3}]}
)


def test_the_snapshot_reads_the_load_and_the_busiest_processes() -> None:
    snapshot = read_snapshot(lambda script: REPLY)
    assert snapshot == CpuSnapshot(7.0, (("python", 41.5), ("ollama", 3.0)))
    assert snapshot.to_json() == {"load_percent": 7.0, "top": [["python", 41.5], ["ollama", 3.0]]}


def test_no_busy_process_is_an_empty_list_not_an_error() -> None:
    reply = json.dumps({"load": 3, "top": []})
    assert read_snapshot(lambda script: reply) == CpuSnapshot(3.0, ())


def test_any_failure_becomes_an_unchecked_snapshot_never_an_exception() -> None:
    def broken(script: str) -> str:
        raise OSError("no powershell here")

    def timed_out(script: str) -> str:
        raise subprocess.TimeoutExpired("powershell", 60)

    assert read_snapshot(broken) == CpuSnapshot(None, ())
    assert read_snapshot(timed_out) == CpuSnapshot(None, ())
    assert read_snapshot(lambda script: "not json") == CpuSnapshot(None, ())
    assert read_snapshot(lambda script: "[1, 2]") == CpuSnapshot(None, ())  # wrong shape


def test_an_idle_pc_passes_at_once_without_waiting() -> None:
    slept: list[float] = []
    reading = wait_until_idle(
        idle_percent=15, read=lambda: CpuSnapshot(4.0, ()), sleep=slept.append
    )
    assert reading.idle is True
    assert reading.waited_s == 0.0
    assert slept == []
    assert reading.after is None


def test_a_busy_pc_is_waited_for_until_it_quietens() -> None:
    loads = iter([90.0, 60.0, 10.0])
    slept: list[float] = []
    reading = wait_until_idle(
        idle_percent=15,
        pause_s=30,
        read=lambda: CpuSnapshot(next(loads), ()),
        sleep=slept.append,
    )
    assert reading.idle is True
    assert slept == [30, 30]
    assert reading.waited_s == 60
    assert reading.before.load_percent == 10.0


def test_a_pc_that_never_quietens_is_flagged_not_assumed_idle() -> None:
    slept: list[float] = []
    reading = wait_until_idle(
        idle_percent=15,
        attempts=3,
        read=lambda: CpuSnapshot(80.0, (("game", 70.0),)),
        sleep=slept.append,
    )
    assert reading.idle is False
    assert len(slept) == 2
    assert reading.to_json()["idle"] is False
    assert reading.to_json()["before"]["top"] == [["game", 70.0]]


def test_an_unreadable_load_is_never_called_idle() -> None:
    reading = wait_until_idle(attempts=2, read=lambda: CpuSnapshot(None, ()), sleep=lambda s: None)
    assert reading.idle is False
