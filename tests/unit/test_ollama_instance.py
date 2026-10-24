"""Tests for step S5.1: the private Ollama server of the KV-cache sweep. A fake process and a fake
clock stand in for ``ollama serve``, so nothing here starts a server or waits."""

import subprocess
from typing import Any

import pytest

from geekbot.evals.ollama_instance import (
    DEFAULT_PORT,
    InstanceSpec,
    OllamaInstanceError,
    PrivateOllama,
)


class FakeProcess:
    """Plays ``subprocess.Popen``: alive until ``exit_code`` is set."""

    def __init__(self, *, exit_code: int | None = None) -> None:
        self.pid = 4242
        self.exit_code = exit_code
        self.killed = False
        self.waited = 0

    def poll(self) -> int | None:
        return self.exit_code

    def wait(self, timeout: float | None = None) -> int:
        self.waited += 1
        return self.exit_code or 0

    def terminate(self) -> None:
        self.exit_code = 0

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Harness:
    """Records what the manager did; ``answers_after`` is how many probes fail before one succeeds."""

    def __init__(self, *, answers_after: int = 2, process: FakeProcess | None = None) -> None:
        self.process = process or FakeProcess()
        self.answers_after = answers_after
        self.probes = 0
        self.popen_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.killed: list[FakeProcess] = []
        self.time = FakeTime()

    def popen(self, command: list[str], **kwargs: Any) -> FakeProcess:
        self.popen_calls.append((command, kwargs))
        return self.process

    def probe(self, host: str) -> bool:
        self.probes += 1
        return self.probes > self.answers_after

    def build(self, spec: InstanceSpec | None = None, **kwargs: Any) -> PrivateOllama:
        return PrivateOllama(
            spec or InstanceSpec(),
            popen=self.popen,
            environ={"PATH": "x", "OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_KV_CACHE_TYPE": "q4_0"},
            probe=self.probe,
            kill=lambda process: self.killed.append(process),  # type: ignore[arg-type]
            clock=self.time.clock,
            sleep=self.time.sleep,
            **kwargs,
        )


def test_the_spec_sets_the_server_the_plan_wants_and_never_the_users_port() -> None:
    env = InstanceSpec(kv_cache_type="q8_0").environment(
        {"PATH": "x", "OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_KV_CACHE_TYPE": "f16"}
    )
    assert env["OLLAMA_HOST"] == f"127.0.0.1:{DEFAULT_PORT}"  # replaced, and loopback only
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"  # the parent's value is overridden
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"  # R-3
    assert env["OLLAMA_NUM_PARALLEL"] == "1"  # R-3
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "2"  # R-2: the embedder and the chat model both stay
    assert env["PATH"] == "x"  # the rest is inherited


def test_the_default_port_is_not_the_users_ollama() -> None:
    assert DEFAULT_PORT != 11434
    assert InstanceSpec().host == f"http://127.0.0.1:{DEFAULT_PORT}"


@pytest.mark.parametrize(
    "bad",
    [
        {"kv_cache_type": "q2"},
        {"kv_cache_type": "q8_0", "flash_attention": False},
        {"port": 0},
        {"port": 70000},
    ],
)
def test_nonsense_specs_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        InstanceSpec(**bad)


def test_start_launches_ollama_serve_with_the_environment_and_waits_until_it_answers() -> None:
    harness = Harness(answers_after=3)  # one probe before the start, then two that fail
    with harness.build(InstanceSpec(kv_cache_type="q8_0")) as host:
        assert host == f"http://127.0.0.1:{DEFAULT_PORT}"
    ((command, kwargs),) = harness.popen_calls
    assert command == ["ollama", "serve"]
    assert kwargs["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert harness.probes == 4
    assert harness.killed == [harness.process]  # leaving the block stops it


def test_a_port_that_already_answers_is_refused_and_nothing_is_started() -> None:
    harness = Harness(answers_after=-1)  # the very first probe already answers
    with pytest.raises(OllamaInstanceError, match="already answers"):
        harness.build().start()
    assert harness.popen_calls == []


def test_a_server_that_dies_before_it_answers_is_an_error_with_its_exit_code() -> None:
    harness = Harness(answers_after=99, process=FakeProcess(exit_code=1))
    with pytest.raises(OllamaInstanceError, match="exited with code 1"):
        harness.build().start()


def test_a_server_that_stays_silent_is_stopped_and_reported() -> None:
    harness = Harness(answers_after=10_000)
    with pytest.raises(OllamaInstanceError, match="did not answer"):
        harness.build(start_timeout_s=2.0).start()
    assert harness.killed == [harness.process]  # it is not left running


def test_a_missing_ollama_program_is_an_error_not_a_traceback() -> None:
    def no_such_program(command: list[str], **kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("ollama")

    manager = PrivateOllama(InstanceSpec(), popen=no_such_program, probe=lambda host: False)
    with pytest.raises(OllamaInstanceError, match="could not start"):
        manager.start()


def test_the_block_stops_the_server_even_when_the_work_inside_it_fails() -> None:
    harness = Harness()
    with pytest.raises(RuntimeError, match="boom"), harness.build():
        raise RuntimeError("boom")
    assert harness.killed == [harness.process]


def test_stop_is_safe_to_repeat_and_to_call_before_start() -> None:
    harness = Harness()
    manager = harness.build()
    manager.stop()  # before start: nothing to do
    manager.start()
    manager.stop()
    manager.stop()
    assert harness.killed == [harness.process]  # killed once


def test_starting_twice_is_an_error() -> None:
    harness = Harness()
    manager = harness.build()
    manager.start()
    with pytest.raises(OllamaInstanceError, match="already started"):
        manager.start()
    manager.stop()


def test_a_stubborn_process_is_killed_after_the_polite_stop_times_out() -> None:
    class Stubborn(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            self.waited += 1
            if not self.killed:
                raise subprocess.TimeoutExpired("ollama", timeout or 0)
            return -9

    harness = Harness(process=Stubborn())
    manager = harness.build()
    manager.start()
    manager.stop()
    assert harness.process.killed
