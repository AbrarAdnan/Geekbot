"""Tests for step C3: the explicit model swap (PLAN.md §6.5 R-4), with a fake Ollama and a fake
clock, so a 30-second timeout takes no time and nothing touches the GPU."""

import pytest

from geekbot.config import Settings
from geekbot.llm.models import (
    OllamaError,
    Patience,
    swap_to,
    swap_to_mode,
    unload_and_wait,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

CHAT = "gemma3:4b"
CSV = "qwen2.5-coder:3b"


class FakeOllama:
    """Plays Ollama's model memory. An unloaded model stays listed for ``lag`` more polls
    (Ollama frees it a moment after the request returns); ``stuck`` never frees it."""

    def __init__(self, loaded: list[str], *, lag: int = 0, stuck: bool = False) -> None:
        self.loaded = list(loaded)
        self.lag, self.stuck = lag, stuck
        self.pending: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []
        self.warm_up_args: list[tuple[str, int, str]] = []
        self.batches: list[int | None] = []

    def loaded_models(self) -> list[str]:
        self.calls.append(("ps", ""))
        for name in list(self.pending):
            if self.pending[name] == 0:
                self.loaded.remove(name)
                del self.pending[name]
            else:
                self.pending[name] -= 1
        return list(self.loaded)

    def unload(self, model: str) -> None:
        self.calls.append(("unload", model))
        if not self.stuck:
            self.pending[model] = self.lag

    def warm_up(
        self, model: str, *, num_ctx: int, keep_alive: str, num_batch: int | None = None
    ) -> float:
        self.calls.append(("warm_up", model))
        self.warm_up_args.append((model, num_ctx, keep_alive))
        self.batches.append(num_batch)
        self.loaded.append(model)
        return 2.5


class FakeTime:
    """A clock that only moves when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def patience(self, *, timeout_s: float = 30.0, poll_s: float = 0.25) -> Patience:
        return Patience(timeout_s=timeout_s, poll_s=poll_s, sleep=self.sleep, clock=self.clock)


def _swap(client: FakeOllama, time: FakeTime, *, release: list[str]) -> tuple[object, ...]:
    result = swap_to(
        client,
        CSV,
        release=release,
        num_ctx=16384,
        keep_alive="30m",
        patience=time.patience(),
    )
    return (result.loaded, result.unloaded, result.unload_s, result.warm_up_s, result.total_s)


def test_the_other_model_is_unloaded_before_the_new_one_is_warmed_up() -> None:
    client, time = FakeOllama([CHAT]), FakeTime()
    loaded, unloaded, _, _, _ = _swap(client, time, release=[CHAT])
    assert (loaded, unloaded) == (CSV, (CHAT,))
    assert ("unload", CHAT) in client.calls
    assert client.calls[-1] == ("warm_up", CSV)  # the warm-up is the very last thing
    assert client.loaded == [CSV]


def test_the_swap_waits_until_ollama_stops_listing_the_old_model() -> None:
    client, time = FakeOllama([CHAT], lag=2), FakeTime()
    _, _, unload_s, warm_up_s, total_s = _swap(client, time, release=[CHAT])
    assert time.sleeps == [0.25, 0.25]  # listed on two polls, gone on the third
    assert unload_s == pytest.approx(0.5)
    assert warm_up_s == 2.5  # as reported by the client
    assert total_s == pytest.approx(3.0)
    assert client.calls.index(("warm_up", CSV)) > max(
        i for i, call in enumerate(client.calls) if call == ("ps", "")
    )


def test_a_model_that_never_goes_away_is_an_error_and_nothing_is_warmed_up() -> None:
    client, time = FakeOllama([CHAT], stuck=True), FakeTime()
    with pytest.raises(OllamaError, match="still loaded"):
        _swap(client, time, release=[CHAT])
    assert ("warm_up", CSV) not in client.calls
    assert sum(time.sleeps) >= 30  # it did wait the whole time before giving up


def test_nothing_is_unloaded_when_the_other_model_is_not_loaded() -> None:
    client, time = FakeOllama([]), FakeTime()
    _, unloaded, unload_s, _, _ = _swap(client, time, release=[CHAT])
    assert unloaded == ()
    assert unload_s == 0
    assert not [call for call in client.calls if call[0] == "unload"]


def test_only_models_that_are_loaded_are_unloaded() -> None:
    client, time = FakeOllama([CHAT, "nomic-embed-text:latest"]), FakeTime()
    _swap(client, time, release=[CHAT, "never-loaded:1b"])
    assert [call for call in client.calls if call[0] == "unload"] == [("unload", CHAT)]


def test_the_new_model_is_never_unloaded_even_if_it_is_in_the_release_list() -> None:
    client, time = FakeOllama([CSV]), FakeTime()
    _, unloaded, _, _, _ = _swap(client, time, release=[CSV])
    assert unloaded == ()
    assert [call for call in client.calls if call[0] == "unload"] == []
    assert ("warm_up", CSV) in client.calls


def test_a_bare_name_and_its_latest_tag_are_the_same_model() -> None:
    client = FakeOllama(["old-model:latest"])
    assert unload_and_wait(client, ["old-model"], patience=FakeTime().patience()) == (
        "old-model:latest",
    )


def test_the_new_model_gets_the_modes_context_and_keep_alive() -> None:
    client = FakeOllama([])
    swap_to(
        client, CSV, release=[], num_ctx=16384, keep_alive="30m", patience=FakeTime().patience()
    )
    assert client.warm_up_args == [(CSV, 16384, "30m")]


def test_swap_to_mode_picks_the_models_and_contexts_from_the_settings() -> None:
    settings = Settings(_env_file=None)  # the proto profile: gemma3:4b and qwen2.5-coder:3b
    client, time = FakeOllama([CHAT]), FakeTime()
    result = swap_to_mode(client, settings, "csv", patience=time.patience())
    assert (result.loaded, result.unloaded) == (CSV, (CHAT,))
    assert client.warm_up_args == [(CSV, settings.csv_num_ctx, settings.keep_alive)]

    back = swap_to_mode(client, settings, "chat", patience=time.patience())
    assert (back.loaded, back.unloaded) == (CHAT, (CSV,))
    assert client.warm_up_args[-1] == (CHAT, settings.chat_num_ctx, settings.keep_alive)
    assert client.loaded == [CHAT]


def test_only_the_chat_model_is_warmed_up_with_a_batch_size() -> None:
    """Step S5.1: a batch size is a load-time option, so the warm-up and the questions must agree
    (R-2); the CSV model is asked with none, and so is warmed up with none."""
    settings = Settings(_env_file=None, chat_num_batch=256)
    client, time = FakeOllama([]), FakeTime()
    swap_to_mode(client, settings, "chat", patience=time.patience())
    swap_to_mode(client, settings, "csv", patience=time.patience())
    assert client.batches == [256, None]


def test_with_no_batch_size_the_warm_up_is_the_old_call_the_old_fakes_accept() -> None:
    class OldFake(FakeOllama):
        def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:  # type: ignore[override]
            return super().warm_up(model, num_ctx=num_ctx, keep_alive=keep_alive)

    client, time = OldFake([]), FakeTime()
    swap_to_mode(client, Settings(_env_file=None), "chat", patience=time.patience())
    assert client.batches == [None]
