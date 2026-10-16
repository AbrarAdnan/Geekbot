"""Tests for step C3: the swap timing run behind ``geekbot csv swap`` (fake Ollama, no GPU)."""

import pytest

from geekbot.config import Settings
from geekbot.llm.vram import CommandError, GpuMemory, LoadedModel
from geekbot.swapcheck import (
    SETUP,
    TO_CHAT,
    TO_CSV,
    SwapRecord,
    format_swap_report,
    run_swap_cycles,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

CHAT, CSV = "gemma3:4b", "qwen2.5-coder:3b"


class FakeOllama:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.warmed: list[str] = []

    def loaded_models(self) -> list[str]:
        return list(self.loaded)

    def unload(self, model: str) -> None:
        self.loaded.remove(model)

    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:
        self.warmed.append(model)
        self.loaded.append(model)
        return 2.0


def _ps(client: FakeOllama, *, gpu: int = 100) -> list[LoadedModel]:
    return [LoadedModel(name, gpu, 100 - gpu, None, "Forever") for name in client.loaded]


def _run(client: FakeOllama, *, cycles: int = 2, gpu: int = 100) -> list[SwapRecord]:
    return run_swap_cycles(
        Settings(_env_file=None),
        client,
        cycles=cycles,
        read_gpu=lambda: GpuMemory(5000, 10240),
        read_ps=lambda: _ps(client, gpu=gpu),
    )


def test_it_sets_up_with_the_chat_model_then_alternates_in_both_directions() -> None:
    client = FakeOllama()
    records = _run(client, cycles=2)
    assert [r.label for r in records] == [SETUP, TO_CSV, TO_CHAT, TO_CSV, TO_CHAT]
    assert client.warmed == [CHAT, CSV, CHAT, CSV, CHAT]
    assert client.loaded == [CHAT]  # one model at a time, all the way through
    assert all(r.ok for r in records)
    assert all(r.gpu_used_mib == 5000 for r in records)


def test_a_model_left_partly_on_the_cpu_fails_that_swap() -> None:
    records = _run(FakeOllama(), cycles=1, gpu=52)
    assert not records[1].ok
    assert records[1].residency is not None
    assert records[1].residency.status == "FAIL"


def test_unreadable_ps_and_gpu_leave_the_checks_out_instead_of_crashing() -> None:
    client = FakeOllama()

    def broken() -> list[LoadedModel]:
        raise CommandError("ollama not found")

    def no_gpu() -> GpuMemory:
        raise CommandError("nvidia-smi not found")

    records = run_swap_cycles(
        Settings(_env_file=None), client, cycles=1, read_gpu=no_gpu, read_ps=broken
    )
    for record in records:
        assert (record.residency, record.exclusive, record.gpu_used_mib) == (None, None, None)
        assert record.ok  # unknown is not a failure; the report shows "?"
    assert "?" in format_swap_report(Settings(_env_file=None), records)


def test_the_report_names_the_models_and_summarises_each_direction() -> None:
    settings = Settings(_env_file=None)
    text = format_swap_report(settings, _run(FakeOllama(), cycles=3))
    assert f"chat {CHAT} <-> csv {CSV}" in text
    assert "profile=proto" in text
    assert f"{TO_CSV}: 3 swap(s)" in text
    assert f"{TO_CHAT}: 3 swap(s)" in text
    assert "min 2.0 s / median 2.0 s / max 2.0 s" in text
    assert "is not counted" in text  # the setup swap
    assert "[FAIL]" not in text


def test_the_report_lists_every_check_that_is_not_a_pass() -> None:
    settings = Settings(_env_file=None)
    text = format_swap_report(settings, _run(FakeOllama(), cycles=1, gpu=52))
    assert f"[FAIL] after {TO_CSV}: residency (R-1)" in text
    assert "52% GPU" in text
