"""Tests for step S0.3: the doctor's checks, with made-up data (no GPU, no Ollama)."""

from pathlib import Path
from typing import Self

import pytest

from geekbot.config import Settings
from geekbot.doctor import (
    BUSY_GPU_USED_MIB,
    Check,
    Status,
    check_embedder_on_cpu,
    check_embedder_pulled,
    check_exclusive,
    check_gpu_memory,
    check_models_pulled,
    check_residency,
    check_storage_lock,
    describe_swap,
    format_checks,
    has_failure,
    run_doctor,
)
from geekbot.index.lock import StorageLock
from geekbot.llm.models import OllamaError, SwapResult
from geekbot.llm.vram import CommandError, GpuMemory, LoadedModel

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

IDLE_GPU = GpuMemory(used_mib=654, total_mib=10240)
EMBED = "nomic-embed-text"
PULLED = ["gemma3:4b", "qwen2.5-coder:3b", "nomic-embed-text:latest"]
CHAT, CSV = "gemma3:4b", "qwen2.5-coder:3b"


def _model(gpu: int = 100, context: int | None = 8192, name: str = CHAT) -> LoadedModel:
    return LoadedModel(name, gpu, 100 - gpu, context, "Forever")


def test_missing_chat_model_fails_but_missing_csv_model_only_warns() -> None:
    checks = check_models_pulled("gemma3:4b", "qwen2.5-coder:3b", available=[])
    assert [c.status for c in checks] == ["FAIL", "WARN"]
    assert "ollama pull gemma3:4b" in checks[0].detail


def test_residency_passes_at_100_percent_gpu() -> None:
    assert check_residency("gemma3:4b", [_model()], expected_context=8192).status == "PASS"


def test_residency_fails_when_the_model_is_split_across_cpu() -> None:
    check = check_residency("gemma3:4b", [_model(gpu=52)], expected_context=8192)
    assert check.status == "FAIL"
    assert "48% CPU" in check.detail


def test_residency_fails_when_not_loaded() -> None:
    assert check_residency("gemma3:4b", [], expected_context=8192).status == "FAIL"


def test_residency_warns_on_a_context_mismatch() -> None:
    # The trap from lesson 03 §4: loaded with another context than queries use -> reload.
    check = check_residency("gemma3:4b", [_model(context=131072)], expected_context=8192)
    assert check.status == "WARN"
    assert "reload" in check.detail


def test_residency_ignores_the_context_when_the_column_is_missing() -> None:
    check = check_residency("gemma3:4b", [_model(context=None)], expected_context=8192)
    assert check.status == "PASS"


def test_a_busy_gpu_warns() -> None:
    busy = GpuMemory(used_mib=BUSY_GPU_USED_MIB + 1, total_mib=10240)
    assert check_gpu_memory(busy, label="GPU").status == "WARN"
    assert check_gpu_memory(IDLE_GPU, label="GPU").status == "PASS"


def test_memory_in_use_after_loading_the_model_does_not_warn() -> None:
    # Found in the first live run (2026-10-15): 5,315 MiB used after warm-up is the model itself.
    loaded = GpuMemory(used_mib=5315, total_mib=10240)
    check = check_gpu_memory(loaded, label="after", model_loaded=True, before=IDLE_GPU)
    assert check.status == "PASS"
    assert "+4661 MiB" in check.detail


def test_warmup_report_shows_the_memory_change_without_warning() -> None:
    readings = iter([GpuMemory(1202, 10240), GpuMemory(5315, 10240)])
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: next(readings),
        read_ps=lambda: [_model()],
    )
    after = _by_name(checks, "GPU memory after")
    assert after.status == "PASS"
    assert "+4113 MiB" in after.detail


def test_a_model_loaded_by_an_earlier_run_does_not_blame_other_programs() -> None:
    # Second live run (2026-10-15): gemma3:4b was still loaded (keep_alive), so "before" was
    # already ~5,300 MiB and the doctor wrongly warned about other programs.
    busy = GpuMemory(used_mib=5294, total_mib=10240)
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: busy,
        read_ps=lambda: [_model()],
    )
    before = _by_name(checks, "GPU memory before")
    assert before.status == "PASS"
    assert "already loaded" in before.detail


def test_a_busy_gpu_with_nothing_of_ours_loaded_still_warns_in_the_report() -> None:
    busy = GpuMemory(used_mib=5294, total_mib=10240)
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: busy,
        read_ps=list,
    )
    assert _by_name(checks, "GPU memory before").status == "WARN"


class FakeClient:
    """A stand-in Ollama. ``loaded`` is what it holds in memory; a warm-up adds to it and an
    unload removes from it at once (the swap's waiting is tested in test_swap.py)."""

    def __init__(
        self,
        *,
        pulled: list[str],
        fail: bool = False,
        loaded: list[str] | None = None,
        warm_up_error: OllamaError | None = None,
    ) -> None:
        self.pulled, self.fail = pulled, fail
        self.loaded = list(loaded or [])
        self.warm_up_error = warm_up_error
        self.warmed: list[tuple[str, int, str]] = []
        self.unloaded: list[str] = []

    def version(self) -> str:
        if self.fail:
            raise OllamaError("connection refused")
        return "0.34.2"

    def list_models(self) -> list[str]:
        return self.pulled

    def loaded_models(self) -> list[str]:
        return list(self.loaded)

    def unload(self, model: str) -> None:
        self.unloaded.append(model)
        self.loaded.remove(model)

    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:
        if self.warm_up_error is not None:
            raise self.warm_up_error
        self.warmed.append((model, num_ctx, keep_alive))
        self.loaded.append(model)
        return 1.5


def _settings() -> Settings:
    return Settings(_env_file=None)


def _by_name(checks: list[Check], prefix: str) -> Check:
    return next(c for c in checks if c.name.startswith(prefix))


def test_unreachable_ollama_is_a_single_fail() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=[], fail=True),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=list,
    )
    assert [c.status for c in checks] == ["FAIL"]
    assert has_failure(checks)


def test_no_warmup_loads_nothing_and_skips_residency_when_not_loaded() -> None:
    client = FakeClient(pulled=PULLED)
    checks = run_doctor(_settings(), client, warm_up=False, read_gpu=lambda: IDLE_GPU, read_ps=list)
    assert client.warmed == []
    assert _by_name(checks, "residency").status == "SKIP"


def test_no_warmup_still_checks_a_model_that_is_already_loaded() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=False,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model()],
    )
    assert _by_name(checks, "residency").status == "PASS"


def test_warmup_uses_the_configured_context_and_keep_alive() -> None:
    client = FakeClient(pulled=PULLED)
    checks = run_doctor(
        _settings(), client, warm_up=True, read_gpu=lambda: IDLE_GPU, read_ps=lambda: [_model()]
    )
    assert client.warmed == [("gemma3:4b", 8192, "30m")]
    assert _by_name(checks, "warm-up").status == "PASS"
    assert _by_name(checks, "GPU memory after").status == "PASS"


def test_unreadable_gpu_or_ps_warns_instead_of_crashing() -> None:
    def broken_gpu() -> GpuMemory:
        raise CommandError("nvidia-smi not found")

    def broken_ps() -> list[LoadedModel]:
        raise CommandError("ollama not found")

    checks = run_doctor(
        _settings(), FakeClient(pulled=PULLED), warm_up=True, read_gpu=broken_gpu, read_ps=broken_ps
    )
    assert _by_name(checks, "GPU memory before").status == "WARN"
    assert _by_name(checks, "residency").status == "WARN"
    assert not has_failure(checks)


def test_no_check_is_a_placeholder_for_a_step_that_is_built() -> None:
    # S1.4 (the storage lock) and S1.5 (the embedder) used to be "not built yet" lines here.
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=False,
        read_gpu=lambda: IDLE_GPU,
        read_ps=list,
    )
    assert not any("not built yet" in c.detail for c in checks)
    assert _by_name(checks, "storage lock free").status == "PASS"
    assert _by_name(checks, "embedder model pulled").status == "PASS"


# --- steps S1.4 and S1.5: the embedder and the storage lock ------------------------------------


def _embedder_entry(gpu: int = 0, name: str = "nomic-embed-text:latest") -> LoadedModel:
    return LoadedModel(name, gpu, 100 - gpu, 2048, "29 minutes from now")


def test_the_embedder_must_be_pulled_in_chat_mode_and_only_warns_in_csv_mode() -> None:
    assert check_embedder_pulled(EMBED, PULLED).status == "PASS"  # "name" is "name:latest"
    missing = check_embedder_pulled(EMBED, ["gemma3:4b"])
    assert missing.status == "FAIL"
    assert "ollama pull nomic-embed-text" in missing.detail
    assert check_embedder_pulled(EMBED, ["gemma3:4b"], mode="csv").status == "WARN"


def test_the_embedder_passes_when_it_is_listed_as_100_percent_cpu() -> None:
    check = check_embedder_on_cpu(EMBED, [_model(), _embedder_entry()])
    assert (check.status, check.name) == ("PASS", "embedder on CPU")
    assert "100% CPU" in check.detail


def test_an_embedder_on_the_gpu_fails_because_it_takes_the_chat_models_memory() -> None:
    for gpu in (100, 40):
        check = check_embedder_on_cpu(EMBED, [_embedder_entry(gpu=gpu)])
        assert check.status == "FAIL"
        assert f"{gpu}% on the GPU" in check.detail


def test_an_embedder_that_ollama_does_not_list_is_a_warning_not_a_pass() -> None:
    assert check_embedder_on_cpu(EMBED, [_model()]).status == "WARN"


def test_the_storage_lock_is_free_when_nobody_holds_it(tmp_path: Path) -> None:
    assert check_storage_lock(tmp_path).status == "PASS"
    assert check_storage_lock(tmp_path).status == "PASS"  # the check let it go again
    with StorageLock(tmp_path):
        pass  # and it can still be taken


def test_a_missing_data_folder_is_free_and_is_not_created(tmp_path: Path) -> None:
    folder = tmp_path / "not-yet"
    check = check_storage_lock(folder)
    assert check.status == "PASS"
    assert "no data folder yet" in check.detail
    assert not folder.exists()


def test_a_held_storage_lock_is_a_warning_naming_the_likely_owner(tmp_path: Path) -> None:
    with StorageLock(tmp_path):
        check = check_storage_lock(tmp_path)
    assert check.status == "WARN"
    assert "another Geekbot process" in check.detail


class _Probe:
    """An embedder probe that records that it was called, and can fail."""

    def __init__(self, error: OllamaError | None = None) -> None:
        self.error, self.calls = error, 0

    def __call__(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def test_the_doctor_probes_the_embedder_and_reads_where_it_landed() -> None:
    probe = _Probe()
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model(), _embedder_entry()],
        probe_embedder=probe,
    )
    assert probe.calls == 1
    assert _by_name(checks, "embedder on CPU").status == "PASS"
    assert not has_failure(checks)


def test_the_doctor_fails_an_embedder_that_ended_up_on_the_gpu() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model(), _embedder_entry(gpu=100)],
        probe_embedder=_Probe(),
    )
    assert _by_name(checks, "embedder on CPU").status == "FAIL"
    assert has_failure(checks)


def test_a_failed_probe_is_a_failed_check_with_the_reason() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model()],
        probe_embedder=_Probe(OllamaError("model requires more memory")),
    )
    check = _by_name(checks, "embedder on CPU")
    assert check.status == "FAIL"
    assert "model requires more memory" in check.detail


def test_no_warmup_loads_nothing_so_the_embedder_is_not_probed() -> None:
    probe = _Probe()
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=False,
        read_gpu=lambda: IDLE_GPU,
        read_ps=list,
        probe_embedder=probe,
    )
    assert probe.calls == 0
    check = _by_name(checks, "embedder on CPU")
    assert check.status == "SKIP"
    assert "--no-warmup" in check.detail


def test_csv_mode_does_not_probe_the_embedder() -> None:
    probe = _Probe()
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model(name=CSV)],
        mode="csv",
        probe_embedder=probe,
    )
    assert probe.calls == 0
    assert _by_name(checks, "embedder on CPU").status == "SKIP"


def test_an_embedder_that_is_not_pulled_is_not_probed_and_the_pull_check_fails() -> None:
    probe = _Probe()
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=["gemma3:4b", "qwen2.5-coder:3b"]),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model()],
        probe_embedder=probe,
    )
    assert probe.calls == 0
    assert _by_name(checks, "embedder model pulled").status == "FAIL"
    assert _by_name(checks, "embedder on CPU").status == "SKIP"


def test_an_unreadable_ps_after_the_probe_is_a_warning() -> None:
    calls = {"n": 0}

    def ps() -> list[LoadedModel]:
        calls["n"] += 1
        if calls["n"] > 2:  # the doctor reads it twice for the chat model, then for the embedder
            raise CommandError("ollama not found")
        return [_model()]

    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=ps,
        probe_embedder=_Probe(),
    )
    assert _by_name(checks, "embedder on CPU").status == "WARN"


def test_without_a_probe_the_embedder_is_skipped_not_passed() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model()],
    )
    assert _by_name(checks, "embedder on CPU").status == "SKIP"


# --- step C3: the doctor checks the active mode's model, reached by an explicit swap ----------


def test_the_chat_report_points_to_the_csv_check() -> None:
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=False,
        read_gpu=lambda: IDLE_GPU,
        read_ps=list,
    )
    pointer = _by_name(checks, "CSV model residency")
    assert pointer.status == "SKIP"
    assert "--mode csv" in pointer.detail


def test_csv_mode_swaps_to_the_csv_model_and_checks_it_at_the_csv_context() -> None:
    settings = _settings()
    client = FakeClient(pulled=PULLED, loaded=[CHAT])
    checks = run_doctor(
        settings,
        client,
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: [_model(name=CSV, context=settings.csv_num_ctx)],
        mode="csv",
    )
    assert client.unloaded == [CHAT]  # R-4: the chat model goes first
    assert client.warmed == [(CSV, settings.csv_num_ctx, settings.keep_alive)]
    warm_up = _by_name(checks, "warm-up")
    assert warm_up.status == "PASS"
    assert f"unloaded {CHAT}" in warm_up.detail
    assert _by_name(checks, "residency").status == "PASS"
    assert _by_name(checks, "no co-residency").status == "PASS"
    assert "checking csv" in _by_name(checks, "profile").detail
    assert not any("CSV model residency" in c.name for c in checks)  # the real check replaced it
    assert not has_failure(checks)


def test_a_chat_warm_up_also_unloads_a_csv_model_that_is_still_loaded() -> None:
    client = FakeClient(pulled=PULLED, loaded=[CSV])
    checks = run_doctor(
        _settings(), client, warm_up=True, read_gpu=lambda: IDLE_GPU, read_ps=lambda: [_model()]
    )
    assert client.unloaded == [CSV]
    assert _by_name(checks, "warm-up").status == "PASS"


def test_both_models_loaded_at_once_fails_the_co_residency_check() -> None:
    both = [_model(), _model(name=CSV, context=16384)]
    checks = run_doctor(
        _settings(),
        FakeClient(pulled=PULLED),
        warm_up=True,
        read_gpu=lambda: IDLE_GPU,
        read_ps=lambda: both,
    )
    exclusive = _by_name(checks, "no co-residency")
    assert exclusive.status == "FAIL"
    assert CSV in exclusive.detail
    assert has_failure(checks)


def test_a_failed_swap_is_a_failed_warm_up_and_stops_the_report() -> None:
    client = FakeClient(pulled=PULLED, warm_up_error=OllamaError("model not found"))
    checks = run_doctor(
        _settings(), client, warm_up=True, read_gpu=lambda: IDLE_GPU, read_ps=list, mode="csv"
    )
    assert checks[-1].name == "warm-up"
    assert checks[-1].status == "FAIL"
    assert "model not found" in checks[-1].detail


def test_the_active_modes_model_missing_fails_and_the_other_only_warns() -> None:
    only_chat = check_models_pulled(CHAT, CSV, available=[CHAT], mode="csv")
    assert [c.status for c in only_chat] == ["PASS", "FAIL"]
    only_csv = check_models_pulled(CHAT, CSV, available=[CSV], mode="csv")
    assert [c.status for c in only_csv] == ["WARN", "PASS"]


def test_exclusive_check_ignores_the_embedder_and_passes_when_the_other_model_is_gone() -> None:
    embedder = LoadedModel("nomic-embed-text:latest", 0, 100, 2048, "Forever")
    assert check_exclusive(CSV, CHAT, [_model(name=CSV), embedder]).status == "PASS"


def test_exclusive_check_fails_when_the_other_model_is_loaded() -> None:
    assert check_exclusive(CSV, CHAT, [_model(name=CSV), _model()]).status == "FAIL"


def test_exclusive_check_passes_when_both_modes_use_one_model() -> None:
    assert check_exclusive(CHAT, CHAT, [_model()]).status == "PASS"


def test_describe_swap_says_what_was_unloaded_only_when_something_was() -> None:
    plain = SwapResult(loaded=CSV, unloaded=(), unload_s=0.0, warm_up_s=3.14)
    assert describe_swap(plain) == f"{CSV} loaded in 3.1 s"
    swapped = SwapResult(loaded=CSV, unloaded=(CHAT,), unload_s=1.25, warm_up_s=3.14)
    assert describe_swap(swapped) == f"{CSV} loaded in 3.1 s; unloaded {CHAT} first (1.2 s)"


@pytest.mark.parametrize(
    ("statuses", "failed"),
    [(["PASS", "WARN", "SKIP"], False), (["PASS", "FAIL"], True)],
)
def test_only_fail_makes_the_doctor_fail(statuses: list[Status], failed: bool) -> None:
    checks = [Check(f"c{i}", s, "") for i, s in enumerate(statuses)]
    assert has_failure(checks) is failed


def test_format_shows_status_name_and_detail() -> None:
    text = format_checks([Check("ollama", "FAIL", "down"), Check("gpu", "PASS", "ok")])
    assert text.splitlines()[0].startswith("[FAIL] ollama")
    assert "down" in text


def test_the_command_probes_with_the_real_embedder_settings_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeEmbedder:
        def __init__(self, host: str, model: str, *, num_gpu: int, keep_alive: str) -> None:
            seen.update(host=host, model=model, num_gpu=num_gpu, keep_alive=keep_alive)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            seen["closed"] = True

        def embed_query(self, text: str) -> list[float]:
            seen["text"] = text
            return [0.1]

    monkeypatch.setattr("geekbot.index.embedder.OllamaEmbeddingsClient", FakeEmbedder)
    from geekbot.cli import _probe_embedder

    _probe_embedder(_settings())
    assert seen == {
        "host": "http://127.0.0.1:11434",
        "model": "nomic-embed-text",
        "num_gpu": 0,
        "keep_alive": "30m",
        "text": "doctor probe",
        "closed": True,
    }
