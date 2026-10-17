"""Live test for step C3: swap between the real chat and CSV models (needs Ollama + the GPU).

Run on purpose with ``pytest -m live`` when the GPU is free. It loads and unloads both models of
the active profile, so it is deselected by default. After each swap the new model must be 100%
on the GPU at its own context size and the old one must be gone (R-1, R-4).
"""

import pytest

from geekbot.config import Settings
from geekbot.doctor import check_exclusive, check_residency
from geekbot.llm.models import OllamaClient, swap_to_mode
from geekbot.llm.vram import OLLAMA_PS_COMMAND, parse_ollama_ps, run_command


@pytest.mark.live
def test_swapping_between_modes_leaves_only_the_new_model_fully_on_the_gpu() -> None:
    settings = Settings()
    chat, csv = settings.resolved_chat_model, settings.resolved_csv_model
    # mode -> (its model, the other mode's model, the context its queries use)
    models = {
        "chat": (chat, csv, settings.chat_num_ctx),
        "csv": (csv, chat, settings.csv_num_ctx),
    }
    with OllamaClient(settings.ollama_host) as client:
        for mode in ("chat", "csv", "chat"):
            active, other, num_ctx = models[mode]
            swap_to_mode(client, settings, mode)  # type: ignore[arg-type]
            loaded = parse_ollama_ps(run_command(OLLAMA_PS_COMMAND))
            residency = check_residency(active, loaded, expected_context=num_ctx)
            assert residency.status == "PASS", f"after the swap to {mode}: {residency.detail}"
            exclusive = check_exclusive(active, other, loaded)
            assert exclusive.status == "PASS", f"after the swap to {mode}: {exclusive.detail}"
