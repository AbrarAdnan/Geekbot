"""Live test for step S0.3: warm up the real chat model and check R-1 (needs Ollama + the GPU).

Run on purpose with ``pytest -m live`` when the GPU is free. It loads the active profile's
chat model, so it is deselected by default.
"""

import pytest

from geekbot.config import Settings
from geekbot.doctor import check_residency
from geekbot.llm.models import OllamaClient
from geekbot.llm.vram import OLLAMA_PS_COMMAND, parse_ollama_ps, run_command


@pytest.mark.live
def test_warmed_up_chat_model_is_fully_on_the_gpu() -> None:
    settings = Settings()
    model = settings.resolved_chat_model
    with OllamaClient(settings.ollama_host) as client:
        client.warm_up(model, num_ctx=settings.chat_num_ctx, keep_alive=settings.keep_alive)
    loaded = parse_ollama_ps(run_command(OLLAMA_PS_COMMAND))
    check = check_residency(model, loaded, expected_context=settings.chat_num_ctx)
    assert check.status == "PASS", check.detail
