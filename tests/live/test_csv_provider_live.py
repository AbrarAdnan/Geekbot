"""Live test for step C2: the real CSV provider against a real Ollama (needs Ollama + the model).

Run on purpose with ``pytest -m live`` when the GPU is free. It loads the active profile's CSV
model (``qwen2.5-coder:3b`` on the default ``proto`` profile) and checks only what the provider
promises: Ollama's JSON mode returns a JSON object, and a JSON Schema fixes its shape. Whether the
model answers CSV questions *well* is not tested here; that is C3 and C6, on the target model.
"""

import json
from collections.abc import Iterator

import pytest

from geekbot.config import Settings
from geekbot.csv_mode.provider import OllamaCsvProvider
from geekbot.llm.models import OllamaClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def provider() -> Iterator[OllamaCsvProvider]:
    settings = Settings()
    model = settings.resolved_csv_model
    # Load the model first with the generous warm-up timeout, so the calls under test measure the
    # provider and not a cold start.
    with OllamaClient(settings.ollama_host) as client:
        client.warm_up(model, num_ctx=settings.csv_num_ctx, keep_alive=settings.keep_alive)
    with OllamaCsvProvider(
        host=settings.ollama_host,
        model=model,
        num_ctx=settings.csv_num_ctx,
        keep_alive=settings.keep_alive,
        timeout_s=float(settings.csv_llm_timeout_s),
    ) as live_provider:
        yield live_provider


def test_json_mode_returns_a_json_object(provider: OllamaCsvProvider) -> None:
    reply = provider.generate_json(
        'Reply with a JSON object that has one key, "ok", set to true.',
        temperature=0.0,
        seed=1,
    )
    assert isinstance(json.loads(reply), dict), f"{provider.model} returned: {reply!r}"


def test_a_schema_fixes_the_shape_of_the_reply(provider: OllamaCsvProvider) -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    reply = provider.generate_json(
        "Is the sky blue? Answer in the requested JSON shape.",
        temperature=0.0,
        seed=1,
        schema=schema,
    )
    parsed = json.loads(reply)
    assert isinstance(parsed.get("ok"), bool), f"{provider.model} returned: {reply!r}"
