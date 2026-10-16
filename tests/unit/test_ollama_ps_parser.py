"""Tests for the ✍️ piece of step S0.3: ``parse_ollama_ps``.

They fail with ``NotImplementedError`` until you write the parser in
``src/geekbot/llm/vram.py``. They are the spec.
"""

from pathlib import Path

from geekbot.llm.vram import LoadedModel, parse_ollama_ps

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_the_real_output_from_this_pc() -> None:
    assert parse_ollama_ps(_read("ollama_ps_one_model.txt")) == [
        LoadedModel(
            name="gemma3:4b",
            gpu_percent=100,
            cpu_percent=0,
            context=131072,
            until="4 minutes from now",
        )
    ]


def test_nothing_loaded_gives_an_empty_list() -> None:
    assert parse_ollama_ps(_read("ollama_ps_empty.txt")) == []
    assert parse_ollama_ps("") == []


def test_split_between_cpu_and_gpu_follows_the_label_order() -> None:
    (model,) = parse_ollama_ps(_read("ollama_ps_split.txt"))
    assert (model.cpu_percent, model.gpu_percent) == (48, 52)
    assert model.until == "Forever"


def test_two_models_keep_their_order_and_processors() -> None:
    chat, embedder = parse_ollama_ps(_read("ollama_ps_two_models.txt"))
    assert (chat.name, chat.gpu_percent, chat.cpu_percent) == ("gemma4:12b", 100, 0)
    assert (embedder.name, embedder.gpu_percent, embedder.cpu_percent) == (
        "nomic-embed-text",
        0,
        100,
    )


def test_a_version_without_a_context_column() -> None:
    (model,) = parse_ollama_ps(_read("ollama_ps_no_context.txt"))
    assert model.context is None
    assert model.until == "4 minutes from now"
