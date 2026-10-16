"""Tests for step S0.3: the nvidia-smi parser (the ollama-ps parser is the ✍️ piece)."""

import pytest

from geekbot.llm.vram import GpuMemory, parse_nvidia_smi_memory


def test_parses_the_real_line() -> None:
    # Real output on this PC, 2026-10-15.
    gpu = parse_nvidia_smi_memory("654, 10240\n")
    assert gpu == GpuMemory(used_mib=654, total_mib=10240)
    assert gpu.free_mib == 9586


def test_with_two_gpus_the_first_is_used() -> None:
    assert parse_nvidia_smi_memory("100, 8000\n200, 24000\n").total_mib == 8000


@pytest.mark.parametrize("text", ["", "\n", "N/A", "1, 2, 3", "abc, def"])
def test_bad_output_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError, match="nvidia-smi"):
        parse_nvidia_smi_memory(text)
