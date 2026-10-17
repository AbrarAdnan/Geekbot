"""The finished answer as text for a terminal (step S3.1).

Pure functions of the graph's final state, so the layout is testable without a terminal, a model or
Ollama. The answer comes first and alone on its lines, then the sources the model was shown (the
numbers match its ``[n]`` marks), then the readings of the run. Everything after the answer is
diagnostic: it says where the time went and whether the model had to be loaded, which is how the
rules of PLAN.md §6.5 (R-1, R-2) are watched while building.
"""

from collections.abc import Mapping
from typing import Any

from geekbot.agent.packing import PackedPrompt

__all__ = ["format_answer", "format_prompt"]


def format_prompt(packed: PackedPrompt) -> str:
    """The exact messages the model receives, labelled by role (``ask --show-prompt``)."""
    parts = [f"--- {message.type} ---\n{message.content}" for message in packed.messages]
    return "\n\n".join(parts)


def _source_line(source: Mapping[str, Any]) -> str:
    where = [str(source["title"])]
    if source["section"]:
        where.append(str(source["section"]))
    if source["page"] is not None:
        where.append(f"page {source['page']}")
    return f"[{source['n']}] {' | '.join(where)}  ({source['path']})"


def _timing_line(label: str, readings: Mapping[str, float], *, order: tuple[str, ...]) -> str:
    shown = [f"{name} {readings[name]:.1f} ms" for name in order if name in readings]
    extra = [f"{name} {value:.1f} ms" for name, value in readings.items() if name not in order]
    return f"{label}: " + " · ".join([*shown, *extra])


def format_answer(final: Mapping[str, Any], *, model: str) -> str:
    """The answer, the sources it could cite, and the readings of the run.

    ``final`` is the graph's final state. Only the first ``used`` sources are listed: those are the
    ones in the prompt, so they are the only ones a ``[n]`` in the answer can mean.
    """
    lines = [str(final["answer"]), ""]
    used = int(final.get("used", 0))
    sources = list(final.get("sources", []))[:used]
    if sources:
        lines.append("Sources:")
        lines.extend(f"  {_source_line(source)}" for source in sources)
        lines.append("")
    retrieval = final.get("retrieval_ms", {})
    if retrieval:
        lines.append(_timing_line("retrieval", retrieval, order=("total",)))
    generation = final.get("generation", {})
    if generation:
        facts = [f"{generation['seconds']:.2f} s"]
        if "load_seconds" in generation:
            facts.append(f"model load {generation['load_seconds']:.2f} s")
        if "output_tokens" in generation:
            facts.append(f"{generation['output_tokens']:.0f} tokens out")
        if "tokens_per_second" in generation:
            facts.append(f"{generation['tokens_per_second']:.1f} tokens/s")
        lines.append(f"{model}: " + ", ".join(facts))
    return "\n".join(lines).rstrip()
