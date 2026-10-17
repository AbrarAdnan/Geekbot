"""The finished answer as text for a terminal (steps S3.1 and S3.2).

Pure functions of the graph's final state, so the layout is testable without a terminal, a model or
Ollama. The answer comes first and alone on its lines, then the sources: the ones the answer cites
(its ``[n]`` marks, checked by ``agent.citations``), then the ones the model read and did not cite.
After those come the labels of the answer (PLAN.md §8.2) and the readings of the run. Everything after
the answer is diagnostic: it says where the time went and whether the model had to be loaded, which is
how the rules of PLAN.md §6.5 (R-1, R-2) are watched while building.
"""

from collections.abc import Mapping
from typing import Any

from geekbot.agent.packing import PackedPrompt, source_label
from geekbot.agent.state import SourceRef
from geekbot.domain import Label, is_tainted

__all__ = ["format_answer", "format_prompt"]


def format_prompt(packed: PackedPrompt) -> str:
    """The exact messages the model receives, labelled by role (``ask --show-prompt``)."""
    parts = [f"--- {message.type} ---\n{message.content}" for message in packed.messages]
    return "\n\n".join(parts)


def _source_line(source: SourceRef) -> str:
    return f"[{source['n']}] {source_label(source)}  ({source['path']})"


def _timing_line(label: str, readings: Mapping[str, float], *, order: tuple[str, ...]) -> str:
    shown = [f"{name} {readings[name]:.1f} ms" for name in order if name in readings]
    extra = [f"{name} {value:.1f} ms" for name, value in readings.items() if name not in order]
    return f"{label}: " + " · ".join([*shown, *extra])


def _source_lines(sources: list[SourceRef], cited: list[int]) -> list[str]:
    """The source listing: cited ones first, then the ones the model read and did not cite."""
    used = [source for source in sources if source["n"] in cited]
    unused = [source for source in sources if source["n"] not in cited]
    lines: list[str] = []
    if used:
        lines.append("Sources cited:")
        lines.extend(f"  {_source_line(source)}" for source in used)
    if unused:
        lines.append("Read, not cited:" if used else "Read, but the answer cites none:")
        lines.extend(f"  {_source_line(source)}" for source in unused)
    return lines


def format_answer(final: Mapping[str, Any], *, model: str) -> str:
    """The answer, its sources, and the readings of the run.

    ``final`` is the graph's final state. Only the first ``used`` sources are listed: those are the
    ones in the prompt, so they are the only ones a ``[n]`` in the answer can mean. A citation the
    check removed is named, so a user who sees a gap knows why.
    """
    lines = [str(final["answer"]), ""]
    used = int(final.get("used", 0))
    sources: list[SourceRef] = list(final.get("sources", []))[:used]
    cited = [int(n) for n in final.get("cited", [])]
    if sources:
        lines.extend(_source_lines(sources, cited))
        lines.append("")
    dropped = [int(n) for n in final.get("dropped_citations", [])]
    if dropped:
        marks = ", ".join(f"[{n}]" for n in dropped)
        lines.append(f"removed from the answer (no such source in the prompt): {marks}")
    names = list(final.get("answer_labels", []))
    if names:
        verdict = "tainted" if is_tainted(Label(name) for name in names) else "not tainted"
        lines.append(f"labels: {', '.join(names)} ({verdict})")
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
