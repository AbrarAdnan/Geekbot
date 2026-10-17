"""The finished answer as text for a terminal (steps S3.1 to S3.3).

Pure functions of the graph's final state, so the layout is testable without a terminal, a model or
Ollama. The answer comes first and alone on its lines, then the sources: the ones the answer cites
(its ``[n]`` marks, checked by ``agent.citations``), then the ones the model read and did not cite.
After those come the labels of the answer (PLAN.md §8.2) and the readings of the run. Everything after
the answer is diagnostic: it says where the time went and whether the model had to be loaded, which is
how the rules of PLAN.md §6.5 (R-1, R-2) are watched while building.

S3.3 adds the streaming side: ``consume_stream`` turns the graph's stream events into writes (answer
pieces to one place, notes to another) and ``format_after_stream`` prints what follows once the
answer has already been shown piece by piece.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from geekbot.agent.packing import PackedPrompt, source_label
from geekbot.agent.state import SourceRef
from geekbot.domain import Label, is_tainted

__all__ = [
    "StreamedAnswer",
    "consume_stream",
    "format_after_stream",
    "format_answer",
    "format_prompt",
]


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


def _generation_line(model: str, generation: Mapping[str, float]) -> str:
    """The readings of the model call: whole time, time to first token, load, prompt, output."""
    facts = [f"{generation['seconds']:.2f} s"]
    if "first_token_seconds" in generation:
        facts.append(f"first token {generation['first_token_seconds']:.2f} s")
    if "load_seconds" in generation:
        facts.append(f"model load {generation['load_seconds']:.2f} s")
    if "prompt_tokens" in generation:
        prompt = f"prompt {generation['prompt_tokens']:.0f} tokens"
        if "prompt_eval_seconds" in generation:
            prompt += f" (read in {generation['prompt_eval_seconds']:.2f} s)"
        facts.append(prompt)
    if "output_tokens" in generation:
        facts.append(f"{generation['output_tokens']:.0f} tokens out")
    if "tokens_per_second" in generation:
        facts.append(f"{generation['tokens_per_second']:.1f} tokens/s")
    if "thinking_chars" in generation:
        facts.append(f"thinking {generation['thinking_chars']:.0f} characters")
    return f"{model}: " + ", ".join(facts)


def format_answer(
    final: Mapping[str, Any],
    *,
    model: str,
    include_answer: bool = True,
    show_thinking: bool = False,
) -> str:
    """The answer, its sources, and the readings of the run.

    ``final`` is the graph's final state. Only the first ``used`` sources are listed: those are the
    ones in the prompt, so they are the only ones a ``[n]`` in the answer can mean. A citation the
    check removed is named, so a user who sees a gap knows why. ``include_answer=False`` leaves the
    answer out (it was already shown while streaming). With ``show_thinking`` the model's reasoning,
    if it produced any, is printed under the answer; the readings count its characters either way.
    """
    lines: list[str] = []
    if include_answer:
        lines.extend([str(final["answer"]), ""])
    thinking = str(final.get("thinking", ""))
    if show_thinking and thinking:
        indented = "\n".join(f"  {line}" for line in thinking.splitlines())
        lines.extend([f"thinking ({len(thinking)} characters):", indented, ""])
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
        lines.append(_generation_line(model, generation))
    return "\n".join(lines).strip()


# --- streaming (step S3.3) -------------------------------------------------------------------


@dataclass(frozen=True)
class StreamedAnswer:
    """What a streamed run left: the final state and the answer text that was written live.

    ``streamed`` is empty when the model was never called (nothing was found, or nothing fit the
    budget): the application's own sentence is then not shown live and has to be printed after.
    """

    final: dict[str, Any]
    streamed: str


def consume_stream(
    events: Iterable[Any],
    *,
    write: Callable[[str], None],
    note: Callable[[str], None],
    show_thinking: bool = False,
) -> StreamedAnswer:
    """Turn ``graph.stream(..., stream_mode=["custom", "values"])`` into live writes.

    ``events`` are ``(mode, payload)`` pairs (LangGraph types its stream as ``Any``, so this does
    too, and unpacks each pair itself). A ``custom`` payload is a ``StreamEvent``: answer
    pieces go to ``write`` as they arrive (the very first one without its leading whitespace, so the
    answer does not start with a blank line); reasoning pieces go to ``note`` when ``show_thinking``
    is on and are otherwise reduced to one "thinking..." note, so the user knows the model is
    working and not stuck. A ``values`` payload is the whole state after a step; the last one is
    the final state. The pieces are the model's untrusted text: this function only passes them on
    as text, and the caller prints them without interpreting them.
    """
    final: dict[str, Any] | None = None
    shown: list[str] = []
    thinking_started = False
    thinking_closed = False
    for mode, payload in events:  # a pair per event; a mismatch fails loudly, here
        if mode == "values":
            final = dict(payload)
            continue
        if mode != "custom" or not isinstance(payload, Mapping):
            continue
        piece = str(payload.get("text", ""))
        kind = payload.get("event")
        if kind == "thinking":
            if not thinking_started:
                thinking_started = True
                note("thinking: " if show_thinking else "thinking...\n")
            if show_thinking:
                note(piece)
        elif kind == "text":
            if thinking_started and show_thinking and not thinking_closed:
                thinking_closed = True
                note("\n")
            if not shown:
                piece = piece.lstrip()
            if piece:
                shown.append(piece)
                write(piece)
    if final is None:
        raise ValueError("the graph ended without a final state")
    return StreamedAnswer(final=final, streamed="".join(shown))


def format_after_stream(streamed: StreamedAnswer, *, model: str) -> str:
    """What to print once the stream is over (the cursor is at the end of the streamed answer).

    - Nothing was streamed: the whole answer, as ``format_answer`` prints it.
    - Something was streamed and the citation check left it alone: the sources, labels and readings.
    - Something was streamed and the check removed a mark: the corrected answer first, because the
      live text could not be taken back and still holds the mark that named no source.
    """
    final = streamed.final
    if not streamed.streamed.strip():
        return format_answer(final, model=model)
    lines: list[str] = []
    answer = str(final["answer"])
    if answer != str(final.get("raw_answer", answer)):
        lines.extend(
            [
                "corrected answer (the citation check removed a mark that names no source):",
                answer,
                "",
            ]
        )
    lines.append(format_answer(final, model=model, include_answer=False))
    return "\n".join(lines).strip()
