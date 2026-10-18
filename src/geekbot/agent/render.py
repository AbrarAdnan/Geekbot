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
    "StreamCollector",
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
    thread: str | None = None,
) -> str:
    """The answer, its sources, and the readings of the run.

    ``final`` is the graph's final state. Only the first ``used`` sources are listed: those are the
    ones in the prompt, so they are the only ones a ``[n]`` in the answer can mean. A citation the
    check removed is named, so a user who sees a gap knows why. ``include_answer=False`` leaves the
    answer out (it was already shown while streaming). With ``show_thinking`` the model's reasoning,
    if it produced any, is printed under the answer; the readings count its characters either way.
    On a follow-up the line ``searched as:`` shows what retrieval was asked for (the rewrite), and a
    rewrite that failed says why. With ``thread`` (the name of a saved conversation) a last line says
    which turn this was and what the thread's taint is (PLAN.md §8.2).
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
    lines.extend(_rewrite_lines(final))
    retrieval = final.get("retrieval_ms", {})
    if retrieval:
        lines.append(_timing_line("retrieval", retrieval, order=("total",)))
    generation = final.get("generation", {})
    if generation:
        lines.append(_generation_line(model, generation))
    if thread is not None:
        lines.append(_thread_line(thread, final))
    return "\n".join(lines).strip()


def _rewrite_lines(final: Mapping[str, Any]) -> list[str]:
    """What retrieval searched for when it was not the user's own words, and a failed rewrite."""
    lines: list[str] = []
    verb = (
        "asked" if final.get("mode") == "csv" else "searched"
    )  # a table is asked, documents searched
    searched = str(final.get("standalone_question", ""))
    if searched and searched != str(final.get("question", searched)):
        lines.append(f'{verb} as: "{searched}"')
    error = str(final.get("condense_error", ""))
    if error:
        lines.append(f"rewrite failed, {verb} with your own words: {error}")
    seconds = float(final.get("condense_seconds", 0.0))
    if seconds > 0:
        lines.append(f"rewrite: {seconds:.2f} s")
    status = str(final.get("csv_status", ""))
    if status:
        lines.append(f"csv status: {status}")
    return lines


def _thread_line(thread: str, final: Mapping[str, Any]) -> str:
    """``thread t1: turn 2, 4 messages remembered, taint KB, USER (tainted)`` (``(csv)`` in CSV mode)."""
    taint = list(final.get("thread_taint", []))
    verdict = "tainted" if is_tainted(Label(name) for name in taint) else "not tainted"
    kept = len(final.get("history", []))
    kind = " (csv)" if final.get("mode") == "csv" else ""
    return (
        f"thread {thread}{kind}: turn {final.get('turn', 0)}, {kept} messages remembered, "
        f"taint {', '.join(taint) or 'none'} ({verdict})"
    )


# --- streaming (step S3.3) -------------------------------------------------------------------


@dataclass(frozen=True)
class StreamedAnswer:
    """What a streamed run left: the final state and the answer text that was written live.

    ``streamed`` is empty when the model was never called (nothing was found, or nothing fit the
    budget): the application's own sentence is then not shown live and has to be printed after.
    """

    final: dict[str, Any]
    streamed: str


class StreamCollector:
    """Turns the events of a streamed run into live writes, one event at a time.

    Feed it every ``(mode, payload)`` pair of ``graph.stream(..., stream_mode=["custom", "values"])``
    (or of ``astream``: the logic is the same, so the synchronous and the asynchronous loops share
    it), then ask for ``result()``. A ``custom`` payload is a ``StreamEvent``: answer pieces go to
    ``write`` as they arrive (the very first one without its leading whitespace, so the answer does
    not start with a blank line); reasoning pieces go to ``note`` when ``show_thinking`` is on and are
    otherwise reduced to one "thinking..." note, so the user knows the model is working and not
    stuck. A ``values`` payload is the whole state after a step; the last one is the final state.
    The pieces are the model's untrusted text: this class only passes them on as text, and the
    caller prints them without interpreting them.
    """

    def __init__(
        self,
        *,
        write: Callable[[str], None],
        note: Callable[[str], None],
        show_thinking: bool = False,
    ) -> None:
        self._write = write
        self._note = note
        self._show_thinking = show_thinking
        self._final: dict[str, Any] | None = None
        self._shown: list[str] = []
        self._thinking_started = False
        self._thinking_closed = False

    def feed(self, mode: str, payload: Any) -> None:
        """Handle one event; events of kinds this does not know are ignored."""
        if mode == "values":
            self._final = dict(payload)
            return
        if mode != "custom" or not isinstance(payload, Mapping):
            return
        piece = str(payload.get("text", ""))
        kind = payload.get("event")
        if kind == "thinking":
            if not self._thinking_started:
                self._thinking_started = True
                self._note("thinking: " if self._show_thinking else "thinking...\n")
            if self._show_thinking:
                self._note(piece)
        elif kind == "text":
            if self._thinking_started and self._show_thinking and not self._thinking_closed:
                self._thinking_closed = True
                self._note("\n")
            if not self._shown:
                piece = piece.lstrip()
            if piece:
                self._shown.append(piece)
                self._write(piece)

    def result(self) -> StreamedAnswer:
        """The final state and the text that was written live; ``ValueError`` if no state came."""
        if self._final is None:
            raise ValueError("the graph ended without a final state")
        return StreamedAnswer(final=self._final, streamed="".join(self._shown))


def consume_stream(
    events: Iterable[Any],
    *,
    write: Callable[[str], None],
    note: Callable[[str], None],
    show_thinking: bool = False,
) -> StreamedAnswer:
    """Run a whole synchronous stream through a ``StreamCollector``.

    ``events`` are ``(mode, payload)`` pairs (LangGraph types its stream as ``Any``, so this does
    too, and unpacks each pair itself: a mismatch fails loudly, here).
    """
    collector = StreamCollector(write=write, note=note, show_thinking=show_thinking)
    for mode, payload in events:
        collector.feed(mode, payload)
    return collector.result()


def format_after_stream(streamed: StreamedAnswer, *, model: str, thread: str | None = None) -> str:
    """What to print once the stream is over (the cursor is at the end of the streamed answer).

    - Nothing was streamed: the whole answer, as ``format_answer`` prints it.
    - Something was streamed and the citation check left it alone: the sources, labels and readings.
    - Something was streamed and the check removed a mark: the corrected answer first, because the
      live text could not be taken back and still holds the mark that named no source.
    """
    final = streamed.final
    if not streamed.streamed.strip():
        return format_answer(final, model=model, thread=thread)
    lines: list[str] = []
    answer = str(final["answer"])
    if answer != str(final.get("raw_answer") or answer):
        lines.extend(
            [
                "corrected answer (the citation check removed a mark that names no source):",
                answer,
                "",
            ]
        )
    lines.append(format_answer(final, model=model, include_answer=False, thread=thread))
    return "\n".join(lines).strip()
