"""Checking the model's ``[n]`` citations against the sources it was shown (step S3.2, PLAN.md §7).

The model is told to cite each claim as ``[n]``, where ``n`` is the number in a source's header. It is
a small model, so it sometimes cites a number that was never shown (``[7]`` when the prompt held
three sources), writes ``[0]``, or groups marks as ``[1, 2]``. A citation that points nowhere is
worse than none: it looks like support and checks out against nothing, and in a document-poisoning
attack it is how a forged source gets a foothold (PLAN.md §8.3, vector 4). So the application, not
the model, decides what counts as a citation: ``check_citations`` keeps the marks that name a source
the prompt really held, removes the rest, and reports both lists. Rendering the citations (the
numbers become titles and paths) is the application's job too, in ``agent.render``.

What this does *not* do: it does not check that the cited source supports the claim. That is a
relevance question for ``cite_check`` (S3.5), and whether answers are grounded is measured offline by
the judge (S5.2). A valid ``[1]`` only means "source 1 was in the prompt".
"""

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["CitationCheck", "check_citations"]

# A number as the model is told to write it: no sign, no leading zero, at most nine digits. Anything
# else (``[01]``, ``[-1]``, ``[1-3]``, ``[a]``, ``[n]``) is ordinary text, not a citation attempt.
_NUMBER: Final = r"(?:0|[1-9]\d{0,8})"
# One mark, or several in one pair of brackets: ``[2]``, ``[1,2]``, ``[1 , 2]``. The spaces and tabs
# before it are captured so that removing the mark does not leave a gap in the sentence.
_MARKER: Final = re.compile(rf"(?P<lead>[ \t]*)\[(?P<body>{_NUMBER}(?:[ \t]*,[ \t]*{_NUMBER})*)\]")


@dataclass(frozen=True)
class CitationCheck:
    """The result of ``check_citations``.

    ``text`` is the answer with the invalid marks removed. ``cited`` are the numbers of the marks
    that are left, each once, in order of first appearance. ``dropped`` are the numbers that were
    removed, each once, in order of first appearance. A number is never in both.
    """

    text: str
    cited: tuple[int, ...]
    dropped: tuple[int, ...]


def check_citations(answer: str, used: int) -> CitationCheck:
    """Keep the ``[n]`` marks of ``answer`` whose ``n`` is a source the prompt held; drop the rest.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. ``used`` is how
    many sources the prompt held, numbered ``1..used`` (``PackedPrompt.used``). The rules:

    1. A mark is ``[n]`` or a comma-separated group ``[n, m]``, with each number written in plain
       decimal without a leading zero (``_NUMBER``). Other bracketed text is left alone.
    2. A number is valid exactly when ``1 <= n <= used``. ``[0]`` and ``[used + 1]`` are not.
    3. A mark with only valid numbers stays byte for byte as the model wrote it. In a group with
       some invalid numbers only the invalid ones go (``[1, 7]`` becomes ``[1]``); a mark with no
       valid number goes entirely, with the spaces in front of it (``"days [7]."`` becomes
       ``"days."``; ``"see [7] below"`` becomes ``"see below"``, one space kept between words).
    4. Removing a mark can bring brackets together into a new one (``[[7]1]`` becomes ``[1]``), so
       the check runs again on its own output until nothing changes. What it returns therefore
       contains no mark that it would still remove, and ``check_citations`` of its own ``text`` finds
       nothing to drop. Each pass that changes anything removes at least one digit and adds none,
       so it ends.
    5. The returned text is stripped at both ends. ``cited`` is read from the final text, so it
       always describes what the user will see; ``dropped`` collects every number removed along
       the way.
    6. With ``used == 0`` every mark is invalid. A negative ``used`` behaves the same way.
    """
    dropped: dict[int, None] = {}  # a dict keeps first-appearance order and holds each key once

    def settle(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match["body"].split(",")]
        valid = [number for number in numbers if 1 <= number <= used]
        for number in numbers:
            if not 1 <= number <= used:
                dropped.setdefault(number)
        if len(valid) == len(numbers):
            return match[0]
        if valid:
            return f"{match['lead']}[{', '.join(str(n) for n in dict.fromkeys(valid))}]"
        following = match.string[match.end() : match.end() + 1]
        return " " if match["lead"] and following.isalnum() else ""

    text = answer
    while (settled := _MARKER.sub(settle, text)) != text:
        text = settled
    text = text.strip()

    cited: dict[int, None] = {}
    for match in _MARKER.finditer(text):
        for part in match["body"].split(","):
            cited.setdefault(int(part))
    return CitationCheck(text=text, cited=tuple(cited), dropped=tuple(dropped))
