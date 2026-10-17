"""Read the golden-questions template into items (step S2.4).

A person fills in ``evals/datasets/golden-questions.md``, which is easier to write and read than
JSON. ``geekbot eval import`` turns it into ``golden.jsonl`` with this module and refuses to write
the file while the validator (``datasets.validate_golden``) finds errors. The markdown is the
source of truth and the JSONL is generated from it: edit the template, never the JSONL.

**What is read.** A block starts at a ``### ID`` heading (``S-001``, anything after the ID on that
line is ignored) and ends at the next heading or ``---``. Its fields are the ``- **Label:** value`` bullets. The kind of the item comes from the
letter of its ID (S single-hop, M multi-hop, T multi-turn, U unanswerable), not from the section
heading above it, so a block pasted into the wrong section is still read as what its ID says.
A value may continue on the following lines, including as ``- `` sub-bullets.

**Two small conventions** for fields that hold lists, because prose cannot be parsed reliably:

- *Where the answer is*: one place per line or separated by ``;``, each written
  ``path/in/corpus.md (page or section)``. The part in parentheses is optional and only for people.
- *Must include*: short phrases, one per line or separated by ``;``.

A block whose fields are all empty is an untouched copy of the template and is skipped (it is
reported, not an error). HTML comments (``<!-- ... -->``) and fenced code blocks (a worked example
in the template) are ignored.
"""

import re
from dataclasses import dataclass
from typing import Final

from geekbot.evals.datasets import (
    ID_PREFIX,
    GoldenItem,
    ItemType,
    Problem,
    Source,
)

__all__ = ["ImportResult", "import_markdown"]

_BLOCK: Final = re.compile(r"^###\s+(?P<id>\S+)(?:\s.*)?$")  # text after the id is ignored
_FIELD: Final = re.compile(r"^-\s+\*\*(?P<label>[^*]+?):\*\*\s*(?P<value>.*)$")
_END_OF_BLOCK: Final = re.compile(r"^(#{1,2}\s|-{3,}\s*$)")
_COMMENT: Final = re.compile(r"<!--.*?-->")
_PLACE: Final = re.compile(r"^(?P<path>.+?)\s*\((?P<locator>[^()]*)\)\s*$")
_TURN: Final = re.compile(r"turn\s*(?P<number>\d+)")
_LIST_MARK: Final = re.compile(r"^[-*]\s+")

_TYPE_BY_LETTER: Final[dict[str, ItemType]] = {letter: kind for kind, letter in ID_PREFIX.items()}


@dataclass(frozen=True)
class ImportResult:
    """The items read, the ids of untouched template blocks, and what could not be read."""

    items: tuple[GoldenItem, ...]
    blank: tuple[str, ...]
    problems: tuple[Problem, ...]


@dataclass
class _Block:
    id: str
    line: int
    fields: dict[str, list[str]]


def _field_key(label: str) -> str | None:
    """The canonical name of a bullet's label, or ``None`` for one we do not know."""
    text = label.strip().lower()
    turn = _TURN.match(text)
    if turn is not None:
        return f"turn{int(turn.group('number'))}"
    for prefix, key in (
        ("question", "question"),
        ("where the answer", "where"),
        ("must include", "must_include"),
        ("why it can", "reason"),
        ("notes", "notes"),
    ):
        if text.startswith(prefix):
            return key
    return None


def _read_blocks(text: str) -> tuple[list[_Block], list[Problem]]:
    """Cut the file into blocks of (canonical label -> lines of value)."""
    blocks: list[_Block] = []
    problems: list[Problem] = []
    current: _Block | None = None
    label: str | None = None
    in_fence = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _COMMENT.sub("", raw).rstrip()
        if line.lstrip().startswith("```"):  # a worked example in the template is not data
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        start = _BLOCK.match(line)
        if start is not None:
            current = _Block(id=start.group("id"), line=number, fields={})
            blocks.append(current)
            label = None
        elif _END_OF_BLOCK.match(line):
            current, label = None, None
        elif current is not None:
            bullet = _FIELD.match(line)
            if bullet is not None:
                label = _field_key(bullet.group("label"))
                if label is None:
                    problems.append(
                        Problem(
                            "error",
                            current.id,
                            f"line {number}: unknown field {bullet.group('label').strip()!r}",
                        )
                    )
                else:
                    current.fields.setdefault(label, []).append(bullet.group("value").strip())
            elif label is not None and line.strip():
                current.fields[label].append(line.strip())
    return blocks, problems


def _text(lines: list[str]) -> str:
    """The lines of a free-text field as one line of text."""
    return " ".join(part for part in (line.strip() for line in lines) if part)


def _list(lines: list[str]) -> list[str]:
    """Split a list field on line ends and on ``;`` outside parentheses; drop bullets and blanks."""
    entries: list[str] = []
    for line in lines:
        depth = 0
        piece: list[str] = []
        for char in line:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            if char == ";" and depth == 0:
                entries.append("".join(piece))
                piece = []
            else:
                piece.append(char)
        entries.append("".join(piece))
    cleaned = (_LIST_MARK.sub("", entry.strip()).strip() for entry in entries)
    return [entry for entry in cleaned if entry]


def _place(text: str) -> Source:
    """``path (locator)`` -> a ``Source``; backticks and ``./`` around the path are dropped."""
    found = _PLACE.match(text)
    path, locator = (found.group("path"), found.group("locator")) if found else (text, "")
    path = path.strip().strip("`").strip().replace("\\", "/").removeprefix("./")
    return Source(path=path, locator=locator.strip())


def _turns(block: _Block) -> list[str]:
    """The block's ``Turn N`` fields in order of N."""
    numbered = sorted(
        (int(key.removeprefix("turn")), _text(lines))
        for key, lines in block.fields.items()
        if key.startswith("turn")
    )
    return [text for _, text in numbered]


def _item_of(block: _Block, kind: ItemType) -> GoldenItem:
    turns = _turns(block)
    if kind == "multi_turn" and turns:
        *earlier, question = turns  # the highest turn is the follow-up that gets graded
    else:
        earlier, question = [], _text(block.fields.get("question", []))
    return GoldenItem(
        id=block.id,
        type=kind,
        question=question,
        sources=tuple(_place(entry) for entry in _list(block.fields.get("where", []))),
        must_include=tuple(_list(block.fields.get("must_include", []))),
        earlier_turns=tuple(earlier),
        reason=_text(block.fields.get("reason", [])),
        notes=_text(block.fields.get("notes", [])),
    )


def _is_blank(block: _Block) -> bool:
    return all(not _text(lines) for lines in block.fields.values())


def import_markdown(text: str) -> ImportResult:
    """Read every ``### ID`` block of the template.

    Returns the items (not yet validated: ``datasets.validate_golden`` is the next step), the ids of
    the blocks that were left blank, and the problems in reading: a heading that is not an item id
    and a field label that is not in the template. Reading never raises on content.
    """
    blocks, problems = _read_blocks(text)
    items: list[GoldenItem] = []
    blank: list[str] = []
    for block in blocks:
        kind = _TYPE_BY_LETTER.get(block.id[:1]) if re.fullmatch(r"[A-Z]-\d+", block.id) else None
        if kind is None:
            problems.append(
                Problem(
                    "error",
                    block.id,
                    f"line {block.line}: the heading is not an item id like S-001, M-001, T-001 "
                    "or U-001",
                )
            )
        elif _is_blank(block):
            blank.append(block.id)
        else:
            if kind != "multi_turn" and any(key.startswith("turn") for key in block.fields):
                problems.append(
                    Problem(
                        "error",
                        block.id,
                        "has Turn fields but its id is not a T- id: the turns would be dropped",
                    )
                )
            items.append(_item_of(block, kind))
    return ImportResult(items=tuple(items), blank=tuple(blank), problems=tuple(problems))
