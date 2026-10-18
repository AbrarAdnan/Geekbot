"""The golden set: the hand-written questions Geekbot is graded on (step S2.4, PLAN.md §9).

The golden set is *system-specific behavioural evaluation*: questions about **our** documents, with
the places that answer them. It says nothing about retrieval quality in general (BEIR does that).
This module is the file format, the checks that keep a hand-made file honest, the development/test
split, and the judgements (``qrels``) that ``evals.metrics`` scores against.

**Four kinds of item** (PLAN.md §9, ``evals/datasets/golden-questions.md``):

- ``single_hop``: the answer is in one place. One source.
- ``multi_hop``: the answer needs facts from two or more places. Two or more sources.
- ``multi_turn``: a follow-up that only makes sense after an earlier question. The item stores the
  earlier turn(s) and the follow-up, which is the question that gets graded.
- ``unanswerable``: the documents do not contain the answer, so the right behaviour is to say so.
  No sources.

**Sources are corpus-relative paths.** A document's ID is a hash of its absolute path (D-054), so
an ID stored in the file would break the moment the repository is cloned somewhere else. The file
holds ``handbook/setup.md`` and the IDs are computed at load time against the folder the corpus was
ingested from (``build_qrels``). The price: the folder used for ``geekbot ingest`` and the folder
given here must be the same one, which ``unindexed_sources`` checks against the store.

**The split** (D-065, answering D-033's first question): 40% of each kind is a *development* item,
the rest is *test*. Whatever is chosen by trying values (the abstain threshold, the rerank depth)
is chosen on development items only, and the reported golden-set score is the test items'. An
item's side is decided by hashing its ID with a fixed seed and taking the first 40% of each kind
in hash order, so it does not depend on the order of the file, and adding an item moves at most
one existing item of its kind across the line. Even so, **finish the golden set before tuning
anything**: a split that moves while you tune is not a split.

**Retrieval metrics see only some items** (D-065): an unanswerable item has no relevant document
(its metrics are undefined, ``evals.metrics`` returns ``None``), and a multi-turn follow-up cannot
be searched as written until ``condense`` (S3.4) rewrites it. ``retrieval_items`` selects what is
left.
"""

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from geekbot.ingest.ids import doc_id_of

__all__ = [
    "DEV_FRACTION",
    "ID_PREFIX",
    "ITEM_TYPES",
    "RETRIEVAL_TYPES",
    "SPLIT_SEED",
    "TARGET_COUNTS",
    "GoldenFormatError",
    "GoldenItem",
    "ItemType",
    "Problem",
    "Source",
    "Split",
    "ValidationReport",
    "assign_splits",
    "build_qrels",
    "corpus_relative",
    "describe_set",
    "dump_golden",
    "list_corpus_files",
    "load_golden",
    "parse_golden",
    "retrieval_items",
    "unindexed_sources",
    "validate_golden",
    "write_golden",
]

ItemType = Literal["single_hop", "multi_hop", "multi_turn", "unanswerable"]
Split = Literal["dev", "test"]

ITEM_TYPES: Final[tuple[ItemType, ...]] = ("single_hop", "multi_hop", "multi_turn", "unanswerable")
# The letter an item's ID starts with, as in the template: S-001, M-001, T-001, U-001.
ID_PREFIX: Final[dict[ItemType, str]] = {
    "single_hop": "S",
    "multi_hop": "M",
    "multi_turn": "T",
    "unanswerable": "U",
}
# PLAN.md §9: ~100 items. A count outside +-20% of its target is a warning, never an error: the
# targets are "about".
TARGET_COUNTS: Final[dict[ItemType, int]] = {
    "single_hop": 50,
    "multi_hop": 15,
    "multi_turn": 15,
    "unanswerable": 20,
}
TARGET_SLACK: Final = 0.2
# The kinds the retrieval metrics score today (D-065).
RETRIEVAL_TYPES: Final[tuple[ItemType, ...]] = ("single_hop", "multi_hop")

DEV_FRACTION: Final = 0.4  # D-065: 40 development items of 100
SPLIT_SEED: Final = "geekbot-golden-split-1"

MIN_MULTI_HOP_SOURCES: Final = 2  # "multi" means two or more places

_TYPE_BY_NAME: Final[dict[str, ItemType]] = {kind: kind for kind in ITEM_TYPES}
_ID_PATTERN: Final = re.compile(r"[SMTU]-\d{3,}")
_DRIVE_PATTERN: Final = re.compile(r"[A-Za-z]:")
_FIELDS: Final = frozenset(
    {"id", "type", "question", "earlier_turns", "sources", "must_include", "reason", "notes"}
)
_SOURCE_FIELDS: Final = frozenset({"path", "locator"})


class GoldenFormatError(ValueError):
    """The file is not a golden-set file: bad JSON, a missing field, a field of the wrong type."""


@dataclass(frozen=True)
class Source:
    """One place where an answer lives: a corpus-relative path and, for a person, where in it."""

    path: str  # relative to the corpus folder, forward slashes, e.g. "handbook/setup.md"
    locator: str = ""  # a page or a section: "p. 3", "Install > Windows". Not machine-checked.


@dataclass(frozen=True)
class GoldenItem:
    """One graded question. ``question`` is the follow-up for a ``multi_turn`` item.

    ``must_include`` are short phrases a correct answer has to contain (the deterministic answer
    check of PLAN.md §9). ``reason`` is why an unanswerable question cannot be answered: a topic
    close to the documents makes a better test, and writing it down keeps the item auditable.
    """

    id: str
    type: ItemType
    question: str
    sources: tuple[Source, ...] = ()
    must_include: tuple[str, ...] = ()
    earlier_turns: tuple[str, ...] = ()
    reason: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Problem:
    """Something wrong (``error``: fix it) or doubtful (``warning``) about an item or the set."""

    severity: Literal["error", "warning"]
    item_id: str | None  # None: about the set as a whole
    message: str

    def __str__(self) -> str:
        where = self.item_id if self.item_id is not None else "set"
        return f"{self.severity}: {where}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """What ``validate_golden`` found. ``ok`` means no errors; warnings do not block."""

    problems: tuple[Problem, ...]
    counts: Mapping[ItemType, int]

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


# --- the file format -------------------------------------------------------------------------


def _fail(line: int, message: str) -> GoldenFormatError:
    return GoldenFormatError(f"line {line}: {message}")


def _text_of(obj: Mapping[str, object], key: str, line: int, *, required: bool = False) -> str:
    value = obj.get(key)
    if value is None:
        if required:
            raise _fail(line, f"missing the field {key!r}")
        return ""
    if not isinstance(value, str):
        raise _fail(line, f"{key!r} must be text, got {type(value).__name__}")
    return value


def _texts_of(obj: Mapping[str, object], key: str, line: int) -> tuple[str, ...]:
    value = obj.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise _fail(line, f"{key!r} must be a list of text")
    return tuple(value)


def _sources_of(obj: Mapping[str, object], line: int) -> tuple[Source, ...]:
    value = obj.get("sources")
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _fail(line, "'sources' must be a list of {path, locator}")
    sources: list[Source] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise _fail(line, "each source must be an object with 'path' and 'locator'")
        unknown = set(entry) - _SOURCE_FIELDS
        if unknown:
            raise _fail(line, f"unknown source field(s): {', '.join(sorted(map(str, unknown)))}")
        sources.append(
            Source(
                path=_text_of(entry, "path", line, required=True),
                locator=_text_of(entry, "locator", line),
            )
        )
    return tuple(sources)


def _item_from_json(obj: object, line: int) -> GoldenItem:
    if not isinstance(obj, dict):
        raise _fail(line, "each line must be a JSON object")
    unknown = set(obj) - _FIELDS
    if unknown:  # a typo such as "must_includes" must not silently drop the field
        raise _fail(line, f"unknown field(s): {', '.join(sorted(map(str, unknown)))}")
    name = _text_of(obj, "type", line, required=True)
    kind = _TYPE_BY_NAME.get(name)
    if kind is None:
        raise _fail(line, f"'type' must be one of {', '.join(ITEM_TYPES)}, got {name!r}")
    return GoldenItem(
        id=_text_of(obj, "id", line, required=True),
        type=kind,
        question=_text_of(obj, "question", line, required=True),
        sources=_sources_of(obj, line),
        must_include=_texts_of(obj, "must_include", line),
        earlier_turns=_texts_of(obj, "earlier_turns", line),
        reason=_text_of(obj, "reason", line),
        notes=_text_of(obj, "notes", line),
    )


def parse_golden(text: str) -> list[GoldenItem]:
    """Read JSON Lines: one item per line, blank lines ignored.

    Strict on purpose: an unknown field, a wrong type or an unknown ``type`` is a
    ``GoldenFormatError`` with the line number. Whether the *content* makes sense (one source for a
    single-hop question, a reason for an unanswerable one) is ``validate_golden``'s job.
    """
    items: list[GoldenItem] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj: object = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _fail(number, f"not valid JSON ({error.msg})") from error
        items.append(_item_from_json(obj, number))
    return items


def load_golden(path: Path) -> list[GoldenItem]:
    """``parse_golden`` of a UTF-8 file. A missing file raises ``FileNotFoundError``."""
    return parse_golden(path.read_text(encoding="utf-8"))


def _item_to_json(item: GoldenItem) -> dict[str, object]:
    return {
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "earlier_turns": list(item.earlier_turns),
        "sources": [{"path": s.path, "locator": s.locator} for s in item.sources],
        "must_include": list(item.must_include),
        "reason": item.reason,
        "notes": item.notes,
    }


def dump_golden(items: Iterable[GoldenItem]) -> str:
    """JSON Lines text, every field of every item, in a fixed order (so a diff shows real edits)."""
    return "".join(json.dumps(_item_to_json(item), ensure_ascii=False) + "\n" for item in items)


def write_golden(path: Path, items: Iterable[GoldenItem]) -> None:
    """Write ``dump_golden`` as UTF-8 with ``\\n`` line ends, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_golden(items), encoding="utf-8", newline="\n")


# --- the checks ------------------------------------------------------------------------------


def _error(item_id: str | None, message: str) -> Problem:
    return Problem("error", item_id, message)


def _warning(item_id: str | None, message: str) -> Problem:
    return Problem("warning", item_id, message)


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _check_source_path(
    item_id: str, path: str, corpus_files: Collection[str] | None
) -> list[Problem]:
    """A path must be corpus-relative with forward slashes, and name a file that exists."""
    if not path.strip():
        return [_error(item_id, "a source has an empty path")]
    bad: list[str] = []
    if "\\" in path:
        bad.append("use forward slashes")
    if PurePosixPath(path).is_absolute() or _DRIVE_PATTERN.match(path):
        bad.append("give the path relative to the corpus folder, not an absolute one")
    if ".." in PurePosixPath(path).parts:
        bad.append("'..' would leave the corpus folder")
    if bad:
        return [_error(item_id, f"source path {path!r}: {'; '.join(bad)}")]
    if corpus_files is not None and path not in corpus_files:
        return [_error(item_id, f"source {path!r} is not a file in the corpus folder")]
    return []


def _check_sources(item: GoldenItem, corpus_files: Collection[str] | None) -> list[Problem]:
    problems: list[Problem] = []
    for source in item.sources:
        problems.extend(_check_source_path(item.id, source.path, corpus_files))
    if len(set(item.sources)) != len(item.sources):
        problems.append(_warning(item.id, "the same place is listed twice"))
    count = len(item.sources)
    if item.type == "single_hop" and count != 1:
        problems.append(_error(item.id, f"a single-hop item needs exactly 1 source, has {count}"))
    elif item.type == "multi_hop":
        if count < MIN_MULTI_HOP_SOURCES:
            problems.append(_error(item.id, f"a multi-hop item needs 2+ sources, has {count}"))
        elif len({s.path for s in item.sources}) == 1:
            problems.append(
                _warning(
                    item.id,
                    "all sources are in one document: document-level recall cannot tell this "
                    "from a single-hop question",
                )
            )
    elif item.type == "multi_turn" and count < 1:
        problems.append(_error(item.id, "a multi-turn question needs at least 1 source"))
    elif item.type == "unanswerable" and count:
        problems.append(_error(item.id, "an unanswerable question has no sources"))
    return problems


def _check_text_fields(item: GoldenItem) -> list[Problem]:
    problems: list[Problem] = []
    if not item.question.strip():
        problems.append(_error(item.id, "the question is empty"))
    if item.type == "multi_turn":
        if not item.earlier_turns:
            problems.append(_error(item.id, "a multi-turn item needs the earlier turn"))
        if any(not turn.strip() for turn in item.earlier_turns):
            problems.append(_error(item.id, "an earlier turn is empty"))
    elif item.earlier_turns:
        problems.append(_error(item.id, "only a multi-turn item has earlier turns"))
    if item.type == "unanswerable":
        if item.must_include:
            problems.append(_error(item.id, "an unanswerable question has no must-include phrases"))
        if not item.reason.strip():
            problems.append(_error(item.id, "say why the documents cannot answer it (reason)"))
    else:
        if not item.must_include:
            problems.append(_error(item.id, "needs at least 1 must-include phrase"))
        if any(not phrase.strip() for phrase in item.must_include):
            problems.append(_error(item.id, "a must-include phrase is empty"))
    return problems


def _check_id(item: GoldenItem) -> list[Problem]:
    if _ID_PATTERN.fullmatch(item.id) is None:
        return [_error(item.id, "the id must look like S-001 (letter, dash, 3 or more digits)")]
    if item.id[0] != ID_PREFIX[item.type]:
        return [_error(item.id, f"a {item.type} item's id starts with {ID_PREFIX[item.type]!r}")]
    return []


def _check_set(items: Sequence[GoldenItem], counts: Mapping[ItemType, int]) -> list[Problem]:
    problems: list[Problem] = []
    if not items:
        return [_error(None, "the golden set is empty")]
    for item_id, count in Counter(item.id for item in items).items():
        if count > 1:
            problems.append(_error(item_id, f"the id is used {count} times"))
    seen: dict[tuple[str, ...], str] = {}
    for item in items:
        key = (*map(_normalized, item.earlier_turns), _normalized(item.question))
        if item.question.strip() and key in seen and seen[key] != item.id:
            problems.append(_error(item.id, f"the same question as {seen[key]}"))
        seen.setdefault(key, item.id)
    for kind in ITEM_TYPES:
        target = TARGET_COUNTS[kind]
        # round() first: 15 * (1 + 0.2) must be 18.0 for the bound, not 18.000000000000004
        low = math.floor(round(target * (1 - TARGET_SLACK), 6))
        high = math.ceil(round(target * (1 + TARGET_SLACK), 6))
        if not low <= counts[kind] <= high:
            problems.append(
                _warning(None, f"{counts[kind]} {kind} items; the plan aims for about {target}")
            )
    return problems


def validate_golden(
    items: Sequence[GoldenItem], *, corpus_files: Collection[str] | None = None
) -> ValidationReport:
    """Check a golden set for the mistakes a hand-made file collects. This step's ✍️ piece.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. What it
    checks, and why each rule exists:

    - **Ids** look like ``S-001`` and start with the letter of the item's type (``S`` single-hop,
      ``M`` multi-hop, ``T`` multi-turn, ``U`` unanswerable): a block copied from the wrong
      template section is caught. Ids are unique, and no two items ask the same question (a
      multi-turn item is compared together with its earlier turns).
    - **Shape by kind.** Single-hop: exactly 1 source. Multi-hop: 2 or more (a warning if they are
      all in one document, because the document-level metrics then cannot tell it from a
      single-hop one). Multi-turn: an earlier turn and at least 1 source. Unanswerable: no
      sources, no must-include phrases, and a ``reason``. Every kind but unanswerable needs at
      least 1 non-empty must-include phrase; only multi-turn items have earlier turns.
    - **Source paths** are relative to the corpus folder, with forward slashes, no ``..``, never
      absolute. If ``corpus_files`` is given (the corpus-relative paths found on disk), each source
      must be one of them. The comparison is exact, including case, so a file that happens to be
      found on Windows is also found on Linux.
    - **Counts.** A kind more than 20% away from its target (50 / 15 / 15 / 20) is a warning: the
      targets are "about", so a set of 90 is not an error.

    Errors block ``geekbot eval import``; warnings are printed. The report also carries the count
    of each kind.
    """
    counts: dict[ItemType, int] = {kind: 0 for kind in ITEM_TYPES}
    for item in items:
        counts[item.type] += 1
    problems: list[Problem] = []
    for item in items:
        problems.extend(_check_id(item))
        problems.extend(_check_text_fields(item))
        problems.extend(_check_sources(item, corpus_files))
    problems.extend(_check_set(items, counts))
    return ValidationReport(problems=tuple(problems), counts=counts)


# --- the split -------------------------------------------------------------------------------


def _split_key(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest()


def assign_splits(
    items: Iterable[GoldenItem], *, dev_fraction: float = DEV_FRACTION, seed: str = SPLIT_SEED
) -> dict[str, Split]:
    """Say, for every item id, whether it is a ``dev`` or a ``test`` item (D-065).

    Within each kind, the ids are put in the order of ``sha256(seed | id)`` and the first
    ``round(dev_fraction * n)`` are development items. So: the split is stratified (every kind is
    cut the same way, 20/30 of 50, 6/9 of 15, 8/12 of 20), it does not depend on the order of the
    items, and adding one item moves at most one existing item of its kind across the line.
    ``ValueError`` if ``dev_fraction`` is not between 0 and 1 or an id appears twice.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError(f"dev_fraction must be between 0 and 1, got {dev_fraction}")
    ids_by_type: dict[str, list[str]] = {}
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"id {item.id!r} appears twice; run validate_golden first")
        seen.add(item.id)
        ids_by_type.setdefault(item.type, []).append(item.id)
    splits: dict[str, Split] = {}
    for ids in ids_by_type.values():
        ordered = sorted(ids, key=lambda item_id: (_split_key(seed, item_id), item_id))
        dev_count = round(len(ordered) * dev_fraction)
        for position, item_id in enumerate(ordered):
            splits[item_id] = "dev" if position < dev_count else "test"
    return splits


def retrieval_items(
    items: Iterable[GoldenItem], splits: Mapping[str, Split], split: Split
) -> list[GoldenItem]:
    """The items the retrieval metrics score on one side of the split (D-065).

    Single-hop and multi-hop only: an unanswerable item has no relevant document, and a multi-turn
    follow-up is not searchable until ``condense`` exists (S3.4). Order of ``items`` is kept.
    """
    return [i for i in items if i.type in RETRIEVAL_TYPES and splits.get(i.id) == split]


# --- from paths to judgements ----------------------------------------------------------------


def build_qrels(items: Iterable[GoldenItem], corpus_dir: Path) -> dict[str, dict[str, int]]:
    """Judgements for ``evals.metrics``: item id -> {doc_id: 1} for each document that answers it.

    The ID of a document is a hash of its absolute path (D-054), so ``corpus_dir`` must be the
    folder that was given to ``geekbot ingest`` (``unindexed_sources`` checks that it was). A
    document that is the source twice is one judgement. An unanswerable item gets ``{}``, for which
    every metric returns ``None``.
    """
    return {
        item.id: {doc_id_of(corpus_dir / source.path): 1 for source in item.sources}
        for item in items
    }


def unindexed_sources(
    items: Iterable[GoldenItem], corpus_dir: Path, ready_doc_ids: Collection[str]
) -> list[tuple[str, str]]:
    """(item id, path) for every source whose document is not a ready document in the store.

    The failure this prevents is silent: if the corpus was ingested from a different path (or a
    file failed to ingest), the judgements name documents the search can never return, recall is 0
    for those questions, and the result looks like a bad retriever instead of a bad setup.
    """
    ready = set(ready_doc_ids)
    return [
        (item.id, source.path)
        for item in items
        for source in item.sources
        if doc_id_of(corpus_dir / source.path) not in ready
    ]


def corpus_relative(path: str | os.PathLike[str], corpus_dir: Path) -> str | None:
    """``path`` as the golden file spells it (relative to ``corpus_dir``, forward slashes).

    ``None`` if the file is not under the corpus folder. Nothing is read from disk.
    """
    base = Path(os.path.normpath(corpus_dir.absolute()))
    target = Path(os.path.normpath(Path(path).absolute()))
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return None


def list_corpus_files(corpus_dir: Path) -> set[str]:
    """Every file under ``corpus_dir`` as a corpus-relative path with forward slashes."""
    return {
        path.relative_to(corpus_dir).as_posix() for path in corpus_dir.rglob("*") if path.is_file()
    }


def describe_set(items: Sequence[GoldenItem], splits: Mapping[str, Split]) -> str:
    """A small table: each kind, how many items, the target, and how many are dev / test."""
    lines = [f"{'kind':<14}{'items':>6}{'target':>8}{'dev':>6}{'test':>6}"]
    for kind in ITEM_TYPES:
        of_kind = [item for item in items if item.type == kind]
        dev = sum(1 for item in of_kind if splits.get(item.id) == "dev")
        lines.append(
            f"{kind:<14}{len(of_kind):>6}{TARGET_COUNTS[kind]:>8}{dev:>6}{len(of_kind) - dev:>6}"
        )
    total_dev = sum(1 for item in items if splits.get(item.id) == "dev")
    lines.append(
        f"{'total':<14}{len(items):>6}{sum(TARGET_COUNTS.values()):>8}"
        f"{total_dev:>6}{len(items) - total_dev:>6}"
    )
    return "\n".join(lines)
