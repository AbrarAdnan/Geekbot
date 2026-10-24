"""Answer evaluation: the deterministic checks of PLAN.md §9 (step S5.2).

PLAN §9 sets a **measurement hierarchy**: retrieval metrics, then *deterministic answer checks*, then
human labels, then a calibrated judge (supporting evidence only). This file is the second rung: what
can be decided about an answer by plain code, with no model and no opinion.

For every golden question the real answer graph is run once (a follow-up on a conversation with its
earlier turns, as a user would ask it), the answer is kept with the sources the model was shown, and
four things are checked:

- **Abstention.** An unanswerable question should be refused and an answerable one should not. The
  graph decides refusal by the model's own wording (D-082); this only reads what it decided.
- **Must-include recall.** The share of the item's ``must_include`` phrases the answer contains
  (``evals.bench.must_include_recall``).
- **Citation hit.** Does the answer cite at least one of the documents the golden item names as its
  source, and what share of those documents does it cite? A citation to the right document does not
  prove the claim is in it; it is the cheapest check that the answer is looking in the right place.
- **Invalid-citation rate.** The graph removes ``[n]`` marks that name no source the prompt held
  (S3.2); how often the model wrote one is a measure of how well it follows the citation format.

None of these says an answer is *true*: an answer can contain the right phrase and cite the right
document and still say something wrong around them. That is what human labels and the judge are for,
and why they are reported next to these numbers and never instead of them.

Everything here is a plain function over data (the tests give it a fake graph); the part that loads the
models is ``answers_run.py``. The golden set is Claude-authored and provisional (D-067): the numbers
are for the system on that set, not general claims.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from geekbot.evals.bench import must_include_recall
from geekbot.evals.datasets import GoldenItem

__all__ = [
    "ANSWERABLE_TYPES",
    "AnswerRecord",
    "AnswerScores",
    "SourceView",
    "corpus_relative",
    "format_answers_report",
    "generate_answers",
    "load_answers",
    "record_from_state",
    "save_answers",
    "score_answer",
    "summarize_scores",
]

ANSWERABLE_TYPES: Final = ("single_hop", "multi_hop", "multi_turn")
ANSWERS_VERSION: Final = 1


@dataclass(frozen=True)
class SourceView:
    """One source the prompt held, as the evaluation keeps it: where it is and what it says."""

    n: int
    path: str  # relative to the corpus folder, forward slashes
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class AnswerRecord:
    """One golden question and what the graph answered, with the sources it was shown."""

    id: str
    type: str
    split: str
    question: str
    earlier_turns: tuple[str, ...]
    standalone: str  # what retrieval searched for (the rewrite of a follow-up); "" on a first turn
    answer: str  # what the user sees
    raw_answer: str  # the model's own text, before the citation check
    abstained: bool
    cited: tuple[int, ...]  # the valid [n] marks left in the answer
    dropped: tuple[int, ...]  # the marks the check removed (they named no source the prompt held)
    sources: tuple[SourceView, ...]  # the ones the prompt held, numbered as the model saw them
    seconds: float
    first_token_s: float | None
    output_tokens: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "split": self.split,
            "question": self.question,
            "earlier_turns": list(self.earlier_turns),
            "standalone": self.standalone,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "abstained": self.abstained,
            "cited": list(self.cited),
            "dropped": list(self.dropped),
            "sources": [
                {"n": s.n, "path": s.path, "title": s.title, "section": s.section, "text": s.text}
                for s in self.sources
            ],
            "seconds": round(self.seconds, 3),
            "first_token_s": None if self.first_token_s is None else round(self.first_token_s, 3),
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AnswerRecord":
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            split=str(data["split"]),
            question=str(data["question"]),
            earlier_turns=tuple(str(t) for t in data.get("earlier_turns", ())),
            standalone=str(data.get("standalone", "")),
            answer=str(data["answer"]),
            raw_answer=str(data.get("raw_answer", "")),
            abstained=bool(data["abstained"]),
            cited=tuple(int(n) for n in data.get("cited", ())),
            dropped=tuple(int(n) for n in data.get("dropped", ())),
            sources=tuple(
                SourceView(
                    int(s["n"]),
                    str(s["path"]),
                    str(s.get("title", "")),
                    str(s.get("section", "")),
                    str(s["text"]),
                )
                for s in data.get("sources", ())
            ),
            seconds=float(data.get("seconds", 0.0)),
            first_token_s=data.get("first_token_s"),
            output_tokens=data.get("output_tokens"),
        )


def corpus_relative(path: str, corpus: Path) -> str:
    """``path`` relative to the corpus folder with forward slashes; only the file name if it is outside.

    The graph reports absolute paths (the files were ingested from where they are); the golden items
    name sources relative to the corpus, and a result file committed to a repository should not carry
    the folders of someone's PC.
    """
    try:
        return Path(path).resolve().relative_to(corpus.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).name


def record_from_state(
    item: GoldenItem,
    split: str,
    final: Mapping[str, Any],
    *,
    corpus: Path,
    seconds: float,
) -> AnswerRecord:
    """The record of one item from the answer graph's final state."""
    used = int(final.get("used", 0))
    sources = tuple(
        SourceView(
            n=int(source["n"]),
            path=corpus_relative(str(source["path"]), corpus),
            title=str(source.get("title", "")),
            section=str(source.get("section", "")),
            text=str(source.get("text", "")),
        )
        for source in list(final.get("sources", []))[:used]
    )
    generation = final.get("generation") or {}
    tokens = generation.get("output_tokens")
    return AnswerRecord(
        id=item.id,
        type=item.type,
        split=split,
        question=item.question,
        earlier_turns=item.earlier_turns,
        standalone=str(final.get("standalone_question", "")) if item.earlier_turns else "",
        answer=str(final.get("answer", "")),
        raw_answer=str(final.get("raw_answer", "")),
        abstained=bool(final.get("abstained", False)),
        cited=tuple(int(n) for n in final.get("cited", [])),
        dropped=tuple(int(n) for n in final.get("dropped_citations", [])),
        sources=sources,
        seconds=seconds,
        first_token_s=generation.get("first_token_seconds"),
        output_tokens=None if tokens is None else int(tokens),
    )


def generate_answers(
    items: Iterable[GoldenItem],
    splits: Mapping[str, str],
    ask: Callable[[GoldenItem], Mapping[str, Any]],
    *,
    corpus: Path,
    clock: Callable[[], float],
    existing: Mapping[str, AnswerRecord] | None = None,
    on_record: Callable[[AnswerRecord, bool], None] | None = None,
) -> list[AnswerRecord]:
    """One record for every item, in the items' order; items already in ``existing`` are not asked again.

    ``ask`` runs one golden item through the graph (a multi-turn item as a conversation) and returns
    the final state of its last turn; an exception from it is not caught, because a run that broke
    must not be counted as a refusal. ``on_record(record, reused)`` is called for progress, so a long
    run can save as it goes and a crash loses only the item in flight.
    """
    found = dict(existing or {})
    records: list[AnswerRecord] = []
    for item in items:
        reused = item.id in found
        if reused:
            record = found[item.id]
        else:
            started = clock()
            final = ask(item)
            record = record_from_state(
                item, splits[item.id], final, corpus=corpus, seconds=clock() - started
            )
        records.append(record)
        if on_record is not None:
            on_record(record, reused)
    return records


# --- saving ----------------------------------------------------------------------------------


def save_answers(path: Path, meta: Mapping[str, Any], records: Sequence[AnswerRecord]) -> None:
    """Write ``answers.json``: the run's facts (``meta``) and every record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": ANSWERS_VERSION,
        **meta,
        "records": [record.to_json() for record in records],
    }
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_answers(path: Path) -> tuple[dict[str, Any], list[AnswerRecord]]:
    """``(meta, records)`` from an ``answers.json``. ``ValueError`` if the file is not one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [AnswerRecord.from_json(entry) for entry in data["records"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path} is not an answers file: {error}") from error
    meta = {key: value for key, value in data.items() if key != "records"}
    return meta, records


# --- the checks ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerScores:
    """The deterministic checks of one answer. ``None`` means the check does not apply, never zero."""

    id: str
    type: str
    split: str
    answered: bool
    abstention_correct: bool
    must_include: float | None
    citation_hit: bool | None
    citation_recall: float | None
    invalid_citations: int


def score_answer(item: GoldenItem, record: AnswerRecord) -> AnswerScores:
    """The checks of PLAN.md §9 for one answer.

    - ``abstention_correct``: refused exactly when the item is unanswerable.
    - ``must_include``: the share of the item's phrases in the answer (``None`` for an item with
      none). A refusal of an answerable question scores 0, which is what it is.
    - ``citation_hit`` and ``citation_recall``: for an answerable item, which of the documents named
      by the golden item the answer **cites** (through the marks that survived the check: ``cited``
      numbers mapped to the sources the prompt held). A refusal cites nothing: hit false, recall 0.
      ``None`` for an unanswerable item (it has no documents to cite).
    - ``invalid_citations``: how many numbers the check removed from the answer.
    """
    unanswerable = item.type == "unanswerable"
    hit: bool | None = None
    recall: float | None = None
    if not unanswerable:
        gold = {source.path for source in item.sources}
        by_number = {source.n: source.path for source in record.sources}
        cited_paths = {by_number[n] for n in record.cited if n in by_number}
        found = gold & cited_paths
        hit = bool(found)
        recall = len(found) / len(gold) if gold else None
    return AnswerScores(
        id=item.id,
        type=item.type,
        split=record.split,
        answered=not record.abstained,
        abstention_correct=record.abstained == unanswerable,
        must_include=must_include_recall(record.answer, item.must_include),
        citation_hit=hit,
        citation_recall=recall,
        invalid_citations=len(record.dropped),
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _counts(scores: Sequence[AnswerScores]) -> dict[str, Any]:
    """The numbers for one group of scores: counts, never a bare rate."""
    answerable = [s for s in scores if s.type in ANSWERABLE_TYPES]
    answered = [s for s in answerable if s.answered]
    unanswerable = [s for s in scores if s.type == "unanswerable"]
    must = [s.must_include for s in answerable if s.must_include is not None]
    recall = [s.citation_recall for s in answered if s.citation_recall is not None]
    return {
        "n": len(scores),
        "answerable": len(answerable),
        "answerable_answered": len(answered),
        "answerable_refused": len(answerable) - len(answered),
        "unanswerable": len(unanswerable),
        "unanswerable_refused": sum(1 for s in unanswerable if not s.answered),
        "unanswerable_answered": sum(1 for s in unanswerable if s.answered),
        "must_include_mean": _mean(must),
        "must_include_n": len(must),
        "citation_hit": sum(1 for s in answered if s.citation_hit),
        "citation_hit_of": len(answered),
        "citation_recall_mean": _mean(recall),
        "answers_with_an_invalid_citation": sum(
            1 for s in scores if s.answered and s.invalid_citations > 0
        ),
        "answers": sum(1 for s in scores if s.answered),
    }


def summarize_scores(scores: Sequence[AnswerScores]) -> dict[str, Any]:
    """The checks per kind of item, per split and overall (counts next to every rate).

    The groups are ``all``, each item type and each split (``dev``, ``test``): the plan's tuning rule
    (D-065) is that anything tuned was tuned on the development items only, so the test items are
    reported on their own.
    """
    summary: dict[str, Any] = {"all": _counts(scores)}
    for kind in ("single_hop", "multi_hop", "multi_turn", "unanswerable"):
        summary[kind] = _counts([s for s in scores if s.type == kind])
    for split in ("dev", "test"):
        summary[split] = _counts([s for s in scores if s.split == split])
    return summary


# --- the report ------------------------------------------------------------------------------


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole}" if whole else "none"


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_answers_report(
    meta: Mapping[str, Any],
    summary: Mapping[str, Any],
    judge: Mapping[str, Any] | None = None,
    kappa: Mapping[str, Any] | None = None,
) -> str:
    """The Markdown report of an answer evaluation: the deterministic checks, then the judge and κ.

    Reads only the dictionaries, so it can be regenerated from the saved files. The judge section is
    labelled supporting evidence and says whether the human labels allow it to be used at all.
    """
    lines: list[str] = []
    add = lines.append
    add(f"# Answer evaluation: {meta.get('name', '?')}")
    add("")
    add(
        f"Chat model `{meta.get('model', '?')}` (profile `{meta.get('profile', '?')}`), retrieval "
        f"hybrid with the cross-encoder, top-k {meta.get('top_k', '?')}; golden v0 "
        "(Claude-authored, provisional: easy mode, never a headline); "
        f"{meta.get('created', '?')}."
    )
    add("")
    add("## Deterministic checks (PLAN.md §9: the primary evidence)")
    add(
        "Counts are shown next to every share. A citation hit says the answer cited a document the "
        "golden item names; it does not say the claim is in it."
    )
    add("")
    add(
        "| group | items | answerable answered | answerable refused wrongly | unanswerable refused | "
        "unanswerable answered | must-include recall | citation hit | citation recall | "
        "answers with an invalid citation |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for key in ("all", "single_hop", "multi_hop", "multi_turn", "unanswerable", "dev", "test"):
        c = summary.get(key)
        if not c or not c["n"]:
            continue
        add(
            f"| {key} | {c['n']} | {_share(c['answerable_answered'], c['answerable'])} | "
            f"{_share(c['answerable_refused'], c['answerable'])} | "
            f"{_share(c['unanswerable_refused'], c['unanswerable'])} | "
            f"{_share(c['unanswerable_answered'], c['unanswerable'])} | "
            f"{_number(c['must_include_mean'])} (n={c['must_include_n']}) | "
            f"{_share(c['citation_hit'], c['citation_hit_of'])} | "
            f"{_number(c['citation_recall_mean'])} | "
            f"{_share(c['answers_with_an_invalid_citation'], c['answers'])} |"
        )
    if judge:
        add("")
        add("## The judge (supporting evidence only)")
        add(
            f"Judge `{judge.get('model', '?')}` (family `{judge.get('family', '?')}`, the generator's "
            f"is `{judge.get('generator_family', '?')}`); {judge.get('judged', 0)} answers judged, "
            f"{judge.get('skipped_refusals', 0)} refusals not judged. **Supported** (every claim is "
            f"stated by the sources): {judge.get('supported', 0)} of {judge.get('judged', 0)}; "
            f"**answers the question**: {judge.get('answers_question', 0)} of {judge.get('judged', 0)}."
        )
        by_type = judge.get("by_type") or {}
        for kind, counts in by_type.items():
            add(
                f"- {kind}: supported {counts.get('supported', 0)} of {counts.get('judged', 0)}, "
                f"answers the question {counts.get('answers_question', 0)} of {counts.get('judged', 0)}"
            )
    if kappa:
        add("")
        add("## Agreement with the human labels (PLAN.md §9)")
        add(f"{kappa.get('verdict', 'no labels')}")
        counts = kappa.get("counts")
        if counts:
            add(
                f"- both supported {counts['both_supported']}, both unsupported "
                f"{counts['both_unsupported']}, human supported and judge not "
                f"{counts['human_supported_judge_not']}, judge supported and human not "
                f"{counts['judge_supported_human_not']}"
            )
    add("")
    add(
        "These are the numbers of one system on a small, easy, provisional set; a deterministic check "
        "is a floor, the judge is a model's opinion, and neither says an answer is true."
    )
    return "\n".join(lines) + "\n"
