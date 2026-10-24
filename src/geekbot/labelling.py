"""Human labels for the answer judge, and Cohen's kappa (step S5.2, PLAN.md §9).

PLAN §9: the judge is a model, models are biased (a judge prefers its own family's writing), and an
unchecked judge is a number nobody should quote. So the plan asks for **~60 answers, shuffled and
labelled blind by a person**, and a rule: **if κ < 0.6 the judge is excluded from headline claims**.
This module is the part of that which both the evaluation (``geekbot.evals``) and the labelling page
of the UI (``geekbot.ui``) need. It sits outside both packages because the UI may import neither
``geekbot.evals`` nor the storage (the import-linter contract of S4.4), and a person labelling answers
needs no model and no database, only the order, a file to write to, and a number at the end.

What a label is: one yes/no question about one answer, the one the judge is asked too: **is every claim
in the answer supported by the sources it was shown?** One binary question keeps an hour of labelling
to about an hour, and gives a κ that means one thing.

**Blind** means the person never sees the judge's verdict, the model's name or the item's id, and the
order is a fixed shuffle (a hash of the item id and a seed) that tells nothing about the kind of
question. Nothing here reads the judge's verdicts: it cannot leak them.

**The labels are the user's own work.** Claude never writes labels (D-097); this code can only record
what a person clicked. A label file is append-only JSON Lines, the latest label of an item wins, so a
mistake is corrected by labelling again, and nothing is ever rewritten.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

__all__ = [
    "BLIND_SEED",
    "KAPPA_THRESHOLD",
    "LABEL_COUNT",
    "MIN_LABELS_FOR_KAPPA",
    "Label",
    "LabelStore",
    "agreement_counts",
    "blind_order",
    "cohens_kappa",
    "kappa_verdict",
    "pick_for_labelling",
]

BLIND_SEED: Final = "geekbot-label-order-1"
LABEL_COUNT: Final = 60  # PLAN.md §9: about 60 answers, about an hour
KAPPA_THRESHOLD: Final = 0.6  # PLAN.md §9: below this the judge is excluded from headline claims
# Fewer labels than this give a κ that a single click can move by more than the threshold's width, so
# no verdict is given (the plan asks for 60; 30 is where the report stops saying "not enough yet").
MIN_LABELS_FOR_KAPPA: Final = 30


def blind_order(ids: Sequence[str], *, seed: str = BLIND_SEED) -> list[str]:
    """``ids`` in a fixed shuffle: sorted by the hash of ``seed``, a separator and the id.

    The same ids and seed always give the same order, whatever order they come in, so the page and
    the report agree on which answers are the first sixty without sharing any state. The order carries
    no information about the item (an id like ``S-004`` does not sort before ``U-001``).
    """
    return sorted(
        dict.fromkeys(ids),
        key=lambda item_id: hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest(),
    )


def pick_for_labelling(
    ids: Sequence[str], *, count: int = LABEL_COUNT, seed: str = BLIND_SEED
) -> list[str]:
    """The first ``count`` of the blind order: the answers a person is asked to label."""
    return blind_order(ids, seed=seed)[: max(count, 0)]


@dataclass(frozen=True)
class Label:
    """One human judgement of one answer."""

    run: str  # the evaluation run the answer belongs to (answers of different runs are different)
    item_id: str
    supported: bool  # every claim in the answer is supported by the sources it was shown
    note: str = ""
    at: str = ""  # when the person clicked, ISO time (set by ``LabelStore.add``)


class LabelStore:
    """An append-only JSON Lines file of labels for one run.

    ``labels()`` reads the whole file and keeps the latest label of each item of this run, so a
    wrong click is fixed by clicking again. A line that is not a label (a half-written line after a
    crash, a hand edit) is skipped, never fatal: an hour of someone's labelling must not be lost to
    one bad line.
    """

    def __init__(self, path: Path, run: str) -> None:
        self.path = path
        self.run = run

    def add(self, item_id: str, supported: bool, note: str = "") -> Label:
        label = Label(
            self.run,
            item_id,
            supported,
            note.strip()[:500],
            datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "run": label.run,
                "id": label.item_id,
                "supported": label.supported,
                "note": label.note,
                "at": label.at,
            },
            ensure_ascii=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        return label

    def labels(self) -> dict[str, Label]:
        """``{item id: its latest label}`` for this run."""
        found: dict[str, Label] = {}
        if not self.path.is_file():
            return found
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                data = json.loads(line)
                if data["run"] != self.run:
                    continue
                supported = data["supported"]
                if not isinstance(supported, bool):
                    continue
                found[str(data["id"])] = Label(
                    self.run,
                    str(data["id"]),
                    supported,
                    str(data.get("note", "")),
                    str(data.get("at", "")),
                )
            except (ValueError, KeyError, TypeError):
                continue
        return found

    def remaining(self, order: Sequence[str]) -> list[str]:
        """The ids of ``order`` that have no label yet, in order."""
        done = self.labels()
        return [item_id for item_id in order if item_id not in done]


def cohens_kappa(human: Sequence[bool], judge: Sequence[bool]) -> float | None:
    """Cohen's κ between two yes/no raters over the same items.

    ✍️ Written by Claude in prototyping mode (D-047, step S5.2); on the backlog in STATUS.md.

    κ = (p_o - p_e) / (1 - p_e): ``p_o`` is the share of items they agree on, ``p_e`` the share they
    would agree on by chance given how often each says yes (``p_e = a·b + (1-a)(1-b)`` with ``a`` and
    ``b`` the two yes-rates). 1 is perfect agreement, 0 is what two raters with these yes-rates would
    reach by guessing, below 0 is worse than guessing. Why not plain agreement: a judge that says
    "supported" every time agrees with a person on 90% of answers when 90% are supported, and has
    told us nothing.

    ``None`` when κ is undefined: no items, or ``p_e == 1`` (both raters said the same thing for every
    item, so there is nothing to agree *beyond*). ``ValueError`` when the lists differ in length.
    """
    if len(human) != len(judge):
        raise ValueError(f"the raters rated {len(human)} and {len(judge)} items; they must match")
    n = len(human)
    if n == 0:
        return None
    observed = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    human_yes = sum(human) / n
    judge_yes = sum(judge) / n
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def agreement_counts(human: Sequence[bool], judge: Sequence[bool]) -> dict[str, int]:
    """The 2x2 table behind κ: both yes, both no, human yes and judge no, human no and judge yes."""
    if len(human) != len(judge):
        raise ValueError(f"the raters rated {len(human)} and {len(judge)} items; they must match")
    pairs = list(zip(human, judge, strict=True))
    return {
        "both_supported": sum(1 for h, j in pairs if h and j),
        "both_unsupported": sum(1 for h, j in pairs if not h and not j),
        "human_supported_judge_not": sum(1 for h, j in pairs if h and not j),
        "judge_supported_human_not": sum(1 for h, j in pairs if not h and j),
    }


def kappa_verdict(
    kappa: float | None,
    n: int,
    *,
    threshold: float = KAPPA_THRESHOLD,
    minimum_n: int = MIN_LABELS_FOR_KAPPA,
) -> str:
    """The sentence the report prints about the judge, by the rule of PLAN.md §9.

    Too few labels: no verdict (and how many are missing). κ undefined: no verdict. κ at or above
    the threshold: the judge may be used for supporting evidence (the plan keeps deterministic
    metrics and human labels as the primary evidence either way). Below: **excluded from headline
    claims**.
    """
    if n < minimum_n:
        return (
            f"no verdict yet: {n} labels, and at least {minimum_n} are needed "
            f"(the plan asks for {LABEL_COUNT})"
        )
    if kappa is None:
        return (
            "no verdict: κ is undefined (the judge and the labels said the same thing every time)"
        )
    if kappa >= threshold:
        return (
            f"κ = {kappa:.2f} over {n} labels, at or above {threshold}: the judge may be used as "
            "supporting evidence (deterministic checks and human labels stay primary)"
        )
    return (
        f"κ = {kappa:.2f} over {n} labels, below {threshold}: the judge is EXCLUDED from headline "
        "claims (PLAN.md §9)"
    )
