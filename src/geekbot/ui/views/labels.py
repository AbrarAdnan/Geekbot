"""The labelling page (step S5.2): a person labels answers, blind, to calibrate the judge.

PLAN.md §9 asks for about 60 answers, **shuffled and labelled blind**, so that the judge's agreement with
a person can be measured (Cohen's κ, ``geekbot.labelling``) and the judge dropped from headline claims
if it is below 0.6. This page is where the person does it: one answer at a time, with the question
and the sources the model was shown, and two buttons.

**Blind** here means this page only ever reads ``answers.json`` (the answers and their sources). It does
not read the judge's file, the item's id or type is never shown, and the order is the fixed shuffle of
``labelling.pick_for_labelling`` (the same one the report uses, so the page and the report agree on which
sixty answers count). Refusals are not offered: "I could not find it" claims nothing to support.

**One question** is asked about every answer: *is every claim in it supported by the sources shown?*
Click "Every claim is supported" or "Something is not supported" (a wrong click is fixed with "Label the
last one again"). Labels are appended to ``evals/labels/labels.jsonl``, which is the person's own work:
nothing else in the project writes it.

**These are files, not storage** (as for the evals page, S4.4): the page reads and writes files in the
repository folder and imports no storage package. The answers are model text built from other people's
documents, so they are shown as plain text, never as Markdown (an image in one must not load).

A button changes a file only inside ``if st.button(...)``, and the page reruns right after: Streamlit
reruns the script on every interaction, and a write outside a click would repeat on every rerun (the
lesson of v1 bug #1, as on the knowledge page).
"""

import json
from pathlib import Path
from typing import Any, Final

import streamlit as st

from geekbot.labelling import LABEL_COUNT, LabelStore, pick_for_labelling
from geekbot.ui.client import GeekbotClient

__all__ = ["items_to_label", "list_runs", "render"]

RESULTS_DIR: Final = Path("evals/results")
LABELS_FILE: Final = Path("evals/labels/labels.jsonl")
MAX_ANSWERS_BYTES: Final = 20_000_000
REDO_KEY: Final = "labels_redo"


def list_runs(root: Path) -> list[str]:
    """The names of the result folders that hold an ``answers.json``, sorted."""
    if not root.is_dir():
        return []
    return sorted(folder.name for folder in root.iterdir() if (folder / "answers.json").is_file())


def read_answers(root: Path, run: str) -> list[dict[str, Any]]:
    """The records of ``<root>/<run>/answers.json``; ``ValueError`` saying why not.

    The real path must stay inside ``root`` (a link that points elsewhere is refused) and the file
    must not be absurdly large.
    """
    real_root = root.resolve()
    real = (root / run / "answers.json").resolve()
    if not real.is_relative_to(real_root):
        raise ValueError(f"{run} is outside the results folder")
    if real.stat().st_size > MAX_ANSWERS_BYTES:
        raise ValueError(f"{run}/answers.json is too large to label from")
    try:
        data = json.loads(real.read_text(encoding="utf-8"))
        records = data["records"]
        if not isinstance(records, list):
            raise TypeError("records is not a list")
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{run}/answers.json is not an answers file: {error}") from error
    return [record for record in records if isinstance(record, dict)]


def items_to_label(records: list[dict[str, Any]], count: int = LABEL_COUNT) -> list[dict[str, Any]]:
    """The answers a person is asked about, in the blind order: no refusals, ``count`` of them."""
    answered = {str(r["id"]): r for r in records if not r.get("abstained") and r.get("answer")}
    return [answered[item_id] for item_id in pick_for_labelling(list(answered), count=count)]


def _show(record: dict[str, Any]) -> None:
    """One answer: the question, the answer and the sources, all as plain text."""
    earlier = record.get("earlier_turns") or []
    if earlier:
        st.caption("Earlier in this conversation the user asked:")
        for turn in earlier:
            st.text(f"- {turn}")
    st.caption("Question")
    st.text(str(record.get("question", "")))
    st.caption("Answer")
    st.text(str(record.get("answer", "")))
    st.caption("The sources the answer was written from")
    for source in record.get("sources", []):
        title = f"[{source.get('n')}] {source.get('path', '')}"
        with st.expander(title):
            st.text(str(source.get("text", "")))


def render(client: GeekbotClient) -> None:
    del client  # this page reads and writes files, not the server
    st.title("Label answers")
    st.caption(
        "Is every claim in the answer supported by the sources shown? Your labels calibrate the "
        "judge (PLAN.md section 9): about sixty, in a fixed shuffle, and you see nothing the judge said."
    )
    runs = list_runs(RESULTS_DIR)
    if not runs:
        st.info(
            f"No answers to label in `{RESULTS_DIR.resolve()}`. Run `geekbot eval answers --phase "
            "generate` first (it needs the GPU), then come back."
        )
        return
    run = st.selectbox("Evaluation run", runs)
    try:
        todo = items_to_label(read_answers(RESULTS_DIR, str(run)))
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    store = LabelStore(LABELS_FILE, str(run))
    done = store.labels()
    ids = [str(record["id"]) for record in todo]
    labelled = sum(1 for item_id in ids if item_id in done)
    st.progress(labelled / len(ids) if ids else 0.0, text=f"{labelled} of {len(ids)} labelled")
    if not ids:
        st.info("This run has no answers to label (every answer was a refusal).")
        return

    redo = st.session_state.get(REDO_KEY)
    remaining = [item_id for item_id in ids if item_id not in done]
    current = redo if redo in ids else (remaining[0] if remaining else None)
    if current is None:
        st.success(
            f"All {len(ids)} answers are labelled. Thank you: `geekbot eval answers --phase "
            "report` now compares the judge with your labels."
        )
        _undo_button(ids, done)
        return
    record = next(r for r in todo if str(r["id"]) == current)
    _show(record)
    if st.button("Every claim is supported", type="primary"):
        store.add(current, True)
        st.session_state.pop(REDO_KEY, None)
        st.rerun()
    if st.button("Something is not supported"):
        store.add(current, False)
        st.session_state.pop(REDO_KEY, None)
        st.rerun()
    _undo_button(ids, done)


def _undo_button(ids: list[str], done: dict[str, Any]) -> None:
    """Offer to show the most recently labelled answer again (a wrong click is fixed by labelling again)."""
    latest = max((done[i] for i in ids if i in done), key=lambda label: label.at, default=None)
    if latest is not None and st.button("Label the last one again"):
        st.session_state[REDO_KEY] = latest.item_id
        st.rerun()
