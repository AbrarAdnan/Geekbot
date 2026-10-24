"""Tests for step S5.2: the labelling page, run headless with Streamlit's own ``AppTest``.

The properties that matter are the ones the plan names: the person labels **blind** (the page never
reads or shows the judge's verdicts, the item's id or its kind), refusals are not offered, the order is
the fixed shuffle of ``geekbot.labelling``, a label is written only by a click and appended, and the
answers are shown as plain text.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from geekbot.labelling import LabelStore, pick_for_labelling
from geekbot.ui.views import labels

RUN = "run1"
JUDGE_MARKER = "JUDGE-SAID-THIS-9f3a"


def answer_record(n: int, *, abstained: bool = False, text: str | None = None) -> dict[str, Any]:
    return {
        "id": f"S-{n:03d}",
        "type": "single_hop",
        "split": "dev",
        "question": f"Question number {n}?",
        "earlier_turns": [],
        "answer": text or f"Answer number {n} [1].",
        "abstained": abstained,
        "sources": [{"n": 1, "path": f"docs/file{n}.md", "text": f"Source text of {n}."}],
    }


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "results"
    (root / RUN).mkdir(parents=True)
    records = [answer_record(n) for n in range(1, 7)]
    records.append(answer_record(7, abstained=True, text="I could not find it in the documents."))
    (root / RUN / "answers.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    # The judge's file sits next to it; the page must never read it.
    (root / RUN / "judge.json").write_text(
        json.dumps({"verdicts": {"S-001": {"reason": JUDGE_MARKER}}}), encoding="utf-8"
    )
    monkeypatch.setattr(labels, "RESULTS_DIR", root)
    monkeypatch.setattr(labels, "LABELS_FILE", tmp_path / "labels" / "labels.jsonl")
    return root


def run_page() -> AppTest:
    script = (
        "import streamlit as st\n"
        "from geekbot.ui.views import labels\n"
        "labels.render(st.session_state['client'])\n"
    )
    app = AppTest.from_string(script, default_timeout=10)
    app.session_state["client"] = object()
    return app.run()


def page_text(app: AppTest) -> str:
    parts: list[str] = []
    for kind in ("text", "caption", "markdown", "title", "success", "info"):
        parts += [str(element.value) for element in getattr(app, kind)]
    parts += [str(element.proto.text) for element in app.get("progress")]  # type: ignore[attr-defined]
    return "\n".join(parts)


def first_in_order() -> str:
    ids = [f"S-{n:03d}" for n in range(1, 7)]  # the refusal (S-007) is never offered
    return pick_for_labelling(ids)[0]


def test_the_page_offers_the_first_answer_of_the_blind_order_and_nothing_else(
    results: Path,
) -> None:
    app = run_page()
    assert not app.exception
    shown = page_text(app)
    number = int(first_in_order().split("-")[1])
    assert f"Question number {number}?" in shown and f"Answer number {number} [1]." in shown
    assert JUDGE_MARKER not in shown  # the judge's verdict is never read
    assert "single_hop" not in shown and first_in_order() not in shown  # no kind, no id
    assert "0 of 6 labelled" in shown


def test_a_refusal_is_not_offered_for_labelling(results: Path) -> None:
    records = labels.read_answers(results, RUN)
    offered = [r["id"] for r in labels.items_to_label(records)]
    assert "S-007" not in offered and len(offered) == 6


def test_the_sources_are_shown_as_plain_text_and_the_answer_is_not_rendered_as_markdown(
    results: Path,
) -> None:
    app = run_page()
    assert any("Source text of" in str(t.value) for t in app.text)
    answers = [str(t.value) for t in app.text if "Answer number" in str(t.value)]
    assert answers  # shown with st.text, which does not interpret Markdown or load an image
    assert not any("Answer number" in str(m.value) for m in app.markdown)


def test_a_click_appends_a_label_and_moves_on_to_the_next_answer(results: Path) -> None:
    app = run_page()
    before = page_text(app)
    app.button[0].click().run()  # "Every claim is supported"
    assert not app.exception
    lines = labels.LABELS_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["id"] == first_in_order() and saved["supported"] is True and saved["run"] == RUN
    after = page_text(app)
    assert after != before and "1 of 6 labelled" in after


def test_the_second_button_labels_the_answer_unsupported(results: Path) -> None:
    app = run_page()
    app.button[1].click().run()
    saved = json.loads(labels.LABELS_FILE.read_text(encoding="utf-8").splitlines()[0])
    assert saved["supported"] is False


def test_a_rerun_without_a_click_writes_nothing(results: Path) -> None:
    app = run_page()
    app.run()
    app.run()
    assert not labels.LABELS_FILE.exists()  # the lesson of v1 bug #1: only a click changes a file


def test_every_answer_is_offered_once_in_the_blind_order(results: Path) -> None:
    app = run_page()
    shown_order: list[str] = []
    for _ in range(6):
        shown = [str(t.value) for t in app.text if str(t.value).startswith("Question number")]
        shown_order.append(shown[0])
        app.button[0].click().run()
    expected = [
        f"Question number {int(i.split('-')[1])}?"
        for i in pick_for_labelling([f"S-{n:03d}" for n in range(1, 7)])
    ]
    assert shown_order == expected
    assert any("All 6 answers are labelled" in str(s.value) for s in app.success)


def test_the_last_label_can_be_given_again_and_the_newer_one_wins(results: Path) -> None:
    app = run_page()
    app.button[0].click().run()  # supported
    redo = [b for b in app.button if b.label == "Label the last one again"]
    assert redo
    redo[0].click().run()
    assert f"Question number {int(first_in_order().split('-')[1])}?" in page_text(app)
    app.button[1].click().run()  # now: not supported
    store = LabelStore(labels.LABELS_FILE, RUN)
    assert store.labels()[first_in_order()].supported is False
    assert len(labels.LABELS_FILE.read_text(encoding="utf-8").splitlines()) == 2  # appended


def test_with_no_answers_the_page_says_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(labels, "RESULTS_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(labels, "LABELS_FILE", tmp_path / "labels.jsonl")
    app = run_page()
    assert not app.exception
    assert any("geekbot eval answers --phase generate" in info.value for info in app.info)


def test_a_damaged_answers_file_is_an_error_message_not_a_crash(results: Path) -> None:
    (results / RUN / "answers.json").write_text("{not json", encoding="utf-8")
    app = run_page()
    assert not app.exception
    assert any("not an answers file" in error.value for error in app.error)


def test_the_helpers_refuse_a_run_outside_the_results_folder(results: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the results folder"):
        labels.read_answers(results, "../escape")


def test_list_runs_finds_only_folders_with_an_answers_file(results: Path) -> None:
    (results / "empty").mkdir()
    (results / "notes.txt").write_text("x", encoding="utf-8")
    assert labels.list_runs(results) == [RUN]
    assert labels.list_runs(results / "nowhere") == []
