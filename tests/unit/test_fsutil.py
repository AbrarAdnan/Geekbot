"""Tests for ``geekbot.fsutil.replace_file``: a rename that is patient when Windows says "access denied".

``os.replace`` is replaced where the module looks it up, so no real file is ever locked: what is
checked is when it is tried again, how long it waits, and what is never retried.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from geekbot import fsutil
from geekbot.fsutil import REPLACE_ATTEMPTS, replace_file


class Flaky:
    """An ``os.replace`` that refuses ``refusals`` times with ``error`` and then really renames."""

    def __init__(self, refusals: int, error: Exception | None = None) -> None:
        self.refusals = refusals
        self.error = error or PermissionError(5, "Access is denied")
        self.calls = 0

    def __call__(self, source: str, target: str) -> None:
        self.calls += 1
        if self.calls <= self.refusals:
            raise self.error
        os.replace(source, target)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make ``fsutil`` see this as its ``os`` (the real ``os`` module is left alone)."""
        monkeypatch.setattr(fsutil, "os", SimpleNamespace(replace=self))


def make_files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "new.json.part"
    target = tmp_path / "new.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    return source, target


def test_a_plain_replace_overwrites_the_target(tmp_path: Path) -> None:
    source, target = make_files(tmp_path)
    replace_file(source, target)
    assert target.read_text(encoding="utf-8") == "new" and not source.exists()


def test_a_refusal_is_tried_again_after_a_growing_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = make_files(tmp_path)
    flaky = Flaky(refusals=3)
    pauses: list[float] = []
    flaky.install(monkeypatch)
    replace_file(source, target, sleep=pauses.append, retry=True, pause_s=0.5)
    assert flaky.calls == 4
    assert pauses == [0.5, 1.0, 1.5]  # n times the pause before attempt n + 1
    assert target.read_text(encoding="utf-8") == "new"


def test_a_refusal_that_does_not_go_away_is_raised_after_all_the_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = make_files(tmp_path)
    flaky = Flaky(refusals=10_000)
    pauses: list[float] = []
    flaky.install(monkeypatch)
    with pytest.raises(PermissionError, match="Access is denied"):
        replace_file(source, target, sleep=pauses.append, retry=True)
    assert flaky.calls == REPLACE_ATTEMPTS
    assert len(pauses) == REPLACE_ATTEMPTS - 1  # no pause after the last attempt
    assert target.read_text(encoding="utf-8") == "old"  # and the old file is untouched


@pytest.mark.parametrize("error", [FileNotFoundError(2, "no such file"), OSError(28, "disk full")])
def test_any_other_error_is_raised_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    source, target = make_files(tmp_path)
    flaky = Flaky(refusals=1, error=error)
    flaky.install(monkeypatch)
    with pytest.raises(type(error)):
        replace_file(source, target, sleep=lambda _: None, retry=True)
    assert flaky.calls == 1


def test_off_windows_a_permission_error_is_a_real_one_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = make_files(tmp_path)
    flaky = Flaky(refusals=1)
    flaky.install(monkeypatch)
    with pytest.raises(PermissionError):
        replace_file(source, target, sleep=lambda _: None, retry=False)
    assert flaky.calls == 1


def test_at_least_one_attempt_is_made_whatever_is_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = make_files(tmp_path)
    flaky = Flaky(refusals=0)
    flaky.install(monkeypatch)
    replace_file(source, target, attempts=0, sleep=lambda _: None, retry=True)
    assert flaky.calls == 1 and target.read_text(encoding="utf-8") == "new"
