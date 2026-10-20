"""Tests for step S4.3: the path allowlist (``PathPolicy``), the ✍️ piece of the step.

The rules are the six in the docstring of ``PathPolicy.check``; each group below takes one. Links are
the point of the step, and a link cannot always be made: creating a symlink on Windows needs
Developer Mode or an elevated shell. Those tests skip (and say why) when the system refuses, and the
Windows-only junction test, which needs no privilege, runs there. CI's Linux job runs the symlink ones.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from geekbot.config import Settings
from geekbot.ingest.pipeline import collect_files
from geekbot.server.security import (
    UPLOADS_DIR,
    PathNotAllowedError,
    PathPolicy,
    PolicyError,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

NETWORK_PATHS = [
    "\\\\server\\share\\x.txt",
    "//server/share/x.txt",
    "\\\\?\\C:\\x.txt",
    "//./pipe/x",
    "/\\server\\share\\x.txt",  # Windows reads a mix of the two slashes as a network path too
    "\\/server/share/x.txt",
]


@dataclass
class World:
    """A temporary world: ``root`` is allowed, ``outside`` is not, ``data`` is Geekbot's own."""

    base: Path
    root: Path
    outside: Path
    data: Path
    settings: Settings
    policy: PathPolicy


def build(base: Path, **options: Any) -> World:
    base = base.resolve()
    root, outside, data = base / "docs", base / "secret", base / "data"
    for folder in (root, outside, data):
        folder.mkdir()
    (outside / "secret.txt").write_text("the secret", encoding="utf-8")
    configured = Settings(_env_file=None, data_dir=data, allowed_roots=str(root), **options)
    return World(base, root, outside, data, configured, PathPolicy.from_settings(configured))


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build(tmp_path)


def make_file_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks (Windows: needs Developer Mode)")


def make_directory_link(link: Path, target: Path) -> None:
    """A directory link that ``os.walk`` may or may not follow: a junction on Windows, else a symlink."""
    if sys.platform == "win32":
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError:
            pytest.skip("could not create a junction here")
        if not link.is_dir():
            pytest.skip("the junction was not created")
        return
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")


# --- rule 4: inside a root, not outside ------------------------------------------------------


def test_a_file_inside_a_root_is_allowed_and_comes_back_as_its_real_path(world: World) -> None:
    note = world.root / "a.txt"
    note.write_text("x", encoding="utf-8")
    assert world.policy.check(str(note)) == note


def test_a_root_itself_can_be_ingested_whole(world: World) -> None:
    assert world.policy.check(str(world.root)) == world.root


def test_a_file_outside_every_root_is_refused_with_a_403_style_error(world: World) -> None:
    with pytest.raises(PathNotAllowedError) as raised:
        world.policy.check(str(world.outside / "secret.txt"))
    assert raised.value.malformed is False
    message = str(raised.value)
    assert "GEEKBOT_ALLOWED_ROOTS" in message
    assert str(world.root) in message  # the caller is told what it may read


def test_a_folder_whose_name_starts_like_a_root_is_not_inside_it(world: World) -> None:
    sibling = world.base / "docs-private"
    sibling.mkdir()
    with pytest.raises(PathNotAllowedError):
        world.policy.check(str(sibling / "a.txt"))


def test_dot_dot_that_leaves_the_root_is_refused_and_dot_dot_that_stays_is_fine(
    world: World,
) -> None:
    escape = os.path.join(str(world.root), "..", "secret", "secret.txt")
    with pytest.raises(PathNotAllowedError):
        world.policy.check(escape)
    (world.root / "sub").mkdir()
    stays = os.path.join(str(world.root), "sub", "..", "a.txt")
    assert world.policy.check(stays) == world.root / "a.txt"


def test_a_path_that_does_not_exist_inside_a_root_passes_the_check(world: World) -> None:
    """Rule 5: the check says nothing about existence; the route reports a missing path itself."""
    assert world.policy.check(str(world.root / "nope.txt")) == world.root / "nope.txt"


def test_outside_every_root_a_missing_path_and_an_existing_one_are_refused_alike(
    world: World,
) -> None:
    """Rule 5: the answer cannot be used to find out what exists on the disk."""
    messages = []
    for target in (world.outside / "secret.txt", world.outside / "not-there.txt"):
        with pytest.raises(PathNotAllowedError) as raised:
            world.policy.check(str(target))
        messages.append(str(raised.value).replace(str(target), "<path>"))
    assert messages[0] == messages[1]


# --- rule 1 and 2: text that is not a usable path --------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "a\x00b", "line\nbreak", "docs/a.txt", "a.txt", "C:relative.txt", *NETWORK_PATHS],
    ids=[
        "empty",
        "blank",
        "nul",
        "newline",
        "relative",
        "bare-name",
        "drive-relative",
        "unc-backslashes",
        "unc-slashes",
        "device",
        "pipe",
        "unc-mixed",
        "unc-mixed-other-way",
    ],
)
def test_text_that_is_not_a_usable_path_is_malformed(world: World, raw: str) -> None:
    with pytest.raises(PathNotAllowedError) as raised:
        world.policy.check(raw)
    assert raised.value.malformed is True


def test_something_that_is_not_text_is_malformed(world: World) -> None:
    with pytest.raises(PathNotAllowedError) as raised:
        world.policy.check(None)  # type: ignore[arg-type]
    assert raised.value.malformed is True


def test_a_colon_after_the_drive_letter_is_malformed(world: World) -> None:
    """An NTFS alternate data stream (``file.txt:stream``) is another way to name a file."""
    with pytest.raises(PathNotAllowedError) as raised:
        world.policy.check(str(world.root / "a.txt") + ":stream")
    assert raised.value.malformed is True


def test_a_network_path_is_refused_before_the_disk_is_touched(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2: resolving ``\\\\server\\share`` would make Windows connect to that server."""

    def forbidden(path: Any) -> str:
        raise AssertionError(f"the disk was consulted for {path!r}")

    monkeypatch.setattr("geekbot.server.security.os.path.realpath", forbidden)
    for raw in NETWORK_PATHS:
        with pytest.raises(PathNotAllowedError):
            world.policy.check(raw)


# --- the data folder, the uploads folder -----------------------------------------------------


def test_the_data_folder_is_never_readable_even_inside_a_root(tmp_path: Path) -> None:
    everything = build(tmp_path)
    wide = Settings(_env_file=None, data_dir=everything.data, allowed_roots=str(everything.base))
    policy = PathPolicy.from_settings(wide)
    (everything.base / "other.txt").write_text("x", encoding="utf-8")
    assert policy.check(str(everything.base / "other.txt")) == everything.base / "other.txt"
    for target in (everything.data, everything.data / "threads.db", everything.data / "api-key"):
        with pytest.raises(PathNotAllowedError, match="data folder"):
            policy.check(str(target))


def test_the_uploads_folder_inside_the_data_folder_is_readable(world: World) -> None:
    upload = world.data / UPLOADS_DIR / "0123456789abcdef-a.txt"
    assert world.policy.check(str(upload)) == upload  # allowed although it does not exist yet
    with pytest.raises(PathNotAllowedError):
        world.policy.check(str(world.data / "threads.db"))


def test_with_no_roots_only_uploads_are_readable(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    bare = Settings(_env_file=None, data_dir=base / "data")
    policy = PathPolicy.from_settings(bare)
    (base / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(PathNotAllowedError):
        policy.check(str(base / "a.txt"))
    uploaded = base / "data" / UPLOADS_DIR / "x.txt"
    assert policy.check(str(uploaded)) == uploaded


# --- rule 3 and 6: links ---------------------------------------------------------------------


def test_a_file_link_that_leaves_the_root_is_refused(world: World) -> None:
    link = world.root / "innocent.txt"
    make_file_link(link, world.outside / "secret.txt")
    with pytest.raises(PathNotAllowedError):
        world.policy.check(str(link))


def test_a_path_through_a_directory_link_that_leaves_the_root_is_refused(world: World) -> None:
    link = world.root / "linked"
    make_directory_link(link, world.outside)
    with pytest.raises(PathNotAllowedError):
        world.policy.check(str(link / "secret.txt"))


def test_a_link_that_stays_inside_comes_back_as_its_target(world: World) -> None:
    real = world.root / "real.txt"
    real.write_text("x", encoding="utf-8")
    alias = world.root / "alias.txt"
    make_file_link(alias, real)
    assert world.policy.check(str(alias)) == real


# --- folder walks ----------------------------------------------------------------------------


def test_split_keeps_real_paths_without_repeats_in_the_order_found(world: World) -> None:
    (world.root / "sub").mkdir()
    first, second = world.root / "a.txt", world.root / "b.txt"
    for file in (first, second):
        file.write_text("x", encoding="utf-8")
    sneaky = world.root / "sub" / ".." / "a.txt"
    admitted, refused = world.policy.split([first, second, first, sneaky])
    assert admitted == [first, second]
    assert refused == []


def test_split_refuses_a_file_link_that_leaves_the_root_and_keeps_the_rest(world: World) -> None:
    good = world.root / "good.txt"
    good.write_text("x", encoding="utf-8")
    leak = world.root / "leak.txt"
    make_file_link(leak, world.outside / "secret.txt")
    admitted, refused = world.policy.split([good, leak])
    assert admitted == [good]
    assert refused == [leak]  # reported as the walk gave it


def test_a_walk_never_admits_a_file_behind_a_directory_link(world: World) -> None:
    """``os.walk`` does not follow a symlink to a directory but does follow a Windows junction.

    Either way no file outside the root may be admitted: on Linux the walk never lists it, on
    Windows it lists it and the policy refuses it.
    """
    (world.root / "fine.txt").write_text("x", encoding="utf-8")
    make_directory_link(world.root / "linked", world.outside)
    found, _unsupported = collect_files([world.root])
    admitted, _refused = world.policy.split(found)
    assert world.root / "fine.txt" in admitted
    assert all(path.name != "secret.txt" for path in admitted)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions exist on Windows only")
def test_on_windows_the_walk_does_list_the_files_behind_a_junction_and_the_policy_refuses_them(
    world: World,
) -> None:
    """The reason ``split`` exists: the first check cannot see inside the folder."""
    make_directory_link(world.root / "linked", world.outside)
    found, _unsupported = collect_files([world.root])
    assert any(path.name == "secret.txt" for path in found)  # os.walk followed the junction
    admitted, refused = world.policy.split(found)
    assert all(path.name != "secret.txt" for path in admitted)
    assert any(path.name == "secret.txt" for path in refused)


# --- building the policy from the settings ---------------------------------------------------


def test_the_roots_are_read_from_the_setting_in_order_with_the_platform_separator(
    tmp_path: Path,
) -> None:
    base = tmp_path.resolve()
    first, second = base / "one", base / "two"
    first.mkdir()
    second.mkdir()
    joined = os.pathsep.join([str(first), "", f" {second} ", ""])
    configured = Settings(_env_file=None, data_dir=base / "data", allowed_roots=joined)
    assert configured.allowed_root_paths == (first, second)
    policy = PathPolicy.from_settings(configured)
    assert policy.roots == (first, second)
    assert policy.uploads == base / "data" / UPLOADS_DIR
    assert policy.forbidden == (base / "data",)
    assert str(policy.uploads) in policy.describe()


def test_no_roots_describe_only_the_uploads_folder(tmp_path: Path) -> None:
    policy = PathPolicy.from_settings(Settings(_env_file=None, data_dir=tmp_path / "data"))
    assert policy.roots == ()
    assert policy.describe() == str(policy.uploads)
    assert PathPolicy().describe() == "nowhere"


@pytest.mark.parametrize("bad", ["relative/folder", "docs"], ids=["relative-path", "bare-name"])
def test_a_root_that_is_not_a_full_path_is_a_configuration_error(tmp_path: Path, bad: str) -> None:
    configured = Settings(_env_file=None, data_dir=tmp_path / "data", allowed_roots=bad)
    with pytest.raises(PolicyError, match="not a full path"):
        PathPolicy.from_settings(configured)


def test_a_root_that_does_not_exist_or_is_a_file_is_a_configuration_error(tmp_path: Path) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("x", encoding="utf-8")
    for bad in (tmp_path / "missing", a_file):
        configured = Settings(_env_file=None, data_dir=tmp_path / "data", allowed_roots=str(bad))
        with pytest.raises(PolicyError, match="not a folder that exists"):
            PathPolicy.from_settings(configured)


# --- the property: a model of "inside the root" ----------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],  # the fixture is never changed
)
@given(segments=st.lists(st.sampled_from(["a", "b", "..", "."]), max_size=8))
def test_a_path_is_allowed_exactly_when_it_stays_inside_the_root(
    world: World, segments: list[str]
) -> None:
    raw = os.path.join(str(world.root), *segments)
    normalized = Path(os.path.normpath(raw))
    if normalized.is_relative_to(world.root):
        assert world.policy.check(raw) == normalized
    else:
        with pytest.raises(PathNotAllowedError):
            world.policy.check(raw)
