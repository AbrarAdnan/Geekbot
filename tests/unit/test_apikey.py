"""Tests for step S4.3: the API key the server asks for and the UI sends (``geekbot.apikey``)."""

from pathlib import Path

import pytest

from geekbot import apikey
from geekbot.apikey import (
    API_KEY_FILE,
    MIN_KEY_CHARS,
    ApiKey,
    ApiKeyError,
    api_key_path,
    check_api_key,
    generate_api_key,
    load_api_key,
    load_or_create_api_key,
)
from geekbot.config import Settings

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

GOOD = "g" * 32
OTHER = "o" * 32


def settings_in(tmp_path: Path, **options: object) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", **options)


def write_key_file(settings: Settings, text: str) -> Path:
    path = api_key_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- the key itself --------------------------------------------------------------------------


def test_a_generated_key_is_long_random_and_usable() -> None:
    keys = {generate_api_key() for _ in range(20)}
    assert len(keys) == 20
    for key in keys:
        assert len(key) >= MIN_KEY_CHARS
        assert check_api_key(key, origin="test") == key


def test_a_key_is_stripped_of_surrounding_whitespace() -> None:
    assert check_api_key(f"  {GOOD}\r\n", origin="test") == GOOD


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "short",
        "x" * (MIN_KEY_CHARS - 1),
        "a" * 30 + " " + "b" * 30,
        '"' + "q" * 30 + '"',
        "k" * 30 + chr(0xE9),
    ],
    ids=["empty", "short", "one-short-of-minimum", "inner-space", "quotes", "non-ascii"],
)
def test_an_unusable_key_is_refused_and_the_error_says_where_it_came_from(bad: str) -> None:
    with pytest.raises(ApiKeyError, match="from the-place"):
        check_api_key(bad, origin="the-place")


def test_the_key_stays_out_of_repr() -> None:
    assert GOOD not in repr(ApiKey(GOOD, "somewhere"))


# --- finding it ------------------------------------------------------------------------------


def test_nothing_is_found_when_nothing_is_set(tmp_path: Path) -> None:
    assert load_api_key(settings_in(tmp_path)) is None


def test_the_setting_wins_over_the_file(tmp_path: Path) -> None:
    settings = settings_in(tmp_path, api_key=GOOD)
    write_key_file(settings, OTHER)
    found = load_api_key(settings)
    assert found is not None
    assert (found.value, found.origin, found.created) == (GOOD, "GEEKBOT_API_KEY", False)


def test_a_blank_setting_counts_as_unset(tmp_path: Path) -> None:
    settings = settings_in(tmp_path, api_key="   ")
    assert load_api_key(settings) is None
    write_key_file(settings, OTHER)
    found = load_api_key(settings)
    assert found is not None and found.value == OTHER


def test_the_file_is_read_whatever_its_line_ending(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    path = write_key_file(settings, GOOD + "\r\n")
    found = load_api_key(settings)
    assert found is not None
    assert (found.value, found.origin) == (GOOD, str(path))


def test_a_bad_setting_is_an_error_not_a_silent_fallback_to_the_file(tmp_path: Path) -> None:
    settings = settings_in(tmp_path, api_key="short")
    write_key_file(settings, OTHER)
    with pytest.raises(ApiKeyError, match="GEEKBOT_API_KEY"):
        load_api_key(settings)


def test_a_bad_file_is_an_error_that_names_the_file(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    path = write_key_file(settings, "oops")
    with pytest.raises(ApiKeyError, match=API_KEY_FILE):
        load_api_key(settings)
    assert path.read_text(encoding="utf-8") == "oops"  # never overwritten behind the user's back


# --- creating it -----------------------------------------------------------------------------


def test_the_first_call_creates_the_data_folder_and_the_file_the_second_reads_it(
    tmp_path: Path,
) -> None:
    settings = settings_in(tmp_path)
    assert not settings.data_dir.exists()
    first = load_or_create_api_key(settings)
    path = api_key_path(settings)
    assert first.created is True
    assert first.origin == str(path)
    assert path.read_text(encoding="utf-8") == first.value + "\n"
    again = load_or_create_api_key(settings)
    assert (again.value, again.created) == (first.value, False)


def test_a_configured_key_creates_no_file(tmp_path: Path) -> None:
    settings = settings_in(tmp_path, api_key=GOOD)
    assert load_or_create_api_key(settings).value == GOOD
    assert not api_key_path(settings).exists()


def test_a_lost_race_reads_the_key_the_winner_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes starting together: ours looked, found nothing, and the other created the file."""
    settings = settings_in(tmp_path)
    write_key_file(settings, GOOD + "\n")
    real = apikey.load_api_key
    calls: list[int] = []

    def blind_the_first_time(given: Settings) -> ApiKey | None:
        calls.append(1)
        return None if len(calls) == 1 else real(given)

    monkeypatch.setattr(apikey, "load_api_key", blind_the_first_time)
    key = load_or_create_api_key(settings)
    assert (key.value, key.created) == (GOOD, False)
    assert len(calls) == 2
