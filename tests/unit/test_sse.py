"""Tests for step S4.2's SSE module: writing frames, reading them back, and the traps in between.

``format_event`` is the step's ✍️ piece (written by Claude in prototyping mode, D-047). Its contract
is the six rules in its docstring; the tests below take them in order, then the parser, then
properties that hold for any JSON value: what is written is what is read, and no line splitter can cut
a frame.

Characters with a special meaning are built with ``chr`` so that the test file contains no invisible
control character.
"""

import json
from collections.abc import Iterator

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.sse import SseEvent, format_event, parse_events, parse_text

NEL = chr(0x85)  # next line
LINE_SEPARATOR = chr(0x2028)
PARAGRAPH_SEPARATOR = chr(0x2029)
LONE_SURROGATE = chr(0xD800)  # a text no UTF-8 encoder accepts

JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=12,
)
EVENT_NAMES = st.from_regex(r"[a-z][a-z0-9_-]{0,31}", fullmatch=True)
# Text that is mostly the troublesome characters, so the property tests meet them every time.
AWKWARD_TEXT = st.text(
    alphabet=st.sampled_from(
        [
            "a",
            " ",
            "\n",
            "\r",
            chr(0x0B),
            chr(0x0C),
            chr(0x1C),
            NEL,
            LINE_SEPARATOR,
            PARAGRAPH_SEPARATOR,
        ]
    )
)


# --- format_event: the six rules -----------------------------------------------------------------


def test_a_frame_is_an_event_line_a_data_line_and_a_blank_line() -> None:
    assert format_event("text", {"text": "hi"}) == 'event: text\ndata: {"text":"hi"}\n\n'


def test_the_data_is_compact_json_on_one_line() -> None:
    frame = format_event("done", {"a": [1, 2, {"b": None}], "c": True, "d": 1.5})
    assert frame == 'event: done\ndata: {"a":[1,2,{"b":null}],"c":true,"d":1.5}\n\n'


@pytest.mark.parametrize(
    "name", ["", "Text", "two words", "a\nb", "x:y", "1abc", "-a", "a" * 33, "text\n", "café"]
)
def test_an_event_name_that_could_break_the_frame_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not a usable event name"):
        format_event(name, {})


@pytest.mark.parametrize("name", ["a", "text", "done", "some_name", "some-name", "a" * 32, "x1"])
def test_ordinary_event_names_are_accepted(name: str) -> None:
    assert format_event(name, 1).startswith(f"event: {name}\n")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_values_that_are_not_json_are_refused(number: float) -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        format_event("text", {"n": number})


def test_a_value_json_cannot_hold_raises_a_type_error() -> None:
    with pytest.raises(TypeError):
        format_event("text", {"n": {1, 2}})


def test_non_ascii_text_is_written_as_utf8_not_as_escapes() -> None:
    frame = format_event("text", {"text": "日本語 café"})
    assert "日本語 café" in frame
    assert "\\u" not in frame


def test_a_newline_in_the_text_travels_inside_the_json_string() -> None:
    frame = format_event("text", {"text": "line one\nline two\r\nline three"})
    assert frame.count("\n") == 3  # the end of the event line, of the data line, and the blank line
    assert "\r" not in frame
    assert "line one\\nline two\\r\\nline three" in frame


@pytest.mark.parametrize("char", [NEL, LINE_SEPARATOR, PARAGRAPH_SEPARATOR])
def test_the_line_breaks_json_leaves_alone_are_escaped(char: str) -> None:
    frame = format_event("text", {"text": f"before{char}after"})
    assert char not in frame
    assert len(frame.splitlines()) == 3  # what httpx's iter_lines and str.splitlines would see
    assert parse_text(frame) == [SseEvent("text", {"text": f"before{char}after"})]


def test_a_text_no_utf8_encoder_accepts_is_written_in_ascii_instead() -> None:
    value = {"text": f"a{LONE_SURROGATE}b 日"}
    frame = format_event("text", value)
    assert frame.encode("utf-8")  # does not raise: the server can send it
    assert frame.isascii()
    assert parse_text(frame) == [SseEvent("text", value)]


# --- parse_events / parse_text -------------------------------------------------------------------


def test_two_frames_are_read_in_order() -> None:
    text = format_event("text", {"text": "a"}) + format_event("done", {"ok": True})
    assert parse_text(text) == [SseEvent("text", {"text": "a"}), SseEvent("done", {"ok": True})]


def test_a_frame_with_crlf_line_endings_is_read() -> None:
    text = 'event: text\r\ndata: {"text":"a"}\r\n\r\n'
    assert parse_text(text) == [SseEvent("text", {"text": "a"})]


def test_a_lone_carriage_return_is_a_line_ending_too() -> None:
    assert parse_text('event: text\rdata: {"text":"a"}\r\r') == [SseEvent("text", {"text": "a"})]


def test_comments_are_skipped_and_the_space_after_the_colon_is_optional() -> None:
    text = ': keep-alive\n\nevent:text\ndata:{"n":1}\n\n'
    assert parse_text(text) == [SseEvent("text", {"n": 1})]


def test_several_data_lines_are_joined_with_a_newline() -> None:
    assert parse_text("data: [1,\ndata: 2]\n\n") == [SseEvent("message", [1, 2])]


def test_a_frame_without_an_event_line_is_called_message() -> None:
    assert parse_text('data: "x"\n\n') == [SseEvent("message", "x")]


def test_unknown_fields_are_ignored() -> None:
    text = 'id: 7\nretry: 100\nflavour: odd\nevent: text\ndata: {"text":"a"}\n\n'
    assert parse_text(text) == [SseEvent("text", {"text": "a"})]


def test_a_frame_with_no_data_delivers_nothing_and_does_not_leak_its_name() -> None:
    text = 'event: ghost\n\ndata: "x"\n\n'
    assert parse_text(text) == [SseEvent("message", "x")]


def test_a_frame_still_open_when_the_stream_ends_is_not_delivered() -> None:
    complete = format_event("text", {"text": "a"})
    assert parse_text(complete + 'event: text\ndata: {"text":"b"}') == [
        SseEvent("text", {"text": "a"})
    ]


def test_data_that_is_not_json_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="property name|Expecting"):
        parse_text("data: {broken\n\n")


def test_parse_events_reads_lazily_from_any_iterable_of_lines() -> None:
    seen: list[str] = []

    def lines() -> Iterator[str]:
        for line in ["event: text", 'data: {"n":1}', "", "event: text", 'data: {"n":2}', ""]:
            seen.append(line)
            yield line

    first = next(parse_events(lines()))
    assert first.data == {"n": 1}
    assert len(seen) == 3  # it stopped at the blank line, without reading the rest


# --- properties ----------------------------------------------------------------------------------


@given(EVENT_NAMES, JSON_VALUES)
def test_what_is_written_is_what_is_read(name: str, value: object) -> None:
    assert parse_text(format_event(name, value)) == [SseEvent(name, value)]


@given(EVENT_NAMES, st.lists(JSON_VALUES, min_size=1, max_size=5))
def test_a_stream_of_frames_is_read_back_frame_for_frame(name: str, values: list[object]) -> None:
    stream = "".join(format_event(name, value) for value in values)
    assert parse_text(stream) == [SseEvent(name, value) for value in values]


@given(EVENT_NAMES, AWKWARD_TEXT)
def test_no_line_splitter_can_cut_a_frame(name: str, text: str) -> None:
    """``str.splitlines`` is the widest splitter in common use: if it sees three lines, all do."""
    frame = format_event(name, {"text": text})
    lines = frame.splitlines()
    assert len(lines) == 3
    assert lines[0] == f"event: {name}"
    assert lines[1].startswith("data: ")
    assert lines[2] == ""
    assert frame.endswith("\n\n")
    assert json.loads(lines[1].removeprefix("data: ")) == {"text": text}


@given(EVENT_NAMES, AWKWARD_TEXT)
def test_awkward_text_survives_the_round_trip(name: str, text: str) -> None:
    assert parse_text(format_event(name, {"text": text})) == [SseEvent(name, {"text": text})]
