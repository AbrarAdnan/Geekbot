"""Tests for step S5.3: the web search tool, its cleaning and its backends. No network: HTTP is a fake
transport, and the DuckDuckGo page is a hand-written one in the documented layout."""

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.config import Settings
from geekbot.tools import WEB_SEARCH, build_tools, web_tool
from geekbot.tools.egress import GatedTool
from geekbot.tools.web import (
    MAX_QUERY_CHARS,
    MAX_RESPONSE_BYTES,
    DuckDuckGoBackend,
    FixtureBackend,
    SearxngBackend,
    WebHit,
    WebQueryError,
    WebSearch,
    WebSearchError,
    clean_hits,
    clean_query,
    make_backend,
)


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


# --- the query -------------------------------------------------------------------------------


def test_a_query_is_one_clean_line() -> None:
    assert clean_query("  how   long\n are \t logs kept? ") == "how long are logs kept?"
    assert clean_query("zero\u200bwidth and bidi\u202e marks") == "zerowidth and bidi marks"


def test_an_empty_or_too_long_query_is_refused_and_never_cut() -> None:
    with pytest.raises(WebQueryError, match="empty"):
        clean_query(" \u200b\n ")
    with pytest.raises(WebQueryError, match="limit"):
        clean_query("x" * (MAX_QUERY_CHARS + 1))
    assert clean_query("x" * MAX_QUERY_CHARS) == "x" * MAX_QUERY_CHARS


@given(st.text())
def test_cleaning_a_query_twice_changes_nothing(text: str) -> None:
    try:
        once = clean_query(text)
    except WebQueryError:
        return
    assert clean_query(once) == once
    assert "\n" not in once and "\u200b" not in once


# --- the results -----------------------------------------------------------------------------


def hit(url: str, title: str = "T", snippet: str = "S") -> WebHit:
    return WebHit(title, url, snippet)


def test_only_public_http_links_survive() -> None:
    kept = clean_hits(
        [
            hit("https://example.org/a"),
            hit("http://example.org/b"),
            hit("javascript:alert(1)"),
            hit("data:text/html,<b>x"),
            hit("file:///etc/passwd"),
            hit("ftp://example.org/x"),
            hit("https://user:pass@example.org/x"),
            hit("http://localhost/x"),
            hit("http://127.0.0.1:8000/x"),
            hit("http://192.168.0.5/x"),
            hit("http://[::1]/x"),
            hit("http://intranet/x"),
            hit("http://printer.local/x"),
            hit("http://a.internal/x"),
            hit("https://example.org/with space"),
            hit(""),
        ],
        limit=20,
    )
    assert [h.url for h in kept] == ["https://example.org/a", "http://example.org/b"]


def test_a_page_listed_twice_is_listed_once_and_fragments_are_dropped() -> None:
    kept = clean_hits(
        [
            hit("https://Example.org/a/#top", "first"),
            hit("https://example.org/a", "second"),
            hit("https://example.org/a?x=1", "other query"),
        ],
        limit=10,
    )
    assert [(h.title, h.url) for h in kept] == [
        ("first", "https://example.org/a/"),
        ("other query", "https://example.org/a?x=1"),
    ]


def test_text_is_cleaned_and_capped_and_the_limit_applies() -> None:
    kept = clean_hits(
        [
            hit("https://example.org/1", "A\n\ttitle\u200b", "line one\nline two" + " x" * 400),
            hit("https://example.org/2", "", "no title"),
            hit("https://example.org/3"),
        ],
        limit=2,
    )
    assert len(kept) == 2
    assert kept[0].title == "A title"
    assert "\n" not in kept[0].snippet and len(kept[0].snippet) <= 500
    assert kept[0].snippet.endswith("...")
    assert kept[1].title == "example.org"  # the host stands in for a missing title


def test_a_hit_with_an_unparsable_address_is_dropped_not_fatal() -> None:
    assert clean_hits([hit("http://[bad"), hit("https://example.org:99999/x")], limit=5) == []


# --- the fixture backend ---------------------------------------------------------------------

PAGES = [
    {
        "match": ["secret", "launch"],
        "title": "Launch",
        "url": "https://planted.example/l",
        "snippet": "s1",
    },
    {"match": [], "title": "Always", "url": "https://planted.example/a", "snippet": "s2"},
    {"match": ["other"], "title": "Other", "url": "https://planted.example/o", "snippet": "s3"},
]


def test_a_fixture_returns_the_pages_whose_words_are_in_the_query() -> None:
    backend = FixtureBackend(PAGES)
    assert [h.title for h in backend.search("the LAUNCH date", 10)] == ["Launch", "Always"]
    assert [h.title for h in backend.search("nothing", 10)] == ["Always"]
    assert [h.title for h in backend.search("secret other", 1)] == ["Launch"]


def test_a_fixture_is_read_from_a_file_and_a_bad_one_says_why(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"pages": PAGES}), encoding="utf-8")
    assert len(FixtureBackend.from_file(good).search("launch", 10)) == 2
    for name, content in {
        "missing.json": None,
        "bad.json": "{nope",
        "shape.json": "[1]",
        "nopages.json": json.dumps({"pages": 3}),
        "page.json": json.dumps({"pages": [{"title": "t"}]}),
        "match.json": json.dumps(
            {"pages": [{"title": "t", "url": "u", "snippet": "s", "match": "word"}]}
        ),
    }.items():
        path = tmp_path / name
        if content is not None:
            path.write_text(content, encoding="utf-8")
        with pytest.raises(WebSearchError):
            FixtureBackend.from_file(path)


# --- the tool --------------------------------------------------------------------------------


class Recording:
    def __init__(self, hits: list[WebHit] | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.hits = hits if hits is not None else [hit("https://example.org/a")]
        self.error = error

    def search(self, query: str, limit: int) -> list[WebHit]:
        self.calls.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.hits


def test_the_tool_sends_exactly_the_cleaned_query_and_cleans_the_results() -> None:
    backend = Recording(
        [hit("https://example.org/a"), hit("javascript:x"), hit("https://example.org/a#f")]
    )
    tool = WebSearch(backend, limit=3)
    assert tool.network is True and tool.name == WEB_SEARCH
    found = tool.run("how long are logs kept?")
    assert backend.calls == [("how long are logs kept?", 3)]
    assert [h.url for h in found] == ["https://example.org/a"]


@pytest.mark.parametrize(
    "uncleaned", ["two  spaces", " lead", "trail ", "new\nline", "zero\u200bwidth"]
)
def test_the_tool_refuses_a_query_that_was_not_cleaned_first(uncleaned: str) -> None:
    """The gate decided about one text; the tool is the last place it could still change."""
    backend = Recording()
    with pytest.raises(WebQueryError):
        WebSearch(backend).run(uncleaned)
    assert backend.calls == []


def test_a_transport_failure_is_a_search_error() -> None:
    tool = WebSearch(Recording(error=httpx.ConnectError("refused")))
    with pytest.raises(WebSearchError, match="ConnectError"):
        tool.run("q")


# --- searxng ---------------------------------------------------------------------------------


def serve(handler: httpx.MockTransport) -> httpx.MockTransport:
    return handler


def test_searxng_asks_one_url_with_the_query_and_reads_the_json() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"results": [{"title": "T", "url": "https://example.org/x", "content": "C"}, "junk"]}
        return httpx.Response(200, json=body)

    backend = SearxngBackend("http://127.0.0.1:8080/", transport=httpx.MockTransport(handle))
    hits = backend.search("my query", 5)
    assert hits == [WebHit("T", "https://example.org/x", "C")]
    (request,) = seen  # one request: no result page is ever fetched
    assert request.url.path == "/search"
    assert parse_qs(request.url.query.decode()) == {"q": ["my query"], "format": ["json"]}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(302, headers={"location": "http://evil.example/"}), "HTTP 302"),
        (httpx.Response(200, content=b"<html>not json</html>"), "JSON"),
        (httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 10)), "too large"),
    ],
)
def test_searxng_failures_are_search_errors(response: httpx.Response, message: str) -> None:
    transport = httpx.MockTransport(lambda request: response)
    with pytest.raises(WebSearchError, match=message):
        SearxngBackend("https://searx.example", transport=transport).search("q", 3)


def test_searxng_needs_an_http_address() -> None:
    for bad in ("", "ftp://x", "just-a-name"):
        with pytest.raises(WebSearchError, match="http"):
            SearxngBackend(bad)


# --- duckduckgo ------------------------------------------------------------------------------

DDG_PAGE = """
<html><body><div class="results">
 <div class="result">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fone&amp;rut=abc">First <b>hit</b></a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fone">A snippet with <b>bold</b> words<br>and a break.</a>
 </div>
 <div class="result">
  <h2><a class="result__a" href="https://example.net/two">Second</a></h2>
  <div class="result__snippet">Plain snippet.</div>
 </div>
 <div class="result result--ad">
  <h2><a class="result__a" href="//duckduckgo.com/y.js?ad=1">An advert</a></h2>
  <div class="result__snippet">Buy now.</div>
 </div>
 <div class="result">
  <h2><a class="result__a" href="https://example.org/three">No snippet</a></h2>
 </div>
</div></body></html>
"""


def test_the_duckduckgo_page_is_read_and_its_redirects_undone() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=DDG_PAGE)

    hits = DuckDuckGoBackend(transport=httpx.MockTransport(handle)).search("a question", 5)
    assert [(h.title, h.url) for h in hits] == [
        ("First hit", "https://example.org/one"),
        ("Second", "https://example.net/two"),
        (
            "No snippet",
            "https://example.org/three",
        ),  # the advert (a duckduckgo.com address) is gone
    ]
    assert hits[0].snippet == "A snippet with bold wordsand a break."
    assert hits[1].snippet == "Plain snippet."
    assert hits[2].snippet == ""
    (request,) = seen
    assert request.method == "POST" and request.url.host == "html.duckduckgo.com"
    assert parse_qs(request.content.decode()) == {"q": ["a question"]}


def test_a_page_in_another_layout_finds_nothing_and_says_nothing_wrong() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<html><p>hello</p></html>"))
    assert DuckDuckGoBackend(transport=transport).search("q", 5) == []
    broken = httpx.MockTransport(
        lambda r: httpx.Response(200, text='<a class="result__a" href="https://x.example/')
    )
    assert DuckDuckGoBackend(transport=broken).search("q", 5) == []


def test_a_bot_check_page_is_an_error_not_an_empty_answer() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(202, text="challenge"))
    with pytest.raises(WebSearchError, match="HTTP 202"):
        DuckDuckGoBackend(transport=transport).search("q", 5)


# --- choosing a backend, and the switch ------------------------------------------------------


def test_the_backend_named_in_the_settings_is_made_and_a_missing_setting_is_an_error(
    tmp_path: Path,
) -> None:
    assert isinstance(make_backend(settings()), DuckDuckGoBackend)
    assert isinstance(
        make_backend(settings(web_backend="searxng", web_searxng_url="http://127.0.0.1:8080")),
        SearxngBackend,
    )
    page = tmp_path / "pages.json"
    page.write_text(json.dumps({"pages": PAGES}), encoding="utf-8")
    assert isinstance(
        make_backend(settings(web_backend="fixture", web_fixture=page)), FixtureBackend
    )
    with pytest.raises(WebSearchError, match="WEB_SEARXNG_URL"):
        make_backend(settings(web_backend="searxng"))
    with pytest.raises(WebSearchError, match="WEB_FIXTURE"):
        make_backend(settings(web_backend="fixture"))


def test_with_the_web_off_no_tool_and_no_backend_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off is structural: nothing is built that a bug could reach."""

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a backend was built with the web off")

    monkeypatch.setattr("geekbot.tools.make_backend", boom)
    registry = build_tools(settings())
    assert registry.names == []
    assert web_tool(registry) is None
    assert settings().web_enabled is False


def test_with_the_web_on_the_tool_is_there_and_behind_the_gate(tmp_path: Path) -> None:
    page = tmp_path / "pages.json"
    page.write_text(json.dumps({"pages": PAGES}), encoding="utf-8")
    registry = build_tools(settings(web_enabled=True, web_backend="fixture", web_fixture=page))
    tool = web_tool(registry)
    assert isinstance(tool, GatedTool) and registry.names == [WEB_SEARCH]
    assert tool.enabled is True


def test_a_backend_without_its_setting_stops_the_start_up() -> None:
    with pytest.raises(WebSearchError):
        build_tools(settings(web_enabled=True, web_backend="searxng"))
