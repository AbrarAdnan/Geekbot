"""The web search tool (step S5.3, PLAN.md §8.1 G2).

The only place in the application where a question's words can leave the machine. Everything around it
exists to make that a decision and not an accident: the tool is only built when ``web_enabled`` is on,
it is only reached through ``tools.egress.EgressGate``, which asks ``agent.taint.egress_decision``
first, and what it sends is exactly the cleaned query text the user saw.

**What it does.** One call, ``WebSearch.run(query)``: ask a search backend, get titles, links and
snippets, clean them, return at most ``limit``. **No result page is ever fetched**, so a page cannot be
made to talk to this program, and the tool has no way to be pointed at an internal address by what it
reads. What comes back is untrusted text like any document (label ``WEB``): it goes into the prompt
data-marked, and the answer built from it is tainted.

**Cleaning** (v1 pasted raw results into the prompt and never de-duplicated them, PLAN.md rows 12 and
3): control and zero-width characters are removed, whitespace collapsed, titles and snippets capped, a
link must be ``http``/``https`` with no user name or password, must not point at this machine or a
private network, and a page listed twice (same host, path and query, a fragment ignored) is listed
once.

**Backends**, chosen by ``Settings.web_backend``:

- ``fixture``: canned results from a JSON file, no network. The injection evaluation plants its pages
  this way, and it lets you try the approval prompt offline.
- ``searxng``: a SearXNG instance you run (``/search?format=json``, a documented, stable interface).
- ``duckduckgo``: DuckDuckGo's HTML page, as v1 used it. A page layout is not an interface: this parser
  is tested on a hand-written page in the documented layout and **has not been tried against the live
  site**; if the layout has changed it finds nothing and says so.
"""

import ipaddress
import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

from geekbot.config import Settings

__all__ = [
    "MAX_QUERY_CHARS",
    "DuckDuckGoBackend",
    "FixtureBackend",
    "SearchBackend",
    "SearxngBackend",
    "WebHit",
    "WebQueryError",
    "WebSearch",
    "WebSearchError",
    "clean_hits",
    "clean_query",
    "make_backend",
]

MAX_QUERY_CHARS: Final = 300
MAX_TITLE_CHARS: Final = 200
MAX_SNIPPET_CHARS: Final = 500
MAX_RESPONSE_BYTES: Final = 1_000_000  # a result page is a few tens of kilobytes; more is not one
_DDG_URL: Final = "https://html.duckduckgo.com/html/"
_USER_AGENT: Final = "Mozilla/5.0 (compatible; geekbot-local)"
_LOCAL_SUFFIXES: Final = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_VOID_TAGS: Final = frozenset({"br", "hr", "img", "input", "link", "meta", "wbr", "source"})


class WebQueryError(ValueError):
    """The search query cannot be sent: it is empty or longer than ``MAX_QUERY_CHARS``."""


class WebSearchError(RuntimeError):
    """The search could not be done (network, a backend that answered badly, a bad fixture)."""


@dataclass(frozen=True)
class WebHit:
    """One search result as the model will see it. All three fields are untrusted text."""

    title: str
    url: str
    snippet: str


class SearchBackend(Protocol):
    """Where a query goes. Returns raw hits; ``WebSearch`` cleans them."""

    def search(self, query: str, limit: int) -> list[WebHit]: ...


# --- cleaning --------------------------------------------------------------------------------


def _printable(text: str) -> str:
    """``text`` without control and format characters (zero-width, bidi marks, ...), spaces kept."""
    return "".join(
        ch for ch in text if ch.isspace() or unicodedata.category(ch) not in {"Cc", "Cf", "Cs"}
    )


def _one_line(text: str, limit: int) -> str:
    squeezed = " ".join(_printable(text).split())
    return squeezed if len(squeezed) <= limit else squeezed[: limit - 3].rstrip() + "..."


def clean_query(text: str) -> str:
    """The query exactly as it will be sent, or ``WebQueryError``.

    One line, without control characters. It never *cuts* a long query: the user approves the text they
    are shown, and a tool that then sent a shortened one would send something nobody approved.
    """
    query = " ".join(_printable(text).split())
    if not query:
        raise WebQueryError("the search query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise WebQueryError(
            f"the search query is {len(query)} characters; the limit is {MAX_QUERY_CHARS}"
        )
    return query


def _is_local_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host or host == "localhost" or host.endswith(_LOCAL_SUFFIXES)
    return not address.is_global


def _clean_url(url: str) -> str | None:
    """``url`` as a plain public web address (no fragment), or ``None`` if it is not one."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
        return None
    if any(ch.isspace() or not ch.isprintable() for ch in url.strip()) or _is_local_host(host):
        return None
    netloc = host if ":" not in host else f"[{host}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def _same_page(url: str) -> tuple[str, str, str]:
    parts = urlsplit(url)
    return (parts.netloc.lower(), parts.path.rstrip("/") or "/", parts.query)


def clean_hits(hits: Iterable[WebHit], limit: int) -> list[WebHit]:
    """At most ``limit`` usable hits: cleaned text, public ``http(s)`` links, each page once."""
    cleaned: list[WebHit] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        url = _clean_url(hit.url)
        if url is None:
            continue
        page = _same_page(url)
        if page in seen:
            continue
        seen.add(page)
        title = _one_line(hit.title, MAX_TITLE_CHARS) or urlsplit(url).netloc
        cleaned.append(WebHit(title, url, _one_line(hit.snippet, MAX_SNIPPET_CHARS)))
        if len(cleaned) >= limit:
            break
    return cleaned


# --- the tool --------------------------------------------------------------------------------


class WebSearch:
    """The search tool. ``network`` marks it for the gate (``tools.registry`` refuses it ungated)."""

    name: Final = "web_search"
    network: Final = True

    def __init__(self, backend: SearchBackend, *, limit: int = 5) -> None:
        self._backend = backend
        self._limit = limit

    def run(self, arguments: str) -> list[WebHit]:
        """Search for ``arguments``, which must already be a cleaned query (``clean_query``).

        Refuses anything else: the gate decided about one exact text, and this is the last place
        that text could still change. Backend failures are a ``WebSearchError``.
        """
        if arguments != clean_query(arguments):
            raise WebQueryError("the query was not cleaned before it was sent")
        try:
            raw = self._backend.search(arguments, self._limit)
        except httpx.HTTPError as error:
            raise WebSearchError(f"the web search failed ({type(error).__name__})") from error
        return clean_hits(raw, self._limit)


# --- backends --------------------------------------------------------------------------------


class FixtureBackend:
    """Canned results from a JSON file: ``{"pages": [{"match": [...], "title", "url", "snippet"}]}``.

    A page is returned when the query contains any of its ``match`` words (case-insensitive); an
    empty ``match`` returns it for every query. No network. Order in the file is order in the answer.
    """

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = [self._check(page) for page in pages]

    @classmethod
    def from_file(cls, path: Path) -> "FixtureBackend":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise WebSearchError(f"cannot read the web fixture {path}: {error}") from error
        pages = data.get("pages") if isinstance(data, dict) else None
        if not isinstance(pages, list):
            raise WebSearchError(f"{path}: expected an object with a 'pages' list")
        return cls(pages)

    @staticmethod
    def _check(page: object) -> dict[str, Any]:
        if not isinstance(page, dict) or not all(
            isinstance(page.get(key), str) for key in ("title", "url", "snippet")
        ):
            raise WebSearchError("a fixture page needs text 'title', 'url' and 'snippet'")
        match = page.get("match", [])
        if not isinstance(match, list) or not all(isinstance(word, str) for word in match):
            raise WebSearchError("a fixture page's 'match' must be a list of words")
        return page

    def search(self, query: str, limit: int) -> list[WebHit]:
        lowered = query.casefold()
        found = [
            WebHit(page["title"], page["url"], page["snippet"])
            for page in self._pages
            if not page.get("match") or any(word.casefold() in lowered for word in page["match"])
        ]
        return found[:limit]


def _read_limited(client: httpx.Client, request: httpx.Request) -> bytes:
    """The response body, read at most ``MAX_RESPONSE_BYTES`` (a bigger one is an error).

    Redirects are not followed: a search service that answers with one is not the service that was
    configured.
    """
    response = client.send(request, stream=True)
    try:
        if response.status_code != httpx.codes.OK:
            raise WebSearchError(f"the search service answered HTTP {response.status_code}")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise WebSearchError("the search service's answer is too large")
        return bytes(body)
    finally:
        response.close()


class SearxngBackend:
    """A SearXNG instance's JSON interface: ``GET <base>/search?q=...&format=json``."""

    def __init__(
        self, base_url: str, *, timeout_s: float = 8.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise WebSearchError(f"web_searxng_url {base_url!r} is not an http(s) address")
        self._url = base_url.rstrip("/") + "/search"
        self._timeout = timeout_s
        self._transport = transport

    def search(self, query: str, limit: int) -> list[WebHit]:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            request = client.build_request("GET", self._url, params={"q": query, "format": "json"})
            body = _read_limited(client, request)
        try:
            results = json.loads(body).get("results", [])
        except (ValueError, AttributeError) as error:
            raise WebSearchError("the SearXNG instance did not answer with JSON results") from error
        hits = [
            WebHit(
                str(item.get("title", "")), str(item.get("url", "")), str(item.get("content", ""))
            )
            for item in results
            if isinstance(item, dict)
        ]
        return hits[: limit * 2]  # a little slack for hits the cleaning drops


class _DuckDuckGoParser(HTMLParser):
    """Reads the result list of DuckDuckGo's HTML page.

    A result is an ``<a class="result__a" href=...>title</a>`` followed by an element with class
    ``result__snippet``. The link goes through a redirect (``/l/?uddg=<address>``) which is undone here.
    Nesting is followed with a depth count, so ``<b>`` inside a snippet does not end it.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[WebHit] = []
        self._depth = 0
        self._open: dict[str, Any] | None = None
        self._field: str | None = None  # "title" or "snippet": the text being collected
        self._field_depth = 0

    def _flush(self) -> None:
        if self._open is not None and self._open["url"]:
            self.hits.append(
                WebHit(
                    "".join(self._open["title"]), self._open["url"], "".join(self._open["snippet"])
                )
            )
        self._open = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._open = {
                "url": _unwrap_redirect(attributes.get("href") or ""),
                "title": [],
                "snippet": [],
            }
            self._field = "title"
        elif self._open is not None and "result__snippet" in classes:
            self._field = "snippet"
        else:
            if tag not in _VOID_TAGS:
                self._depth += 1
            return
        self._field_depth = self._depth
        if tag not in _VOID_TAGS:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        self._depth = max(0, self._depth - 1)
        if self._field is not None and self._depth <= self._field_depth:
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field is not None and self._open is not None:
            self._open[self._field].append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def _unwrap_redirect(href: str) -> str:
    """The address behind one of DuckDuckGo's ``/l/?uddg=`` redirects; ``""`` for DuckDuckGo's own."""
    link = "https:" + href if href.startswith("//") else href
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()
    if host == "duckduckgo.com" or host.endswith(".duckduckgo.com"):
        target = parse_qs(parts.query).get("uddg")
        return target[0] if parts.path.startswith("/l/") and target else ""
    return link


class DuckDuckGoBackend:
    """DuckDuckGo's HTML search page, as v1 used it. See the module docstring for what is verified."""

    def __init__(
        self, *, timeout_s: float = 8.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._timeout = timeout_s
        self._transport = transport

    def search(self, query: str, limit: int) -> list[WebHit]:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            request = client.build_request(
                "POST", _DDG_URL, data={"q": query}, headers={"User-Agent": _USER_AGENT}
            )
            page = _read_limited(client, request).decode("utf-8", errors="replace")
        parser = _DuckDuckGoParser()
        parser.feed(page)
        parser.close()
        return parser.hits[: limit * 2]


def make_backend(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> SearchBackend:
    """The backend ``settings`` names. ``WebSearchError`` if its required setting is missing."""
    if settings.web_backend == "fixture":
        if settings.web_fixture is None:
            raise WebSearchError("web_backend is 'fixture' but GEEKBOT_WEB_FIXTURE is not set")
        return FixtureBackend.from_file(settings.web_fixture)
    if settings.web_backend == "searxng":
        if not settings.web_searxng_url:
            raise WebSearchError("web_backend is 'searxng' but GEEKBOT_WEB_SEARXNG_URL is not set")
        return SearxngBackend(
            settings.web_searxng_url, timeout_s=settings.web_timeout_s, transport=transport
        )
    return DuckDuckGoBackend(timeout_s=settings.web_timeout_s, transport=transport)
