"""Tests for step S4.3: what stands in front of every route (``PerimeterMiddleware``, ``Perimeter``).

The app is the real one (real lifespan, real SQLite and Chroma with a fake embedder, the real
middleware); only the models are fakes. Layout: the Host check, the API key, "every route is
protected" (a registry-style test, so a route added later cannot be forgotten), the body-size caps,
CORS (there is none), and that ``create_app`` with no arguments is the safe configuration.
"""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from serverkit import make_app, settings_for

from geekbot.apikey import api_key_path
from geekbot.server import create_app
from geekbot.server.security import (
    LOOPBACK_HOSTS,
    PUBLIC_PATHS,
    PathPolicy,
    Perimeter,
    hostname,
)

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

KEY = "perimeter-test-key-" + "k" * 16
LOCAL = "http://127.0.0.1:8000"
AUTH = {"Authorization": f"Bearer {KEY}"}
HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete"}
)  # an OpenAPI path item has more keys


@pytest.fixture
def strict(tmp_path: Path) -> Iterator[TestClient]:
    """The real configuration: a key is required and only the loopback host names are served."""
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS)
    with TestClient(app, base_url=LOCAL) as client:
        yield client


@pytest.fixture
def small_limits(tmp_path: Path) -> Iterator[TestClient]:
    """Like ``strict`` but with tiny caps, so the tests need no 100 MB body."""
    app = make_app(
        tmp_path,
        api_key=KEY,
        hosts=LOOPBACK_HOSTS,
        max_body_bytes=200,
        max_upload_bytes=5000,
    )
    with TestClient(app, base_url=LOCAL) as client:
        yield client


# --- the Host check (DNS rebinding) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:8000", "127.0.0.1"),
        ("Localhost:8000", "localhost"),
        ("  LOCALHOST  ", "localhost"),
        ("[::1]:8000", "[::1]:8000"),
        ("", ""),
    ],
)
def test_hostname_drops_the_port_and_the_case(header: str, name: str) -> None:
    assert hostname(header) == name


@pytest.mark.parametrize(
    ("host", "status"),
    [
        ("127.0.0.1", 200),
        ("127.0.0.1:8000", 200),
        ("localhost", 200),
        ("localhost:8000", 200),
        ("LocalHost", 200),
        ("evil.example", 400),
        ("evil.example:8000", 400),
        ("localhost.evil.example", 400),
        ("127.0.0.1.evil.example", 400),
        ("evil.example:127.0.0.1", 400),
        ("[::1]", 400),
        ("testserver", 400),
    ],
)
def test_only_the_loopback_host_names_are_served(
    strict: TestClient, host: str, status: int
) -> None:
    """A web page after a DNS rebinding sends its own host name; the server will not answer to it."""
    assert strict.get("/v1/health", headers={**AUTH, "Host": host}).status_code == status


def test_the_host_is_checked_before_the_key_and_for_the_public_route_too(
    strict: TestClient,
) -> None:
    assert strict.get("/healthz", headers={"Host": "evil.example"}).status_code == 400
    assert strict.get("/v1/health", headers={"Host": "evil.example"}).status_code == 400


# --- the API key -----------------------------------------------------------------------------


def test_healthz_needs_no_key(strict: TestClient) -> None:
    assert strict.get("/healthz").status_code == 200


def test_a_request_without_the_key_is_401_and_says_how_to_send_it(strict: TestClient) -> None:
    response = strict.get("/v1/health")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "missing or wrong API key"}


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong-key-wrong-key-wrong-key",
        f"Bearer {KEY}x",
        f"Bearer {KEY[:-1]}",
        f"Basic {KEY}",
        f"Token {KEY}",
        KEY,
        "Bearer",
    ],
    ids=[
        "wrong",
        "one-too-long",
        "one-too-short",
        "basic-scheme",
        "token-scheme",
        "no-scheme",
        "empty",
    ],
)
def test_a_wrong_key_or_scheme_is_401(strict: TestClient, header: str) -> None:
    assert strict.get("/v1/health", headers={"Authorization": header}).status_code == 401


def test_the_right_key_is_accepted_with_any_case_of_the_scheme(strict: TestClient) -> None:
    assert strict.get("/v1/health", headers=AUTH).status_code == 200
    assert strict.get("/v1/health", headers={"Authorization": f"bearer {KEY}"}).status_code == 200


def test_the_key_is_checked_before_the_body_is_read(strict: TestClient) -> None:
    """Without the key a malformed body is 401, not the 422 that reading it would have produced."""
    response = strict.post(
        "/v1/kb/ingest", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/healthz/", "/HEALTHZ", "//healthz", "/healthz%2F"])
def test_only_the_exact_public_path_is_public(strict: TestClient, path: str) -> None:
    """Anything that is not exactly ``/healthz`` wants the key, so no spelling slips through."""
    assert strict.get(path).status_code == 401


def route_calls(app: FastAPI) -> list[tuple[str, str]]:
    """Every (method, path) of the app except the public ones, path parameters filled with ``x``."""
    # The OpenAPI document, not ``app.routes``: FastAPI 0.146 keeps an included router as one lazy
    # ``_IncludedRouter`` entry there, so the routes themselves are not in that list.
    calls: list[tuple[str, str]] = []
    for path, operations in app.openapi()["paths"].items():
        if path in PUBLIC_PATHS:
            continue
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        calls.extend((method.upper(), concrete) for method in operations if method in HTTP_METHODS)
    return calls


def test_every_route_but_the_public_one_wants_the_key(tmp_path: Path) -> None:
    """A route added in a later step is covered without anyone remembering to protect it."""
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS)
    calls = route_calls(app)
    assert len(calls) >= 12  # health, kb (list, ingest, upload, delete), jobs (2), threads (5)
    with TestClient(app, base_url=LOCAL) as client:
        for method, path in calls:
            assert client.request(method, path).status_code == 401, (method, path)


def test_with_no_key_configured_nothing_is_asked_and_the_docs_pages_exist(tmp_path: Path) -> None:
    """The test configuration (``geekbot serve`` always has a key)."""
    with TestClient(make_app(tmp_path)) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_with_a_key_the_docs_pages_are_off(strict: TestClient, path: str) -> None:
    assert strict.get(path, headers=AUTH).status_code == 404


def test_a_refusal_is_logged_with_the_route_and_never_with_the_key(strict: TestClient) -> None:
    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    logger = logging.getLogger("geekbot.server.security")
    handler = Collect()
    logger.addHandler(handler)
    try:
        strict.get("/v1/health", headers={"Authorization": "Bearer guessed-guessed-guessed-guess"})
    finally:
        logger.removeHandler(handler)
    text = "\n".join(lines)
    assert "refused GET /v1/health" in text
    assert "no valid API key" in text
    assert "guessed-guessed" not in text
    assert KEY not in text


def test_the_perimeter_keeps_the_key_out_of_its_repr() -> None:
    assert KEY not in repr(Perimeter(paths=PathPolicy(), api_key=KEY))


# --- body size -------------------------------------------------------------------------------


def test_a_json_body_over_the_cap_is_413_and_nothing_is_queued(small_limits: TestClient) -> None:
    response = small_limits.post("/v1/kb/ingest", json={"paths": ["x" * 300]}, headers=AUTH)
    assert response.status_code == 413
    assert small_limits.get("/v1/jobs", headers=AUTH).json() == []


def test_a_body_under_the_cap_reaches_the_routes_own_validation(small_limits: TestClient) -> None:
    assert small_limits.post("/v1/kb/ingest", json={"paths": []}, headers=AUTH).status_code == 422


def test_a_body_without_a_length_is_411(small_limits: TestClient) -> None:
    """Chunked bodies are refused: nobody could know their size before reading them."""

    def pieces() -> Iterator[bytes]:
        yield b'{"paths": '
        yield b'["x"]}'

    response = small_limits.post("/v1/kb/ingest", content=pieces(), headers=AUTH)
    assert response.status_code == 411


def test_a_length_that_is_not_a_number_is_400(small_limits: TestClient) -> None:
    response = small_limits.post(
        "/v1/kb/ingest", content=b"{}", headers={**AUTH, "Content-Length": "abc"}
    )
    assert response.status_code == 400


def test_the_upload_route_has_its_own_larger_cap(small_limits: TestClient) -> None:
    params = {"filename": "a.txt"}
    fits = small_limits.post("/v1/kb/upload", params=params, content=b"x" * 1000, headers=AUTH)
    assert fits.status_code == 202  # over the 200 bytes of a JSON body, under the 5000 of an upload
    too_big = small_limits.post("/v1/kb/upload", params=params, content=b"x" * 6000, headers=AUTH)
    assert too_big.status_code == 413


# --- CORS ------------------------------------------------------------------------------------


def test_no_cors_header_is_ever_sent(strict: TestClient) -> None:
    """The UI is a Python client with no origin; a browser must never be let read a response."""
    origin = {"Origin": "https://evil.example"}
    for headers in (origin, {**origin, **AUTH}):
        response = strict.get("/v1/health", headers=headers)
        assert not [name for name in response.headers if name.lower().startswith("access-control")]
    preflight = strict.options(
        "/v1/health", headers={**origin, **AUTH, "Access-Control-Request-Method": "GET"}
    )
    assert preflight.status_code == 405
    assert not [name for name in preflight.headers if name.lower().startswith("access-control")]


# --- the safe default ------------------------------------------------------------------------


def test_create_app_with_no_perimeter_builds_the_one_serve_uses(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    app = create_app(settings)
    perimeter = app.state.perimeter
    assert perimeter.api_key  # a key is required
    assert api_key_path(settings).read_text(encoding="utf-8").strip() == perimeter.api_key
    assert perimeter.hosts == LOOPBACK_HOSTS
    assert (app.docs_url, app.redoc_url, app.openapi_url) == (None, None, None)
