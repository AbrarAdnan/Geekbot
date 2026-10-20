"""The server's perimeter: who may ask, from where, how much, and which files it may read (step S4.3).

PLAN.md section 8.1 lists the perimeter guarantees; this module is where they live, so that one file
can be read to see what the server refuses. In the order a request meets them:

1. **Host check (DNS rebinding).** The server binds ``127.0.0.1`` (``app.HOST``, a constant). A web
   page open in the user's browser can still send requests to that address, and with "DNS rebinding"
   (its own host name made to point at ``127.0.0.1``) it can read the answers as if they were its
   own. Such a request carries the page's host name in its ``Host`` header, so only ``127.0.0.1`` and
   ``localhost`` are served; anything else gets 400. No ``Access-Control-*`` header is ever sent, so a
   browser never lets another origin read a response: the UI is a Python client, which has no origin.
2. **The API key.** Every route except ``/healthz`` wants ``Authorization: Bearer <key>``
   (``geekbot.apikey`` says where the key comes from). It is checked here, in front of the router,
   so a request without it is refused before FastAPI reads or parses a byte of its body. While a key
   is required the interactive ``/docs`` pages are off: a browser cannot send the header to open them.
3. **Body size.** A JSON request may carry ``MAX_BODY_BYTES``, an upload ``MAX_UPLOAD_BYTES``. The
   size is the declared ``Content-Length`` (the web server reads exactly that many bytes), and a body
   without one (``Transfer-Encoding: chunked``) is refused with 411, so there is no streamed body whose
   size nobody knows in advance. These numbers are starting points, not measurements.
4. **The path allowlist** (``PathPolicy``): which files an ingest may read. See its docstring.

**What this is not.** It is not rate limiting and not a login (one shared key; D-084 dropped what only
outside clients need), it does not encrypt anything (loopback), and a program running as the same
Windows user can read the key file as easily as the UI can. It stops web pages, other machines and
other accounts; it does not stop the user's own malware.
"""

import hmac
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from geekbot.apikey import load_or_create_api_key
from geekbot.config import Settings

__all__ = [
    "LOOPBACK_HOSTS",
    "MAX_BODY_BYTES",
    "MAX_UPLOAD_BYTES",
    "PUBLIC_PATHS",
    "UPLOADS_DIR",
    "UPLOAD_PATHS",
    "PathNotAllowedError",
    "PathPolicy",
    "Perimeter",
    "PerimeterMiddleware",
    "PolicyError",
]

log = logging.getLogger(__name__)

# Host names the server answers to: what `http://127.0.0.1:8000` and `http://localhost:8000` send.
LOOPBACK_HOSTS: Final = ("127.0.0.1", "localhost")
# The only routes that need no key: a liveness probe that touches nothing (the UI waits on it).
PUBLIC_PATHS: Final = frozenset({"/healthz"})
# The routes that take a file. C5 adds the CSV upload here.
UPLOAD_PATHS: Final = frozenset({"/v1/kb/upload"})
MAX_BODY_BYTES: Final = 1_000_000  # a JSON request; the largest real one (200 paths) is under 50 KB
MAX_UPLOAD_BYTES: Final = 100_000_000  # one uploaded file; the same as the loaders' file limit
UPLOADS_DIR: Final = "uploads"  # inside the data folder
MAX_REFUSED_LISTED: Final = 20  # how many refused files an ingest job names

_DIGITS: Final = re.compile(r"[0-9]+")


# --- the path allowlist ----------------------------------------------------------------------


class PathNotAllowedError(ValueError):
    """A path the server will not read. The message is written for the caller and is safe to show.

    ``malformed`` means the text is not a usable path at all (empty, relative, a colon in the wrong
    place): the caller made a mistake, HTTP 422. Otherwise it is a well-formed path that is not
    permitted, HTTP 403.
    """

    def __init__(self, message: str, *, malformed: bool = False) -> None:
        super().__init__(message)
        self.malformed = malformed


class PolicyError(ValueError):
    """The allowed folders in the settings cannot be used (not absolute, missing, not a folder)."""


def _real(path: str | os.PathLike[str]) -> Path:
    """The path with every symlink and Windows junction resolved; it need not exist.

    ``os.path.realpath`` (not ``Path.resolve``) for both roots and candidates, so the two sides of
    every comparison were produced by the same function.
    """
    return Path(os.path.realpath(path))


def _inside(path: Path, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or below it. Both must already be real paths.

    ``Path`` compares case-insensitively on Windows and case-sensitively elsewhere, which is how the
    file systems behave.
    """
    return path.is_relative_to(folder)


def _lexical(raw: object) -> None:
    """Refuse text that is not a usable absolute path, before anything touches the disk.

    The order matters for one rule: a network path (``\\\\server\\share``, ``//server/share``, a
    ``\\\\?\\`` device path) must be refused *here*. Resolving it, which is the next thing the
    policy does, makes Windows open a connection to that server, and the connection carries the
    user's credentials: an attacker's share name would be enough to collect them.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PathNotAllowedError("the path is empty", malformed=True)
    if not raw.isprintable():
        raise PathNotAllowedError("the path has a character that cannot be printed", malformed=True)
    if raw.replace("/", "\\").startswith("\\\\"):  # a mix of the two slashes counts as well
        raise PathNotAllowedError(
            "network and device paths (two slashes first) are not accepted; "
            "use a drive letter or a mounted folder",
            malformed=True,
        )
    if ":" in raw[2:]:
        # `C:\x` has its colon second. Any later colon is an NTFS alternate data stream
        # (`file.txt:stream`) or a device name, and on other systems a rare file name.
        raise PathNotAllowedError("a colon may only follow the drive letter", malformed=True)
    if not Path(raw).is_absolute():
        raise PathNotAllowedError(
            "give the full path, starting with a drive letter or a slash", malformed=True
        )


@dataclass(frozen=True)
class PathPolicy:
    """Which files an ingest may read: the uploads folder, and the roots the user listed.

    ``roots``, ``uploads`` and ``forbidden`` are *real* paths (symlinks resolved), made once by
    ``from_settings``. The rule, in this order:

    1. inside ``uploads`` (where the server puts what it was sent): allowed;
    2. inside a ``forbidden`` folder (the data folder, which holds the databases and the key): refused,
       even when a listed root contains it;
    3. inside one of the ``roots`` (``GEEKBOT_ALLOWED_ROOTS``): allowed;
    4. anywhere else: refused.

    With no roots configured, the server reads nothing but uploads. That is deliberate: a path typed
    into a request is the one way this API reaches the user's own disk, so it is closed until the user
    opens a folder.
    """

    roots: tuple[Path, ...] = ()
    uploads: Path | None = None
    forbidden: tuple[Path, ...] = ()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """The policy for these settings. ``PolicyError`` for a root that cannot be used.

        A bad root is an error at start-up and not a warning: a typo that silently removed a folder
        from the list would look like "my documents are not found" much later.
        """
        roots: list[Path] = []
        for entry in settings.allowed_root_paths:
            if not entry.is_absolute():
                raise PolicyError(
                    f"GEEKBOT_ALLOWED_ROOTS: {str(entry)!r} is not a full path "
                    "(start it with a drive letter or a slash)"
                )
            real = _real(entry)
            if not real.is_dir():
                raise PolicyError(
                    f"GEEKBOT_ALLOWED_ROOTS: {str(entry)!r} is not a folder that exists"
                )
            roots.append(real)
        data = _real(settings.data_dir)
        return cls(roots=tuple(roots), uploads=data / UPLOADS_DIR, forbidden=(data,))

    def admits(self, real: Path) -> bool:
        """Whether this *real* path may be read (the four rules in the class docstring)."""
        if self.uploads is not None and _inside(real, self.uploads):
            return True
        if any(_inside(real, folder) for folder in self.forbidden):
            return False
        return any(_inside(real, root) for root in self.roots)

    def check(self, raw: str) -> Path:
        """The real path for ``raw``, or ``PathNotAllowedError``. This is the allowlist.

        ✍️ written by Claude (prototyping mode, D-047). Six rules:

        1. Text that is not a path (empty or blank, a character that cannot be printed, not
           absolute) is refused as *malformed*; so is a colon anywhere but after a drive letter.
        2. A network or device path (it starts with two slashes of either kind) is refused as
           malformed **before any disk access**: resolving it would contact the server it names.
        3. Symlinks and Windows junctions are resolved first (``realpath``), and the *result* is what
           is compared. A link inside an allowed folder that points outside is therefore refused,
           and ``..`` that leaves a root is refused; ``..`` that stays inside is fine.
        4. The path must be inside the uploads folder or a listed root and not inside the data
           folder (``admits``); otherwise ``PathNotAllowedError`` (403). A folder's own path is
           allowed, so a root can be ingested whole.
        5. Nothing is said about whether the path exists. A path outside every root gets the same
           answer whether or not it exists, so the check cannot be used to probe the disk; existence
           is the caller's business for the paths that pass.
        6. The returned path is the *real* one. What was checked is what is read, and two spellings of
           one file (a link and its target) become one document.

        Blocking (it touches the disk): call it from a worker thread.
        """
        _lexical(raw)
        real = _real(raw)
        if self.admits(real):
            return real
        if any(_inside(real, folder) for folder in self.forbidden):
            raise PathNotAllowedError(
                f"{raw} is inside Geekbot's own data folder, which an ingest never reads"
            )
        raise PathNotAllowedError(
            f"{raw} is not inside a folder this server may read "
            f"(it may read {self.describe()}); "
            "add a folder with GEEKBOT_ALLOWED_ROOTS, or upload the file"
        )

    def split(self, files: Iterable[Path]) -> tuple[list[Path], list[Path]]:
        """``(admitted, refused)`` for files found by a folder walk.

        The top path was checked, but what is *inside* it was not: a symlink (or, on Windows, a
        junction, which ``os.walk`` follows) in an allowed folder can lead anywhere. So every file
        found is resolved and checked on its own. ``admitted`` holds real paths, without repeats and
        in the order found; ``refused`` holds the paths as the walk gave them.
        """
        admitted: list[Path] = []
        refused: list[Path] = []
        seen: set[Path] = set()
        for file in files:
            real = _real(file)
            if not self.admits(real):
                refused.append(file)
            elif real not in seen:
                seen.add(real)
                admitted.append(real)
        return admitted, refused

    def describe(self) -> str:
        """The folders this policy reads from, comma-separated (the uploads folder first)."""
        places = [*([self.uploads] if self.uploads is not None else []), *self.roots]
        return ", ".join(str(place) for place in places) or "nowhere"


# --- everything a request is checked against -------------------------------------------------


@dataclass(frozen=True)
class Perimeter:
    """The perimeter's settings, built once and handed to ``create_app``.

    ``api_key`` of ``None`` means no key is asked for: that is for tests and nothing else, and
    ``geekbot serve`` always passes one. The key is kept out of ``repr``.
    """

    paths: PathPolicy
    api_key: str | None = field(default=None, repr=False)
    hosts: tuple[str, ...] = LOOPBACK_HOSTS
    max_body_bytes: int = MAX_BODY_BYTES
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    key_origin: str = field(default="", compare=False)  # for the start-up message, not the secret

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """What ``geekbot serve`` runs with: a key is always required (made on first use).

        Raises ``geekbot.apikey.ApiKeyError`` or ``PolicyError`` for a configuration that cannot
        be used; the command line turns both into one line.
        """
        # The policy first: a bad root must not leave a new key file behind.
        paths = PathPolicy.from_settings(settings)
        key = load_or_create_api_key(settings)
        return cls(
            paths=paths,
            api_key=key.value,
            key_origin=key.origin + (" (new)" if key.created else ""),
        )


# --- the middleware --------------------------------------------------------------------------


def _problem(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """A refusal in the same shape as FastAPI's own errors: ``{"detail": ...}``."""
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def hostname(host_header: str) -> str:
    """The name in a ``Host`` header, lower-cased and without its port.

    ``Localhost:8000`` gives ``localhost``. An IPv6 literal (``[::1]``) is returned as it is, which no allowed host equals: the server does
    not listen on IPv6.
    """
    name = host_header.strip().lower()
    if name.startswith("["):
        return name
    return name.rsplit(":", 1)[0] if ":" in name else name


class PerimeterMiddleware:
    """Host check, API key and body size, in front of everything (a pure ASGI middleware).

    Not Starlette's ``BaseHTTPMiddleware``: that class wraps the response in a way that has broken
    streaming responses and client disconnects, and this server's main route is a stream. A plain ASGI
    wrapper does its checks, answers itself on a refusal, and otherwise passes the request through
    untouched.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str | None,
        hosts: Sequence[str] = LOOPBACK_HOSTS,
        max_body_bytes: int = MAX_BODY_BYTES,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self.app = app
        self._key = api_key.encode("utf-8") if api_key else None
        self._hosts = frozenset(host.lower() for host in hosts)
        self._max_body = max_body_bytes
        self._max_upload = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # this API has none
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":  # lifespan
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        refusal = self._refusal(scope, headers)
        if refusal is not None:
            await refusal(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _refusal(self, scope: Scope, headers: Headers) -> JSONResponse | None:
        """The response that turns this request away, or ``None`` to let it through."""
        path = str(scope["path"])
        host = headers.get("host", "")
        if hostname(host) not in self._hosts:
            self._log(scope, f"host header {host!r}")
            return _problem(400, "invalid host header")
        if self._key is not None and path not in PUBLIC_PATHS and not self._has_key(headers):
            self._log(scope, "no valid API key")
            return _problem(401, "missing or wrong API key", {"WWW-Authenticate": "Bearer"})
        return self._size_refusal(scope, headers, path)

    def _has_key(self, headers: Headers) -> bool:
        scheme, _, token = headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or self._key is None:
            return False
        # compare_digest: the time it takes does not tell how many leading characters were right.
        return hmac.compare_digest(token.strip().encode("utf-8"), self._key)

    def _size_refusal(self, scope: Scope, headers: Headers, path: str) -> JSONResponse | None:
        if "transfer-encoding" in headers:
            self._log(scope, "a body without Content-Length")
            return _problem(
                411, "send a Content-Length; streamed (chunked) bodies are not accepted"
            )
        declared = headers.get("content-length")
        if declared is None:
            return None  # no body
        if _DIGITS.fullmatch(declared.strip()) is None:
            return _problem(400, "bad Content-Length")
        limit = self._max_upload if path in UPLOAD_PATHS else self._max_body
        if int(declared) > limit:
            self._log(scope, f"a body of {declared} bytes")
            return _problem(413, f"the request is larger than the {limit} bytes this route accepts")
        return None

    @staticmethod
    def _log(scope: Scope, reason: str) -> None:
        """One warning per refusal: the route and the caller, never the Authorization header."""
        client = scope.get("client")
        log.warning(
            "refused %s %s from %s: %s",
            scope.get("method", "?"),
            scope.get("path", "?"),
            client[0] if client else "?",
            reason,
        )
