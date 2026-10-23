"""The UI's one way to the server: a small synchronous HTTP client (step S4.4).

Streamlit scripts are plain synchronous Python, so this is an ``httpx.Client``, not an async one. It
does four things the pages should not each do for themselves:

- **send the API key** (S4.3) as ``Authorization: Bearer <key>`` on every call but ``/healthz``;
- **turn failures into two exceptions** a page can show as one sentence: ``ServerUnavailableError``
  (the server cannot be reached, the key is missing, or an answer stream broke off) and ``ApiError``
  (the server answered with an error status and a ``detail``);
- **read the answer stream** (``ask``): a generator of ``geekbot.sse.SseEvent``, which ends in a
  ``done`` or an ``error`` event, or raises ``ServerUnavailableError`` if the connection closed
  before either (a server that died mid-answer must not look like a finished answer);
- **wait for a job** (``wait_for_job``): ingest and delete are queued jobs, polled until they end.

It returns the server's JSON as plain dicts. The shapes are the ones in ``server/schemas.py``; the UI
may not import that module (it imports the storage), so the few fields a page reads are named in
``display.py`` and in the pages. The key is never printed: ``GeekbotClient`` has no ``__repr__`` that
shows it, and the error messages carry only the server's own ``detail``.

Only ``geekbot.apikey``, ``geekbot.config`` and ``geekbot.sse`` are imported from Geekbot, which is
what lets ``import-linter`` forbid the rest (PLAN.md section 6.2).
"""

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, Final, Protocol, Self

import httpx

from geekbot.apikey import ApiKeyError, load_api_key
from geekbot.config import Settings
from geekbot.sse import SseEvent, parse_events

__all__ = [
    "LOCAL_HOST",
    "ApiError",
    "ClientError",
    "GeekbotClient",
    "JobReader",
    "JobTimeoutError",
    "ServerUnavailableError",
    "connect",
    "wait_for_job",
]

LOCAL_HOST: Final = "127.0.0.1"  # the server listens here and nowhere else (PLAN.md section 6.2)
CONNECT_TIMEOUT_S: Final = 3.0  # a server on this machine either answers at once or is not there
REQUEST_TIMEOUT_S: Final = 30.0
UPLOAD_TIMEOUT_S: Final = 300.0  # up to 100 MB written to the disk by the server
# The model can be silent for a while before its first token (a load from disk, a long prompt) and
# the stream has no keep-alive frames yet (S4.2), so the read timeout has to be long.
STREAM_READ_TIMEOUT_S: Final = 300.0
JOB_DONE_STATES: Final = frozenset({"done", "failed"})
STREAM_END_EVENTS: Final = frozenset({"done", "error"})
NOT_FOUND: Final = 404
UNAUTHORIZED: Final = 401
DEFAULT_POLL_S: Final = 0.5
DEFAULT_JOB_TIMEOUT_S: Final = 900.0


class ClientError(Exception):
    """Base of the client's errors. ``str(error)`` is a sentence that can be shown to the user."""


class ServerUnavailableError(ClientError):
    """The server cannot be talked to: not reachable, no usable key, or the stream broke off."""


class ApiError(ClientError):
    """The server answered with an error status. ``detail`` is its own explanation."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class JobTimeoutError(ClientError):
    """A job was still running when the wait ended. It keeps running on the server."""


def _detail(response: httpx.Response) -> str:
    """The server's explanation of an error response, in one string.

    Geekbot's own errors, FastAPI's and the middleware's all have the shape ``{"detail": ...}``; a
    validation error has a list of ``{"loc", "msg"}`` items instead of a string.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail:
        return "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail
        )
    return f"HTTP {response.status_code} {response.reason_phrase}".strip()


class GeekbotClient:
    """The server at ``base_url``, spoken to with the API key.

    ``http`` is for tests: any ``httpx.Client`` (Starlette's ``TestClient`` is one) whose base URL
    reaches the app. Left out, a client for ``base_url`` is made.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        key_origin: str = "",
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.key_origin = key_origin  # where the key came from, for the settings page; not the key
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._http = http or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        )

    @classmethod
    def from_settings(cls, settings: Settings, *, http: httpx.Client | None = None) -> Self:
        """The client for the server these settings describe: its port, and the key it keeps.

        Creates nothing: the server makes the key file on its first start, so a missing key means the
        server has never run here (``ServerUnavailableError``, with the command to start it).
        """
        try:
            key = load_api_key(settings)
        except ApiKeyError as error:
            raise ServerUnavailableError(str(error)) from error
        if key is None:
            raise ServerUnavailableError(
                "there is no API key yet: start the server first (`geekbot up` or `geekbot serve`), "
                "which makes one"
            )
        return cls(
            f"http://{LOCAL_HOST}:{settings.server_port}",
            key.value,
            key_origin=key.origin,
            http=http,
        )

    def close(self) -> None:
        self._http.close()

    # --- the transport ---------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = self._http.request(
                method,
                path,
                headers={**self._auth, **(headers or {})},
                timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_S),
                **kwargs,
            )
        except httpx.TransportError as error:
            raise ServerUnavailableError(
                f"cannot reach the Geekbot server at {self.base_url} ({error.__class__.__name__}); "
                "is it running? (`geekbot up` or `geekbot serve`)"
            ) from error
        if response.is_error:
            raise ApiError(response.status_code, _detail(response))
        return response

    # --- health ----------------------------------------------------------------------------

    def ping(self) -> bool:
        """Whether the server answers ``/healthz``, the one route that asks for no key."""
        try:
            return self._http.get("/healthz", timeout=CONNECT_TIMEOUT_S).is_success
        except httpx.TransportError:
            return False

    def health(self) -> dict[str, Any]:
        """``/v1/health``: what the storage holds, the chat model, what is running. Needs the key."""
        result: dict[str, Any] = self._request("GET", "/v1/health").json()
        return result

    # --- the knowledge base ----------------------------------------------------------------

    def formats(self) -> list[str]:
        """The file extensions the server can read, with the dot (``[".md", ".pdf", ...]``)."""
        result: list[str] = self._request("GET", "/v1/kb/formats").json()
        return result

    def documents(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request("GET", "/v1/kb/documents").json()
        return result

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Send one file (the raw bytes as the body, S4.3); the answer names the ingest job.

        ``Content-Length`` is set by ``httpx`` for a ``bytes`` body, which the server requires
        (it refuses a body of unknown size).
        """
        response = self._request(
            "POST",
            "/v1/kb/upload",
            params={"filename": filename},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=UPLOAD_TIMEOUT_S,
        )
        result: dict[str, Any] = response.json()
        return result

    def ingest(self, paths: list[str]) -> dict[str, Any]:
        """Queue an ingest of full paths the server may read; 403 names the folders it may."""
        response = self._request("POST", "/v1/kb/ingest", json={"paths": paths})
        result: dict[str, Any] = response.json()
        return result

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("DELETE", f"/v1/kb/documents/{doc_id}").json()
        return result

    def job(self, job_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", f"/v1/jobs/{job_id}").json()
        return result

    # --- threads ---------------------------------------------------------------------------

    def new_thread(self, mode: str = "chat") -> str:
        """A fresh thread id. The server stores nothing until the first question is answered.

        ``mode`` is what the conversation is meant to be about (``chat`` or ``csv``); the server
        refuses any other name. It is only a statement: a CSV file uploaded to the thread is what
        makes it a CSV conversation (``upload_csv``).
        """
        created: dict[str, Any] = self._request("POST", "/v1/threads", json={"mode": mode}).json()
        return str(created["thread_id"])

    def upload_csv(self, thread_id: str, filename: str, data: bytes) -> dict[str, Any]:
        """Give a CSV conversation its file (step C5): the raw bytes as the body, the thread and the
        file name in the address. The server saves it, loads the table and answers with the rows,
        the columns and what the loader changed (``CsvUploadOut``); those come from the file and are
        untrusted text. Loading a big table takes a while, so the wait is as long as an upload's.

        ``ApiError``: 409 (the thread is a chat conversation, or another question is running),
        413 (too big for the file cap or the memory bound), 415 (not ``.csv`` / ``.tsv``),
        422 (not a usable CSV).
        """
        response = self._request(
            "POST",
            "/v1/csv/upload",
            params={"thread": thread_id, "filename": filename},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=UPLOAD_TIMEOUT_S,
        )
        result: dict[str, Any] = response.json()
        return result

    def threads(self, limit: int = 50) -> list[dict[str, Any]]:
        response = self._request("GET", "/v1/threads", params={"limit": limit})
        result: list[dict[str, Any]] = response.json()
        return result

    def thread(self, thread_id: str) -> dict[str, Any] | None:
        """One thread with its messages, or ``None`` if the server has none by that name."""
        try:
            result: dict[str, Any] = self._request("GET", f"/v1/threads/{thread_id}").json()
        except ApiError as error:
            if error.status == NOT_FOUND:
                return None
            raise
        return result

    def delete_thread(self, thread_id: str) -> bool:
        """Forget a thread: ``False`` if there was none, ``ApiError`` 409 while it is answering."""
        try:
            self._request("DELETE", f"/v1/threads/{thread_id}")
        except ApiError as error:
            if error.status == NOT_FOUND:
                return False
            raise
        return True

    def ask(
        self,
        thread_id: str,
        question: str,
        *,
        think: bool = False,
        mode: str | None = None,
        steps: bool = False,
    ) -> Iterator[SseEvent]:
        """Ask a question and yield the answer's events as they arrive.

        Ends after a ``done`` or an ``error`` event (the server's own ending). Raises ``ApiError``
        before the first event for a refusal (422 bad question, 409 another question running or the
        thread is the other kind of conversation), and ``ServerUnavailableError`` if the connection
        fails or closes before a final event. Closing the generator early closes the connection,
        which the server takes as "the client went away". ``mode`` (``"csv"``) says what the page
        believes the conversation is; left out, the server decides from the thread. ``steps`` asks
        the server for progress events while a chat question runs (``status`` frames that name the
        step: searching, ranking, writing; S3.6); left out, the request is what it was before.
        """
        finished = False
        body: dict[str, Any] = {"question": question, "think": think}
        if mode is not None:
            body["mode"] = mode
        if steps:
            body["steps"] = True
        try:
            with self._http.stream(
                "POST",
                f"/v1/threads/{thread_id}/messages",
                json=body,
                headers=self._auth,
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S, read=STREAM_READ_TIMEOUT_S
                ),
            ) as response:
                if response.is_error:
                    response.read()
                    raise ApiError(response.status_code, _detail(response))
                for event in parse_events(response.iter_lines()):
                    finished = finished or event.event in STREAM_END_EVENTS
                    yield event
        except httpx.TransportError as error:
            raise ServerUnavailableError(
                f"the connection to the server broke ({error.__class__.__name__}); "
                "the question was not answered"
            ) from error
        except json.JSONDecodeError as error:
            raise ServerUnavailableError(
                "the server sent a frame this client cannot read"
            ) from error
        if not finished:
            raise ServerUnavailableError(
                "the server closed the connection before the answer was finished"
            )


def connect(settings: Settings, *, http: httpx.Client | None = None) -> GeekbotClient:
    """A client the server has accepted, or a ``ClientError`` that says what to do about it.

    Three things can be wrong and each gets its own sentence: no key yet (the server never ran
    here), no server answering on the port, and a server that is up but was started with a different
    key (``GEEKBOT_API_KEY`` changed after it started, or two data folders). The key is checked with
    one real call, ``/v1/health``, because ``/healthz`` asks for no key and would say nothing.
    """
    client = GeekbotClient.from_settings(settings, http=http)
    if not client.ping():
        raise ServerUnavailableError(
            f"the Geekbot server is not answering at {client.base_url}. Start it with `geekbot up` "
            "(or `geekbot serve`), then try again"
        )
    try:
        client.health()
    except ApiError as error:
        if error.status == UNAUTHORIZED:
            raise ServerUnavailableError(
                "the server did not accept this UI's API key. The server is running with another "
                f"key than the one in {client.key_origin or 'the settings'}: restart both together "
                "(`geekbot up`)"
            ) from error
        raise
    return client


class JobReader(Protocol):
    """Anything with ``job(job_id)``: the client, or a fake in a test."""

    def job(self, job_id: str, /) -> dict[str, Any]: ...


def wait_for_job(
    client: JobReader,
    job_id: str,
    *,
    poll_seconds: float = DEFAULT_POLL_S,
    timeout_seconds: float = DEFAULT_JOB_TIMEOUT_S,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll a job until it is ``done`` or ``failed`` and return its last state.

    ``on_update`` sees every state read, the last included, so a page can show progress. After
    ``timeout_seconds`` the wait ends with ``JobTimeoutError``. The job itself is not cancelled
    (the server has no such route): it finishes later and its documents show on the next refresh.
    ``sleep`` and ``clock`` are parameters so a test can run the loop without waiting.
    """
    deadline = clock() + timeout_seconds
    while True:
        job = client.job(job_id)
        if on_update is not None:
            on_update(job)
        if job["state"] in JOB_DONE_STATES:
            return job
        if clock() >= deadline:
            raise JobTimeoutError(
                f"the job {job_id} is still {job['state']} after {timeout_seconds:.0f} s; "
                "it keeps running on the server"
            )
        sleep(poll_seconds)
