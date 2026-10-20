"""The API key the server asks for and the UI sends (step S4.3, PLAN.md section 8.1 "Perimeter").

**What it is, and what it is not.** A local password between Geekbot's own server and its own UI,
checked by Geekbot's own code. It is *not* the key of any cloud service (no Anthropic, OpenAI or
other provider key): Geekbot calls no cloud API, and this key is never sent anywhere.

**Why a key on a server that listens on 127.0.0.1 only.** Loopback keeps other *machines* out. It does
not keep out a web page open in the user's browser (it can send requests to ``127.0.0.1`` too, and
"DNS rebinding" lets it read the answers), another Windows user on the same PC, or a program that
should not be talking to Geekbot at all. A secret that only the UI and the server know closes that.

**Where the key comes from** (first match wins):

1. ``GEEKBOT_API_KEY`` (environment or ``.env``), when set and not blank;
2. the file ``<data_dir>/api-key``;
3. nothing yet: ``load_or_create_api_key`` makes a random key and writes it to that file.

So ``geekbot serve`` needs no set-up, and the UI of step S4.4, which reads the same settings, finds the
same key. The file is created with mode ``0o600`` where the OS has modes; **on Windows it has whatever
rights its folder gives it**, so a PC shared with other Windows users needs the key in an environment
variable of the user's own session instead. This module imports only ``config`` and the standard
library on purpose: the UI is an API client and must not import the storage modules (PLAN.md section
6.2), and ``geekbot.server`` imports them.
"""

import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from geekbot.config import Settings

__all__ = [
    "API_KEY_FILE",
    "MIN_KEY_CHARS",
    "ApiKey",
    "ApiKeyError",
    "api_key_path",
    "check_api_key",
    "generate_api_key",
    "load_api_key",
    "load_or_create_api_key",
]

API_KEY_FILE: Final = "api-key"
# ``secrets.token_urlsafe(32)`` gives 43 characters (256 bits). A key typed by hand must still be long
# enough that guessing it over loopback is out of the question.
MIN_KEY_CHARS: Final = 24
_TOKEN_BYTES: Final = 32
# What RFC 6750 allows in a bearer token, so the key is always safe to put in a header.
_VALID: Final = re.compile(r"[A-Za-z0-9._~+/=-]+")


class ApiKeyError(ValueError):
    """The configured key is unusable. The message names where it came from and what is wrong."""


@dataclass(frozen=True)
class ApiKey:
    """A key and where it was found. ``value`` is hidden from ``repr`` so a log line cannot leak it."""

    value: str = field(repr=False)
    origin: str  # "GEEKBOT_API_KEY", or the path of the key file
    created: bool = False  # true when this call made the file


def generate_api_key() -> str:
    """A fresh random key: 256 bits, URL-safe characters only."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def check_api_key(value: str, *, origin: str) -> str:
    """The key with its surrounding whitespace removed, or ``ApiKeyError`` if it is unusable.

    Too short is refused rather than accepted: a key of ``1234`` set for a quick try would be the
    whole protection. The allowed characters are those of an HTTP bearer token.
    """
    text = value.strip()
    if len(text) < MIN_KEY_CHARS or _VALID.fullmatch(text) is None:
        raise ApiKeyError(
            f"the API key from {origin} must be at least {MIN_KEY_CHARS} characters long and use "
            "only letters, digits and . _ ~ + / = - (delete it to get a generated one)"
        )
    return text


def api_key_path(settings: Settings) -> Path:
    """Where the generated key is kept: ``<data_dir>/api-key``."""
    return settings.data_dir / API_KEY_FILE


def load_api_key(settings: Settings) -> ApiKey | None:
    """The key that is already there, or ``None`` if there is none yet. Creates nothing.

    A key in ``GEEKBOT_API_KEY`` wins over the file. A blank variable counts as unset, because a
    ``GEEKBOT_API_KEY=`` line left in ``.env`` is far more likely to mean "no key" than "an empty key".
    """
    configured = settings.api_key.get_secret_value() if settings.api_key is not None else ""
    if configured.strip():
        origin = "GEEKBOT_API_KEY"
        return ApiKey(check_api_key(configured, origin=origin), origin)
    path = api_key_path(settings)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return ApiKey(check_api_key(text, origin=str(path)), str(path))


def load_or_create_api_key(settings: Settings) -> ApiKey:
    """The key to use: the one that exists, or a new random one saved for next time.

    The file is created with ``O_EXCL``, so two processes starting together on a fresh data folder
    cannot both write one: the loser reads what the winner wrote. In the instant between the
    winner's create and its write the file is empty and the loser fails with the "too short" error;
    starting again fixes it (the window is microseconds, and a server and a UI are not started in the
    same instant by anything in this project).
    """
    found = load_api_key(settings)
    if found is not None:
        return found
    path = api_key_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_api_key()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raced = load_api_key(settings)
        if raced is None:  # it vanished again: not a state this code can explain
            raise ApiKeyError(f"could not create or read the key file {path}") from None
        return raced
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return ApiKey(key, str(path), created=True)
