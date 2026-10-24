"""A private Ollama server for the bench's KV-cache sweep (step S5.1, PLAN.md §6.5 rule R-3).

The KV-cache type (``f16`` or ``q8_0``) and flash attention are settings of the Ollama **server**
(``OLLAMA_KV_CACHE_TYPE``, ``OLLAMA_FLASH_ATTENTION``), not of a request. Changing them means
restarting the server, and the server on port 11434 is the user's: other programs use it, and
restarting it, or editing its environment, would be changing their PC under them. So the sweep starts
its **own** server on another loopback port with the settings it wants, lets the models load there,
and stops it afterwards. It reads the same model files (the same ``OLLAMA_MODELS`` or default folder),
so nothing is downloaded, and it holds nothing when the sweep is over.

Two servers share one GPU, so the bench's idle-GPU guard (``evals.bench``) runs first: it refuses to
call a number official if anything else is using the card. This module only starts and stops the
process; every part that touches the machine is a parameter, so the tests need no Ollama.
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol

import httpx

__all__ = [
    "DEFAULT_PORT",
    "InstanceSpec",
    "OllamaInstanceError",
    "PrivateOllama",
    "answers",
]

DEFAULT_PORT: Final = 11435  # the user's own Ollama is on 11434
KV_CACHE_TYPES: Final = ("f16", "q8_0", "q4_0")


class OllamaInstanceError(RuntimeError):
    """The private server could not be started, or something already listens where it should."""


@dataclass(frozen=True)
class InstanceSpec:
    """The server settings one sweep configuration needs (PLAN.md §6.5, R-3).

    ``num_parallel`` is 1 and flash attention is on in every instance, as rule R-3 says; the KV-cache
    type is what the sweep varies. ``max_loaded_models`` is 2 so the embedder (on the CPU) and the
    chat model can both stay loaded (R-2).
    """

    port: int = DEFAULT_PORT
    kv_cache_type: str = "f16"
    flash_attention: bool = True
    num_parallel: int = 1
    max_loaded_models: int = 2

    def __post_init__(self) -> None:
        if self.kv_cache_type not in KV_CACHE_TYPES:
            raise ValueError(
                f"unknown KV-cache type {self.kv_cache_type!r}; expected one of "
                f"{', '.join(KV_CACHE_TYPES)}"
            )
        if self.kv_cache_type != "f16" and not self.flash_attention:
            raise ValueError("a quantized KV cache needs flash attention")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port out of range: {self.port}")

    @property
    def host(self) -> str:
        """The base URL of the instance (loopback only)."""
        return f"http://127.0.0.1:{self.port}"

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        """``base`` (the parent's environment) with this instance's Ollama settings on top.

        ``OLLAMA_HOST`` is always replaced, so an instance can never end up on the user's port.
        """
        env = dict(base)
        env.update(
            {
                "OLLAMA_HOST": f"127.0.0.1:{self.port}",
                "OLLAMA_KV_CACHE_TYPE": self.kv_cache_type,
                "OLLAMA_FLASH_ATTENTION": "1" if self.flash_attention else "0",
                "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
                "OLLAMA_MAX_LOADED_MODELS": str(self.max_loaded_models),
            }
        )
        return env


class ProcessLike(Protocol):
    """The part of ``subprocess.Popen`` the manager uses."""

    pid: int

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


def answers(host: str, *, timeout_s: float = 1.0) -> bool:
    """Whether an Ollama server answers ``GET /api/version`` at ``host``."""
    try:
        return httpx.get(f"{host}/api/version", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False


def _kill_tree(process: ProcessLike) -> None:
    """Stop ``ollama serve`` and the runner processes it started.

    On Windows the runners are children that outlive a plain ``terminate()`` and keep the model in
    video memory, so the whole tree goes (``taskkill /T``). Elsewhere ``terminate`` is enough.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
            timeout=30,
        )
    else:
        process.terminate()


class PrivateOllama:
    """``ollama serve`` on a loopback port of its own, as a context manager.

    ``with PrivateOllama(InstanceSpec(kv_cache_type="q8_0")) as host:`` gives the URL to point
    ``Settings.ollama_host`` at; leaving the block stops the server and its runners. Starting refuses
    when something already answers on the port (a second server there would be a mystery) and when
    the process dies or stays silent for ``start_timeout_s``.
    """

    def __init__(
        self,
        spec: InstanceSpec,
        *,
        command: tuple[str, ...] = ("ollama", "serve"),
        popen: Callable[..., ProcessLike] = subprocess.Popen,
        environ: Mapping[str, str] | None = None,
        probe: Callable[[str], bool] = answers,
        kill: Callable[[ProcessLike], None] = _kill_tree,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        start_timeout_s: float = 60.0,
    ) -> None:
        self.spec = spec
        self._command = command
        self._popen = popen
        self._environ = dict(os.environ if environ is None else environ)
        self._probe = probe
        self._kill = kill
        self._clock = clock
        self._sleep = sleep
        self._timeout = start_timeout_s
        self._process: ProcessLike | None = None

    def start(self) -> str:
        """Start the server and wait until it answers; return its URL."""
        if self._process is not None:
            raise OllamaInstanceError("this private Ollama is already started")
        if self._probe(self.spec.host):
            raise OllamaInstanceError(
                f"something already answers at {self.spec.host}; choose another port for the "
                "private Ollama (it must never share a port with a server that is not its own)"
            )
        try:
            process = self._popen(
                list(self._command),
                env=self.spec.environment(self._environ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise OllamaInstanceError(
                f"could not start `{' '.join(self._command)}`: {error}"
            ) from error
        self._process = process
        deadline = self._clock() + self._timeout
        while True:
            if process.poll() is not None:
                self._process = None
                raise OllamaInstanceError(
                    f"`{' '.join(self._command)}` exited with code {process.poll()} before it "
                    f"answered at {self.spec.host} (is the port in use?)"
                )
            if self._probe(self.spec.host):
                return self.spec.host
            if self._clock() >= deadline:
                self.stop()
                raise OllamaInstanceError(
                    f"the private Ollama did not answer at {self.spec.host} within "
                    f"{self._timeout:.0f} s"
                )
            self._sleep(0.25)

    def stop(self) -> None:
        """Stop the server and its runners; safe to call twice or before ``start``."""
        process, self._process = self._process, None
        if process is None:
            return
        self._kill(process)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def __enter__(self) -> str:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
