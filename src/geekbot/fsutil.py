"""Small file-system helpers that must behave on Windows too (found in C5's chat).

``replace_file`` is the last step of every "write to a temporary file, then rename it over the real
one" in this project (the evaluation result files, the rewrite cache, a downloaded dataset, an
uploaded file). That pattern is what makes a crash leave either the old file or the new one and never
half of one. On POSIX the rename always succeeds. On Windows ``os.replace`` fails with
``PermissionError: [WinError 5] Access is denied`` (or WinError 32, a sharing violation) whenever
another program has the target or the source open at that moment without allowing a rename, and the
programs that do that to a file written a millisecond ago are the Windows virus scanner and the search
indexer. The failure is real, rare, and gone a few tens of milliseconds later: it failed
``test_the_rewrite_phase_calls_the_model_once_per_follow_up_then_uses_the_cache`` once in a full run,
and the same class of failure is the likeliest reason a Windows CI job once failed in its Tests step
and passed again on the same code. So the rename is tried again, a few times, with a growing pause, and
only on Windows; a ``PermissionError`` that is still there after about a second is a real one and is
raised as it is.
"""

import os
import sys
import time
from collections.abc import Callable
from typing import Final

__all__ = ["REPLACE_ATTEMPTS", "REPLACE_PAUSE_S", "replace_file"]

REPLACE_ATTEMPTS: Final = 8
REPLACE_PAUSE_S: Final = 0.05  # the pause before attempt n + 1 is n times this: 1.4 s in all
_RETRY_PERMISSION_ERRORS: Final = sys.platform == "win32"


def replace_file(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    attempts: int = REPLACE_ATTEMPTS,
    pause_s: float = REPLACE_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
    retry: bool = _RETRY_PERMISSION_ERRORS,
) -> None:
    """``os.replace(source, target)``, tried again on Windows when it is refused (see the module).

    Only ``PermissionError`` is retried, and only when ``retry`` is true (Windows by default); any
    other error, and the last ``PermissionError``, propagate unchanged. ``sleep`` and ``retry`` are
    parameters so the tests need neither a clock nor a Windows machine.
    """
    tries = max(attempts, 1) if retry else 1
    for attempt in range(1, tries + 1):
        try:
            os.replace(source, target)
        except PermissionError:
            if attempt == tries:
                raise
            sleep(pause_s * attempt)
        else:
            return
