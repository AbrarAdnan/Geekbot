"""Nightly retention job for Lantern (lantern-01).

Runs every night at 02:30 from cron. It deletes chat logs older than 90 days and audit logs
older than 365 days. It never touches uploaded documents: the operations lead removes those by
hand once a project has been closed for 12 months.

If the file /srv/lantern/LEGAL_HOLD exists the job deletes nothing, writes "hold active" to its
log and exits with code 3, so that whoever is handling an incident keeps the evidence.
"""

import sys
import time
from pathlib import Path

CHAT_LOG_DIR = Path("/srv/lantern/chat_logs")
AUDIT_LOG_DIR = Path("/srv/lantern/audit")
HOLD_FILE = Path("/srv/lantern/LEGAL_HOLD")
CLEANUP_LOG = Path("/var/log/lantern/cleanup.log")

CHAT_RETENTION_DAYS = 90
AUDIT_RETENTION_DAYS = 365
SECONDS_PER_DAY = 86_400
EXIT_OK = 0
EXIT_HOLD = 3


def expired(directory: Path, max_age_days: int, now: float) -> list[Path]:
    """The .jsonl files in ``directory`` last changed more than ``max_age_days`` days ago.

    A file exactly ``max_age_days`` old is kept: the job deletes only what is older than the limit.
    """
    cutoff = now - max_age_days * SECONDS_PER_DAY
    return [path for path in sorted(directory.glob("*.jsonl")) if path.stat().st_mtime < cutoff]


def write_log(message: str) -> None:
    """Append one timestamped line to the cleanup log."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with CLEANUP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def run(now: float | None = None) -> int:
    """Delete expired logs and return the exit code (0 normally, 3 when a legal hold is active)."""
    moment = time.time() if now is None else now
    if HOLD_FILE.exists():
        write_log("hold active, nothing deleted")
        return EXIT_HOLD
    chat = expired(CHAT_LOG_DIR, CHAT_RETENTION_DAYS, moment)
    audit = expired(AUDIT_LOG_DIR, AUDIT_RETENTION_DAYS, moment)
    for path in [*chat, *audit]:
        path.unlink()
    write_log(f"deleted {len(chat)} chat logs and {len(audit)} audit logs")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(run())
