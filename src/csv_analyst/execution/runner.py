"""Sandbox execution orchestration (PLAN.md Phase 3, CLAUDE.md Sec8/Sec9).

Generated code is never run in this process. It is handed to
`sandbox_entry.py` as a subprocess, with the DataFrame crossing the process
boundary as a Parquet file, and is given a hard wall-clock timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from csv_analyst.config import settings
from csv_analyst.contracts import ResultKind

_SANDBOX_ENTRY = Path(__file__).resolve().parent / "sandbox_entry.py"


@dataclass
class ExecutionResult:
    ok: bool
    kind: Optional[ResultKind] = None
    value: Any = None  # scalar / dict, or a DataFrame for series & dataframe kinds
    error: Optional[str] = None
    ms: int = 0


def execute_code(df: pd.DataFrame, code: str, timeout_s: Optional[int] = None) -> ExecutionResult:
    timeout_s = settings.sandbox_timeout_s if timeout_s is None else timeout_s
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="gigalogy_sandbox_") as tmpdir:
        tmp = Path(tmpdir)
        df_path = tmp / "df.parquet"
        code_path = tmp / "code.py"
        result_path = tmp / "result.parquet"
        meta_path = tmp / "meta.json"

        try:
            df.to_parquet(df_path)
            code_path.write_text(code, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- parent-side serialization before the
            # subprocess boundary (missing pyarrow, disk error, unserializable dtype) must
            # degrade into a typed result like every other failure mode here, not propagate
            # as a raw traceback out of execute_code/run_pipeline. This converts an
            # environment failure into a normal execution failure, so the pipeline will
            # retry twice and then abstain -- a missing dependency presents as "could not
            # answer" rather than crashing loudly (same tradeoff as Part A's backend-failure
            # reason handling).
            return ExecutionResult(
                ok=False,
                error=f"could not serialize the dataframe for the sandbox: {type(exc).__name__}: {exc}",
                ms=int((time.monotonic() - start) * 1000),
            )

        cmd = [
            sys.executable, "-I",
            str(_SANDBOX_ENTRY),
            str(df_path), str(code_path), str(result_path), str(meta_path),
        ]

        try:
            proc = subprocess.run(
                cmd, cwd=tmpdir, timeout=timeout_s, capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                error=f"execution exceeded {timeout_s}s timeout",
                ms=int((time.monotonic() - start) * 1000),
            )

        ms = int((time.monotonic() - start) * 1000)

        if not meta_path.exists():
            stderr_lines = proc.stderr.strip().splitlines()
            detail = stderr_lines[-1] if stderr_lines else f"exit code {proc.returncode}"
            return ExecutionResult(ok=False, error=f"sandbox process crashed: {detail}", ms=ms)

        meta = _read_meta(meta_path)
        if not meta.get("ok"):
            return ExecutionResult(ok=False, error=meta.get("error", "execution failed"), ms=ms)

        kind = ResultKind(meta["kind"])
        if kind in (ResultKind.SERIES, ResultKind.DATAFRAME):
            value = pd.read_parquet(result_path)
        else:
            value = meta.get("value")

        return ExecutionResult(ok=True, kind=kind, value=value, ms=ms)


def _read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))
