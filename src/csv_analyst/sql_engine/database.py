"""Loading a DataFrame into SQLite, once per dataset.

The CSV is written into an in-memory SQLite database (table `data`) the
first time a dataset is queried; the database is then kept as serialized
bytes, and each query gets a fresh private copy (`deserialize`, a few ms).
Fresh copies keep queries isolated and thread-safe (the API runs requests on
a thread pool) without re-running `to_sql` every time.

SQLite as compiled into Python has no POWER/SQRT/MEDIAN (checked: `2 ^ 3`,
`pow()`, `sqrt()` all fail), which cost hospital/h01 a whole run. They are
registered here as ordinary functions -- still only callable from a SELECT.

`real_columns` (settings.sql_real_measures): integer measure columns stored
as REAL, so `height / 100` divides like pandas instead of SQLite's integer
division (156 / 100 = 1). Identifier columns stay INTEGER.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
import threading
import weakref
from collections import OrderedDict

import pandas as pd

from csv_analyst.sql_engine.entities import DATA_TABLE, create_sql

TABLE_NAME = DATA_TABLE
_CACHE_SIZE = 4
# (id(df), real columns) -> (weakref to df, serialized database). The weakref check means a
# reused id() after garbage collection can never return another frame's data.
_cache: "OrderedDict[tuple, tuple[weakref.ref, bytes]]" = OrderedDict()
_cache_lock = threading.Lock()


def _safe(fn):
    def wrapper(*args):
        if any(a is None for a in args):
            return None
        try:
            return fn(*args)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    return wrapper


class _Median:
    def __init__(self):
        self.values: list[float] = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return statistics.median(self.values) if self.values else None


class _Stdev:
    def __init__(self):
        self.values: list[float] = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return statistics.stdev(self.values) if len(self.values) > 1 else None


EXTRA_FUNCTIONS = {
    "POWER": (2, _safe(math.pow)),
    "POW": (2, _safe(math.pow)),
    "SQRT": (1, _safe(math.sqrt)),
    "LN": (1, _safe(math.log)),
    "LOG10": (1, _safe(math.log10)),
    "EXP": (1, _safe(math.exp)),
    "FLOOR": (1, _safe(math.floor)),
    "CEIL": (1, _safe(math.ceil)),
}
EXTRA_AGGREGATES = {"MEDIAN": _Median, "STDEV": _Stdev}


def _sql_frame(df: pd.DataFrame, real_columns: frozenset = frozenset()) -> pd.DataFrame:
    """Datetime columns as ISO text (date only when there is no time part),
    so SQLite's date functions and string comparisons work on them."""
    out = df.copy()
    for col in out.columns:
        if col in real_columns and pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype("float64")
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            s = out[col]
            midnight = s.dropna().eq(s.dropna().dt.normalize()).all()
            out[col] = s.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S")
    return out


def _serialized(df: pd.DataFrame, real_columns: frozenset, tables: tuple = ()) -> bytes:
    key = (id(df), real_columns, tables)
    with _cache_lock:  # a host app may run several conversations on threads
        hit = _cache.get(key)
        if hit is not None and hit[0]() is df:
            _cache.move_to_end(key)
            return hit[1]
    conn = sqlite3.connect(":memory:")
    try:
        _sql_frame(df, real_columns).to_sql(TABLE_NAME, conn, index=False)
        for table in tables:  # card H-B1 entity tables, built from `data`
            conn.execute(create_sql(table))
        data = conn.serialize()
    finally:
        conn.close()
    with _cache_lock:
        _cache[key] = (weakref.ref(df), data)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return data


def connect(df: pd.DataFrame, real_columns=(), tables: tuple = ()) -> sqlite3.Connection:
    """A private in-memory copy of `df` as table `data` (plus the H-B1 entity
    `tables`, if given), with the extra functions registered. The caller owns
    (and closes) the connection."""
    conn = sqlite3.connect(":memory:")
    conn.deserialize(_serialized(df, frozenset(real_columns), tuple(tables)))
    for name, (n_args, fn) in EXTRA_FUNCTIONS.items():
        conn.create_function(name, n_args, fn, deterministic=True)
    for name, cls in EXTRA_AGGREGATES.items():
        conn.create_aggregate(name, 1, cls)
    return conn
