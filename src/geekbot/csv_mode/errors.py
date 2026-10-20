"""Errors of CSV mode that code outside this package has to name (step C5).

They live in a module of their own, with no imports, because ``geekbot.csv_mode.adapter`` pulls in
pandas through the vendored package and the server only needs the *names* of these errors to tell the
user what went wrong. ``adapter`` re-exports ``CsvLoadError``, so older imports keep working.
"""

__all__ = ["CsvLoadError"]


class CsvLoadError(Exception):
    """The file is not a usable CSV. Geekbot's own type, so callers outside this package never
    need the vendored package's ``IngestionError`` (the import boundary, ``csv_mode/__init__``)."""
