"""CSV analyst mode: Geekbot's side of the connection to the vendored ``csv_analyst`` package.

This is the only package in Geekbot that imports ``csv_analyst`` (PLAN.md §6.7, "import
boundary"). Everything else reaches the CSV analyst through here, so the copied code stays behind
one door: its settings, its model calls and its quirks are translated once, in this folder
(lesson: project/learn/06-ports-adapters.md). ``tests/unit/test_csv_import_boundary.py`` checks the
rule until the import-linter contract arrives in Phase 4.
"""
