"""Tests for step C2: who may import whom around the vendored ``csv_analyst`` package.

PLAN.md §6.7 says only ``geekbot.csv_mode`` imports ``csv_analyst``. The real enforcement is an
import-linter contract (Phase 4); until then this test reads every source file's imports with
``ast`` (no code is run) and fails if the rule is broken. Imports inside functions count too.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
GEEKBOT = SRC / "geekbot"
CSV_MODE = GEEKBOT / "csv_mode"
CSV_ANALYST = SRC / "csv_analyst"


def _imported_packages(path: Path) -> set[str]:
    """The top-level package of every absolute import in the file, wherever the import sits."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            packages.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            packages.add(node.module.split(".")[0])
    return packages


def _files_importing(root: Path, package: str, *, skip: Path | None = None) -> list[str]:
    """Files under ``root`` (paths relative to ``src/``) that import ``package``, minus ``skip``."""
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if skip is not None and skip in path.parents:
            continue
        if package in _imported_packages(path):
            found.append(path.relative_to(SRC).as_posix())
    return found


def test_only_csv_mode_imports_csv_analyst() -> None:
    assert _files_importing(GEEKBOT, "csv_analyst", skip=CSV_MODE) == []


def test_the_scan_does_see_the_imports_inside_csv_mode() -> None:
    """Without this, a broken scanner would pass the test above by finding nothing at all."""
    importers = _files_importing(CSV_MODE, "csv_analyst")
    assert "geekbot/csv_mode/adapter.py" in importers
    assert "geekbot/csv_mode/provider.py" in importers
    assert "geekbot/csv_mode/grading.py" in importers


def test_the_vendored_package_does_not_import_geekbot() -> None:
    """The copy must stay usable on its own, so a re-sync from Gigalogy stays a plain copy."""
    assert _files_importing(CSV_ANALYST, "geekbot") == []


def test_the_cli_imports_csv_mode_only_inside_its_commands() -> None:
    """Pandas and pyarrow are slow to import; `geekbot --version` must not wait for them."""
    tree = ast.parse((GEEKBOT / "cli.py").read_text(encoding="utf-8"))
    top_level = [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    heavy = (
        "geekbot.csv_mode",
        "geekbot.evals.csv_repro",
    )  # both pull in pandas (C3 added the second)
    assert not [module for module in top_level if module.startswith(heavy)]
