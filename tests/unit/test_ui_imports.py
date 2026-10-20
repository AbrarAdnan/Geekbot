"""Tests for step S4.4: the UI never imports the storage (PLAN.md section 6.2).

``lint-imports`` (the contract in ``pyproject.toml``) is the real enforcement, and runs in CI. This
test says the same thing with the standard library's ``ast`` so that plain ``pytest`` also fails when
the rule is broken, and so that the rule does not depend on a tool being configured right: it reads
the source, follows every ``geekbot.*`` import of the UI to the file it names, and checks that nothing
reachable that way imports a forbidden module. It sees *imports written in the code*; a module loaded
by ``importlib`` or a string is not seen (nothing in the UI does that).
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
UI = SRC / "geekbot" / "ui"

# The storage, the server, everything that talks to a model, and the libraries behind them.
FORBIDDEN = (
    "geekbot.index",
    "geekbot.ingest",
    "geekbot.retrieval",
    "geekbot.agent",
    "geekbot.llm",
    "geekbot.server",
    "geekbot.csv_mode",
    "geekbot.evals",
    "geekbot.doctor",
    "geekbot.cli",
    "csv_analyst",
    "chromadb",
    "langgraph",
    "langchain_core",
    "langchain_ollama",
    "fastapi",
    "starlette",
    "uvicorn",
    "sqlite3",
    "aiosqlite",
)


def imported_names(path: Path) -> set[str]:
    """Every module name a file imports, and ``module.name`` for each ``from module import name``
    (which may itself be a module, as in ``from geekbot.ui import launch``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: use absolute imports, so this check can follow them"
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def source_file(module: str) -> Path | None:
    """The file of a ``geekbot.*`` module in this repository, or ``None`` (a name, not a module)."""
    base = SRC.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def is_forbidden(name: str) -> bool:
    return any(name == banned or name.startswith(banned + ".") for banned in FORBIDDEN)


def reachable_from_the_ui() -> dict[Path, set[str]]:
    """Each UI file and each ``geekbot.*`` file it imports, however indirectly, with its imports."""
    seen: dict[Path, set[str]] = {}
    pending = sorted(UI.rglob("*.py"))
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen[path] = imported_names(path)
        for name in seen[path]:
            if name.startswith("geekbot.") and (found := source_file(name)) is not None:
                pending.append(found)
    return seen


def test_nothing_the_ui_imports_reaches_the_storage_or_the_server() -> None:
    violations = [
        f"{path.relative_to(SRC).as_posix()} imports {name}"
        for path, names in sorted(reachable_from_the_ui().items())
        for name in sorted(names)
        if is_forbidden(name)
    ]
    assert violations == []


def test_the_walk_really_follows_imports() -> None:
    """If this fails the check above is looking at too little to mean anything."""
    files = {path.relative_to(SRC).as_posix() for path in reachable_from_the_ui()}
    assert {
        "geekbot/ui/client.py",
        "geekbot/ui/views/chat.py",
        "geekbot/apikey.py",  # the key, shared with the server
        "geekbot/sse.py",  # the stream format, shared with the server
        "geekbot/config.py",
    } <= files
    assert not any(name.startswith("geekbot/server") for name in files)


def test_the_detector_sees_a_forbidden_import_when_there_is_one(tmp_path: Path) -> None:
    module = tmp_path / "sneaky.py"
    module.write_text(
        "import chromadb\nfrom geekbot.index import sqlite\nfrom geekbot import server\n",
        encoding="utf-8",
    )
    names = imported_names(module)
    assert {name for name in names if is_forbidden(name)} == {
        "chromadb",
        "geekbot.index",
        "geekbot.index.sqlite",
        "geekbot.server",  # `from geekbot import server` imports the module geekbot.server
    }
    assert not is_forbidden("geekbot.config")
    assert not is_forbidden("geekbot.indexer")  # a prefix of the name is not the package
