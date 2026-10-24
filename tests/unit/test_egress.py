"""Tests for step S5.3: the egress gate, the tool registry and the inventory of ways out.

G2 says no network call with tainted arguments without the user's approval of those exact arguments.
These tests check the parts that make it hold: a gated tool refuses on its own, a network tool cannot be
registered without the gate, and the list of modules that can reach the network is a list somebody edits
on purpose."""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import geekbot.tools as tools_package
from geekbot.agent.taint import EgressAction
from geekbot.domain import Label
from geekbot.tools.egress import (
    EgressGate,
    EgressRefused,
    GatedTool,
    ToolRegistry,
    UngatedToolError,
)
from geekbot.tools.web import WebHit, WebSearch

SRC = Path(__file__).resolve().parents[2] / "src"


class Counting:
    """A network tool that counts how often it really ran."""

    name = "count"
    network = True

    def __init__(self) -> None:
        self.runs: list[str] = []

    def run(self, arguments: str) -> list[str]:
        self.runs.append(arguments)
        return ["ran"]


class Local:
    name = "local"
    network = False

    def run(self, arguments: str) -> str:
        return arguments


def gated(enabled: bool = True) -> tuple[GatedTool, Counting]:
    tool = Counting()
    return EgressGate(enabled=enabled).wrap(tool), tool


# --- the gated tool --------------------------------------------------------------------------


def test_a_gated_tool_cannot_be_made_without_the_gate() -> None:
    with pytest.raises(TypeError, match="EgressGate.wrap"):
        GatedTool(Counting(), enabled=True)
    with pytest.raises(TypeError, match="EgressGate.wrap"):
        GatedTool(Counting(), enabled=True, _token=object())


def test_the_users_own_words_go_through() -> None:
    tool, inner = gated()
    assert tool.call("q", labels=[Label.USER]) == ["ran"]
    assert inner.runs == ["q"]


def test_tainted_arguments_are_refused_by_the_tool_itself_not_only_by_the_graph() -> None:
    tool, inner = gated()
    with pytest.raises(EgressRefused) as refused:
        tool.call("q", labels=[Label.USER, Label.KB])
    assert refused.value.decision.action is EgressAction.ASK
    assert refused.value.decision.tainted is True
    assert inner.runs == []


def test_the_exact_approval_lets_the_tainted_call_through_and_another_text_does_not() -> None:
    tool, inner = gated()
    assert tool.call("q", labels=[Label.WEB], approved="q") == ["ran"]
    with pytest.raises(EgressRefused):
        tool.call("q changed", labels=[Label.WEB], approved="q")
    assert inner.runs == ["q"]


def test_a_switched_off_gate_refuses_everything() -> None:
    tool, inner = gated(enabled=False)
    for labels, approved in (([Label.USER], None), ([Label.KB], "q"), ([], "q")):
        with pytest.raises(EgressRefused) as refused:
            tool.call("q", labels=labels, approved=approved)
        assert refused.value.decision.action is EgressAction.DENY
    assert inner.runs == []


def test_a_call_with_no_labels_is_treated_as_tainted() -> None:
    tool, inner = gated()
    with pytest.raises(EgressRefused):
        tool.call("q", labels=[])
    assert inner.runs == []


def test_check_decides_without_calling() -> None:
    tool, inner = gated()
    assert tool.check("q", [Label.KB]).action is EgressAction.ASK
    assert tool.check("q", [Label.KB], approved="q").action is EgressAction.ALLOW
    assert tool.check("q", [Label.KB], approved="q", denied=True).action is EgressAction.DENY
    assert inner.runs == []


def test_only_a_network_tool_is_wrapped() -> None:
    with pytest.raises(ValueError, match="not a network tool"):
        EgressGate(enabled=True).wrap(Local())


# --- the registry ----------------------------------------------------------------------------


def test_a_network_tool_cannot_be_registered_without_the_gate() -> None:
    registry = ToolRegistry()
    with pytest.raises(UngatedToolError, match="EgressGate.wrap"):
        registry.register(Counting())
    assert registry.names == []
    tool, _ = gated()
    registry.register(tool)
    assert registry.names == ["count"] and registry.get("count") is tool


def test_a_local_tool_needs_no_gate_and_a_name_is_used_once() -> None:
    registry = ToolRegistry()
    registry.register(Local())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Local())
    assert registry.get("nothing") is None


def test_every_network_tool_class_in_the_tools_package_is_known_and_cannot_skip_the_gate() -> None:
    """A new network tool added to ``geekbot.tools`` fails here until this list is edited on purpose."""
    found: set[str] = set()
    for module_info in pkgutil.iter_modules(tools_package.__path__):
        module = importlib.import_module(f"{tools_package.__name__}.{module_info.name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and getattr(cls, "network", False) is True:
                found.add(name)
    assert found == {"GatedTool", "WebSearch"}
    registry = ToolRegistry()
    with pytest.raises(UngatedToolError):
        registry.register(WebSearch(_Nothing()))


class _Nothing:
    def search(self, query: str, limit: int) -> list[WebHit]:
        return []


# --- the inventory of ways out ---------------------------------------------------------------

# Modules that import a library which can open a connection, and what each one talks to. A module
# that is not in this table must not be able to reach the network; adding one is a decision, made
# here, in a diff somebody reads. ``geekbot.tools.web`` is the only one that sends text a user's words
# went into to somewhere other than this machine.
NETWORK_MODULES: dict[str, str] = {
    "geekbot.agent.resilience": "httpx exception types, to tell a transient failure from another",
    "geekbot.cli": "uvicorn for `serve`; httpx for `eval fetch` (dataset download)",
    "geekbot.csv_mode.adapter": "the local Ollama (a model-presence check)",
    "geekbot.csv_mode.provider": "the local Ollama",
    "geekbot.evals.beir": "download of the BEIR datasets (no user text)",
    "geekbot.evals.ollama_instance": "a private Ollama on a loopback port",
    "geekbot.index.embedder": "the local Ollama (embeddings)",
    "geekbot.index.workspace": "the local Ollama (model checks)",
    "geekbot.llm.chat": "the local Ollama through langchain-ollama",
    "geekbot.llm.models": "the local Ollama",
    "geekbot.retrieval.rerank": "fastembed: downloads the reranker's weights once (no user text)",
    "geekbot.tools": "httpx type of a test transport",
    "geekbot.tools.web": "the web search: the one place a query leaves the machine (G2)",
    "geekbot.ui.client": "this program's own server on 127.0.0.1",
    "geekbot.ui.launch": "this program's own server on 127.0.0.1",
    "csv_analyst.llm.ollama_provider": "vendored: requests to Ollama (unused: Geekbot's provider replaces it)",
}
NETWORK_LIBRARIES = {
    "httpx", "requests", "urllib3", "aiohttp", "socket", "ftplib", "smtplib", "websockets",
    "websocket", "ollama", "langchain_ollama", "huggingface_hub", "fastembed", "uvicorn",
}  # fmt: skip
NETWORK_SUBMODULES = {"urllib.request", "http.client", "xmlrpc.client"}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_network(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
            if node.module in {"urllib", "http", "xmlrpc"}:
                names += [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            if name.split(".")[0] in NETWORK_LIBRARIES or any(
                name == sub or name.startswith(sub + ".") for sub in NETWORK_SUBMODULES
            ):
                return True
    return False


def test_the_modules_that_can_reach_the_network_are_the_ones_on_the_list() -> None:
    found = {
        _module_name(path)
        for path in SRC.rglob("*.py")
        if _imports_network(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert found == set(NETWORK_MODULES), (
        "a module now imports a network library (or one stopped): "
        f"new {sorted(found - set(NETWORK_MODULES))}, gone {sorted(set(NETWORK_MODULES) - found)}"
    )


def test_the_graph_and_the_nodes_never_import_a_network_library() -> None:
    """The decision to send is made in ``egress_gate`` and made by the tool's gate; the agent package
    holds no way to send anything itself (``agent.resilience`` only names exception types)."""
    offenders = [
        module
        for module in NETWORK_MODULES
        if module.startswith("geekbot.agent.") and module != "geekbot.agent.resilience"
    ]
    assert offenders == []


def test_the_agent_context_reaches_the_web_only_through_a_gated_tool() -> None:
    from geekbot.agent.runtime import AgentContext

    hints = inspect.get_annotations(AgentContext)
    assert "GatedTool" in str(hints["web"])
    assert "WebSearch" not in str(hints["web"])


def test_the_sources_scan_sees_what_it_should() -> None:
    """The scan itself works: it finds the one module that is known to import httpx at top level."""
    tree = ast.parse("def f():\n    import httpx\n")
    assert _imports_network(tree)
    assert not _imports_network(ast.parse("from urllib.parse import quote\nimport json\n"))
    assert _imports_network(ast.parse("from urllib import request\n"))
    assert _imports_network(ast.parse("import http.client\n"))
    unrelated: Any = ast.parse("import http.server\n")
    assert not _imports_network(unrelated)
