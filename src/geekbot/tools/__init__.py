"""The tools the answer graph may call (step S5.3, PLAN.md §6.6, §8.1).

There is one: ``web_search``, the only network tool. ``build_tools`` makes the registry for a set of
settings. With ``web_enabled`` off (the default) the web tool is **not built**: no backend object, no
HTTP client, nothing that could be reached by a bug. The graph then has no web tool and the
``egress_gate`` step says so; "web off: no egress" is a structural fact, not a flag read at call time.

``geekbot.tools.web`` is the only module of the program that imports an HTTP library for a query that
came from a user's words (the others talk to the local Ollama, download evaluation datasets, or are the
UI's client of this program's own server); a test lists them all.
"""

import httpx

from geekbot.config import Settings
from geekbot.tools.egress import EgressGate, GatedTool, ToolRegistry
from geekbot.tools.web import WebSearch, make_backend

__all__ = ["WEB_SEARCH", "build_tools"]

WEB_SEARCH = WebSearch.name


def build_tools(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> ToolRegistry:
    """The registry for ``settings``: empty with the web off, else the web tool behind the gate.

    ``transport`` is for tests (a fake HTTP server). ``WebSearchError`` if the chosen backend lacks a
    required setting, so a mistake stops ``geekbot serve`` at start-up and not at the first search.
    """
    registry = ToolRegistry()
    if settings.web_enabled:
        gate = EgressGate(enabled=True)
        backend = make_backend(settings, transport=transport)
        registry.register(gate.wrap(WebSearch(backend, limit=settings.web_max_results)))
    return registry


def web_tool(registry: ToolRegistry) -> GatedTool | None:
    """The gated web tool of a registry, or ``None`` when the web is off."""
    tool = registry.get(WEB_SEARCH)
    return tool if isinstance(tool, GatedTool) else None
