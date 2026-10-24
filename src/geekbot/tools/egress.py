"""The egress gate: the one door a network tool is called through (step S5.3, PLAN.md §8.1 G2).

**The rule** (``agent.taint.egress_decision``): no network call whose arguments were built from text
the user did not write, unless the user approved those exact arguments. This module is the part that
makes the rule hold: a network tool is wrapped by ``EgressGate.wrap`` into a ``GatedTool``, and a
``GatedTool`` runs the tool only after the rule said ``ALLOW``, every time, whoever calls it. The graph
asks the user *before* it calls (the ``egress_gate`` node, with ``interrupt()``), and passes what the
user answered; the tool asks the rule again. A bug in the graph that skipped the question would reach
a tool that still refuses.

**Registry.** ``ToolRegistry.register`` refuses any tool that says ``network = True`` and is not a
``GatedTool``, so "every network tool goes through the gate" is not a convention but an error at
start-up, and ``tests/unit/test_egress.py`` fails if a network tool is added to ``geekbot.tools``
without it. The same test lists which modules of the program import a network library at all, so a new
way out of the machine cannot be added without that list being edited on purpose.

What this does **not** do: it cannot tell whether a *tool's own* code sends more than it should (the
web tool sends one query and fetches nothing, which is checked in its tests), and it says nothing about
what the model writes into its answer (that is G3, the output filter, and the injection evaluation).
"""

from collections.abc import Iterable
from typing import Any, Final, Protocol

from geekbot.agent.taint import EgressAction, EgressDecision, egress_decision
from geekbot.domain import Label

__all__ = [
    "EgressGate",
    "EgressRefused",
    "GatedTool",
    "NetworkTool",
    "ToolRegistry",
    "UngatedToolError",
]


class NetworkTool(Protocol):
    """What the gate wraps: a tool with a name, a ``network`` flag and one text argument."""

    @property
    def name(self) -> str: ...

    @property
    def network(self) -> bool: ...

    def run(self, arguments: str) -> Any: ...


class EgressRefused(RuntimeError):
    """The gate did not let a call through. ``decision`` says why (``ASK`` or ``DENY``)."""

    def __init__(self, decision: EgressDecision) -> None:
        super().__init__(f"{decision.action.value}: {decision.reason}")
        self.decision = decision


class GatedTool:
    """A network tool that only runs when the rule allows it. Make one with ``EgressGate.wrap``."""

    network: Final = True
    _MADE_BY_GATE: Final = object()  # a token only ``EgressGate.wrap`` has

    def __init__(self, tool: NetworkTool, *, enabled: bool, _token: object = None) -> None:
        if _token is not self._MADE_BY_GATE:
            raise TypeError("a GatedTool is made with EgressGate.wrap, not directly")
        self._tool = tool
        self._enabled = enabled

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(
        self,
        arguments: str,
        labels: Iterable[Label],
        *,
        approved: str | None = None,
        denied: bool = False,
    ) -> EgressDecision:
        """What the rule says about this call, without making it."""
        return egress_decision(
            enabled=self._enabled,
            labels=labels,
            arguments=arguments,
            approved=approved,
            denied=denied,
        )

    def call(self, arguments: str, *, labels: Iterable[Label], approved: str | None = None) -> Any:
        """Run the tool if, and only if, the rule says ``ALLOW``; else raise ``EgressRefused``."""
        decision = self.check(arguments, labels, approved=approved)
        if decision.action is not EgressAction.ALLOW:
            raise EgressRefused(decision)
        return self._tool.run(arguments)


class EgressGate:
    """Wraps network tools. One per server; ``enabled`` is the switch (``Settings.web_enabled``)."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def wrap(self, tool: NetworkTool) -> GatedTool:
        """``tool`` behind the gate. A tool that is not a network tool does not need one."""
        if not tool.network:
            raise ValueError(f"{tool.name} is not a network tool; it does not go through the gate")
        return GatedTool(tool, enabled=self._enabled, _token=GatedTool._MADE_BY_GATE)


class UngatedToolError(TypeError):
    """A network tool was registered without the gate."""


class ToolRegistry:
    """The tools a run may use, by name. A network tool is accepted only behind the gate."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        name = str(tool.name)
        if getattr(tool, "network", False) and not isinstance(tool, GatedTool):
            raise UngatedToolError(
                f"{name} is a network tool and must be registered through EgressGate.wrap"
            )
        if name in self._tools:
            raise ValueError(f"a tool called {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)
