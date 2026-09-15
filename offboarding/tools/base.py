"""Tool contract and the registry that enforces the allowlist.

Every tool call in the system goes through a :class:`ToolRegistry` lookup, and
every lookup is scoped to the calling step's allowlist. A step cannot reach a
tool it did not declare, which is the guardrail the brief asks for: it is
enforced at the call site rather than trusted to prompt text.

A tool that changes the world also implements :class:`SideEffectingTool`, whose
``lookup`` method answers "did the effect for this idempotency key land?". Real
providers expose exactly this (Stripe's idempotency keys, a message id lookup);
our mocks expose it so the reconciliation path in ``ledger.py`` is real code
rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from offboarding.domain.errors import ToolNotAllowed


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Static declaration of what a tool is and how it must be treated."""

    name: str
    description: str
    side_effecting: bool
    #: Whether a human must approve before this tool may run at all. Enforced by
    #: the orchestrator's approval gate, not by the tool itself.
    requires_approval: bool = False


@runtime_checkable
class Tool(Protocol):
    """Anything callable through the registry."""

    spec: ToolSpec

    def call(self, **kwargs: Any) -> dict[str, Any]:
        """Perform the tool's work and return a JSON-serialisable result."""
        ...


@runtime_checkable
class SideEffectingTool(Tool, Protocol):
    """A tool whose effects can be looked up after the fact."""

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return the recorded effect for ``idempotency_key``, or ``None``.

        Used only during recovery, to decide whether an interrupted call
        actually landed. Must not itself have side effects.
        """
        ...


class ToolRegistry:
    """Name-to-tool mapping with allowlist enforcement on every resolution."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool. Re-registering a name is a programming error."""
        if tool.spec.name in self._tools:
            raise ValueError(f"tool {tool.spec.name!r} is already registered")
        self._tools[tool.spec.name] = tool

    def names(self) -> list[str]:
        """Every registered tool name."""
        return sorted(self._tools)

    def spec(self, name: str) -> ToolSpec:
        """Return a tool's spec without resolving it for a call."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotAllowed(f"tool {name!r} is not registered")
        return tool.spec

    def resolve(self, name: str, allowlist: frozenset[str]) -> Tool:
        """Return a tool, refusing anything outside ``allowlist``.

        Both failure modes are :class:`ToolNotAllowed`, which is a permanent
        error: retrying a call the guardrail refused would breach it again.
        """
        if name not in allowlist:
            raise ToolNotAllowed(
                f"tool {name!r} is not in this step's allowlist "
                f"({sorted(allowlist) or 'empty'})"
            )
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotAllowed(f"tool {name!r} is not registered")
        return tool
