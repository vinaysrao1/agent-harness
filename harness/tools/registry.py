"""Tool registry and dispatch (DESIGN.md §3 "Tool router", §4.1).

A :class:`Tool` binds three things together: the provider-neutral
:class:`~harness.types.ToolSpec` the model sees, the
:class:`~harness.permissions.ToolMeta` the permission engine gates on, and an
async ``handler`` that actually executes the call and returns the string the
model reads back. :mod:`harness.tools.builtin` supplies the concrete
handlers; this module only knows how to hold and dispatch them.

:class:`ToolRegistry` is the harness's tool router: the agent loop calls
:meth:`ToolRegistry.dispatch` for every :class:`~harness.types.ToolCall` the
model emits (after the permission engine has already decided ``ALLOW``).
Dispatch never lets a handler exception escape — an unknown tool name or an
exception raised inside a handler both become an error
:class:`~harness.types.ToolResult` rather than crashing the agent loop, and
oversized results are truncated with a marker rather than blowing up the
context window.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from harness.permissions import ToolMeta
from harness.types import ToolCall, ToolResult, ToolSpec

__all__ = [
    "MAX_RESULT_BYTES",
    "ToolHandler",
    "Tool",
    "DuplicateToolError",
    "ToolRegistry",
]

#: Tool results larger than this are truncated (mirrors the truncation
#: pattern used for sandbox exec output in `harness.sandbox.base`, applied
#: here to the text every tool call ultimately returns to the model).
MAX_RESULT_BYTES: Final[int] = 50_000

#: A tool handler: takes the call's raw ``arguments`` dict, returns the
#: string the model reads back as the tool result. Handlers are free to
#: raise on bad input or backend failure -- `ToolRegistry.dispatch` turns
#: any exception into an error `ToolResult` rather than propagating it.
ToolHandler = Callable[[dict], Awaitable[str]]


class DuplicateToolError(Exception):
    """Raised by :meth:`ToolRegistry.register` when a tool name is reused."""


@dataclass(frozen=True)
class Tool:
    """One registerable tool: its model-facing spec, permission metadata,
    and the handler that executes it.

    A plain (non-pydantic) frozen dataclass, since ``handler`` is an async
    callable that pydantic has nothing useful to validate.
    """

    spec: ToolSpec
    meta: ToolMeta
    handler: ToolHandler


def _truncate_result(content: str) -> str:
    """Truncate ``content`` to :data:`MAX_RESULT_BYTES`, marker on overflow.

    Mirrors :func:`harness.sandbox.base.truncate_output`: encode as UTF-8,
    keep at most the byte limit, decode leniently, and append a marker
    naming the limit and the true original size so the truncation is never
    mistaken for the whole result.
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= MAX_RESULT_BYTES:
        return content
    head = encoded[:MAX_RESULT_BYTES]
    marker = (
        f"\n...[tool result truncated at {MAX_RESULT_BYTES} bytes; "
        f"{len(encoded)} bytes total]...\n"
    )
    return head.decode("utf-8", errors="replace") + marker


class ToolRegistry:
    """Holds every :class:`Tool` available to an agent and dispatches calls.

    Registration happens once at harness setup (via the factories in
    :mod:`harness.tools.builtin`); dispatch happens on the agent loop's hot
    path, once per :class:`~harness.types.ToolCall` the model emits.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add ``tool`` to the registry.

        Raises :class:`DuplicateToolError` if a tool with the same
        ``tool.spec.name`` is already registered -- silently overwriting a
        tool would make the registry order-dependent and could shadow the
        wrong handler.
        """
        name = tool.spec.name
        if name in self._tools:
            raise DuplicateToolError(f"tool already registered: {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Fetch the registered tool named ``name``.

        Raises :class:`KeyError` if no such tool is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    def specs(self) -> list[ToolSpec]:
        """Return every registered tool's :class:`~harness.types.ToolSpec`.

        This is what gets handed to the model adapter's ``tools`` parameter
        each turn (DESIGN.md §4.2).
        """
        return [tool.spec for tool in self._tools.values()]

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute ``call`` and return its :class:`~harness.types.ToolResult`.

        Never raises: an unknown tool name and any exception raised by the
        handler both become ``ToolResult(is_error=True, ...)`` with a clear
        message, so one bad tool call can never crash the agent loop
        (DESIGN.md §4.1). Successful results over :data:`MAX_RESULT_BYTES`
        are truncated with a marker.
        """
        try:
            tool = self._tools[call.name]
        except KeyError:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name!r}",
                is_error=True,
            )
        try:
            content = await tool.handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            return ToolResult(
                tool_call_id=call.id,
                content=f"tool {call.name!r} raised {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=call.id,
            content=_truncate_result(content),
            is_error=False,
        )
