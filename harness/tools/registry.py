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
from harness.persistence import RunStore
from harness.types import ToolCall, ToolResult, ToolSpec

__all__ = [
    "MAX_RESULT_BYTES",
    "ToolHandler",
    "SpillWriter",
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

#: Writes a tool result that overflowed :data:`MAX_RESULT_BYTES` somewhere
#: the model can read it back, and returns the path it wrote (absolute,
#: inside the sandbox — see :data:`harness.sandbox.base.SPILL_DIR`).
#: :func:`harness.sandbox.base.spill_tool_output` is the implementation the
#: orchestrator wires in; the registry only holds the callable, so a
#: registry built without one (every pre-existing construction site) keeps
#: today's truncate-and-warn behaviour exactly. Free to raise: the registry
#: treats any failure as "no spill".
SpillWriter = Callable[[str], Awaitable[str]]


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


def _truncate_result(content: str, spill_path: str | None = None) -> str:
    """Truncate ``content`` to :data:`MAX_RESULT_BYTES`, marker on overflow.

    Mirrors :func:`harness.sandbox.base.truncate_output`: encode as UTF-8,
    keep at most the byte limit, decode leniently, and append a marker
    naming the limit and the true original size so the truncation is never
    mistaken for the whole result.

    ``spill_path``, when given, is where the *full* result was written
    inside the sandbox, and the marker changes from a warning into an
    instruction with an object: it names that path and the exact next call
    to make. The distinction is deliberate and measured — the bare marker
    below was ignored 12 times out of 12, and "advisory" and "unactionable"
    are two different diagnoses for that. Keeping both markers in the code,
    selected only by whether a spill actually happened, is what lets the
    two be told apart in the run data instead of conflated.
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= MAX_RESULT_BYTES:
        return content
    head = encoded[:MAX_RESULT_BYTES]
    if spill_path is None:
        marker = (
            f"\n...[tool result truncated at {MAX_RESULT_BYTES} bytes; "
            f"{len(encoded)} bytes total]...\n"
        )
    else:
        marker = (
            f"\n...[tool result truncated at {MAX_RESULT_BYTES} bytes; "
            f"{len(encoded)} bytes total. The full output is in "
            f"{spill_path} -- read the part you need with bash, e.g. "
            f"`grep -n PATTERN {spill_path}` or "
            f"`sed -n '1,200p' {spill_path}`, rather than re-running this "
            f"call]...\n"
        )
    return head.decode("utf-8", errors="replace") + marker


class ToolRegistry:
    """Holds every :class:`Tool` available to an agent and dispatches calls.

    Registration happens once at harness setup (via the factories in
    :mod:`harness.tools.builtin`); dispatch happens on the agent loop's hot
    path, once per :class:`~harness.types.ToolCall` the model emits.
    """

    def __init__(
        self,
        *,
        spill: SpillWriter | None = None,
        store: RunStore | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Build an empty registry.

        ``spill`` (see :data:`SpillWriter`) makes an oversized result
        retrievable: the full text is written into the sandbox and the
        truncation marker names that path plus the call to read it back.
        ``None`` -- the default, and every construction site that predates
        this -- keeps the plain truncate-and-warn marker. ``store`` and
        ``agent_id``, when both given, record each spill as a
        ``tool_output_spilled`` transcript event on that agent's stream
        (measurement first: whether a named path is actually read is the
        question the mechanism exists to answer).
        """
        self._tools: dict[str, Tool] = {}
        self._spill = spill
        self._store = store
        self._agent_id = agent_id

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

    async def _spill_full_result(
        self, tool_name: str, content: str, full_bytes: int
    ) -> str | None:
        """Write the pre-truncation ``content`` out; return its path or None.

        Fails open in every direction: no spill writer configured, a writer
        that raises, a writer that returns nothing, or a failed transcript
        write all end with the caller falling back to the plain truncation
        marker. Truncation exists to protect the context window, so it must
        never be the step that turns a successful tool call into an error.
        """
        if self._spill is None:
            return None
        try:
            path = await self._spill(content)
        except Exception:  # noqa: BLE001 - fail open; see docstring
            return None
        if not path:
            return None
        if self._store is not None and self._agent_id is not None:
            try:
                self._store.append_event(
                    self._agent_id,
                    "tool_output_spilled",
                    {
                        "tool": tool_name,
                        "full_bytes": full_bytes,
                        "shown_bytes": MAX_RESULT_BYTES,
                        "path": path,
                    },
                )
            except Exception:  # noqa: BLE001 - telemetry never breaks a call
                pass
        return path

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute ``call`` and return its :class:`~harness.types.ToolResult`.

        Never raises: an unknown tool name and any exception raised by the
        handler both become ``ToolResult(is_error=True, ...)`` with a clear
        message, so one bad tool call can never crash the agent loop
        (DESIGN.md §4.1). Successful results over :data:`MAX_RESULT_BYTES`
        are truncated with a marker; if this registry was built with a
        ``spill`` writer, the full result is first written into the sandbox
        and the marker names that path and how to read it.
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
        full_bytes = len(content.encode("utf-8"))
        spill_path: str | None = None
        if full_bytes > MAX_RESULT_BYTES:
            spill_path = await self._spill_full_result(
                call.name, content, full_bytes
            )
        return ToolResult(
            tool_call_id=call.id,
            content=_truncate_result(content, spill_path),
            is_error=False,
        )
