"""Scripted fake adapter for tests and smoke runs (DESIGN.md §4.2).

:class:`FakeAdapter` replays a fixed script of :class:`ModelResponse` objects
— given directly or loaded from a JSONL file — returning them in order,
recording every ``complete()`` call for assertions, and raising a clear
:class:`~harness.adapters.base.AdapterError` when the script runs out.

JSONL script format, one JSON object per line::

    {"content": "thinking...", "tool_calls": [{"name": "bash", "arguments": {"cmd": "ls"}}]}
    {"content": "all done"}

Lines with ``tool_calls`` become :attr:`StopReason.TOOL_USE` responses;
lines without become :attr:`StopReason.END_TURN`. Call ids are generated
(``fake_<line>_<index>``) unless a tool call provides an ``"id"``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.adapters.base import AdapterError, ModelAdapter
from harness.types import (
    Capabilities,
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = ["FakeAdapter", "RecordedCall"]


@dataclass
class RecordedCall:
    """Snapshot of one ``complete()`` invocation, kept for test assertions."""

    messages: list[Message]
    tools: list[ToolSpec]
    system: str | None
    params: dict[str, Any] = field(default_factory=dict)


def _response_from_line(line: str, line_number: int) -> ModelResponse:
    """Build one scripted :class:`ModelResponse` from a JSONL script line."""
    try:
        data = json.loads(line)
    except ValueError as exc:
        raise ValueError(
            f"invalid JSON on line {line_number} of FakeAdapter script: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"line {line_number} of FakeAdapter script must be a JSON "
            f"object, got {type(data).__name__}"
        )
    tool_calls: list[ToolCall] = []
    for index, entry in enumerate(data.get("tool_calls") or []):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                f"line {line_number} of FakeAdapter script has a malformed "
                f"tool_calls entry at index {index}: expected a JSON object "
                f"with a 'name' key, got {entry!r}"
            )
        tool_calls.append(
            ToolCall(
                id=entry.get("id", f"fake_{line_number}_{index}"),
                name=entry["name"],
                arguments=entry.get("arguments", {}),
            )
        )
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            content=data.get("content"),
            tool_calls=tool_calls,
        ),
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
    )


class FakeAdapter(ModelAdapter):
    """A deterministic, scripted stand-in for a real model adapter.

    Construct with either a sequence of :class:`ModelResponse` objects or a
    path to a JSONL script file (see module docstring for the line format).
    Every ``complete()`` call is appended to :attr:`calls`; responses are
    returned in script order (as deep copies, so callers cannot mutate the
    script), and a call past the end of the script raises a clear
    :class:`AdapterError`.
    """

    def __init__(
        self, script: Sequence[ModelResponse] | str | Path = ()
    ) -> None:
        if isinstance(script, (str, Path)):
            path = Path(script)
            text = path.read_text(encoding="utf-8")
            self._responses = [
                _response_from_line(line, number)
                for number, line in enumerate(text.splitlines(), start=1)
                if line.strip()
            ]
        else:
            self._responses = list(script)
        #: Every ``complete()`` invocation, in order, for test assertions.
        self.calls: list[RecordedCall] = []
        self._next = 0

    @property
    def capabilities(self) -> Capabilities:
        """Permissive capabilities so no harness feature is gated off."""
        return Capabilities(
            max_context=1_000_000,
            supports_cache_control=False,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        """Record the call and return the next scripted response.

        Raises :class:`AdapterError` (non-retryable) when the script is
        exhausted, stating how many responses were scripted.
        """
        self.calls.append(
            RecordedCall(
                messages=list(messages),
                tools=list(tools),
                system=system,
                params=dict(params),
            )
        )
        if self._next >= len(self._responses):
            raise AdapterError(
                f"FakeAdapter script exhausted: {len(self._responses)} "
                f"scripted response(s), but complete() was called "
                f"{len(self.calls)} time(s)"
            )
        response = self._responses[self._next]
        self._next += 1
        return response.model_copy(deep=True)
