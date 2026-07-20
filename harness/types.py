"""Provider-neutral core types for the harness (DESIGN.md §4.2).

Every layer above the model adapters — the agent loop, context manager,
permission engine, persistence — speaks exclusively in these types. Adapters
translate to and from each provider's SDK at the boundary, so nothing here may
reference any provider concept.

All models are pydantic v2 with strict-ish validation; enums are plain
``str``-valued ``Enum`` subclasses so they serialize cleanly to JSON/SQLite.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Role",
    "StopReason",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "Message",
    "Usage",
    "ModelResponse",
    "Capabilities",
]


class Role(str, Enum):
    """Who authored a message in the transcript."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    """Why the model stopped generating, normalized across providers."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    ERROR = "error"


class ToolSpec(BaseModel):
    """Declaration of a tool the model may call.

    ``input_schema`` is a JSON Schema dict describing the tool's arguments;
    adapters translate it into each provider's tool-definition format.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A single tool invocation requested by the model.

    ``id`` is the provider-assigned call id; it is echoed back in the matching
    :class:`ToolResult` so providers can pair calls with results.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing one :class:`ToolCall`.

    ``content`` is the textual payload returned to the model; ``is_error``
    marks failures so adapters can flag them in provider-specific ways.
    """

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """One transcript entry in the provider-neutral conversation format.

    Shape conventions:

    - ``role=assistant`` messages may carry ``tool_calls`` (and optionally
      accompanying text in ``content``).
    - ``role=tool`` messages carry exactly one ``tool_result`` and typically
      no ``content`` (the payload lives on the result).
    - ``system``/``user`` messages carry only ``content``.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_result: ToolResult | None = None


class Usage(BaseModel):
    """Token accounting for a single model call, accumulable via ``+``.

    Convention: ``input_tokens`` counts uncached input only — it *excludes*
    prompt-cache reads and writes, which are tracked separately in
    ``cache_read_tokens``/``cache_write_tokens`` (they stay 0 on providers
    without prompt caching). Total input-side traffic is therefore always
    ``input_tokens + cache_read_tokens + cache_write_tokens``, regardless of
    adapter. This matches the Anthropic API's fields directly; adapters for
    APIs whose prompt total is cache-inclusive (e.g. OpenAI ``prompt_tokens``)
    must subtract cache traffic when mapping to ``input_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: object) -> "Usage":
        """Field-wise sum, so per-call usage can be rolled up per run/agent."""
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def __radd__(self, other: object) -> "Usage":
        """Treat the int ``0`` as additive identity so bare ``sum(usages)`` works."""
        if other == 0:
            return self.model_copy()
        if isinstance(other, Usage):
            return self.__add__(other)
        return NotImplemented


class ModelResponse(BaseModel):
    """Everything the harness needs back from one adapter ``complete()`` call.

    ``raw`` optionally holds the provider's original response (as a dict) for
    debugging and trace logging; nothing above the adapter layer may depend
    on its shape.
    """

    message: Message
    usage: Usage
    stop_reason: StopReason
    raw: dict | None = None


class Capabilities(BaseModel):
    """What a model/adapter pair can do, used for capability negotiation.

    The harness queries this rather than assuming a lowest common denominator:
    e.g. it sets cache breakpoints only when ``supports_cache_control`` is
    True.

    ``extra="forbid"``: constructing with an unknown field is an error, so
    a capability field that has been removed (e.g. the A3-deleted
    ``parallel_tool_calls``) fails loudly at every construction site
    instead of being silently swallowed by pydantic's default ignore.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_context: int
    supports_cache_control: bool
