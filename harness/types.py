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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Role",
    "StopReason",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "DROPPED_ARGUMENTS_PREFIX_CHARS",
    "DroppedToolCall",
    "Message",
    "Usage",
    "IncompleteReason",
    "ModelResponse",
    "Capabilities",
]

#: Why an adapter judged a turn to have ended without producing anything the
#: agent loop can act on. The loop selects its re-prompt wording from this, so
#: every member must correspond to advice a model can actually follow:
#:
#: - ``"dropped_calls"`` — at least one tool call was discarded because its
#:   arguments could not be parsed (typically cut off mid-JSON).
#: - ``"max_tokens"`` — the turn hit the output-token cap before producing a
#:   tool call or a complete answer.
#: - ``"no_finish_reason"`` — the provider ended the response without a stop
#:   reason we recognise *and* left nothing usable in the message.
IncompleteReason = Literal[
    "max_tokens",
    "dropped_calls",
    "no_finish_reason",
    "refusal",
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


#: Cap on :attr:`DroppedToolCall.raw_arguments_prefix`: enough to see where
#: the provider cut the payload off, without carrying — or persisting — whole
#: truncated file bodies (one observed fragment was ~21 KB). Adapters trim to
#: this when reporting a drop and the agent loop trims again when writing the
#: ``tool_call_dropped`` event, so neither side can widen it alone.
DROPPED_ARGUMENTS_PREFIX_CHARS: int = 512


class DroppedToolCall(BaseModel):
    """A tool call the adapter discarded because it could not be parsed.

    Providers cut a response off mid-generation — at the output-token cap, or
    because the stream died — and a tool call caught by that cut arrives with
    truncated JSON arguments. Raising would kill the run non-retryably over a
    payload that is already spent, so adapters *drop* the call and report it
    here instead; the agent loop persists one ``tool_call_dropped`` event per
    entry, which is the only durable trace such a call leaves.

    ``raw_arguments_prefix`` holds the leading characters of the provider's
    unparseable argument string (enough to see where it was cut) and
    ``raw_arguments_len`` the full length that prefix was taken from, so a
    21 KB fragment is diagnosable without being stored.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    raw_arguments_prefix: str = ""
    raw_arguments_len: int = 0


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

    ``reasoning_tokens`` counts hidden reasoning/thinking output reported by
    the provider (currently populated on ``openai_compat`` only, from
    ``usage.completion_tokens_details.reasoning_tokens``; Anthropic stays 0
    until the round-3 thinking-block passthrough lands). Unlike the cache
    fields above, this is **not** subtracted from anything: it is a *subset*
    of ``output_tokens``, not a sibling bucket, because the provider already
    includes reasoning tokens in ``completion_tokens``/``output_tokens``.
    ``output_tokens`` is therefore left as-is — do not subtract
    ``reasoning_tokens`` from it the way ``cache_read_tokens`` /
    ``cache_write_tokens`` are subtracted out of OpenAI's cache-inclusive
    ``prompt_tokens`` when computing ``input_tokens``. This is pure telemetry
    for now: it has no effect on behaviour, config, or token accounting.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: object) -> "Usage":
        """Field-wise sum, so per-call usage can be rolled up per run/agent."""
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
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

    ``stop_reason`` is the harness's normalized enum; ``provider_stop_reason``
    is the provider's **untranslated** stop string, verbatim (OpenAI
    ``finish_reason``, Anthropic ``stop_reason``), or ``None`` when the
    provider sent none. It exists because normalization is lossy: every
    unknown or missing value collapses to :attr:`StopReason.ERROR`, so
    "the stream ended with no terminal chunk", "the provider said ``stop``",
    and "the provider sent a value we do not map" are indistinguishable after
    mapping — and that distinction is exactly what post-hoc triage of a dead
    turn needs. The agent loop persists it on each ``model_turn`` event,
    which is the only place it is durably recorded (``raw`` is not
    persisted). Nothing above the adapter layer may *branch* on its value:
    it is evidence, not control flow.

    ``incomplete`` says the turn ended without producing anything the loop can
    act on, and ``incomplete_reason`` says why (see :data:`IncompleteReason`).
    It is deliberately *not* derived from ``stop_reason`` at the loop: the two
    answer different questions. ``stop_reason`` is a faithful translation of
    what the provider said; ``incomplete`` is the adapter's judgement about
    whether the translated message is usable — a turn can stop at
    ``MAX_TOKENS`` and still carry a perfectly good tool call (not incomplete),
    and one can stop at ``END_TURN`` having had its only tool call dropped
    (incomplete). Keeping both means ``stop_reason`` stays truthful while the
    loop still knows which of the three re-prompts to send.

    ``dropped_tool_calls`` lists the calls the adapter discarded while
    translating (see :class:`DroppedToolCall`). Translation is
    side-effect-free by module contract, so adapters only *report* drops here;
    persisting them is the agent loop's job.

    ``raw`` optionally holds the provider's original response (as a dict) for
    debugging and trace logging; nothing above the adapter layer may depend
    on its shape.
    """

    message: Message
    usage: Usage
    stop_reason: StopReason
    provider_stop_reason: str | None = None
    incomplete: bool = False
    incomplete_reason: IncompleteReason | None = None
    dropped_tool_calls: list[DroppedToolCall] = Field(default_factory=list)
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
