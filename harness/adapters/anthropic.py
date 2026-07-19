"""Anthropic model adapter (DESIGN.md §4.2).

Translates the harness's provider-neutral types to and from the Anthropic
Messages API via the ``anthropic`` AsyncAnthropic SDK:

- ``system`` rides as the top-level ``system`` parameter (as content blocks).
- Assistant tool calls become ``tool_use`` content blocks; tool results become
  ``tool_result`` blocks inside ``user``-role messages.
- ``cache_control: ephemeral`` breakpoints are set on the system prompt and on
  the last transcript message — per DESIGN.md the single biggest cost lever
  for long runs (every turn re-reads the whole stable prefix from cache).
- SDK failures are wrapped in :class:`~harness.adapters.base.AdapterError`
  with ``retryable`` set from the HTTP status (429/5xx/timeouts retry;
  400/401-class errors do not), and ``complete()`` runs under the shared
  :func:`~harness.adapters.base.retry_with_backoff` policy.

The translation functions are module-level and side-effect free so they can
be unit-tested without a client or network.
"""

from __future__ import annotations

from typing import Any

from harness.adapters.base import AdapterError, ModelAdapter, retry_with_backoff
from harness.types import (
    Capabilities,
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "AnthropicAdapter",
    "to_anthropic_messages",
    "to_anthropic_system",
    "to_anthropic_tools",
    "from_anthropic_response",
    "map_stop_reason",
    "wrap_anthropic_error",
]

#: Anthropic requires ``max_tokens``; used when the caller does not pass one.
DEFAULT_MAX_TOKENS = 8192

#: ``cache_control`` value for ephemeral prompt-cache breakpoints.
_EPHEMERAL = {"type": "ephemeral"}

#: Provider stop reasons -> harness :class:`StopReason`.
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "pause_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}


def map_stop_reason(stop_reason: str | None) -> StopReason:
    """Map an Anthropic ``stop_reason`` string to a harness :class:`StopReason`.

    Unknown or missing values map to :attr:`StopReason.ERROR` so new provider
    stop reasons fail loudly in traces rather than masquerading as clean ends.
    """
    if stop_reason is None:
        return StopReason.ERROR
    return _STOP_REASONS.get(stop_reason, StopReason.ERROR)


def _message_blocks(message: Message) -> list[dict[str, Any]]:
    """Render one harness message as a list of Anthropic content blocks."""
    blocks: list[dict[str, Any]] = []
    if message.role is Role.TOOL:
        if message.tool_result is None:
            raise AdapterError(
                "tool-role message has no tool_result; cannot translate to "
                "an Anthropic tool_result block"
            )
        result = message.tool_result
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        blocks.append(block)
        return blocks
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    return blocks


def to_anthropic_messages(
    messages: list[Message], *, cache: bool = True
) -> list[dict[str, Any]]:
    """Translate harness messages to Anthropic Messages API ``messages``.

    - ``tool``-role messages become ``user`` messages carrying a
      ``tool_result`` block; consecutive same-role messages are merged into
      one API message (Anthropic wants tool results grouped in the user turn
      that follows the tool-using assistant turn).
    - ``system``-role messages are rejected: the system prompt must ride as
      the top-level ``system`` parameter (see :func:`to_anthropic_system`).
    - When ``cache`` is true, a ``cache_control: ephemeral`` breakpoint is
      set on the final content block of the last message, marking the whole
      transcript so far as a cacheable stable prefix for the next turn.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            raise AdapterError(
                "system-role messages are not allowed in the transcript for "
                "the Anthropic adapter; pass the system prompt via the "
                "top-level 'system' parameter instead"
            )
        blocks = _message_blocks(message)
        if not blocks:
            raise AdapterError(
                f"message with role {message.role.value!r} has no content, "
                "tool calls, or tool result; cannot translate"
            )
        role = "user" if message.role in (Role.USER, Role.TOOL) else "assistant"
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    if cache and out:
        out[-1]["content"][-1]["cache_control"] = dict(_EPHEMERAL)
    return out


def to_anthropic_system(
    system: str | None, *, cache: bool = True
) -> list[dict[str, Any]] | None:
    """Render the system prompt as Anthropic system content blocks.

    Returns ``None`` when there is no system prompt. When ``cache`` is true
    the final block carries a ``cache_control: ephemeral`` breakpoint so the
    system prompt is a stable cached prefix.
    """
    if system is None:
        return None
    block: dict[str, Any] = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = dict(_EPHEMERAL)
    return [block]


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Translate harness tool specs to Anthropic tool definitions."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def from_anthropic_response(response: Any) -> ModelResponse:
    """Translate an Anthropic Messages API response to a :class:`ModelResponse`.

    ``response`` is duck-typed (SDK ``Message`` object or anything with the
    same attribute shape): ``content`` blocks with ``type``/``text``/``id``/
    ``name``/``input``, ``stop_reason``, and ``usage`` with token counts
    including ``cache_read_input_tokens``/``cache_creation_input_tokens``.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )
        # Unknown block types (e.g. future thinking blocks) are ignored.
    usage = getattr(response, "usage", None)
    raw: dict | None = None
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            raw = dump()
        except Exception:  # pragma: no cover - raw is best-effort debug data
            raw = None
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
        stop_reason=map_stop_reason(getattr(response, "stop_reason", None)),
        raw=raw,
    )


def wrap_anthropic_error(exc: Exception) -> AdapterError:
    """Wrap an SDK exception in an :class:`AdapterError` with ``retryable`` set.

    Classification: HTTP 408/429 and all 5xx (including Anthropic's 529
    ``overloaded_error``) are retryable; other statuses (400 invalid request,
    401 auth, ...) are not. Statusless connection/timeout failures are
    retryable. Anything else is treated as a permanent adapter bug.
    """
    if isinstance(exc, AdapterError):
        return exc
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        retryable = status in (408, 429) or status >= 500
        return AdapterError(
            f"anthropic API error (HTTP {status}): {exc}", retryable=retryable
        )
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name or "Connection" in name:
        return AdapterError(f"anthropic connection error: {exc}", retryable=True)
    return AdapterError(f"anthropic SDK error: {name}: {exc}", retryable=False)


class AnthropicAdapter(ModelAdapter):
    """Model adapter for the Anthropic Messages API.

    ``client`` is injectable for tests; when omitted, an
    ``anthropic.AsyncAnthropic`` client is built with ``api_key`` (``None``
    lets the SDK fall back to ``ANTHROPIC_API_KEY``) and ``base_url`` (``None``
    lets the SDK fall back to ``api.anthropic.com``; set it to point at a
    proxy/gateway per DESIGN.md's per-model registry field). ``retry``
    overrides keyword arguments to :func:`retry_with_backoff` (tests inject a
    no-op ``sleep``).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: Any | None = None,
        max_context: int = 200_000,
        retry: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._retry = dict(retry or {})
        self._capabilities = Capabilities(
            parallel_tool_calls=True,
            max_context=max_context,
            supports_cache_control=True,
        )

    @property
    def capabilities(self) -> Capabilities:
        """Anthropic models: parallel tools and prompt caching supported."""
        return self._capabilities

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        """Run one Messages API call with caching, retry, and translation.

        ``max_tokens`` defaults to :data:`DEFAULT_MAX_TOKENS` (the API
        requires it); other ``params`` pass through to
        ``messages.create``. Raises :class:`AdapterError` on failure.
        """
        cache = self._capabilities.supports_cache_control
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": params.pop("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": to_anthropic_messages(messages, cache=cache),
            **params,
        }
        system_blocks = to_anthropic_system(system, cache=cache)
        if system_blocks is not None:
            kwargs["system"] = system_blocks
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)

        async def _call() -> Any:
            try:
                return await self._client.messages.create(**kwargs)
            except Exception as exc:
                raise wrap_anthropic_error(exc) from exc

        response = await retry_with_backoff(_call, **self._retry)
        return from_anthropic_response(response)
