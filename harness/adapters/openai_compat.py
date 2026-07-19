"""OpenAI-compatible model adapter (DESIGN.md §4.2).

Speaks the plain ``chat.completions`` dialect with ``function`` tools via the
``openai`` AsyncOpenAI SDK, with a configurable ``base_url`` — the same code
path covers api.openai.com and any OpenAI-compatible endpoint (validated
against ``https://api.moonshot.ai/v1`` with model ``kimi-k3``). No
OpenAI-only exotica (responses API, structured outputs, etc.).

Translation notes:

- The system prompt rides as a leading ``role="system"`` message.
- Tool results ride as ``role="tool"`` messages with ``tool_call_id``; the
  format has no error flag, so ``is_error`` results are prefixed ``Error:``.
- Tool-call ``arguments`` arrive as a JSON *string*; malformed JSON from a
  provider surfaces as a clear non-retryable
  :class:`~harness.adapters.base.AdapterError`, never a raw crash.
- SDK failures are wrapped with ``retryable`` derived from HTTP status
  (429/5xx/timeouts retry; 400/401-class do not) and ``complete()`` runs
  under :func:`~harness.adapters.base.retry_with_backoff`.

Translation functions are module-level and side-effect free for direct
unit testing without a client or network.
"""

from __future__ import annotations

import json
from typing import Any

from harness.adapters.base import AdapterError, ModelAdapter, retry_with_backoff
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

__all__ = [
    "OpenAICompatAdapter",
    "to_openai_messages",
    "to_openai_tools",
    "from_openai_response",
    "map_finish_reason",
    "wrap_openai_error",
]

#: Provider finish reasons -> harness :class:`StopReason`.
_FINISH_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


def map_finish_reason(finish_reason: str | None) -> StopReason:
    """Map an OpenAI ``finish_reason`` string to a harness :class:`StopReason`.

    Unknown or missing values map to :attr:`StopReason.ERROR` so new provider
    finish reasons fail loudly rather than masquerading as clean ends.
    """
    if finish_reason is None:
        return StopReason.ERROR
    return _FINISH_REASONS.get(finish_reason, StopReason.ERROR)


def to_openai_messages(
    messages: list[Message], system: str | None = None
) -> list[dict[str, Any]]:
    """Translate harness messages to chat.completions message dicts.

    ``system``, if given, is prepended as a ``role="system"`` message
    (``system``-role messages inside ``messages`` are also honored).
    Assistant tool calls are serialized with JSON-string ``arguments``; tool
    results become ``role="tool"`` messages with ``tool_call_id``, with
    ``is_error`` results prefixed ``Error:`` since the format has no flag.
    A non-tool message with neither ``content`` nor ``tool_calls`` raises
    :class:`AdapterError` at translation time (mirroring the Anthropic
    adapter) instead of silently emitting ``content: null``.
    """
    out: list[dict[str, Any]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role is Role.TOOL:
            if message.tool_result is None:
                raise AdapterError(
                    "tool-role message has no tool_result; cannot translate "
                    "to an OpenAI tool message"
                )
            result = message.tool_result
            content = result.content
            if result.is_error and not content.startswith("Error:"):
                content = f"Error: {content}"
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": content,
                }
            )
            continue
        if not message.content and not message.tool_calls:
            raise AdapterError(
                f"message with role {message.role.value!r} has no content "
                "or tool calls; cannot translate"
            )
        entry: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.tool_calls:
            if message.role is not Role.ASSISTANT:
                raise AdapterError(
                    f"only assistant messages may carry tool calls, got role "
                    f"{message.role.value!r}"
                )
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        out.append(entry)
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Translate harness tool specs to chat.completions ``function`` tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _parse_arguments(name: str, raw: Any) -> dict:
    """Parse a tool call's JSON-string arguments defensively.

    Providers occasionally emit malformed JSON; that must surface as a clear
    :class:`AdapterError` (non-retryable — the payload is already consumed),
    not a raw ``json`` crash. ``None``/empty means no arguments, and a
    non-object payload is likewise rejected.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):  # some compatible providers pre-parse
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AdapterError(
            f"provider returned malformed JSON arguments for tool call "
            f"{name!r}: {exc}: {raw!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise AdapterError(
            f"provider returned non-object JSON arguments for tool call "
            f"{name!r}: {parsed!r}"
        )
    return parsed


def from_openai_response(response: Any) -> ModelResponse:
    """Translate a chat.completions response to a :class:`ModelResponse`.

    ``response`` is duck-typed (SDK object or same-shaped stand-in):
    ``choices[0].message`` with ``content``/``tool_calls``,
    ``choices[0].finish_reason``, and ``usage`` with
    ``prompt_tokens``/``completion_tokens`` (cache read/write tokens read
    from ``usage.prompt_tokens_details.cached_tokens``/``cache_write_tokens``
    when present).

    Usage normalization: the OpenAI API's ``prompt_tokens`` *includes* cache
    traffic (``prompt_tokens_details`` fields are subsets of it), but
    :class:`~harness.types.Usage` defines ``input_tokens`` as *excluding*
    cache reads/writes (the Anthropic convention — see the ``Usage``
    docstring). So cache tokens are subtracted from ``prompt_tokens`` here,
    clamped at zero for providers that report cache counts outside the
    prompt total.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise AdapterError("provider response contained no choices")
    choice = choices[0]
    provider_message = choice.message
    tool_calls: list[ToolCall] = []
    for call in getattr(provider_message, "tool_calls", None) or []:
        function = call.function
        tool_calls.append(
            ToolCall(
                id=call.id,
                name=function.name,
                arguments=_parse_arguments(function.name, function.arguments),
            )
        )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read_tokens = getattr(details, "cached_tokens", 0) or 0
    cache_write_tokens = getattr(details, "cache_write_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    # prompt_tokens is cache-inclusive per the OpenAI API; Usage.input_tokens
    # is cache-exclusive by convention, so peel the cache traffic off here.
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
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
            content=getattr(provider_message, "content", None) or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        stop_reason=map_finish_reason(getattr(choice, "finish_reason", None)),
        raw=raw,
    )


def wrap_openai_error(exc: Exception) -> AdapterError:
    """Wrap an SDK exception in an :class:`AdapterError` with ``retryable`` set.

    Classification: HTTP 408/429 and all 5xx are retryable; other statuses
    (400 invalid request, 401 auth, ...) are not. Statusless connection or
    timeout failures are retryable. Anything else is non-retryable.
    """
    if isinstance(exc, AdapterError):
        return exc
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        retryable = status in (408, 429) or status >= 500
        return AdapterError(
            f"openai-compatible API error (HTTP {status}): {exc}",
            retryable=retryable,
        )
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name or "Connection" in name:
        return AdapterError(
            f"openai-compatible connection error: {exc}", retryable=True
        )
    return AdapterError(f"openai SDK error: {name}: {exc}", retryable=False)


class OpenAICompatAdapter(ModelAdapter):
    """Model adapter for OpenAI-compatible chat.completions endpoints.

    ``base_url`` points the SDK at any compatible endpoint (e.g.
    ``https://api.moonshot.ai/v1`` for Kimi). ``client`` is injectable for
    tests; when omitted, an ``openai.AsyncOpenAI`` client is built from
    ``api_key``/``base_url``. ``retry`` overrides keyword arguments to
    :func:`retry_with_backoff`.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: Any | None = None,
        max_context: int = 128_000,
        retry: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._retry = dict(retry or {})
        self._capabilities = Capabilities(
            parallel_tool_calls=True,
            max_context=max_context,
            supports_cache_control=False,
        )

    @property
    def capabilities(self) -> Capabilities:
        """chat.completions: parallel tools, no explicit cache control."""
        return self._capabilities

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        """Run one chat.completions call with retry and translation.

        Extra ``params`` (temperature, max_tokens, ...) pass through to
        ``chat.completions.create``. Raises :class:`AdapterError` on
        failure, including malformed tool-call JSON from the provider.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(messages, system),
            **params,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)

        async def _call() -> Any:
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                raise wrap_openai_error(exc) from exc

        response = await retry_with_backoff(_call, **self._retry)
        return from_openai_response(response)
