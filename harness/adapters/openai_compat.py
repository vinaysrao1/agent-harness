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
  :class:`~harness.adapters.base.AdapterError`, never a raw crash — *except*
  when the turn was truncated at the output-token limit
  (``finish_reason == "length"``): a tool call cut off mid-arguments is then
  an expected casualty of truncation, so the malformed call is dropped and
  the turn survives as a plain ``MAX_TOKENS`` response the agent loop's
  truncation-continue path can re-prompt (see :func:`from_openai_response`).
  TODO: the Anthropic adapter has no analogue of this truncated-tool-call
  degradation (no current exposure); port it if truncation is observed there.
- SDK failures are wrapped with ``retryable`` derived from HTTP status
  (429/5xx/timeouts retry; 400/401-class do not) and ``complete()`` runs
  under :func:`~harness.adapters.base.retry_with_backoff`.
- Some gateways (notably OpenRouter) report *transient upstream* faults as an
  HTTP 200 whose body carries an inline ``error`` object and/or an empty
  ``choices`` list rather than as an HTTP error status. These are treated as
  retryable and — crucially — response translation runs *inside* the retried
  call (see :meth:`OpenAICompatAdapter.complete`), so a transient empty
  response is retried instead of killing the turn.
- The retry policy lives in exactly one layer: the SDK client is built with
  ``max_retries=0`` so :func:`~harness.adapters.base.retry_with_backoff` is
  the sole retrier (DESIGN.md §4.1), and with an explicit request ``timeout``
  so a hung upstream fails fast (as a retryable timeout) instead of blocking.

Translation functions are module-level and side-effect free for direct
unit testing without a client or network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
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
    "accumulate_stream_chunks",
    "map_finish_reason",
    "wrap_openai_error",
]

#: Default per-request SDK timeout (seconds): bounds one hung call so it
#: surfaces as a retryable timeout instead of blocking. Overridable per adapter.
#: Used as the whole-call ceiling on the non-streaming path only.
_DEFAULT_REQUEST_TIMEOUT = 120.0

#: Default *idle* timeout (seconds) for the streaming path: the maximum gap
#: allowed between successive streamed chunks (and to the first chunk). A
#: healthy generation — however long and however slow the provider's
#: throughput — keeps emitting tokens and never trips this; a genuine upstream
#: stall (a live socket producing no bytes) trips it fast and surfaces as a
#: retryable error. This is the fix for the failure mode where a *whole-call*
#: timeout could not tell a slow-but-healthy long generation apart from a
#: stall and guillotined both (see :meth:`OpenAICompatAdapter.complete`).
_DEFAULT_STREAM_IDLE_TIMEOUT = 60.0

#: Default wall-clock ceiling for the whole retry sequence of one ``complete()``
#: call, passed through to :func:`retry_with_backoff`. Keeps
#: ``request_timeout × max_attempts`` from overrunning an upstream agent-
#: execution deadline (e.g. a benchmark harness's per-agent timeout) when a
#: provider hangs on every attempt. Overridable via the ``retry`` kwarg.
_DEFAULT_RETRY_MAX_ELAPSED = 300.0

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


def _code_retryable(code: int | None) -> bool:
    """Whether a provider error code denotes a transient (retryable) fault.

    Mirrors :func:`wrap_openai_error`'s HTTP-status classification: 408/429 and
    5xx are transient, 4xx client errors are permanent. ``None`` (no usable
    code) defaults to retryable — an inline provider error without a code is
    most often a transient upstream hiccup, and the bounded retry policy makes
    the occasional wasted retry cheap. Non-HTTP codes (e.g. OpenRouter's
    negative sentinels) are likewise treated as transient.
    """
    if code is None:
        return True
    # code >= 500 is treated as transient wholesale (mirrors wrap_openai_error's
    # HTTP-status rule). This is deliberately over-inclusive — a permanent 501
    # would burn the bounded retry budget — but keeping one classification rule
    # for gateway codes and HTTP statuses is worth the rare wasted retries.
    if code in (408, 429) or code >= 500:
        return True
    if 400 <= code < 500:
        return False
    return True


def _provider_error(response: Any) -> tuple[int | None, str] | None:
    """Extract an inline gateway ``error`` object from a 200-status response.

    OpenRouter (and some other gateways) return HTTP 200 with an ``error``
    object in the body — and usually empty ``choices`` — when the upstream
    provider rate-limits or fails, rather than a proper HTTP error status. The
    SDK surfaces it as ``response.error``. Returns ``(code, message)`` when
    present (``code`` is ``None`` if not an int), else ``None``. Duck-typed to
    accept both dict and attribute-style error payloads.
    """
    error = getattr(response, "error", None)
    if not error:
        return None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or str(error)
    else:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None) or str(error)
    return (code if isinstance(code, int) else None), message


def from_openai_response(response: Any) -> ModelResponse:
    """Translate a chat.completions response to a :class:`ModelResponse`.

    ``response`` is duck-typed (SDK object or same-shaped stand-in):
    ``choices[0].message`` with ``content``/``tool_calls``,
    ``choices[0].finish_reason``, and ``usage`` with
    ``prompt_tokens``/``completion_tokens`` (cache read/write tokens read
    from ``usage.prompt_tokens_details.cached_tokens``/``cache_write_tokens``
    when present).

    Transient responses: an inline gateway ``error`` object (see
    :func:`_provider_error`) or an empty ``choices`` list is raised as a
    **retryable** :class:`AdapterError`, because these signal a transient
    upstream fault rather than a malformed reply. :meth:`OpenAICompatAdapter.complete`
    runs this translation inside its retried call, so such faults are retried.
    Malformed tool-call JSON, by contrast, stays non-retryable (the payload is
    already consumed and will not change on retry).

    Truncation graceful degradation: when ``finish_reason == "length"`` (the
    turn hit the output-token limit) a tool call with malformed arguments is
    almost certainly one cut off mid-JSON by the cap, not a provider fault.
    Raising would kill the whole run non-retryably (observed on a real trial:
    a large inline ``write_file`` truncated at the output cap), and retrying
    cannot help — the payload is spent. So each such call is *dropped* from
    the translated message instead, and the response reaches the agent loop
    as ``stop_reason=MAX_TOKENS`` with the malformed calls gone, where the
    existing truncation-continue path re-prompts the model to re-issue the
    call. If dropping leaves the message with no content and no tool calls,
    a placeholder content string is substituted, because
    :func:`to_openai_messages` correctly rejects empty messages when the
    transcript is replayed next turn (an assistant message must not dangle
    without its tool calls). Any other ``finish_reason`` with malformed
    arguments keeps the non-retryable :class:`AdapterError` — a genuine
    provider fault, not truncation.

    Usage normalization: the OpenAI API's ``prompt_tokens`` *includes* cache
    traffic (``prompt_tokens_details`` fields are subsets of it), but
    :class:`~harness.types.Usage` defines ``input_tokens`` as *excluding*
    cache reads/writes (the Anthropic convention — see the ``Usage``
    docstring). So cache tokens are subtracted from ``prompt_tokens`` here,
    clamped at zero for providers that report cache counts outside the
    prompt total.
    """
    provider_error = _provider_error(response)
    if provider_error is not None:
        code, message = provider_error
        suffix = f" (code {code})" if code is not None else ""
        raise AdapterError(
            f"provider returned an inline error{suffix}: {message}",
            retryable=_code_retryable(code),
        )
    choices = getattr(response, "choices", None)
    if not choices:
        raise AdapterError(
            "provider response contained no choices", retryable=True
        )
    choice = choices[0]
    provider_message = choice.message
    stop_reason = map_finish_reason(getattr(choice, "finish_reason", None))
    truncated = stop_reason is StopReason.MAX_TOKENS
    tool_calls: list[ToolCall] = []
    dropped_calls = 0
    for call in getattr(provider_message, "tool_calls", None) or []:
        function = call.function
        try:
            arguments = _parse_arguments(function.name, function.arguments)
        except AdapterError:
            if truncated:
                # A tool call cut off mid-arguments at the output-token cap:
                # drop it so the turn survives as MAX_TOKENS and the loop's
                # truncation-continue path re-prompts (see docstring).
                dropped_calls += 1
                continue
            raise
        tool_calls.append(
            ToolCall(id=call.id, name=function.name, arguments=arguments)
        )
    content = getattr(provider_message, "content", None) or None
    if dropped_calls and content is None and not tool_calls:
        # Dropping left the assistant message empty; substitute a placeholder
        # so next-turn transcript translation does not reject it.
        content = (
            "(response truncated at the output-token limit while emitting "
            "a tool call)"
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
            content=content,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        stop_reason=stop_reason,
        raw=raw,
    )


def accumulate_stream_chunks(chunks: Any) -> Any:
    """Reduce streamed ``chat.completions`` chunks into a full-response shape.

    The streaming API delivers a completion as a sequence of delta chunks —
    ``choices[0].delta.content`` fragments, ``choices[0].delta.tool_calls``
    fragments (each carrying an ``index`` and partial ``id``/``function.name``/
    ``function.arguments``), a terminal ``finish_reason``, and — when
    ``stream_options={"include_usage": True}`` is requested — a final
    usage-only chunk (``choices == []``). This function folds them back into an
    object shaped exactly like a non-streamed response so
    :func:`from_openai_response` can translate it **unchanged** — keeping one
    translation path for both modes.

    ``chunks`` is any iterable of duck-typed chunk objects (SDK objects or
    same-shaped stand-ins). Text fragments are concatenated in arrival order;
    tool-call fragments are merged by ``index`` (``id``/``name`` take the first
    non-empty value, ``arguments`` fragments are concatenated). A stream that
    yields no content, no tool calls, no ``finish_reason`` and no inline error
    collapses to an empty ``choices`` list, so
    :func:`from_openai_response` raises the same *retryable* "no choices" error
    an empty non-streamed reply would — a content-less stream is a transient
    fault, not a clean empty turn.
    """
    content_parts: list[str] = []
    tool_frags: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: Any = None
    error: Any = None

    for chunk in chunks:
        chunk_error = getattr(chunk, "error", None)
        if chunk_error:
            error = chunk_error
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        reason = getattr(choice, "finish_reason", None)
        if reason is not None:
            finish_reason = reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)
        for call in getattr(delta, "tool_calls", None) or []:
            index = getattr(call, "index", 0) or 0
            frag = tool_frags.setdefault(
                index, {"id": None, "name": None, "arguments": ""}
            )
            call_id = getattr(call, "id", None)
            if call_id:
                frag["id"] = call_id
            function = getattr(call, "function", None)
            if function is not None:
                name = getattr(function, "name", None)
                if name:
                    frag["name"] = name
                arguments = getattr(function, "arguments", None)
                if arguments:
                    frag["arguments"] += arguments

    # A stream carrying nothing usable is a transient empty reply: emit empty
    # choices so translation raises the retryable "no choices" error, matching
    # the non-streamed empty-response contract.
    if not content_parts and not tool_frags and finish_reason is None and not error:
        empty = SimpleNamespace(choices=[], usage=usage)
        if error:
            empty.error = error
        return empty

    tool_calls = [
        SimpleNamespace(
            id=tool_frags[index]["id"] or "",
            function=SimpleNamespace(
                name=tool_frags[index]["name"] or "",
                arguments=tool_frags[index]["arguments"] or "",
            ),
        )
        for index in sorted(tool_frags)
    ]
    message = SimpleNamespace(
        content="".join(content_parts) if content_parts else None,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    response = SimpleNamespace(choices=[choice], usage=usage)
    if error:
        response.error = error
    return response


#: Substrings of exception *class names* that denote a transient transport
#: fault worth retrying. These are the httpx/openai transport errors that can
#: interrupt a call — and, because streaming reads the response body
#: incrementally, are markedly more likely mid-stream than on a buffered
#: whole-call read: a dropped/reset connection surfaces as ``ReadError`` or
#: ``RemoteProtocolError`` partway through the token stream. A statusless
#: transport error is transient by nature (the request may simply not have
#: been served), so retrying is safe and correct; classifying ``ReadError`` as
#: permanent forfeited whole trials on a single mid-stream blip.
_TRANSIENT_ERROR_MARKERS = (
    "Timeout",  # ReadTimeout, ConnectTimeout, PoolTimeout, APITimeoutError
    "Connection",  # ConnectError, APIConnectionError
    "ReadError",
    "WriteError",
    "ProtocolError",  # Remote/LocalProtocolError
    "StreamError",
    "IncompleteRead",
)


def wrap_openai_error(exc: Exception) -> AdapterError:
    """Wrap an SDK exception in an :class:`AdapterError` with ``retryable`` set.

    Classification: HTTP 408/429 and all 5xx are retryable; other statuses
    (400 invalid request, 401 auth, ...) are not. Statusless transport faults
    — connection errors, timeouts, and mid-stream read/protocol errors (see
    :data:`_TRANSIENT_ERROR_MARKERS`) — are retryable. Anything else is
    non-retryable.
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
    if isinstance(exc, TimeoutError) or any(
        marker in name for marker in _TRANSIENT_ERROR_MARKERS
    ):
        return AdapterError(
            f"openai-compatible transport error ({name}): {exc}", retryable=True
        )
    return AdapterError(f"openai SDK error: {name}: {exc}", retryable=False)


class OpenAICompatAdapter(ModelAdapter):
    """Model adapter for OpenAI-compatible chat.completions endpoints.

    ``base_url`` points the SDK at any compatible endpoint (e.g.
    ``https://api.moonshot.ai/v1`` for Kimi). ``client`` is injectable for
    tests; when omitted, an ``openai.AsyncOpenAI`` client is built from
    ``api_key``/``base_url`` with an explicit ``request_timeout`` and
    ``max_retries=0`` — the latter keeps :func:`retry_with_backoff` the single
    retry layer (the SDK defaults to retrying twice on its own, which would
    both violate that invariant and compound latency against an upstream
    agent-execution deadline). ``request_timeout`` bounds a single call so a
    hung upstream surfaces as a retryable timeout rather than blocking
    indefinitely; it is ignored when an explicit ``client`` is injected.
    ``retry`` overrides keyword arguments to :func:`retry_with_backoff`.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: Any | None = None,
        max_context: int = 128_000,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        stream: bool = True,
        stream_idle_timeout: float = _DEFAULT_STREAM_IDLE_TIMEOUT,
        extra_body: dict[str, Any] | None = None,
        retry: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout,
                max_retries=0,
            )
        self._client = client
        self._model = model
        # Streaming (default) vs. whole-call mode. Streaming replaces the
        # whole-call ``request_timeout`` deadline with a per-chunk *idle*
        # timeout (:data:`_DEFAULT_STREAM_IDLE_TIMEOUT`), which is the only way
        # to bound a genuine upstream stall *without* also killing a healthy
        # long generation from a slow provider — a whole-call timeout cannot
        # tell the two apart. Non-streaming mode is retained for endpoints or
        # tests that do not stream; it keeps the hard whole-call backstop.
        self._stream = stream
        self._stream_idle_timeout = stream_idle_timeout
        # Provider-specific request fields (e.g. a reasoning/thinking control)
        # merged into every request's ``extra_body``. Kept out of the typed
        # ``create`` kwargs so an arbitrary gateway field rides through the SDK
        # untouched; an unsupported field is ignored by the gateway.
        self._extra_body = extra_body
        # A hard per-attempt deadline enforced with asyncio.wait_for, so a
        # single in-flight call cannot outlive request_timeout even if the SDK
        # transport timeout fails to fire (a stalled-but-alive connection whose
        # read timeout keeps resetting) — the failure mode behind an observed
        # turns=0 900s hang. The SDK's own ``timeout`` (set on the client
        # above) still applies and normally fires first with a cleaner error;
        # this is the guaranteed backstop at the same deadline. Streaming mode
        # supersedes it with the idle timeout above.
        self._request_timeout = request_timeout
        # Default the retry sequence's wall-clock ceiling; an explicit
        # ``retry`` mapping may override it (or any other retry knob).
        self._retry = {"max_elapsed": _DEFAULT_RETRY_MAX_ELAPSED, **(retry or {})}
        self._capabilities = Capabilities(
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

        Both the network call *and* response translation run inside the
        retried body, so a transient reply (an inline gateway error or an
        empty ``choices`` list — see :func:`from_openai_response`) is retried
        rather than propagated. Non-retryable translation errors (malformed
        tool JSON) still surface immediately: the retry helper only retries
        :class:`AdapterError`\\ s flagged ``retryable``.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(messages, system),
            **params,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if self._extra_body:
            # A per-call ``extra_body`` (rare) wins over the adapter default so
            # a caller can still override, but both are preserved.
            kwargs["extra_body"] = {**self._extra_body, **(kwargs.get("extra_body") or {})}

        async def _call() -> ModelResponse:
            try:
                if self._stream:
                    response = await self._consume_stream(kwargs)
                else:
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(**kwargs),
                        timeout=self._request_timeout,
                    )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                # Streaming: no chunk arrived within the idle window (a stall on
                # a live socket). Non-streaming: the whole call outlived the
                # hard timeout. Both are transient — retry on a fresh call.
                detail = (
                    f"stalled: no data for {self._stream_idle_timeout}s"
                    if self._stream
                    else f"exceeded {self._request_timeout}s hard timeout "
                    "(no response)"
                )
                raise AdapterError(
                    f"model call {detail}", retryable=True
                ) from exc
            except Exception as exc:
                raise wrap_openai_error(exc) from exc
            try:
                return from_openai_response(response)
            except AdapterError:
                raise  # already classified (retryable empty-choices, etc.)
            except Exception as exc:
                # A structural surprise in the reply (missing message, odd
                # shape) must end the run cleanly as a non-retryable adapter
                # error, not crash out past the loop's AdapterError handler.
                raise AdapterError(
                    f"failed to translate provider response: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        return await retry_with_backoff(_call, **self._retry)

    async def _consume_stream(self, kwargs: dict[str, Any]) -> Any:
        """Drive one streamed completion, bounded by a per-chunk idle timeout.

        Opens the stream (``stream=True`` plus ``stream_options`` to get a
        final usage chunk) and pulls chunks, wrapping **each** await —
        including the initial connect/first-chunk await — in
        :func:`asyncio.wait_for` with :attr:`_stream_idle_timeout`. A healthy
        generation keeps the gaps short and completes no matter how long it
        runs in total; a stall (no bytes within the idle window) raises
        :class:`asyncio.TimeoutError`, which :meth:`complete` turns into a
        retryable :class:`AdapterError`. The stream is always closed, so a
        timed-out or partially-read connection is not leaked. The accumulated
        chunks fold back into a full-response shape
        (:func:`accumulate_stream_chunks`) for the shared translation path.
        """
        idle = self._stream_idle_timeout
        stream = await asyncio.wait_for(
            self._client.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True}
            ),
            timeout=idle,
        )
        chunks: list[Any] = []
        try:
            iterator = stream.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        iterator.__anext__(), timeout=idle
                    )
                except StopAsyncIteration:
                    break
                chunks.append(chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
        return accumulate_stream_chunks(chunks)
