"""Unit tests for harness.adapters.openai_compat (no network, no real API)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from harness.adapters.base import AdapterError
from harness.adapters.openai_compat import (
    OpenAICompatAdapter,
    accumulate_stream_chunks,
    from_openai_response,
    map_finish_reason,
    to_openai_messages,
    to_openai_tools,
    wrap_openai_error,
)
from harness.types import (
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def fake_response(
    *,
    content: str | None = "hi",
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = "stop",
    usage: Any = None,
) -> SimpleNamespace:
    """Build an SDK-shaped chat.completions response without the SDK."""
    if usage is None:
        usage = SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
        )
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def fake_tool_call(id: str, name: str, arguments: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class FakeCompletionsAPI:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def fake_client(results: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletionsAPI(results))
    )


# ---------------------------------------------------------------- translation


class TestToOpenAIMessages:
    def test_system_prepended(self) -> None:
        out = to_openai_messages(
            [Message(role=Role.USER, content="hi")], system="rules"
        )
        assert out == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]

    def test_assistant_tool_calls_are_json_strings(self) -> None:
        out = to_openai_messages(
            [
                Message(
                    role=Role.ASSISTANT,
                    content="running",
                    tool_calls=[
                        ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
                    ],
                )
            ]
        )
        (entry,) = out
        assert entry["role"] == "assistant"
        assert entry["content"] == "running"
        (call,) = entry["tool_calls"]
        assert call["id"] == "c1"
        assert call["type"] == "function"
        assert call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {"cmd": "ls"}

    def test_tool_result_rides_as_role_tool(self) -> None:
        out = to_openai_messages(
            [
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id="c1", content="ok"),
                )
            ]
        )
        assert out == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]

    def test_error_tool_result_gets_prefix(self) -> None:
        out = to_openai_messages(
            [
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id="c1", content="boom", is_error=True
                    ),
                )
            ]
        )
        assert out[0]["content"] == "Error: boom"

    def test_tool_message_without_result_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no tool_result"):
            to_openai_messages([Message(role=Role.TOOL)])

    def test_non_assistant_tool_calls_rejected(self) -> None:
        with pytest.raises(AdapterError, match="only assistant"):
            to_openai_messages(
                [
                    Message(
                        role=Role.USER,
                        content="hi",
                        tool_calls=[ToolCall(id="c", name="x", arguments={})],
                    )
                ]
            )

    def test_empty_user_message_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no content"):
            to_openai_messages([Message(role=Role.USER)])

    def test_empty_assistant_message_without_tool_calls_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no content"):
            to_openai_messages([Message(role=Role.ASSISTANT)])

    def test_assistant_tool_calls_without_content_allowed(self) -> None:
        out = to_openai_messages(
            [
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=[ToolCall(id="c1", name="bash", arguments={})],
                )
            ]
        )
        (entry,) = out
        assert entry["content"] is None
        assert entry["tool_calls"][0]["function"]["name"] == "bash"


class TestToOpenAITools:
    def test_function_tool_shape(self) -> None:
        schema = {"type": "object", "properties": {"cmd": {"type": "string"}}}
        spec = ToolSpec(name="bash", description="run a command", input_schema=schema)
        assert to_openai_tools([spec]) == [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "run a command",
                    "parameters": schema,
                },
            }
        ]


class TestFromOpenAIResponse:
    def test_text_response(self) -> None:
        result = from_openai_response(fake_response(content="hello"))
        assert result.message.role is Role.ASSISTANT
        assert result.message.content == "hello"
        assert result.message.tool_calls == []
        assert result.stop_reason is StopReason.END_TURN
        # prompt_tokens=7 is cache-inclusive; Usage.input_tokens excludes
        # the 4 cached tokens per the harness-wide Usage convention.
        assert result.usage.input_tokens == 3
        assert result.usage.output_tokens == 3
        assert result.usage.cache_read_tokens == 4

    def test_tool_call_arguments_parsed_from_json_string(self) -> None:
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": "ls"}')],
            finish_reason="tool_calls",
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == [
            ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        ]
        assert result.stop_reason is StopReason.TOOL_USE

    def test_empty_arguments_ok(self) -> None:
        for raw in (None, ""):
            resp = fake_response(
                content=None,
                tool_calls=[fake_tool_call("c1", "bash", raw)],
                finish_reason="tool_calls",
            )
            assert from_openai_response(resp).message.tool_calls[0].arguments == {}

    def test_malformed_arguments_surface_as_adapter_error(self) -> None:
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": ')],
            finish_reason="tool_calls",
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert "bash" in str(excinfo.value)
        assert "malformed JSON" in str(excinfo.value)
        assert excinfo.value.retryable is False

    def test_non_object_arguments_rejected(self) -> None:
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '["not", "a", "dict"]')],
            finish_reason="tool_calls",
        )
        with pytest.raises(AdapterError, match="non-object"):
            from_openai_response(resp)

    def test_no_choices_rejected_and_retryable(self) -> None:
        # Empty choices is a transient upstream fault (e.g. OpenRouter under
        # rate-limit returns HTTP 200 with []), so it must be retryable — a
        # regression here silently kills whole tasks on a single blip.
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(SimpleNamespace(choices=[], usage=None))
        assert "no choices" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_inline_provider_error_retryable_by_code(self) -> None:
        # OpenRouter reports upstream faults as an inline error object on an
        # HTTP 200 body; a 429 there means the same as an HTTP 429 → retry.
        resp = SimpleNamespace(
            choices=[], usage=None,
            error={"code": 429, "message": "rate limited upstream"},
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert "inline error" in str(excinfo.value)
        assert "429" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_inline_provider_error_non_retryable_client_code(self) -> None:
        resp = SimpleNamespace(
            choices=[], usage=None,
            error={"code": 400, "message": "bad request upstream"},
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is False

    def test_inline_provider_error_without_code_defaults_retryable(self) -> None:
        resp = SimpleNamespace(
            choices=[], usage=None, error={"message": "something transient"}
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is True

    def test_inline_error_object_attribute_style(self) -> None:
        # Duck-typed: some SDK stand-ins expose error as an object, not a dict.
        resp = SimpleNamespace(
            choices=[], usage=None,
            error=SimpleNamespace(code=503, message="upstream down"),
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is True

    def test_inline_error_wins_over_present_choices(self) -> None:
        # A gateway may return BOTH an error object and (stale/partial) choices;
        # the error must take precedence so a fault is never silently parsed.
        good_choice = fake_response(content="ignore me").choices[0]
        resp = SimpleNamespace(
            choices=[good_choice], usage=None,
            error={"code": 429, "message": "rate limited"},
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is True

    def test_falsy_error_field_is_ignored(self) -> None:
        # error: null and error: {} are not faults — the response parses normally.
        for empty in (None, {}, ""):
            resp = fake_response(content="fine")
            resp.error = empty
            result = from_openai_response(resp)
            assert result.message.content == "fine"

    def test_missing_usage_details_default_zero(self) -> None:
        resp = fake_response(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        )
        usage = from_openai_response(resp).usage
        assert usage.input_tokens == 1
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_input_tokens_exclude_cache_traffic(self) -> None:
        """Regression: OpenAI's prompt_tokens INCLUDES cached tokens
        (prompt_tokens_details fields are subsets of it), while the harness
        Usage convention is cache-exclusive input_tokens. The adapter must
        subtract cache reads/writes so downstream consumers that sum
        input + cache_read + cache_write (e.g. the Harbor bridge) recover
        the true prompt total instead of double-counting cache traffic.
        """
        resp = fake_response(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=3,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=700, cache_write_tokens=100
                ),
            )
        )
        usage = from_openai_response(resp).usage
        assert usage.input_tokens == 200  # 1000 - 700 - 100
        assert usage.cache_read_tokens == 700
        assert usage.cache_write_tokens == 100
        # Invariant the Harbor bridge relies on: the sum reconstructs
        # the provider's cache-inclusive prompt total.
        total = (
            usage.input_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
        )
        assert total == 1000

    def test_input_tokens_clamped_when_cache_counts_exceed_prompt_total(
        self,
    ) -> None:
        """Providers that report cache counts outside prompt_tokens must not
        produce negative input_tokens."""
        resp = fake_response(
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=1,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=4, cache_write_tokens=3
                ),
            )
        )
        usage = from_openai_response(resp).usage
        assert usage.input_tokens == 0

    def test_cache_write_tokens_populated_when_reported(self) -> None:
        resp = fake_response(
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=4, cache_write_tokens=9
                ),
            )
        )
        usage = from_openai_response(resp).usage
        assert usage.cache_read_tokens == 4
        assert usage.cache_write_tokens == 9

    def test_round_trip_ours_to_provider_to_ours(self) -> None:
        resp = fake_response(
            content="running",
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": "ls"}')],
            finish_reason="tool_calls",
        )
        message = from_openai_response(resp).message
        (entry,) = to_openai_messages([message])
        assert entry["content"] == "running"
        assert entry["tool_calls"][0]["id"] == "c1"
        assert json.loads(entry["tool_calls"][0]["function"]["arguments"]) == {
            "cmd": "ls"
        }

    @pytest.mark.parametrize(
        ("provider", "ours"),
        [
            ("stop", StopReason.END_TURN),
            ("tool_calls", StopReason.TOOL_USE),
            ("length", StopReason.MAX_TOKENS),
            ("content_filter", StopReason.REFUSAL),
            ("weird_new_reason", StopReason.ERROR),
            (None, StopReason.ERROR),
        ],
    )
    def test_finish_reason_mapping(
        self, provider: str | None, ours: StopReason
    ) -> None:
        assert map_finish_reason(provider) is ours


# ------------------------------------------------------------- error mapping


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestWrapOpenAIError:
    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    def test_retryable_statuses(self, status: int) -> None:
        assert wrap_openai_error(FakeStatusError(status)).retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_non_retryable_statuses(self, status: int) -> None:
        assert wrap_openai_error(FakeStatusError(status)).retryable is False

    def test_timeout_and_connection_retryable(self) -> None:
        class APITimeoutError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        assert wrap_openai_error(APITimeoutError()).retryable is True
        assert wrap_openai_error(APIConnectionError()).retryable is True

    @pytest.mark.parametrize(
        "exc_name",
        [
            "ReadError",
            "ReadTimeout",
            "WriteError",
            "RemoteProtocolError",
            "LocalProtocolError",
            "StreamError",
            "IncompleteRead",
            "PoolTimeout",
        ],
    )
    def test_streaming_transport_errors_retryable(self, exc_name: str) -> None:
        # Reading the body incrementally makes mid-stream httpx transport
        # errors likely; they are transient and must retry, not forfeit the run.
        exc = type(exc_name, (Exception,), {})()
        wrapped = wrap_openai_error(exc)
        assert wrapped.retryable is True
        assert exc_name in str(wrapped)

    def test_unknown_not_retryable(self) -> None:
        assert wrap_openai_error(RuntimeError("boom")).retryable is False

    def test_adapter_error_passthrough(self) -> None:
        original = AdapterError("wrapped", retryable=True)
        assert wrap_openai_error(original) is original


# ------------------------------------------------------------------ complete


class TestComplete:
    async def test_translates_request_and_response(self) -> None:
        client = fake_client([fake_response(content="done")])
        adapter = OpenAICompatAdapter("kimi-k3", client=client, stream=False)
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")],
            [ToolSpec(name="bash", description="run", input_schema={})],
            system="rules",
            temperature=0.2,
        )
        assert result.message.content == "done"
        (kwargs,) = client.chat.completions.calls
        assert kwargs["model"] == "kimi-k3"
        assert kwargs["temperature"] == 0.2
        assert kwargs["messages"][0] == {"role": "system", "content": "rules"}
        assert kwargs["tools"][0]["function"]["name"] == "bash"

    async def test_omits_tools_when_absent(self) -> None:
        client = fake_client([fake_response()])
        adapter = OpenAICompatAdapter("m", client=client, stream=False)
        await adapter.complete([Message(role=Role.USER, content="hi")], [])
        (kwargs,) = client.chat.completions.calls
        assert "tools" not in kwargs

    async def test_retries_retryable_then_succeeds(self) -> None:
        client = fake_client([FakeStatusError(503), fake_response(content="ok")])
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False,
            retry={"sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        result = await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert result.message.content == "ok"
        assert len(client.chat.completions.calls) == 2
        assert sleeps == [1.0]

    async def test_non_retryable_raises_immediately(self) -> None:
        client = fake_client([FakeStatusError(400)])
        adapter = OpenAICompatAdapter("m", client=client, stream=False)
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is False
        assert len(client.chat.completions.calls) == 1

    async def test_empty_choices_response_is_retried(self) -> None:
        # Regression for the production failure: an HTTP-200 empty-choices
        # reply must be retried (translation happens inside the retried call),
        # not raised straight out and killing the run.
        empty = SimpleNamespace(choices=[], usage=None)
        client = fake_client([empty, fake_response(content="recovered")])
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False,
            retry={"sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        result = await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert result.message.content == "recovered"
        assert len(client.chat.completions.calls) == 2
        assert sleeps == [1.0]

    async def test_inline_gateway_error_response_is_retried(self) -> None:
        err = SimpleNamespace(
            choices=[], usage=None,
            error={"code": 429, "message": "upstream rate limit"},
        )
        client = fake_client([err, fake_response(content="ok")])

        async def fake_sleep(_delay: float) -> None:
            pass

        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False,
            retry={"sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        result = await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert result.message.content == "ok"
        assert len(client.chat.completions.calls) == 2

    async def test_empty_choices_exhaustion_reraises_retryable(self) -> None:
        # If every attempt returns empty choices, the final error still carries
        # retryable=True (it was transient; we just ran out of attempts).
        empties = [SimpleNamespace(choices=[], usage=None) for _ in range(3)]
        client = fake_client(empties)

        async def fake_sleep(_delay: float) -> None:
            pass

        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False,
            retry={"max_attempts": 3, "sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is True
        assert len(client.chat.completions.calls) == 3

    async def test_hard_timeout_interrupts_hung_call(self) -> None:
        # Regression for the observed turns=0 900s hang: a single in-flight
        # call that outlives request_timeout must be interrupted and surfaced
        # as a retryable error, not awaited indefinitely (the SDK/httpx
        # transport timeout is not trusted to fire on a stalled connection).
        class HungAPI:
            calls = 0

            async def create(self, **kwargs: Any) -> Any:
                HungAPI.calls += 1
                await asyncio.sleep(10.0)  # far longer than request_timeout
                return fake_response()

        client = SimpleNamespace(chat=SimpleNamespace(completions=HungAPI()))
        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False,
            request_timeout=0.02, retry={"max_attempts": 1},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is True
        assert "hard timeout" in str(excinfo.value)
        assert HungAPI.calls == 1

    def test_default_retry_config_bounds_wall_clock(self) -> None:
        # The adapter defaults a retry budget so request_timeout × max_attempts
        # cannot silently overrun an upstream agent deadline.
        adapter = OpenAICompatAdapter("m", client=fake_client([]))
        assert adapter._retry["max_elapsed"] == 300.0
        # An explicit retry mapping still overrides it.
        override = OpenAICompatAdapter(
            "m", client=fake_client([]), retry={"max_elapsed": 42.0}
        )
        assert override._retry["max_elapsed"] == 42.0

    async def test_malformed_tool_json_not_retried(self) -> None:
        # Non-retryable translation errors must still fail fast even though
        # translation now runs inside the retried call.
        bad = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": ')],
            finish_reason="tool_calls",
        )
        client = fake_client([bad, fake_response(content="unreached")])
        adapter = OpenAICompatAdapter("m", client=client, stream=False)
        with pytest.raises(AdapterError, match="malformed JSON"):
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert len(client.chat.completions.calls) == 1

    def test_capabilities(self) -> None:
        adapter = OpenAICompatAdapter("m", client=fake_client([]))
        caps = adapter.capabilities
        assert caps.supports_cache_control is False
        assert caps.max_context == 128_000

    def test_base_url_reaches_real_sdk_client(self) -> None:
        # Dummy key, never used for a request: verifies base_url plumbing for
        # OpenAI-compatible endpoints like Moonshot (DESIGN.md model registry).
        adapter = OpenAICompatAdapter(
            "kimi-k3",
            api_key="dummy-key",
            base_url="https://api.moonshot.ai/v1",
        )
        assert str(adapter._client.base_url).startswith("https://api.moonshot.ai/v1")

    def test_real_client_has_single_retry_layer_and_timeout(self) -> None:
        # The SDK must not add its own retry layer (retry_with_backoff is the
        # sole retrier, DESIGN.md §4.1), and a request timeout must bound hung
        # calls so they fail fast as retryable timeouts.
        adapter = OpenAICompatAdapter(
            "kimi-k3",
            api_key="dummy-key",
            base_url="https://openrouter.ai/api/v1",
            request_timeout=90.0,
        )
        assert adapter._client.max_retries == 0
        assert adapter._client.timeout == 90.0

    def test_streaming_is_the_default(self) -> None:
        adapter = OpenAICompatAdapter("m", client=fake_client([]))
        assert adapter._stream is True

    def test_get_adapter_threads_extra_body_from_config(self) -> None:
        import warnings as _warnings

        from harness.adapters import get_adapter
        from harness.config import ModelConfig

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # literal api_key warns; irrelevant
            mc = ModelConfig(
                adapter="openai",
                model="m",
                api_key="sk-dummy",
                extra_body={"reasoning": {"effort": "low"}},
            )
            adapter = get_adapter(mc)
        assert adapter._extra_body == {"reasoning": {"effort": "low"}}


# --------------------------------------------------------------- stream chunks


def stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
    error: Any = None,
) -> SimpleNamespace:
    """Build one streamed chat.completion chunk (SDK-shaped stand-in).

    A usage-only terminal chunk (``stream_options`` include_usage) carries
    ``choices == []``; model this by passing only ``usage``.
    """
    chunk = SimpleNamespace(choices=[], usage=usage)
    if content is not None or tool_calls is not None or finish_reason is not None:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        chunk.choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    if error is not None:
        chunk.error = error
    return chunk


def tc_delta(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """A streamed tool-call delta fragment."""
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeStream:
    """Async-iterable stand-in for the SDK's ``AsyncStream``.

    ``gaps[i]`` (seconds) is awaited before yielding chunk ``i`` — used to
    simulate an inter-chunk stall the idle timeout must catch.
    """

    def __init__(self, chunks: list[Any], *, gaps: list[float] | None = None) -> None:
        self._chunks = list(chunks)
        self._gaps = list(gaps) if gaps is not None else [0.0] * len(chunks)
        self._i = 0
        self.closed = False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> Any:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        gap = self._gaps[self._i] if self._i < len(self._gaps) else 0.0
        if gap:
            await asyncio.sleep(gap)
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    async def close(self) -> None:
        self.closed = True


class FakeStreamingCompletionsAPI:
    def __init__(self, streams: list[Any]) -> None:
        self.streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.streams.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def fake_streaming_client(streams: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeStreamingCompletionsAPI(streams))
    )


class TestAccumulateStreamChunks:
    def test_text_fragments_concatenated(self) -> None:
        resp = accumulate_stream_chunks(
            [
                stream_chunk(content="he"),
                stream_chunk(content="llo"),
                stream_chunk(finish_reason="stop"),
            ]
        )
        translated = from_openai_response(resp)
        assert translated.message.content == "hello"
        assert translated.stop_reason is StopReason.END_TURN

    def test_tool_calls_merged_by_index_across_chunks(self) -> None:
        resp = accumulate_stream_chunks(
            [
                stream_chunk(
                    tool_calls=[tc_delta(0, id="c1", name="bash", arguments='{"cmd"')]
                ),
                stream_chunk(tool_calls=[tc_delta(0, arguments=': "ls"}')]),
                stream_chunk(finish_reason="tool_calls"),
            ]
        )
        translated = from_openai_response(resp)
        assert translated.message.tool_calls == [
            ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        ]
        assert translated.stop_reason is StopReason.TOOL_USE

    def test_two_parallel_tool_calls_kept_in_index_order(self) -> None:
        resp = accumulate_stream_chunks(
            [
                stream_chunk(tool_calls=[tc_delta(0, id="a", name="one", arguments="{}")]),
                stream_chunk(tool_calls=[tc_delta(1, id="b", name="two", arguments="{}")]),
                stream_chunk(finish_reason="tool_calls"),
            ]
        )
        calls = from_openai_response(resp).message.tool_calls
        assert [c.id for c in calls] == ["a", "b"]
        assert [c.name for c in calls] == ["one", "two"]

    def test_usage_captured_from_terminal_chunk(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=42,
            prompt_tokens_details=SimpleNamespace(cached_tokens=700),
        )
        resp = accumulate_stream_chunks(
            [stream_chunk(content="hi"), stream_chunk(finish_reason="stop", usage=usage)]
        )
        translated = from_openai_response(resp)
        assert translated.usage.input_tokens == 300  # 1000 - 700 cached
        assert translated.usage.output_tokens == 42
        assert translated.usage.cache_read_tokens == 700

    def test_content_less_stream_collapses_to_retryable_no_choices(self) -> None:
        # A stream that yields only a usage chunk (no content, no tool calls, no
        # finish_reason) is a transient empty reply — it must translate to the
        # SAME retryable "no choices" error a non-streamed empty response does.
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0)
        resp = accumulate_stream_chunks([stream_chunk(usage=usage)])
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert "no choices" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_inline_error_chunk_preserved_for_translation(self) -> None:
        resp = accumulate_stream_chunks(
            [stream_chunk(error={"code": 429, "message": "rate limited"})]
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is True


class TestStreamingComplete:
    async def test_streams_text_and_requests_usage(self) -> None:
        client = fake_streaming_client(
            [
                FakeStream(
                    [
                        stream_chunk(content="done"),
                        stream_chunk(
                            finish_reason="stop",
                            usage=SimpleNamespace(
                                prompt_tokens=7, completion_tokens=3
                            ),
                        ),
                    ]
                )
            ]
        )
        adapter = OpenAICompatAdapter("kimi-k3", client=client)
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")], []
        )
        assert result.message.content == "done"
        (kwargs,) = client.chat.completions.calls
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    async def test_streamed_tool_call_round_trips(self) -> None:
        client = fake_streaming_client(
            [
                FakeStream(
                    [
                        stream_chunk(
                            tool_calls=[
                                tc_delta(0, id="c1", name="bash", arguments="")
                            ]
                        ),
                        stream_chunk(
                            tool_calls=[tc_delta(0, arguments='{"cmd": "ls"}')]
                        ),
                        stream_chunk(finish_reason="tool_calls"),
                    ]
                )
            ]
        )
        adapter = OpenAICompatAdapter("m", client=client)
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")], []
        )
        assert result.message.tool_calls == [
            ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        ]

    async def test_slow_but_steady_stream_is_not_a_stall(self) -> None:
        # Each chunk arrives just under the idle window: a long, slow-but-healthy
        # generation must complete regardless of total wall time — the exact
        # case a whole-call timeout wrongly guillotined.
        chunks = [stream_chunk(content=str(i)) for i in range(6)]
        chunks.append(stream_chunk(finish_reason="stop"))
        client = fake_streaming_client(
            [FakeStream(chunks, gaps=[0.03] * len(chunks))]
        )
        adapter = OpenAICompatAdapter(
            "m", client=client, stream_idle_timeout=0.2
        )
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")], []
        )
        assert result.message.content == "012345"

    async def test_idle_stall_surfaces_retryable(self) -> None:
        # A gap between chunks longer than the idle timeout is a stall.
        stream = FakeStream(
            [stream_chunk(content="a"), stream_chunk(content="b")],
            gaps=[0.0, 0.3],
        )
        client = fake_streaming_client([stream])
        adapter = OpenAICompatAdapter(
            "m", client=client, stream_idle_timeout=0.1,
            retry={"max_attempts": 1},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is True
        assert "stalled" in str(excinfo.value)
        assert stream.closed is True  # the stalled stream was closed, not leaked

    async def test_stall_then_retry_succeeds(self) -> None:
        # A transient stall on the first attempt self-heals on a fresh stream.
        stalled = FakeStream([stream_chunk(content="x")], gaps=[0.3])
        healthy = FakeStream(
            [stream_chunk(content="recovered"), stream_chunk(finish_reason="stop")]
        )
        client = fake_streaming_client([stalled, healthy])
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        adapter = OpenAICompatAdapter(
            "m", client=client, stream_idle_timeout=0.1,
            retry={"sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")], []
        )
        assert result.message.content == "recovered"
        assert len(client.chat.completions.calls) == 2
        assert sleeps == [1.0]

    async def test_extra_body_forwarded_to_request(self) -> None:
        # A reasoning/thinking control (or any gateway field) set on the adapter
        # rides through as ``extra_body`` on every request.
        client = fake_streaming_client(
            [FakeStream([stream_chunk(content="ok"), stream_chunk(finish_reason="stop")])]
        )
        adapter = OpenAICompatAdapter(
            "m", client=client, extra_body={"reasoning": {"effort": "low"}}
        )
        await adapter.complete([Message(role=Role.USER, content="hi")], [])
        (kwargs,) = client.chat.completions.calls
        assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}

    async def test_empty_stream_is_retryable(self) -> None:
        stream = FakeStream(
            [stream_chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=0))]
        )
        client = fake_streaming_client([stream])
        adapter = OpenAICompatAdapter(
            "m", client=client, retry={"max_attempts": 1}
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert "no choices" in str(excinfo.value)
        assert excinfo.value.retryable is True
