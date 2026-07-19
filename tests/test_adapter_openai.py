"""Unit tests for harness.adapters.openai_compat (no network, no real API)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from harness.adapters.base import AdapterError
from harness.adapters.openai_compat import (
    OpenAICompatAdapter,
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

    def test_no_choices_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no choices"):
            from_openai_response(SimpleNamespace(choices=[], usage=None))

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

    def test_unknown_not_retryable(self) -> None:
        assert wrap_openai_error(RuntimeError("boom")).retryable is False

    def test_adapter_error_passthrough(self) -> None:
        original = AdapterError("wrapped", retryable=True)
        assert wrap_openai_error(original) is original


# ------------------------------------------------------------------ complete


class TestComplete:
    async def test_translates_request_and_response(self) -> None:
        client = fake_client([fake_response(content="done")])
        adapter = OpenAICompatAdapter("kimi-k3", client=client)
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
        adapter = OpenAICompatAdapter("m", client=client)
        await adapter.complete([Message(role=Role.USER, content="hi")], [])
        (kwargs,) = client.chat.completions.calls
        assert "tools" not in kwargs

    async def test_retries_retryable_then_succeeds(self) -> None:
        client = fake_client([FakeStatusError(503), fake_response(content="ok")])
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        adapter = OpenAICompatAdapter(
            "m", client=client, retry={"sleep": fake_sleep, "jitter": lambda: 0.0}
        )
        result = await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert result.message.content == "ok"
        assert len(client.chat.completions.calls) == 2
        assert sleeps == [1.0]

    async def test_non_retryable_raises_immediately(self) -> None:
        client = fake_client([FakeStatusError(400)])
        adapter = OpenAICompatAdapter("m", client=client)
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is False
        assert len(client.chat.completions.calls) == 1

    def test_capabilities(self) -> None:
        adapter = OpenAICompatAdapter("m", client=fake_client([]))
        caps = adapter.capabilities
        assert caps.supports_cache_control is False
        assert caps.parallel_tool_calls is True
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
