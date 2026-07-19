"""Unit tests for harness.adapters.anthropic (no network, no real client)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from harness.adapters.anthropic import (
    DEFAULT_MAX_TOKENS,
    AnthropicAdapter,
    from_anthropic_response,
    map_stop_reason,
    to_anthropic_messages,
    to_anthropic_system,
    to_anthropic_tools,
    wrap_anthropic_error,
)
from harness.adapters.base import AdapterError
from harness.types import (
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)

EPHEMERAL = {"type": "ephemeral"}


def fake_response(
    *,
    content: list[Any] | None = None,
    stop_reason: str | None = "end_turn",
    usage: Any = None,
) -> SimpleNamespace:
    """Build an SDK-shaped response object without the SDK."""
    if usage is None:
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
        )
    return SimpleNamespace(
        content=content if content is not None else [],
        stop_reason=stop_reason,
        usage=usage,
    )


class FakeMessagesAPI:
    """Records create() kwargs and replays scripted results/exceptions."""

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
    return SimpleNamespace(messages=FakeMessagesAPI(results))


# ---------------------------------------------------------------- translation


class TestToAnthropicMessages:
    def test_user_and_assistant_text(self) -> None:
        out = to_anthropic_messages(
            [
                Message(role=Role.USER, content="hi"),
                Message(role=Role.ASSISTANT, content="hello"),
            ],
            cache=False,
        )
        assert out == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]

    def test_assistant_tool_calls_become_tool_use_blocks(self) -> None:
        out = to_anthropic_messages(
            [
                Message(
                    role=Role.ASSISTANT,
                    content="running",
                    tool_calls=[
                        ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
                    ],
                )
            ],
            cache=False,
        )
        assert out[0]["content"] == [
            {"type": "text", "text": "running"},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
        ]

    def test_tool_results_ride_as_user_messages_and_merge(self) -> None:
        out = to_anthropic_messages(
            [
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(id="c1", name="bash", arguments={}),
                        ToolCall(id="c2", name="bash", arguments={}),
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id="c1", content="ok"),
                ),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(
                        tool_call_id="c2", content="boom", is_error=True
                    ),
                ),
            ],
            cache=False,
        )
        # Two consecutive tool messages merge into ONE user message.
        assert [m["role"] for m in out] == ["assistant", "user"]
        assert out[1]["content"] == [
            {"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
            {
                "type": "tool_result",
                "tool_use_id": "c2",
                "content": "boom",
                "is_error": True,
            },
        ]

    def test_cache_breakpoint_on_last_block_only(self) -> None:
        out = to_anthropic_messages(
            [
                Message(role=Role.USER, content="a"),
                Message(role=Role.ASSISTANT, content="b"),
                Message(role=Role.USER, content="c"),
            ],
            cache=True,
        )
        assert out[-1]["content"][-1]["cache_control"] == EPHEMERAL
        for message in out[:-1]:
            for block in message["content"]:
                assert "cache_control" not in block

    def test_system_role_message_rejected(self) -> None:
        with pytest.raises(AdapterError, match="top-level 'system'"):
            to_anthropic_messages(
                [Message(role=Role.SYSTEM, content="rules")], cache=False
            )

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no content"):
            to_anthropic_messages([Message(role=Role.USER)], cache=False)

    def test_tool_message_without_result_rejected(self) -> None:
        with pytest.raises(AdapterError, match="no tool_result"):
            to_anthropic_messages([Message(role=Role.TOOL)], cache=False)


class TestSystemAndTools:
    def test_system_none(self) -> None:
        assert to_anthropic_system(None) is None

    def test_system_cache_breakpoint(self) -> None:
        assert to_anthropic_system("rules", cache=True) == [
            {"type": "text", "text": "rules", "cache_control": EPHEMERAL}
        ]
        assert to_anthropic_system("rules", cache=False) == [
            {"type": "text", "text": "rules"}
        ]

    def test_tools(self) -> None:
        spec = ToolSpec(
            name="bash",
            description="run a command",
            input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )
        assert to_anthropic_tools([spec]) == [
            {
                "name": "bash",
                "description": "run a command",
                "input_schema": spec.input_schema,
            }
        ]


class TestFromAnthropicResponse:
    def test_text_and_tool_use(self) -> None:
        resp = fake_response(
            content=[
                SimpleNamespace(type="text", text="let me check"),
                SimpleNamespace(
                    type="tool_use", id="c9", name="bash", input={"cmd": "ls"}
                ),
            ],
            stop_reason="tool_use",
        )
        result = from_anthropic_response(resp)
        assert result.message.role is Role.ASSISTANT
        assert result.message.content == "let me check"
        assert result.message.tool_calls == [
            ToolCall(id="c9", name="bash", arguments={"cmd": "ls"})
        ]
        assert result.stop_reason is StopReason.TOOL_USE
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.cache_read_tokens == 3
        assert result.usage.cache_write_tokens == 2

    def test_missing_cache_fields_default_zero(self) -> None:
        resp = fake_response(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        )
        usage = from_anthropic_response(resp).usage
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_empty_content_is_none(self) -> None:
        result = from_anthropic_response(fake_response(content=[]))
        assert result.message.content is None
        assert result.message.tool_calls == []

    def test_round_trip_ours_to_provider_to_ours(self) -> None:
        resp = fake_response(
            content=[
                SimpleNamespace(type="text", text="running"),
                SimpleNamespace(
                    type="tool_use", id="c1", name="bash", input={"cmd": "ls"}
                ),
            ],
            stop_reason="tool_use",
        )
        message = from_anthropic_response(resp).message
        back = to_anthropic_messages([message], cache=False)
        assert back == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running"},
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "bash",
                        "input": {"cmd": "ls"},
                    },
                ],
            }
        ]

    @pytest.mark.parametrize(
        ("provider", "ours"),
        [
            ("end_turn", StopReason.END_TURN),
            ("stop_sequence", StopReason.END_TURN),
            ("tool_use", StopReason.TOOL_USE),
            ("max_tokens", StopReason.MAX_TOKENS),
            ("refusal", StopReason.REFUSAL),
            ("some_new_reason", StopReason.ERROR),
            (None, StopReason.ERROR),
        ],
    )
    def test_stop_reason_mapping(
        self, provider: str | None, ours: StopReason
    ) -> None:
        assert map_stop_reason(provider) is ours


# ------------------------------------------------------------- error mapping


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeAPITimeoutError(Exception):
    pass


class TestWrapAnthropicError:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 529])
    def test_retryable_statuses(self, status: int) -> None:
        wrapped = wrap_anthropic_error(FakeStatusError(status))
        assert isinstance(wrapped, AdapterError)
        assert wrapped.retryable is True
        assert str(status) in str(wrapped)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_non_retryable_statuses(self, status: int) -> None:
        assert wrap_anthropic_error(FakeStatusError(status)).retryable is False

    def test_timeout_retryable(self) -> None:
        assert wrap_anthropic_error(FakeAPITimeoutError("slow")).retryable is True
        assert wrap_anthropic_error(TimeoutError()).retryable is True

    def test_real_sdk_connection_error_shape_retryable(self) -> None:
        class APIConnectionError(Exception):
            pass

        assert wrap_anthropic_error(APIConnectionError("nope")).retryable is True

    def test_unknown_error_not_retryable(self) -> None:
        wrapped = wrap_anthropic_error(ValueError("bad input"))
        assert wrapped.retryable is False
        assert "bad input" in str(wrapped)

    def test_adapter_error_passthrough(self) -> None:
        original = AdapterError("already wrapped", retryable=True)
        assert wrap_anthropic_error(original) is original


# ------------------------------------------------------------------ complete


class TestComplete:
    async def test_translates_request_and_response(self) -> None:
        client = fake_client(
            [
                fake_response(
                    content=[SimpleNamespace(type="text", text="done")],
                    stop_reason="end_turn",
                )
            ]
        )
        adapter = AnthropicAdapter("claude-opus-4-8", client=client)
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")],
            [ToolSpec(name="bash", description="run", input_schema={})],
            system="rules",
            temperature=0.5,
        )
        assert result.message.content == "done"
        assert result.stop_reason is StopReason.END_TURN

        (kwargs,) = client.messages.calls
        assert kwargs["model"] == "claude-opus-4-8"
        assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
        assert kwargs["temperature"] == 0.5
        # system as top-level param with a cache breakpoint
        assert kwargs["system"] == [
            {"type": "text", "text": "rules", "cache_control": EPHEMERAL}
        ]
        assert kwargs["tools"] == [
            {"name": "bash", "description": "run", "input_schema": {}}
        ]
        # last transcript block carries the second cache breakpoint
        assert kwargs["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL

    async def test_omits_system_and_tools_when_absent(self) -> None:
        client = fake_client(
            [fake_response(content=[SimpleNamespace(type="text", text="ok")])]
        )
        adapter = AnthropicAdapter("m", client=client)
        await adapter.complete([Message(role=Role.USER, content="hi")], [])
        (kwargs,) = client.messages.calls
        assert "system" not in kwargs
        assert "tools" not in kwargs

    async def test_retries_retryable_then_succeeds(self) -> None:
        client = fake_client(
            [
                FakeStatusError(429),
                fake_response(content=[SimpleNamespace(type="text", text="ok")]),
            ]
        )
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        adapter = AnthropicAdapter(
            "m",
            client=client,
            retry={"sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        result = await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert result.message.content == "ok"
        assert len(client.messages.calls) == 2
        assert sleeps == [1.0]

    async def test_non_retryable_raises_immediately(self) -> None:
        client = fake_client([FakeStatusError(401)])
        adapter = AnthropicAdapter("m", client=client)
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is False
        assert len(client.messages.calls) == 1

    def test_capabilities(self) -> None:
        adapter = AnthropicAdapter("m", client=fake_client([]))
        caps = adapter.capabilities
        assert caps.supports_cache_control is True
        assert caps.parallel_tool_calls is True
        assert caps.max_context == 200_000

    def test_base_url_reaches_real_sdk_client(self) -> None:
        # Dummy key, never used for a request: verifies base_url plumbing for
        # proxies/gateways in front of the Messages API (DESIGN.md model
        # registry's generic base_url field).
        adapter = AnthropicAdapter(
            "claude-opus-4-8",
            api_key="dummy-key",
            base_url="https://my-gateway.example/",
        )
        assert str(adapter._client.base_url).startswith(
            "https://my-gateway.example/"
        )
