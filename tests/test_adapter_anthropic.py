"""Unit tests for harness.adapters.anthropic (no network, no real client)."""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from harness.adapters.anthropic import (
    DEFAULT_MAX_TOKENS,
    EMPTY_ASSISTANT_PLACEHOLDER,
    EMPTY_MESSAGE_PLACEHOLDERS,
    AnthropicAdapter,
    from_anthropic_response,
    map_stop_reason,
    to_anthropic_messages,
    to_anthropic_system,
    to_anthropic_tools,
    wrap_anthropic_error,
)
from harness.adapters.base import AdapterError
from harness.diligence import looks_unfinished
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

    def test_empty_assistant_message_gets_placeholder_not_error(self) -> None:
        # C1 replay backstop, Anthropic parity: an assistant message with
        # neither content nor tool calls must stay translatable.
        with pytest.warns(UserWarning, match="neither content nor tool calls"):
            out = to_anthropic_messages(
                [Message(role=Role.ASSISTANT)], cache=False
            )
        assert out == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": EMPTY_ASSISTANT_PLACEHOLDER}
                ],
            }
        ]

    def test_empty_assistant_placeholder_is_diligence_safe(self) -> None:
        assert not EMPTY_ASSISTANT_PLACEHOLDER.rstrip().endswith("?")
        unfinished, _reason = looks_unfinished(EMPTY_ASSISTANT_PLACEHOLDER, 0)
        assert unfinished is False

    def test_empty_assistant_placeholder_not_written_back(self) -> None:
        message = Message(role=Role.ASSISTANT)
        with pytest.warns(UserWarning):
            to_anthropic_messages([message], cache=False)
        assert message.content is None
        assert message.tool_calls == []

    def test_empty_assistant_placeholder_takes_cache_breakpoint(self) -> None:
        # The placeholder is a real block, so the trailing cache breakpoint
        # still lands on it rather than on nothing.
        with pytest.warns(UserWarning):
            out = to_anthropic_messages(
                [Message(role=Role.USER, content="go"), Message(role=Role.ASSISTANT)],
                cache=True,
            )
        assert out[-1]["content"][-1] == {
            "type": "text",
            "text": EMPTY_ASSISTANT_PLACEHOLDER,
            "cache_control": EPHEMERAL,
        }

    def test_empty_assistant_merges_into_preceding_assistant_turn(self) -> None:
        with pytest.warns(UserWarning):
            out = to_anthropic_messages(
                [
                    Message(role=Role.ASSISTANT, content="first"),
                    Message(role=Role.ASSISTANT),
                ],
                cache=False,
            )
        assert out == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": EMPTY_ASSISTANT_PLACEHOLDER},
                ],
            }
        ]

    def test_assistant_tool_calls_without_content_emit_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = to_anthropic_messages(
                [
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=[ToolCall(id="c1", name="bash", arguments={})],
                    )
                ],
                cache=False,
            )
        assert out[0]["content"] == [
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {}}
        ]


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

    def test_reasoning_tokens_always_zero(self) -> None:
        # Anthropic thinking is not wired up (round-3 prerequisite: a
        # thinking-block passthrough field on Message, which does not exist
        # yet). This adapter must never report reasoning_tokens, whatever the
        # response carries — a stray attribute must not leak through.
        resp = fake_response(content=[SimpleNamespace(type="text", text="hi")])
        assert from_anthropic_response(resp).usage.reasoning_tokens == 0

    def test_empty_content_gets_a_placeholder_body(self) -> None:
        # Parity with the OpenAI-compatible adapter: an assistant message with
        # nothing in it must never be produced, because replaying one forces
        # the translator to invent a body (and a real run died at turn 1 that
        # way). stop_reason="end_turn" is recognised, so this is NOT incomplete
        # — the M1 gate confines re-prompting to unrecognised stop reasons.
        result = from_anthropic_response(fake_response(content=[]))
        assert result.message.content == EMPTY_MESSAGE_PLACEHOLDERS[None]
        assert result.message.tool_calls == []
        assert result.incomplete is False
        assert result.incomplete_reason is None

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

    @pytest.mark.parametrize(
        ("provider", "ours"),
        [
            ("end_turn", StopReason.END_TURN),
            ("tool_use", StopReason.TOOL_USE),
            ("max_tokens", StopReason.MAX_TOKENS),
            ("some_new_reason", StopReason.ERROR),
            (None, StopReason.ERROR),
        ],
    )
    def test_provider_stop_reason_carried_through_verbatim(
        self, provider: str | None, ours: StopReason
    ) -> None:
        """§C2 parity with the OpenAI adapter: the API's raw stop_reason
        survives translation untouched while stop_reason maps as before.
        The last two rows are the point — mapping alone cannot distinguish
        an unmapped provider string from a missing one."""
        result = from_anthropic_response(fake_response(stop_reason=provider))
        assert result.provider_stop_reason == provider
        assert result.stop_reason is ours

    def test_provider_stop_reason_absent_attribute_is_none(self) -> None:
        """A response object with no stop_reason attribute at all (not
        merely None) must not raise; provenance is best-effort."""
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        result = from_anthropic_response(response)
        assert result.provider_stop_reason is None
        assert result.stop_reason is StopReason.ERROR


# ------------------------------- incomplete-turn parity with openai_compat


class TestIncompleteDerivation:
    """§C4 parity. Anthropic delivers ``tool_use`` inputs pre-parsed, so this
    adapter has no malformed-arguments case and never drops a call; it mirrors
    the other two precedence rows and the empty-message placeholder table."""

    def test_max_tokens_is_incomplete(self) -> None:
        result = from_anthropic_response(
            fake_response(content=[], stop_reason="max_tokens")
        )
        assert result.incomplete is True
        assert result.incomplete_reason == "max_tokens"
        assert result.message.content == EMPTY_MESSAGE_PLACEHOLDERS[
            "max_tokens"
        ]

    def test_max_tokens_with_content_is_still_incomplete(self) -> None:
        # max_tokens is unconditional per the precedence table.
        result = from_anthropic_response(
            fake_response(
                content=[SimpleNamespace(type="text", text="half a thou")],
                stop_reason="max_tokens",
            )
        )
        assert result.incomplete_reason == "max_tokens"
        assert result.message.content == "half a thou"

    def test_missing_stop_reason_with_empty_message_is_no_finish_reason(
        self,
    ) -> None:
        result = from_anthropic_response(
            fake_response(content=[], stop_reason=None)
        )
        assert result.incomplete_reason == "no_finish_reason"
        assert result.message.content == EMPTY_MESSAGE_PLACEHOLDERS[
            "no_finish_reason"
        ]

    def test_missing_stop_reason_with_content_is_not_incomplete(self) -> None:
        """The M1 gate, mirrored: an unmapped or absent stop reason on a
        response that still said something useful must not earn re-prompts."""
        result = from_anthropic_response(
            fake_response(
                content=[SimpleNamespace(type="text", text="The answer is 42.")],
                stop_reason="some_future_reason",
            )
        )
        assert result.stop_reason is StopReason.ERROR
        assert result.incomplete is False
        assert result.incomplete_reason is None

    def test_unmapped_stop_reason_with_only_a_tool_use_is_not_incomplete(
        self,
    ) -> None:
        result = from_anthropic_response(
            fake_response(
                content=[
                    SimpleNamespace(
                        type="tool_use", id="c1", name="bash", input={"cmd": "ls"}
                    )
                ],
                stop_reason=None,
            )
        )
        assert result.incomplete is False

    def test_clean_end_turn_is_not_incomplete(self) -> None:
        result = from_anthropic_response(
            fake_response(
                content=[SimpleNamespace(type="text", text="done")],
                stop_reason="end_turn",
            )
        )
        assert result.incomplete is False
        assert result.dropped_tool_calls == []

    @pytest.mark.parametrize(
        "reason", ["max_tokens", "no_finish_reason", None]
    )
    def test_placeholders_replay_and_do_not_look_unfinished(
        self, reason: str | None
    ) -> None:
        # Each placeholder becomes assistant content, so it must survive
        # re-translation without the C1 warning and must not read as promised
        # future work if it reaches the diligence check.
        text = EMPTY_MESSAGE_PLACEHOLDERS[reason]
        message = Message(role=Role.ASSISTANT, content=text)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (entry,) = to_anthropic_messages([message], cache=False)
        assert entry["content"][0]["text"] == text
        assert looks_unfinished(text, 0)[0] is False

    def test_placeholder_table_matches_the_openai_adapter_verbatim(
        self,
    ) -> None:
        # Kept textually identical on purpose (same rationale as
        # EMPTY_ASSISTANT_PLACEHOLDER): a transcript must read the same
        # whichever provider produced it.
        from harness.adapters.openai_compat import (
            EMPTY_MESSAGE_PLACEHOLDERS as OPENAI_PLACEHOLDERS,
        )

        for reason, text in EMPTY_MESSAGE_PLACEHOLDERS.items():
            assert OPENAI_PLACEHOLDERS[reason] == text


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


class _FakeStatusErrorWithBody(Exception):
    """A status error whose str() carries the provider's error body."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code


class TestWrapAnthropicErrorFault:
    """Provider-fault classification must be identical across adapters.

    Both wrappers delegate to the one shared
    :func:`~harness.adapters.base.classify_http_fault` rule precisely so the
    taxonomy cannot drift between providers; these are the parity tests.
    """

    @pytest.mark.parametrize(
        "status,want",
        [
            (401, "auth"),
            (402, "quota"),
            (429, "rate_limit"),
            (500, "server"),
            (529, "server"),  # Anthropic's overloaded_error
            (400, None),
            (404, None),
            (422, None),
        ],
    )
    def test_status_classification(self, status: int, want: str | None) -> None:
        assert wrap_anthropic_error(FakeStatusError(status)).fault == want

    def test_403_quota_body(self) -> None:
        exc = _FakeStatusErrorWithBody(403, "Your credit balance is too low")
        assert wrap_anthropic_error(exc).fault == "quota"

    def test_403_non_quota_stays_a_scored_failure(self) -> None:
        exc = _FakeStatusErrorWithBody(403, "request blocked in your region")
        assert wrap_anthropic_error(exc).fault is None

    def test_timeout_and_connection_are_transport_faults(self) -> None:
        class APIConnectionError(Exception):
            pass

        assert wrap_anthropic_error(TimeoutError()).fault == "transport"
        assert wrap_anthropic_error(FakeAPITimeoutError("slow")).fault == "transport"
        assert wrap_anthropic_error(APIConnectionError("nope")).fault == "transport"

    def test_unknown_error_unclassified(self) -> None:
        assert wrap_anthropic_error(ValueError("bad input")).fault is None


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

    async def test_never_sends_thinking_or_output_config(self) -> None:
        # Guard against an accidental re-introduction of Anthropic reasoning
        # mapping: thinking={"type":"enabled","budget_tokens":N} returns
        # HTTP 400 on every current Anthropic model, and this harness
        # discards thinking blocks the API requires echoed back on
        # multi-turn tool use (round-3 prerequisite, not yet built). No
        # reasoning-shaped key may ever be sent until that lands.
        client = fake_client(
            [fake_response(content=[SimpleNamespace(type="text", text="ok")])]
        )
        adapter = AnthropicAdapter("claude-opus-4-8", client=client)
        await adapter.complete(
            [Message(role=Role.USER, content="hi")],
            [ToolSpec(name="bash", description="run", input_schema={})],
            system="rules",
        )
        (kwargs,) = client.messages.calls
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

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

    async def test_hard_timeout_interrupts_hung_call(self) -> None:
        # Parity with the OpenAI adapter: a single in-flight Messages call that
        # outlives request_timeout is interrupted and raised as retryable.
        class HungAPI:
            calls = 0

            async def create(self, **kwargs: Any) -> Any:
                HungAPI.calls += 1
                await asyncio.sleep(10.0)

        client = SimpleNamespace(messages=HungAPI())
        adapter = AnthropicAdapter(
            "m", client=client, request_timeout=0.02, retry={"max_attempts": 1}
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.retryable is True
        assert "hard timeout" in str(excinfo.value)
        assert HungAPI.calls == 1

    async def test_hard_timeout_is_a_transport_fault_after_retries(self) -> None:
        # Parity with the OpenAI adapter: complete()'s own timeout branch runs
        # instead of wrap_anthropic_error, so it must carry the fault itself,
        # and the classification has to survive an exhausted retry budget —
        # that is the AdapterError the loop reads exc.fault off.
        class HungAPI:
            async def create(self, **kwargs: Any) -> Any:
                await asyncio.sleep(10.0)

        async def fake_sleep(delay: float) -> None:
            return None

        adapter = AnthropicAdapter(
            "m",
            client=SimpleNamespace(messages=HungAPI()),
            request_timeout=0.02,
            retry={"max_attempts": 3, "sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.fault == "transport"

    def test_capabilities(self) -> None:
        adapter = AnthropicAdapter("m", client=fake_client([]))
        caps = adapter.capabilities
        assert caps.supports_cache_control is True
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

    def test_real_client_has_single_retry_layer_and_timeout(self) -> None:
        # Same invariant as the OpenAI adapter: retry_with_backoff is the sole
        # retrier (SDK max_retries=0) and a request timeout bounds hung calls.
        adapter = AnthropicAdapter(
            "claude-opus-4-8",
            api_key="dummy-key",
            request_timeout=90.0,
        )
        assert adapter._client.max_retries == 0
        assert adapter._client.timeout == 90.0

    def test_default_retry_config_bounds_wall_clock(self) -> None:
        adapter = AnthropicAdapter("m", client=fake_client([]))
        assert adapter._retry["max_elapsed"] == 300.0
