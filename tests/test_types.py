"""Unit tests for harness.types."""

import pytest
from pydantic import ValidationError

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


class TestEnums:
    def test_role_values(self):
        assert [r.value for r in Role] == ["system", "user", "assistant", "tool"]

    def test_stop_reason_values(self):
        assert [s.value for s in StopReason] == [
            "end_turn",
            "tool_use",
            "max_tokens",
            "refusal",
            "error",
        ]

    def test_enums_are_str(self):
        # str-valued enums serialize cleanly to JSON/SQLite.
        assert isinstance(Role.USER, str)
        assert isinstance(StopReason.END_TURN, str)


class TestMessage:
    def test_minimal_user_message(self):
        msg = Message(role=Role.USER, content="hi")
        assert msg.tool_calls == []
        assert msg.tool_result is None

    def test_role_coerces_from_string(self):
        assert Message(role="assistant", content="x").role is Role.ASSISTANT

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="robot", content="x")

    def test_assistant_message_with_tool_calls(self):
        call = ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[call])
        assert msg.tool_calls[0].arguments == {"cmd": "ls"}

    def test_tool_message_with_result(self):
        result = ToolResult(tool_call_id="c1", content="ok")
        msg = Message(role=Role.TOOL, tool_result=result)
        assert msg.tool_result is not None
        assert msg.tool_result.is_error is False

    def test_tool_result_error_flag(self):
        result = ToolResult(tool_call_id="c1", content="boom", is_error=True)
        assert result.is_error is True

    def test_default_tool_calls_lists_are_independent(self):
        a = Message(role=Role.ASSISTANT)
        b = Message(role=Role.ASSISTANT)
        a.tool_calls.append(ToolCall(id="c1", name="t", arguments={}))
        assert b.tool_calls == []

    def test_serialization_round_trip(self):
        msg = Message(
            role=Role.ASSISTANT,
            content="running it",
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
        )
        restored = Message.model_validate_json(msg.model_dump_json())
        assert restored == msg

    def test_tool_message_round_trip(self):
        msg = Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="c1", content="out", is_error=True),
        )
        restored = Message.model_validate(msg.model_dump())
        assert restored == msg


class TestToolSpec:
    def test_fields(self):
        spec = ToolSpec(
            name="bash",
            description="Run a command",
            input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )
        assert spec.input_schema["type"] == "object"

    def test_round_trip(self):
        spec = ToolSpec(name="t", description="d", input_schema={"a": [1, 2]})
        assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec

    def test_frozen(self):
        spec = ToolSpec(name="t", description="d")
        with pytest.raises(ValidationError):
            spec.name = "other"


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert (u.input_tokens, u.output_tokens) == (0, 0)
        assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)

    def test_addition(self):
        a = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=2)
        b = Usage(input_tokens=1, output_tokens=2, cache_write_tokens=7)
        total = a + b
        assert total == Usage(
            input_tokens=11,
            output_tokens=7,
            cache_read_tokens=2,
            cache_write_tokens=7,
        )

    def test_addition_does_not_mutate_operands(self):
        a = Usage(input_tokens=1)
        b = Usage(input_tokens=2)
        _ = a + b
        assert a.input_tokens == 1 and b.input_tokens == 2

    def test_sum_accumulation(self):
        usages = [Usage(input_tokens=i, output_tokens=1) for i in range(4)]
        total = sum(usages, Usage())
        assert total.input_tokens == 6
        assert total.output_tokens == 4

    def test_add_non_usage_rejected(self):
        with pytest.raises(TypeError):
            Usage() + "nope"  # type: ignore[operator]

    def test_round_trip(self):
        u = Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3)
        assert Usage.model_validate_json(u.model_dump_json()) == u


class TestModelResponse:
    def test_round_trip_with_raw(self):
        resp = ModelResponse(
            message=Message(role=Role.ASSISTANT, content="done"),
            usage=Usage(input_tokens=100, output_tokens=20),
            stop_reason=StopReason.END_TURN,
            raw={"provider": "anthropic", "id": "msg_1"},
        )
        restored = ModelResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp
        assert restored.stop_reason is StopReason.END_TURN

    def test_raw_optional(self):
        resp = ModelResponse(
            message=Message(role=Role.ASSISTANT, content="x"),
            usage=Usage(),
            stop_reason=StopReason.TOOL_USE,
        )
        assert resp.raw is None


class TestCapabilities:
    def test_fields_required(self):
        with pytest.raises(ValidationError):
            Capabilities(max_context=200_000)  # type: ignore[call-arg]

    def test_round_trip(self):
        caps = Capabilities(
            max_context=200_000,
            supports_cache_control=True,
        )
        assert Capabilities.model_validate(caps.model_dump()) == caps

    def test_unknown_fields_rejected(self):
        """Regression (A3 sweep): a removed capability field must fail
        loudly at construction, not be silently swallowed — a stub passing
        the deleted ``parallel_tool_calls`` survived the sweep exactly
        because pydantic's default is to ignore unknown kwargs."""
        with pytest.raises(ValidationError):
            Capabilities(
                parallel_tool_calls=True,  # type: ignore[call-arg]
                max_context=200_000,
                supports_cache_control=True,
            )
