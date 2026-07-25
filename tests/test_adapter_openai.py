"""Unit tests for harness.adapters.openai_compat (no network, no real API)."""

from __future__ import annotations

import asyncio
import json
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from harness.adapters.base import AdapterError
from harness.adapters.openai_compat import (
    EMPTY_ASSISTANT_PLACEHOLDER,
    EMPTY_MESSAGE_PLACEHOLDERS,
    OpenAICompatAdapter,
    accumulate_stream_chunks,
    drop_notice,
    from_openai_response,
    map_finish_reason,
    to_openai_messages,
    to_openai_tools,
    wrap_openai_error,
)
from harness.diligence import looks_unfinished
from harness.types import (
    DROPPED_ARGUMENTS_PREFIX_CHARS,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from tests.test_adapters_base import QUOTA_403_BODY


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

    def test_empty_assistant_message_gets_placeholder_not_error(self) -> None:
        # C1 replay backstop: an assistant message with neither content nor
        # tool calls must stay translatable, or every transcript already
        # holding one (including a persisted one being resumed) is dead.
        with pytest.warns(UserWarning, match="neither content nor tool calls"):
            out = to_openai_messages([Message(role=Role.ASSISTANT)])
        assert out == [
            {"role": "assistant", "content": EMPTY_ASSISTANT_PLACEHOLDER}
        ]

    def test_empty_assistant_placeholder_is_diligence_safe(self) -> None:
        # Must not look like a promise of future work or a question if it
        # ever reaches the diligence check.
        assert not EMPTY_ASSISTANT_PLACEHOLDER.rstrip().endswith("?")
        unfinished, _reason = looks_unfinished(EMPTY_ASSISTANT_PLACEHOLDER, 0)
        assert unfinished is False

    def test_empty_assistant_placeholder_not_written_back(self) -> None:
        # Translation is pure: the caller's Message must be untouched so the
        # persisted event log stays a faithful record of the provider output.
        message = Message(role=Role.ASSISTANT)
        with pytest.warns(UserWarning):
            to_openai_messages([message])
        assert message.content is None
        assert message.tool_calls == []

    def test_empty_assistant_message_among_others(self) -> None:
        with pytest.warns(UserWarning):
            out = to_openai_messages(
                [
                    Message(role=Role.USER, content="go"),
                    Message(role=Role.ASSISTANT),
                    Message(role=Role.USER, content="continue"),
                ]
            )
        assert [entry["content"] for entry in out] == [
            "go",
            EMPTY_ASSISTANT_PLACEHOLDER,
            "continue",
        ]

    def test_assistant_tool_calls_without_content_allowed(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
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

    def test_malformed_arguments_drop_the_call_instead_of_raising(
        self,
    ) -> None:
        # The arguments are already spent — no retry can repair them — so the
        # call is dropped and reported, never raised on.
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": ')],
            finish_reason="tool_calls",
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == []
        (dropped,) = result.dropped_tool_calls
        assert dropped.tool_name == "bash"
        assert dropped.raw_arguments_prefix == '{"cmd": '
        assert result.incomplete_reason == "dropped_calls"

    def test_non_object_arguments_also_dropped(self) -> None:
        # _parse_arguments rejects non-object payloads too; that is the same
        # unusable-payload case and degrades identically.
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '["not", "a", "dict"]')],
            finish_reason="tool_calls",
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == []
        assert [d.tool_name for d in result.dropped_tool_calls] == ["bash"]

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
        assert usage.reasoning_tokens == 0

    def test_reasoning_tokens_read_from_completion_tokens_details(self) -> None:
        resp = fake_response(
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=100,
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=1234
                ),
            )
        )
        usage = from_openai_response(resp).usage
        assert usage.reasoning_tokens == 1234
        # reasoning_tokens is a SUBSET of completion_tokens, not additional
        # traffic on top of it — output_tokens must not be reduced by it,
        # unlike the cache-token subtraction applied to input_tokens.
        assert usage.output_tokens == 100

    def test_reasoning_tokens_absent_completion_tokens_details_default_zero(
        self,
    ) -> None:
        resp = fake_response(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        )
        assert from_openai_response(resp).usage.reasoning_tokens == 0

    def test_reasoning_tokens_none_default_zero(self) -> None:
        resp = fake_response(
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=None
                ),
            )
        )
        assert from_openai_response(resp).usage.reasoning_tokens == 0

    def test_reasoning_tokens_usage_none_default_zero(self) -> None:
        resp = fake_response(content="hi")
        resp.usage = None
        assert from_openai_response(resp).usage.reasoning_tokens == 0

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

    @pytest.mark.parametrize(
        ("provider", "ours"),
        [
            ("stop", StopReason.END_TURN),
            ("tool_calls", StopReason.TOOL_USE),
            ("length", StopReason.MAX_TOKENS),
            ("content_filter", StopReason.REFUSAL),
            ("weird_reason", StopReason.ERROR),
            (None, StopReason.ERROR),
        ],
    )
    def test_provider_stop_reason_carried_through_verbatim(
        self, provider: str | None, ours: StopReason
    ) -> None:
        """§C2: the raw finish_reason survives translation untouched while
        stop_reason maps exactly as before. The two ERROR rows are the point
        — after mapping alone, an unmapped provider string and a missing one
        are the same value, and post-hoc triage cannot tell them apart."""
        result = from_openai_response(fake_response(finish_reason=provider))
        assert result.provider_stop_reason == provider
        assert result.stop_reason is ours

    def test_provider_stop_reason_absent_attribute_is_none(self) -> None:
        """A response object with no finish_reason attribute at all (not
        merely None) must not raise; provenance is best-effort."""
        message = SimpleNamespace(content="hi", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        result = from_openai_response(response)
        assert result.provider_stop_reason is None
        assert result.stop_reason is StopReason.ERROR


# ------------------------- truncated tool calls (graceful degradation)


#: The real failure shape (trial make-mips-interpreter__KSxCFCR): a large
#: inline write_file whose JSON arguments were cut mid-string at the
#: output-token cap. json.loads fails on it; the run must survive anyway.
TRUNCATED_WRITE_FILE_ARGS = (
    '{"path": "interp.py", "content": "def main():\\n    regs = [0] * 32\\n'
    "    # ... hundreds of lines of interpreter code that hit the cap"
)


class TestTruncatedToolCallDegradation:
    def test_length_truncated_malformed_call_dropped_not_raised(self) -> None:
        # Pinned regression: finish_reason="length" + args cut mid-JSON must
        # translate (call dropped, MAX_TOKENS), not raise a non-retryable
        # AdapterError that kills the whole run.
        resp = fake_response(
            content=None,
            tool_calls=[
                fake_tool_call("c1", "write_file", TRUNCATED_WRITE_FILE_ARGS)
            ],
            finish_reason="length",
        )
        result = from_openai_response(resp)
        assert result.stop_reason is StopReason.MAX_TOKENS
        assert result.message.tool_calls == []

    @pytest.mark.parametrize(
        "finish_reason",
        [None, "stop", "tool_calls", "error", "length", "weird_reason"],
    )
    def test_malformed_args_never_raise_whatever_the_finish_reason(
        self, finish_reason: str | None
    ) -> None:
        """The old rule keyed degradation on ``finish_reason == "length"``,
        which is wrong: a call cut off mid-JSON is equally unrecoverable under
        every other finish reason, and the real make-mips trial died under a
        *missing* one. Whatever the provider says it stopped for, an
        unparseable call is dropped and the reason is ``dropped_calls``."""
        resp = fake_response(
            content=None,
            tool_calls=[
                fake_tool_call("c1", "write_file", TRUNCATED_WRITE_FILE_ARGS)
            ],
            finish_reason=finish_reason,
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == []
        assert result.incomplete is True
        assert result.incomplete_reason == "dropped_calls"
        (dropped,) = result.dropped_tool_calls
        assert dropped.tool_name == "write_file"
        assert dropped.raw_arguments_len == len(TRUNCATED_WRITE_FILE_ARGS)
        assert drop_notice(1) in result.message.content
        # stop_reason stays a faithful translation of what the provider said.
        assert result.provider_stop_reason == finish_reason

    def test_dropped_call_reported_with_capped_prefix(self) -> None:
        # The loop persists these; the prefix must show the cut point without
        # carrying a whole truncated file body (one real fragment was ~21 KB).
        huge = '{"path": "big.py", "content": "' + "x" * 21_000
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "write_file", huge)],
            finish_reason="length",
        )
        (dropped,) = from_openai_response(resp).dropped_tool_calls
        assert dropped.tool_name == "write_file"
        assert dropped.raw_arguments_len == len(huge)
        assert dropped.raw_arguments_prefix == huge[
            :DROPPED_ARGUMENTS_PREFIX_CHARS
        ]
        assert len(dropped.raw_arguments_prefix) == (
            DROPPED_ARGUMENTS_PREFIX_CHARS
        )

    def test_all_calls_dropped_and_no_content_gets_drop_notice(self) -> None:
        # An empty assistant message would force to_openai_messages to invent
        # a body when the transcript is replayed next turn; the notice keeps
        # the transcript translatable *and* tells the model what happened.
        resp = fake_response(
            content=None,
            tool_calls=[
                fake_tool_call("c1", "write_file", TRUNCATED_WRITE_FILE_ARGS)
            ],
            finish_reason="length",
        )
        result = from_openai_response(resp)
        assert result.message.content == drop_notice(1)
        # Next-turn translation of the transcript does not raise or warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (entry,) = to_openai_messages([result.message])
        assert entry["role"] == "assistant"
        assert "dropped" in entry["content"]

    def test_drop_notice_appended_even_when_a_sibling_survives(self) -> None:
        """M2: the drop must be announced even on the sibling-survivor path.
        The loop takes its tool-call branch there and never consults
        ``incomplete``, so this notice — read back as the model's own prior
        turn — is the only way it learns call 2 vanished."""
        resp = fake_response(
            content="Listing, then writing the file.",
            tool_calls=[
                fake_tool_call("c1", "bash", '{"cmd": "ls"}'),
                fake_tool_call("c2", "write_file", TRUNCATED_WRITE_FILE_ARGS),
            ],
            finish_reason="tool_calls",
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == [
            ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        ]
        assert result.message.content == (
            "Listing, then writing the file.\n\n" + drop_notice(1)
        )
        assert [d.tool_name for d in result.dropped_tool_calls] == ["write_file"]
        assert result.incomplete_reason == "dropped_calls"

    def test_existing_content_is_kept_and_the_notice_appended(self) -> None:
        resp = fake_response(
            content="Writing the interpreter now.",
            tool_calls=[
                fake_tool_call("c1", "write_file", TRUNCATED_WRITE_FILE_ARGS)
            ],
            finish_reason="length",
        )
        result = from_openai_response(resp)
        assert result.message.content.startswith("Writing the interpreter now.")
        assert result.message.content.endswith(drop_notice(1))
        assert result.message.tool_calls == []

    def test_two_dropped_calls_are_reported_and_counted_in_the_notice(
        self,
    ) -> None:
        resp = fake_response(
            content=None,
            tool_calls=[
                fake_tool_call("c1", "write_file", TRUNCATED_WRITE_FILE_ARGS),
                fake_tool_call("c2", "bash", '{"cmd": '),
            ],
            finish_reason="length",
        )
        result = from_openai_response(resp)
        assert len(result.dropped_tool_calls) == 2
        assert result.message.content == drop_notice(2)
        assert "2 tool calls were dropped" in result.message.content

    def test_non_object_args_also_dropped_when_truncated(self) -> None:
        # _parse_arguments rejects non-object payloads too; that is the same
        # salvageable failure.
        resp = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '"cut off mid')],
            finish_reason="length",
        )
        result = from_openai_response(resp)
        assert result.message.tool_calls == []
        assert result.stop_reason is StopReason.MAX_TOKENS
        assert result.incomplete_reason == "dropped_calls"

    def test_injected_texts_cannot_trip_the_diligence_check(self) -> None:
        # Every one of these becomes assistant content and can reach
        # looks_unfinished; none may read as promised future work, and none
        # may end in '?'. A false "unfinished" here would spend a nudge on a
        # string the harness wrote itself.
        texts = [drop_notice(1), drop_notice(2)]
        texts += list(EMPTY_MESSAGE_PLACEHOLDERS.values())
        for text in texts:
            assert not text.rstrip().endswith("?")
            assert looks_unfinished(text, 0)[0] is False

    def test_streaming_truncated_args_behave_identically(self) -> None:
        # The same truncated arguments arriving as stream fragments fold into
        # a length-finished response and degrade the same way.
        resp = accumulate_stream_chunks(
            [
                stream_chunk(
                    tool_calls=[
                        tc_delta(
                            0,
                            id="c1",
                            name="write_file",
                            arguments='{"path": "interp.py", "content": "def ',
                        )
                    ]
                ),
                stream_chunk(
                    tool_calls=[tc_delta(0, arguments="main():\\n    regs")]
                ),
                stream_chunk(finish_reason="length"),
            ]
        )
        result = from_openai_response(resp)
        assert result.stop_reason is StopReason.MAX_TOKENS
        assert result.message.tool_calls == []
        assert result.message.content == drop_notice(1)

    def test_streaming_truncated_args_with_no_terminal_chunk(self) -> None:
        """The actual make-mips shape: the tool-call fragments arrived but no
        terminal chunk ever did, so ``finish_reason`` was None — the one case
        the ``"length"``-keyed rule did not cover, and the one that killed the
        run. Same fragments as the test above, minus the terminal chunk."""
        resp = accumulate_stream_chunks(
            [
                stream_chunk(
                    tool_calls=[
                        tc_delta(
                            0,
                            id="c1",
                            name="write_file",
                            arguments='{"path": "interp.py", "content": "def ',
                        )
                    ]
                ),
                stream_chunk(
                    tool_calls=[tc_delta(0, arguments="main():\\n    regs")]
                ),
            ]
        )
        result = from_openai_response(resp)
        assert result.provider_stop_reason is None
        assert result.message.tool_calls == []
        assert result.incomplete_reason == "dropped_calls"
        assert result.message.content == drop_notice(1)


# ------------------------- incomplete-turn derivation (M1 gating)


class TestIncompleteDerivation:
    def test_content_without_a_terminal_chunk_is_not_incomplete(self) -> None:
        """**The M1 regression guard.** A stream that delivered a real answer
        but no terminal chunk maps to StopReason.ERROR. Treating that alone as
        incomplete would give any gateway that omits ``finish_reason`` three
        spurious re-prompts at the end of *every* run, each telling a model
        that just answered correctly it had been cut off — false, and at
        17-40s a turn, fatal after wind-down."""
        resp = accumulate_stream_chunks(
            [stream_chunk(content="The answer is 42. All tests pass.")]
        )
        result = from_openai_response(resp)
        assert result.stop_reason is StopReason.ERROR
        assert result.provider_stop_reason is None
        assert result.incomplete is False
        assert result.incomplete_reason is None

    def test_surviving_tool_call_without_terminal_chunk_is_not_incomplete(
        self,
    ) -> None:
        # Same gate, via the other half of "usable": a parseable tool call.
        resp = accumulate_stream_chunks(
            [
                stream_chunk(
                    tool_calls=[
                        tc_delta(
                            0, id="c1", name="bash", arguments='{"cmd": "ls"}'
                        )
                    ]
                )
            ]
        )
        result = from_openai_response(resp)
        assert result.stop_reason is StopReason.ERROR
        assert result.incomplete is False

    def test_empty_message_without_terminal_chunk_is_no_finish_reason(
        self,
    ) -> None:
        # Nothing usable *and* no recognised stop reason: the only shape the
        # no_finish_reason branch is allowed to fire on.
        result = from_openai_response(
            fake_response(content=None, tool_calls=None, finish_reason=None)
        )
        assert result.incomplete is True
        assert result.incomplete_reason == "no_finish_reason"

    def test_unmapped_finish_reason_with_empty_message_is_no_finish_reason(
        self,
    ) -> None:
        result = from_openai_response(
            fake_response(
                content=None, tool_calls=None, finish_reason="weird_reason"
            )
        )
        assert result.incomplete_reason == "no_finish_reason"
        assert result.provider_stop_reason == "weird_reason"

    def test_max_tokens_beats_no_finish_reason_but_loses_to_dropped_calls(
        self,
    ) -> None:
        # Precedence: dropped_calls > max_tokens > no_finish_reason.
        truncated_empty = from_openai_response(
            fake_response(content=None, tool_calls=None, finish_reason="length")
        )
        assert truncated_empty.incomplete_reason == "max_tokens"

        truncated_with_drop = from_openai_response(
            fake_response(
                content=None,
                tool_calls=[
                    fake_tool_call(
                        "c1", "write_file", TRUNCATED_WRITE_FILE_ARGS
                    )
                ],
                finish_reason="length",
            )
        )
        assert truncated_with_drop.incomplete_reason == "dropped_calls"

    def test_max_tokens_with_a_surviving_call_is_still_incomplete(self) -> None:
        """``max_tokens`` is unconditional per the precedence table — it does
        not require an empty message. The loop still dispatches the surviving
        call (its tool-call branch runs first), so this costs no re-prompt;
        the flag is what a caller inspects to know the turn was cut short."""
        result = from_openai_response(
            fake_response(
                content="writing",
                tool_calls=[fake_tool_call("c1", "bash", '{"cmd": "ls"}')],
                finish_reason="length",
            )
        )
        assert result.incomplete_reason == "max_tokens"
        assert len(result.message.tool_calls) == 1

    def test_clean_finish_is_not_incomplete(self) -> None:
        result = from_openai_response(fake_response(content="done"))
        assert result.incomplete is False
        assert result.incomplete_reason is None
        assert result.dropped_tool_calls == []

    @pytest.mark.parametrize(
        ("finish_reason", "reason", "text"),
        [
            (
                "length",
                "max_tokens",
                "(response truncated at the output-token limit before "
                "producing any output)",
            ),
            (
                None,
                "no_finish_reason",
                "(provider response ended without completing)",
            ),
            (
                "stop",
                None,
                "(provider returned an empty assistant message)",
            ),
        ],
    )
    def test_empty_message_gets_a_cause_specific_placeholder(
        self, finish_reason: str | None, reason: str | None, text: str
    ) -> None:
        """A turn with no content and no tool calls used to be persisted empty
        and then rejected on replay, killing a real run at turn 1. The
        substitute names the cause, since the model reads it back as its own
        prior turn. Note the ``"stop"`` row: a *recognised* stop reason with an
        empty body still gets a body, but is NOT incomplete — that is the M1
        gate, which confines re-prompting to unrecognised stop reasons."""
        result = from_openai_response(
            fake_response(
                content=None, tool_calls=None, finish_reason=finish_reason
            )
        )
        assert result.message.content == text
        assert result.message.content == EMPTY_MESSAGE_PLACEHOLDERS[reason]
        assert result.incomplete is (reason is not None)
        assert result.incomplete_reason == reason
        # Replay-safe and diligence-safe.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_openai_messages([result.message])
        assert looks_unfinished(text, 0)[0] is False


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


class FakeStatusErrorWithBody(Exception):
    """An SDK error whose str() carries the provider's error body, which is
    how the openai SDK actually renders an APIStatusError."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code


class TestWrapOpenAIErrorFault:
    """The provider-fault taxonomy, which is orthogonal to ``retryable``.

    The classification is what lets a benchmark harness record an exhausted
    key as *infrastructure* rather than as an agent capability failure. It
    must be conservative: anything it cannot positively identify stays
    ``None`` and is scored as an ordinary failure.
    """

    @pytest.mark.parametrize(
        "status,want",
        [
            (401, "auth"),
            (402, "quota"),
            (429, "rate_limit"),
            (500, "server"),
            (503, "server"),
            (400, None),
            (404, None),
            (408, None),
        ],
    )
    def test_status_classification(self, status: int, want: str | None) -> None:
        assert wrap_openai_error(FakeStatusError(status)).fault == want

    def test_403_quota_body_from_the_real_run(self) -> None:
        """The load-bearing case: 7 of 22 trials in one Terminal-Bench run
        died on exactly this body and were recorded by Harbor as clean
        completions with reward 0."""
        exc = FakeStatusErrorWithBody(403, QUOTA_403_BODY)
        wrapped = wrap_openai_error(exc)
        assert wrapped.fault == "quota"
        assert wrapped.retryable is False  # credit is gone; do not re-burn it

    @pytest.mark.parametrize(
        "body",
        [
            "Error code: 403 - this model is not permitted in your region",
            "Error code: 403 - blocked by moderation",
            "Error code: 403 - your key may not use this model",
        ],
    )
    def test_403_non_quota_stays_a_scored_failure(self, body: str) -> None:
        # 403 alone is over-broad; only a quota-shaped body may be laundered
        # into an infrastructure fault.
        assert wrap_openai_error(FakeStatusErrorWithBody(403, body)).fault is None

    @pytest.mark.parametrize(
        "exc_name",
        [
            "ReadError",
            "ReadTimeout",
            "WriteError",
            "RemoteProtocolError",
            "StreamError",
            "IncompleteRead",
            "PoolTimeout",
            "APIConnectionError",
        ],
    )
    def test_statusless_transport_errors_are_transport_faults(
        self, exc_name: str
    ) -> None:
        exc = type(exc_name, (Exception,), {})()
        assert wrap_openai_error(exc).fault == "transport"

    def test_builtin_timeout_is_a_transport_fault(self) -> None:
        assert wrap_openai_error(TimeoutError()).fault == "transport"

    def test_unknown_sdk_error_unclassified(self) -> None:
        assert wrap_openai_error(RuntimeError("boom")).fault is None

    def test_passthrough_adapter_error_keeps_its_own_fault(self) -> None:
        original = AdapterError("already classified", fault="quota")
        assert wrap_openai_error(original).fault == "quota"


class TestInlineGatewayErrorFault:
    """OpenRouter reports faults as HTTP 200 bodies with an inline error
    object; the same taxonomy has to apply there or the most common real
    failure path stays unclassified."""

    @staticmethod
    def _raise(code: object, message: str) -> AdapterError:
        resp = SimpleNamespace(
            choices=[], usage=None, error={"code": code, "message": message}
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        return excinfo.value

    @pytest.mark.parametrize(
        "code,want",
        [
            (401, "auth"),
            (402, "quota"),
            (429, "rate_limit"),
            (500, "server"),
            (400, None),
        ],
    )
    def test_code_classification(self, code: int, want: str | None) -> None:
        assert self._raise(code, "upstream said so").fault == want

    def test_403_quota_body(self) -> None:
        assert self._raise(403, "Key limit exceeded (total limit)").fault == "quota"

    def test_403_non_quota_body(self) -> None:
        assert self._raise(403, "not permitted in your region").fault is None

    def test_codeless_inline_error_unclassified(self) -> None:
        """No code means no evidence: stay retryable (as before) but refuse
        to guess a fault kind."""
        resp = SimpleNamespace(
            choices=[], usage=None, error={"message": "something transient"}
        )
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(resp)
        assert excinfo.value.retryable is True
        assert excinfo.value.fault is None

    def test_empty_choices_unclassified(self) -> None:
        with pytest.raises(AdapterError) as excinfo:
            from_openai_response(SimpleNamespace(choices=[], usage=None))
        assert excinfo.value.fault is None


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

    async def test_hard_timeout_is_a_transport_fault_after_retries(self) -> None:
        # The adapter raises its *own* AdapterError for a timeout, before
        # wrap_openai_error ever sees it (asyncio.TimeoutError is TimeoutError
        # on 3.11+), so the taxonomy has to be attached at that construction
        # site. Asserted through complete() with the retry budget spent,
        # because that is the exception the agent loop turns into
        # AgentResult.error_kind -> Harbor's NetworkConnectionError.
        class HungAPI:
            async def create(self, **kwargs: Any) -> Any:
                await asyncio.sleep(10.0)

        async def fake_sleep(delay: float) -> None:
            return None

        client = SimpleNamespace(chat=SimpleNamespace(completions=HungAPI()))
        adapter = OpenAICompatAdapter(
            "m", client=client, stream=False, request_timeout=0.02,
            retry={"max_attempts": 3, "sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.fault == "transport"

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

    async def test_malformed_tool_json_degrades_without_retrying(self) -> None:
        # Malformed arguments are a spent payload: retrying cannot repair
        # them, so the call is dropped in place and exactly one request is
        # made. (This asserted a raise before C4; the run-killing behaviour it
        # pinned is what the change removes.)
        bad = fake_response(
            content=None,
            tool_calls=[fake_tool_call("c1", "bash", '{"cmd": ')],
            finish_reason="tool_calls",
        )
        client = fake_client([bad, fake_response(content="unreached")])
        adapter = OpenAICompatAdapter("m", client=client, stream=False)
        result = await adapter.complete(
            [Message(role=Role.USER, content="hi")], []
        )
        assert result.incomplete_reason == "dropped_calls"
        assert [d.tool_name for d in result.dropped_tool_calls] == ["bash"]
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

    def test_reasoning_tokens_carried_through_terminal_usage_chunk(self) -> None:
        # Regression: the fold must carry completion_tokens_details through
        # the final usage-only chunk (choices == []), the same object the
        # non-streamed path reads reasoning_tokens off of.
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=42,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=17),
        )
        resp = accumulate_stream_chunks(
            [stream_chunk(content="hi"), stream_chunk(finish_reason="stop", usage=usage)]
        )
        translated = from_openai_response(resp)
        assert translated.usage.reasoning_tokens == 17
        assert translated.usage.output_tokens == 42

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

    def test_provider_stop_reason_survives_the_streaming_path(self) -> None:
        """§C2: the accumulated shape feeds the same translator, so the
        terminal chunk's verbatim finish_reason reaches the response."""
        resp = accumulate_stream_chunks(
            [stream_chunk(content="hi"), stream_chunk(finish_reason="length")]
        )
        assert from_openai_response(resp).provider_stop_reason == "length"

    def test_stream_with_no_terminal_chunk_reports_no_stop_reason(self) -> None:
        """The observed mid-generation death (trial make-mips-interpreter):
        content arrived, then the stream ended with no finish_reason chunk.
        stop_reason maps to ERROR — the same value an unmapped provider
        string yields — so ``provider_stop_reason is None`` is the only
        durable evidence that nothing terminal was ever sent."""
        resp = accumulate_stream_chunks([stream_chunk(content="partial")])
        translated = from_openai_response(resp)
        assert translated.provider_stop_reason is None
        assert translated.stop_reason is StopReason.ERROR
        assert translated.message.content == "partial"


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

    async def test_stall_surviving_retries_is_a_transport_fault(self) -> None:
        # Mid-stream death is the trial-forfeiting fault class this taxonomy
        # exists for. A stall that outlives the retry budget must reach the
        # caller classified as ``transport`` (-> NetworkConnectionError, which
        # Harbor does *not* exclude from -r retries), not unclassified — which
        # would score an infrastructure failure as a capability failure.
        streams = [
            FakeStream([stream_chunk(content="a")], gaps=[0.3]) for _ in range(3)
        ]
        client = fake_streaming_client(streams)

        async def fake_sleep(delay: float) -> None:
            return None

        adapter = OpenAICompatAdapter(
            "m", client=client, stream_idle_timeout=0.05,
            retry={"max_attempts": 3, "sleep": fake_sleep, "jitter": lambda: 0.0},
        )
        with pytest.raises(AdapterError) as excinfo:
            await adapter.complete([Message(role=Role.USER, content="hi")], [])
        assert excinfo.value.fault == "transport"
        assert excinfo.value.retryable is True
        assert len(client.chat.completions.calls) == 3

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
