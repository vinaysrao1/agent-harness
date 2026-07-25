"""Unit tests for harness.adapters.base."""

import pytest

from harness.adapters.base import (
    AdapterError,
    ModelAdapter,
    classify_http_fault,
    retry_with_backoff,
)
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


class FakeAdapter(ModelAdapter):
    """Minimal concrete adapter for exercising the base class."""

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            max_context=100_000,
            supports_cache_control=False,
        )

    async def complete(self, messages, tools, system=None, **params):
        return ModelResponse(
            message=Message(role=Role.ASSISTANT, content="ok"),
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason=StopReason.END_TURN,
        )


class TestModelAdapter:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ModelAdapter()  # type: ignore[abstract]

    async def test_concrete_adapter_completes(self):
        adapter = FakeAdapter()
        resp = await adapter.complete(
            [Message(role=Role.USER, content="hi")],
            [ToolSpec(name="t", description="d")],
            system="be brief",
        )
        assert resp.stop_reason is StopReason.END_TURN
        assert adapter.capabilities.max_context == 100_000

    def test_count_tokens_chars_over_four(self):
        adapter = FakeAdapter()
        # 40 chars of content -> 10 tokens.
        msgs = [Message(role=Role.USER, content="x" * 40)]
        assert adapter.count_tokens(msgs) == 10

    def test_count_tokens_includes_tool_calls_and_results(self):
        adapter = FakeAdapter()
        call = ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
        msgs = [
            Message(role=Role.ASSISTANT, tool_calls=[call]),
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="c1", content="y" * 80),
            ),
        ]
        expected = (len("bash") + len(repr({"cmd": "ls"})) + 80) // 4
        assert adapter.count_tokens(msgs) == expected

    def test_count_tokens_empty(self):
        assert FakeAdapter().count_tokens([]) == 0
        assert FakeAdapter().count_tokens([Message(role=Role.USER)]) == 0


class TestAdapterError:
    def test_defaults_not_retryable(self):
        assert AdapterError("bad request").retryable is False

    def test_retryable_flag(self):
        assert AdapterError("rate limited", retryable=True).retryable is True

    def test_message_preserved(self):
        assert str(AdapterError("boom")) == "boom"

    def test_fault_defaults_none(self):
        """Unclassified is the default, so every pre-taxonomy construction
        site keeps meaning 'an ordinary failure, score it as one'."""
        assert AdapterError("boom").fault is None

    @pytest.mark.parametrize(
        "fault",
        ["auth", "quota", "rate_limit", "server", "transport", "malformed_response"],
    )
    def test_fault_round_trips(self, fault):
        assert AdapterError("boom", fault=fault).fault == fault

    def test_fault_is_orthogonal_to_retryable(self):
        """The two flags answer different questions: retryable governs the
        adapter's own bounded retry loop, fault describes the surviving
        failure for whoever is above the harness."""
        exc = AdapterError("throttled", retryable=True, fault="rate_limit")
        assert (exc.retryable, exc.fault) == (True, "rate_limit")


#: Verbatim OpenRouter 403 body captured from a real Terminal-Bench run in
#: which an exhausted key killed 7 of 22 trials (the workspace-URL key id is
#: redacted; nothing in the classifier reads it). This is the exact text the
#: quota classification has to recognise.
QUOTA_403_BODY = (
    "Error code: 403 - {'error': {'message': 'Key limit exceeded (total "
    "limit). Manage it using https://openrouter.ai/workspaces/default/keys/"
    "<redacted>', 'code': 403}}"
)


class TestClassifyHTTPFault:
    """The single shared HTTP-status -> Fault rule, used by every adapter."""

    @pytest.mark.parametrize(
        "status,want",
        [
            (401, "auth"),
            (402, "quota"),
            (429, "rate_limit"),
            (500, "server"),
            (502, "server"),
            (529, "server"),  # Anthropic's overloaded_error
            (400, None),
            (404, None),
            (409, None),
            (None, None),
        ],
    )
    def test_status_table(self, status, want):
        assert classify_http_fault(status, "some body") == want

    def test_403_quota_body_from_real_run(self):
        assert classify_http_fault(403, QUOTA_403_BODY) == "quota"

    @pytest.mark.parametrize(
        "body",
        [
            "insufficient_quota",
            "You have exceeded your monthly QUOTA",
            "Your credit balance is too low",
            "Rate limit exceeded for this key",
        ],
    )
    def test_403_quota_markers_case_insensitive(self, body):
        assert classify_http_fault(403, body) == "quota"

    @pytest.mark.parametrize(
        "body",
        [
            "This model is not permitted in your region",
            "Your request was blocked by our moderation system",
            "Your key may not use this model",
            "",
        ],
    )
    def test_403_non_quota_stays_unclassified(self, body):
        """403 alone is over-broad. A region block, a moderation refusal, or
        a model-permission denial is a genuine scored failure and must NOT be
        laundered into an infrastructure fault — mis-classifying it would
        hide a real capability/config problem AND (for a benchmark harness
        keyed on the exception type) silently retry something that will never
        succeed."""
        assert classify_http_fault(403, body) is None

    def test_403_body_defaults_to_empty(self):
        assert classify_http_fault(403) is None


class TestRetryWithBackoff:
    """All tests inject a fake sleep: nothing here actually waits."""

    @staticmethod
    def make_sleep_recorder(delays: list[float]):
        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        return fake_sleep

    async def test_success_first_try_no_sleep(self):
        delays: list[float] = []

        async def fn():
            return 42

        result = await retry_with_backoff(
            fn, sleep=self.make_sleep_recorder(delays)
        )
        assert result == 42
        assert delays == []

    async def test_retries_retryable_then_succeeds(self):
        delays: list[float] = []
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise AdapterError("429", retryable=True)
            return "ok"

        result = await retry_with_backoff(
            fn,
            max_attempts=5,
            backoff_base=1.0,
            jitter=lambda: 0.0,
            sleep=self.make_sleep_recorder(delays),
        )
        assert result == "ok"
        assert attempts == 3
        assert delays == [1.0, 2.0]  # exponential, zero jitter

    async def test_non_retryable_raises_immediately(self):
        delays: list[float] = []
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            raise AdapterError("401 unauthorized", retryable=False)

        with pytest.raises(AdapterError):
            await retry_with_backoff(fn, sleep=self.make_sleep_recorder(delays))
        assert attempts == 1
        assert delays == []

    async def test_exhausts_attempts_and_reraises(self):
        delays: list[float] = []
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            raise AdapterError("still down", retryable=True)

        with pytest.raises(AdapterError, match="still down"):
            await retry_with_backoff(
                fn,
                max_attempts=4,
                jitter=lambda: 0.0,
                sleep=self.make_sleep_recorder(delays),
            )
        assert attempts == 4
        assert delays == [1.0, 2.0, 4.0]  # no sleep after the final attempt

    async def test_backoff_capped(self):
        delays: list[float] = []

        async def fn():
            raise AdapterError("down", retryable=True)

        with pytest.raises(AdapterError):
            await retry_with_backoff(
                fn,
                max_attempts=6,
                backoff_base=10.0,
                backoff_cap=15.0,
                jitter=lambda: 0.0,
                sleep=self.make_sleep_recorder(delays),
            )
        assert delays == [10.0, 15.0, 15.0, 15.0, 15.0]

    async def test_jitter_applied_multiplicatively(self):
        delays: list[float] = []

        async def fn():
            raise AdapterError("down", retryable=True)

        with pytest.raises(AdapterError):
            await retry_with_backoff(
                fn,
                max_attempts=2,
                backoff_base=2.0,
                jitter=lambda: 0.5,
                sleep=self.make_sleep_recorder(delays),
            )
        assert delays == [3.0]  # 2.0 * (1 + 0.5)

    async def test_other_exceptions_not_retried(self):
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            raise ValueError("not an adapter error")

        with pytest.raises(ValueError):
            await retry_with_backoff(fn, sleep=self.make_sleep_recorder([]))
        assert attempts == 1

    async def test_invalid_max_attempts(self):
        async def fn():
            return 1

        with pytest.raises(ValueError):
            await retry_with_backoff(fn, max_attempts=0)

    async def test_max_elapsed_stops_before_attempts_exhausted(self):
        # A fake clock that advances 100s per read simulates slow-hanging
        # attempts; with a 250s budget the sequence must give up early even
        # though max_attempts (5) has not been reached.
        delays: list[float] = []
        attempts = 0
        ticks = iter([0.0, 100.0, 200.0, 300.0, 400.0])

        async def fn():
            nonlocal attempts
            attempts += 1
            raise AdapterError("hanging", retryable=True)

        with pytest.raises(AdapterError, match="hanging"):
            await retry_with_backoff(
                fn,
                max_attempts=5,
                jitter=lambda: 0.0,
                sleep=self.make_sleep_recorder(delays),
                max_elapsed=250.0,
                clock=lambda: next(ticks),
            )
        # start=0; after attempt1 clock=100, 100+1<250 -> sleep; after attempt2
        # clock=200, 200+2<250 -> sleep; after attempt3 clock=300, 300+4>=250
        # -> re-raise without sleeping. Three attempts, two sleeps.
        assert attempts == 3
        assert delays == [1.0, 2.0]

    async def test_max_elapsed_none_is_unbounded_by_time(self):
        # Default (None) preserves the old behavior: bounded only by attempts.
        delays: list[float] = []
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            raise AdapterError("down", retryable=True)

        with pytest.raises(AdapterError):
            await retry_with_backoff(
                fn,
                max_attempts=3,
                jitter=lambda: 0.0,
                sleep=self.make_sleep_recorder(delays),
            )
        assert attempts == 3
        assert delays == [1.0, 2.0]
