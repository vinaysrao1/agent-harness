"""S-110: a refusal is reported as a refusal, never as an answer.

Observed on the Opus 5 benchmark run: three security-flavoured tasks came
back at turn 1 with `finish_reason: content_filter`, no content and no tool
calls. That arrives as an empty assistant turn, which the adapter labelled
with the generic placeholder "(provider returned an empty assistant
message)" -- and `looks_unfinished` accepts a non-empty sentence with no open
tasks as a finished run. So a model declining the task was recorded as the
agent completing and producing that sentence.

Two of the three were solved by other models, so this is a real scored loss.
But the defect worth fixing is the disguise, not the decline: the harness
reports what happened and does **not** re-prompt to get around a safety
decision.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.adapters.openai_compat import (
    EMPTY_MESSAGE_PLACEHOLDERS,
    from_openai_response,
)
from harness.loop import REFUSAL_EVENT, AgentResult
from harness.types import StopReason, Usage


def _refusal_response():
    """A provider response shaped exactly like the observed refusals."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="content_filter",
                message=SimpleNamespace(content=None, tool_calls=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=415,
            completion_tokens=0,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


class TestRefusalIsDistinguishable:
    def test_S110_refusal_gets_its_own_reason(self) -> None:
        response = from_openai_response(_refusal_response())
        assert response.stop_reason is StopReason.REFUSAL
        assert response.incomplete_reason == "refusal"

    def test_S110_refusal_text_does_not_read_as_an_answer(self) -> None:
        response = from_openai_response(_refusal_response())
        text = response.message.content or ""
        assert "declined" in text
        # The exact regression: it must no longer be the generic placeholder,
        # which said nothing about why and read like a provider glitch.
        assert text != EMPTY_MESSAGE_PLACEHOLDERS[None]

    def test_S110_generic_empty_message_is_still_generic(self) -> None:
        # A provider that returns nothing for no stated reason is a different
        # fact and must keep its own text.
        empty = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=None, tool_calls=None),
                )
            ],
            usage=None,
        )
        response = from_openai_response(empty)
        assert response.incomplete_reason != "refusal"
        assert "declined" not in (response.message.content or "")

    def test_S110_a_refusal_carrying_text_is_not_relabelled(self) -> None:
        # If the provider explains its refusal in prose, that prose is the
        # turn's content and must survive; the placeholder exists only for an
        # empty turn.
        spoken = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(
                        content="I can't help with that.", tool_calls=None
                    ),
                )
            ],
            usage=None,
        )
        response = from_openai_response(spoken)
        assert response.message.content == "I can't help with that."
        assert response.incomplete_reason != "refusal"


class TestResultCarriesTheRefusal:
    def test_S110_defaults_to_not_refused(self) -> None:
        result = AgentResult(
            status="completed", final_text="done", usage=Usage(), turns=1
        )
        assert result.refused is False

    def test_S110_refused_round_trips(self) -> None:
        result = AgentResult(
            status="completed",
            final_text="(the model declined...)",
            usage=Usage(),
            turns=1,
            refused=True,
        )
        assert result.refused is True

    def test_S110_refused_is_distinct_from_error(self) -> None:
        # Nothing failed. The model chose not to act, and a caller that
        # reported that as an infrastructure error would misattribute it.
        result = AgentResult(
            status="completed",
            final_text="x",
            usage=Usage(),
            turns=1,
            refused=True,
        )
        assert result.status == "completed"
        assert result.error_kind is None


class TestTelemetry:
    def test_S110_event_is_registered_against_this_spec(self) -> None:
        from harness.specs import EVENT_KIND_SPECS

        assert EVENT_KIND_SPECS.get(REFUSAL_EVENT) == "S-110"


class TestTheHarnessDoesNotArgueWithTheModel:
    """The fix reports the refusal. It does not try to get around it."""

    def test_S110_no_retry_or_reframing_on_refusal(self) -> None:
        from pathlib import Path

        source = Path("harness/loop.py").read_text(encoding="utf-8")
        block = source[source.index('== "refusal"') : source.index('== "refusal"') + 900]
        for forbidden in ("continue", "_append_message", "reminder", "retry"):
            assert forbidden not in block.lower(), (
                f"the refusal branch contains {forbidden!r}; the harness must "
                "record a decline, not re-prompt around it"
            )

    def test_S110_control_flow_is_unchanged(self) -> None:
        # Lane A depends on this: the refusal handling adds no completion path
        # and no nudge source, so N5's frozen counts still hold.
        from harness.loop import COMPLETION_GATES, NUDGE_SOURCES

        assert COMPLETION_GATES == ("diligence", "self_verification")
        assert NUDGE_SOURCES == ("diligence", "self_verification")
