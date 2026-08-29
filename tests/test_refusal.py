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
from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig
from harness.loop import REFUSAL_EVENT, AgentResult
from harness.orchestrator import Orchestrator
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.types import Message, ModelResponse, Role, StopReason, Usage

pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")


@pytest.fixture(autouse=True)
def _no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerSandbox, "availability", classmethod(lambda cls: False))


def _orchestrator(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return Orchestrator(HarnessConfig(home=home), RunStore(tmp_path / "state.db"))


def _clean_finish() -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.ASSISTANT, content="Task complete; verified."),
        usage=Usage(),
        stop_reason=StopReason.END_TURN,
    )


def _refusing_run(tmp_path):
    """An orchestrator plus an adapter whose first turn is a real refusal."""
    refusal = from_openai_response(_refusal_response())
    # Extra clean finishes so that if the loop DOES re-prompt, it has
    # somewhere to go -- otherwise the test would pass by the script running
    # out rather than by the loop declining to continue.
    return _orchestrator(tmp_path), FakeAdapter(
        [refusal, _clean_finish(), _clean_finish(), _clean_finish(), _clean_finish()]
    )


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
    """The fix reports the refusal. It does not try to get around it.

    The first version of this class asserted that a 900-character slice of
    loop.py contained none of four words. It passed while the loop was
    re-prompting refusing models three times, ~670 lines away, with a message
    telling them they had been cut off at the token limit. A wordlist over the
    wrong region is not a test. These drive the loop.
    """

    async def test_S110_a_refusal_is_not_re_prompted(self, tmp_path) -> None:
        orchestrator, adapter = _refusing_run(tmp_path)
        await orchestrator.run_task(
            "Do the thing.", "fake-model", adapter_override=adapter,
            workspace=tmp_path / "ws",
        )
        assert len(adapter.calls) == 1, (
            f"the harness called the model {len(adapter.calls)} times after a "
            "refusal; it must accept the decline, not argue with it"
        )

    async def test_S110_no_truncation_continue_on_a_refusal(self, tmp_path) -> None:
        # The specific regression: "refusal" became an IncompleteReason, which
        # made `incomplete` True, which routed it into truncation-continue --
        # where the reminder fallback told the model it had been cut off at the
        # output-token limit. False, and pressure on a safety decision.
        orchestrator, adapter = _refusing_run(tmp_path)
        run_id, _ = await orchestrator.run_task(
            "Do the thing.", "fake-model", adapter_override=adapter,
            workspace=tmp_path / "ws2",
        )
        agent_id = orchestrator.store.list_agents(run_id)[0].id
        kinds = [e.kind for e in orchestrator.store.load_events(agent_id)]
        assert "truncation_continue" not in kinds, (
            "a refusal entered the truncation-continue path"
        )

    async def test_S110_the_run_records_the_refusal(self, tmp_path) -> None:
        # Acceptance (3)'s untested half: `refused` True after a refusing turn.
        orchestrator, adapter = _refusing_run(tmp_path)
        run_id, result = await orchestrator.run_task(
            "Do the thing.", "fake-model", adapter_override=adapter,
            workspace=tmp_path / "ws3",
        )
        assert result.refused is True, "the run did not record the refusal"
        agent_id = orchestrator.store.list_agents(run_id)[0].id
        kinds = [e.kind for e in orchestrator.store.load_events(agent_id)]
        assert REFUSAL_EVENT in kinds

    async def test_S110_a_truncated_response_IS_still_re_prompted(
        self, tmp_path
    ) -> None:
        # The control. Without it, "no re-prompt on refusal" could pass simply
        # because truncation-continue stopped working for everything.
        from harness.adapters.fake import FakeAdapter

        orchestrator = _orchestrator(tmp_path)
        truncated = from_openai_response(
            SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content=None, tool_calls=None))],
                usage=None))
        adapter = FakeAdapter([truncated, _clean_finish()])
        await orchestrator.run_task(
            "Do the thing.", "fake-model", adapter_override=adapter,
            workspace=tmp_path / "ws4",
        )
        assert len(adapter.calls) > 1, (
            "a genuinely truncated response was not re-prompted; the refusal "
            "fix broke truncation-continue for everything"
        )

    def test_S110_refusal_is_declared_non_continuable(self) -> None:
        from harness.loop import NON_CONTINUABLE_REASONS

        assert "refusal" in NON_CONTINUABLE_REASONS

    def test_S110_control_flow_is_unchanged(self) -> None:
        from harness.loop import COMPLETION_GATES, NUDGE_SOURCES

        assert COMPLETION_GATES == ("diligence", "self_verification")
        assert NUDGE_SOURCES == ("diligence", "self_verification")
