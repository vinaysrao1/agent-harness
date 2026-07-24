"""Tests for harness.loop (DESIGN.md §4.1).

No network, no API keys, no Docker: every test drives :class:`AgentLoop`
with a scripted :class:`FakeAdapter`, a tmp-dir :class:`RunStore`, and —
where a real sandbox tool is exercised — :class:`LocalSandbox` on
``tmp_path``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from harness.adapters.fake import FakeAdapter
from harness.config import PermissionMode
from harness.context import COMPACTION_SUMMARY_PREFIX, ContextManager
from harness.deadline import Deadline
from harness.loop import AgentLoop, AgentResult, Budgets, wind_down_threshold
from harness.permissions import Policy, ToolMeta
from harness.persistence import RunStore
from harness.sandbox.local import LocalSandbox
from harness.tools.builtin import bash_tool
from harness.tools.registry import Tool, ToolRegistry
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

GOAL = "Ship the widget."

#: A final message the diligence check accepts as finished.
CLEAN_FINISH = "Task complete. All tests pass: 3 passed in 0.02s."


def resp(
    content: str | None = None,
    calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
    stop_reason: StopReason | None = None,
) -> ModelResponse:
    """Build one scripted assistant response.

    ``stop_reason`` defaults to TOOL_USE when there are tool calls else
    END_TURN; pass it explicitly to script a truncated turn (MAX_TOKENS).
    """
    tool_calls = calls or []
    if stop_reason is None:
        stop_reason = (
            StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
        )
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT, content=content, tool_calls=tool_calls
        ),
        usage=usage or Usage(),
        stop_reason=stop_reason,
    )


def call(id: str, name: str, **arguments: object) -> ToolCall:
    """Build one tool call."""
    return ToolCall(id=id, name=name, arguments=dict(arguments))


def simple_tool(
    name: str,
    *,
    side_effect: bool = False,
    delay: float = 0.0,
    log: list[str] | None = None,
) -> Tool:
    """A test tool that echoes its ``text`` argument, optionally after a
    delay (for completion-order tests) and logging its execution."""

    async def handler(arguments: dict) -> str:
        if delay:
            await asyncio.sleep(delay)
        if log is not None:
            log.append(name)
        return f"{name}:{arguments.get('text', '')}"

    return Tool(
        spec=ToolSpec(name=name, description=f"test tool {name}"),
        meta=ToolMeta(side_effect=side_effect),
        handler=handler,
    )


async def stub_summarize(messages: list[Message]) -> str:
    return f"STUB SUMMARY of {len(messages)} messages"


@dataclass
class Harness:
    """Everything a test needs to poke at one wired-up AgentLoop."""

    loop: AgentLoop
    adapter: FakeAdapter
    store: RunStore
    run_id: str
    agent_id: str
    ask_log: list[tuple[str, dict, ToolMeta]] = field(default_factory=list)

    def event_kinds(self) -> list[str]:
        return [e.kind for e in self.store.load_events(self.agent_id)]

    def events(self, kind: str) -> list[dict]:
        return [
            e.payload
            for e in self.store.load_events(self.agent_id)
            if e.kind == kind
        ]


def make_harness(
    tmp_path: Path,
    script: list[ModelResponse],
    *,
    tools: list[Tool] = (),
    policy: Policy | None = None,
    budgets: Budgets | None = None,
    ask_answer: bool = True,
    context: ContextManager | None = None,
    clock: "Callable[[], float] | None" = None,
    sandbox: LocalSandbox | None = None,
    deadline: Deadline | None = None,
) -> Harness:
    """Wire a full AgentLoop from real lower layers on ``tmp_path``."""
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(GOAL, "fake-model", "auto")
    agent_id = store.create_agent(run_id, GOAL)
    adapter = FakeAdapter(script)
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    if context is None:
        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=adapter.count_tokens,
            max_context=adapter.capabilities.max_context,
            summarize=stub_summarize,
        )
    ask_log: list[tuple[str, dict, ToolMeta]] = []

    async def ask(tool_name: str, arguments: dict, meta: ToolMeta) -> bool:
        ask_log.append((tool_name, arguments, meta))
        return ask_answer

    loop = AgentLoop(
        adapter,
        registry,
        policy or Policy(mode=PermissionMode.AUTO),
        store,
        run_id,
        agent_id,
        context,
        budgets or Budgets(),
        ask,
        model="fake-model",
        sandbox=sandbox,
        deadline=deadline,
        **({"clock": clock} if clock is not None else {}),
    )
    return Harness(
        loop=loop,
        adapter=adapter,
        store=store,
        run_id=run_id,
        agent_id=agent_id,
        ask_log=ask_log,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_multi_turn_with_tool_calls(self, tmp_path: Path) -> None:
        """Two tool turns (one via a real LocalSandbox bash tool), then a
        clean finish: completed, with usage and turns accounted."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        sandbox = LocalSandbox(workspace)
        script = [
            resp(
                "listing",
                [call("c1", "bash", command="echo hello-from-sandbox")],
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            resp(
                "echoing",
                [call("c2", "echo", text="hi")],
                usage=Usage(input_tokens=20, output_tokens=6),
            ),
            resp(CLEAN_FINISH, usage=Usage(input_tokens=30, output_tokens=7)),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox), simple_tool("echo")],
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert result.turns == 3
        assert result.usage == Usage(input_tokens=60, output_tokens=18)

        # The bash result (with real sandbox output) went back to the model.
        second_call_messages = h.adapter.calls[1].messages
        tool_payloads = [
            m.tool_result.content
            for m in second_call_messages
            if m.tool_result is not None
        ]
        assert any("hello-from-sandbox" in p for p in tool_payloads)
        # Goal seeded as the first user message.
        assert h.adapter.calls[0].messages[0] == Message(
            role=Role.USER, content=GOAL
        )
        assert h.store.get_agent(h.agent_id).status == "completed"

    async def test_unknown_tool_becomes_error_result(
        self, tmp_path: Path
    ) -> None:
        script = [
            resp("trying", [call("c1", "no_such_tool")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)
        assert result.status == "completed"
        (payload,) = h.events("tool_result")
        assert payload["is_error"] is True
        assert "unknown tool" in payload["content"]


# ---------------------------------------------------------------------------
# Parallel dispatch ordering
# ---------------------------------------------------------------------------


class TestParallelDispatch:
    async def test_results_keep_original_call_order(
        self, tmp_path: Path
    ) -> None:
        """The slow first call finishes after the fast second one, yet
        results are appended in the original tool-call order."""
        completion_order: list[str] = []
        script = [
            resp(
                None,
                [
                    call("slow-id", "slow", text="a"),
                    call("fast-id", "fast", text="b"),
                ],
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[
                simple_tool("slow", delay=0.05, log=completion_order),
                simple_tool("fast", log=completion_order),
            ],
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        # They genuinely ran concurrently: fast completed first ...
        assert completion_order == ["fast", "slow"]
        # ... but persisted results follow the original call order.
        results = h.events("tool_result")
        assert [r["tool_call_id"] for r in results] == ["slow-id", "fast-id"]
        assert results[0]["content"] == "slow:a"
        assert results[1]["content"] == "fast:b"
        # And the transcript fed back to the model has the same order.
        feedback = [
            m.tool_result.tool_call_id
            for m in h.adapter.calls[1].messages
            if m.tool_result is not None
        ]
        assert feedback == ["slow-id", "fast-id"]


# ---------------------------------------------------------------------------
# Permissions: ASK / DENY
# ---------------------------------------------------------------------------


class TestPermissions:
    async def test_ask_approved_dispatches(self, tmp_path: Path) -> None:
        ran: list[str] = []
        script = [
            resp(None, [call("c1", "send", text="msg")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("send", side_effect=True, log=ran)],
            policy=Policy(mode=PermissionMode.GATED),
            ask_answer=True,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert ran == ["send"]  # the handler actually executed
        assert h.ask_log == [
            ("send", {"text": "msg"}, ToolMeta(side_effect=True))
        ]
        (decision,) = h.events("decision")
        assert decision["decision"] == "allow"
        assert decision["decided_by"] == "user"
        (approval,) = h.store.list_approvals(h.run_id)
        assert (approval.decision, approval.decided_by) == ("allow", "user")

    async def test_ask_denied_returns_error_result(
        self, tmp_path: Path
    ) -> None:
        ran: list[str] = []
        script = [
            resp(None, [call("c1", "send", text="msg")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("send", side_effect=True, log=ran)],
            policy=Policy(mode=PermissionMode.GATED),
            ask_answer=False,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert ran == []  # handler never executed
        assert len(h.ask_log) == 1  # asked exactly once
        (payload,) = h.events("tool_result")
        assert payload == {
            "tool_call_id": "c1",
            "content": "denied by user",
            "is_error": True,
        }
        # The denial went back to the model as a tool result.
        feedback = [
            m.tool_result.content
            for m in h.adapter.calls[1].messages
            if m.tool_result is not None
        ]
        assert feedback == ["denied by user"]

    async def test_policy_deny_skips_ask_and_preserves_order(
        self, tmp_path: Path
    ) -> None:
        """A denied call and an allowed call in one turn: no ask() for the
        deny, and results stay in original order."""
        ran: list[str] = []
        script = [
            resp(
                None,
                [
                    call("d1", "danger_zone", text="x"),
                    call("a1", "echo", text="y"),
                ],
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[
                simple_tool("danger_zone", log=ran),
                simple_tool("echo", log=ran),
            ],
            policy=Policy(mode=PermissionMode.AUTO, deny=("danger*",)),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.ask_log == []  # DENY never consults the user
        assert ran == ["echo"]
        results = h.events("tool_result")
        assert [r["tool_call_id"] for r in results] == ["d1", "a1"]
        assert results[0] == {
            "tool_call_id": "d1",
            "content": "denied by policy",
            "is_error": True,
        }
        assert results[1]["content"] == "echo:y"
        decisions = h.events("decision")
        assert [(d["decision"], d["decided_by"]) for d in decisions] == [
            ("deny", "policy"),
            ("allow", "policy"),
        ]


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class TestBudgets:
    async def test_turn_budget_pauses_resumably(self, tmp_path: Path) -> None:
        script = [
            resp(None, [call("c1", "echo", text="a")]),
            resp(None, [call("c2", "echo", text="b")]),  # never reached
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=1),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.final_text is None
        assert result.turns == 1
        assert len(h.adapter.calls) == 1  # no second model call
        # State persisted for resume: events + agent status.
        assert h.store.get_agent(h.agent_id).status == "paused_budget"
        assert h.event_kinds() == [
            "message",  # goal
            "message",  # assistant turn 1
            "tool_call",
            "decision",
            "tool_result",
        ]

    async def test_token_budget_pauses_resumably(self, tmp_path: Path) -> None:
        script = [
            resp(
                None,
                [call("c1", "echo", text="a")],
                usage=Usage(input_tokens=80, output_tokens=30),
            ),
            resp(CLEAN_FINISH),  # never reached
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            budgets=Budgets(max_tokens=100),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 1
        assert result.usage == Usage(input_tokens=80, output_tokens=30)
        assert len(h.adapter.calls) == 1


# ---------------------------------------------------------------------------
# Diligence nudges
# ---------------------------------------------------------------------------


class TestNudges:
    async def test_nudge_fires_on_promised_future_work(
        self, tmp_path: Path
    ) -> None:
        script = [
            resp("I will run the tests next."),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert result.turns == 2
        nudges = h.events("nudge")
        assert len(nudges) == 1
        assert "promises future work" in nudges[0]["reason"]
        # The reminder reached the model as a user message on turn 2.
        last = h.adapter.calls[1].messages[-1]
        assert last.role is Role.USER
        assert "unfinished" in (last.content or "")

    async def test_nudges_respect_max_nudges(self, tmp_path: Path) -> None:
        unfinished = "I will keep going after this."
        script = [resp(unfinished), resp(unfinished), resp(unfinished)]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        # Two nudges (MAX_NUDGES), then the third answer is accepted as-is.
        assert result.status == "completed"
        assert result.final_text == unfinished
        assert result.turns == 3
        assert [n["nudge_number"] for n in h.events("nudge")] == [1, 2]

    async def test_open_ledger_items_trigger_nudge(
        self, tmp_path: Path
    ) -> None:
        script = [resp(CLEAN_FINISH), resp(CLEAN_FINISH), resp(CLEAN_FINISH)]
        h = make_harness(tmp_path, script)
        h.store.upsert_task_item(
            h.run_id, "t1", "write the report", "in_progress"
        )
        result = await h.loop.run(GOAL)

        # Item never closed: nudged twice, then accepted.
        assert result.status == "completed"
        nudges = h.events("nudge")
        assert len(nudges) == 2
        assert "task-ledger item" in nudges[0]["reason"]

    async def test_closed_ledger_items_do_not_nudge(
        self, tmp_path: Path
    ) -> None:
        script = [resp(CLEAN_FINISH)]
        h = make_harness(tmp_path, script)
        h.store.upsert_task_item(
            h.run_id, "t1", "write the report", "done", "report.md exists"
        )
        result = await h.loop.run(GOAL)
        assert result.status == "completed"
        assert h.events("nudge") == []
        assert result.turns == 1


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class TestEventLog:
    async def test_events_persisted_in_order_with_kinds(
        self, tmp_path: Path
    ) -> None:
        script = [
            resp("working", [call("c1", "echo", text="x")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        await h.loop.run(GOAL)

        events = h.store.load_events(h.agent_id)
        assert [e.kind for e in events] == [
            "message",  # goal (user)
            "message",  # assistant with tool call
            "tool_call",
            "decision",
            "tool_result",
            "message",  # final assistant
        ]
        assert [e.seq for e in events] == list(range(1, 7))
        assert events[0].payload["role"] == "user"
        assert events[0].payload["content"] == GOAL
        assert events[1].payload["role"] == "assistant"
        assert events[2].payload == {
            "id": "c1",
            "name": "echo",
            "arguments": {"text": "x"},
        }
        assert events[4].payload["content"] == "echo:x"
        assert events[5].payload["content"] == CLEAN_FINISH

    async def test_usage_recorded_per_model_call(self, tmp_path: Path) -> None:
        script = [
            resp(
                None,
                [call("c1", "echo", text="x")],
                usage=Usage(input_tokens=7, output_tokens=3),
            ),
            resp(CLEAN_FINISH, usage=Usage(input_tokens=11, output_tokens=4)),
        ]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        result = await h.loop.run(GOAL)

        records = h.store.list_usage(h.run_id)
        assert len(records) == 2
        assert records[0].usage == Usage(input_tokens=7, output_tokens=3)
        assert records[0].model == "fake-model"
        assert records[0].agent_id == h.agent_id
        totals = h.store.total_usage(h.run_id)
        # Real (monotonic) durations: non-negative, exact value not asserted
        # here — TestDurationRecording drives the clock deterministically.
        assert totals.pop("duration_ms") >= 0
        assert totals == {
            "input_tokens": 18,
            "output_tokens": 7,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        assert result.usage == Usage(input_tokens=18, output_tokens=7)


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    async def test_compaction_path_with_tiny_context(
        self, tmp_path: Path
    ) -> None:
        """A tiny max_context forces compaction mid-run; the evicted span is
        persisted as a 'compaction' event and the summary (with the verbatim
        goal) reaches the model."""

        def count_by_message(messages: list[Message]) -> int:
            return 100 * len(messages)

        summarizer_calls: list[int] = []

        async def summarize(messages: list[Message]) -> str:
            summarizer_calls.append(len(messages))
            return "TINY SUMMARY"

        # threshold = 0.8 * 500 = 400 tokens -> compaction once the full
        # assembly (system + transcript) exceeds 4 messages.
        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=count_by_message,
            max_context=500,
            summarize=summarize,
        )
        script = [
            resp("step 1", [call("c1", "echo", text="a")]),
            resp("step 2", [call("c2", "echo", text="b")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path, script, tools=[simple_tool("echo")], context=context
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert summarizer_calls == [3]  # compacted exactly once
        (compaction,) = h.events("compaction")
        # The boundary snapped past turn 1's tool result, so the evicted
        # span is the goal + first assistant message + its tool result.
        assert compaction["evicted_count"] == 3
        assert compaction["evicted"][0]["content"] == GOAL
        assert compaction["evicted"][1]["content"] == "step 1"
        assert compaction["evicted"][2]["tool_result"]["tool_call_id"] == "c1"
        # The summary text is persisted with the event (resume substitutes
        # it for the evicted span when replaying).
        assert "TINY SUMMARY" in compaction["summary"]
        assert GOAL in compaction["summary"]
        # The model's next call saw the summary with the verbatim goal.
        summary_messages = [
            m.content
            for m in h.adapter.calls[2].messages
            if m.content and COMPACTION_SUMMARY_PREFIX in m.content
        ]
        assert len(summary_messages) == 1
        assert GOAL in summary_messages[0]
        assert "TINY SUMMARY" in summary_messages[0]


    async def test_compaction_runs_to_fixpoint_within_one_turn(
        self, tmp_path: Path
    ) -> None:
        """Regression: one halving may not bring a heavy transcript under
        the threshold; the loop keeps compacting until it fits instead of
        calling the model with an over-window assembly."""

        def count_by_message(messages: list[Message]) -> int:
            return 100 * len(messages)

        summarizer_calls: list[int] = []

        async def summarize(messages: list[Message]) -> str:
            summarizer_calls.append(len(messages))
            return "S"

        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=count_by_message,
            max_context=500,  # threshold: 400 tokens = 4 messages
            summarize=summarize,
        )
        # Pre-seed a long plain-message history (as if replayed): 12
        # messages + goal + system = 14 messages, far over threshold, and a
        # single halving still leaves it over.
        for i in range(12):
            context.append(Message(role=Role.USER, content=f"note {i}"))
        script = [resp(CLEAN_FINISH)]
        h = make_harness(tmp_path, script, context=context)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        # Compacted more than once in the same turn ...
        compactions = h.events("compaction")
        assert len(compactions) >= 2
        assert len(summarizer_calls) == len(compactions)
        # ... and every compaction event carries its summary for resume.
        assert all("summary" in c and c["summary"] for c in compactions)
        # The model call finally happened on a genuinely shrunken assembly.
        assert len(h.adapter.calls) == 1
        assert len(h.adapter.calls[0].messages) < 14

    async def test_summarizer_adapter_error_ends_run_with_error(
        self, tmp_path: Path
    ) -> None:
        """Regression: an AdapterError raised during compaction
        summarization must finish the run with status 'error' (like any
        model-call failure), not escape AgentLoop.run as an exception."""
        from harness.adapters.base import AdapterError

        async def failing_summarize(messages: list[Message]) -> str:
            raise AdapterError("rate limited during summarization", retryable=True)

        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=lambda messages: 100 * len(messages),
            max_context=500,
            summarize=failing_summarize,
        )
        script = [
            resp("step 1", [call("c1", "echo", text="a")]),
            resp("step 2", [call("c2", "echo", text="b")]),
            resp(CLEAN_FINISH),  # never reached: compaction fails first
        ]
        h = make_harness(
            tmp_path, script, tools=[simple_tool("echo")], context=context
        )
        result = await h.loop.run(GOAL)  # must not raise

        assert result.status == "error"
        assert "rate limited" in (result.final_text or "")
        assert h.store.get_agent(h.agent_id).status == "error"


# ---------------------------------------------------------------------------
# Adapter errors
# ---------------------------------------------------------------------------


class TestAdapterError:
    async def test_exhausted_script_ends_run_with_error(
        self, tmp_path: Path
    ) -> None:
        h = make_harness(tmp_path, [])  # empty script -> AdapterError
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.turns == 0
        assert "exhausted" in (result.final_text or "")
        assert h.store.get_agent(h.agent_id).status == "error"
        # The goal was still persisted before the failure.
        assert h.event_kinds() == ["message"]

    async def test_error_after_successful_turns_keeps_usage(
        self, tmp_path: Path
    ) -> None:
        script = [
            resp(
                None,
                [call("c1", "echo", text="x")],
                usage=Usage(input_tokens=5, output_tokens=2),
            ),
        ]  # second complete() call exhausts the script
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.turns == 1
        assert result.usage == Usage(input_tokens=5, output_tokens=2)

    async def test_loop_adds_no_second_retry_layer(self, tmp_path: Path) -> None:
        """Regression: retries live in exactly one layer — the adapters
        (which wrap their provider calls in retry_with_backoff). The loop
        must call complete() exactly once per turn, even for a retryable
        failure, instead of multiplying the adapter's attempts."""
        from typing import Any

        from harness.adapters.base import AdapterError, ModelAdapter
        from harness.types import Capabilities, ToolSpec

        class AlwaysRetryableAdapter(ModelAdapter):
            def __init__(self) -> None:
                self.attempts = 0

            @property
            def capabilities(self) -> Capabilities:
                return Capabilities(
                    max_context=1_000_000,
                    supports_cache_control=False,
                )

            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                system: str | None = None,
                **params: Any,
            ) -> ModelResponse:
                # A real adapter would have exhausted its *internal*
                # retries before raising; the loop must not restart them.
                self.attempts += 1
                raise AdapterError("still throttled", retryable=True)

        h = make_harness(tmp_path, [])
        adapter = AlwaysRetryableAdapter()
        h.loop.adapter = adapter
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert "throttled" in (result.final_text or "")
        assert adapter.attempts == 1  # exactly one complete() per turn


class TestNudgePersistence:
    async def test_nudge_reminder_persisted_as_message_event(
        self, tmp_path: Path
    ) -> None:
        """Regression: the continue-reminder user message must be persisted
        as a 'message' event (not just the 'nudge' bookkeeping event), so a
        resumed transcript matches what the model actually saw."""
        script = [
            resp("I will run the tests next."),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        message_events = h.events("message")
        reminders = [
            payload
            for payload in message_events
            if payload["role"] == "user"
            and "unfinished" in (payload["content"] or "")
        ]
        assert len(reminders) == 1
        # The persisted reminder is exactly the message the model saw.
        live_reminder = next(
            m
            for m in h.adapter.calls[1].messages
            if m.role is Role.USER and "unfinished" in (m.content or "")
        )
        assert reminders[0]["content"] == live_reminder.content
        # The bookkeeping event is still recorded alongside.
        assert len(h.events("nudge")) == 1


class TestResultModel:
    def test_agent_result_rejects_unknown_status(self) -> None:
        with pytest.raises(Exception):
            AgentResult(
                status="exploded", final_text=None, usage=Usage(), turns=0
            )


# ---------------------------------------------------------------------------
# Per-call output cap + wall-clock wind-down
# ---------------------------------------------------------------------------


class TestMaxOutputTokens:
    async def test_cap_passed_through_as_max_tokens(
        self, tmp_path: Path
    ) -> None:
        h = make_harness(
            tmp_path,
            [resp(CLEAN_FINISH)],
            budgets=Budgets(max_output_tokens=1234),
        )
        await h.loop.run(GOAL)
        assert h.adapter.calls[0].params == {"max_tokens": 1234}

    async def test_no_cap_by_default(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path, [resp(CLEAN_FINISH)])
        await h.loop.run(GOAL)
        assert "max_tokens" not in h.adapter.calls[0].params


class TestWallClockWindDown:
    def _clock(self, values: list[float]) -> Callable[[], float]:
        """A clock returning ``values`` in order, then repeating the last."""
        it = iter(values)
        last = values[-1]

        def clock() -> float:
            nonlocal last
            last = next(it, last)
            return last

        return clock

    async def test_reminder_injected_when_deadline_near_and_nudge_suppressed(
        self, tmp_path: Path
    ) -> None:
        # Deadline anchored at 0, then the turn-1 check reads 700 → 200s of a
        # 900s budget left (below the clamped 300s threshold but above the 60s
        # hard-stop floor), so wind-down fires. The final message "looks
        # unfinished" (promises future work) but the nudge is suppressed
        # once wound down, so the run completes on turn 1 rather than looping.
        clock = self._clock([0.0, 700.0])
        h = make_harness(
            tmp_path,
            [resp("I will keep going after this.")],
            budgets=Budgets(wall_clock_seconds=900.0),
            clock=clock,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 1
        # The wind-down bookkeeping event landed once...
        wind_downs = h.events("wind_down")
        assert len(wind_downs) == 1
        assert wind_downs[0]["remaining_seconds"] == pytest.approx(200.0)
        assert wind_downs[0]["threshold"] == pytest.approx(300.0)
        # ...and the reminder reached the model on that turn's call.
        turn1_texts = [
            m.content for m in h.adapter.calls[0].messages if m.content
        ]
        assert any("approaching your hard time limit" in t for t in turn1_texts)
        # No diligence nudge was injected (it was suppressed by wind-down).
        assert h.events("nudge") == []

    async def test_no_wind_down_when_budget_unset(self, tmp_path: Path) -> None:
        # Even with a clock far past any deadline, no wall_clock_seconds means
        # no wind-down event and normal behaviour.
        clock = self._clock([0.0, 100_000.0])
        h = make_harness(
            tmp_path,
            [resp(CLEAN_FINISH)],
            clock=clock,
        )
        result = await h.loop.run(GOAL)
        assert result.status == "completed"
        assert h.events("wind_down") == []

    async def test_injected_pre_aged_deadline_wins_over_budgets(
        self, tmp_path: Path
    ) -> None:
        """An injected Deadline is the source of truth: a loop handed a
        pre-aged shared deadline is born wound-down (the reminder lands on
        its turn 1), even though budgets.wall_clock_seconds is unset —
        the late-spawned-subagent shape."""
        # Anchored at 0 elsewhere; every remaining() read sees 700 → 200s
        # of 900s left, inside the clamped 300s band, above the 60s floor.
        deadline = Deadline(900.0, self._clock([0.0, 700.0]))
        h = make_harness(
            tmp_path,
            [resp("I will keep going after this.")],
            deadline=deadline,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 1
        (wind_down,) = h.events("wind_down")
        assert wind_down["remaining_seconds"] == pytest.approx(200.0)
        assert wind_down["threshold"] == pytest.approx(300.0)
        turn1_texts = [
            m.content for m in h.adapter.calls[0].messages if m.content
        ]
        assert any("approaching your hard time limit" in t for t in turn1_texts)


class TestWindDownThreshold:
    @pytest.mark.parametrize(
        ("budget", "expected"),
        [
            (600.0, 300.0),  # 0.5 * budget beats the 300s floor
            (900.0, 300.0),  # raw fraction (180s) raised to the floor
            (2400.0, 480.0),  # raw fraction, inside the band
            (3600.0, 600.0),  # raw fraction (720s) clamped to the ceiling
            (12000.0, 600.0),  # ceiling: no 40-minute nudge-free tail
        ],
    )
    def test_clamp_table(self, budget: float, expected: float) -> None:
        assert wind_down_threshold(budget) == expected


class TestWallClockHardStop:
    def _clock(self, values: list[float]) -> Callable[[], float]:
        """A clock returning ``values`` in order, then repeating the last."""
        it = iter(values)
        last = values[-1]

        def clock() -> float:
            nonlocal last
            last = next(it, last)
            return last

        return clock

    async def test_below_floor_pauses_before_any_model_call(
        self, tmp_path: Path
    ) -> None:
        """Remaining below WALL_CLOCK_STOP_FLOOR (60s): the loop refuses to
        start a model call that cannot finish — zero adapter calls, a
        persisted wall_clock_stop event, and a resumable paused_budget."""
        # Anchor at 0; the turn-1 check reads 850 → 50s of 900s left.
        clock = self._clock([0.0, 850.0])
        h = make_harness(
            tmp_path,
            [resp(CLEAN_FINISH)],  # scripted but must never be consumed
            budgets=Budgets(wall_clock_seconds=900.0),
            clock=clock,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 0
        assert result.final_text is None
        assert h.adapter.calls == []  # zero model calls started
        (stop,) = h.events("wall_clock_stop")
        assert stop["remaining_seconds"] == pytest.approx(50.0)
        # It stopped, it did not wind down: no reminder for a call that
        # will never happen.
        assert h.events("wind_down") == []

    async def test_wind_down_fires_once_before_the_hard_stop(
        self, tmp_path: Path
    ) -> None:
        """A clock passing through the band: turn 1 winds down (200s left),
        turn 2 hard-stops (50s left) — exactly one wind_down, then the stop."""
        # Reads: anchor 0; turn-1 check 650 (remaining 250 ≤ 300 →
        # wind-down); call start/end 650/660; turn-2 check 850 (remaining
        # 50 < 60 → hard stop).
        clock = self._clock([0.0, 650.0, 650.0, 660.0, 850.0])
        h = make_harness(
            tmp_path,
            [resp("working", [call("c1", "echo", text="x")])],
            tools=[simple_tool("echo")],
            budgets=Budgets(wall_clock_seconds=900.0),
            clock=clock,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 1
        assert len(h.adapter.calls) == 1
        (wind_down,) = h.events("wind_down")
        assert wind_down["remaining_seconds"] == pytest.approx(250.0)
        (stop,) = h.events("wall_clock_stop")
        assert stop["remaining_seconds"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Truncated-turn continuation (output-token cap hit with no action)
# ---------------------------------------------------------------------------


class TestTruncationContinue:
    async def test_truncated_actionless_turn_is_continued_not_finished(
        self, tmp_path: Path
    ) -> None:
        # Turn 1 hit the output cap mid-thought (MAX_TOKENS) with no tool call
        # and an empty message; the loop must re-prompt to act rather than
        # accept the empty turn as the final answer.
        script = [
            resp(content=None, stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert result.turns == 2
        assert len(h.events("truncation_continue")) == 1
        # The reminder reached the model on the retry turn.
        turn2_texts = [m.content for m in h.adapter.calls[1].messages if m.content]
        assert any("cut off at the output-token limit" in t for t in turn2_texts)

    async def test_truncation_continues_are_bounded(self, tmp_path: Path) -> None:
        # A model that truncates every turn cannot loop forever: after
        # MAX_TRUNCATION_CONTINUES the truncated turn is accepted as final.
        from harness.loop import MAX_TRUNCATION_CONTINUES

        script = [
            resp(content="still thinking", stop_reason=StopReason.MAX_TOKENS)
            for _ in range(MAX_TRUNCATION_CONTINUES + 1)
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == MAX_TRUNCATION_CONTINUES + 1
        assert len(h.events("truncation_continue")) == MAX_TRUNCATION_CONTINUES

    async def test_truncated_turn_with_tool_call_dispatches_normally(
        self, tmp_path: Path
    ) -> None:
        # MAX_TOKENS but a complete tool call is present → the tool path runs;
        # the truncation guard only fires when no action was produced.
        script = [
            resp(
                content=None,
                calls=[call("c1", "echo", text="hi")],
                stop_reason=StopReason.MAX_TOKENS,
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("truncation_continue") == []
        assert len(h.events("tool_result")) == 1

    async def test_provider_truncated_tool_call_survives_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """Pinned end-to-end regression (trial make-mips-interpreter__KSxCFCR):
        a provider turn cut off at the output-token cap mid-tool-call used to
        kill the run with a non-retryable AdapterError from the argument
        parser. Through the real OpenAI-compat adapter, the malformed call is
        dropped, the loop's truncation-continue path fires, and the run
        completes."""
        from types import SimpleNamespace

        from harness.adapters.openai_compat import OpenAICompatAdapter

        def sdk_response(
            content: str | None, tool_calls: list | None, finish_reason: str
        ) -> SimpleNamespace:
            message = SimpleNamespace(content=content, tool_calls=tool_calls)
            choice = SimpleNamespace(
                message=message, finish_reason=finish_reason
            )
            return SimpleNamespace(
                choices=[choice],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        # A large inline write_file whose JSON arguments were cut mid-string
        # at the cap — the real trial's failure shape.
        truncated_call = SimpleNamespace(
            id="c1",
            function=SimpleNamespace(
                name="write_file",
                arguments='{"path": "interp.py", "content": "def main():',
            ),
        )

        class FakeCompletions:
            def __init__(self, results: list) -> None:
                self.results = list(results)
                self.calls: list[dict] = []

            async def create(self, **kwargs: object) -> SimpleNamespace:
                self.calls.append(kwargs)
                return self.results.pop(0)

        completions = FakeCompletions(
            [
                sdk_response(None, [truncated_call], "length"),
                sdk_response(CLEAN_FINISH, None, "stop"),
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAICompatAdapter("fake-model", client=client, stream=False)

        store = RunStore(tmp_path / "state.db")
        run_id = store.create_run(GOAL, "fake-model", "auto")
        agent_id = store.create_agent(run_id, GOAL)
        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=adapter.count_tokens,
            max_context=adapter.capabilities.max_context,
            summarize=stub_summarize,
        )

        async def ask(tool_name: str, arguments: dict, meta: ToolMeta) -> bool:
            return True

        loop = AgentLoop(
            adapter,
            ToolRegistry(),
            Policy(mode=PermissionMode.AUTO),
            store,
            run_id,
            agent_id,
            context,
            Budgets(),
            ask,
            model="fake-model",
        )
        result = await loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        kinds = [e.kind for e in store.load_events(agent_id)]
        assert kinds.count("truncation_continue") == 1
        # Turn 2's request replayed the transcript — the placeholder assistant
        # message translated cleanly (no empty-message rejection) — and the
        # reminder covering the cut-off-tool-call case reached the model.
        assert len(completions.calls) == 2
        turn2_contents = [
            str(m.get("content") or "") for m in completions.calls[1]["messages"]
        ]
        assert any(
            "truncated at the output-token limit" in c for c in turn2_contents
        )
        assert any("cut off mid-arguments" in c for c in turn2_contents)


# ---------------------------------------------------------------------------
# Per-turn duration recording (§10.2 A5)
# ---------------------------------------------------------------------------


def scripted_clock(values: list[float]) -> Callable[[], float]:
    """A clock returning ``values`` in order, then repeating the last."""
    it = iter(values)
    last = values[-1]

    def clock() -> float:
        nonlocal last
        last = next(it, last)
        return last

    return clock


class TestDurationRecording:
    async def test_duration_measured_around_each_model_call(
        self, tmp_path: Path
    ) -> None:
        """Each usage row records the wall-clock duration of exactly its
        model call, from the injected monotonic clock. Clock reads with no
        wall-clock budget/deadline: (call start, call end) per turn — no
        deadline means no per-turn remaining() read and no anchor read."""
        clock = scripted_clock([10.0, 10.5, 20.0, 20.25])
        script = [
            resp(None, [call("c1", "echo", text="a")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path, script, tools=[simple_tool("echo")], clock=clock
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        records = h.store.list_usage(h.run_id)
        assert [record.duration_ms for record in records] == [500, 250]
        assert h.store.total_usage(h.run_id)["duration_ms"] == 750


# ---------------------------------------------------------------------------
# Self-verification (§10.3 B1)
# ---------------------------------------------------------------------------


def declare(id: str, command: str, description: str = "proves the goal") -> ToolCall:
    """Build one declare_verification tool call."""
    return call(id, "declare_verification", command=command, description=description)


def verification_harness(
    tmp_path: Path, script: list[ModelResponse]
) -> Harness:
    """A harness with a real LocalSandbox wired to both the loop (as the
    verification runner) and its bash/declare_verification tools."""
    from harness.tools.builtin import declare_verification_tool

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    sandbox = LocalSandbox(workspace)
    return make_harness(
        tmp_path,
        script,
        tools=[bash_tool(sandbox), declare_verification_tool()],
        sandbox=sandbox,
    )


class TestVerification:
    async def test_declared_check_passes_and_run_completes(
        self, tmp_path: Path
    ) -> None:
        """Pass path: the declared command is re-run at completion time;
        exit 0 persists verification_passed (with output) and finishes."""
        script = [
            resp("declaring", [declare("v1", "echo verified-ok")]),
            resp(CLEAN_FINISH),
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        (declared,) = h.events("verification_declared")
        assert declared == {
            "command": "echo verified-ok",
            "description": "proves the goal",
        }
        (passed,) = h.events("verification_passed")
        assert passed["command"] == "echo verified-ok"
        assert passed["exit_code"] == 0
        assert "verified-ok" in passed["output"]
        assert h.events("verification_failed") == []

    async def test_failed_check_nudges_then_fixed_check_passes(
        self, tmp_path: Path
    ) -> None:
        """Fail-then-fix: a failing check bounces the final answer back with
        the failure output; once the agent fixes the workspace, the same
        check passes and the run completes."""
        script = [
            resp("declaring", [declare("v1", "test -f done.txt")]),
            resp(CLEAN_FINISH),  # bounced: done.txt does not exist yet
            resp("fixing", [call("c1", "bash", command="touch done.txt")]),
            resp(CLEAN_FINISH),  # now verification passes
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 4
        (failed,) = h.events("verification_failed")
        assert failed["command"] == "test -f done.txt"
        assert failed["exit_code"] != 0
        assert failed["nudge_number"] == 1
        assert "nudges_exhausted" not in failed
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        # The failure reminder reached the model as a user message on the
        # turn after the bounced answer, and was persisted as a message.
        turn3_last = h.adapter.calls[2].messages[-1]
        assert turn3_last.role is Role.USER
        assert "verification command failed" in (turn3_last.content or "")
        assert "test -f done.txt" in (turn3_last.content or "")
        assert any(
            payload["role"] == "user"
            and "verification command failed" in (payload["content"] or "")
            for payload in h.events("message")
        )

    async def test_permanently_failing_check_is_bounded_by_nudges(
        self, tmp_path: Path
    ) -> None:
        """Fail-exhausted: verification failures consume MAX_NUDGES; after
        that the run completes anyway, with the final failure persisted
        (nudges_exhausted) so it stays auditable."""
        from harness.diligence import MAX_NUDGES

        script = [
            resp("declaring", [declare("v1", "exit 1")]),
            *[resp(CLEAN_FINISH) for _ in range(MAX_NUDGES + 1)],
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        failures = h.events("verification_failed")
        assert len(failures) == MAX_NUDGES + 1
        assert [f.get("nudge_number") for f in failures[:-1]] == list(
            range(1, MAX_NUDGES + 1)
        )
        assert failures[-1]["nudges_exhausted"] is True
        assert h.events("verification_passed") == []
        assert h.store.get_agent(h.agent_id).status == "completed"

    async def test_redeclaring_replaces_the_previous_command(
        self, tmp_path: Path
    ) -> None:
        script = [
            resp("first", [declare("v1", "exit 1")]),
            resp("second", [declare("v2", "echo second-ok")]),
            resp(CLEAN_FINISH),
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        declared = h.events("verification_declared")
        assert [d["command"] for d in declared] == ["exit 1", "echo second-ok"]
        (passed,) = h.events("verification_passed")
        assert passed["command"] == "echo second-ok"
        assert h.events("verification_failed") == []

    async def test_heuristic_nudges_and_verification_share_one_budget(
        self, tmp_path: Path
    ) -> None:
        """A looks_unfinished nudge and verification failures draw from the
        same MAX_NUDGES pool, so the combination cannot loop forever."""
        script = [
            resp("declaring", [declare("v1", "exit 1")]),
            resp("I will keep going after this."),  # heuristic nudge (1)
            resp(CLEAN_FINISH),  # verification fail (nudge 2)
            resp(CLEAN_FINISH),  # budget spent: completes despite failure
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 4
        assert len(h.events("nudge")) == 1
        failures = h.events("verification_failed")
        assert len(failures) == 2
        assert failures[0]["nudge_number"] == 2
        assert failures[1]["nudges_exhausted"] is True

    async def test_invalid_declaration_does_not_arm_the_gate(
        self, tmp_path: Path
    ) -> None:
        """A declare_verification call that errors (missing command) never
        arms the gate: no verification events, heuristic path unchanged."""
        script = [
            resp("declaring", [call("v1", "declare_verification")]),
            resp(CLEAN_FINISH),
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        (tool_result,) = h.events("tool_result")
        assert tool_result["is_error"] is True
        assert h.events("verification_declared") == []
        assert h.events("verification_passed") == []
        assert h.events("verification_failed") == []

    async def test_no_declaration_leaves_heuristic_behavior_unchanged(
        self, tmp_path: Path
    ) -> None:
        """With a sandbox wired but nothing declared, completion is decided
        by looks_unfinished alone — no verification events at all."""
        script = [resp(CLEAN_FINISH)]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 1
        assert h.events("verification_passed") == []
        assert h.events("verification_failed") == []

    async def test_verification_execution_records_an_allow_decision(
        self, tmp_path: Path
    ) -> None:
        """Regression (§4.11): the harness-initiated verification execution
        is itself a logged permission decision — even an auto-allow —
        under a synthetic verification-exec tool-call id."""
        script = [
            resp("declaring", [declare("v1", "echo verified-ok")]),
            resp(CLEAN_FINISH),
        ]
        h = verification_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        ver_decisions = [
            d
            for d in h.events("decision")
            if d["tool_call_id"].startswith("verification-exec-")
        ]
        assert len(ver_decisions) == 1
        assert ver_decisions[0]["tool_name"] == "bash"
        assert ver_decisions[0]["arguments"] == {"command": "echo verified-ok"}
        assert ver_decisions[0]["decision"] == "allow"
        assert ver_decisions[0]["decided_by"] == "policy"

    async def test_policy_deny_glob_blocks_verification_execution(
        self, tmp_path: Path
    ) -> None:
        """Regression (§4.11): an explicit user deny glob on bash is the
        highest-precedence rule and must cover the B1 verification
        execution too — the model cannot run arbitrary shell by declaring
        it as a verification command. The command never reaches the
        sandbox, the deny is logged, and no nudge budget is burned (a
        policy deny is not something the model can fix)."""
        from harness.tools.builtin import declare_verification_tool

        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        sandbox = LocalSandbox(workspace)
        script = [
            resp("declaring", [declare("v1", "touch PWNED")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox), declare_verification_tool()],
            policy=Policy(mode=PermissionMode.GATED, deny=("bash",)),
            sandbox=sandbox,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 2
        assert not (workspace / "PWNED").exists()  # never executed
        (failed,) = h.events("verification_failed")
        assert failed["denied"] is True
        assert failed["exit_code"] is None
        assert "not executed" in failed["output"]
        assert h.events("verification_passed") == []
        assert h.events("nudge") == []
        # The deny outranks ASK: the callback was never consulted.
        assert h.ask_log == []
        ver_decisions = [
            d
            for d in h.events("decision")
            if d["tool_call_id"].startswith("verification-exec-")
        ]
        assert len(ver_decisions) == 1
        assert ver_decisions[0]["decision"] == "deny"
        assert ver_decisions[0]["decided_by"] == "policy"

    async def test_missing_bash_tool_gates_execution_in_gated_mode(
        self, tmp_path: Path
    ) -> None:
        """Regression: a registry with declare_verification but no bash
        tool has no policy-blessed shell meta, so the execution is treated
        as side-effecting — GATED mode routes it through ask instead of
        auto-allowing via the benign unknown-tool default."""
        from harness.tools.builtin import declare_verification_tool

        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        sandbox = LocalSandbox(workspace)
        script = [
            resp("declaring", [declare("v1", "touch GATED")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[declare_verification_tool()],
            policy=Policy(mode=PermissionMode.GATED),
            ask_answer=False,
            sandbox=sandbox,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert not (workspace / "GATED").exists()
        assert h.ask_log == [
            ("bash", {"command": "touch GATED"}, ToolMeta(side_effect=True))
        ]
        (failed,) = h.events("verification_failed")
        assert failed["denied"] is True
        ver_decisions = [
            d
            for d in h.events("decision")
            if d["tool_call_id"].startswith("verification-exec-")
        ]
        assert len(ver_decisions) == 1
        assert ver_decisions[0]["decision"] == "deny"
        assert ver_decisions[0]["decided_by"] == "user"

    async def test_wound_down_failure_is_labeled_wound_down_not_exhausted(
        self, tmp_path: Path
    ) -> None:
        """Regression: a verification failure accepted because the run
        wound down — with nudge budget remaining — must be stamped
        wound_down, not nudges_exhausted, or the B2/B4 failure
        classification mines a corrupted audit signal."""
        from harness.tools.builtin import declare_verification_tool

        # Deadline anchored at 0; turn-1 reads 0/0/1; the turn-2 check reads
        # 700 → 200s of a 900s budget left (inside the clamped 300s wind-down
        # band, above the 60s hard-stop floor), so wind-down fires before
        # turn 2.
        values = iter([0.0, 0.0, 0.0, 1.0, 700.0])
        last = 700.0

        def clock() -> float:
            nonlocal last
            last = next(values, last)
            return last

        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        sandbox = LocalSandbox(workspace)
        script = [
            resp("declaring", [declare("v1", "exit 1")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox), declare_verification_tool()],
            budgets=Budgets(wall_clock_seconds=900.0),
            clock=clock,
            sandbox=sandbox,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert len(h.events("wind_down")) == 1
        assert h.events("nudge") == []  # budget untouched
        (failed,) = h.events("verification_failed")
        assert failed["wound_down"] is True
        assert "nudges_exhausted" not in failed

    async def test_sandbox_exec_error_is_a_failed_verification_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        """A verification command that cannot even execute (sandbox raises)
        is treated as a failed check, never an exception out of run()."""
        from harness.sandbox.base import SandboxError
        from harness.tools.builtin import declare_verification_tool

        class ExplodingSandbox:
            async def exec(self, command: str, timeout: float = 120):
                raise SandboxError("no such sandbox backend")

        script = [
            resp("declaring", [declare("v1", "pytest -q")]),
            resp(CLEAN_FINISH),
            resp(CLEAN_FINISH),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path, script, tools=[declare_verification_tool()]
        )
        h.loop.sandbox = ExplodingSandbox()  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        failures = h.events("verification_failed")
        assert len(failures) == 3
        assert failures[0]["exit_code"] is None
        assert "failed to execute" in failures[0]["output"]


class ScriptedTimeoutSandbox:
    """Sandbox stub that records the ``timeout`` it received and returns a
    scripted :class:`~harness.sandbox.base.ExecResult` -- lets §Fix 3b's
    tests drive a verification timeout deterministically, with no real
    sleep (mirrors :class:`ExplodingSandbox` above)."""

    def __init__(self, result) -> None:
        self.result = result
        self.received_timeout: float | None = None

    async def exec(self, command: str, timeout: float = 120):
        self.received_timeout = timeout
        return self.result


class TestVerificationTimeoutCap:
    """Wind-down plan §Fix 3b: the verification re-run's own timeout is
    capped by the run's remaining wall-clock, and a timeout that only
    happened *because* of that cap is inconclusive -- not a failure the
    model could act on -- so it skips the nudge/reminder and is accepted."""

    async def test_capped_timeout_is_inconclusive_and_skips_the_nudge(
        self, tmp_path: Path
    ) -> None:
        from harness.diligence import VERIFICATION_TIMEOUT_SECONDS
        from harness.sandbox.base import ExecResult
        from harness.tools.builtin import declare_verification_tool

        script = [
            resp("declaring", [declare("v1", "pytest -q")]),
            resp(CLEAN_FINISH),
        ]
        # remaining=100 throughout (fixed clock); 100 - EXEC_RESERVE(60) = 40,
        # below VERIFICATION_TIMEOUT_SECONDS(300) -> capped to 40s. 100 is
        # also above the wind-down threshold for a 100s "budget" (50s), so
        # wind-down does not fire and cannot be confused with this path.
        deadline = Deadline(100.0, clock=lambda: 0.0)
        h = make_harness(
            tmp_path,
            script,
            tools=[declare_verification_tool()],
            deadline=deadline,
        )
        fake = ScriptedTimeoutSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        h.loop.sandbox = fake  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert h.events("wind_down") == []
        assert h.events("nudge") == []
        (failed,) = h.events("verification_failed")
        assert failed["timed_out"] is True
        assert failed["timeout_capped"] is True
        assert failed["inconclusive"] is True
        assert failed["timeout_seconds"] == pytest.approx(40.0)
        assert failed["timeout_seconds"] < VERIFICATION_TIMEOUT_SECONDS
        assert "nudge_number" not in failed
        assert "nudges_exhausted" not in failed
        # No VERIFICATION_FAILED_REMINDER reached the model as a message.
        assert not any(
            payload["role"] == "user"
            and "verification command failed" in (payload["content"] or "")
            for payload in h.events("message")
        )

    async def test_uncapped_timeout_still_nudges_as_today(
        self, tmp_path: Path
    ) -> None:
        """Contrast: with no deadline (so no cap applies), a verification
        timeout keeps today's fail-and-nudge semantics exactly."""
        from harness.diligence import MAX_NUDGES
        from harness.sandbox.base import ExecResult
        from harness.tools.builtin import declare_verification_tool

        script = [
            resp("declaring", [declare("v1", "pytest -q")]),
            *[resp(CLEAN_FINISH) for _ in range(MAX_NUDGES + 1)],
        ]
        h = make_harness(tmp_path, script, tools=[declare_verification_tool()])
        fake = ScriptedTimeoutSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        h.loop.sandbox = fake  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        failures = h.events("verification_failed")
        assert len(failures) == MAX_NUDGES + 1
        assert all(f.get("timed_out") for f in failures)
        assert all("timeout_capped" not in f for f in failures)
        assert all("inconclusive" not in f for f in failures)
        assert [f.get("nudge_number") for f in failures[:-1]] == list(
            range(1, MAX_NUDGES + 1)
        )
        assert failures[-1]["nudges_exhausted"] is True
        reminders = [
            payload
            for payload in h.events("message")
            if payload["role"] == "user"
            and "verification command failed" in (payload["content"] or "")
        ]
        assert len(reminders) == MAX_NUDGES

    async def test_capped_but_passing_verification_still_counts_as_passed(
        self, tmp_path: Path
    ) -> None:
        from harness.sandbox.base import ExecResult
        from harness.tools.builtin import declare_verification_tool

        script = [
            resp("declaring", [declare("v1", "echo ok")]),
            resp(CLEAN_FINISH),
        ]
        deadline = Deadline(100.0, clock=lambda: 0.0)  # forces a 40s cap
        h = make_harness(
            tmp_path,
            script,
            tools=[declare_verification_tool()],
            deadline=deadline,
        )
        fake = ScriptedTimeoutSandbox(ExecResult(exit_code=0, stdout="ok", stderr=""))
        h.loop.sandbox = fake  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert fake.received_timeout == pytest.approx(40.0)
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        assert passed["timeout_capped"] is True
        assert passed["timeout_seconds"] == pytest.approx(40.0)
        assert h.events("verification_failed") == []
        assert h.events("nudge") == []


class TestVerificationOrchestratorWiring:
    async def test_orchestrated_run_executes_declared_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end through the Orchestrator: declare_verification is in
        the default coding toolset and the lead loop gets the run's sandbox,
        so the declared command really executes before completion."""
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )
        with pytest.warns(UserWarning, match="no Docker daemon"):
            with RunStore(tmp_path / "orch.db") as store:
                orchestrator = Orchestrator(
                    HarnessConfig(home=tmp_path / "home"), store
                )
                adapter = FakeAdapter(
                    [
                        resp("declaring", [declare("v1", "echo wired-ok")]),
                        resp(CLEAN_FINISH),
                    ]
                )
                run_id, result = await orchestrator.run_task(
                    GOAL, "fake-model", adapter_override=adapter
                )
                assert result.status == "completed"
                # The declaration was offered as a tool and the check ran.
                names = [spec.name for spec in adapter.calls[0].tools]
                assert "declare_verification" in names
                lead = store.list_agents(run_id)[0]
                events = store.load_events(lead.id)
                passed = [
                    e.payload
                    for e in events
                    if e.kind == "verification_passed"
                ]
                assert len(passed) == 1
                assert "wired-ok" in passed[0]["output"]


class TestVerificationResume:
    async def test_resume_replays_verification_events_without_breaking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resume-safety: verification_* event kinds in the persisted log
        are ignored by the orchestrator's transcript replay (not crashed
        on), and the resumed run completes normally."""
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )
        with pytest.warns(UserWarning, match="no Docker daemon"):
            with RunStore(tmp_path / "orch.db") as store:
                orchestrator = Orchestrator(
                    HarnessConfig(home=tmp_path / "home"), store
                )
                first = FakeAdapter(
                    [resp("declaring", [declare("v1", "echo resumed-ok")])]
                )
                run_id, paused = await orchestrator.run_task(
                    GOAL,
                    "fake-model",
                    adapter_override=first,
                    budgets=Budgets(max_turns=1),
                )
                assert paused.status == "paused_budget"
                lead = store.list_agents(run_id)[0]
                kinds = [e.kind for e in store.load_events(lead.id)]
                assert "verification_declared" in kinds

                second = FakeAdapter([resp(CLEAN_FINISH)])
                result = await orchestrator.resume_task(
                    run_id, adapter_override=second
                )
                assert result.status == "completed"
                assert store.get_run(run_id).status == "completed"
                # The replayed transcript still carries the declaration's
                # tool call/result pair (regular events), goal first.
                messages = second.calls[0].messages
                assert messages[0].content == GOAL
                assert any(
                    m.tool_result is not None
                    and m.tool_result.tool_call_id == "v1"
                    for m in messages
                )

    async def test_resume_rearms_the_last_declared_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: resume must re-arm the B1 gate from the last
        persisted verification_declared event — the replayed transcript
        promises the model its check "will be re-run before your answer is
        accepted", and a resume that silently disarms it breaks that
        promise. The *last* declaration wins, mirroring the live loop."""
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )
        with pytest.warns(UserWarning, match="no Docker daemon"):
            with RunStore(tmp_path / "orch.db") as store:
                orchestrator = Orchestrator(
                    HarnessConfig(home=tmp_path / "home"), store
                )
                first = FakeAdapter(
                    [
                        resp("declaring", [declare("v1", "exit 1")]),
                        resp(
                            "redeclaring",
                            [declare("v2", "echo resumed-ok")],
                        ),
                    ]
                )
                run_id, paused = await orchestrator.run_task(
                    GOAL,
                    "fake-model",
                    adapter_override=first,
                    budgets=Budgets(max_turns=2),
                )
                assert paused.status == "paused_budget"

                second = FakeAdapter([resp(CLEAN_FINISH)])
                result = await orchestrator.resume_task(
                    run_id, adapter_override=second
                )
                assert result.status == "completed"
                lead = store.list_agents(run_id)[0]
                events = store.load_events(lead.id)
                passed = [
                    e.payload
                    for e in events
                    if e.kind == "verification_passed"
                ]
                assert len(passed) == 1
                assert passed[0]["command"] == "echo resumed-ok"
                assert "resumed-ok" in passed[0]["output"]
                assert [
                    e for e in events if e.kind == "verification_failed"
                ] == []
