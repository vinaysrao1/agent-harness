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
from harness.deadline import WALL_CLOCK_STOP_FLOOR, Deadline
from harness.diligence import looks_unfinished
from harness.loop import (
    DROPPED_ARGUMENTS_PREFIX_CHARS,
    LANDING_TURN_NOTICE,
    AgentLoop,
    AgentResult,
    Budgets,
    wind_down_threshold,
)
from harness.permissions import Policy, ToolMeta
from harness.persistence import RunStore
from harness.sandbox.local import LocalSandbox
from harness.tools.builtin import LANDING_REFUSAL, bash_tool
from harness.tools.registry import Tool, ToolRegistry
from harness.types import (
    DroppedToolCall,
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
    provider_stop_reason: str | None = None,
    dropped: list[DroppedToolCall] | None = None,
    incomplete_reason: str | None = None,
) -> ModelResponse:
    """Build one scripted assistant response.

    ``stop_reason`` defaults to TOOL_USE when there are tool calls else
    END_TURN; pass it explicitly to script a truncated turn (MAX_TOKENS).
    ``provider_stop_reason`` scripts the provider's verbatim stop string
    (what the loop records on its ``model_turn`` event); ``None`` — the
    default — models a provider that sent none.

    ``incomplete``/``incomplete_reason`` are *derived* here using the same
    precedence the real adapters apply (dropped_calls > max_tokens), so a
    scripted response cannot claim a combination no adapter would produce.
    Pass ``incomplete_reason`` explicitly for the ``no_finish_reason`` case,
    which depends on the translated message being empty.
    """
    tool_calls = calls or []
    if stop_reason is None:
        stop_reason = (
            StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
        )
    if incomplete_reason is None:
        if dropped:
            incomplete_reason = "dropped_calls"
        elif stop_reason is StopReason.MAX_TOKENS:
            incomplete_reason = "max_tokens"
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT, content=content, tool_calls=tool_calls
        ),
        usage=usage or Usage(),
        stop_reason=stop_reason,
        provider_stop_reason=provider_stop_reason,
        incomplete=incomplete_reason is not None,
        incomplete_reason=incomplete_reason,
        dropped_tool_calls=dropped or [],
    )


def drop(
    tool_name: str = "write_file", arguments: str = '{"path": "a.py", "c'
) -> DroppedToolCall:
    """One scripted dropped tool call, as an adapter would report it."""
    return DroppedToolCall(
        tool_name=tool_name,
        raw_arguments_prefix=arguments,
        raw_arguments_len=len(arguments),
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
            "model_turn",  # turn 1 provenance (§C2)
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


class TestDeadlineDerivedMaxTurns:
    """A turn count must not end a run that still has wall clock to spend.

    Observed: a trial hit its 80-turn ceiling at 1150s of a 1800s budget and
    paused with 36% of its time unused. When a deadline exists and the caller
    did not choose the ceiling, the clock is the rail (:data:`MIN_SECONDS_PER
    _TURN`); the turn count only survives as a floor.
    """

    @staticmethod
    def _busy_script(count: int) -> list[ModelResponse]:
        """``count`` turns that each do work, so nothing else stops the run."""
        return [
            resp("working", [call(f"c{i}", "echo", text="x")])
            for i in range(count)
        ]

    @staticmethod
    def _stepping_clock(step: float) -> "Callable[[], float]":
        """A clock that advances ``step`` seconds on every read."""
        now = 0.0

        def clock() -> float:
            nonlocal now
            now += step
            return now

        return clock

    async def test_deadline_lifts_a_low_turn_ceiling(
        self, tmp_path: Path
    ) -> None:
        """max_turns=3 against a 3600s deadline: the derived rail is
        ceil(3600/5)=720, so the run works past turn 3 and ends on the clock
        instead."""
        clock = self._stepping_clock(300.0)
        deadline = Deadline(3600.0, clock)
        h = make_harness(
            tmp_path,
            self._busy_script(10),
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=3),
            clock=clock,
            deadline=deadline,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns > 3
        # Ended because the wall clock ran out, not because of a turn count.
        assert len(h.events("wall_clock_stop")) == 1

    async def test_without_a_deadline_the_turn_ceiling_binds(
        self, tmp_path: Path
    ) -> None:
        """The default for every non-deadline caller is unchanged."""
        h = make_harness(
            tmp_path,
            self._busy_script(10),
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=3),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 3
        assert h.events("wall_clock_stop") == []

    async def test_an_explicit_ceiling_is_never_lifted(
        self, tmp_path: Path
    ) -> None:
        """A caller who chose the ceiling gets exactly it, deadline or not."""
        clock = self._stepping_clock(300.0)
        deadline = Deadline(3600.0, clock)
        h = make_harness(
            tmp_path,
            self._busy_script(10),
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=3, max_turns_is_hard=True),
            clock=clock,
            deadline=deadline,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 3
        assert h.events("wall_clock_stop") == []

    async def test_the_derived_rail_is_a_floor_not_a_ceiling(
        self, tmp_path: Path
    ) -> None:
        """A caller's *larger* turn budget survives the derivation: a 60s
        deadline derives 12 turns, but max_turns=13 still buys 13."""
        h = make_harness(
            tmp_path,
            self._busy_script(20),
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=13),
            clock=lambda: 0.0,
            deadline=Deadline(60.0, lambda: 0.0),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 13
        assert h.events("wall_clock_stop") == []

    async def test_a_budgetless_deadline_derives_nothing(
        self, tmp_path: Path
    ) -> None:
        """``Deadline(None)`` is the no-deadline object: it must not silently
        become an unbounded turn budget."""
        h = make_harness(
            tmp_path,
            self._busy_script(10),
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=3),
            deadline=Deadline(None),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 3

    async def test_the_token_ceiling_still_binds_independently(
        self, tmp_path: Path
    ) -> None:
        """Lifting the turn rail must not lift the token rail with it."""
        script = [
            resp(
                "working",
                [call("c1", "echo", text="x")],
                usage=Usage(input_tokens=80, output_tokens=30),
            ),
            resp(CLEAN_FINISH),  # never reached
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=3, max_tokens=100),
            clock=lambda: 0.0,
            deadline=Deadline(3600.0, lambda: 0.0),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.turns == 1


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
            "model_turn",  # turn 1 provenance (§C2)
            "message",  # assistant with tool call
            "tool_call",
            "decision",
            "tool_result",
            "model_turn",  # turn 2 provenance (§C2)
            "message",  # final assistant
        ]
        assert [e.seq for e in events] == list(range(1, 9))
        assert events[0].payload["role"] == "user"
        assert events[0].payload["content"] == GOAL
        assert events[2].payload["role"] == "assistant"
        assert events[3].payload == {
            "id": "c1",
            "name": "echo",
            "arguments": {"text": "x"},
        }
        assert events[5].payload["content"] == "echo:x"
        assert events[7].payload["content"] == CLEAN_FINISH

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
            "reasoning_tokens": 0,
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

    async def test_low_utilization_run_never_prunes_a_tool_result(
        self, tmp_path: Path
    ) -> None:
        """G1 end-to-end fence: 25 tool turns with 2 KB results against a
        1M-token window is ~1% utilization, so nothing the model already saw
        may be replaced by a stub on any later call. Pre-G1 every result
        older than three assistant turns was stubbed regardless."""
        payload = "x" * 2048
        script = [
            resp(f"step {i}", [call(f"c{i}", "echo", text=payload)])
            for i in range(25)
        ] + [resp(CLEAN_FINISH)]
        adapter = FakeAdapter(script)
        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=adapter.count_tokens,
            max_context=1_000_000,
            summarize=stub_summarize,
        )
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            context=context,
            budgets=Budgets(max_turns=30),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert len(h.adapter.calls) == 26
        for index, recorded in enumerate(h.adapter.calls):
            for message in recorded.messages:
                content = message.content or (
                    message.tool_result.content
                    if message.tool_result is not None
                    else ""
                )
                assert "[pruned:" not in content, f"call {index} saw a stub"
        # The very first result is still verbatim in the final call.
        final = [
            m.tool_result.content
            for m in h.adapter.calls[-1].messages
            if m.tool_result is not None
        ]
        assert final[0] == f"echo:{payload}"
        assert not h.events("compaction")

    async def test_pruning_engages_and_hands_off_to_compaction_under_pressure(
        self, tmp_path: Path
    ) -> None:
        """The other end of the ladder: the same script against an
        8,000-token window prunes *and* eventually compacts."""
        payload = "x" * 2048
        script = [
            resp(f"step {i}", [call(f"c{i}", "echo", text=payload)])
            for i in range(25)
        ] + [resp(CLEAN_FINISH)]
        adapter = FakeAdapter(script)
        context = ContextManager(
            base_system_prompt="You are a test agent.",
            count_tokens=adapter.count_tokens,
            max_context=8_000,
            summarize=stub_summarize,
        )
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            context=context,
            budgets=Budgets(max_turns=30),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        stubs = [
            m.tool_result.content
            for recorded in h.adapter.calls
            for m in recorded.messages
            if m.tool_result is not None
            and m.tool_result.content.startswith("[pruned:")
        ]
        assert stubs
        assert h.events("compaction")

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
        # The goal was still persisted before the failure, and the failure
        # itself is on the record (§C2) — not only in the returned result.
        assert h.event_kinds() == ["message", "run_error"]

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


class TestProviderFaultKind:
    """AgentResult.error_kind carries the adapter's provider-fault
    classification out of the loop, and the run_error event records it.

    This is what lets an embedding benchmark harness tell an infrastructure
    outage (an exhausted key killed 7 of 22 trials in one real run) apart
    from an agent capability failure — without it, both look like a clean
    completion with reward 0.
    """

    @staticmethod
    def _failing_loop(h, fault: str | None, message: str = "provider died"):
        """Point ``h``'s loop at an adapter that always raises with ``fault``."""
        from typing import Any

        from harness.adapters.base import AdapterError, ModelAdapter
        from harness.types import Capabilities, ToolSpec

        class FaultingAdapter(ModelAdapter):
            @property
            def capabilities(self) -> Capabilities:
                return Capabilities(
                    max_context=1_000_000, supports_cache_control=False
                )

            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                system: str | None = None,
                **params: Any,
            ) -> ModelResponse:
                raise AdapterError(message, fault=fault)

        h.loop.adapter = FaultingAdapter()
        return h

    @pytest.mark.parametrize(
        "fault",
        ["quota", "auth", "rate_limit", "server", "transport"],
    )
    async def test_fault_becomes_error_kind(
        self, tmp_path: Path, fault: str
    ) -> None:
        h = self._failing_loop(make_harness(tmp_path, []), fault)
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.error_kind == fault

    async def test_run_error_event_carries_error_kind(
        self, tmp_path: Path
    ) -> None:
        """The classification must survive on disk, not only in the returned
        object: post-hoc triage reads the transcript."""
        h = self._failing_loop(
            make_harness(tmp_path, []),
            "quota",
            "openai-compatible API error (HTTP 403): Key limit exceeded",
        )
        result = await h.loop.run(GOAL)

        assert result.error_kind == "quota"
        events = h.events("run_error")
        assert len(events) == 1
        assert events[0]["error_kind"] == "quota"
        assert "Key limit exceeded" in events[0]["message"]
        assert events[0]["turns"] == 0

    async def test_unclassified_adapter_error_has_no_error_kind(
        self, tmp_path: Path
    ) -> None:
        """An adapter that could not classify the failure leaves error_kind
        None, so the run stays an ordinary scored failure. This is the
        default and covers every pre-taxonomy AdapterError construction
        site — including FakeAdapter's script-exhausted error."""
        h = make_harness(tmp_path, [])  # empty script -> plain AdapterError
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.error_kind is None
        assert h.events("run_error")[0]["error_kind"] is None

    async def test_completed_run_has_no_error_kind(self, tmp_path: Path) -> None:
        h = make_harness(tmp_path, [resp(CLEAN_FINISH)])
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.error_kind is None

    async def test_budget_pause_has_no_error_kind(self, tmp_path: Path) -> None:
        h = make_harness(
            tmp_path,
            [resp(None, [call("c1", "echo", text="x")])],
            tools=[simple_tool("echo")],
            budgets=Budgets(max_turns=1),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        assert result.error_kind is None

    async def test_fault_after_real_work_keeps_usage_and_turns(
        self, tmp_path: Path
    ) -> None:
        """The case that matters most: a trial that did real work before the
        provider died still reports what it accrued, alongside the fault."""
        h = make_harness(
            tmp_path,
            [
                resp(
                    None,
                    [call("c1", "echo", text="x")],
                    usage=Usage(input_tokens=5, output_tokens=2),
                )
            ],
            tools=[simple_tool("echo")],
        )
        # First call succeeds from the script; the adapter faults afterwards.
        original = h.loop.adapter

        from typing import Any

        from harness.adapters.base import AdapterError, ModelAdapter
        from harness.types import Capabilities, ToolSpec

        class FaultAfterScript(ModelAdapter):
            @property
            def capabilities(self):
                return original.capabilities

            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                system: str | None = None,
                **params: Any,
            ) -> ModelResponse:
                try:
                    return await original.complete(
                        messages, tools, system, **params
                    )
                except AdapterError:
                    raise AdapterError(
                        "Key limit exceeded (total limit)", fault="quota"
                    ) from None

        h.loop.adapter = FaultAfterScript()
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.error_kind == "quota"
        assert result.turns == 1
        assert result.usage == Usage(input_tokens=5, output_tokens=2)


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


class _MovableClock:
    """A monotonic clock the test moves by hand — no real sleeps."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _TimedFakeAdapter(FakeAdapter):
    """A :class:`FakeAdapter` whose calls cost wall-clock on a fake clock,
    so the loop's model-call observations are exercised deterministically."""

    def __init__(
        self,
        script: list[ModelResponse],
        clock: _MovableClock,
        seconds_per_call: float,
    ) -> None:
        super().__init__(script)
        self._clock = clock
        self._seconds_per_call = seconds_per_call

    async def complete(  # type: ignore[no-untyped-def]
        self, messages, tools, system=None, **params
    ):
        self._clock.advance(self._seconds_per_call)
        return await super().complete(messages, tools, system, **params)


class _BurnsItsWindowSandbox:
    """A sandbox stub whose ``exec`` consumes exactly its ``timeout``.

    The worst case for the exec cap: a command that runs until it is
    killed, so what is left when it returns is exactly what the cap
    reserved.
    """

    def __init__(self, clock: _MovableClock) -> None:
        self._clock = clock
        self.received_timeout: float | None = None

    async def exec(  # type: ignore[no-untyped-def]
        self, command: str, timeout: float = 120
    ):
        from harness.sandbox.base import ExecResult

        self.received_timeout = timeout
        self._clock.advance(timeout)
        return ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)


class TestLandingReserveLeavesATurn:
    """Change 0: the exec reserve funds a landing turn, not just a clean stop.

    Before the split, ``EXEC_RESERVE_SECONDS == WALL_CLOCK_STOP_FLOOR ==
    60``, so an exec capped at ``remaining - 60`` returned with *exactly*
    the hard-stop floor left: step 1a fired, the run paused, and the agent
    that had just been told to "finalize your answer now" never got the
    turn to do it.
    """

    async def test_capped_exec_is_followed_by_another_model_turn(
        self, tmp_path: Path
    ) -> None:
        # The write-compressor shape: at remaining=336 of a 900s budget the
        # model asks for a 300s command that runs until killed.
        clock = _MovableClock()
        deadline = Deadline(900.0, clock)
        clock.now = 564.0  # remaining 336

        sandbox = _BurnsItsWindowSandbox(clock)
        script = [
            resp(
                "compressing",
                [call("c1", "bash", command="./compress", timeout=300)],
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox, deadline=deadline)],  # type: ignore[arg-type]
            clock=clock,
            deadline=deadline,
        )
        result = await h.loop.run(GOAL)

        # The cap: reserve = 60 floor + 15 allowance (a fake clock makes the
        # scripted model calls cost 0s, so the allowance clamps to its
        # minimum) = 75s, and Change 1's band softener holds back a further
        # 0.25 x 336 = 84s because the exec starts above the 300s wind-down
        # threshold -> 336 - 84 = 252s, not the old 336 - 60 = 276s.
        assert sandbox.received_timeout == pytest.approx(252.0)
        # ...which is the whole point: the exec returns above the floor.
        assert deadline.remaining() == pytest.approx(84.0)
        assert deadline.remaining() > WALL_CLOCK_STOP_FLOOR
        # So the run gets its landing turn instead of a wall_clock_stop.
        assert h.events("wall_clock_stop") == []
        assert len(h.events("model_turn")) == 2
        assert len(h.adapter.calls) == 2
        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        # And the landing turn carries the wind-down reminder (it is now
        # reachable, where before step 1a returned ahead of it).
        assert len(h.events("wind_down")) == 1

    async def test_loop_feeds_the_deadlines_model_call_window(
        self, tmp_path: Path
    ) -> None:
        """The allowance is adaptive because the loop reports every call."""
        clock = _MovableClock()
        deadline = Deadline(1800.0, clock)
        script = [
            resp("thinking", [call(f"c{i}", "echo", text="x")]) for i in range(3)
        ] + [resp(CLEAN_FINISH)]
        h = make_harness(
            tmp_path,
            script,
            tools=[simple_tool("echo")],
            clock=clock,
            deadline=deadline,
        )
        timed = _TimedFakeAdapter(script, clock, 20.0)
        h.loop.adapter = timed  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.turns == 4
        # Four 20s calls observed: p75 = 20 (inside the bounds), so the
        # reserve tracks this provider rather than the 30s default.
        assert deadline.recent_call_median() == pytest.approx(20.0)
        assert deadline.landing_allowance() == pytest.approx(20.0)
        assert deadline.landing_reserve() == pytest.approx(80.0)

    async def test_no_deadline_records_nothing_and_caps_nothing(
        self, tmp_path: Path
    ) -> None:
        """Without a deadline the loop must not touch a deadline at all."""
        clock = _MovableClock()
        sandbox = _BurnsItsWindowSandbox(clock)
        script = [
            resp("running", [call("c1", "bash", command="./x", timeout=300)]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox)],  # type: ignore[arg-type]
            clock=clock,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert sandbox.received_timeout == 300.0  # pure passthrough
        assert h.events("wall_clock_stop") == []


class _RecordingBurnSandbox(_BurnsItsWindowSandbox):
    """:class:`_BurnsItsWindowSandbox` that logs what it was handed and
    what the run had left when it was handed it — the two numbers the exec
    cap is judged on."""

    def __init__(self, clock: _MovableClock, deadline: Deadline) -> None:
        super().__init__(clock)
        self._deadline = deadline
        #: ``(remaining_before, timeout)`` for every exec, in order.
        self.log: list[tuple[float, float]] = []

    async def exec(  # type: ignore[no-untyped-def]
        self, command: str, timeout: float = 120
    ):
        remaining = self._deadline.remaining()
        assert remaining is not None
        self.log.append((remaining, timeout))
        return await super().exec(command, timeout)


class TestExecCapEndsTheBandSkip:
    """Change 1 at the loop level, on the four trials that proved the bug.

    All four emitted ``wall_clock_stop`` with **no ``wind_down`` event
    ever**: a single exec carried the run from above the wind-down
    threshold to below the hard-stop floor, so step 1b was never reached.
    The cap now guarantees the agent regains control inside the band.

    The model is scripted to keep asking for the same long command until
    the clock is gone — the pathological shape, not a cooperative one.
    """

    #: ``(id, budget, remaining_at_first_exec, requested)`` — see plan §0.2.
    SHAPES = [
        ("mcmc-sampling-stan", 1800.0, 1706.9, 1800.0),
        ("caffe-cifar-10", 1200.0, 619.7, 600.0),
        ("extract-moves-from-video", 1800.0, 926.0, 600.0),
        ("write-compressor", 900.0, 335.5, 300.0),
    ]

    def _run_shape(
        self,
        tmp_path: Path,
        budget: float,
        remaining: float,
        requested: float,
        seconds_per_call: float = 0.0,
    ):
        """Wire one shape. ``seconds_per_call`` makes model calls cost
        wall-clock, which is what lets the run actually reach the end of its
        budget — with free calls the clock only moves inside ``exec``."""
        clock = _MovableClock()
        deadline = Deadline(budget, clock)
        clock.now = budget - remaining
        sandbox = _RecordingBurnSandbox(clock, deadline)
        script = [
            resp(
                "still working",
                [call(f"c{i}", "bash", command="./long", timeout=requested)],
            )
            for i in range(40)
        ] + [resp(CLEAN_FINISH)]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox, deadline=deadline)],  # type: ignore[arg-type]
            clock=clock,
            deadline=deadline,
            budgets=Budgets(max_turns=100),
        )
        if seconds_per_call:
            h.loop.adapter = _TimedFakeAdapter(  # type: ignore[assignment]
                script, clock, seconds_per_call
            )
        return h, sandbox, deadline

    @pytest.mark.parametrize(
        "budget,remaining,requested",
        [shape[1:] for shape in SHAPES],
        ids=[shape[0] for shape in SHAPES],
    )
    async def test_wind_down_always_precedes_the_landing_turn(
        self, tmp_path: Path, budget: float, remaining: float, requested: float
    ) -> None:
        # 5s model calls, so the run really does spend its budget rather
        # than looping for free once the exec windows reach zero.
        h, sandbox, _deadline = self._run_shape(
            tmp_path, budget, remaining, requested, seconds_per_call=5.0
        )
        result = await h.loop.run(GOAL)

        kinds = h.event_kinds()
        assert "wind_down" in kinds
        # Change 1c: every one of these shapes now ends on the landing turn
        # instead of on the hard stop. The wind-down reminder still has to
        # come first — it is what pushes the answer to disk while there is
        # still a shell — and the landing turn is where the run ends: the
        # agent gets its last model call and the loop finishes on the text
        # it produced, so the bare `wall_clock_stop` pause (no final text
        # at all) never happens.
        assert "landing_turn" in kinds
        assert kinds.index("wind_down") < kinds.index("landing_turn")
        assert kinds.index("landing_finish") > kinds.index("landing_turn")
        assert "wall_clock_stop" not in kinds
        assert result.status == "completed"
        # And it really is the last turn: nothing runs after it.
        assert kinds.index("landing_finish") == len(kinds) - 1

    @pytest.mark.parametrize(
        "budget,remaining,requested",
        [shape[1:] for shape in SHAPES],
        ids=[shape[0] for shape in SHAPES],
    )
    async def test_a_model_turn_follows_every_capped_exec(
        self, tmp_path: Path, budget: float, remaining: float, requested: float
    ) -> None:
        h, sandbox, _deadline = self._run_shape(
            tmp_path, budget, remaining, requested, seconds_per_call=5.0
        )
        await h.loop.run(GOAL)

        # The bug was one exec owning the rest of the run. Now the agent
        # comes back and gets at least one further turn — which is what the
        # four failing trials never got.
        assert len(sandbox.log) >= 2
        assert len(h.events("model_turn")) >= 2
        # Change 1a strengthened this: it now holds for *every* exec, not
        # every exec but the last. The old version had to exempt the last
        # one because the exec floor could be paid out of the stop floor —
        # that escape is what closed. `>=` rather than `>` because the
        # clamp lands exactly on the floor when it binds.
        for before, timeout in sandbox.log:
            assert before - timeout >= WALL_CLOCK_STOP_FLOOR

    async def test_mcmc_regains_control_mid_install(self, tmp_path: Path) -> None:
        # The headline number: the runaway `while pgrep ...; do sleep 20;
        # done` owned 1647s of 1800s and the agent never came back. Capped
        # at half the budget, it comes back at ~807s remaining — with the
        # install still running and most of the run left to use.
        h, sandbox, _deadline = self._run_shape(tmp_path, 1800.0, 1706.9, 1800.0)
        await h.loop.run(GOAL)

        first_remaining, first_timeout = sandbox.log[0]
        assert first_timeout == 900.0  # 0.5 x budget, the share cap
        assert first_remaining - first_timeout == pytest.approx(806.9)
        assert h.events("exec_capped") == []  # tool not wired to a store here

    async def test_the_cap_reason_reaches_the_model(self, tmp_path: Path) -> None:
        # The agent has to be able to tell "your command was too long" from
        # "the run is nearly over" to respond correctly to either.
        h, sandbox, _deadline = self._run_shape(tmp_path, 1800.0, 1706.9, 1800.0)
        await h.loop.run(GOAL)

        results = [
            payload
            for payload in h.events("tool_result")
            if "half the run's total budget" in (payload.get("content") or "")
        ]
        assert results


class TestLandingTurnBand:
    """Change 1c: the loop-side half of the landing guarantee.

    The hard stop is checked at the loop top; the exec it protects is
    issued *after* the model call has already spent part of the margin.
    Both round-2 hard-stops died in that gap — ``qemu-alpine-ssh`` seq 238
    passed the check at 63.0s remaining, generated for 4.55s and issued a
    30s exec at 58.4s; ``make-doom-for-mips`` seq 553 passed at 62.6s,
    generated for 15.83s and issued at 46.8s. No amount of tool-layer
    arithmetic recovers a turn the loop has already given away, which is
    why 1a alone buys nothing here.

    So the loop takes one explicit final turn instead: below
    ``WALL_CLOCK_STOP_FLOOR + recent_call_median()`` it injects
    :data:`LANDING_TURN_NOTICE`, persists ``landing_turn``, and arms the
    ``bash`` refusal — from state, not from arithmetic on any requested
    timeout.

    **Expected recovery on the measured corpus: zero tasks.** Neither trial
    had an answer worth landing. This makes the guarantee true; it does not
    claim a win.
    """

    def _wire(
        self,
        tmp_path: Path,
        script: list[ModelResponse],
        *,
        remaining: float,
        budget: float = 900.0,
        observations: tuple[float, ...] = (),
        tools: list[Tool] | None = None,
    ):
        """A run frozen at ``remaining`` with a clock only ``exec`` moves.

        ``observations`` pre-loads the deadline's call window, which is what
        sets the band's width. Model calls are free here on purpose: the
        band's *arming* is what is under test, and a free call keeps the
        arithmetic exact.
        """
        clock = _MovableClock()
        deadline = Deadline(budget, clock)
        clock.now = budget - remaining
        for seconds in observations:
            deadline.observe_model_call(seconds)
        sandbox = _RecordingBurnSandbox(clock, deadline)
        h = make_harness(
            tmp_path,
            script,
            tools=(
                tools
                if tools is not None
                else [bash_tool(sandbox, deadline=deadline)]  # type: ignore[arg-type]
            ),
            clock=clock,
            deadline=deadline,
            budgets=Budgets(max_turns=20),
        )
        return h, sandbox, deadline

    async def test_landing_turn_refuses_bash_and_lands_the_answer(
        self, tmp_path: Path
    ) -> None:
        # remaining 75 with a 20s typical call: 75 < 60 + 20, so the very
        # next turn is the landing turn — and it is the last one, so the
        # text it carries alongside its (refused) command is the answer.
        script = [
            resp(
                "my answer is in /app/out.txt",
                [call("c1", "bash", command="./slow")],
            ),
            resp("never reached"),
        ]
        h, sandbox, deadline = self._wire(
            tmp_path, script, remaining=75.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        # Armed exactly once, with the numbers that justified it.
        (event,) = h.events("landing_turn")
        assert event["remaining_seconds"] == pytest.approx(75.0)
        assert event["expected_call_seconds"] == pytest.approx(20.0)
        assert deadline.landing is True
        # The notice rode the very next model call.
        first_call_messages = [
            message.content for message in h.adapter.calls[0].messages
        ]
        assert LANDING_TURN_NOTICE.format(remaining=75) in first_call_messages
        # The bash call was refused without the sandbox ever being asked.
        assert sandbox.log == []
        (tool_result,) = h.events("tool_result")
        assert tool_result["content"] == LANDING_REFUSAL
        assert tool_result["is_error"] is False
        # And the run ends on the model's own words, not a budget pause.
        assert result.status == "completed"
        assert result.final_text == "my answer is in /app/out.txt"
        assert h.events("wall_clock_stop") == []
        # The landing turn is the last turn: the loop does not go round
        # again, so the second scripted response is never asked for.
        (finish,) = h.events("landing_finish")
        assert finish == {"tool_calls": 1, "has_text": True}
        assert len(h.adapter.calls) == 1
        assert result.turns == 1

    async def test_the_landing_turn_is_the_last_turn(
        self, tmp_path: Path
    ) -> None:
        """Regression: ``landing`` is a latch and
        :meth:`Deadline.begin_landing` has no reset, so the loop must stop
        on the landing turn. Before this fix it kept issuing ordinary turns
        with the shell permanently dead — the agent was stranded, not
        landed."""
        script = [
            resp("checking", [call("c1", "bash", command="./a")]),
            resp("checking again", [call("c2", "bash", command="./b")]),
            resp("done: /app/out.txt"),
        ]
        h, sandbox, _deadline = self._wire(
            tmp_path, script, remaining=75.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        assert len(h.events("landing_turn")) == 1
        notice = LANDING_TURN_NOTICE.format(remaining=75)
        messages = [
            payload.get("content") for payload in h.events("message")
        ]
        assert messages.count(notice) == 1
        # Exactly one turn happens after the notice, so exactly one command
        # is refused — a second refusal would mean a turn the agent spent
        # without a shell it could still have used.
        assert sandbox.log == []
        assert [r["content"] for r in h.events("tool_result")] == [
            LANDING_REFUSAL
        ]
        assert len(h.adapter.calls) == 1
        assert result.turns == 1
        assert result.status == "completed"

    async def test_a_stale_call_median_cannot_strand_the_agent(
        self, tmp_path: Path
    ) -> None:
        """The reviewed failure, verbatim: a 16-call window whose median
        lags the provider arms the band on evidence that is already stale,
        and the latch can never re-evaluate it. What must be bounded is the
        *consequence* — one turn, not the rest of the run.

        Before the fix this produced 40 consecutive bash refusals and zero
        execs while the provider was running at 5s/call, burning 200s of
        wall-clock the agent could have worked in.
        """
        clock = _MovableClock()
        deadline = Deadline(3600.0, clock)
        clock.now = 3600.0 - 259.0
        for _ in range(16):
            deadline.observe_model_call(200.0)  # stale: the provider is fast now
        sandbox = _RecordingBurnSandbox(clock, deadline)
        script = [
            resp(
                f"attempt {i}: answer at /app/out.txt",
                [call(f"c{i}", "bash", command="./probe", timeout=5)],
            )
            for i in range(40)
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox, deadline=deadline)],  # type: ignore[arg-type]
            clock=clock,
            deadline=deadline,
            budgets=Budgets(max_turns=100),
        )
        h.loop.adapter = _TimedFakeAdapter(script, clock, 5.0)  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        # The band still arms — 259 < 60 + 200 is what the evidence says.
        assert len(h.events("landing_turn")) == 1
        # But it costs exactly one turn and one refusal, not forty.
        refusals = [
            payload
            for payload in h.events("tool_result")
            if payload["content"] == LANDING_REFUSAL
        ]
        assert len(refusals) == 1
        assert sandbox.log == []
        assert len(h.loop.adapter.calls) == 1  # type: ignore[attr-defined]
        # And the run ends with the model's own answer rather than spinning
        # down to a bare pause with no final text.
        assert result.status == "completed"
        assert result.final_text == "attempt 0: answer at /app/out.txt"
        assert h.events("wall_clock_stop") == []
        assert deadline.remaining() == pytest.approx(254.0)

    async def test_an_empty_call_window_disables_the_band(
        self, tmp_path: Path
    ) -> None:
        # No model call has been observed yet, so there is no evidence for
        # how wide the band should be. Behaviour is exactly today's: the
        # command runs.
        script = [
            resp("working", [call("c1", "bash", command="./a", timeout=5)]),
            resp(CLEAN_FINISH),
        ]
        h, sandbox, deadline = self._wire(tmp_path, script, remaining=75.0)
        assert deadline.recent_call_median() is None
        result = await h.loop.run(GOAL)

        assert h.events("landing_turn") == []
        assert deadline.landing is False
        assert len(sandbox.log) == 1
        assert result.status == "completed"

    async def test_no_deadline_means_no_band(self, tmp_path: Path) -> None:
        script = [
            resp("working", [call("c1", "bash", command="./a", timeout=5)]),
            resp(CLEAN_FINISH),
        ]
        clock = _MovableClock()
        sandbox = _BurnsItsWindowSandbox(clock)
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox)],  # type: ignore[arg-type]
            clock=clock,
        )
        result = await h.loop.run(GOAL)

        assert h.events("landing_turn") == []
        assert sandbox.received_timeout == 5.0
        assert result.status == "completed"

    async def test_the_band_does_not_arm_above_it(self, tmp_path: Path) -> None:
        # remaining 81 with a 20s typical call: 81 >= 60 + 20, so turn 1 is
        # an ordinary working turn and the shell stays available. The 5s
        # exec then carries the run to 76, which *is* inside the band — so
        # the event that eventually appears records 76, never 81.
        script = [
            resp("working", [call("c1", "bash", command="./a", timeout=5)]),
            resp(CLEAN_FINISH),
        ]
        h, sandbox, _deadline = self._wire(
            tmp_path, script, remaining=81.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        assert len(sandbox.log) == 1
        remaining_before, timeout = sandbox.log[0]
        assert remaining_before == pytest.approx(81.0)
        assert timeout == 5.0  # ran in full, nothing refused, nothing capped
        assert [
            event["remaining_seconds"] for event in h.events("landing_turn")
        ] == [pytest.approx(76.0)]
        assert result.status == "completed"

    async def test_the_band_tracks_the_provider_not_a_constant(
        self, tmp_path: Path
    ) -> None:
        # The same remaining, a fast provider: a 2s typical call means the
        # run can still afford a working turn where a slow one could not.
        script = [
            resp("working", [call("c1", "bash", command="./a", timeout=5)]),
            resp(CLEAN_FINISH),
        ]
        h, sandbox, _deadline = self._wire(
            tmp_path, script, remaining=70.0, observations=(2.0,) * 16
        )
        await h.loop.run(GOAL)
        assert h.events("landing_turn") == []
        assert len(sandbox.log) == 1

        slow_h, slow_sandbox, _slow = self._wire(
            tmp_path / "slow", script, remaining=70.0, observations=(30.0,) * 16
        )
        await slow_h.loop.run(GOAL)
        assert len(slow_h.events("landing_turn")) == 1
        assert slow_sandbox.log == []

    async def test_the_refusal_costs_no_nudge_or_truncation_budget(
        self, tmp_path: Path
    ) -> None:
        # A normal ToolResult, so the loop treats the landing turn as a
        # working turn that produced a result — not as an error to
        # re-prompt, which is the one thing the final turn cannot afford.
        script = [
            resp("trying", [call("c1", "bash", command="./a")]),
            resp("answer at /app/out.txt"),
        ]
        h, _sandbox, _deadline = self._wire(
            tmp_path, script, remaining=75.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        assert h.events("truncation_continue") == []
        assert h.events("nudge") == []
        assert result.status == "completed"

    async def test_a_truncated_landing_turn_is_not_re_prompted(
        self, tmp_path: Path
    ) -> None:
        # 5a normally re-prompts a turn that ran to the output cap. On the
        # landing turn there is no turn left to re-prompt into, and the
        # re-prompt would spend the last of the wall-clock on a reply
        # nobody can read. Take the partial text as the answer instead.
        script = [
            resp(
                "the answer is at /app/out.txt and it cont",
                stop_reason=StopReason.MAX_TOKENS,
            ),
            resp("never reached"),
        ]
        h, _sandbox, _deadline = self._wire(
            tmp_path, script, remaining=75.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        assert len(h.events("landing_turn")) == 1
        assert h.events("truncation_continue") == []
        assert result.status == "completed"
        assert result.final_text == "the answer is at /app/out.txt and it cont"
        assert result.turns == 1

    async def test_a_landing_turn_that_looks_unfinished_is_not_nudged(
        self, tmp_path: Path
    ) -> None:
        # Same rule for the diligence nudge: "I will now verify" reads as
        # unfinished, but nudging spends a turn the run does not have.
        script = [
            resp("I will now verify the output at /app/out.txt"),
            resp("never reached"),
        ]
        h, _sandbox, _deadline = self._wire(
            tmp_path, script, remaining=75.0, observations=(20.0,) * 16
        )
        result = await h.loop.run(GOAL)

        assert looks_unfinished(script[0].message.content, 0)[0] is True
        assert len(h.events("landing_turn")) == 1
        assert h.events("nudge") == []
        assert result.status == "completed"
        assert result.turns == 1

    async def test_verification_still_runs_on_the_landing_turn(
        self, tmp_path: Path
    ) -> None:
        # The completion gate is not the shell: it is the loop's own exec,
        # exempt by design, and disarming it would silently demote a
        # passing gate to a non-gate.
        workspace = tmp_path / "ws"
        workspace.mkdir()
        real_sandbox = LocalSandbox(workspace)
        from harness.tools.builtin import declare_verification_tool

        clock = _MovableClock()
        deadline = Deadline(900.0, clock)
        clock.now = 825.0  # remaining 75
        for _ in range(16):
            deadline.observe_model_call(20.0)
        script = [
            resp("declaring", [call("v1", "declare_verification",
                                    command="true", description="it works")]),
            resp("answer at /app/out.txt"),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[
                bash_tool(real_sandbox, deadline=deadline),
                declare_verification_tool(),
            ],
            clock=clock,
            deadline=deadline,
            sandbox=real_sandbox,
        )
        result = await h.loop.run(GOAL)

        assert len(h.events("landing_turn")) == 1
        assert len(h.events("verification_passed")) == 1
        assert result.status == "completed"


class TestVerificationIsExemptFromTheExecCap:
    """X7: the completion gate must not be shortened into a non-gate.

    ``_run_verification`` runs at completion by definition. If the band
    softener or the share cap applied, a legitimate check would be cut into
    ``timed_out`` + ``timeout_capped``, which this loop reads as
    *inconclusive* and accepts — turning a passing gate into no gate at all.
    """

    async def test_a_250s_check_still_gets_its_full_window(
        self, tmp_path: Path
    ) -> None:
        from harness.diligence import VERIFICATION_TIMEOUT_SECONDS
        from harness.sandbox.base import ExecResult
        from harness.tools.builtin import declare_verification_tool

        # The reviewer's exact case: budget 900, remaining 400. Exploratory
        # arithmetic would hold back the band softener's share; verification
        # holds back only the landing reserve, so the 300s gate is untouched
        # and the 250s check inside it completes.
        clock = _MovableClock()
        deadline = Deadline(900.0, clock)
        clock.now = 500.0  # remaining 400
        script = [
            resp("declaring", [declare("v1", "pytest -q tests/acceptance.py")]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[declare_verification_tool()],
            clock=clock,
            deadline=deadline,
        )
        fake = ScriptedTimeoutSandbox(
            ExecResult(exit_code=0, stdout="250 passed", stderr="")
        )
        h.loop.sandbox = fake  # type: ignore[assignment]
        result = await h.loop.run(GOAL)

        assert fake.received_timeout is not None
        assert fake.received_timeout >= 300.0
        assert fake.received_timeout == VERIFICATION_TIMEOUT_SECONDS
        assert fake.received_timeout > 250.0  # the check fits
        assert result.status == "completed"
        (passed,) = h.events("verification_passed")
        assert "timeout_capped" not in passed
        assert h.events("verification_failed") == []

    async def test_the_same_shape_would_be_cut_for_an_exploratory_exec(
        self, tmp_path: Path
    ) -> None:
        # The contrast that makes the exemption load-bearing rather than
        # cosmetic: identical budget and remaining, ordinary bash tool.
        clock = _MovableClock()
        deadline = Deadline(900.0, clock)
        clock.now = 500.0  # remaining 400
        assert deadline.exec_cap(300.0) == (300.0, False, None)
        # ...and twenty seconds more of elapsed time is enough to split them.
        clock.now = 520.0  # remaining 380
        assert deadline.exec_cap(300.0) == (285.0, True, "band")
        assert deadline.exec_cap(300.0, purpose="verification") == (
            290.0,
            True,
            "reserve",
        )

    async def test_no_share_cap_on_a_long_verification(
        self, tmp_path: Path
    ) -> None:
        # VERIFICATION_TIMEOUT_SECONDS is well under 0.5 x any real budget,
        # so the share cap would not bite today — but the exemption is
        # explicit rather than accidental, and this pins it.
        deadline = Deadline(400.0, clock=lambda: 0.0)  # share cap = 200s
        for _ in range(4):
            deadline.observe_model_call(5.0)  # reserve 75
        assert deadline.exec_cap(300.0) == (200.0, True, "share")
        assert deadline.exec_cap(300.0, purpose="verification") == (
            300.0,
            False,
            None,
        )


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
        # Turn 2's request replayed the transcript — the assistant message
        # carrying the drop notice translated cleanly (no empty-message
        # rejection) — and the dropped-call reminder reached the model.
        assert len(completions.calls) == 2
        turn2_contents = [
            str(m.get("content") or "") for m in completions.calls[1]["messages"]
        ]
        assert any("tool call was dropped" in c for c in turn2_contents)
        assert any("cut off mid-arguments" in c for c in turn2_contents)
        # The drop is on the record, with the cut point visible.
        (dropped,) = [
            e.payload
            for e in store.load_events(agent_id)
            if e.kind == "tool_call_dropped"
        ]
        assert dropped["tool_name"] == "write_file"
        assert dropped["raw_arguments_prefix"].startswith('{"path": "interp.py"')


# ---------------------------------------------------------------------------
# Incomplete-turn degradation (§C4)
# ---------------------------------------------------------------------------


class TestIncompleteTurnReminders:
    """The continue now fires on ``incomplete``, not on MAX_TOKENS alone, and
    the re-prompt names the actual cause. One text for three causes was
    factually wrong for two of them, and a false diagnosis steers the model
    away from the one action that would fix the turn."""

    @pytest.mark.parametrize(
        ("incomplete_reason", "phrase", "dropped"),
        [
            ("dropped_calls", "cut off mid-arguments", [drop()]),
            ("max_tokens", "cut off at the output-token limit", None),
            ("no_finish_reason", "ended without completing", None),
        ],
    )
    async def test_reminder_text_matches_the_reason(
        self,
        tmp_path: Path,
        incomplete_reason: str,
        phrase: str,
        dropped: list[DroppedToolCall] | None,
    ) -> None:
        script = [
            resp(
                content="(placeholder)",
                incomplete_reason=incomplete_reason,
                dropped=dropped,
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        (event,) = h.events("truncation_continue")
        assert event["incomplete_reason"] == incomplete_reason
        turn2 = [m.content for m in h.adapter.calls[1].messages if m.content]
        assert any(phrase in text for text in turn2)
        # Exactly one reminder text is used — no cross-contamination.
        assert sum(phrase in text for text in turn2) == 1

    async def test_no_finish_reason_incomplete_is_continued(
        self, tmp_path: Path
    ) -> None:
        # A turn that ended with nothing usable and no recognised stop reason
        # used to be banked as a clean final answer.
        script = [
            resp(
                content="(provider response ended without completing)",
                stop_reason=StopReason.ERROR,
                incomplete_reason="no_finish_reason",
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert len(h.events("truncation_continue")) == 1

    async def test_complete_turn_with_unmapped_stop_reason_is_not_continued(
        self, tmp_path: Path
    ) -> None:
        """The M1 gate at the loop's end of the wire: an adapter that reports
        StopReason.ERROR but ``incomplete=False`` (a real answer arrived, the
        gateway just omitted its finish reason) must be accepted as final. The
        alternative is three spurious re-prompts at the end of every run."""
        script = [resp(CLEAN_FINISH, stop_reason=StopReason.ERROR)]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert result.turns == 1
        assert h.events("truncation_continue") == []

    async def test_incomplete_with_a_surviving_tool_call_dispatches(
        self, tmp_path: Path
    ) -> None:
        """Sibling-survivor: one call ran, one was dropped. The tool branch
        wins (no re-prompt, no continue consumed) and the drop is still on the
        record — otherwise the vanished call leaves no trace at all."""
        script = [
            resp(
                content="running one of two",
                calls=[call("c1", "echo", text="a")],
                dropped=[drop()],
            ),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert [r["content"] for r in h.events("tool_result")] == ["echo:a"]
        assert h.events("truncation_continue") == []
        assert [d["tool_name"] for d in h.events("tool_call_dropped")] == [
            "write_file"
        ]


class TestTruncationExhaustion:
    """C2-of-review: exhausting the continue budget on nothing but unparseable
    provider replies is a provider failure, and must not be laundered into a
    clean completion carrying a placeholder for an answer."""

    async def test_all_continues_dropping_calls_ends_as_error(
        self, tmp_path: Path
    ) -> None:
        from harness.loop import MAX_TRUNCATION_CONTINUES

        script = [
            resp(content="(1 tool call was dropped)", dropped=[drop()])
            for _ in range(MAX_TRUNCATION_CONTINUES + 1)
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.error_kind == "malformed_response"
        assert "unparseable tool-call arguments" in result.final_text
        assert result.turns == MAX_TRUNCATION_CONTINUES + 1
        assert len(h.events("truncation_continue")) == MAX_TRUNCATION_CONTINUES
        # Every drop is auditable — the only thing that makes this diagnosable.
        assert len(h.events("tool_call_dropped")) == (
            MAX_TRUNCATION_CONTINUES + 1
        )
        (run_error,) = h.events("run_error")
        assert run_error["error_kind"] == "malformed_response"

    async def test_max_tokens_exhaustion_without_drops_still_completes(
        self, tmp_path: Path
    ) -> None:
        # Unchanged from before C4: a model that merely thinks itself out of
        # budget every turn is a capability failure, not a provider one.
        from harness.loop import MAX_TRUNCATION_CONTINUES

        script = [
            resp(content="still thinking", stop_reason=StopReason.MAX_TOKENS)
            for _ in range(MAX_TRUNCATION_CONTINUES + 1)
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.error_kind is None
        assert h.events("run_error") == []

    async def test_a_single_clean_continue_prevents_the_error(
        self, tmp_path: Path
    ) -> None:
        """"Every continue dropped calls" is literal: one continue spent on a
        plain truncation means the provider was not uniformly broken, so the
        run is accepted rather than failed."""
        from harness.loop import MAX_TRUNCATION_CONTINUES

        script = [resp(content="thinking", stop_reason=StopReason.MAX_TOKENS)]
        script += [
            resp(content="(dropped)", dropped=[drop()])
            for _ in range(MAX_TRUNCATION_CONTINUES)
        ]
        h = make_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.error_kind is None


class TestTruncationDeadlineSkip:
    """M4a. A re-prompt is only worth its cost if the answer it provokes can
    arrive *and* be acted on. gpt2-codegolf: one 8192-token reasoning call
    cost 299s of a 900s budget, so 3 more continues would guarantee a
    wall-clock kill with an empty workspace — turning a fast death into a slow
    one that also loses the budget the continue exists to protect."""

    async def test_continue_is_skipped_when_two_calls_will_not_fit(
        self, tmp_path: Path
    ) -> None:
        # gpt2-codegolf's numbers exactly. Clock reads, in order: deadline
        # anchor, the turn's remaining() check, call_started, call end, then
        # step 5a's fresh remaining(). So the call cost 300s and 200s remain —
        # less than the 600s two more calls at that price would need.
        clock = scripted_clock([0.0, 0.0, 0.0, 300.0, 700.0])
        deadline = Deadline(900.0, clock)
        script = [
            resp(content="(truncated)", stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, clock=clock, deadline=deadline)
        result = await h.loop.run(GOAL)

        # Paused resumably — every event is already on disk — not "completed"
        # with a placeholder banked as the answer.
        assert result.status == "paused_budget"
        assert result.final_text is None
        (event,) = h.events("truncation_continue_skipped")
        assert event["reason"] == "deadline"
        assert event["last_call_seconds"] == pytest.approx(300.0)
        assert event["remaining_seconds"] == pytest.approx(200.0)
        assert event["incomplete_reason"] == "max_tokens"
        # No continue fired, and crucially no further model call was made.
        assert h.events("truncation_continue") == []
        assert len(h.adapter.calls) == 1

    async def test_continue_proceeds_when_the_budget_can_fund_it(
        self, tmp_path: Path
    ) -> None:
        # Same shape, cheap call (1s) against a 900s budget: the re-prompt
        # comfortably fits, so nothing is skipped.
        clock = scripted_clock([0.0, 10.0, 11.0, 12.0])
        deadline = Deadline(900.0, clock)
        script = [
            resp(content="(truncated)", stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, clock=clock, deadline=deadline)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert h.events("truncation_continue_skipped") == []
        assert len(h.events("truncation_continue")) == 1

    async def test_no_deadline_means_no_skip(self, tmp_path: Path) -> None:
        # Deadline-free runs (the default for direct callers) keep the
        # pre-C4 behaviour exactly: the guard is inert without a budget.
        clock = scripted_clock([0.0, 10.0, 1000.0, 2000.0])
        script = [
            resp(content="(truncated)", stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, clock=clock)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("truncation_continue_skipped") == []

    async def test_the_guard_is_priced_on_the_median_not_the_last_call(
        self, tmp_path: Path
    ) -> None:
        """make-doom-for-mips' numbers exactly. One 225.5s call — the turn
        that ran to the output cap, so by construction the expensive one —
        priced the whole decision under the old rule and paused the run with
        267.6s (30% of the budget) still on the clock. The window's median is
        9.0s, so two more calls plainly fit."""
        clock = scripted_clock([0.0, 0.0, 408.0, 633.0, 633.0])
        deadline = Deadline(900.0, clock)
        # Three earlier 9.0s calls, as an ordinary run would have recorded.
        for _ in range(3):
            deadline.observe_model_call(9.0)
        script = [
            resp(content="(truncated)", stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, clock=clock, deadline=deadline)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("truncation_continue_skipped") == []
        assert len(h.events("truncation_continue")) == 1
        assert len(h.adapter.calls) == 2

    async def test_the_skip_still_fires_inside_the_stop_floor(
        self, tmp_path: Path
    ) -> None:
        """Cheap calls do not buy a re-prompt below the hard-stop floor: the
        loop refuses to start a call there anyway, so the answer the
        re-prompt provokes could never be acted on."""
        clock = scripted_clock([0.0, 0.0, 800.0, 801.0, 870.0])
        deadline = Deadline(900.0, clock)
        for _ in range(3):
            deadline.observe_model_call(1.0)
        script = [
            resp(content="(truncated)", stop_reason=StopReason.MAX_TOKENS),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(tmp_path, script, clock=clock, deadline=deadline)
        result = await h.loop.run(GOAL)

        assert result.status == "paused_budget"
        (event,) = h.events("truncation_continue_skipped")
        assert event["remaining_seconds"] == pytest.approx(30.0)
        assert event["remaining_seconds"] < WALL_CLOCK_STOP_FLOOR
        assert event["typical_call_seconds"] == pytest.approx(1.0)
        assert h.events("truncation_continue") == []
        assert len(h.adapter.calls) == 1


class TestLengthBoundTruncation:
    """A tool call dropped on a turn that also ran to the output-token cap is
    a *length* failure, not a corrupted one. The observed drop carried 26 KB
    of C source: telling that model "your call was cut off, re-issue it"
    invites the same oversized call again, and lowering the per-call cap
    makes the truncation more likely — so the wording is the only lever, and
    the retry it provokes is attributed to the model rather than the
    provider."""

    @staticmethod
    def _at_cap(cap: int = 8192) -> ModelResponse:
        """A turn whose only tool call was dropped at the output-token cap."""
        return resp(
            content="(1 tool call was dropped)",
            usage=Usage(output_tokens=cap),
            stop_reason=StopReason.MAX_TOKENS,
            dropped=[drop()],
        )

    async def test_drop_at_the_output_cap_asks_for_smaller_pieces(
        self, tmp_path: Path
    ) -> None:
        from harness.loop import TRUNCATION_REMINDERS

        h = make_harness(
            tmp_path,
            [self._at_cap(), resp(CLEAN_FINISH)],
            budgets=Budgets(max_output_tokens=8192),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        reminder = h.adapter.calls[1].messages[-1]
        assert reminder.role is Role.USER
        assert reminder.content == TRUNCATION_REMINDERS["max_tokens"]
        assert "smaller pieces" in (reminder.content or "")
        (event,) = h.events("truncation_continue")
        assert event["length_bound"] is True
        assert event["incomplete_reason"] == "dropped_calls"

    async def test_the_per_call_cap_is_left_alone(self, tmp_path: Path) -> None:
        """The retry must not be squeezed: the dropped call was 26 KB of
        arguments, so a smaller cap makes the next attempt truncate sooner.
        The instruction is the lever, not the cap."""
        h = make_harness(
            tmp_path,
            [self._at_cap(), resp(CLEAN_FINISH)],
            budgets=Budgets(max_output_tokens=8192),
        )
        await h.loop.run(GOAL)

        assert [c.params["max_tokens"] for c in h.adapter.calls] == [8192, 8192]

    async def test_a_drop_below_the_cap_keeps_the_dropped_call_wording(
        self, tmp_path: Path
    ) -> None:
        """Only the coincidence with the output cap means "too long". A drop
        on a short turn is a corrupted call, and must still be diagnosed as
        one."""
        from harness.loop import TRUNCATION_REMINDERS

        short = resp(
            content="(1 tool call was dropped)",
            usage=Usage(output_tokens=100),
            dropped=[drop()],
        )
        h = make_harness(
            tmp_path,
            [short, resp(CLEAN_FINISH)],
            budgets=Budgets(max_output_tokens=8192),
        )
        await h.loop.run(GOAL)

        reminder = h.adapter.calls[1].messages[-1]
        assert reminder.content == TRUNCATION_REMINDERS["dropped_calls"]
        (event,) = h.events("truncation_continue")
        assert event["length_bound"] is False

    async def test_no_cap_configured_keeps_the_dropped_call_wording(
        self, tmp_path: Path
    ) -> None:
        """With no per-call cap there is no cap to have hit, so nothing can
        be attributed to length."""
        from harness.loop import TRUNCATION_REMINDERS

        h = make_harness(tmp_path, [self._at_cap(), resp(CLEAN_FINISH)])
        await h.loop.run(GOAL)

        reminder = h.adapter.calls[1].messages[-1]
        assert reminder.content == TRUNCATION_REMINDERS["dropped_calls"]
        (event,) = h.events("truncation_continue")
        assert event["length_bound"] is False

    async def test_a_constrained_retry_that_drops_again_is_not_a_fault(
        self, tmp_path: Path
    ) -> None:
        """The misattribution this guards against: a model that cannot
        shorten its writes would otherwise exhaust the continue budget on
        drops and be reported as a broken provider — the exact fault the
        round-1 taxonomy exists to keep *honest*."""
        from harness.loop import MAX_TRUNCATION_CONTINUES

        h = make_harness(
            tmp_path,
            [self._at_cap() for _ in range(MAX_TRUNCATION_CONTINUES + 1)],
            budgets=Budgets(max_output_tokens=8192),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert result.error_kind != "malformed_response"
        assert result.error_kind is None
        assert h.events("run_error") == []
        # The first drop is still provider evidence; the ones the loop had
        # already warned about are counted apart from it.
        counts = [
            e["constrained_retries"]
            for e in h.events("truncation_continue")
        ]
        assert counts == [0, 1, 2]

    async def test_all_unprompted_drops_still_end_as_malformed_response(
        self, tmp_path: Path
    ) -> None:
        """The round-1 exhaustion path is untouched when every continue was
        an unprompted provider drop — here with a per-call cap configured and
        every dropping turn comfortably under it."""
        from harness.loop import MAX_TRUNCATION_CONTINUES

        short = resp(
            content="(1 tool call was dropped)",
            usage=Usage(output_tokens=100),
            dropped=[drop()],
        )
        h = make_harness(
            tmp_path,
            [short] * (MAX_TRUNCATION_CONTINUES + 1),
            budgets=Budgets(max_output_tokens=8192),
        )
        result = await h.loop.run(GOAL)

        assert result.status == "error"
        assert result.error_kind == "malformed_response"
        assert all(
            e["constrained_retries"] == 0
            for e in h.events("truncation_continue")
        )


class TestIncompleteTurnEndToEnd:
    async def test_reasoning_only_turn_replays_with_its_placeholder(
        self, tmp_path: Path
    ) -> None:
        """The gpt2-codegolf reproduction, through the real OpenAI-compat
        adapter: turn 1 spent its whole 8192-token cap on hidden reasoning and
        returned ``content=null, tool_calls=[]``. That empty assistant message
        was persisted and then rejected on the *next* turn's translation,
        killing the run at turn 1. It must now carry a placeholder and replay
        cleanly. Fails on main."""
        from types import SimpleNamespace

        from harness.adapters.openai_compat import (
            EMPTY_MESSAGE_PLACEHOLDERS,
            OpenAICompatAdapter,
        )

        def sdk_response(
            content: str | None,
            tool_calls: list | None,
            finish_reason: str,
            completion_tokens: int = 5,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=content, tool_calls=tool_calls
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=completion_tokens
                ),
            )

        real_call = SimpleNamespace(
            id="c1",
            function=SimpleNamespace(
                name="echo", arguments='{"text": "hello"}'
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
                # Turn 1: the whole cap spent thinking, nothing emitted.
                sdk_response(None, None, "length", completion_tokens=8192),
                sdk_response(None, [real_call], "tool_calls"),
                sdk_response(CLEAN_FINISH, None, "stop"),
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAICompatAdapter("fake-model", client=client, stream=False)

        store = RunStore(tmp_path / "state.db")
        run_id = store.create_run(GOAL, "fake-model", "auto")
        agent_id = store.create_agent(run_id, GOAL)
        registry = ToolRegistry()
        registry.register(simple_tool("echo"))
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
            registry,
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
        assert len(completions.calls) == 3
        # The replayed transcript carried the placeholder, not an empty
        # assistant message — this is the assertion that fails on main.
        turn2_contents = [
            str(m.get("content") or "") for m in completions.calls[1]["messages"]
        ]
        assert EMPTY_MESSAGE_PLACEHOLDERS["max_tokens"] in turn2_contents
        assert "" not in [
            c
            for c, m in zip(turn2_contents, completions.calls[1]["messages"])
            if m["role"] == "assistant" and not m.get("tool_calls")
        ]


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
# Turn provenance and drop auditability (§C2)
# ---------------------------------------------------------------------------


class TestTurnProvenance:
    async def test_one_model_turn_event_per_call_in_order(
        self, tmp_path: Path
    ) -> None:
        """§C2: every model call leaves exactly one model_turn event, in
        order, carrying the provider's verbatim stop string, the normalized
        one, the turn's output tokens, and the same duration the usage row
        got (from the injected clock). This is the record that made the
        make-mips post-mortem impossible: today nothing on disk says why a
        turn ended."""
        clock = scripted_clock([10.0, 10.5, 20.0, 20.25])
        script = [
            resp(
                "working",
                [call("c1", "echo", text="a")],
                usage=Usage(input_tokens=7, output_tokens=3),
                provider_stop_reason="tool_calls",
            ),
            resp(
                CLEAN_FINISH,
                usage=Usage(input_tokens=11, output_tokens=4),
                provider_stop_reason="stop",
            ),
        ]
        h = make_harness(
            tmp_path, script, tools=[simple_tool("echo")], clock=clock
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        turns = h.events("model_turn")
        assert turns == [
            {
                "turn": 1,
                "provider_stop_reason": "tool_calls",
                "stop_reason": "tool_use",
                "output_tokens": 3,
                "reasoning_tokens": 0,
                "duration_ms": 500,
                "tool_call_count": 1,
                "content_chars": len("working"),
            },
            {
                "turn": 2,
                "provider_stop_reason": "stop",
                "stop_reason": "end_turn",
                "output_tokens": 4,
                "reasoning_tokens": 0,
                "duration_ms": 250,
                "tool_call_count": 0,
                "content_chars": len(CLEAN_FINISH),
            },
        ]
        # The event's duration agrees with the usage row's by construction —
        # one clock read feeds both.
        assert [r.duration_ms for r in h.store.list_usage(h.run_id)] == [
            500,
            250,
        ]

    async def test_model_turn_event_carries_reasoning_tokens(
        self, tmp_path: Path
    ) -> None:
        """Reasoning-token telemetry (openai_compat only): the model_turn
        event carries reasoning_tokens alongside output_tokens, straight
        from the scripted response's usage."""
        script = [
            resp(
                CLEAN_FINISH,
                usage=Usage(output_tokens=100, reasoning_tokens=42),
                provider_stop_reason="stop",
            ),
        ]
        h = make_harness(tmp_path, script)
        await h.loop.run(GOAL)

        (turn,) = h.events("model_turn")
        assert turn["output_tokens"] == 100
        assert turn["reasoning_tokens"] == 42

    async def test_unmapped_and_missing_stop_reasons_stay_distinguishable(
        self, tmp_path: Path
    ) -> None:
        """The whole point of §C2: both turns normalize to ``error``, but
        the persisted provenance still tells "the provider sent a value we
        do not map" apart from "the stream ended with nothing terminal"."""
        script = [
            resp(
                "still going",
                [call("c1", "echo", text="a")],
                stop_reason=StopReason.ERROR,
                provider_stop_reason="weird_reason",
            ),
            resp(
                CLEAN_FINISH,
                stop_reason=StopReason.ERROR,
                provider_stop_reason=None,
            ),
        ]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        await h.loop.run(GOAL)

        turns = h.events("model_turn")
        assert [t["stop_reason"] for t in turns] == ["error", "error"]
        assert [t["provider_stop_reason"] for t in turns] == [
            "weird_reason",
            None,
        ]

    async def test_model_response_raw_is_never_persisted(
        self, tmp_path: Path
    ) -> None:
        """Deliberate scope limit: raw is the whole response body per turn.
        The targeted provenance fields are recorded; raw is not."""
        response = resp(CLEAN_FINISH, provider_stop_reason="stop")
        response.raw = {"secret": "the entire provider response body"}
        h = make_harness(tmp_path, [response])
        await h.loop.run(GOAL)

        dumped = [
            e.payload for e in h.store.load_events(h.agent_id)
        ]
        assert not any("secret" in str(payload) for payload in dumped)
        assert h.events("model_turn")[0]["provider_stop_reason"] == "stop"

    async def test_adapter_error_persists_a_run_error_event(
        self, tmp_path: Path
    ) -> None:
        """§C2: a run that dies on an adapter failure says so in the
        transcript. Previously the only on-disk trace was the agent-status
        row, so triaging one meant crawling the caller's result.json."""
        script = [resp(None, [call("c1", "echo", text="a")])]
        h = make_harness(tmp_path, script, tools=[simple_tool("echo")])
        result = await h.loop.run(GOAL)  # second call exhausts the script

        assert result.status == "error"
        (event,) = h.events("run_error")
        assert "exhausted" in event["message"]
        assert event["turns"] == 1
        # error_kind is populated by the provider-fault taxonomy change;
        # until it lands the field is present and None, never missing.
        assert event["error_kind"] is None
        # It is written before the run ends, and it is the last event.
        assert h.event_kinds()[-1] == "run_error"

    async def test_no_run_error_event_on_success_or_budget_pause(
        self, tmp_path: Path
    ) -> None:
        h = make_harness(tmp_path, [resp(CLEAN_FINISH)])
        assert (await h.loop.run(GOAL)).status == "completed"
        assert h.events("run_error") == []

        h2 = make_harness(
            tmp_path / "b",
            [resp(CLEAN_FINISH)],
            budgets=Budgets(max_turns=0),
        )
        assert (await h2.loop.run(GOAL)).status == "paused_budget"
        assert h2.events("run_error") == []

    async def test_dropped_tool_calls_are_persisted_with_a_capped_prefix(
        self, tmp_path: Path
    ) -> None:
        """§C2 defines the emission site and schema so the degradation
        change that starts *reporting* drops is pure behaviour. The prefix
        is capped at 512 chars: an observed truncated write_file fragment
        was ~21 KB, and state.db must not swallow whole file bodies."""
        huge = "x" * 21_000
        response = ModelResponse(
            message=Message(
                role=Role.ASSISTANT, content="(a tool call was dropped)"
            ),
            usage=Usage(output_tokens=8192),
            stop_reason=StopReason.MAX_TOKENS,
            provider_stop_reason="length",
            incomplete=True,
            incomplete_reason="dropped_calls",
            dropped_tool_calls=[
                DroppedToolCall(
                    tool_name="write_file",
                    raw_arguments_prefix=huge,
                    raw_arguments_len=len(huge),
                )
            ],
        )
        h = make_harness(tmp_path, [response, resp(CLEAN_FINISH)])
        await h.loop.run(GOAL)

        (dropped,) = h.events("tool_call_dropped")
        assert dropped["turn"] == 1
        assert dropped["tool_name"] == "write_file"
        assert dropped["provider_stop_reason"] == "length"
        assert dropped["raw_arguments_len"] == 21_000
        assert dropped["raw_arguments_prefix"] == "x" * 512
        assert len(dropped["raw_arguments_prefix"]) == (
            DROPPED_ARGUMENTS_PREFIX_CHARS
        )
        # Recorded with, and immediately after, its turn's provenance.
        kinds = h.event_kinds()
        assert kinds[1:3] == ["model_turn", "tool_call_dropped"]

    async def test_dropped_calls_recorded_when_a_sibling_call_survives(
        self, tmp_path: Path
    ) -> None:
        """The drop must be on the record even on the tool-call branch —
        that branch dispatches and continues, so a turn where one call ran
        and another vanished would otherwise leave no trace at all."""
        response = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                content="running one of two",
                tool_calls=[call("c1", "echo", text="a")],
            ),
            usage=Usage(),
            stop_reason=StopReason.TOOL_USE,
            provider_stop_reason="tool_calls",
            incomplete=True,
            incomplete_reason="dropped_calls",
            dropped_tool_calls=[
                DroppedToolCall(
                    tool_name="write_file",
                    raw_arguments_prefix='{"path": "a.py", "content": "def ',
                    raw_arguments_len=33,
                )
            ],
        )
        h = make_harness(
            tmp_path,
            [response, resp(CLEAN_FINISH)],
            tools=[simple_tool("echo")],
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert [r["content"] for r in h.events("tool_result")] == ["echo:a"]
        (dropped,) = h.events("tool_call_dropped")
        assert dropped["tool_name"] == "write_file"
        assert dropped["raw_arguments_len"] == 33

    async def test_no_dropped_events_when_the_adapter_reports_none(
        self, tmp_path: Path
    ) -> None:
        """No drops reported, no events written — the common case must stay
        free of noise."""
        h = make_harness(tmp_path, [resp(CLEAN_FINISH)])
        await h.loop.run(GOAL)
        assert h.events("tool_call_dropped") == []


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
        # remaining=200 throughout (fixed clock); the same fixed clock makes
        # every model call take 0s, so the landing allowance clamps to its
        # 15s minimum and the reserve is 60 + 15 = 75. 200 - 75 = 125, below
        # VERIFICATION_TIMEOUT_SECONDS(300) -> capped to 125s. 200 is also
        # above the wind-down threshold for a 200s "budget" (100s), so
        # wind-down does not fire and cannot be confused with this path.
        deadline = Deadline(200.0, clock=lambda: 0.0)
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
        assert failed["timeout_seconds"] == pytest.approx(125.0)
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
        # remaining=200 with a fixed clock -> reserve 75 (see above) -> 125s
        deadline = Deadline(200.0, clock=lambda: 0.0)
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
        assert fake.received_timeout == pytest.approx(125.0)
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        assert passed["timeout_capped"] is True
        assert passed["timeout_seconds"] == pytest.approx(125.0)
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


# ---------------------------------------------------------------------------
# Verification-quality lint (Change 4) — warn-only, no control-flow change
# ---------------------------------------------------------------------------


def lint_harness(
    tmp_path: Path, script: list[ModelResponse]
) -> "Harness":
    """A verification harness whose declare_verification tool lints against
    the loop's own live written-data map.

    The tool is registered *after* :func:`make_harness` because the
    accessor has to close over the loop that maintains the map — exactly
    the ordering the orchestrator faces (registry first, loop second).
    """
    from harness.tools.builtin import declare_verification_tool, write_file_tool

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    sandbox = LocalSandbox(workspace)
    h = make_harness(
        tmp_path,
        script,
        tools=[bash_tool(sandbox), write_file_tool(sandbox)],
        sandbox=sandbox,
    )
    h.loop.registry.register(
        declare_verification_tool(
            written_data=lambda: h.loop.written_data,
            store=h.store,
            agent_id=h.agent_id,
        )
    )
    return h


class TestVerificationLint:
    """The mteb-leaderboard shape, replayed offline through FakeAdapter."""

    MTEB_WRITE = 'echo "Qwen/Qwen3-Embedding-8B" > result.txt'
    MTEB_CHECK = "grep -q '^Qwen/Qwen3-Embedding-8B$' result.txt"

    async def test_mteb_replay_warns_but_still_completes(
        self, tmp_path: Path
    ) -> None:
        """The circular check is flagged everywhere it should be — and the
        run finishes exactly as it did before, because the lint warns."""
        script = [
            resp("writing the answer", [call("b1", "bash", command=self.MTEB_WRITE)]),
            resp("declaring", [declare("v1", self.MTEB_CHECK)]),
            resp(CLEAN_FINISH),
        ]
        h = lint_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        # 1. No control-flow change: same completion, same turn count as an
        #    unflagged run, no nudge, no failure.
        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH
        assert result.turns == 3
        assert h.events("nudge") == []
        assert h.events("verification_failed") == []

        # 2. The tool warned the model on the turn it declared.
        (lint,) = h.events("verification_lint")
        assert lint["command"] == self.MTEB_CHECK
        assert lint["action"] == "warn"
        # Both provenance detectors fire: the literal was echoed into the
        # very file the grep reads (tautology), and nothing between the
        # echo and the grep ran anything at all (no_execution, T1).
        assert [f["kind"] for f in lint["findings"]] == [
            "tautology",
            "no_execution",
        ]

        # 3. The gate recorded the pass *and* that it was flagged weak.
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        assert [f["kind"] for f in passed["lint_findings"]] == [
            "tautology",
            "no_execution",
        ]
        assert passed["lint_findings"][0]["details"]["path"] == "result.txt"
        assert passed["lint_findings"][1]["details"]["authored_paths"] == [
            "result.txt"
        ]

    async def test_honest_check_is_never_flagged(self, tmp_path: Path) -> None:
        """The X9 acceptance case end to end: the agent writes a program
        that prints PASS, runs it into test.log, and greps that — no
        finding anywhere, and no lint_findings key on the pass."""
        script = [
            resp(
                "writing the solver",
                [
                    call(
                        "w1",
                        "write_file",
                        path="solve.py",
                        content='print("ALL_CHECKS_PASSED")\n',
                    )
                ],
            ),
            resp(
                "running it",
                [
                    call(
                        "b1",
                        "bash",
                        command="python3 solve.py > test.log",
                    )
                ],
            ),
            resp(
                "declaring",
                [declare("v1", 'grep -q "ALL_CHECKS_PASSED" test.log')],
            ),
            resp(CLEAN_FINISH),
        ]
        h = lint_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("verification_lint") == []
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        assert "lint_findings" not in passed
        # Regression (T1 review): the advisory is model-facing text, so a
        # detector firing here does not merely mislabel a corpus row — it
        # tells the model, in the transcript, that a correct check proves
        # nothing. Nothing in what the model saw may say so.
        tool_results = [
            payload.get("content") or ""
            for payload in h.events("tool_result")
        ]
        assert not any("Advisory" in content for content in tool_results)
        assert not any(
            "runs the solution" in content for content in tool_results
        )

    async def test_written_data_ignores_failed_tool_calls(
        self, tmp_path: Path
    ) -> None:
        """A denied write taught the lint nothing, so the same declaration
        is not flagged."""
        script = [
            resp(
                "writing the answer",
                [call("b1", "bash", command=self.MTEB_WRITE)],
            ),
            resp("declaring", [declare("v1", self.MTEB_CHECK)]),
            resp(CLEAN_FINISH),
        ]
        h = lint_harness(tmp_path, script)
        # Deny bash so its result is an error result: nothing was written.
        h.loop.policy = Policy(mode=PermissionMode.AUTO, deny=("bash",))
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("verification_lint") == []
        assert len(h.loop.written_data) == 0

    async def test_neutralized_exit_does_not_reject_the_declaration(
        self, tmp_path: Path
    ) -> None:
        """`|| true` is warned about and then honored — the declaration is
        recorded, executed, and the run completes."""
        script = [
            resp("declaring", [declare("v1", "echo verified-ok || true")]),
            resp(CLEAN_FINISH),
        ]
        h = lint_harness(tmp_path, script)
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        (declared,) = h.events("verification_declared")
        assert declared["command"] == "echo verified-ok || true"
        (lint,) = h.events("verification_lint")
        assert [f["kind"] for f in lint["findings"]] == ["neutralized_exit"]
        (passed,) = h.events("verification_passed")
        assert passed["exit_code"] == 0
        assert passed["lint_findings"][0]["kind"] == "neutralized_exit"

    async def test_loop_lints_even_without_a_wired_tool_accessor(
        self, tmp_path: Path
    ) -> None:
        """The gate reads the loop's own map, so a registry built without
        the accessor still produces the round-3 corpus (minus the
        model-facing advisory)."""
        from harness.tools.builtin import declare_verification_tool

        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        sandbox = LocalSandbox(workspace)
        script = [
            resp(
                "writing the answer",
                [call("b1", "bash", command=self.MTEB_WRITE)],
            ),
            resp("declaring", [declare("v1", self.MTEB_CHECK)]),
            resp(CLEAN_FINISH),
        ]
        h = make_harness(
            tmp_path,
            script,
            tools=[bash_tool(sandbox), declare_verification_tool()],
            sandbox=sandbox,
        )
        result = await h.loop.run(GOAL)

        assert result.status == "completed"
        assert h.events("verification_lint") == []  # no store wired in
        (passed,) = h.events("verification_passed")
        assert [f["kind"] for f in passed["lint_findings"]] == [
            "tautology",
            "no_execution",
        ]
