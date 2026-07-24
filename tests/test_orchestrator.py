"""Tests for harness.orchestrator (DESIGN.md §4.12).

No network, no API keys, no Docker daemon: ``DockerSandbox.availability`` is
patched to False (so every run falls back to :class:`LocalSandbox` on
``tmp_path``), and every model is a scripted :class:`FakeAdapter` injected
via ``adapter_override`` — real provider adapters are never constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from collections.abc import Callable

from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig, PermissionMode
from harness.deadline import Deadline
from harness.loop import Budgets
from harness.orchestrator import (
    Orchestrator,
    ToolDeps,
    UnknownModelError,
    UnknownRunError,
)
from harness.permissions import ToolMeta
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.tools.registry import Tool
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

#: The LocalSandbox-fallback warning fires on every run in these tests
#: (Docker availability is patched off); only the dedicated warning test
#: asserts on it.
pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")

GOAL = "Write hello.txt containing hi."

#: A final message the diligence check accepts as finished.
CLEAN_FINISH = "Task complete. Wrote the file; contents verified."


@pytest.fixture(autouse=True)
def no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the LocalSandbox fallback: tests must not need a Docker daemon."""
    monkeypatch.setattr(
        DockerSandbox, "availability", classmethod(lambda cls: False)
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A throwaway HARNESS_HOME directory."""
    return tmp_path / "home"


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    """A RunStore on a tmp database, closed after the test."""
    with RunStore(tmp_path / "state.db") as run_store:
        yield run_store


@pytest.fixture
def orchestrator(home: Path, store: RunStore) -> Orchestrator:
    """An Orchestrator over an empty default config rooted at ``home``."""
    return Orchestrator(HarnessConfig(home=home), store)


def resp(
    content: str | None = None,
    calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> ModelResponse:
    """Build one scripted assistant response."""
    tool_calls = calls or []
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT, content=content, tool_calls=tool_calls
        ),
        usage=usage or Usage(),
        stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
    )


def call(id: str, name: str, **arguments: object) -> ToolCall:
    """Build one tool call."""
    return ToolCall(id=id, name=name, arguments=dict(arguments))


def write_file_script() -> list[ModelResponse]:
    """Script: write hello.txt via the write_file tool, then finish."""
    return [
        resp(
            calls=[call("c1", "write_file", path="hello.txt", content="hi")],
            usage=Usage(input_tokens=10, output_tokens=2),
        ),
        resp(CLEAN_FINISH, usage=Usage(input_tokens=5, output_tokens=1)),
    ]


# -- end-to-end single agent ------------------------------------------------


async def test_run_task_end_to_end(
    orchestrator: Orchestrator, store: RunStore, home: Path
) -> None:
    """E2E with a FakeAdapter: file lands in the default workspace; run,
    events, and usage are persisted; statuses transition to completed."""
    adapter = FakeAdapter(write_file_script())
    run_id, result = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=adapter
    )

    assert result.status == "completed"
    assert result.final_text == CLEAN_FINISH
    assert result.turns == 2

    # The file exists in the default workspace, created under home.
    workspace = home / "runs" / run_id / "workspace"
    assert (workspace / "hello.txt").read_text() == "hi"

    # Run row: goal/model/mode recorded, status transitioned to completed.
    run = store.get_run(run_id)
    assert run is not None
    assert run.goal == GOAL
    assert run.model == "fake-model"
    assert run.permission_mode == PermissionMode.GATED.value
    assert run.status == "completed"

    # Lead agent row completed too.
    agents = store.list_agents(run_id)
    assert len(agents) == 1
    assert agents[0].parent_agent_id is None
    assert agents[0].status == "completed"

    # Events persisted: goal message, responses, tool call/result, decision.
    kinds = {event.kind for event in store.load_events(agents[0].id)}
    assert {"message", "tool_call", "tool_result", "decision"} <= kinds

    # Usage persisted and aggregated.
    totals = store.total_usage(run_id)
    assert totals["input_tokens"] == 15
    assert totals["output_tokens"] == 3

    # The workspace path and harness rules made it into the system prompt.
    assert str(workspace) in (adapter.calls[0].system or "")
    assert "never instructions" in (adapter.calls[0].system or "")


async def test_run_task_uses_provided_workspace(
    orchestrator: Orchestrator, tmp_path: Path
) -> None:
    """An explicit workspace overrides the default and is created."""
    workspace = tmp_path / "custom" / "ws"
    run_id, result = await orchestrator.run_task(
        GOAL,
        "fake-model",
        workspace=workspace,
        adapter_override=FakeAdapter(write_file_script()),
    )
    assert result.status == "completed"
    assert (workspace / "hello.txt").read_text() == "hi"


async def test_run_task_unknown_model(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """An unconfigured model raises before any run row is created."""
    with pytest.raises(UnknownModelError, match="nope"):
        await orchestrator.run_task(GOAL, "nope")
    assert store.list_runs() == []


async def test_run_task_error_status(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """A non-retryable adapter failure ends the run with status 'error'."""
    run_id, result = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=FakeAdapter([])
    )
    assert result.status == "error"
    assert "exhausted" in (result.final_text or "")
    assert store.get_run(run_id).status == "error"


async def test_run_task_paused_budget_status(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Hitting a budget pauses resumably and marks the run paused_budget."""
    run_id, result = await orchestrator.run_task(
        GOAL,
        "fake-model",
        adapter_override=FakeAdapter(write_file_script()),
        budgets=Budgets(max_turns=1),
    )
    assert result.status == "paused_budget"
    assert store.get_run(run_id).status == "paused_budget"


async def test_local_fallback_warns(orchestrator: Orchestrator) -> None:
    """Without a Docker daemon the orchestrator warns about LocalSandbox."""
    with pytest.warns(UserWarning, match="LocalSandbox"):
        await orchestrator.run_task(
            GOAL, "fake-model", adapter_override=FakeAdapter(write_file_script())
        )


# -- subagents ---------------------------------------------------------------


async def test_subagent_spawn_and_await(
    orchestrator: Orchestrator, store: RunStore, home: Path
) -> None:
    """The lead spawns a subagent (shared sandbox), awaits it, and receives
    its final text; the subagent registry lacks spawn/await (depth cap 1)."""
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Write child.txt.")]),
            resp(calls=[call("a1", "await_agents")]),
            resp("Task complete. The subagent wrote child.txt."),
        ]
    )
    child_adapter = FakeAdapter(
        [
            resp(
                calls=[
                    call("w1", "write_file", path="child.txt", content="from child")
                ]
            ),
            resp("Child task complete. Wrote child.txt."),
        ]
    )
    adapters = iter([lead_adapter, child_adapter])

    run_id, result = await orchestrator.run_task(
        "Fan out the work.",
        "fake-model",
        adapter_override=lambda: next(adapters),
    )
    assert result.status == "completed"

    # Shared sandbox: the child's file is in the run workspace.
    workspace = home / "runs" / run_id / "workspace"
    assert (workspace / "child.txt").read_text() == "from child"

    # Agent rows: lead plus one subagent parented to it, both completed.
    agents = store.list_agents(run_id)
    assert len(agents) == 2
    lead, child = agents
    assert lead.parent_agent_id is None
    assert child.parent_agent_id == lead.id
    assert lead.status == "completed"
    assert child.status == "completed"
    assert child.prompt == "Write child.txt."

    # await_agents returned the child's final text to the lead as data.
    await_results = [
        event.payload["content"]
        for event in store.load_events(lead.id)
        if event.kind == "tool_result"
        and event.payload["tool_call_id"] == "a1"
    ]
    assert len(await_results) == 1
    assert "Child task complete." in await_results[0]
    assert child.id in await_results[0]

    # Depth cap 1 by construction: the lead's toolset has spawn/await, the
    # subagent's does not.
    lead_tools = {spec.name for spec in lead_adapter.calls[0].tools}
    child_tools = {spec.name for spec in child_adapter.calls[0].tools}
    assert {"spawn_agent", "await_agents"} <= lead_tools
    assert "spawn_agent" not in child_tools
    assert "await_agents" not in child_tools
    # Both share the rest of the builtin toolset.
    assert {"bash", "write_file", "task_update", "load_skill"} <= child_tools


async def test_subagent_spawn_returns_id_immediately(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """spawn_agent's tool result carries the new agent id without waiting
    for the child to finish."""
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Child work.")]),
            resp(calls=[call("a1", "await_agents")]),
            resp("Task complete. Verified."),
        ]
    )
    child_adapter = FakeAdapter([resp("Child task complete.")])
    adapters = iter([lead_adapter, child_adapter])

    run_id, result = await orchestrator.run_task(
        "Spawn one child.", "fake-model", adapter_override=lambda: next(adapters)
    )
    assert result.status == "completed"

    lead, child = store.list_agents(run_id)
    spawn_results = [
        event.payload["content"]
        for event in store.load_events(lead.id)
        if event.kind == "tool_result"
        and event.payload["tool_call_id"] == "s1"
    ]
    assert spawn_results == [f"spawned subagent {child.id}"]


async def test_unfinished_subagent_is_cancelled(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """A spawned-but-never-awaited subagent is cancelled at run end and its
    agent row marked cancelled."""
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Never awaited.")]),
            resp("Task complete. Verified."),
        ]
    )
    # The child starts but blocks on its tool dispatch (a real await); the
    # lead finishes without awaiting it, so the task is cancelled mid-turn.
    child_adapter = FakeAdapter(
        [
            resp(calls=[call("w1", "write_file", path="x.txt", content="x")]),
            resp("Child task complete."),
        ]
    )
    adapters = iter([lead_adapter, child_adapter])

    run_id, result = await orchestrator.run_task(
        "Spawn and forget.", "fake-model", adapter_override=lambda: next(adapters)
    )
    assert result.status == "completed"
    lead, child = store.list_agents(run_id)
    assert child.status == "cancelled"


# -- session grants ----------------------------------------------------------


async def test_grant_updates_live_loop_policies(
    orchestrator: Orchestrator,
) -> None:
    """grant() rewrites every live loop's policy via Policy.with_grant."""
    # A run establishes a live lead loop; builtin tools are side-effect-free
    # so no ASK fires during it — grant() is exercised directly, as the CLI
    # ask callback's 'a' answer would.
    await orchestrator.run_task(
        GOAL,
        "fake-model",
        adapter_override=FakeAdapter(write_file_script()),
    )
    assert orchestrator._live_loops[0].policy.allow == ()
    orchestrator.grant("bash")
    assert "bash" in orchestrator._live_loops[0].policy.allow
    assert orchestrator._grants == ["bash"]


# -- resume ------------------------------------------------------------------


async def test_resume_task_continues_run(
    orchestrator: Orchestrator, store: RunStore, home: Path
) -> None:
    """A budget-paused run resumes: transcript replayed, remaining budgets
    applied, and the run completes."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL,
        "fake-model",
        adapter_override=first,
        budgets=Budgets(max_turns=1),
    )
    assert paused.status == "paused_budget"
    assert store.get_run(run_id).status == "paused_budget"

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id, adapter_override=second, budgets=Budgets(max_turns=3)
    )
    assert result.status == "completed"
    assert result.final_text == CLEAN_FINISH
    assert store.get_run(run_id).status == "completed"

    # Artifact from the first phase survives (same default workspace).
    assert (home / "runs" / run_id / "workspace" / "hello.txt").exists()

    # The replayed transcript reached the resumed adapter: original goal
    # first, the first phase's tool activity in between, resume marker last.
    messages = second.calls[0].messages
    assert messages[0].content == GOAL
    assert any(
        message.tool_result is not None
        and message.tool_result.tool_call_id == "c1"
        for message in messages
    )
    assert "resumed" in (messages[-1].content or "")


async def test_resume_task_unknown_run(orchestrator: Orchestrator) -> None:
    """Resuming a nonexistent run raises UnknownRunError."""
    with pytest.raises(UnknownRunError, match="no such run"):
        await orchestrator.resume_task("deadbeef")


async def test_resume_task_exhausted_budget_pauses_again(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """When nothing remains of the budget, resume pauses again immediately
    instead of making model calls."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=first, budgets=Budgets(max_turns=1)
    )
    assert paused.status == "paused_budget"

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id, adapter_override=second, budgets=Budgets(max_turns=1)
    )
    assert result.status == "paused_budget"
    assert second.calls == []  # no model call was made


async def test_resume_carries_output_token_and_wall_clock_budgets(
    orchestrator: Orchestrator,
    store: RunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: resume_task used to rebuild the remaining Budgets from
    max_turns/max_tokens only, silently dropping ``max_output_tokens`` (a
    per-call cap with nothing to subtract) and ``wall_clock_seconds`` (which
    restarts fresh — the external deadline it mirrors is per-invocation).
    Both must carry through to the resumed execution."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=first, budgets=Budgets(max_turns=1)
    )
    assert paused.status == "paused_budget"

    captured: dict[str, Budgets] = {}
    real_execute = Orchestrator._execute

    async def spy_execute(self, **kwargs):
        captured["budgets"] = kwargs["budgets"]
        return await real_execute(self, **kwargs)

    monkeypatch.setattr(Orchestrator, "_execute", spy_execute)

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id,
        adapter_override=second,
        budgets=Budgets(
            max_turns=3,
            max_output_tokens=4096,
            wall_clock_seconds=2400.0,
        ),
    )
    assert result.status == "completed"

    remaining = captured["budgets"]
    assert remaining.max_turns == 2  # one turn already consumed
    assert remaining.max_output_tokens == 4096  # carried, not dropped
    assert remaining.wall_clock_seconds == 2400.0  # restarts fresh, full value


# -- crash-resume: dangling tool calls ---------------------------------------


async def test_resume_synthesizes_results_for_dangling_tool_calls(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (§4.10/§5 Flow B): a crash between tool-call persistence
    and tool-result persistence leaves an assistant tool_use with no result;
    resume must synthesize an error result (and surface the dangling intent)
    instead of replaying a provider-invalid transcript."""
    run_id = store.create_run(GOAL, "fake-model", "auto")
    lead_id = store.create_agent(run_id, GOAL)
    store.append_event(
        lead_id, "message", Message(role=Role.USER, content=GOAL).model_dump(mode="json")
    )
    assistant = Message(
        role=Role.ASSISTANT,
        content="sending",
        tool_calls=[ToolCall(id="t1", name="send_email", arguments={"to": "x"})],
    )
    store.append_event(lead_id, "message", assistant.model_dump(mode="json"))
    store.append_event(
        lead_id, "tool_call", assistant.tool_calls[0].model_dump(mode="json")
    )
    # ... crash: no tool_result event was ever persisted.

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(run_id, adapter_override=second)
    assert result.status == "completed"

    messages = second.calls[0].messages
    # The dangling call got a synthesized error result, in place (right
    # after its assistant message, before the resume marker).
    tool_messages = [m for m in messages if m.tool_result is not None]
    assert len(tool_messages) == 1
    synthesized = tool_messages[0].tool_result
    assert synthesized.tool_call_id == "t1"
    assert synthesized.is_error is True
    assert "may or may not" in synthesized.content
    assert messages.index(tool_messages[0]) > messages.index(
        next(m for m in messages if m.role is Role.ASSISTANT)
    )
    # No assistant tool call is left unanswered anywhere in the replay.
    answered = {
        m.tool_result.tool_call_id for m in messages if m.tool_result is not None
    }
    for message in messages:
        for tool_call in message.tool_calls:
            assert tool_call.id in answered
    # The dangling intent is surfaced in the resume marker.
    assert "may or may not have actually executed" in (messages[-1].content or "")
    assert "send_email" in (messages[-1].content or "")
    # The synthesized result is persisted, so a second resume replays it
    # instead of re-synthesizing.
    synthesized_events = [
        event
        for event in store.load_events(lead_id)
        if event.kind == "tool_result" and event.payload["tool_call_id"] == "t1"
    ]
    assert len(synthesized_events) == 1


# -- crash-resume: compacted history ------------------------------------------


async def test_resume_replays_compacted_span_as_its_summary(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (G2/§4.3): resume must reconstruct the *compacted*
    transcript — substituting each compaction event's persisted summary for
    its evicted span — not replay the full pre-compaction history."""
    run_id = store.create_run(GOAL, "fake-model", "auto")
    lead_id = store.create_agent(run_id, GOAL)

    def add_message(role: Role, content: str) -> None:
        store.append_event(
            lead_id,
            "message",
            Message(role=role, content=content).model_dump(mode="json"),
        )

    add_message(Role.USER, GOAL)
    add_message(Role.ASSISTANT, "old thinking that was evicted")
    summary = f"[COMPACTION SUMMARY]\nOriginal goal (verbatim...):\n{GOAL}\n---\nS"
    store.append_event(
        lead_id,
        "compaction",
        {
            "evicted_count": 2,
            "evicted": [],
            "summary": summary,
        },
    )
    add_message(Role.ASSISTANT, "recent thinking, kept")

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(run_id, adapter_override=second)
    assert result.status == "completed"

    messages = second.calls[0].messages
    assert messages[0].content == summary
    assert messages[1].content == "recent thinking, kept"
    assert all(
        "old thinking that was evicted" != (m.content or "") for m in messages
    )


# -- crashed subagents ---------------------------------------------------------


class _CrashingAdapter(FakeAdapter):
    """An adapter whose complete() raises a non-AdapterError bug."""

    async def complete(self, messages, tools, system=None, **params):
        raise RuntimeError("subagent adapter bug")


async def test_await_agents_survives_one_crashed_subagent(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression: one crashed subagent must not discard its siblings'
    reports, and the crashed agent's row must not stay 'running' forever."""
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Crash, please.")]),
            resp(calls=[call("s2", "spawn_agent", prompt="Succeed, please.")]),
            resp(calls=[call("a1", "await_agents")]),
            resp("Task complete. Verified."),
        ]
    )
    crashing = _CrashingAdapter([])
    healthy = FakeAdapter([resp("Child task complete. All good.")])
    adapters = iter([lead_adapter, crashing, healthy])

    run_id, result = await orchestrator.run_task(
        "Fan out.", "fake-model", adapter_override=lambda: next(adapters)
    )
    assert result.status == "completed"

    lead, crashed, ok = store.list_agents(run_id)
    (await_result,) = [
        event.payload["content"]
        for event in store.load_events(lead.id)
        if event.kind == "tool_result" and event.payload["tool_call_id"] == "a1"
    ]
    # The healthy sibling's report survived ...
    assert "All good." in await_result
    # ... and the crash is its own per-agent section, not the whole result.
    assert "crashed" in await_result
    assert "RuntimeError" in await_result
    assert "subagent adapter bug" in await_result
    # The crashed agent's row is closed out as 'error', not left 'running'.
    assert crashed.prompt == "Crash, please."
    assert store.get_agent(crashed.id).status == "error"
    assert store.get_agent(ok.id).status == "completed"


async def test_unawaited_crashed_subagent_failure_is_recorded(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression: a subagent that crashes but is never awaited must not
    have its exception silently discarded — the failure is persisted as an
    agent_error event and the row marked 'error'."""
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Crash, please.")]),
            resp("Task complete. Verified."),
        ]
    )
    crashing = _CrashingAdapter([])
    adapters = iter([lead_adapter, crashing])

    run_id, result = await orchestrator.run_task(
        "Spawn and forget.", "fake-model", adapter_override=lambda: next(adapters)
    )
    assert result.status == "completed"

    lead, child = store.list_agents(run_id)
    assert store.get_agent(child.id).status == "error"
    error_events = [
        event.payload
        for event in store.load_events(child.id)
        if event.kind == "agent_error"
    ]
    assert len(error_events) == 1
    assert "RuntimeError" in error_events[0]["error"]
    assert "subagent adapter bug" in error_events[0]["error"]


# -- budget scoping on resume --------------------------------------------------


async def test_resume_budget_counts_only_the_leads_own_usage(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression: live runs give each subagent its own fresh budget, so
    resume must charge only the lead's own spend against the lead's budget —
    not the whole run's (which would re-pause immediately after heavy
    subagent activity)."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=first, budgets=Budgets(max_turns=1)
    )
    assert paused.status == "paused_budget"
    # A subagent burned (way) more than the whole token budget.
    lead = store.list_agents(run_id)[0]
    sub_id = store.create_agent(run_id, "big spender", parent_agent_id=lead.id)
    store.set_agent_status(sub_id, "completed")
    store.record_usage(
        run_id, sub_id, "fake-model", Usage(input_tokens=10_000_000)
    )

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id, adapter_override=second, budgets=Budgets(max_turns=3)
    )
    # Charged run-wide, the lead would pause again with zero model calls.
    assert result.status == "completed"
    assert len(second.calls) == 1


# -- grant persistence ---------------------------------------------------------


async def test_grants_are_persisted_and_restored_on_resume(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (§4.11 'persisted to the run's policy'): an
    always-for-this-run grant survives interruption — the resumed run's
    policy allows the pattern without re-asking."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL,
        "fake-model",
        mode=PermissionMode.GATED,
        adapter_override=first,
        budgets=Budgets(max_turns=1),
    )
    assert paused.status == "paused_budget"
    orchestrator.grant("bash")

    grant_events = [
        event
        for agent in store.list_agents(run_id)
        for event in store.load_events(agent.id)
        if event.kind == "grant"
    ]
    assert [event.payload for event in grant_events] == [{"pattern": "bash"}]

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(run_id, adapter_override=second)
    assert result.status == "completed"
    assert "bash" in orchestrator._live_loops[0].policy.allow


# -- config permission patterns ------------------------------------------------


async def test_config_permission_patterns_reach_the_policy(
    home: Path, store: RunStore
) -> None:
    """Regression (§4.11): [permissions] allow/deny patterns from
    config.toml must be threaded into every run's Policy."""
    config = HarnessConfig(home=home, permission_deny=("write_file",))
    orchestrator = Orchestrator(config, store)
    adapter = FakeAdapter(write_file_script())
    run_id, result = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=adapter
    )
    assert result.status == "completed"
    lead = store.list_agents(run_id)[0]
    (denial,) = [
        event.payload
        for event in store.load_events(lead.id)
        if event.kind == "tool_result"
    ]
    assert denial["is_error"] is True
    assert denial["content"] == "denied by policy"


# -- instruction ledger seeding ------------------------------------------------


async def test_goal_is_seeded_into_instruction_ledger(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (§4.5): the instruction-ledger machinery is actually
    wired — the goal is extracted at run start, persisted, and rendered
    into the system prompt's ledger section (so reminders never re-inject
    '(no instructions recorded)')."""
    adapter = FakeAdapter(write_file_script())
    run_id, result = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=adapter
    )
    assert result.status == "completed"

    items = store.list_instructions(run_id)
    assert [(item.item_id, item.instruction, item.source) for item in items] == [
        ("goal", GOAL, "user")
    ]
    system = adapter.calls[0].system or ""
    assert "Instruction ledger" in system
    assert GOAL in system


async def test_instructions_reloaded_into_context_on_resume(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (§4.5): instructions persist across interruption and are
    reloaded into the resumed context's ledger."""
    first = FakeAdapter(
        [resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")])]
    )
    run_id, paused = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=first, budgets=Budgets(max_turns=1)
    )
    assert paused.status == "paused_budget"
    store.upsert_instruction(run_id, "i1", "never touch prod", "user")

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id, adapter_override=second, budgets=Budgets(max_turns=3)
    )
    assert result.status == "completed"
    system = second.calls[0].system or ""
    assert "never touch prod" in system


# -- skill bodies survive pruning and resume -----------------------------------


def _add_skill(home: Path) -> str:
    body = "Always greet in French. UNIQUE-SKILL-MARKER."
    skill_dir = home / "skills" / "greet"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: greet\ndescription: Greets politely\n---\n{body}\n",
        encoding="utf-8",
    )
    return body


async def test_loaded_skill_body_rides_system_prompt_for_rest_of_run(
    orchestrator: Orchestrator, store: RunStore, home: Path
) -> None:
    """Regression (§4.6): load_skill splices the body into the system
    prompt (immune to tool-result pruning), not just into a tool result
    that gets pruned after 3 assistant turns."""
    body = _add_skill(home)
    script = [
        resp(
            calls=[ToolCall(id="k1", name="load_skill", arguments={"name": "greet"})]
        ),
        # 4 more assistant turns: past PRUNE_KEEP_TURNS, the load_skill
        # tool result is stubbed out of the transcript.
        resp(calls=[call("c1", "write_file", path="a.txt", content="a")]),
        resp(calls=[call("c2", "write_file", path="b.txt", content="b")]),
        resp(calls=[call("c3", "write_file", path="c.txt", content="c")]),
        resp(calls=[call("c4", "write_file", path="d.txt", content="d")]),
        resp(CLEAN_FINISH),
    ]
    adapter = FakeAdapter(script)
    run_id, result = await orchestrator.run_task(
        "Greet in French.", "fake-model", adapter_override=adapter
    )
    assert result.status == "completed"

    final_call = adapter.calls[-1]
    # The tool result itself has been pruned to a stub by now ...
    stubbed = [
        m.tool_result.content
        for m in final_call.messages
        if m.tool_result is not None and "[pruned:" in m.tool_result.content
    ]
    assert stubbed, "expected the old load_skill result to be pruned"
    # ... but the body still applies, via the system prompt.
    assert body in (final_call.system or "")


async def test_loaded_skill_body_restored_on_resume(
    orchestrator: Orchestrator, store: RunStore, home: Path
) -> None:
    """Regression (§4.6): a skill loaded before an interruption is
    re-spliced into the system prompt when the run resumes."""
    body = _add_skill(home)
    first = FakeAdapter(
        [
            resp(
                calls=[
                    ToolCall(
                        id="k1", name="load_skill", arguments={"name": "greet"}
                    )
                ]
            )
        ]
    )
    run_id, paused = await orchestrator.run_task(
        "Greet in French.",
        "fake-model",
        adapter_override=first,
        budgets=Budgets(max_turns=1),
    )
    assert paused.status == "paused_budget"

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(
        run_id, adapter_override=second, budgets=Budgets(max_turns=3)
    )
    assert result.status == "completed"
    assert body in (second.calls[0].system or "")


# -- search_history availability -----------------------------------------------


async def test_search_history_recovers_persisted_output(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Regression (§4.3.4): the retrieval backstop is a real registered
    tool — the agent can grep content out of its own persisted event log."""
    script = [
        resp(calls=[call("c1", "write_file", path="hello.txt", content="hi")]),
        resp(calls=[call("c2", "search_history", query="hello.txt")]),
        resp(CLEAN_FINISH),
    ]
    adapter = FakeAdapter(script)
    run_id, result = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=adapter
    )
    assert result.status == "completed"
    assert "search_history" in {spec.name for spec in adapter.calls[0].tools}

    lead = store.list_agents(run_id)[0]
    (search_result,) = [
        event.payload
        for event in store.load_events(lead.id)
        if event.kind == "tool_result" and event.payload["tool_call_id"] == "c2"
    ]
    assert search_result["is_error"] is False
    assert "hello.txt" in search_result["content"]


# -- wall-clock deadline seam (wind-down plan §2b/§2d) -------------------------


def scripted_clock(values: list[float]) -> Callable[[], float]:
    """A clock returning ``values`` in order, then repeating the last."""
    it = iter(values)
    last = values[-1]

    def clock() -> float:
        nonlocal last
        last = next(it, last)
        return last

    return clock


def spy_tool_factory(captured: dict) -> Callable[[ToolDeps], Tool]:
    """A tool factory that records the ToolDeps bundle it is invoked with."""

    def factory(deps: ToolDeps) -> Tool:
        captured["deps"] = deps

        async def handler(arguments: dict) -> str:
            return "ok"

        return Tool(
            spec=ToolSpec(name="spy", description="records its deps"),
            meta=ToolMeta(side_effect=False),
            handler=handler,
        )

    return factory


async def test_run_task_deadline_reaches_tool_factories(
    orchestrator: Orchestrator,
) -> None:
    """Ordering regression (§2b blocker): the deadline must be resolved at
    the top of _execute, BEFORE the registry/ToolDeps build — the ToolDeps
    bundle every factory receives carries the very instance passed to
    run_task, not None."""
    captured: dict = {}
    deadline = Deadline(3600.0)
    run_id, result = await orchestrator.run_task(
        GOAL,
        "fake-model",
        adapter_override=FakeAdapter([resp(CLEAN_FINISH)]),
        tool_factories=[spy_tool_factory(captured)],
        deadline=deadline,
    )
    assert result.status == "completed"
    assert captured["deps"].deadline is deadline
    # The lead loop shares the identical instance.
    assert orchestrator._live_loops[0].deadline is deadline


async def test_wall_clock_budget_builds_one_shared_deadline(
    orchestrator: Orchestrator,
) -> None:
    """Back-compat path: no injected deadline but budgets.wall_clock_seconds
    set — _execute constructs ONE Deadline from it (before the registry
    build) and both the tool factories and the lead loop see that same
    object."""
    captured: dict = {}
    run_id, result = await orchestrator.run_task(
        GOAL,
        "fake-model",
        adapter_override=FakeAdapter([resp(CLEAN_FINISH)]),
        tool_factories=[spy_tool_factory(captured)],
        budgets=Budgets(wall_clock_seconds=3600.0),
    )
    assert result.status == "completed"
    deadline = captured["deps"].deadline
    assert deadline is not None
    assert deadline.budget == 3600.0
    assert orchestrator._live_loops[0].deadline is deadline


async def test_late_spawned_subagent_shares_the_aged_deadline(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Subagent sharing (§2b): the run's ONE deadline is handed to every
    loop, so a subagent whose first check is already inside the wind-down
    band is born wound-down — the reminder lands on its turn 1."""
    # Anchored at 0; every later read sees 700 → 200s of 900s remain:
    # inside the clamped 300s wind-down band, above the 60s stop floor.
    deadline = Deadline(900.0, scripted_clock([0.0, 700.0]))
    lead_adapter = FakeAdapter(
        [
            resp(calls=[call("s1", "spawn_agent", prompt="Child work.")]),
            resp(calls=[call("a1", "await_agents")]),
            resp("Task complete. Verified."),
        ]
    )
    child_adapter = FakeAdapter([resp("Child task complete.")])
    adapters = iter([lead_adapter, child_adapter])

    run_id, result = await orchestrator.run_task(
        "Fan out late.",
        "fake-model",
        adapter_override=lambda: next(adapters),
        deadline=deadline,
    )
    assert result.status == "completed"

    lead, child = store.list_agents(run_id)
    child_wind_downs = [
        event.payload
        for event in store.load_events(child.id)
        if event.kind == "wind_down"
    ]
    assert len(child_wind_downs) == 1
    assert child_wind_downs[0]["remaining_seconds"] == pytest.approx(200.0)
    # The reminder reached the child on its very first model call.
    turn1_texts = [
        m.content for m in child_adapter.calls[0].messages if m.content
    ]
    assert any("approaching your hard time limit" in t for t in turn1_texts)


async def test_hard_stopped_run_is_resumable(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    """Hard stop (§2d): an already-expired deadline pauses the run with
    zero model calls and a persisted wall_clock_stop; a later resume (a
    fresh invocation, no deadline) completes it."""
    expired = Deadline(100.0, scripted_clock([0.0, 200.0]))
    first = FakeAdapter([resp(CLEAN_FINISH)])
    run_id, paused = await orchestrator.run_task(
        GOAL, "fake-model", adapter_override=first, deadline=expired
    )
    assert paused.status == "paused_budget"
    assert first.calls == []  # no model call was started
    assert store.get_run(run_id).status == "paused_budget"
    lead = store.list_agents(run_id)[0]
    stops = [
        event.payload
        for event in store.load_events(lead.id)
        if event.kind == "wall_clock_stop"
    ]
    assert len(stops) == 1
    assert stops[0]["remaining_seconds"] == 0.0

    second = FakeAdapter([resp(CLEAN_FINISH)])
    result = await orchestrator.resume_task(run_id, adapter_override=second)
    assert result.status == "completed"
    assert store.get_run(run_id).status == "completed"
