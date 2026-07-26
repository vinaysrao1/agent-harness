"""Unit tests for harness.tools.registry and harness.tools.builtin."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.deadline import (
    EXEC_CAP_FLOOR_SECONDS,
    LANDING_ALLOWANCE_DEFAULT,
    WALL_CLOCK_STOP_FLOOR,
    Deadline,
)
from harness.diligence import WrittenData, record_written_data
from harness.memory.store import FactNotFoundError, MemoryStore
from harness.permissions import ToolMeta
from harness.persistence import RunStore
from harness.sandbox.base import ExecResult, SandboxError
from harness.sandbox.local import LocalSandbox
from harness.skills import SkillLibrary
from harness.context import ContextManager
from harness.tools.builtin import (
    DEFAULT_EXEC_TIMEOUT,
    MIN_EXEC_SECONDS,
    LANDING_REFUSAL,
    MissingArgumentError,
    add_instruction_tool,
    bash_tool,
    declare_verification_tool,
    edit_file_tool,
    load_skill_tool,
    memory_read_fact_tool,
    memory_search_tool,
    memory_write_fact_tool,
    read_file_tool,
    search_history_tool,
    task_list_tool,
    task_update_tool,
    write_file_tool,
)
from harness.tools.registry import (
    MAX_RESULT_BYTES,
    DuplicateToolError,
    Tool,
    ToolRegistry,
)
from harness.types import Message, Role, ToolCall, ToolSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(tmp_path / "workspace")


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def run_store(tmp_path: Path):
    store = RunStore(tmp_path / "state.db")
    yield store
    store.close()


@pytest.fixture
def run_id(run_store: RunStore) -> str:
    return run_store.create_run(
        goal="test goal", model="fake", permission_mode="gated"
    )


@pytest.fixture
def skill_library(tmp_path: Path) -> SkillLibrary:
    root = tmp_path / "skills"
    skill_dir = root / "greet"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Says hello politely\n---\n"
        "Say hello and ask how you can help.\n",
        encoding="utf-8",
    )
    return SkillLibrary(root)


def _call(name: str, arguments: dict | None = None, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


def _fake_tool(name: str, handler=None, *, side_effect: bool = False) -> Tool:
    async def default_handler(arguments: dict) -> str:
        return "ok"

    return Tool(
        spec=ToolSpec(
            name=name,
            description="a fake tool for registry tests",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        meta=ToolMeta(side_effect=side_effect),
        handler=handler or default_handler,
    )


def _assert_valid_object_schema(schema: dict) -> None:
    """Minimal structural check that ``schema`` is a JSON Schema object shape."""
    assert schema.get("type") == "object"
    assert isinstance(schema.get("properties"), dict)
    assert isinstance(schema.get("required"), list)
    for key in schema["required"]:
        assert key in schema["properties"], f"required key {key!r} not in properties"
    for prop_name, prop_schema in schema["properties"].items():
        assert isinstance(prop_schema, dict)
        assert "type" in prop_schema, f"property {prop_name!r} missing 'type'"


def _make_context(reminder_interval: int = 1) -> ContextManager:
    """A ContextManager for context-binding tests (stub counter/summarizer)."""

    async def summarize(messages):
        return "stub"

    return ContextManager(
        base_system_prompt="base prompt",
        count_tokens=lambda messages: 0,
        max_context=1_000_000,
        summarize=summarize,
        reminder_interval=reminder_interval,
    )


class FakeExecSandbox:
    """A ``Sandbox`` stub that records the ``timeout`` it received from
    ``bash_tool`` and returns a scripted :class:`ExecResult` -- used to test
    the deadline-driven exec cap (wind-down plan §Fix 3a) without a real
    subprocess or a real sleep."""

    def __init__(self, result: ExecResult | None = None) -> None:
        self.received_timeout: float | None = None
        self._result = result or ExecResult(exit_code=0, stdout="ok", stderr="")

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        self.received_timeout = timeout
        return self._result


def _fixed_deadline(
    budget: float,
    remaining: float | None = None,
    observations: tuple[float, ...] = (),
) -> Deadline:
    """A ``Deadline`` of ``budget`` frozen at ``remaining`` seconds left --
    a clock that never advances, so cap-math tests are exact and don't race
    real wall-clock time. ``remaining`` defaults to the whole ``budget``.

    Budget and remaining are separate because the exec cap reads both: the
    share cap is a fraction of the *budget* while the reserve and the band
    guarantee work off what *remains*.

    ``observations`` pre-loads model-call durations so the landing reserve
    (``WALL_CLOCK_STOP_FLOOR + landing_allowance()``) is pinned; with none,
    the reserve is the no-observations default of 60 + 30 = 90s.
    """
    elapsed = 0.0 if remaining is None else budget - remaining
    calls = iter([0.0])
    deadline = Deadline(budget, clock=lambda: next(calls, elapsed))
    for seconds in observations:
        deadline.observe_model_call(seconds)
    return deadline


# ---------------------------------------------------------------------------
# ToolRegistry: register/get/specs
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_register_then_get_returns_same_tool(self):
        registry = ToolRegistry()
        tool = _fake_tool("noop")
        registry.register(tool)
        assert registry.get("noop") is tool

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        with pytest.raises(DuplicateToolError, match="noop"):
            registry.register(_fake_tool("noop"))

    def test_get_unknown_raises_keyerror(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="nope"):
            registry.get("nope")

    def test_specs_returns_every_registered_spec(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("a"))
        registry.register(_fake_tool("b"))
        names = {spec.name for spec in registry.specs()}
        assert names == {"a", "b"}
        assert all(isinstance(spec, ToolSpec) for spec in registry.specs())

    def test_specs_empty_for_fresh_registry(self):
        assert ToolRegistry().specs() == []


# ---------------------------------------------------------------------------
# ToolRegistry.dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_success_returns_content_untouched(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        result = await registry.dispatch(_call("noop"))
        assert result.tool_call_id == "call-1"
        assert result.content == "ok"
        assert result.is_error is False

    async def test_unknown_tool_is_error_not_exception(self):
        registry = ToolRegistry()
        result = await registry.dispatch(_call("does-not-exist", call_id="call-9"))
        assert result.tool_call_id == "call-9"
        assert result.is_error is True
        assert "does-not-exist" in result.content

    async def test_handler_exception_becomes_error_result(self):
        async def boom(arguments: dict) -> str:
            raise RuntimeError("kaboom")

        registry = ToolRegistry()
        registry.register(_fake_tool("boom", handler=boom))
        result = await registry.dispatch(_call("boom"))
        assert result.is_error is True
        assert "kaboom" in result.content
        assert "boom" in result.content
        assert "RuntimeError" in result.content

    async def test_handler_exception_does_not_propagate(self):
        # The whole point of dispatch: a crashing handler must not raise
        # out of dispatch() and take the agent loop down with it.
        async def boom(arguments: dict) -> str:
            raise ValueError("nope")

        registry = ToolRegistry()
        registry.register(_fake_tool("boom", handler=boom))
        result = await registry.dispatch(_call("boom"))  # must not raise
        assert result.is_error is True

    async def test_oversized_result_is_truncated_with_marker(self):
        original_size = MAX_RESULT_BYTES + 5000

        async def huge(arguments: dict) -> str:
            return "x" * original_size

        registry = ToolRegistry()
        registry.register(_fake_tool("huge", handler=huge))
        result = await registry.dispatch(_call("huge"))
        assert result.is_error is False
        assert len(result.content.encode("utf-8")) < MAX_RESULT_BYTES + 200
        assert "truncated" in result.content
        assert str(original_size) in result.content

    async def test_small_result_not_truncated(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        result = await registry.dispatch(_call("noop"))
        assert "truncated" not in result.content

    async def test_result_at_exact_limit_not_truncated(self):
        async def exact(arguments: dict) -> str:
            return "y" * MAX_RESULT_BYTES

        registry = ToolRegistry()
        registry.register(_fake_tool("exact", handler=exact))
        result = await registry.dispatch(_call("exact"))
        assert "truncated" not in result.content
        assert len(result.content) == MAX_RESULT_BYTES


# ---------------------------------------------------------------------------
# Sandbox tools: bash / read_file / write_file / edit_file
# ---------------------------------------------------------------------------


class TestBashTool:
    def test_schema_is_valid(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        assert tool.spec.name == "bash"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert "command" in tool.spec.input_schema["required"]

    def test_side_effect_false(self, sandbox: LocalSandbox):
        assert bash_tool(sandbox).meta.side_effect is False

    async def test_runs_command_and_formats_output(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        output = await tool.handler({"command": "echo hello"})
        assert "exit code: 0" in output
        assert "hello" in output
        assert "stdout" in output

    async def test_empty_stdout_and_stderr_omit_boilerplate_sections(
        self, sandbox: LocalSandbox
    ):
        """Regression (DESIGN.md §10.2 A4): an empty stdout/stderr section is
        no longer printed as an empty ``--- stdout/stderr ---`` header --
        that's boilerplate with no information content, and `bash` is the
        highest-volume tool, so it adds up."""
        tool = bash_tool(sandbox)
        output = await tool.handler({"command": "true"})
        assert "exit code: 0" in output
        assert "--- stdout ---" not in output
        assert "--- stderr ---" not in output
        assert "(no output)" in output

    async def test_captures_nonzero_exit_and_stderr(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        output = await tool.handler({"command": "echo problem 1>&2; exit 3"})
        assert "exit code: 3" in output
        assert "problem" in output
        assert "--- stderr ---" in output
        # No stdout was produced, so its section is omitted.
        assert "--- stdout ---" not in output

    async def test_missing_command_raises(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        with pytest.raises(MissingArgumentError):
            await tool.handler({})

    async def test_end_to_end_via_registry_dispatch(self, sandbox: LocalSandbox):
        registry = ToolRegistry()
        registry.register(bash_tool(sandbox))
        result = await registry.dispatch(_call("bash", {"command": "echo via-dispatch"}))
        assert result.is_error is False
        assert "via-dispatch" in result.content


class TestBashToolDeadlineCap:
    """Wind-down plan §Fix 3a: the ``bash`` tool caps its exec timeout by
    the run's remaining wall-clock, using a fake sandbox that records the
    timeout it actually received -- no real sleeps."""

    async def test_no_deadline_is_a_pure_passthrough(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox)
        await tool.handler({"command": "echo hi", "timeout": 45})
        assert sandbox.received_timeout == 45

    async def test_ample_remaining_leaves_requested_timeout_untouched(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(10_000.0))
        await tool.handler({"command": "echo hi", "timeout": 45})
        assert sandbox.received_timeout == 45

    async def test_cap_applies_when_remaining_is_tight(self):
        # remaining=200 of 900, reserve=90 (60 floor + 30 default allowance)
        # -> allowed 110s, below the 120s default. In-band (200 < the 300s
        # wind-down threshold), so the band softener does not add. Since
        # 1b, the omitted timeout is shrunk to 110s up front rather than
        # asked for at 120s and capped, but the exec window is identical.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, 200.0))
        await tool.handler({"command": "echo hi"})
        assert sandbox.received_timeout == 110.0

    async def test_cap_uses_the_observed_landing_allowance(self):
        # Same remaining, a fast provider: p75 of 5s calls clamps up to the
        # 15s minimum, so the reserve is 75s and the exec gets 125s.
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(
            900.0, 200.0, observations=(5.0, 5.0, 5.0, 5.0)
        )
        tool = bash_tool(sandbox, deadline=deadline)
        await tool.handler({"command": "echo hi", "timeout": 600})
        assert sandbox.received_timeout == 125.0

    async def test_capped_exec_leaves_more_than_the_stop_floor(self):
        # The Change 0 contract, at the tool boundary: after a capped exec
        # burns its whole window, the loop can still start a model call.
        sandbox = FakeExecSandbox()
        remaining = 336.0
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, remaining))
        await tool.handler({"command": "sleep 300", "timeout": 300})
        assert sandbox.received_timeout == 246.0  # 336 - 90
        assert remaining - sandbox.received_timeout > WALL_CLOCK_STOP_FLOOR

    async def test_floor_respected_when_remaining_minus_reserve_below_floor(
        self,
    ):
        # remaining=100, reserve=90 -> 10s, below EXEC_CAP_FLOOR_SECONDS
        # (30): the floor wins so trivial commands don't spuriously fail.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, 100.0))
        await tool.handler({"command": "echo hi", "timeout": 120})
        assert sandbox.received_timeout == EXEC_CAP_FLOOR_SECONDS

    async def test_requested_below_the_cap_is_left_untouched(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(
            sandbox, deadline=_fixed_deadline(900.0, 200.0)
        )  # allows 110s
        await tool.handler({"command": "echo hi", "timeout": 10})
        assert sandbox.received_timeout == 10.0

    async def test_capped_timeout_result_names_both_values(self):
        sandbox = FakeExecSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        tool = bash_tool(
            sandbox, deadline=_fixed_deadline(900.0, 200.0)
        )  # allows 110s
        # An *explicit* timeout: 1b makes "capped from your requested Ns"
        # reachable only for a number the agent actually chose.
        output = await tool.handler({"command": "sleep 1000", "timeout": 1000})
        assert "timed out after 110.0s" in output
        assert "capped from your requested 1000.0s" in output
        assert "~200.0s of wall-clock remain" in output
        assert "do not start another long command" in output

    async def test_capped_success_result_carries_a_note(self):
        sandbox = FakeExecSandbox(ExecResult(exit_code=0, stdout="done", stderr=""))
        tool = bash_tool(
            sandbox, deadline=_fixed_deadline(900.0, 200.0)
        )  # allows 110s
        output = await tool.handler({"command": "echo hi", "timeout": 120})
        assert "note: timeout was capped to 110.0s" in output
        assert "exit code: 0" in output

    async def test_uncapped_success_carries_no_note(self):
        sandbox = FakeExecSandbox(ExecResult(exit_code=0, stdout="done", stderr=""))
        tool = bash_tool(sandbox, deadline=_fixed_deadline(10_000.0))
        output = await tool.handler({"command": "echo hi"})
        assert "note: timeout was capped" not in output

    async def test_description_mentions_the_cap(self):
        assert "cap" in bash_tool(FakeExecSandbox()).spec.description.lower()

    def test_exec_reserve_and_floor_constants_are_exported(self):
        # Sanity: the constants this test module relies on actually live in
        # harness.deadline (§Fix 3a: shared with the loop's hard stop).
        assert WALL_CLOCK_STOP_FLOOR == 60.0
        assert LANDING_ALLOWANCE_DEFAULT == 30.0
        assert EXEC_CAP_FLOOR_SECONDS == 30.0
        assert _fixed_deadline(900.0).landing_reserve() == 90.0

    async def test_share_cap_binds_a_command_issued_early(self):
        # The mcmc shape: 1800s requested at remaining 1706.9 of 1800. Only
        # the share cap contains it; the band guarantee alone allows 1346.9s.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(1800.0, 1706.9))
        await tool.handler({"command": "R CMD INSTALL rstan", "timeout": 1800})
        assert sandbox.received_timeout == 900.0

    async def test_band_guarantee_binds_above_the_threshold(self):
        # remaining 335.5 of 900 is above the 300s wind-down threshold, so
        # the softener holds back 0.25 x 335.5 = 83.875s.
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(
            900.0, 335.5, observations=(5.0, 5.0, 5.0, 5.0)
        )
        tool = bash_tool(sandbox, deadline=deadline)
        await tool.handler({"command": "./compress", "timeout": 300})
        assert sandbox.received_timeout == pytest.approx(251.625)

    async def test_cap_message_names_the_share_reason(self):
        sandbox = FakeExecSandbox(ExecResult(exit_code=0, stdout="ok", stderr=""))
        tool = bash_tool(sandbox, deadline=_fixed_deadline(1800.0, 1706.9))
        output = await tool.handler({"command": "make", "timeout": 1800})
        assert "half the run's total budget" in output

    async def test_cap_message_names_the_landing_reason(self):
        sandbox = FakeExecSandbox(ExecResult(exit_code=0, stdout="ok", stderr=""))
        tool = bash_tool(
            sandbox, deadline=_fixed_deadline(900.0, 200.0)
        )  # in-band: reserve, not share
        output = await tool.handler({"command": "make", "timeout": 120})
        assert "land your answer" in output


class TestBashToolDeadlineAwareDefault:
    """Change 1b: an omitted ``timeout`` is the harness's number.

    380 of 513 observed bash calls (74%) named no timeout, and 24 of the 63
    ``exec_capped`` events were raised against the resulting bare 120s
    default — telling the model it "requested 120s" when it requested
    nothing. The default now shrinks to what the run can afford instead, so
    the command still runs and no cap is reported for a number the agent
    never chose.
    """

    async def test_default_is_the_named_constant_without_a_deadline(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox)
        await tool.handler({"command": "echo hi"})
        assert sandbox.received_timeout == DEFAULT_EXEC_TIMEOUT

    async def test_ample_remaining_leaves_the_default_alone(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(10_000.0))
        output = await tool.handler({"command": "echo hi"})
        assert sandbox.received_timeout == DEFAULT_EXEC_TIMEOUT
        assert "no timeout was given" not in output

    async def test_default_shrinks_to_what_the_run_can_afford(self):
        # remaining 200 of 900: affordable = 200 - 90 = 110s.
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(900.0, 200.0)
        tool = bash_tool(sandbox, deadline=deadline)
        output = await tool.handler({"command": "echo hi"})
        assert sandbox.received_timeout == min(
            DEFAULT_EXEC_TIMEOUT, deadline.affordable_exec_seconds()
        )
        assert sandbox.received_timeout == 110.0
        # It is described as what it is, never as a cap on a request.
        assert "capped from your requested" not in output
        assert "no timeout was given" in output

    async def test_no_exec_capped_event_for_a_shrunk_default(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        sandbox = FakeExecSandbox()
        tool = bash_tool(
            sandbox,
            deadline=_fixed_deadline(900.0, 200.0),
            store=run_store,
            agent_id=agent_id,
        )
        await tool.handler({"command": "echo hi"})
        assert [
            event
            for event in run_store.load_events(agent_id)
            if event.kind == "exec_capped"
        ] == []

    async def test_an_explicit_timeout_is_still_capped_and_recorded(
        self, run_store: RunStore, run_id: str
    ):
        # The other half: intent the agent did express is still bounded,
        # still reported, still telemetry.
        agent_id = run_store.create_agent(run_id, "goal")
        sandbox = FakeExecSandbox()
        tool = bash_tool(
            sandbox,
            deadline=_fixed_deadline(900.0, 200.0),
            store=run_store,
            agent_id=agent_id,
        )
        output = await tool.handler({"command": "echo hi", "timeout": 400})
        assert sandbox.received_timeout == 110.0
        assert "timeout was capped to 110.0s" in output
        (event,) = [
            event
            for event in run_store.load_events(agent_id)
            if event.kind == "exec_capped"
        ]
        assert event.payload["requested"] == 400.0
        assert event.payload["effective"] == 110.0

    async def test_a_quick_copy_still_runs_at_sixty_four_seconds_left(self):
        """The F3 anti-regression: v1 would have refused this.

        At remaining 64s of 900 an agent writing its answer out issues
        ``cp /tmp/answer.txt /app/`` with no timeout. v1's proposal refused
        any exec whose window fell below MIN_USEFUL_EXEC_SECONDS = 5.0,
        keyed on ``requested`` — which here is the harness's own 120s
        default, not intent. The corpus says otherwise: the median capped
        exec ran 0.79s, and this one is a file copy. It must execute.
        """
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(900.0, 64.0, observations=(5.0,) * 4)
        tool = bash_tool(sandbox, deadline=deadline)
        output = await tool.handler({"command": "cp /tmp/answer.txt /app/"})
        # Reserve 75 -> the old floor would have said 30s and eaten the
        # stop floor; 1a says 4s, and 4s copies a file.
        assert sandbox.received_timeout == pytest.approx(4.0)
        assert sandbox.received_timeout > 0.0
        assert "exit code: 0" in output
        assert "capped from your requested" not in output

    async def test_a_shrunk_default_that_times_out_says_why_honestly(self):
        sandbox = FakeExecSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, 200.0))
        output = await tool.handler({"command": "sleep 1000"})
        assert "timed out after 110.0s" in output
        assert "no timeout was given" in output
        assert "capped from your requested" not in output

    async def test_description_mentions_the_adaptive_default(self):
        description = bash_tool(FakeExecSandbox()).spec.description.lower()
        assert "omit it" in description
        assert "remaining wall-clock" in description


class TestBashToolNeverDispatchesAZeroWindow:
    """The exec cap may answer 0.0; the tool must never *run* a 0.0.

    ``affordable_exec_seconds()`` is exactly 0.0 once ``remaining`` reaches
    :data:`WALL_CLOCK_STOP_FLOOR`, which is reachable with the landing band
    disarmed — on a run's first turn the call window is empty by
    construction, and generation can spend the margin between the loop-top
    check and the exec. Dispatching that 0.0 does not decline to run the
    command: it runs it and kills it instantly, so an agent's one-line
    ``cp`` comes back as a *timeout* where before the wind-down work it
    would simply have succeeded. A refusal is at least honest; an instant
    timeout is refusal-by-arithmetic wearing an exec's clothes.
    """

    async def test_an_omitted_timeout_inside_the_stop_floor_still_runs(self):
        # The exact reviewed repro: Deadline(3600) at remaining 59.0, no
        # landing declared, no timeout given.
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(3600.0, 59.0)
        assert deadline.affordable_exec_seconds() == 0.0
        tool = bash_tool(sandbox, deadline=deadline)
        output = await tool.handler({"command": "cp /tmp/answer.txt /app/"})
        assert sandbox.received_timeout == MIN_EXEC_SECONDS
        assert sandbox.received_timeout > 0.0
        assert "exit code: 0" in output
        # Still never reported as a cap on a number the agent never chose.
        assert "capped from your requested" not in output
        assert "shortened to 0.0s" not in output

    async def test_an_explicit_timeout_inside_the_stop_floor_still_runs(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, 59.0))
        output = await tool.handler({"command": "cp a b", "timeout": 30})
        assert sandbox.received_timeout == MIN_EXEC_SECONDS
        # The agent asked for 30 and got 1: that is a real cap and is said.
        assert "timeout was capped to 1.0s" in output

    async def test_the_floor_never_lengthens_an_explicit_short_timeout(self):
        # The floor exists to stop the arithmetic reaching zero, not to
        # overrule an agent that deliberately wants a very short command.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, 59.0))
        await tool.handler({"command": "cp a b", "timeout": 0.25})
        assert sandbox.received_timeout == 0.25

    async def test_a_floor_restored_ask_is_not_reported_as_a_cap(self):
        # The floor lifts an explicit sub-second ask back to exactly what
        # was requested, so `exec_decision` says "capped" while nothing was
        # taken. Saying so would tell the agent its 0.25s was "capped to
        # 0.25s" — and the timed-out spelling contradicts itself inside a
        # single sentence.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, 59.0))
        output = await tool.handler({"command": "cp a b", "timeout": 0.25})
        assert sandbox.received_timeout == 0.25
        assert "capped" not in output
        # A cap that really bit is still reported, unchanged.
        capped_out = await bash_tool(
            FakeExecSandbox(), deadline=_fixed_deadline(3600.0, 59.0)
        ).handler({"command": "cp a b", "timeout": 30})
        assert "timeout was capped to 1.0s" in capped_out

    async def test_a_floor_restored_ask_records_no_exec_capped_event(
        self, run_store: RunStore, run_id: str
    ):
        # The same falsehood as telemetry: a corpus row with
        # requested == effective would mis-calibrate every reader of the
        # exec_capped series.
        agent_id = run_store.create_agent(run_id, "goal")
        tool = bash_tool(
            FakeExecSandbox(),
            deadline=_fixed_deadline(3600.0, 59.0),
            store=run_store,
            agent_id=agent_id,
        )
        await tool.handler({"command": "cp a b", "timeout": 0.25})
        assert [
            event
            for event in run_store.load_events(agent_id)
            if event.kind == "exec_capped"
        ] == []

    @pytest.mark.parametrize("timeout", [0.05, 0.25, 0.5, 0.999, 1.0])
    async def test_no_explicit_sub_second_ask_is_ever_a_false_cap(
        self, timeout: float
    ):
        # Every explicit timeout in (0, MIN_EXEC_SECONDS] is restored by the
        # floor once `remaining` is inside the reserve; none of them may
        # claim a cap. Swept, because the band is a whole interval rather
        # than the one value the repro used.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, 59.0))
        output = await tool.handler({"command": "cp a b", "timeout": timeout})
        assert sandbox.received_timeout == timeout
        assert "capped" not in output

    @pytest.mark.parametrize("remaining", [0.5, 5.0, 30.0, 59.0, 60.0])
    async def test_no_reachable_remaining_produces_a_zero_window(
        self, remaining: float
    ):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, remaining))
        await tool.handler({"command": "ls"})
        assert sandbox.received_timeout >= MIN_EXEC_SECONDS

    @pytest.mark.parametrize("remaining", [60.001, 60.1, 60.5, 60.9, 60.999])
    async def test_the_band_just_above_the_stop_floor_is_not_sub_second(
        self, remaining: float
    ):
        # `remaining` strictly between the stop floor and one second above
        # it clamps to a *positive fraction of a second* — a window no
        # command can use, and a number the agent never chose. Guarding
        # only the exact 0.0 left this one-second-wide band dispatching
        # e.g. 0.0999s, which is the same instant timeout in a different
        # spelling. The floor is MIN_EXEC_SECONDS, as its docstring says.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, remaining))
        await tool.handler({"command": "cp a b"})
        assert sandbox.received_timeout == MIN_EXEC_SECONDS

    @pytest.mark.parametrize("remaining", [0.25, 1.0, 59.5, 60.1, 60.75, 61.5])
    async def test_the_floor_holds_across_the_whole_wind_down_tail(
        self, remaining: float
    ):
        # The invariant stated by MIN_EXEC_SECONDS, swept rather than
        # spot-checked: an omitted timeout is never dispatched below a
        # second, whatever the arithmetic produces.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, remaining))
        await tool.handler({"command": "ls"})
        assert sandbox.received_timeout is not None
        assert sandbox.received_timeout >= MIN_EXEC_SECONDS

    @pytest.mark.parametrize("remaining", [60.1, 60.9])
    async def test_an_explicit_sub_second_ask_is_still_honoured_in_the_band(
        self, remaining: float
    ):
        # The one documented exemption survives the widened floor: a caller
        # that explicitly asks for less than a second gets what it asked
        # for — never less, and never lengthened to the floor.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, remaining))
        await tool.handler({"command": "cp a b", "timeout": 0.25})
        assert sandbox.received_timeout == 0.25

    async def test_a_zero_window_is_still_a_normal_exec_not_a_refusal(self):
        # The landing refusal is the *only* way bash declines to run, and
        # it comes from explicit loop state. Arithmetic never refuses.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(3600.0, 1.0))
        output = await tool.handler({"command": "ls"})
        assert output != LANDING_REFUSAL
        assert sandbox.received_timeout == MIN_EXEC_SECONDS


class TestBashToolLandingRefusal:
    """Change 1c, tool side: the loop's final turn disables the shell.

    The gate is :attr:`Deadline.landing` — explicit state the loop sets —
    and nothing else. No arithmetic on the requested timeout is involved,
    which is what makes the F3 failure (refusing an agent because of a
    number the *harness* supplied) impossible by construction.
    """

    async def test_bash_refuses_without_touching_the_sandbox(self):
        sandbox = FakeExecSandbox()
        deadline = _fixed_deadline(900.0, 62.0)
        deadline.begin_landing()
        tool = bash_tool(sandbox, deadline=deadline)
        output = await tool.handler({"command": "make test", "timeout": 5})
        assert output == LANDING_REFUSAL
        assert sandbox.received_timeout is None

    async def test_the_refusal_tells_the_model_what_to_do_instead(self):
        deadline = _fixed_deadline(900.0, 62.0)
        deadline.begin_landing()
        tool = bash_tool(FakeExecSandbox(), deadline=deadline)
        output = await tool.handler({"command": "ls"})
        assert "final turn" in output
        assert "write_file" in output
        assert "Do not retry" in output

    async def test_the_refusal_is_a_normal_result_not_an_error(self):
        # It must not spend nudge or truncation budget: the landing turn is
        # the one turn the run cannot afford to have re-prompted.
        registry = ToolRegistry()
        deadline = _fixed_deadline(900.0, 62.0)
        deadline.begin_landing()
        registry.register(bash_tool(FakeExecSandbox(), deadline=deadline))
        result = await registry.dispatch(
            ToolCall(id="c1", name="bash", arguments={"command": "ls"})
        )
        assert result.is_error is False
        assert result.content == LANDING_REFUSAL

    async def test_nothing_is_refused_before_the_loop_declares_landing(self):
        sandbox = FakeExecSandbox()
        # Deep inside the stop floor, but the loop has not declared the
        # landing turn: the tool does not decide this for itself. The
        # command runs — in the smallest window the tool will dispatch,
        # never in a 0s one, which would fail it outright.
        tool = bash_tool(sandbox, deadline=_fixed_deadline(900.0, 5.0))
        output = await tool.handler({"command": "ls"})
        assert output != LANDING_REFUSAL
        assert sandbox.received_timeout == MIN_EXEC_SECONDS

    async def test_no_deadline_means_no_gate(self):
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox)
        assert await tool.handler({"command": "ls"}) != LANDING_REFUSAL
        assert sandbox.received_timeout == DEFAULT_EXEC_TIMEOUT

    async def test_a_refusal_records_no_exec_capped_event(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        deadline = _fixed_deadline(900.0, 62.0)
        deadline.begin_landing()
        tool = bash_tool(
            FakeExecSandbox(),
            deadline=deadline,
            store=run_store,
            agent_id=agent_id,
        )
        await tool.handler({"command": "make", "timeout": 600})
        assert [
            event
            for event in run_store.load_events(agent_id)
            if event.kind == "exec_capped"
        ] == []


class TestBashToolExecCappedEvent:
    """X13: every capped exec is telemetry, so round 3 retunes the cap's
    constants from real runs instead of another simulation."""

    def _wired(
        self,
        run_store: RunStore,
        run_id: str,
        deadline,
        result: ExecResult | None = None,
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        sandbox = FakeExecSandbox(result) if result else FakeExecSandbox()
        tool = bash_tool(
            sandbox, deadline=deadline, store=run_store, agent_id=agent_id
        )
        return agent_id, sandbox, tool

    def _events(self, run_store: RunStore, agent_id: str) -> list[dict]:
        return [
            event.payload
            for event in run_store.load_events(agent_id)
            if event.kind == "exec_capped"
        ]

    async def test_payload_carries_every_tuning_field(
        self, run_store: RunStore, run_id: str
    ):
        deadline = _fixed_deadline(1800.0, 1706.9)
        agent_id, _sandbox, tool = self._wired(run_store, run_id, deadline)
        await tool.handler({"command": "R CMD INSTALL rstan", "timeout": 1800})

        (payload,) = self._events(run_store, agent_id)
        assert payload == {
            "requested": 1800.0,
            "effective": 900.0,
            "remaining": pytest.approx(1706.9),
            "budget": 1800.0,
            # The base landing reserve (60 floor + 30 default allowance) and
            # what the cap arithmetic actually held back. They differ here:
            # remaining is above the 360s wind-down threshold, so the band
            # softener raised the reserve to 360 even though the share cap
            # is what ultimately bound.
            "reserve": 90.0,
            "applied_reserve": 360.0,
            "reason": "share",
            "purpose": "exploratory",
        }

    async def test_applied_reserve_is_the_softened_band_reserve(
        self, run_store: RunStore, run_id: str
    ):
        """Regression: the payload reported only the *base* landing reserve,
        so the softener's real contribution — the number round 3 retunes
        LANDING_RESERVE_FRACTION from — was absent. It is not always
        recoverable either: for reason "band"/"reserve" it equals
        remaining - effective, but for "share" nothing in the payload
        determines it."""
        # The caffe-cifar-10 row: budget 1200, remaining 619.7, requested
        # 600 on a fast provider (reserve floor 60 + allowance 15 = 75).
        # Threshold is 240; 0.25 x 619.7 = 154.925 softens the band down to
        # it, so 154.925 — not 75 — is what was held back.
        deadline = _fixed_deadline(
            1200.0, 619.7, observations=(15.0, 15.0, 15.0, 15.0)
        )
        agent_id, _sandbox, tool = self._wired(run_store, run_id, deadline)
        await tool.handler({"command": "./train.sh", "timeout": 600})

        (payload,) = self._events(run_store, agent_id)
        assert payload["reason"] == "band"
        assert payload["reserve"] == 75.0  # base: floor + adaptive allowance
        assert payload["applied_reserve"] == pytest.approx(154.925)
        assert payload["effective"] == pytest.approx(464.775)
        # Consistent with the arithmetic the consumer can see.
        assert payload["remaining"] - payload["effective"] == pytest.approx(
            payload["applied_reserve"]
        )

    async def test_applied_reserve_is_the_base_reserve_when_in_band(
        self, run_store: RunStore, run_id: str
    ):
        """Below the wind-down threshold the softener never fires, so the
        two reserve fields agree — that agreement is what tells round 3 the
        softener was not involved."""
        deadline = _fixed_deadline(900.0, 200.0)  # 200 < the 300s threshold
        agent_id, _sandbox, tool = self._wired(run_store, run_id, deadline)
        await tool.handler({"command": "make", "timeout": 300})

        (payload,) = self._events(run_store, agent_id)
        assert payload["reason"] == "reserve"
        assert payload["reserve"] == 90.0
        assert payload["applied_reserve"] == 90.0

    @pytest.mark.parametrize(
        "budget,remaining,requested,reason",
        [
            (1800.0, 1706.9, 1800.0, "share"),
            (900.0, 335.5, 300.0, "band"),
            (900.0, 200.0, 300.0, "reserve"),
        ],
    )
    async def test_reason_is_recorded_for_each_bound(
        self,
        run_store: RunStore,
        run_id: str,
        budget: float,
        remaining: float,
        requested: float,
        reason: str,
    ):
        deadline = _fixed_deadline(
            budget, remaining, observations=(5.0, 5.0, 5.0, 5.0)
        )
        agent_id, _sandbox, tool = self._wired(run_store, run_id, deadline)
        await tool.handler({"command": "make", "timeout": requested})
        assert self._events(run_store, agent_id)[0]["reason"] == reason

    async def test_written_even_when_the_command_then_succeeds(
        self, run_store: RunStore, run_id: str
    ):
        # The cap's cost is what we are measuring, not the command's fate.
        deadline = _fixed_deadline(900.0, 200.0)
        agent_id, _sandbox, tool = self._wired(
            run_store,
            run_id,
            deadline,
            ExecResult(exit_code=0, stdout="done", stderr=""),
        )
        await tool.handler({"command": "make", "timeout": 120})
        assert len(self._events(run_store, agent_id)) == 1

    async def test_nothing_written_when_the_cap_does_not_bite(
        self, run_store: RunStore, run_id: str
    ):
        deadline = _fixed_deadline(10_000.0)
        agent_id, _sandbox, tool = self._wired(run_store, run_id, deadline)
        await tool.handler({"command": "echo hi", "timeout": 45})
        assert self._events(run_store, agent_id) == []

    async def test_no_deadline_emits_nothing(
        self, run_store: RunStore, run_id: str
    ):
        agent_id, sandbox, tool = self._wired(run_store, run_id, None)
        await tool.handler({"command": "echo hi", "timeout": 45})
        assert sandbox.received_timeout == 45
        assert self._events(run_store, agent_id) == []

    async def test_the_default_toolset_wires_store_and_agent_id(
        self,
        memory_store: MemoryStore,
        skill_library: SkillLibrary,
        run_store: RunStore,
        run_id: str,
    ):
        """The telemetry is only worth anything if the real registry build
        carries it — and to the *owning agent's* stream, not the run's."""
        from harness.orchestrator import CODING_TOOL_FACTORIES, ToolDeps

        agent_id = run_store.create_agent(run_id, "goal")
        sandbox = FakeExecSandbox()
        deps = ToolDeps(
            sandbox=sandbox,
            memory=memory_store,
            skills=skill_library,
            store=run_store,
            run_id=run_id,
            agent_id=agent_id,
            context=_make_context(),
            deadline=_fixed_deadline(1800.0, 1706.9),
        )
        registry = ToolRegistry()
        for factory in CODING_TOOL_FACTORIES:
            registry.register(factory(deps))
        await registry.get("bash").handler(
            {"command": "R CMD INSTALL rstan", "timeout": 1800}
        )

        assert sandbox.received_timeout == 900.0
        (payload,) = self._events(run_store, agent_id)
        assert payload["reason"] == "share"

    async def test_unwired_store_still_caps(self):
        # store/agent_id are optional: the cap is the contract, the event
        # is telemetry on top of it.
        sandbox = FakeExecSandbox()
        tool = bash_tool(sandbox, deadline=_fixed_deadline(1800.0, 1706.9))
        await tool.handler({"command": "make", "timeout": 1800})
        assert sandbox.received_timeout == 900.0


class TestReadWriteEditFileTools:
    def test_read_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = read_file_tool(sandbox)
        assert tool.spec.name == "read_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    def test_write_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = write_file_tool(sandbox)
        assert tool.spec.name == "write_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {"path", "content"}
        assert tool.meta.side_effect is False

    def test_edit_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = edit_file_tool(sandbox)
        assert tool.spec.name == "edit_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "path",
            "old_string",
            "new_string",
        }
        assert tool.meta.side_effect is False

    async def test_write_then_read_round_trip(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        read = read_file_tool(sandbox)
        write_result = await write.handler({"path": "notes.txt", "content": "hello world"})
        assert "notes.txt" in write_result
        assert await read.handler({"path": "notes.txt"}) == "hello world"

    async def test_read_missing_file_raises_sandbox_error(self, sandbox: LocalSandbox):
        await sandbox.start()
        tool = read_file_tool(sandbox)
        with pytest.raises(SandboxError):
            await tool.handler({"path": "nope.txt"})

    async def test_edit_file_applies_unique_replacement(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        read = read_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "x = 1\ny = 2\n"})
        await edit.handler({"path": "f.py", "old_string": "x = 1", "new_string": "x = 100"})
        assert await read.handler({"path": "f.py"}) == "x = 100\ny = 2\n"

    async def test_edit_file_replace_all(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        read = read_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "dup\ndup\n"})
        await edit.handler(
            {"path": "f.py", "old_string": "dup", "new_string": "one", "replace_all": True}
        )
        assert await read.handler({"path": "f.py"}) == "one\none\n"

    async def test_edit_file_ambiguous_match_raises(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "dup\ndup\n"})
        with pytest.raises(SandboxError, match="not unique"):
            await edit.handler({"path": "f.py", "old_string": "dup", "new_string": "one"})

    async def test_write_file_missing_content_raises(self, sandbox: LocalSandbox):
        tool = write_file_tool(sandbox)
        with pytest.raises(MissingArgumentError):
            await tool.handler({"path": "f.txt"})

    async def test_dispatch_error_on_path_traversal(self, sandbox: LocalSandbox):
        registry = ToolRegistry()
        registry.register(read_file_tool(sandbox))
        result = await registry.dispatch(
            _call("read_file", {"path": "../outside.txt"})
        )
        assert result.is_error is True


class TestReadFileRangeAndPattern:
    """DESIGN.md §10.2 A4: `read_file` gains `offset`/`limit`/`pattern`."""

    _CONTENT = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"  # line1..line10

    @pytest.fixture
    async def populated(self, sandbox: LocalSandbox) -> LocalSandbox:
        await write_file_tool(sandbox).handler(
            {"path": "f.txt", "content": self._CONTENT}
        )
        return sandbox

    def test_schema_declares_offset_limit_pattern_as_optional(
        self, sandbox: LocalSandbox
    ):
        tool = read_file_tool(sandbox)
        _assert_valid_object_schema(tool.spec.input_schema)
        props = tool.spec.input_schema["properties"]
        assert {"offset", "limit", "pattern"} <= props.keys()
        # Only `path` is required -- the new arguments are all optional.
        assert tool.spec.input_schema["required"] == ["path"]

    async def test_no_range_args_is_unchanged_whole_file_read(
        self, populated: LocalSandbox
    ):
        """Backward compatibility: omitting offset/limit/pattern must return
        the exact, unmodified file content -- no line numbers, no footer."""
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt"})
        assert result == self._CONTENT

    async def test_offset_and_limit_return_numbered_line_range(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "offset": 3, "limit": 2})
        assert "3:line3" in result
        assert "4:line4" in result
        assert "line2" not in result
        assert "line5" not in result
        assert "lines 3-4 of 10" in result

    async def test_offset_without_limit_reads_to_end_of_file(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "offset": 9})
        assert "9:line9" in result
        assert "10:line10" in result
        assert "line8" not in result
        assert "lines 9-10 of 10" in result

    async def test_limit_without_offset_defaults_to_start_of_file(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "limit": 2})
        assert "1:line1" in result
        assert "2:line2" in result
        assert "line3" not in result

    async def test_offset_beyond_end_of_file_reports_clearly_not_a_crash(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "offset": 100})
        assert "100" in result
        assert "10" in result  # total line count is reported

    async def test_offset_zero_is_a_clear_error(self, populated: LocalSandbox):
        tool = read_file_tool(populated)
        with pytest.raises(ValueError, match="offset"):
            await tool.handler({"path": "f.txt", "offset": 0})

    async def test_limit_zero_is_a_clear_error(self, populated: LocalSandbox):
        tool = read_file_tool(populated)
        with pytest.raises(ValueError, match="limit"):
            await tool.handler({"path": "f.txt", "limit": 0})

    async def test_pattern_returns_matching_lines_grep_n_style(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "pattern": r"line[13]$"})
        assert "1:line1" in result
        assert "3:line3" in result
        assert "line2" not in result
        assert "match(es)" in result

    async def test_pattern_no_match_returns_clear_message_not_error(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        result = await tool.handler({"path": "f.txt", "pattern": "nope-not-here"})
        assert "no lines" in result
        assert "nope-not-here" in result

    async def test_pattern_within_offset_limit_window(self, populated: LocalSandbox):
        """`pattern` composes with `offset`/`limit`: it greps only within the
        given range, and reported line numbers stay absolute."""
        tool = read_file_tool(populated)
        result = await tool.handler(
            {"path": "f.txt", "offset": 1, "limit": 3, "pattern": "line3"}
        )
        assert "3:line3" in result
        # line5 also matches /line\d/ loosely but is outside the window and
        # must not appear.
        await tool.handler({"path": "f.txt", "offset": 5, "limit": 1})  # sanity call
        assert "match(es) among lines 1-3 of 10" in result

    async def test_invalid_regex_is_a_clear_error_not_a_crash(
        self, populated: LocalSandbox
    ):
        tool = read_file_tool(populated)
        with pytest.raises(ValueError, match="regex"):
            await tool.handler({"path": "f.txt", "pattern": "(unclosed["})

    async def test_invalid_regex_via_dispatch_is_error_result_not_a_crash(
        self, populated: LocalSandbox
    ):
        registry = ToolRegistry()
        registry.register(read_file_tool(populated))
        result = await registry.dispatch(
            _call("read_file", {"path": "f.txt", "pattern": "(unclosed["})
        )
        assert result.is_error is True
        assert "regex" in result.content


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


class TestMemoryTools:
    def test_read_fact_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_read_fact_tool(memory_store)
        assert tool.spec.name == "memory_read_fact"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    def test_write_fact_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_write_fact_tool(memory_store)
        assert tool.spec.name == "memory_write_fact"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "name",
            "description",
            "type",
            "body",
        }
        assert tool.meta.side_effect is False

    def test_search_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_search_tool(memory_store)
        assert tool.spec.name == "memory_search"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    async def test_write_then_read_round_trip(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        read = memory_read_fact_tool(memory_store)
        write_result = await write.handler(
            {
                "name": "prefers-dark-mode",
                "description": "User prefers dark mode",
                "type": "user",
                "body": "Observed across sessions.",
                "sources": ["2026-07-19-onboarding.md"],
            }
        )
        assert "prefers-dark-mode" in write_result
        read_result = await read.handler({"name": "prefers-dark-mode"})
        assert "prefers-dark-mode" in read_result
        assert "Observed across sessions." in read_result

    async def test_read_unknown_fact_raises(self, memory_store: MemoryStore):
        tool = memory_read_fact_tool(memory_store)
        with pytest.raises(FactNotFoundError):
            await tool.handler({"name": "no-such-fact"})

    async def test_search_finds_written_fact(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        search = memory_search_tool(memory_store)
        await write.handler(
            {
                "name": "likes-tea",
                "description": "User likes tea",
                "type": "user",
                "body": "Drinks green tea every morning.",
            }
        )
        result = await search.handler({"query": "green tea"})
        assert "likes-tea" in result
        assert "green tea" in result

    async def test_search_no_matches(self, memory_store: MemoryStore):
        search = memory_search_tool(memory_store)
        result = await search.handler({"query": "nonexistent-topic"})
        assert "no matches" in result

    async def test_write_fact_invalid_type_raises(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        with pytest.raises(Exception):
            await write.handler(
                {
                    "name": "bad-fact",
                    "description": "desc",
                    "type": "not-a-real-type",
                    "body": "body",
                }
            )

    async def test_dispatch_end_to_end(self, memory_store: MemoryStore):
        registry = ToolRegistry()
        registry.register(memory_write_fact_tool(memory_store))
        registry.register(memory_read_fact_tool(memory_store))
        await registry.dispatch(
            _call(
                "memory_write_fact",
                {
                    "name": "test-fact",
                    "description": "desc",
                    "type": "project",
                    "body": "body text",
                },
            )
        )
        result = await registry.dispatch(
            _call("memory_read_fact", {"name": "test-fact"}, call_id="call-2")
        )
        assert result.is_error is False
        assert "body text" in result.content


# ---------------------------------------------------------------------------
# Task ledger tools
# ---------------------------------------------------------------------------


class TestTaskLedgerTools:
    def test_task_update_schema_and_meta(self, run_store: RunStore, run_id: str):
        tool = task_update_tool(run_store, run_id)
        assert tool.spec.name == "task_update"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "item_id",
            "description",
            "status",
        }
        assert tool.meta.side_effect is False

    def test_task_list_schema_and_meta(self, run_store: RunStore, run_id: str):
        tool = task_list_tool(run_store, run_id)
        assert tool.spec.name == "task_list"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.spec.input_schema["required"] == []
        assert tool.meta.side_effect is False

    async def test_list_empty_ledger(self, run_store: RunStore, run_id: str):
        tool = task_list_tool(run_store, run_id)
        result = await tool.handler({})
        assert "empty" in result

    async def test_update_then_list_round_trip(self, run_store: RunStore, run_id: str):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {
                "item_id": "step-1",
                "description": "Write the tests",
                "status": "in_progress",
            }
        )
        result = await listing.handler({})
        assert "step-1" in result
        assert "Write the tests" in result
        assert "in_progress" in result

    async def test_update_with_evidence_shown_in_listing(
        self, run_store: RunStore, run_id: str
    ):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {
                "item_id": "step-1",
                "description": "Run tests",
                "status": "done",
                "evidence": "pytest: 42 passed",
            }
        )
        result = await listing.handler({})
        assert "pytest: 42 passed" in result

    async def test_repeated_update_overwrites_item(
        self, run_store: RunStore, run_id: str
    ):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {"item_id": "step-1", "description": "first", "status": "pending"}
        )
        await update.handler(
            {"item_id": "step-1", "description": "first", "status": "done"}
        )
        result = await listing.handler({})
        assert result.count("step-1") == 1
        assert "done" in result

    async def test_dispatch_end_to_end(self, run_store: RunStore, run_id: str):
        registry = ToolRegistry()
        registry.register(task_update_tool(run_store, run_id))
        registry.register(task_list_tool(run_store, run_id))
        await registry.dispatch(
            _call(
                "task_update",
                {"item_id": "a", "description": "d", "status": "pending"},
            )
        )
        result = await registry.dispatch(_call("task_list", {}, call_id="call-2"))
        assert result.is_error is False
        assert "a" in result.content


# ---------------------------------------------------------------------------
# Skill tool
# ---------------------------------------------------------------------------


class TestLoadSkillTool:
    def test_schema_and_meta(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        assert tool.spec.name == "load_skill"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.spec.input_schema["required"] == ["name"]
        assert tool.meta.side_effect is False

    async def test_loads_full_body(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        body = await tool.handler({"name": "greet"})
        assert body == "Say hello and ask how you can help."

    async def test_unknown_skill_raises_keyerror(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        with pytest.raises(KeyError):
            await tool.handler({"name": "does-not-exist"})

    async def test_dispatch_unknown_skill_is_error_result(
        self, skill_library: SkillLibrary
    ):
        registry = ToolRegistry()
        registry.register(load_skill_tool(skill_library))
        result = await registry.dispatch(
            _call("load_skill", {"name": "does-not-exist"})
        )
        assert result.is_error is True
        assert "does-not-exist" in result.content


# ---------------------------------------------------------------------------
# All builtins together
# ---------------------------------------------------------------------------


class TestAllBuiltinsTogether:
    def test_register_all_builtins_without_name_collisions(
        self,
        sandbox: LocalSandbox,
        memory_store: MemoryStore,
        run_store: RunStore,
        run_id: str,
        skill_library: SkillLibrary,
    ):
        registry = ToolRegistry()
        tools = [
            bash_tool(sandbox),
            read_file_tool(sandbox),
            write_file_tool(sandbox),
            edit_file_tool(sandbox),
            memory_read_fact_tool(memory_store),
            memory_write_fact_tool(memory_store),
            memory_search_tool(memory_store),
            task_update_tool(run_store, run_id),
            task_list_tool(run_store, run_id),
            load_skill_tool(skill_library),
            add_instruction_tool(run_store, run_id),
            search_history_tool(run_store, run_id),
        ]
        for tool in tools:
            registry.register(tool)
        specs = registry.specs()
        names = [spec.name for spec in specs]
        assert len(names) == len(set(names)) == 12
        for spec in specs:
            _assert_valid_object_schema(spec.input_schema)
        for tool in tools:
            assert tool.meta.side_effect is False


# ---------------------------------------------------------------------------
# Context binding: skill splicing, task snapshot, instruction ledger
# ---------------------------------------------------------------------------


class TestContextBoundTools:
    async def test_load_skill_with_context_splices_body_into_system_prompt(
        self, skill_library: SkillLibrary
    ):
        """Regression (DESIGN.md §4.6): with a bound context the skill body
        rides the system prompt — exempt from tool-result pruning — and the
        tool result is a short acknowledgment, not the body."""
        context = _make_context()
        tool = load_skill_tool(skill_library, context)
        ack = await tool.handler({"name": "greet"})
        assert "greet" in ack
        assert "system prompt" in ack
        assert "Say hello and ask how you can help." not in ack
        system, _ = context.assemble()
        assert "Say hello and ask how you can help." in system
        assert "## Loaded skill: greet" in system

    async def test_load_skill_without_context_returns_body(
        self, skill_library: SkillLibrary
    ):
        tool = load_skill_tool(skill_library)
        assert (
            await tool.handler({"name": "greet"})
            == "Say hello and ask how you can help."
        )

    async def test_task_update_with_context_refreshes_snapshot(
        self, run_store: RunStore, run_id: str
    ):
        """Regression (DESIGN.md §4.9): task updates mirror the live ledger
        into the context's trailing reminder."""
        context = _make_context(reminder_interval=1)
        tool = task_update_tool(run_store, run_id, context)
        await tool.handler(
            {
                "item_id": "step-1",
                "description": "Write the tests",
                "status": "in_progress",
            }
        )
        context.append(Message(role=Role.USER, content="goal"))
        context.append(Message(role=Role.ASSISTANT, content="turn 1"))
        _, messages = context.assemble()
        reminder = messages[-1].content or ""
        assert reminder.startswith("<system-reminder>")
        assert "Current task ledger:" in reminder
        assert "- [in_progress] step-1: Write the tests" in reminder

    async def test_add_instruction_persists_and_joins_context_ledger(
        self, run_store: RunStore, run_id: str
    ):
        """Regression (DESIGN.md §4.5): recorded instructions land in the
        instruction_ledger table and the context's reminder ledger."""
        context = _make_context(reminder_interval=1)
        tool = add_instruction_tool(run_store, run_id, context)
        assert tool.spec.name == "add_instruction"
        assert tool.meta.side_effect is False
        result = await tool.handler({"instruction": "never push to main"})
        assert "never push to main" in result

        items = run_store.list_instructions(run_id)
        assert len(items) == 1
        assert items[0].instruction == "never push to main"
        assert items[0].source == "user"
        assert items[0].item_id == "instr-1"

        context.append(Message(role=Role.USER, content="goal"))
        context.append(Message(role=Role.ASSISTANT, content="turn 1"))
        _, messages = context.assemble()
        assert "never push to main" in (messages[-1].content or "")

    async def test_add_instruction_auto_ids_do_not_collide(
        self, run_store: RunStore, run_id: str
    ):
        tool = add_instruction_tool(run_store, run_id)
        await tool.handler({"instruction": "first"})
        await tool.handler({"instruction": "second", "source": "task"})
        items = run_store.list_instructions(run_id)
        assert [(i.item_id, i.instruction) for i in items] == [
            ("instr-1", "first"),
            ("instr-2", "second"),
        ]
        assert items[1].source == "task"


# ---------------------------------------------------------------------------
# search_history tool (the §4.3 layer-4 retrieval backstop)
# ---------------------------------------------------------------------------


class TestSearchHistoryTool:
    @pytest.fixture
    def agent_id(self, run_store: RunStore, run_id: str) -> str:
        return run_store.create_agent(run_id, "test agent")

    async def test_finds_persisted_tool_output(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        """Regression (DESIGN.md §4.3.4): content evicted from context is
        still reachable through the run's event log."""
        run_store.append_event(
            agent_id,
            "tool_result",
            {
                "tool_call_id": "c1",
                "content": "the secret port is 54321",
                "is_error": False,
            },
        )
        tool = search_history_tool(run_store, run_id)
        assert tool.spec.name == "search_history"
        result = await tool.handler({"query": "secret port"})
        assert "54321" in result
        assert "tool_result" in result
        assert f"agent {agent_id}" in result

    async def test_search_is_case_insensitive_and_reports_no_matches(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        run_store.append_event(
            agent_id, "message", {"role": "user", "content": "Deploy THE WIDGET"}
        )
        tool = search_history_tool(run_store, run_id)
        assert "widget" in (await tool.handler({"query": "the widget"})).lower()
        assert "no matches" in await tool.handler({"query": "zebra"})

    async def test_limit_caps_matches_but_reports_total(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        for i in range(5):
            run_store.append_event(
                agent_id, "message", {"role": "user", "content": f"needle {i}"}
            )
        tool = search_history_tool(run_store, run_id)
        result = await tool.handler({"query": "needle", "limit": 2})
        assert "5 match(es)" in result
        assert "showing first 2" in result
        assert result.count("[agent") == 2

    async def test_registered_dispatch_round_trip(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        run_store.append_event(
            agent_id, "message", {"role": "user", "content": "haystack needle"}
        )
        registry = ToolRegistry()
        registry.register(search_history_tool(run_store, run_id))
        result = await registry.dispatch(
            _call("search_history", {"query": "needle"})
        )
        assert result.is_error is False
        assert "needle" in result.content


# ---------------------------------------------------------------------------
# declare_verification tool (DESIGN.md §10.3 B1)
# ---------------------------------------------------------------------------


class TestDeclareVerificationTool:
    def test_schema_and_meta(self):
        tool = declare_verification_tool()
        assert tool.spec.name == "declare_verification"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "command",
            "description",
        }
        assert tool.meta.side_effect is False

    def test_description_tells_the_model_to_declare_early(self):
        tool = declare_verification_tool()
        assert "early" in tool.spec.description.lower()

    async def test_handler_acknowledges_the_declaration(self):
        tool = declare_verification_tool()
        result = await tool.handler(
            {"command": "pytest -q", "description": "tests pass"}
        )
        assert "pytest -q" in result
        assert "tests pass" in result
        assert "exit 0" in result

    async def test_missing_command_raises(self):
        tool = declare_verification_tool()
        with pytest.raises(MissingArgumentError, match="command"):
            await tool.handler({"description": "tests pass"})

    async def test_missing_description_raises(self):
        tool = declare_verification_tool()
        with pytest.raises(MissingArgumentError, match="description"):
            await tool.handler({"command": "pytest -q"})

    async def test_blank_command_is_rejected(self):
        tool = declare_verification_tool()
        with pytest.raises(ValueError, match="non-empty"):
            await tool.handler({"command": "   ", "description": "d"})

    async def test_dispatch_error_result_not_exception(self):
        registry = ToolRegistry()
        registry.register(declare_verification_tool())
        result = await registry.dispatch(_call("declare_verification", {}))
        assert result.is_error is True
        assert "command" in result.content

    def test_registered_in_default_coding_toolset(
        self,
        sandbox: LocalSandbox,
        memory_store: MemoryStore,
        skill_library: SkillLibrary,
        run_store: RunStore,
        run_id: str,
    ):
        """B1 wiring: the default CODING_TOOL_FACTORIES build a registry
        that contains declare_verification."""
        from harness.orchestrator import CODING_TOOL_FACTORIES, ToolDeps

        deps = ToolDeps(
            sandbox=sandbox,
            memory=memory_store,
            skills=skill_library,
            store=run_store,
            run_id=run_id,
            agent_id=run_store.create_agent(run_id, "goal"),
            context=_make_context(),
        )
        registry = ToolRegistry()
        for factory in CODING_TOOL_FACTORIES:
            registry.register(factory(deps))
        names = [spec.name for spec in registry.specs()]
        assert "declare_verification" in names


# ---------------------------------------------------------------------------
# declare_verification: verification-quality lint (Change 4) — warn-only
# ---------------------------------------------------------------------------


class TestDeclareVerificationLint:
    """The tool appends advisories and emits ``verification_lint``.

    Every assertion here also pins the warn-only contract: the declaration
    is acknowledged, the result is not an error, and nothing is rejected.
    """

    @staticmethod
    def _written(*commands: str) -> WrittenData:
        written = WrittenData()
        for command in commands:
            record_written_data(written, "bash", {"command": command})
        return written

    @staticmethod
    def _lint_events(run_store: RunStore, agent_id: str) -> list[dict]:
        return [
            event.payload
            for event in run_store.load_events(agent_id)
            if event.kind == "verification_lint"
        ]

    async def test_clean_command_is_unchanged_and_emits_no_event(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        written = self._written('echo "DISCRIMINATING_VALUE" > /app/other.txt')
        tool = declare_verification_tool(
            written_data=lambda: written,
            store=run_store,
            agent_id=agent_id,
        )
        result = await tool.handler(
            {"command": "pytest -q", "description": "tests pass"}
        )
        assert "Advisory" not in result
        assert self._lint_events(run_store, agent_id) == []

    async def test_tautology_advisory_and_event(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        written = self._written(
            'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'
        )
        tool = declare_verification_tool(
            written_data=lambda: written,
            store=run_store,
            agent_id=agent_id,
        )
        command = "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt"
        result = await tool.handler(
            {"command": command, "description": "the answer is right"}
        )
        # Warn-only: the declaration still stands, in full.
        assert "verification declared" in result
        assert command in result
        assert "Advisory" in result

        (payload,) = self._lint_events(run_store, agent_id)
        assert payload["command"] == command
        assert payload["action"] == "warn"
        # The echoed file is both the source of the literal (tautology)
        # and a file nothing ever ran to produce (no_execution, T1).
        assert [f["kind"] for f in payload["findings"]] == [
            "tautology",
            "no_execution",
        ]

    async def test_neutralized_exit_is_warned_not_rejected(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        tool = declare_verification_tool(
            store=run_store, agent_id=agent_id
        )
        registry = ToolRegistry()
        registry.register(tool)
        result = await registry.dispatch(
            _call(
                "declare_verification",
                {"command": "pytest -q || true", "description": "tests pass"},
            )
        )
        assert result.is_error is False
        assert "verification declared" in result.content
        (payload,) = self._lint_events(run_store, agent_id)
        assert payload["findings"][0]["kind"] == "neutralized_exit"
        assert payload["findings"][0]["details"]["terminal"] is True

    async def test_non_terminal_neutralizer_is_also_recorded(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        tool = declare_verification_tool(store=run_store, agent_id=agent_id)
        result = await tool.handler(
            {
                "command": "pkill -f server || true; pytest -q",
                "description": "tests pass with the server down",
            }
        )
        assert "verification declared" in result
        (payload,) = self._lint_events(run_store, agent_id)
        details = payload["findings"][0]["details"]
        assert details["terminal"] is False
        assert details["token_index"] == 3

    async def test_redeclaration_is_linted_independently(
        self, run_store: RunStore, run_id: str
    ):
        agent_id = run_store.create_agent(run_id, "goal")
        tool = declare_verification_tool(store=run_store, agent_id=agent_id)
        await tool.handler(
            {"command": "pytest -q || true", "description": "first"}
        )
        await tool.handler({"command": "pytest -q", "description": "second"})
        await tool.handler(
            {"command": "make check; exit 0", "description": "third"}
        )
        payloads = self._lint_events(run_store, agent_id)
        # The clean redeclaration emits nothing; the other two each emit.
        assert [payload["command"] for payload in payloads] == [
            "pytest -q || true",
            "make check; exit 0",
        ]

    async def test_lint_is_silent_without_an_accessor(
        self, run_store: RunStore, run_id: str
    ):
        # The tautology detector needs written_data; without it the tool
        # behaves exactly as it did before Change 4.
        agent_id = run_store.create_agent(run_id, "goal")
        tool = declare_verification_tool(store=run_store, agent_id=agent_id)
        result = await tool.handler(
            {
                "command": "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/r.txt",
                "description": "the answer is right",
            }
        )
        assert "Advisory" not in result
        assert self._lint_events(run_store, agent_id) == []

    async def test_advisory_without_a_store_still_reaches_the_model(self):
        written = self._written(
            'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'
        )
        tool = declare_verification_tool(written_data=lambda: written)
        result = await tool.handler(
            {
                "command": "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt",
                "description": "the answer is right",
            }
        )
        assert "Advisory" in result

    async def test_accessor_returning_none_is_tolerated(self):
        tool = declare_verification_tool(written_data=lambda: None)
        result = await tool.handler(
            {"command": "pytest -q", "description": "tests pass"}
        )
        assert "verification declared" in result

    def test_default_toolset_threads_the_accessor(
        self,
        sandbox: LocalSandbox,
        memory_store: MemoryStore,
        skill_library: SkillLibrary,
        run_store: RunStore,
        run_id: str,
    ):
        """Change 4 wiring: ToolDeps carries the written-data accessor."""
        from harness.orchestrator import CODING_TOOL_FACTORIES, ToolDeps

        written = WrittenData()
        deps = ToolDeps(
            sandbox=sandbox,
            memory=memory_store,
            skills=skill_library,
            store=run_store,
            run_id=run_id,
            agent_id=run_store.create_agent(run_id, "goal"),
            context=_make_context(),
            written_data=lambda: written,
        )
        registry = ToolRegistry()
        for factory in CODING_TOOL_FACTORIES:
            registry.register(factory(deps))
        assert "declare_verification" in [
            spec.name for spec in registry.specs()
        ]

    async def test_default_toolset_lints_against_the_live_map(
        self,
        sandbox: LocalSandbox,
        memory_store: MemoryStore,
        skill_library: SkillLibrary,
        run_store: RunStore,
        run_id: str,
    ):
        """The accessor is read at call time, so writes recorded after the
        registry was built are still linted against."""
        from harness.orchestrator import CODING_TOOL_FACTORIES, ToolDeps

        written = WrittenData()
        agent_id = run_store.create_agent(run_id, "goal")
        deps = ToolDeps(
            sandbox=sandbox,
            memory=memory_store,
            skills=skill_library,
            store=run_store,
            run_id=run_id,
            agent_id=agent_id,
            context=_make_context(),
            written_data=lambda: written,
        )
        registry = ToolRegistry()
        for factory in CODING_TOOL_FACTORIES:
            registry.register(factory(deps))
        # Recorded *after* the registry exists.
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        result = await registry.dispatch(
            _call(
                "declare_verification",
                {
                    "command": (
                        "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt"
                    ),
                    "description": "the answer is right",
                },
            )
        )
        assert result.is_error is False
        assert "Advisory" in result.content
        assert self._lint_events(run_store, agent_id)
