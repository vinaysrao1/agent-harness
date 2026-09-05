"""S-105: the condenser seam.

Compaction was the only subsystem in the harness with no seam and no
measurement. A 100+ step run compacts five to ten times and each compaction
summarizes the last one's output, so quality compounds multiplicatively --
and nothing anywhere read the summary back.

Three claims are tested here:

1. Extracting the strategy changed nothing. The corpus asserts that
   separately (`tests/conformance/test_n7_n8.py`, zero drift); these are the
   unit-level twin.
2. The transcript is no longer rewritten, and every index-bearing reader
   moved to the condensed view rather than half of them.
3. The strategy is swappable and names itself in the event log.

S-404 measured the only retention strategy this spec shipped
(`PivotalCondenser`, marker-based) at **0% recall** against a 25% loss rate.
It was deleted, and so was the retention plumbing behind it once mutation
testing showed every line of that could be removed with the full suite still
green. What is left is the seam, the non-destructive transcript, and
`effective_size`.
"""

from __future__ import annotations

import pytest

from harness.condenser import (
    COMPACTION_SUMMARY_PREFIX,
    CondenseContext,
    Condensation,
    DefaultCondenser,
    condenser_for,
)
from harness.context import ContextManager
from harness.types import Message, Role, ToolCall, ToolResult


async def _summary(messages: list[Message]) -> str:
    return f"STUB SUMMARY of {len(messages)} messages"


def user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def assistant(text: str, calls: list[ToolCall] | None = None) -> Message:
    return Message(role=Role.ASSISTANT, content=text, tool_calls=calls or [])


def tool(call_id: str, content: str, is_error: bool = False) -> Message:
    return Message(
        role=Role.TOOL,
        tool_result=ToolResult(
            tool_call_id=call_id, content=content, is_error=is_error
        ),
    )


def _size(message: Message) -> int:
    result = message.tool_result
    return len(message.content or "") + (len(result.content) if result else 0)


def text_of(messages: list[Message]) -> str:
    """Every message's text, tool results included."""
    parts = []
    for message in messages:
        parts.append(message.content or "")
        if message.tool_result is not None:
            parts.append(message.tool_result.content)
    return "\n".join(parts)


class _NamesItself:
    """A stand-in strategy, to prove the seam actually swaps.

    It used to retain turns as well -- the seam once carried
    `Condensation.kept_refs` and the plumbing to apply it. S-404 killed the
    only strategy that used that (0% recall against a 25% loss rate), and
    mutation testing then showed every line of the plumbing could be deleted
    with the whole suite still green. It was removed rather than propped up
    with tests that exercised nothing else.
    """

    strategy_id = "names-itself"

    async def condense(self, span, ctx):
        return Condensation(
            summary=f"[COMPACTION SUMMARY]\n{ctx.goal}\n---\nS{len(span)}",
            strategy_id=self.strategy_id,
        )


def make_cm(condenser=None, max_context: int = 100_000) -> ContextManager:
    return ContextManager(
        base_system_prompt="SYS",
        # Tool results carry their text in `tool_result.content`, not
        # `content`. Counting only `content` made every tool-heavy transcript
        # weigh nothing, so the pruning test silently measured an empty plan.
        count_tokens=lambda msgs: sum(_size(m) for m in msgs) // 4,
        max_context=max_context,
        summarize=_summary,
        reminder_interval=50,
        condenser=condenser,
    )


class TestTheDefaultIsTodaysBehaviour:
    """Acceptance 1. The corpus proves zero drift over 128K-window traces;
    this proves the bytes at the unit level, where a diff is readable."""

    async def test_S105_the_summary_header_is_unchanged(self) -> None:
        condenser = DefaultCondenser(_summary)
        result = await condenser.condense(
            [user("a"), user("b")],
            CondenseContext(goal="THE GOAL", refs=(1, 2)),
        )
        assert result.summary == (
            f"{COMPACTION_SUMMARY_PREFIX}\n"
            "Original goal (verbatim, never summarized):\n"
            "THE GOAL\n"
            "---\n"
            "STUB SUMMARY of 2 messages"
        )

    async def test_S105_the_default_is_what_a_bare_manager_uses(self) -> None:
        cm = make_cm()
        assert isinstance(cm.condenser, DefaultCondenser)
        assert cm.condenser.strategy_id == "summarize-halve"

    async def test_S105_the_goal_never_rides_on_the_summarizer(self) -> None:
        # The header is built by `_header`, not by the summarize callable, so
        # a summarizer that returns the empty string still carries the goal.
        async def useless(_messages: list[Message]) -> str:
            return ""

        result = await DefaultCondenser(useless).condense(
            [user("a")], CondenseContext(goal="DO NOT TOUCH main", refs=(1,))
        )
        assert "DO NOT TOUCH main" in result.summary


class TestTheTranscriptIsNoLongerRewritten:
    async def test_S105_compaction_leaves_the_raw_transcript_intact(
        self,
    ) -> None:
        cm = make_cm()
        contents = ["goal", "a", "b", "c", "d", "e"]
        for text in contents:
            cm.append(user(text))
        await cm.compact()
        assert [m.content for m in cm.transcript] == contents
        assert cm.effective_size < len(cm.transcript)

    async def test_S105_the_effective_view_is_what_the_model_sees(
        self,
    ) -> None:
        cm = make_cm()
        for text in ["goal", "a", "b", "c"]:
            cm.append(user(text))
        await cm.compact()
        _system, assembled = cm.assemble()
        # Compaction arms the instruction reminder, so the assembly carries
        # one trailing message the effective view does not.
        assert [m.content for m in assembled[:-1]] == [
            m.content for m in cm.effective_messages()
        ]

    async def test_S105_a_second_compaction_layers_on_the_first(self) -> None:
        # Each condensation is a prefix replacement over the view the
        # previous ones produce. Applying them against the raw transcript
        # instead would make the second one evict messages the first had
        # already replaced.
        cm = make_cm()
        for i in range(12):
            cm.append(user(f"m{i}"))
        await cm.compact()
        first = cm.effective_messages()
        assert first[0].content is not None
        assert first[0].content.startswith(COMPACTION_SUMMARY_PREFIX)
        await cm.compact()
        second = cm.effective_messages()
        assert second[0].content is not None
        assert second[0].content.startswith(COMPACTION_SUMMARY_PREFIX)
        assert len(second) < len(first)
        # The second summary summarized the first summary, not the raw head.
        assert "STUB SUMMARY of 3 messages" in (second[0].content or "")

    async def test_S105_refs_stay_unique_across_condensations(self) -> None:
        # The summary takes a fresh ref. Reusing `_next_ref` without bumping
        # it gives the summary the same ref as the next appended message, so
        # two different messages in the effective view cite the same event --
        # and a pruning stub then points at the wrong one.
        cm = make_cm(condenser=_NamesItself())
        cm.append(user("goal"))
        for i in range(10):
            cm.append(assistant(f"t{i}", [ToolCall(id=f"c{i}", name="bash")]))
            cm.append(tool(f"c{i}", f"out {i}", i % 3 == 0))
        for _ in range(3):
            await cm.compact()
            for i in range(4):
                cm.append(user(f"after {i}"))
            refs = cm._effective()[1]
            assert len(refs) == len(set(refs)), refs
            assert len(refs) == cm.effective_size

    async def test_S105_effective_size_is_what_the_shrink_guard_reads(
        self,
    ) -> None:
        # `len(transcript)` no longer falls, so a fixpoint loop reading it
        # would never terminate: it would summarize until the budget was
        # gone, before the first model call.
        cm = make_cm()
        for i in range(8):
            cm.append(user(f"m{i}"))
        before_raw = len(cm.transcript)
        before_effective = cm.effective_size
        await cm.compact()
        assert len(cm.transcript) == before_raw
        assert cm.effective_size < before_effective

    async def test_S105_the_prune_plan_indexes_the_condensed_view(
        self,
    ) -> None:
        # The plan is a frozenset of indices; the assembly consumes it. If
        # one derived from the raw transcript and the other from the
        # condensed view, pruning would stub whichever message happened to
        # sit at that offset -- silently, and only after a compaction.
        cm = make_cm(max_context=400)
        cm.append(user("goal"))
        for i in range(10):
            cm.append(assistant(f"turn {i}", [ToolCall(id=f"c{i}", name="bash")]))
            cm.append(tool(f"c{i}", "X" * 400))
        await cm.compact()
        plan = cm._prune_plan()
        effective = cm.effective_messages()
        assert plan, "the corpus for this test must actually prune"
        for index in plan:
            assert index < len(effective)
            assert effective[index].role is Role.TOOL


class TestStrategiesAreSwappableAndNamed:
    """Acceptance 2."""

    def test_S105_a_strategy_is_selected_by_name(self) -> None:
        assert isinstance(
            condenser_for("summarize-halve", _summary), DefaultCondenser
        )

    def test_S105_an_unknown_name_raises_rather_than_defaulting(self) -> None:
        # Falling back silently would make a typo look exactly like a working
        # configuration: the report names the strategy the operator asked
        # for, the run uses another one.
        with pytest.raises(ValueError, match="unknown condenser"):
            condenser_for("summarise-halve", _summary)

    async def test_S105_the_condensation_names_its_strategy(self) -> None:
        cm = make_cm(condenser=_NamesItself())
        for i in range(6):
            cm.append(user(f"m{i}"))
        await cm.compact()
        assert cm.last_condensation is not None
        assert cm.last_condensation.strategy_id == "names-itself"

    async def test_S105_a_custom_strategy_is_honoured(self) -> None:
        class KeepNothingSayNothing:
            strategy_id = "silent"

            async def condense(self, span, ctx):
                return Condensation(summary="", strategy_id=self.strategy_id)

        cm = make_cm(condenser=KeepNothingSayNothing())
        for i in range(6):
            cm.append(user(f"m{i}"))
        await cm.compact()
        assert cm.effective_messages()[0].content == ""
        assert cm.last_condensation is not None
        assert cm.last_condensation.strategy_id == "silent"


class TestTheBenchmarkPathIsUnchanged:
    def test_S105_the_config_default_is_the_default_strategy(self) -> None:
        from harness.config import HarnessConfig

        assert HarnessConfig().condenser == DefaultCondenser.strategy_id

    async def test_S105_a_typo_raises_rather_than_running_the_default(
        self,
    ) -> None:
        # The wiring, not just `condenser_for`. There used to be a profile
        # gate here that returned the default unless the profile declared
        # `pivotal_retention`; validating *after* it meant a typo was
        # silently corrected on the benchmark path, the one place nobody
        # would notice. The gate is gone with the second strategy, but the
        # raise has to survive it.
        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.persistence import RunStore
        from harness.sandbox.docker import DockerSandbox
        from harness.types import ModelResponse, StopReason, Usage

        DockerSandbox.availability = classmethod(lambda cls: False)
        store = RunStore(self._tmp / "state.db")
        orchestrator = Orchestrator(
            HarnessConfig(home=self._tmp / "home", condenser="summarise-halve"),
            store,
        )
        with pytest.raises(ValueError, match="unknown condenser"):
            await orchestrator.run_task(
                "goal", "fake-model",
                adapter_override=FakeAdapter([
                    ModelResponse(
                        message=assistant("Task complete. Did it."),
                        usage=Usage(), stop_reason=StopReason.END_TURN,
                    )
                ]),
                budgets=Budgets(max_turns=2),
            )

    @pytest.fixture(autouse=True)
    def _paths(self, tmp_path, monkeypatch):
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )
        self._tmp = tmp_path


def _note_tool():
    """A tool that succeeds and says nothing interesting."""
    from tests.test_loop import simple_tool

    return simple_tool("note")


class TestConstraintSurvival:
    """Acceptance 4 — the test that closes open question §9.1.

    §9.1 asked whether compaction preserves what the run must not forget. It
    stayed open because nothing measured it. This drives the real
    `AgentLoop` for forty turns with a window small enough to force repeated
    compaction, plants three constraints at turn 3 by three different routes,
    and reports which survive.

    They do not all survive, and that is the finding rather than a failure:
    the two routes the harness treats as first-class do, and unregistered
    prose does not.
    """

    async def _run(self, tmp_path, turns: int, condenser=None):
        from harness.adapters.fake import FakeAdapter
        from harness.config import PermissionMode
        from harness.loop import AgentLoop, Budgets
        from harness.permissions import Policy
        from harness.persistence import RunStore
        from harness.tools.registry import ToolRegistry
        from harness.types import ModelResponse, StopReason, Usage

        goal = "Port the parser. NEVER modify vendor/ — it is generated."
        store = RunStore(tmp_path / "state.db")
        run_id = store.create_run(goal, "m", "auto")
        agent_id = store.create_agent(run_id, goal)

        # Each turn must call a tool, or the loop finishes on turn 1 and the
        # forty responses are never consumed -- a "forty turns of compaction"
        # test that compacts zero times and passes anyway.
        responses = [
            ModelResponse(
                message=assistant(
                    f"working, step {i} " + "x" * 600,
                    [ToolCall(id=f"c{i}", name="note", arguments={"text": "ok"})],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            for i in range(turns)
        ]
        responses[-1] = ModelResponse(
            message=assistant(
                "Task complete. I ported the parser and ran the tests."
            ),
            usage=Usage(),
            stop_reason=StopReason.END_TURN,
        )

        context = ContextManager(
            base_system_prompt="SYS",
            count_tokens=lambda msgs: sum(_size(m) for m in msgs) // 4,
            # Small enough that forty turns compact repeatedly. A window that
            # only compacts once or twice would let a "survives forty turns"
            # test pass on a transcript that was barely touched.
            max_context=1_200,
            summarize=_summary,
            reminder_interval=50,
            condenser=condenser,
        )
        registry = ToolRegistry()
        registry.register(_note_tool())
        loop = AgentLoop(
            adapter=FakeAdapter(responses),
            registry=registry,
            policy=Policy(mode=PermissionMode.AUTO),
            store=store,
            run_id=run_id,
            agent_id=agent_id,
            context=context,
            budgets=Budgets(max_turns=turns + 5),
            ask=None,
            sandbox=None,
        )
        # Seed the goal before anything else is planted. `_goal_text` is
        # whatever message is appended first, and it is what every summary
        # header carries verbatim -- so a test that plants its constraint
        # before the goal makes that constraint immortal and proves nothing.
        # `run` appends the goal again; the duplicate is evicted early and
        # only the header copy survives, which is the mechanism under test.
        context.append(user(goal))
        return loop, context, goal

    async def test_S105_constraints_survive_forty_turns_of_compaction(
        self, tmp_path
    ) -> None:
        loop, context, goal = await self._run(tmp_path, turns=40)

        # Three routes, all planted at the start of the run.
        context.add_instruction(
            "Never modify files under vendor/", "user"
        )  # the instruction ledger
        loop._append_message(
            user("Reminder: the deadline is Friday and vendor/ is off limits.")
        )  # unregistered prose

        await loop.run(goal)

        system, messages = context.assemble()
        assembled = system + "\n" + text_of(messages)

        # 1. The ledger survives: it is re-rendered into the system prompt on
        #    every assembly, so compaction cannot reach it.
        assert "Never modify files under vendor/" in assembled
        # 2. The goal survives: it is folded verbatim into every summary
        #    header and never rides on the summarizer.
        assert goal in assembled
        # 3. Unregistered prose does not. This is the §9.1 answer, not a bug
        #    in compaction: a constraint stated in a user message and never
        #    registered has no representation the harness knows to preserve.
        #    The mitigation is `add_instruction`, which is why that tool
        #    exists -- and this is the number that says how much it matters.
        assert "deadline is Friday" not in assembled

    async def test_S105_the_run_actually_compacted(self, tmp_path) -> None:
        # Without this the test above passes on a run that never compacted at
        # all, which is the failure mode it exists to rule out.
        loop, context, goal = await self._run(tmp_path, turns=40)
        await loop.run(goal)
        events = [
            e
            for e in loop.store.load_events(loop.agent_id)
            if e.kind == "compaction"
        ]
        assert len(events) >= 3, f"only {len(events)} compactions"
        assert all(e.payload["strategy_id"] == "summarize-halve" for e in events)

    async def test_S105_the_event_names_a_non_default_strategy(
        self, tmp_path
    ) -> None:
        # The assertion above is satisfied by hardcoding the literal in the
        # payload. This is the other half.
        loop, _context, goal = await self._run(
            tmp_path, turns=40, condenser=_NamesItself()
        )
        await loop.run(goal)
        events = [
            e
            for e in loop.store.load_events(loop.agent_id)
            if e.kind == "compaction"
        ]
        assert events
        assert all(
            e.payload["strategy_id"] == "names-itself" for e in events
        )

    async def test_S105_the_condensation_reaches_the_event_log(
        self, tmp_path
    ) -> None:
        # The seam's one remaining promise beyond the summary text: which
        # strategy ran is recoverable from the run. There used to be a
        # `kept_refs` assertion here too; S-404 deleted retention, and this
        # is what is left that a future strategy still depends on.
        loop, _context, goal = await self._run(
            tmp_path, turns=40, condenser=_NamesItself()
        )
        await loop.run(goal)
        events = [
            e
            for e in loop.store.load_events(loop.agent_id)
            if e.kind == "compaction"
        ]
        assert events
        assert all(e.payload["strategy_id"] == "names-itself" for e in events)
        assert all("S" in (e.payload["summary"] or "") for e in events)