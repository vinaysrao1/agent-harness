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
3. Pivotal retention keeps what the summarizer would have flattened -- and
   keeps it in a form the provider will accept.
"""

from __future__ import annotations

import pytest

from harness.condenser import (
    COMPACTION_SUMMARY_PREFIX,
    MAX_PIVOTAL_KEPT,
    CondenseContext,
    Condensation,
    DefaultCondenser,
    PivotalCondenser,
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

    async def test_S105_the_default_keeps_nothing(self) -> None:
        result = await DefaultCondenser(_summary).condense(
            [user("a"), user("b")],
            CondenseContext(goal="g", refs=(1, 2)),
        )
        assert result.kept_refs == ()
        assert result.dropped_refs == (1, 2)

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
        cm = make_cm(condenser=PivotalCondenser(_summary))
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


class TestPivotalRetention:
    """Acceptance 3."""

    async def test_S105_a_marked_turn_survives_eviction(self) -> None:
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        cm.append(assistant("looking"))
        pivotal = cm.append(
            user("VERIFICATION FAILED: pytest -q exited 1: missing fixture 'db'")
        )
        cm.mark_pivotal(pivotal, "verification_failed")
        for i in range(5):
            cm.append(user(f"later {i}"))
        await cm.compact()
        text = text_of(cm.effective_messages())
        assert "missing fixture 'db'" in text

    async def test_S105_the_default_strategy_drops_the_same_turn(self) -> None:
        # The control. Without it the test above passes whenever the
        # eviction boundary happens to fall after the marked message.
        cm = make_cm()
        cm.append(user("goal"))
        cm.append(assistant("looking"))
        pivotal = cm.append(
            user("VERIFICATION FAILED: pytest -q exited 1: missing fixture 'db'")
        )
        cm.mark_pivotal(pivotal, "verification_failed")
        for i in range(5):
            cm.append(user(f"later {i}"))
        await cm.compact()
        text = text_of(cm.effective_messages())
        assert "missing fixture 'db'" not in text

    async def test_S105_a_retained_tool_result_brings_its_assistant_turn(
        self,
    ) -> None:
        # A `tool_result` with no preceding `tool_use` is rejected by the
        # provider. Retaining the failing result alone would have produced a
        # 400 on the next model call, mid-run, when context is tightest.
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        cm.append(assistant("run it", [ToolCall(id="c1", name="bash")]))
        ref = cm.append(tool("c1", "FAILED: segfault in parser.c:88", True))
        cm.mark_pivotal(ref, "tool_error")
        for i in range(6):
            cm.append(user(f"later {i}"))
        await cm.compact()
        effective = cm.effective_messages()
        assert "segfault in parser.c:88" in text_of(effective)
        # The assistant message that owns the tool_use came with it.
        tool_index = next(
            i for i, m in enumerate(effective) if m.role is Role.TOOL
        )
        owner = effective[tool_index - 1]
        assert owner.role is Role.ASSISTANT
        assert [c.id for c in owner.tool_calls] == ["c1"]

    async def test_S105_the_kept_transcript_never_starts_with_a_tool_result(
        self,
    ) -> None:
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        for i in range(4):
            cm.append(assistant(f"t{i}", [ToolCall(id=f"c{i}", name="bash")]))
            ref = cm.append(tool(f"c{i}", f"boom {i}", True))
            cm.mark_pivotal(ref, "tool_error")
        for i in range(8):
            cm.append(user(f"later {i}"))
        await cm.compact()
        effective = cm.effective_messages()
        assert effective[0].role is Role.USER  # the summary
        for index, message in enumerate(effective):
            if message.role is Role.TOOL:
                assert effective[index - 1].role in (Role.ASSISTANT, Role.TOOL)

    async def test_S105_retention_is_capped(self) -> None:
        # Uncapped, a run that fails often retains most of its own span, the
        # assembly barely shrinks, and the loop's fixpoint pass spins:
        # compaction returns a span every time and never makes progress.
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        for i in range(20):
            ref = cm.append(user(f"FAILURE {i}"))
            cm.mark_pivotal(ref, "tool_error")
        for i in range(4):
            cm.append(user(f"tail {i}"))
        before = cm.effective_size
        await cm.compact()
        assert cm.effective_size < before
        kept = cm.last_condensation
        assert kept is not None
        # A literal, not the constant: `assert n <= MAX_PIVOTAL_KEPT` is
        # satisfied by setting MAX_PIVOTAL_KEPT to 40.
        assert MAX_PIVOTAL_KEPT == 4
        assert len(kept.reasons) <= 4

    async def test_S105_the_reasons_say_what_was_kept_and_why(self) -> None:
        # `pivotal_reasons` is the telemetry the spec sells: a retention that
        # never retains, or always retains, should be visible in the event
        # log. Returning an empty tuple left every other test green.
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        cm.append(assistant("run it", [ToolCall(id="c1", name="bash")]))
        cm.append(tool("c1", "boom", True))  # auto-marked `tool_error`
        verification = cm.append(user("VERIFICATION FAILED: pytest exited 1"))
        cm.mark_pivotal(verification, "verification_failed")
        for i in range(8):
            cm.append(user(f"later {i}"))
        await cm.compact()
        assert cm.last_condensation is not None
        reasons = cm.last_condensation.reasons
        assert [r.split(" (")[0] for r in reasons] == [
            "tool_error",
            "verification_failed",
        ]
        # The ref range names the whole turn, not just the marked message:
        # the tool error's turn is its assistant message plus the result.
        assert reasons[0] == "tool_error (refs 2-3)"
        assert reasons[1] == "verification_failed (refs 4-4)"

    async def test_S105_the_most_recent_pivotal_turns_win(self) -> None:
        # Oldest-first was wrong: an early failure that has since been fixed
        # is exactly the one worth summarizing away.
        cm = make_cm(condenser=PivotalCondenser(_summary))
        cm.append(user("goal"))
        for i in range(10):
            ref = cm.append(user(f"FAILURE {i}"))
            cm.mark_pivotal(ref, "tool_error")
        for i in range(4):
            cm.append(user(f"tail {i}"))
        await cm.compact()
        text = text_of(cm.effective_messages())
        assert "FAILURE 0" not in text
        assert "FAILURE 9" in text

    async def test_S105_compaction_still_shrinks_when_everything_is_pivotal(
        self,
    ) -> None:
        cm = make_cm(condenser=PivotalCondenser(_summary))
        for i in range(30):
            ref = cm.append(user(f"m{i}"))
            cm.mark_pivotal(ref, "tool_error")
        for _ in range(6):
            before = cm.effective_size
            await cm.compact()
            assert cm.effective_size < before, "the fixpoint pass would spin"

    def test_S105_the_first_reason_for_a_ref_wins(self) -> None:
        # A turn marked as a failed verification must not be relabelled as a
        # generic tool error by a later mark.
        cm = make_cm()
        ref = cm.append(user("x"))
        cm.mark_pivotal(ref, "verification_failed")
        cm.mark_pivotal(ref, "tool_error")
        assert cm._pivotal[ref] == "verification_failed"


class TestStrategiesAreSwappableAndNamed:
    """Acceptance 2."""

    def test_S105_a_strategy_is_selected_by_name(self) -> None:
        assert isinstance(
            condenser_for("summarize-halve", _summary), DefaultCondenser
        )
        assert isinstance(
            condenser_for("summarize-halve+pivotal", _summary), PivotalCondenser
        )

    def test_S105_an_unknown_name_raises_rather_than_defaulting(self) -> None:
        # Falling back silently would make a typo look exactly like a working
        # configuration: the report names the strategy the operator asked
        # for, the run uses another one.
        with pytest.raises(ValueError, match="unknown condenser"):
            condenser_for("summarise-halve", _summary)

    def test_S105_a_typo_raises_on_the_benchmark_path_too(self) -> None:
        # Validating *after* the profile gate meant a typo was silently
        # corrected to the default under `CODING` -- the default profile, and
        # the one path where nobody would notice. It only ever raised for
        # repo runs.
        from harness.orchestrator import select_condenser_strategy
        from harness.profiles import CODING, CODING_REPO

        for profile in (CODING, CODING_REPO, None):
            with pytest.raises(ValueError, match="unknown condenser"):
                select_condenser_strategy("summarise-halve", profile)

    async def test_S105_the_condensation_names_its_strategy(self) -> None:
        cm = make_cm(condenser=PivotalCondenser(_summary))
        for i in range(6):
            cm.append(user(f"m{i}"))
        await cm.compact()
        assert cm.last_condensation is not None
        assert cm.last_condensation.strategy_id == "summarize-halve+pivotal"

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
    def test_S105_coding_does_not_enable_pivotal_retention(self) -> None:
        from harness.profiles import CODING, CODING_REPO

        assert not CODING.enables("pivotal_retention")
        assert CODING_REPO.enables("pivotal_retention")

    def test_S105_the_config_default_is_the_default_strategy(self) -> None:
        from harness.config import HarnessConfig

        assert HarnessConfig().condenser == DefaultCondenser.strategy_id

    def test_S105_config_cannot_move_the_benchmark_path(self) -> None:
        # The wiring, not just the capability. Making the orchestrator ignore
        # `config.condenser` entirely -- so no configuration could ever select
        # a strategy -- left every other test in this file green.
        from harness.orchestrator import select_condenser_strategy
        from harness.profiles import CODING, CODING_REPO

        assert (
            select_condenser_strategy("summarize-halve+pivotal", CODING)
            == "summarize-halve"
        )
        assert (
            select_condenser_strategy("summarize-halve+pivotal", CODING_REPO)
            == "summarize-halve+pivotal"
        )
        # No profile at all is the benchmark case too: `CODING` is what runs
        # when nothing was selected.
        assert (
            select_condenser_strategy("summarize-halve+pivotal", None)
            == "summarize-halve"
        )

    def test_S105_a_repo_run_gets_the_strategy_config_asked_for(self) -> None:
        from harness.orchestrator import select_condenser_strategy
        from harness.profiles import CODING_REPO

        chosen = select_condenser_strategy(
            "summarize-halve+pivotal", CODING_REPO
        )
        assert isinstance(condenser_for(chosen, _summary), PivotalCondenser)


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
            tmp_path, turns=40, condenser=PivotalCondenser(_summary)
        )
        await loop.run(goal)
        events = [
            e
            for e in loop.store.load_events(loop.agent_id)
            if e.kind == "compaction"
        ]
        assert events
        assert all(
            e.payload["strategy_id"] == "summarize-halve+pivotal"
            for e in events
        )

    async def test_S105_pivotal_retention_keeps_a_failure_the_default_loses(
        self, tmp_path
    ) -> None:
        marker = "FATAL: undefined symbol _zlib_inflate at link time"

        async def survives(condenser) -> bool:
            loop, context, goal = await self._run(
                tmp_path, turns=40, condenser=condenser
            )
            ref = loop._append_message(user(marker))
            context.mark_pivotal(ref, "verification_failed")
            await loop.run(goal)
            return marker in text_of(context.assemble()[1])

        assert not await survives(None), "the control never lost it"
        assert await survives(PivotalCondenser(_summary))


def _failing_note_tool():
    """`note`, but it raises on "boom" -- an error result marks the turn."""
    from harness.permissions import ToolMeta
    from harness.tools.registry import Tool
    from harness.types import ToolSpec

    async def handler(arguments: dict) -> str:
        if arguments.get("text") == "boom":
            raise RuntimeError("linker failed")
        return "noted"

    return Tool(
        spec=ToolSpec(name="note", description="test tool note"),
        meta=ToolMeta(side_effect=False),
        handler=handler,
    )


class TestResumeCarriesWhatRetentionKept:
    """Resume rebuilds the transcript from events. Splicing in only the
    summary dropped exactly the turns pivotal retention exists to keep -- and
    dropped them silently, on a path no unit test touches."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the LocalSandbox fallback, the way the rest of the suite
        does. Assigning to `DockerSandbox.availability` directly was never
        reverted and leaked into every later module that calls it."""
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def _orchestrator(self, tmp_path, strategy: str):
        from harness.config import HarnessConfig
        from harness.persistence import RunStore
        from harness.orchestrator import Orchestrator

        store = RunStore(tmp_path / "state.db")
        config = HarnessConfig(home=tmp_path / "home", condenser=strategy)
        return Orchestrator(config, store), store

    async def _paused_run_that_compacted(self, tmp_path, strategy, marker):
        from harness.adapters.fake import FakeAdapter
        from harness.loop import Budgets
        from harness.profiles import CODING_REPO
        from harness.types import ModelResponse, StopReason, Usage

        orchestrator, store = await self._orchestrator(tmp_path, strategy)
        goal = "Build the thing and keep it building."

        turns = 24
        responses = [
            ModelResponse(
                message=assistant(
                    (marker if i == 2 else f"step {i} ") + "y" * 600,
                    [
                        ToolCall(
                            id=f"c{i}",
                            name="note",
                            # Turn 2's call fails, which is what marks the
                            # turn pivotal (`tool_error`). Without a failure
                            # nothing is ever marked and retention has
                            # nothing to retain -- the run compacts and the
                            # test proves nothing.
                            arguments={"text": "boom" if i == 2 else "ok"},
                        )
                    ],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            # More responses than the budget allows: a failing tool draws a
            # diligence nudge, which spends an extra model call, and an
            # exhausted script ends the run in `error` rather than
            # `paused_budget` -- with nothing to resume.
            for i in range(turns + 8)
        ]
        # A small window so the run compacts. `Capabilities` is frozen and
        # `capabilities` is a property, so this subclasses rather than
        # assigning -- assigning raised, which is the right failure.
        class _Narrow(FakeAdapter):
            @property
            def capabilities(self):
                from harness.types import Capabilities

                return Capabilities(
                    max_context=2_400, supports_cache_control=False
                )

        adapter = _Narrow(responses)
        run_id, result = await orchestrator.run_task(
            goal,
            "fake-model",
            adapter_override=adapter,
            budgets=Budgets(max_turns=turns),
            profile=CODING_REPO,
            tool_factories=[lambda deps: _failing_note_tool()],
        )
        return orchestrator, store, run_id, result

    async def test_S105_a_resumed_run_still_has_the_retained_turn(
        self, tmp_path
    ) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.loop import Budgets
        from harness.types import ModelResponse, StopReason, Usage

        marker = "FATAL: undefined symbol _zlib_inflate "
        orchestrator, store, run_id, result = (
            await self._paused_run_that_compacted(
                tmp_path, "summarize-halve+pivotal", marker
            )
        )
        events = [
            e for e in store.load_events(
                store.list_agents(run_id)[0].id
            ) if e.kind == "compaction"
        ]
        assert result.status == "paused_budget", (
            result.status, result.final_text
        )
        assert events, "the run never compacted; the test proves nothing"
        assert any(e.payload["kept"] for e in events), (
            "retention kept nothing, so resume has nothing to lose"
        )

        second = FakeAdapter(
            [
                ModelResponse(
                    message=assistant("Task complete. Built and verified it."),
                    usage=Usage(),
                    stop_reason=StopReason.END_TURN,
                )
            ]
        )
        resumed = await orchestrator.resume_task(
            # Turn budgets are cumulative across a resume, so a small
            # number here pauses again immediately without ever calling the
            # adapter -- and every assertion below would read an empty list.
            run_id, adapter_override=second, budgets=Budgets(max_turns=64)
        )
        assert second.calls, (resumed.status, resumed.final_text)
        replayed = text_of(second.calls[0].messages)
        assert marker in replayed, (
            "the resumed transcript lost the turn retention had kept"
        )


class TestRetentionSurvivesTheOtherEvictionLayer:
    """Retention puts the kept turn at the *front* of the effective view, and
    the prune shed is oldest-first — so the retained failure was the first
    thing stubbed, on every turn after the compaction, while `kept_refs` and
    `pivotal_reasons` went on saying it had survived.

    Every other test in this file reads `effective_messages()`, which is the
    *pre-prune* view. That is why the suite could not see this.
    """

    def _cm(self, condenser):
        return ContextManager(
            base_system_prompt="SYS",
            count_tokens=lambda ms: sum(_size(m) for m in ms) // 4,
            max_context=3_000,
            summarize=_summary,
            reminder_interval=50,
            condenser=condenser,
        )

    def _load(self, cm, marker: str) -> int:
        cm.append(user("goal"))
        pivotal_ref = 0
        for i in range(24):
            cm.append(assistant(f"t{i}", [ToolCall(id=f"c{i}", name="bash")]))
            ref = cm.append(
                tool(f"c{i}", (marker if i == 3 else f"output {i} ") * 60, i == 3)
            )
            if i == 3:
                cm.mark_pivotal(ref, "verification_failed")
                pivotal_ref = ref
        return pivotal_ref

    async def test_S105_a_retained_turn_is_not_then_pruned_away(self) -> None:
        marker = "MARKER undefined symbol _zlib_inflate "
        cm = self._cm(PivotalCondenser(_summary))
        self._load(cm, marker)
        await cm.compact()

        assert marker in text_of(cm.effective_messages()), "nothing was retained"
        plan = cm._prune_plan()
        assert plan, "pruning must be active, or this asserts nothing"
        # The within-test control: other old tool results *are* stubbed in the
        # same assembly, so the marker's survival is protection and not an
        # inactive shed.
        assembled = text_of(cm.assemble()[1])
        assert "[pruned:" in assembled
        assert marker in assembled, "the retained turn was stubbed by pruning"

    async def test_S105_an_unretained_tool_result_is_still_pruned(self) -> None:
        # The control for the protection itself. Without it, a shed that
        # never touched anything would make the test above pass.
        cm = self._cm(DefaultCondenser(_summary))
        self._load(cm, "MARKER ")
        await cm.compact()
        assembled = text_of(cm.assemble()[1])
        assert "[pruned:" in assembled
        assert "MARKER" not in assembled

    async def test_S105_protection_is_keyed_on_what_was_kept(self) -> None:
        # Not on `_pivotal`: marks are recorded on every profile including
        # the benchmark one, so protecting marks directly would change the
        # `CODING` assembly and break N7. `DefaultCondenser` keeps nothing,
        # so the protected set is empty there.
        cm = self._cm(DefaultCondenser(_summary))
        self._load(cm, "MARKER ")
        await cm.compact()
        assert cm._pivotal, "the run did mark a turn"
        assert cm._retained_refs() == frozenset()


class TestARealTurnIsMoreThanOneMessage:
    """`room` capped *turns* with a *message* budget. A turn is an assistant
    message plus its tool results, and `tool_error` only ever marks tool
    results — so every realistic pivotal turn is at least two messages and
    the floor did nothing."""

    def _cm(self):
        return ContextManager(
            base_system_prompt="SYS",
            count_tokens=lambda ms: sum(_size(m) for m in ms) // 4,
            max_context=2_000,
            summarize=_summary,
            reminder_interval=50,
            condenser=PivotalCondenser(_summary),
        )

    async def test_S105_a_condensation_never_grows_the_view(self) -> None:
        # Two three-message turns, everything marked: the old cap allowed all
        # six to be kept, so the condensation replaced six messages with
        # seven.
        cm = self._cm()
        for i in range(2):
            cm.append(assistant(f"t{i}", [ToolCall(id=f"a{i}", name="bash")]))
            for _ in range(2):
                ref = cm.append(tool(f"a{i}", "x" * 300, True))
                cm.mark_pivotal(ref, "tool_error")
        before = cm.effective_size
        await cm.compact()
        assert cm.effective_size < before, (
            f"the condensation grew the view: {before} -> {cm.effective_size}"
        )

    async def test_S105_repeated_compaction_of_multi_message_turns_shrinks(
        self,
    ) -> None:
        # Through the loop this presented as a hard floor the transcript
        # could never compact below, plus one wasted real summarizer call
        # every turn for the rest of the run.
        cm = self._cm()
        for i in range(12):
            cm.append(assistant(f"t{i}", [ToolCall(id=f"a{i}", name="bash")]))
            for _ in range(2):
                ref = cm.append(tool(f"a{i}", "x" * 300, True))
                cm.mark_pivotal(ref, "tool_error")
        for round_number in range(8):
            before = cm.effective_size
            await cm.compact()
            assert cm.effective_size < before, (
                f"stalled at {before} on round {round_number}"
            )


class TestTheLoopMarksAFailedVerification:
    """The spec's primary pivotal signal. It was wired and never tested:
    deleting the `mark_pivotal` call in `harness/loop.py` left the whole
    2,532-test suite green, because every retention test marked by hand."""

    async def test_S105_a_failed_verification_marks_its_reminder(
        self, tmp_path
    ) -> None:
        from harness.sandbox.base import ExecResult
        from harness.tools.builtin import declare_verification_tool
        from tests.test_loop import declare, make_harness, resp, GOAL

        class _Failing:
            workspace_root = tmp_path

            async def start(self) -> None: ...

            async def stop(self) -> None: ...

            async def exec(self, command: str, timeout: float = 120):
                return ExecResult(
                    exit_code=1,
                    stdout="E   assert 0 == 1  # tests/test_parser.py:88",
                    stderr="",
                )

        h = make_harness(
            tmp_path,
            [
                resp("declaring", [declare("v1", "pytest -q")]),
                resp("Task complete. Ported the parser and ran the tests."),
                resp("Task complete. Ported the parser and ran the tests."),
            ],
            tools=[declare_verification_tool()],
        )
        h.loop.sandbox = _Failing()  # type: ignore[assignment]
        await h.loop.run(GOAL)

        marks = h.loop.context._pivotal
        assert "verification_failed" in marks.values(), marks
        # The marked message is the one carrying the failing output -- the
        # thing a summarizer would render as "ran the tests".
        ref = next(r for r, why in marks.items() if why == "verification_failed")
        index = h.loop.context._event_refs.index(ref)
        marked = h.loop.context.transcript[index]
        assert "tests/test_parser.py:88" in (marked.content or "")
