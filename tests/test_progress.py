"""S-107: is this run going anywhere?

Which detectors exist was decided by measurement, and the measurement had to
be done twice. Both corrections are pinned here because both were the same
mistake in different clothes.

**Lift needs a length-matched base rate.** Long runs fail more (32.7% under 20
calls, 77.8% over 80), so a detector that needs a long run to fire inherits
that as apparent skill. The null `len(calls) >= 25` — no logic at all — scores
1.31x globally. A `no_progress` detector shipped on a 1.33x global lift and was
deleted when the control showed it at **1.01x matched**: it was measuring run
length.

**A failing command is not `is_error`.** `CONSECUTIVE_FAILURE` shipped
*rejected*, on a measurement keyed off `ToolResult.is_error` — which means the
tool failed, not that the command exited non-zero. Across the corpus
`is_error` is set 8 times, a non-zero exit appears 299 times, and they never
coincide. The rejection came from an instrument that could not see the thing
it was rejecting.

Nothing nudges: N5 hashes `NUDGE_SOURCES` and the count of `nudges += 1`
sites, so a third source is Lane B by construction, and the best detector here
is 1.31x on sixteen firings.
"""

from __future__ import annotations

import pytest

from harness.progress import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    MIN_SCOREABLE_CALLS,
    REPEATED_CALL_THRESHOLD,
    STUCK_EVENT,
    Detector,
    ProgressMonitor,
    audit,
    command_head,
    result_failed,
)


def drive(monitor: ProgressMonitor, calls) -> list:
    """Feed `(name, arguments)` pairs; ids are synthesised."""
    out = []
    for i, (name, arguments) in enumerate(calls):
        signal = monitor.observe_call(f"c{i}", name, arguments)
        if signal:
            out.append(signal)
    return out


def drive_results(monitor: ProgressMonitor, pairs) -> list:
    """Feed `(command, result_content)` pairs through call *and* result."""
    out = []
    for i, (command, content) in enumerate(pairs):
        for signal in (
            monitor.observe_call(f"c{i}", "bash", {"command": command}),
            monitor.observe_result(f"c{i}", content, False),
        ):
            if signal:
                out.append(signal)
    return out


class TestTheRepeatedCallDetector:
    def test_S107_four_identical_calls_trip_it(self) -> None:
        monitor = ProgressMonitor()
        calls = [("bash", {"command": "pytest -q"})] * 4
        (signal,) = drive(monitor, calls)
        assert signal.detector is Detector.REPEATED_CALL
        assert signal.at_call == 4
        assert "4 times" in signal.evidence

    def test_S107_three_do_not(self) -> None:
        # Not a threshold picked for roundness: at three the precision is
        # 54.8% against a 45.4% base rate, which is barely a signal at all.
        assert REPEATED_CALL_THRESHOLD == 4
        monitor = ProgressMonitor()
        assert drive(monitor, [("bash", {"command": "pytest -q"})] * 3) == []

    def test_S107_different_arguments_are_different_calls(self) -> None:
        monitor = ProgressMonitor()
        calls = [("bash", {"command": f"ls {i}"}) for i in range(8)]
        assert drive(monitor, calls) == []

    def test_S107_argument_order_does_not_change_the_identity(self) -> None:
        # Providers do not promise argument order. A detector that depended on
        # it would fire or not depending on which provider served the turn.
        monitor = ProgressMonitor()
        calls = [
            ("edit_file", {"path": "a.py", "old_string": "x"}),
            ("edit_file", {"old_string": "x", "path": "a.py"}),
            ("edit_file", {"path": "a.py", "old_string": "x"}),
            ("edit_file", {"old_string": "x", "path": "a.py"}),
        ]
        assert len(drive(monitor, calls)) == 1

    def test_S107_unserialisable_arguments_do_not_take_the_run_down(
        self,
    ) -> None:
        class Awkward:
            def __repr__(self) -> str:
                return "<awkward>"

        monitor = ProgressMonitor()
        calls = [("bash", {"x": Awkward()})] * 4
        assert len(drive(monitor, calls)) == 1

    def test_S107_repeats_need_not_be_consecutive(self) -> None:
        # The failure this is for is a model that has forgotten it already
        # made the call, which does not require the repeats to be adjacent.
        monitor = ProgressMonitor()
        same = ("bash", {"command": "cat x"})
        other = ("bash", {"command": "ls"})
        assert len(drive(monitor, [same, other, same, other, same, other, same])) == 1


class TestTheConsecutiveFailureDetector:
    """The detector that shipped rejected. The first measurement keyed failure
    off `ToolResult.is_error`, which means the *tool* failed -- a command
    exiting non-zero is a perfectly good tool result carrying bad news."""

    def test_S107_a_non_zero_exit_is_a_failure(self) -> None:
        # 8 `is_error` against 299 non-zero exits in the corpus, never
        # coinciding. A detector keyed on the flag cannot see a failing
        # command at all.
        assert result_failed("exit code: 1\n--- stdout ---\n", False)
        assert not result_failed("exit code: 0\nfine", False)
        assert result_failed("anything", True)

    def test_S107_three_failures_of_the_same_command_trip_it(self) -> None:
        monitor = ProgressMonitor()
        pairs = [("pytest -q tests/a.py", "exit code: 1")] * 3
        (signal,) = drive_results(monitor, pairs)
        assert signal.detector is Detector.CONSECUTIVE_FAILURE
        assert "'pytest'" in signal.evidence

    def test_S107_two_do_not(self) -> None:
        assert CONSECUTIVE_FAILURE_THRESHOLD == 3
        monitor = ProgressMonitor()
        assert drive_results(monitor, [("pytest -q", "exit code: 1")] * 2) == []

    def test_S107_a_success_resets_the_streak(self) -> None:
        monitor = ProgressMonitor()
        # Distinct arguments so REPEATED_CALL stays out of it: this is a
        # claim about the streak, not about repetition.
        pairs = [
            ("pytest -q a.py", "exit code: 1"),
            ("pytest -q b.py", "exit code: 1"),
            ("pytest -q c.py", "exit code: 0"),
            ("pytest -q d.py", "exit code: 1"),
            ("pytest -q e.py", "exit code: 1"),
        ]
        assert drive_results(monitor, pairs) == []

    def test_S107_the_head_is_the_first_token_not_the_whole_line(self) -> None:
        # `pytest -q tests/a.py` then `pytest -q tests/b.py` is the same thing
        # going wrong twice. Keying on the full command took the corpus fire
        # count from 34 to 4, which is what made this look refuted.
        assert command_head({"command": "pytest -q tests/a.py"}) == "pytest"
        assert command_head({"command": "pytest -q tests/b.py"}) == "pytest"
        monitor = ProgressMonitor()
        pairs = [
            ("pytest -q tests/a.py", "exit code: 1"),
            ("pytest -q tests/b.py", "exit code: 1"),
            ("pytest -q tests/c.py", "exit code: 2"),
        ]
        assert len(drive_results(monitor, pairs)) == 1

    def test_S107_a_different_command_breaks_the_streak(self) -> None:
        monitor = ProgressMonitor()
        pairs = [
            ("pytest -q", "exit code: 1"),
            ("make", "exit code: 1"),
            ("pytest -q", "exit code: 1"),
        ]
        assert drive_results(monitor, pairs) == []

    def test_S107_only_bash_is_tracked(self) -> None:
        # A failing `read_file` is the model guessing a path, not a command
        # that will not work.
        monitor = ProgressMonitor()
        for i in range(5):
            monitor.observe_call(f"c{i}", "read_file", {"path": "nope.py"})
            assert monitor.observe_result(f"c{i}", "no such file", True) is None

    def test_S107_a_result_for_an_unseen_call_is_ignored(self) -> None:
        # Resume replays results whose calls this monitor never observed.
        monitor = ProgressMonitor()
        assert monitor.observe_result("never-seen", "exit code: 1", False) is None


class TestEachDetectorFiresOnce:
    """A signal that re-fires every turn once tripped makes "how often does
    this happen" unanswerable from the log: one long run would swamp the
    count."""

    def test_S107_a_repeated_call_signals_once_however_long_it_runs(
        self,
    ) -> None:
        monitor = ProgressMonitor()
        signals = drive(monitor, [("bash", {"command": "x"})] * 40)
        assert [s.detector for s in signals] == [Detector.REPEATED_CALL]

    def test_S107_a_failing_command_signals_once(self) -> None:
        monitor = ProgressMonitor()
        # Distinct commands sharing a head, so REPEATED_CALL stays quiet and
        # this is a claim about CONSECUTIVE_FAILURE alone.
        pairs = [(f"pytest -q t{i}.py", "exit code: 1") for i in range(30)]
        signals = drive_results(monitor, pairs)
        assert [s.detector for s in signals] == [Detector.CONSECUTIVE_FAILURE]

    def test_S107_both_detectors_can_fire_in_one_run(self) -> None:
        monitor = ProgressMonitor()
        pairs = [("pytest -q", "exit code: 1")] * 5
        assert {s.detector for s in drive_results(monitor, pairs)} == {
            Detector.REPEATED_CALL,
            Detector.CONSECUTIVE_FAILURE,
        }


class TestItCannotBeWhatMakesARunSlow:
    """Acceptance 4. A detector that walked the transcript would be slowest on
    exactly the long runs it exists for."""

    def test_S107_cost_per_call_does_not_grow_with_run_length(self) -> None:
        import time

        def elapsed(n: int) -> float:
            monitor = ProgressMonitor()
            calls = [("bash", {"command": f"c{i}"}) for i in range(n)]
            start = time.perf_counter()
            for i, (name, arguments) in enumerate(calls):
                monitor.observe_call(f"c{i}", name, arguments)
            return (time.perf_counter() - start) / n

        short = elapsed(200)
        long = elapsed(20_000)
        # Generous: this catches O(n) per call (100x), not constant-factor
        # noise. A wall-clock assertion, which is ordinarily a smell -- it is
        # here because linear-per-call is only observable as wall clock.
        assert long < short * 20, (short, long)


class TestThePayloadCarriesItsEvidence:
    def test_S107_the_payload_names_the_spec_and_the_detector(self) -> None:
        monitor = ProgressMonitor()
        (signal,) = drive(monitor, [("bash", {"command": "x"})] * 4)
        payload = signal.payload()
        assert payload["spec"] == "S-107"
        assert payload["detector"] == "repeated_call"
        assert payload["at_call"] == 4
        assert "bash" in payload["evidence"]

    def test_S107_the_event_kind_is_owned_by_this_spec(self) -> None:
        from harness.specs import EVENT_KIND_SPECS

        assert EVENT_KIND_SPECS[STUCK_EVENT] == "S-107"


class TestTheAuditIsHonestAboutNoData:
    def test_S107_no_matching_trials_says_so(self, tmp_path) -> None:
        # Rather than printing a table of zeroes, which reads as a
        # measurement.
        text = audit(str(tmp_path / "nothing" / "**" / "state.db"))
        assert "no scoreable trials" in text
        assert "%" not in text

    def test_S107_the_audit_scores_a_synthetic_trial(self, tmp_path) -> None:
        import json

        from harness.persistence import RunStore

        trial = tmp_path / "t1"
        (trial / "agent" / "harness-home").mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}),
            encoding="utf-8",
        )
        with RunStore(trial / "agent" / "harness-home" / "state.db") as store:
            run_id = store.create_run("goal", "m", "auto")
            agent_id = store.create_agent(run_id, "goal")
            for i in range(6):
                store.append_event(
                    agent_id,
                    "tool_call",
                    {"id": f"c{i}", "name": "bash",
                     "arguments": {"command": "same"}},
                )
        text = audit(str(tmp_path / "**" / "state.db"))
        assert "trials scored : 1" in text
        assert "repeated_call" in text
        assert "100.0%" in text  # the one trial failed, and it fired


class TestTheLoopEmitsButDoesNotAct:
    """The whole point of the Lane A framing: `stuck_signal` is an event, and
    events do not reach the model."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def _run(self, tmp_path, calls_per_turn):
        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.persistence import RunStore
        from harness.types import (
            Message,
            ModelResponse,
            Role,
            StopReason,
            ToolCall,
            Usage,
        )
        from tests.test_loop import simple_tool

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        script = [
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content="working",
                    tool_calls=[
                        ToolCall(id=f"c{i}", name="note", arguments=args)
                    ],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            for i, args in enumerate(calls_per_turn)
        ]
        script.append(
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content="Task complete. I did the thing and checked it.",
                ),
                usage=Usage(),
                stop_reason=StopReason.END_TURN,
            )
        )
        run_id, result = await orchestrator.run_task(
            "goal",
            "fake-model",
            adapter_override=FakeAdapter(script),
            budgets=Budgets(max_turns=len(script) + 2),
            tool_factories=[lambda deps: simple_tool("note")],
        )
        agent_id = next(
            a.id for a in store.list_agents(run_id) if a.parent_agent_id is None
        )
        return store, agent_id, result

    async def test_S107_a_looping_run_emits_the_signal(self, tmp_path) -> None:
        store, agent_id, result = await self._run(
            tmp_path, [{"text": "same"}] * 5
        )
        events = [
            e for e in store.load_events(agent_id) if e.kind == STUCK_EVENT
        ]
        assert len(events) == 1
        assert events[0].payload["detector"] == "repeated_call"

    async def test_S107_a_healthy_run_emits_nothing(self, tmp_path) -> None:
        store, agent_id, _ = await self._run(
            tmp_path, [{"text": f"n{i}"} for i in range(5)]
        )
        assert [
            e for e in store.load_events(agent_id) if e.kind == STUCK_EVENT
        ] == []

    async def test_S107_the_signal_changes_nothing_the_run_does(
        self, tmp_path
    ) -> None:
        # No nudge, no extra turn, no different outcome -- N5 stays intact.
        looping_store, looping_agent, looping = await self._run(
            tmp_path / "a", [{"text": "same"}] * 5
        )
        healthy_store, healthy_agent, healthy = await self._run(
            tmp_path / "b", [{"text": f"n{i}"} for i in range(5)]
        )
        assert looping.status == healthy.status == "completed"
        assert looping.turns == healthy.turns

        def nudges(store, agent_id):
            return [e for e in store.load_events(agent_id) if e.kind == "nudge"]

        assert nudges(looping_store, looping_agent) == []
        assert nudges(healthy_store, healthy_agent) == []


class TestTheAuditReportsItsConfound:
    """The correction that deleted a detector. Long runs fail more, so lift
    against a global base rate is mostly run length — and a detector that
    needs a long run to fire inherits that as apparent skill."""

    def _corpus(self, tmp_path, trials):
        """`trials` are `(passed, n_calls, command)` triples."""
        import json

        from harness.persistence import RunStore

        for index, (ok, n, command) in enumerate(trials):
            trial = tmp_path / f"t{index}"
            (trial / "agent" / "harness-home").mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps(
                    {"verifier_result": {"rewards": {"reward": 1.0 if ok else 0.0}}}
                ),
                encoding="utf-8",
            )
            db = trial / "agent" / "harness-home" / "state.db"
            with RunStore(db) as store:
                run_id = store.create_run("goal", "m", "auto")
                agent_id = store.create_agent(run_id, "goal")
                for i in range(n):
                    store.append_event(
                        agent_id,
                        "tool_call",
                        {"id": f"c{i}", "name": "bash",
                         "arguments": {"command": command}},
                    )
        return audit(str(tmp_path / "**" / "state.db"))

    def test_S107_the_null_baselines_are_printed(self) -> None:
        # Without them, a length-driven detector reads as a 1.31x signal.
        from harness.progress import NULL_BASELINES

        assert NULL_BASELINES == (25, 50)
        text = self._corpus(
            __import__("pathlib").Path(
                __import__("tempfile").mkdtemp()
            ),
            [(False, 30, f"c{i}") for i in range(3)]
            + [(True, 30, f"d{i}") for i in range(3)],
        )
        assert "[null] len >= 25" in text
        assert "no detector logic at all" in text

    def test_S107_a_null_baseline_scores_1x_matched(self, tmp_path) -> None:
        # By construction: its firing population *is* its comparison
        # population. That is what makes it the yardstick.
        text = self._corpus(
            tmp_path,
            [(False, 30, f"c{i}") for i in range(4)]
            + [(True, 6, f"d{i}") for i in range(4)],
        )
        null_line = next(
            l for l in text.splitlines() if l.startswith("[null] len >= 25")
        )
        assert null_line.rstrip().endswith("1.00x")

    def test_S107_the_excluded_short_trials_are_disclosed(self, tmp_path) -> None:
        # 108 of 727 real trials are under the floor and 84% of them failed:
        # dropping them silently moves the base rate the table is quoted
        # against, from 51.2% to 45.4%.
        assert MIN_SCOREABLE_CALLS == 5
        text = self._corpus(
            tmp_path,
            [(False, 2, "x")] * 3 + [(False, 30, f"c{i}") for i in range(3)],
        )
        assert "excluded      : 3 trials under 5 tool calls" in text
        assert "of which 3 failed" in text


class TestTheAuditsArithmeticIsPinned:
    """Every number in the table was unverified: flipping precision's
    direction, swapping recall's denominator, or turning the lift division
    into a multiplication all passed the suite while inverting the output."""

    def _one(self, tmp_path, passed: bool, repeats: bool = True):
        import json

        from harness.persistence import RunStore

        trial = tmp_path / f"{'p' if passed else 'f'}{repeats}"
        (trial / "agent" / "harness-home").mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {"verifier_result": {"rewards": {"reward": 1.0 if passed else 0.0}}}
            ),
            encoding="utf-8",
        )
        with RunStore(trial / "agent" / "harness-home" / "state.db") as store:
            run_id = store.create_run("goal", "m", "auto")
            agent_id = store.create_agent(run_id, "goal")
            for i in range(6):
                store.append_event(
                    agent_id,
                    "tool_call",
                    {"id": f"c{i}", "name": "bash",
                     "arguments": {"command": "same" if repeats else f"c{i}"}},
                )

    def test_S107_precision_counts_failures_not_successes(
        self, tmp_path
    ) -> None:
        # One firing trial, and it passed. Precision must be 0.0%, not 100%.
        self._one(tmp_path, passed=True)
        self._one(tmp_path, passed=False, repeats=False)
        text = audit(str(tmp_path / "**" / "state.db"))
        line = next(
            l for l in text.splitlines() if l.startswith("repeated_call")
        )
        assert "  0.0%" in line, line

    def test_S107_lift_is_precision_over_base(self, tmp_path) -> None:
        # Two trials, both failing, both firing: precision 100%, base 100%,
        # so lift is exactly 1.00x. A multiplication would give 1.00x too --
        # so make the base differ from the precision.
        self._one(tmp_path, passed=False)
        self._one(tmp_path, passed=True, repeats=False)
        text = audit(str(tmp_path / "**" / "state.db"))
        line = next(
            l for l in text.splitlines() if l.startswith("repeated_call")
        )
        # By column, not by substring: `"2.00x" in line` was satisfied by the
        # *matched* column while the global one read 0.50x, so turning the
        # division into a multiplication passed.
        _name, fires, precision, global_lift, matched_lift = line.split()
        assert (fires, precision) == ("1", "100.0%"), line
        assert global_lift == "2.00x", line   # precision 100% / base 50%
        assert matched_lift == "2.00x", line

    def test_S107_the_two_lift_columns_are_computed_differently(
        self, tmp_path
    ) -> None:
        # A long failing run that fires, and a short passing one that cannot.
        # Global base is 50%; the matched base -- runs long enough to fire --
        # is 100%. So the columns must disagree, which is the entire point of
        # printing both.
        self._one(tmp_path, passed=False)
        import json

        from harness.persistence import RunStore

        short = tmp_path / "short"
        (short / "agent" / "harness-home").mkdir(parents=True)
        (short / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}),
            encoding="utf-8",
        )
        with RunStore(short / "agent" / "harness-home" / "state.db") as store:
            run_id = store.create_run("goal", "m", "auto")
            agent_id = store.create_agent(run_id, "goal")
            for i in range(5):
                store.append_event(
                    agent_id, "tool_call",
                    {"id": f"s{i}", "name": "bash",
                     "arguments": {"command": f"u{i}"}},
                )
        text = audit(str(tmp_path / "**" / "state.db"))
        line = next(
            l for l in text.splitlines() if l.startswith("repeated_call")
        )
        _n, _f, _p, global_lift, matched_lift = line.split()
        assert global_lift == "2.00x", line
        assert matched_lift == "1.00x", line

    def test_S107_an_all_passing_corpus_does_not_crash(self, tmp_path) -> None:
        # A clean sweep is the one input that divided by zero.
        self._one(tmp_path, passed=True)
        self._one(tmp_path, passed=True, repeats=False)
        text = audit(str(tmp_path / "**" / "state.db"))
        assert "trials scored : 2" in text


class TestTheLoopFeedsBothHalves:
    """`CONSECUTIVE_FAILURE` needs the tool *result*, which arrives on a
    different code path from the call. Constructing a fresh monitor per result
    -- so nothing ever accumulates -- passed the whole suite."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def test_S107_a_repeatedly_failing_command_signals_from_the_loop(
        self, tmp_path
    ) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.permissions import ToolMeta
        from harness.persistence import RunStore
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

        async def failing(arguments: dict) -> str:
            # What a non-zero exit actually looks like coming back: a normal
            # result whose text carries the code. Not `is_error`.
            return "exit code: 1\n--- stdout ---\nboom"

        def bash_like(_deps=None) -> Tool:
            return Tool(
                spec=ToolSpec(name="bash", description="run a command"),
                meta=ToolMeta(side_effect=True),
                handler=failing,
            )

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        script = [
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content="trying",
                    tool_calls=[
                        ToolCall(
                            id=f"c{i}",
                            name="bash",
                            arguments={"command": f"pytest -q t{i}.py"},
                        )
                    ],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            for i in range(4)
        ]
        script.append(
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content="Task complete. I ran the tests and checked them.",
                ),
                usage=Usage(),
                stop_reason=StopReason.END_TURN,
            )
        )
        run_id, _ = await orchestrator.run_task(
            "goal",
            "fake-model",
            adapter_override=FakeAdapter(script),
            budgets=Budgets(max_turns=8),
            tool_factories=[bash_like],
        )
        agent_id = next(
            a.id for a in store.list_agents(run_id) if a.parent_agent_id is None
        )
        signals = [
            e.payload["detector"]
            for e in store.load_events(agent_id)
            if e.kind == STUCK_EVENT
        ]
        assert "consecutive_failure" in signals, signals
