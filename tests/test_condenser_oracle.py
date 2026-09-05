"""S-404: scoring the retention decision against what the run did next.

The eval this replaces would have been a pass-rate A/B of `summarize-halve`
against `summarize-halve+pivotal`. It would have returned "no difference", and
that result would have been uninterpretable: across every run this harness has
recorded — 760 agents, 13,159 model turns — `compaction` has been emitted zero
times, and only 3 of 686 runs would have crossed the threshold at a 128K
window. The treatment never fires.

So the tests here are about whether the *label* is trustworthy, because the
label is the whole eval. The one that matters most is the confound control:
re-reading a file you have since edited is correct behaviour, not evidence
that compaction destroyed something.
"""

from __future__ import annotations

import pytest

from harness.eval.condenser_oracle import (
    OracleReport,
    _turns_from_span,
    render_report,
    score_agent,
)
from harness.types import Message, Role, ToolCall, ToolResult


class _Event:
    """A `TranscriptEvent`-shaped stand-in: the scorer reads only these two."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.kind = kind
        self.payload = payload


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


def read_turn(call_id: str, path: str) -> list[Message]:
    return [
        assistant(
            "reading",
            [ToolCall(id=call_id, name="read_file", arguments={"path": path})],
        ),
        tool(call_id, "file contents"),
    ]


def call_event(name: str, **arguments) -> _Event:
    return _Event("tool_call", {"name": name, "arguments": arguments})


def compaction_event(span: list[Message]) -> _Event:
    return _Event(
        "compaction",
        {
            "evicted_count": len(span),
            "evicted": [m.model_dump(mode="json") for m in span],
            "summary": "[COMPACTION SUMMARY]\n...",
        },
    )


class TestTheLabel:
    def test_S404_a_reread_of_an_evicted_file_is_a_loss(self) -> None:
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [
            compaction_event(span),
            call_event("read_file", path="src/parser.py"),
        ]
        report = score_agent(events)
        assert report.lost_turns == 1
        assert report.loss_rate == pytest.approx(0.5)  # 1 of 2 turns

    def test_S404_a_reread_after_editing_that_file_is_not(self) -> None:
        # THE confound this label exists to exclude. An agent that re-reads a
        # file it has just edited is doing the correct thing -- it is fetching
        # something new, not something compaction destroyed. Without this
        # check, the "loss rate" is really a measure of how much editing the
        # run did, and every edit-heavy run scores as catastrophic loss.
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [
            compaction_event(span),
            call_event("edit_file", path="src/parser.py"),
            call_event("read_file", path="src/parser.py"),
        ]
        report = score_agent(events)
        assert report.lost_turns == 0
        assert report.loss_rate == 0.0

    def test_S404_a_write_to_a_different_file_does_not_excuse_the_reread(
        self,
    ) -> None:
        # The control for the control: matching any write, rather than a write
        # to *that path*, would let one unrelated edit suppress every loss in
        # the run.
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [
            compaction_event(span),
            call_event("edit_file", path="src/lexer.py"),
            call_event("read_file", path="src/parser.py"),
        ]
        assert score_agent(events).lost_turns == 1

    def test_S404_a_file_never_read_again_is_not_a_loss(self) -> None:
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [compaction_event(span), call_event("read_file", path="other.py")]
        assert score_agent(events).lost_turns == 0

    def test_S404_a_read_before_the_compaction_does_not_count(self) -> None:
        # "Afterwards" has to mean afterwards. Scanning the whole run would
        # label every evicted read as rediscovered, because the read that put
        # it in the span is itself in the log.
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [
            call_event("read_file", path="src/parser.py"),
            compaction_event(span),
        ]
        assert score_agent(events).lost_turns == 0

    def test_S404_a_repeated_command_is_counted_but_not_scored(self) -> None:
        # Re-running a test suite after an edit is correct behaviour. Folding
        # it into the primary number would make the loss rate track how often
        # the run tested, which is the opposite of a quality signal.
        span = [
            user("goal"),
            assistant(
                "testing",
                [ToolCall(id="c1", name="bash", arguments={"command": "pytest -q"})],
            ),
            tool("c1", "1 failed"),
        ]
        events = [compaction_event(span), call_event("bash", command="pytest -q")]
        report = score_agent(events)
        assert report.repeated_commands == 1
        assert report.lost_turns == 0
        assert report.loss_rate == 0.0


class TestTheMarkerIsRederivedNotTrusted:
    """The marker is reconstructed from the evicted messages, so this scores
    transcripts recorded before S-105 existed — including every run already on
    disk."""

    def test_S404_a_failing_tool_result_marks_its_turn(self) -> None:
        span = [
            user("goal"),
            assistant("run", [ToolCall(id="c1", name="bash")]),
            tool("c1", "boom", True),
        ]
        turns = _turns_from_span(span)
        assert [t.marked for t in turns] == [None, "tool_error"]

    def test_S404_the_verification_reminder_marks_its_turn(self) -> None:
        from harness.diligence import VERIFICATION_FAILED_REMINDER

        span = [
            user("goal"),
            assistant("done"),
            user(
                VERIFICATION_FAILED_REMINDER.format(
                    command="pytest -q", exit_code=1, output="1 failed"
                )
            ),
        ]
        turns = _turns_from_span(span)
        assert turns[-1].marked == "verification_failed"

    def test_S404_a_successful_turn_is_not_marked(self) -> None:
        span = [user("goal"), *read_turn("c1", "a.py")]
        assert [t.marked for t in _turns_from_span(span)] == [None, None]

    def test_S404_a_turn_is_an_assistant_message_and_its_results(self) -> None:
        span = [
            user("goal"),
            assistant("a", [ToolCall(id="c1", name="bash")]),
            tool("c1", "x"),
            tool("c1", "y"),
            assistant("b"),
        ]
        assert [t.index for t in _turns_from_span(span)] == [0, 1, 2]


class TestTheRecencyBaseline:
    """"Keep the last N turns" costs no marking heuristic. A marker that does
    not beat it is not earning its complexity, so it is scored on the same
    spans rather than in a separate arm."""

    def _report(self, lost_index: int, marked_index: int) -> OracleReport:
        span = [user("goal")]
        for i in range(6):
            span.extend(read_turn(f"c{i}", f"f{i}.py"))
        events = [
            compaction_event(span),
            call_event("read_file", path=f"f{lost_index}.py"),
        ]
        report = score_agent(events, keep=2)
        # Plant the marker by hand: the point here is the comparison, not the
        # marker's own derivation, which is covered above.
        for score in report.compactions:
            for turn in score.turns:
                turn.marked = "tool_error" if turn.index == marked_index else None
        return report

    def test_S404_a_marker_that_names_the_lost_turn_beats_recency(self) -> None:
        report = self._report(lost_index=0, marked_index=1)  # turn 1 holds f0
        assert report.marker_scores() == (1.0, 1.0)
        assert report.recency_scores()[1] == 0.0  # the last 2 turns miss it

    def test_S404_recency_wins_when_the_loss_is_recent(self) -> None:
        report = self._report(lost_index=5, marked_index=1)
        assert report.marker_scores() == (0.0, 0.0)
        assert report.recency_scores()[1] == 1.0

    def test_S404_precision_is_none_when_nothing_was_kept(self) -> None:
        # Not 0.0: a precision of zero is a claim that what was kept was
        # wrong, and nothing was kept, so there was no claim to be wrong
        # about. Reporting 0.0 would make "kept nothing" look like the worst
        # possible strategy rather than an abstention.
        report = score_agent([compaction_event([user("goal")])], keep=0)
        assert report.marker_scores() == (None, None)
        assert report.recency_scores() == (None, None)


class TestTheReportRefusesToBeVacuous:
    def test_S404_no_compaction_says_so_instead_of_printing_zeroes(self) -> None:
        # The failure this whole spec exists because of. A run that never
        # compacted has no retention decision to score, and a report of
        # "loss rate 0.0%, precision n/a" reads as a measurement.
        text = render_report([score_agent([call_event("read_file", path="a.py")])])
        assert "NOTHING COMPACTED" in text
        assert "vacuous" in text

    def test_S404_a_zero_loss_rate_is_distinguished_from_no_data(self) -> None:
        span = [user("goal"), *read_turn("c1", "a.py")]
        text = render_report([score_agent([compaction_event(span)])])
        assert "NOTHING COMPACTED" not in text
        assert "destroyed nothing" in text

    def test_S404_the_denominators_are_always_printed(self) -> None:
        # A precision without the count behind it reads as a measurement when
        # it may be one observation.
        span = [user("goal"), *read_turn("c1", "a.py")]
        text = render_report(
            [score_agent([compaction_event(span), call_event("read_file", path="a.py")])]
        )
        assert "1/2 evicted turns" in text

    def test_S404_loss_rate_is_none_rather_than_zero_with_no_evictions(
        self,
    ) -> None:
        assert score_agent([]).loss_rate is None


class TestTheWindowOverrideActuallyForcesCompaction:
    """Acceptance 2. Asserted rather than assumed: the override is the only
    reason there is anything to score, and a silently ignored parameter would
    produce an eval that reports 'nothing compacted' forever."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the LocalSandbox fallback the way the rest of the suite does.
        Assigning to `DockerSandbox.availability` directly is never reverted
        and leaks into every later module that calls it."""
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def test_S404_a_narrow_window_makes_a_run_compact(
        self, tmp_path
    ) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.persistence import RunStore
        from harness.types import ModelResponse, StopReason, Usage
        from tests.test_condenser import _note_tool

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        script = [
            ModelResponse(
                message=assistant(
                    f"step {i} " + "y" * 800,
                    [ToolCall(id=f"c{i}", name="note", arguments={"text": "ok"})],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            for i in range(30)
        ]

        async def compactions(max_context: int | None) -> int:
            run_id, _ = await orchestrator.run_task(
                "do the thing",
                "fake-model",
                adapter_override=FakeAdapter(list(script)),
                budgets=Budgets(max_turns=20),
                tool_factories=[lambda deps: _note_tool()],
                max_context=max_context,
            )
            agent_id = next(
                a.id for a in store.list_agents(run_id) if a.parent_agent_id is None
            )
            return sum(
                1 for e in store.load_events(agent_id) if e.kind == "compaction"
            )

        # FakeAdapter reports a 1M window, which is why the control compacts
        # zero times -- the same reason every real run has.
        assert await compactions(None) == 0
        assert await compactions(2_000) > 0


class TestEveryRunInAStoreIsScored:
    """A suite writes one run per task into a single store. Scoring only
    `runs[0]` reported one task's numbers as the suite's — with a denominator
    small enough to still read as a measurement."""

    def test_S404_all_lead_agents_are_returned(self, tmp_path) -> None:
        from harness.eval.condenser_oracle import lead_agents
        from harness.persistence import RunStore

        with RunStore(tmp_path / "state.db") as store:
            expected = []
            for i in range(3):
                run_id = store.create_run(f"goal {i}", "m", "auto")
                lead = store.create_agent(run_id, f"goal {i}")
                store.create_agent(run_id, "sub", parent_agent_id=lead)
                store.append_event(lead, "message", {"role": "user"})
                expected.append(lead)
        found = [agent_id for agent_id, _ in lead_agents(str(tmp_path / "state.db"))]
        assert sorted(found) == sorted(expected)


class TestTheLabelCanActuallyFire:
    """The instrument shipped null. `read_file` is 38 of 583 tool calls on the
    click corpus (6.5%), **zero** on the 53-turn exemplar run, and zero of the
    19 evicted turns in the only run that compacted. The agent reads files with
    `sed -n`, `grep`, `head`, `tail`, `cat` — 616 segments against 38
    `read_file` calls — so the loss rate was 0.0 by arithmetic and the report
    narrated it as "compaction destroyed nothing"."""

    def test_S404_a_shell_read_counts_as_a_read(self) -> None:
        from harness.eval.condenser_oracle import paths_read_by

        assert paths_read_by("sed -n '1,80p' tests/test_termui.py") == {
            "tests/test_termui.py"
        }
        assert paths_read_by('grep -n -i "pager" tests/test_termui.py') == {
            "tests/test_termui.py"
        }
        assert paths_read_by("cd /x && cat src/y.py") == {"src/y.py"}

    def test_S404_a_pipe_consumer_is_not_a_read(self) -> None:
        # `head -30` after a pipe is reading stdin. Counting it would
        # attribute a read to whatever path happened to be on the line.
        from harness.eval.condenser_oracle import paths_read_by

        assert paths_read_by("pytest -q | head -30") == set()
        assert paths_read_by("tail -20") == set()

    def test_S404_a_redirect_is_not_a_path(self) -> None:
        from harness.eval.condenser_oracle import paths_read_by

        assert paths_read_by("cat README.md 2>/dev/null") == {"README.md"}

    def test_S404_a_shell_reread_of_an_evicted_shell_read_is_a_loss(
        self,
    ) -> None:
        span = [
            user("goal"),
            assistant(
                "look",
                [
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "sed -n '1,80p' src/parser.py"},
                    )
                ],
            ),
            tool("c1", "..."),
        ]
        events = [
            compaction_event(span),
            call_event("bash", command="grep -n 'def parse' src/parser.py"),
        ]
        assert score_agent(events).lost_turns == 1

    def test_S404_a_report_with_no_scoreable_read_says_so(self) -> None:
        # The guard whose absence let a null instrument print a conclusion.
        span = [
            user("goal"),
            assistant("run", [ToolCall(id="c1", name="bash",
                                       arguments={"command": "pytest -q"})]),
            tool("c1", "ok"),
        ]
        text = render_report([score_agent([compaction_event(span)])])
        assert "LABEL NEVER APPLICABLE" in text
        assert "arithmetic, not evidence" in text
        assert "destroyed nothing" not in text


class TestTheConfoundWindowStartsAtTheRead:
    """The write that excuses a re-read is usually *before* the compaction —
    often inside the evicted span itself. Scanning forward from the compaction
    event missed every one of them, and read → edit → compact → re-read is the
    most common agent shape there is."""

    def test_S404_a_write_inside_the_span_excuses_the_reread(self) -> None:
        span = [
            user("goal"),
            *read_turn("c1", "src/parser.py"),
            assistant(
                "fix",
                [ToolCall(id="c2", name="edit_file",
                          arguments={"path": "src/parser.py"})],
            ),
            tool("c2", "edited"),
        ]
        events = [compaction_event(span), call_event("read_file", path="src/parser.py")]
        assert score_agent(events).lost_turns == 0

    def test_S404_a_write_seen_only_as_a_pre_compaction_call_is_missed(
        self,
    ) -> None:
        # A known limit, asserted rather than left as `in (0, 1)` -- which
        # asserts nothing and would have passed whichever way the code went.
        # Writes are read from the evicted span's own messages, so a write
        # recorded only as a `tool_call` event outside the span is invisible.
        # In production every write before the compaction *is* in the span
        # (compaction evicts a prefix), so this shape does not arise there;
        # it is pinned so the limit is visible rather than latent.
        span = [user("goal"), *read_turn("c1", "src/parser.py")]
        events = [
            call_event("edit_file", path="src/parser.py"),
            compaction_event(span),
            call_event("read_file", path="src/parser.py"),
        ]
        assert score_agent(events).lost_turns == 1

    def test_S404_a_write_after_the_read_but_before_a_later_turn(self) -> None:
        # Only writes at or after the reading turn excuse it: an edit made
        # *earlier* in the span does not make a later read stale.
        span = [
            user("goal"),
            assistant("fix", [ToolCall(id="c0", name="edit_file",
                                       arguments={"path": "a.py"})]),
            tool("c0", "edited"),
            *read_turn("c1", "a.py"),
        ]
        events = [compaction_event(span), call_event("read_file", path="a.py")]
        assert score_agent(events).lost_turns == 1

    def test_S404_multi_edit_counts_as_a_write(self) -> None:
        span = [user("goal"), *read_turn("c1", "a.py")]
        events = [
            compaction_event(span),
            call_event("multi_edit", path="a.py"),
            call_event("read_file", path="a.py"),
        ]
        assert score_agent(events).lost_turns == 0


class TestTheReportIsPinned:
    """`render_report` is the CLI's only output and had five surviving
    mutants, including one that printed the marker's numbers on the recency
    line — the single comparison the eval rests on."""

    def _lossy(self, keep: int):
        span = [user("goal")]
        for i in range(6):
            span.extend(read_turn(f"c{i}", f"f{i}.py"))
        report = score_agent(
            [compaction_event(span), call_event("read_file", path="f5.py")],
            keep=keep,
        )
        for score in report.compactions:
            for turn in score.turns:
                turn.marked = "tool_error" if turn.index == 1 else None
        return report

    def test_S404_the_recency_line_is_not_the_marker_line(self) -> None:
        report = self._lossy(keep=2)
        text = render_report([report])
        marker = next(l for l in text.splitlines() if l.startswith("marker"))
        recency = next(l for l in text.splitlines() if l.startswith("recency"))
        assert "recall:   0.0%" in marker      # marked turn 1, lost turn 6
        assert "recall: 100.0%" in recency     # the last 2 turns hold it
        assert marker.split("precision")[1] != recency.split("precision")[1]

    def test_S404_the_keep_flag_reaches_the_baseline(self) -> None:
        # `render_report` rebuilt the merged report without `keep`, so
        # `--keep 2` printed numbers computed at 4.
        assert "recency-2" in render_report([self._lossy(keep=2)])
        assert "recency-4" in render_report([self._lossy(keep=4)])

    def test_S404_the_repeated_command_count_is_printed(self) -> None:
        span = [
            user("goal"),
            assistant("t", [ToolCall(id="c1", name="bash",
                                     arguments={"command": "pytest -q"})]),
            tool("c1", "1 failed"),
        ]
        text = render_report(
            [score_agent([compaction_event(span), call_event("bash", command="pytest -q")])]
        )
        assert "repeated commands    : 1" in text

    def test_S404_an_empty_span_is_not_read_as_destroyed_nothing(self) -> None:
        # A compaction with no turns gives loss_rate None, not 0.0. `not
        # loss_rate` would print the "destroyed nothing" conclusion over no
        # data at all.
        text = render_report([score_agent([compaction_event([])])])
        assert "destroyed nothing" not in text


class TestTheMarkerIsScoredAsItShips:
    def test_S404_marker_scoring_is_capped_like_the_strategy(self) -> None:
        # `PivotalCondenser` keeps at most `keep` turns, most recent first.
        # Scoring every marked turn measured an idealisation that keeps
        # arbitrarily much and reported its recall as the real one's.
        span = [user("goal")]
        for i in range(6):
            span.extend(read_turn(f"c{i}", f"f{i}.py"))
        report = score_agent(
            [compaction_event(span), call_event("read_file", path="f0.py")],
            keep=2,
        )
        for score in report.compactions:
            for turn in score.turns:
                turn.marked = "tool_error"  # every turn marked
        # Turn 1 holds f0 and is lost, but with keep=2 only turns 5 and 6
        # survive -- so the marker misses it, exactly as the strategy would.
        assert report.marker_scores() == (0.0, 0.0)

    def test_S404_a_turn_is_one_non_tool_message_and_its_results(self) -> None:
        # The oracle now owns this definition. It used to be cross-checked
        # against `condenser._turn_bounds`, which went away with
        # `PivotalCondenser` (S-404 measured it at 0% recall). The rule is
        # unchanged and still load-bearing: splitting only on ASSISTANT folds
        # a USER message into the preceding turn, and a USER message is what
        # the loop used to mark for `verification_failed`. Whatever retention
        # strategy comes next must group the same way, or its precision and
        # recall describe a different object than the one it produces.
        from harness.eval.condenser_oracle import _group_turns

        span = [
            user("goal"),
            assistant("a", [ToolCall(id="c1", name="bash")]),
            tool("c1", "x"),
            user("VERIFICATION FAILED"),
            assistant("b", [ToolCall(id="c2", name="read_file")]),
            tool("c2", "y"),
        ]
        bounds, start = [], 0
        for group in _group_turns(span):
            bounds.append((start, start + len(group)))
            start += len(group)
        assert bounds == [(0, 1), (1, 3), (3, 4), (4, 6)]

class TestTheRecencyWidthAndCommandKey:
    def test_S404_the_recency_window_is_exactly_keep_turns(self) -> None:
        from harness.eval.condenser_oracle import _recency_kept, EvictedTurn

        turns = [EvictedTurn(index=i) for i in range(6)]
        assert _recency_kept(turns, 2) == {4, 5}
        assert _recency_kept(turns, 3) == {3, 4, 5}
        assert _recency_kept(turns, 0) == set()

    def test_S404_the_command_key_is_two_tokens_not_one(self) -> None:
        # 46 of 62 tool calls in the exemplar run are `bash` starting with
        # `cd <path> && ...`. With a one-token key everything collides on
        # `cd` and `repeated_commands` becomes noise.
        from harness.eval.condenser_oracle import _command_key

        a = _command_key({"command": "cd /x && pytest -q"})
        b = _command_key({"command": "cd /x && ls"})
        assert a != b
        assert _command_key({"command": "pytest -q tests/x.py"}) == "pytest -q"

    def test_S404_an_errored_read_does_not_record_its_path(self) -> None:
        # The agent guessed a wrong path. Recording it would make a later
        # successful read of that path score as a loss for content the run
        # never had.
        span = [
            user("goal"),
            assistant("look", [ToolCall(id="c1", name="read_file",
                                        arguments={"path": "nope.py"})]),
            tool("c1", "no such file", True),
        ]
        assert [sorted(t.reads) for t in _turns_from_span(span)] == [[], []]

    async def test_S404_the_forced_window_is_recorded_on_the_run(
        self, tmp_path
    ) -> None:
        # `condenser-oracle` globs `state.db` files. Without this event it
        # cannot tell a forced-window run from a production one, and would
        # pool them into one denominator and report the mixture.
        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.loop import Budgets
        from harness.orchestrator import WINDOW_OVERRIDE_EVENT, Orchestrator
        from harness.persistence import RunStore
        from harness.types import ModelResponse, StopReason, Usage

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        script = [
            ModelResponse(
                message=assistant("Task complete. Did the thing."),
                usage=Usage(),
                stop_reason=StopReason.END_TURN,
            )
        ]

        async def events_for(max_context):
            run_id, _ = await orchestrator.run_task(
                "goal", "fake-model",
                adapter_override=FakeAdapter(list(script)),
                budgets=Budgets(max_turns=3), max_context=max_context,
            )
            agent_id = next(
                a.id for a in store.list_agents(run_id) if a.parent_agent_id is None
            )
            return [
                e for e in store.load_events(agent_id)
                if e.kind == WINDOW_OVERRIDE_EVENT
            ]

        assert await events_for(None) == []
        (recorded,) = await events_for(4_000)
        assert recorded.payload["max_context"] == 4_000


class TestOneFileTwoSpellings:
    """The agent reads one file two ways: `read_file` gets a workspace-relative
    path, a shell command routinely carries the absolute one. Measured on the
    corpus, 35 of 51 distinct shell-read paths never matched a tool path by
    string equality — every one a rediscovery the label could not see."""

    def test_S404_an_absolute_path_matches_its_relative_form(self) -> None:
        from harness.eval.condenser_oracle import same_file

        assert same_file("/tmp/work/t0/tests/test_x.py", "tests/test_x.py")
        assert same_file("tests/test_x.py", "/tmp/work/t0/tests/test_x.py")
        assert same_file("a/b/c.py", "a/b/c.py")

    def test_S404_a_shared_basename_is_not_a_shared_file(self) -> None:
        from harness.eval.condenser_oracle import same_file

        assert not same_file("src/click/utils.py", "tests/utils.py")
        assert not same_file("src/a.py", "src/b.py")
        assert not same_file("", "tests/x.py")

    def test_S404_a_shell_reread_matches_an_absolute_tool_read(self) -> None:
        span = [user("goal"), *read_turn("c1", "tests/test_x.py")]
        events = [
            compaction_event(span),
            call_event(
                "bash",
                command="sed -n '1,50p' /tmp/work/t0/tests/test_x.py",
            ),
        ]
        assert score_agent(events).lost_turns == 1

    def test_S404_the_write_check_matches_across_spellings_too(self) -> None:
        # Otherwise the confound control fails open: an edit by one spelling
        # would not excuse a re-read by the other.
        span = [user("goal"), *read_turn("c1", "tests/test_x.py")]
        events = [
            compaction_event(span),
            call_event("edit_file", path="/tmp/work/t0/tests/test_x.py"),
            call_event("read_file", path="tests/test_x.py"),
        ]
        assert score_agent(events).lost_turns == 0


class TestTheWindowIsReadBackNotJustWritten:
    """`context_window_override` was emitted, registered in
    `EVENT_KIND_SPECS`, and asserted by a test — and read by nothing, while
    `render_report` pooled the runs it existed to separate. That is the defect
    archetype this whole spec is about, in code added to prevent it."""

    def test_S404_the_window_is_recovered_from_the_events(self) -> None:
        from harness.eval.condenser_oracle import window_of
        from harness.orchestrator import WINDOW_OVERRIDE_EVENT

        assert window_of([]) is None
        assert window_of([call_event("bash", command="ls")]) is None
        assert (
            window_of([_Event(WINDOW_OVERRIDE_EVENT, {"max_context": 6000})])
            == 6000
        )

    def test_S404_a_scored_run_carries_its_window(self) -> None:
        from harness.orchestrator import WINDOW_OVERRIDE_EVENT

        span = [user("goal"), *read_turn("c1", "a.py")]
        events = [
            _Event(WINDOW_OVERRIDE_EVENT, {"max_context": 6000}),
            compaction_event(span),
        ]
        assert score_agent(events).window == 6000

    def test_S404_mixed_windows_are_called_out(self) -> None:
        # Pooling a 128K run with a 6K one puts runs that *could not* compact
        # into the same denominator as runs that did.
        from harness.orchestrator import WINDOW_OVERRIDE_EVENT

        span = [user("goal"), *read_turn("c1", "a.py")]
        forced = score_agent(
            [
                _Event(WINDOW_OVERRIDE_EVENT, {"max_context": 6000}),
                compaction_event(span),
                call_event("read_file", path="a.py"),
            ]
        )
        native = score_agent([compaction_event(span)])
        text = render_report([forced, native])
        assert "MIXED WINDOWS" in text
        assert "6,000" in text and "model's own window" in text
        assert "MIXED WINDOWS" not in render_report([forced, forced])


class TestTheDegenerateBaselineIsLabelled:
    """The first reading of this eval reported recency-1 at 81% recall. 12 of
    its 17 true positives came from spans too short for that baseline to be a
    condensation at all — where it keeps everything and scores for free."""

    def _span(self, turns: int):
        span = [user("goal")]
        for i in range(turns - 1):
            span.extend(read_turn(f"c{i}", f"f{i}.py"))
        return span

    def test_S404_feasibility_is_one_plus_keep_below_the_span(self) -> None:
        from harness.eval.condenser_oracle import EvictedTurn, is_feasible

        turns = [EvictedTurn(index=i) for i in range(3)]
        assert not is_feasible(turns, 2)  # 1 + 2 == 3, no shrink
        assert is_feasible(turns, 1)  # 1 + 1 < 3
        assert not is_feasible(turns[:2], 1)

    def test_S404_the_report_names_the_infeasible_spans(self) -> None:
        events = [
            compaction_event(self._span(2)),
            call_event("read_file", path="f0.py"),
        ]
        text = render_report([score_agent(events, keep=1)])
        assert "too short for recency-1 to be a condensation" in text
        assert "scores for free" in text

    def test_S404_the_feasible_row_excludes_them(self) -> None:
        report = score_agent(
            [compaction_event(self._span(2)), call_event("read_file", path="f0.py")],
            keep=1,
        )
        assert report.feasible_spans == 0
        # All-spans recall counts it; the feasible row has nothing to score.
        assert report.recency_scores()[1] == 1.0
        assert report.feasible_recency_scores() == (None, None)


class TestTheParserGapsThatWereMeasured:
    """Two defects the corpus surfaced, both of which silently shrank the
    label. Pinned with the exact strings that produced them."""

    def test_S404_an_escaped_alternation_does_not_shred_the_path(self) -> None:
        # `_SEGMENTS` split on a bare `|`, which cut through the quoted
        # `"pager\|PAGER"` and lost the path entirely. 20 of 127 real bash
        # calls carry this shape, including reads of the two most-read files
        # in the corpus.
        from harness.eval.condenser_oracle import paths_read_by

        assert paths_read_by(
            'grep -n "pager\\|Pager\\|PAGER" src/click/_termui_impl.py | head -60'
        ) == {"src/click/_termui_impl.py"}
        assert paths_read_by(
            'grep -n -A15 "pytest\\|mypy" pyproject.toml | sed -n 1,80p'
        ) == {"pyproject.toml"}

    def test_S404_a_pipeline_still_does_not_read_its_consumer(self) -> None:
        # The reason the bare-`|` split existed. Removing it is safe because
        # only the segment's first token is consulted.
        from harness.eval.condenser_oracle import paths_read_by

        assert paths_read_by("pytest -q tests/x.py | head -30") == set()
        assert paths_read_by("cat src/a.py | grep foo") == {"src/a.py"}

    def test_S404_a_bare_basename_does_not_match_a_real_file(self) -> None:
        # `grep -v types.py` is an exclusion pattern the parser reads as a
        # path. Suffix-matching it against `src/click/types.py` was the only
        # spurious pair in the whole corpus.
        from harness.eval.condenser_oracle import same_file

        assert not same_file("types.py", "src/click/types.py")
        assert same_file("tests/t.py", "/workspace/tests/t.py")
        assert same_file("README.md", "README.md")

    def test_S404_the_component_boundary_is_required(self) -> None:
        from harness.eval.condenser_oracle import same_file

        assert not same_file("ils.py", "src/click/utils.py")
        assert not same_file("click/utils.py", "src/xclick/utils.py")
        assert same_file("click/utils.py", "src/click/utils.py")

    def test_S404_a_missing_database_is_not_reported_as_no_compaction(
        self, tmp_path, capsys
    ) -> None:
        # `RunStore` creates the file, so the report used to blame the window
        # for a typo'd path -- the one place a "no data" tool gives the wrong
        # reason for no data.
        from harness.eval.condenser_oracle import main

        missing = tmp_path / "typo.db"
        main([str(missing)])
        out = capsys.readouterr().out
        assert "no such database" in out
        assert not missing.exists(), "the scorer created the database it read"


class TestTheReachTableIsDeterministic:
    """Two reviewers got different reach tables from the same run. The cause
    was the summarizer stand-in: each summary lands in the transcript the next
    compaction measures, so its size compounds."""

    def test_S404_the_standins_are_pinned_to_the_corpus(self) -> None:
        from harness.eval.condenser_oracle import (
            SUMMARY_STANDIN_CHARS,
            SYSTEM_PROMPT_STANDIN_CHARS,
        )
        from tests.conformance.replay import SUMMARY_CHARS

        assert SUMMARY_STANDIN_CHARS == SUMMARY_CHARS
        assert SYSTEM_PROMPT_STANDIN_CHARS == 2_365

    def test_S404_reach_does_not_import_the_test_package(self) -> None:
        # It used to import `tests.conformance.replay`, and `pyproject.toml`
        # packages `harness*` only -- so the command that generates the
        # spec's window table, added so six empirical claims would be
        # re-runnable, ran from exactly one directory.
        import inspect

        from harness.eval import condenser_oracle

        source = inspect.getsource(condenser_oracle)
        assert "from tests." not in source
        assert "import tests" not in source
