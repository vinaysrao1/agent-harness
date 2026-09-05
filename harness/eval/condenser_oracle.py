"""Score a compaction's retention decision against what the run did next (S-404).

S-105 shipped `PivotalCondenser` with unit tests and no evidence. The obvious
way to get evidence — an A/B of pass rates with and without retention — would
have measured a treatment that never fires: across every run this harness has
recorded (738 agents, 12,729 model turns) `compaction` has been emitted
**zero** times, and only 4 of 686 runs would have crossed the threshold at a
128K window.

This scorer then killed the strategy it was written to measure. On 12 tasks at
a forced 6K window: **loss rate 21%** (compaction does destroy things runs come
back for), **marker recall 0%** (it marked 1 turn of 100, and not one of the
21), against **81% for keeping the single newest turn**. The marker watched
errors; what runs lose is knowledge. `PivotalCondenser` is gone; this stays,
because the next strategy needs the same measurement.

So this scores the decision directly instead, offline, against a label derived
from the run's own subsequent behaviour. No model calls beyond the runs already
on disk.

**The label.** For each compaction, walk forward and ask which evicted turns
the run demonstrably needed again:

- `REDISCOVERED_READ` — the run reads a file it had already read inside the
  evicted span, with no intervening write to that path. Unambiguous repeated
  work: the content did not change, and the run went back for it because it no
  longer had it.
- `REPEATED_COMMAND` — the run re-issues a command whose head and first
  argument appeared in the span. Reported separately and never folded into the
  primary number, because re-running a test suite after an edit is correct
  behaviour, not evidence of loss.

Both are proxies for "the run needed this and no longer had it", and are named
as proxies rather than as truth.

**What is scored.** The marker is re-derived from the evicted messages
themselves — `is_error` is on the message, and the verification reminder is
matched by its own prefix — so nothing has to have been persisted and this
runs against transcripts recorded before S-105 existed.

**The baseline that matters.** "Keep the last N turns" costs no marking
heuristic at all. A marker that does not beat it is not earning its
complexity, so it is computed on the same spans rather than in a separate arm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from harness.orchestrator import WINDOW_OVERRIDE_EVENT
from harness.types import Message, Role

__all__ = [
    "EvictedTurn",
    "CompactionScore",
    "OracleReport",
    "DEFAULT_RECENCY_KEEP",
    "is_feasible",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "lead_agents",
    "same_file",
    "score_agent",
    "render_report",
]

#: Tools whose call means "the agent looked at this path".
READ_TOOLS = frozenset({"read_file"})

#: Shell verbs that read a file. `read_file` alone was not the label -- it was
#: a null instrument. On the click corpus `read_file` is 38 of 583 tool calls
#: (6.5%), it is **zero** on the 53-turn exemplar run, and zero of the 19
#: evicted turns in the only compacting run carried one. The agent reads files
#: with `sed -n`, `grep`, `head`, `tail`, `cat` -- 616 segments against 38
#: `read_file` calls -- so the loss rate was 0.0 by arithmetic and the report
#: narrated that as "compaction destroyed nothing".
READ_VERBS = frozenset(
    {"cat", "sed", "head", "tail", "grep", "less", "more", "awk", "rg", "nl"}
)

#: How many turns the recency baseline keeps by default. Its own constant now:
#: it used to borrow `DEFAULT_RECENCY_KEEP` from the strategy it was benchmarking,
#: which S-404 then deleted. On the measured corpus this value is degenerate --
#: spans average 2.9 turns, so keeping 4 keeps 89% of them -- and `--keep 1` is
#: the row worth reading.
DEFAULT_RECENCY_KEEP = 4

#: Splits a command line into the segments a verb can start. Deliberately
#: **not** on a bare `|`: `grep -n "pager\|PAGER" src/x.py` carries an escaped
#: alternation inside quotes, and splitting there shredded it and lost the
#: path entirely -- 20 of 127 real bash calls, including the two most-read
#: files in the corpus. A pipeline needs no split anyway, because only the
#: segment's *first* token is consulted: `pytest -q | head -30` starts with
#: `pytest`, so nothing is read, and `cat x.py | head -60` starts with `cat`.
_SEGMENTS = re.compile(r"&&|\|\||;|\n")

#: Tools whose call means "the agent changed this path", so a later read of it
#: is fetching something new rather than something it lost.
WRITE_TOOLS = frozenset({"write_file", "edit_file", "multi_edit"})

#: First line of the loop's failed-verification reminder. Matching the whole
#: template would break on every wording change; this prefix is what the
#: message is.
_VERIFICATION_PREFIX = "Your declared verification command failed"


def same_file(left: str, right: str) -> bool:
    """Whether two spellings name the same file.

    The agent reads one file two ways. `read_file` gets a workspace-relative
    path (`tests/test_commands.py`); a shell command routinely carries the
    absolute one (`/private/tmp/.../work/<task>/tests/test_commands.py`),
    because the agent pastes what `find` or an error message gave it. Measured
    on the corpus: 35 of 51 distinct shell-read paths never matched a tool
    path by string equality, and the top ones are exactly this. Every one of
    those was a rediscovery the label could not see.

    So: equal, or one is a suffix of the other on a component boundary *and*
    carries a directory of its own. The last clause matters: a bare basename
    is usually not a path at all but a grep pattern the parser mistook for
    one (`grep -v types.py`), and without it that phantom matched the real
    `src/click/types.py` -- the only spurious pair in the whole corpus.

    Two files with the same trailing components in different subtrees would
    still be conflated; inside one repo checkout that is rare (click's
    `tests/conftest.py` and `tests/typing/conftest.py` do not collide, because
    the boundary is required), and a missed rediscovery is the worse error for
    an eval whose pre-registered conclusion is "there is nothing here".
    """
    if left == right:
        return True
    a, b = left.rstrip("/"), right.rstrip("/")
    if len(a) > len(b):
        a, b = b, a
    return "/" in a and b.endswith("/" + a)


def _path_of(arguments: dict) -> str | None:
    # `path` only: every path-taking tool in the registry (`read_file`,
    # `write_file`, `edit_file`, `multi_edit`) names it that. A `file_path`
    # fallback was carried for a tool that does not exist.
    value = arguments.get("path")
    return value if isinstance(value, str) and value else None


def _looks_like_a_path(token: str) -> bool:
    """A bare token that names a file rather than a flag or a pattern.

    Deliberately crude. The alternative is a shell parser, and the cost of a
    wrong answer here is a mislabelled turn in an eval, not a wrong edit.
    """
    if not token or token[0] in "-<>$'\"" or "=" in token:
        return False
    if token.startswith("/dev/") or ">" in token or "<" in token:
        return False
    return "/" in token or ("." in token[1:] and not token.endswith("."))


def paths_read_by(command: str) -> set[str]:
    """Files a shell command reads, best effort.

    Only segments that *start* with a read verb: `head -30` after a pipe is
    consuming stdin, not opening a file, and counting it would attribute a
    read to whatever path happened to appear elsewhere on the line.
    """
    found: set[str] = set()
    for segment in _SEGMENTS.split(command or ""):
        tokens = segment.split()
        if not tokens or tokens[0] not in READ_VERBS:
            continue
        found.update(t.strip("'\"") for t in tokens[1:] if _looks_like_a_path(t))
    return found


def _paths_touched(name: str, arguments: dict) -> set[str]:
    """Paths one tool call read, whichever tool it was."""
    if name in READ_TOOLS:
        path = _path_of(arguments)
        return {path} if path else set()
    if name == "grep" or name == "glob":
        path = _path_of(arguments) or arguments.get("pattern")
        return {path} if isinstance(path, str) and "/" in path else set()
    if name == "bash":
        command = arguments.get("command")
        return paths_read_by(command) if isinstance(command, str) else set()
    return set()


#: Size of the stand-in system prompt `--reach` replays against. The real
#: assembled `CODING` prompt is ~2,365 characters; this matches it so the
#: table is comparable, without importing the test fixtures that build it.
SYSTEM_PROMPT_STANDIN_CHARS = 2_365

#: Size of the stand-in summary each condensation produces. Not cosmetic: the
#: summary is in the transcript the next compaction measures, so this
#: compounds. Two reviewers got different reach tables from the same run for
#: exactly this reason before it was pinned.
SUMMARY_STANDIN_CHARS = 2_000

#: Splits a command line into the steps a `&&` chain runs, keeping pipelines
#: whole: the work in `pytest -q | head -30` is the pytest, not the head.
_STEPS = re.compile(r"&&|\|\||;|\n")


def _command_key(arguments: dict) -> str | None:
    """A command's real head plus its first argument, e.g. ``pytest -q``.

    The *last* step of a `&&` chain, not the first token of the line. 46 of
    the 62 tool calls in the exemplar run are `bash` starting with
    `cd <path> && ...`, so keying on the line's head collided every single one
    of them on `cd <path>` and made `repeated_commands` pure noise.

    Two tokens, not one: every call would otherwise collide on its verb. Not
    the whole line either -- an agent rerunning the same test with one flag
    changed is doing the same work, and a key that distinguished them would
    score that as new.
    """
    command = arguments.get("command")
    if not isinstance(command, str):
        return None
    steps = [step for step in _STEPS.split(command) if step.split()]
    if not steps:
        return None
    return " ".join(steps[-1].split()[:2])


@dataclass
class EvictedTurn:
    """One assistant turn inside an evicted span, and what it touched."""

    #: Position of the turn within the span, 0 = oldest.
    index: int
    #: Paths this turn read.
    reads: frozenset[str] = frozenset()
    #: Command keys this turn ran.
    commands: frozenset[str] = frozenset()
    #: Paths this turn *wrote*. A later read of one of these fetches new
    #: content, so it is not a rediscovery -- and the write is frequently
    #: inside the evicted span itself, which is why scanning only forward
    #: from the compaction was not enough.
    writes: frozenset[str] = frozenset()
    #: Why the marker would have kept it, or None.
    marked: str | None = None
    #: Set when the run later read one of `reads` with no intervening write.
    rediscovered_read: str | None = None
    #: Set when the run later re-issued one of `commands`.
    repeated_command: str | None = None

    @property
    def needed_again(self) -> bool:
        """The primary label. Deliberately *not* `or repeated_command`."""
        return self.rediscovered_read is not None


def _group_turns(span: Sequence[Message]) -> list[list[Message]]:
    """A turn is one **non-TOOL** message plus the tool results following it.

    This used to mirror `condenser._turn_bounds`, which went away with
    `PivotalCondenser`; the oracle now owns the definition. The rule is
    unchanged and still load-bearing. Splitting only on ASSISTANT folded a
    USER message into the preceding assistant turn, and the USER message was
    exactly what the loop marked for `verification_failed` -- so the oracle
    credited the marker with retaining a whole assistant turn and its tool
    output while the strategy retained the lone reminder and dropped the turn.
    Precision and recall were computed over an object the shipping strategy
    did not produce.

    Whatever retention strategy comes next has to group the same way, for the
    same reason.
    """
    groups: list[list[Message]] = []
    current: list[Message] = []
    for message in span:
        if message.role is not Role.TOOL and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def _turns_from_span(span: Sequence[Message]) -> list[EvictedTurn]:
    """Group an evicted span into turns and record what each one touched."""
    turns: list[EvictedTurn] = []
    for index, group in enumerate(_group_turns(span)):
        reads: set[str] = set()
        writes: set[str] = set()
        commands: set[str] = set()
        marked: str | None = None
        pending: dict[str, tuple[str, dict]] = {}
        for message in group:
            if message.role is Role.ASSISTANT:
                pending = {
                    call.id: (call.name, call.arguments)
                    for call in message.tool_calls
                }
            elif message.role is Role.TOOL and message.tool_result is not None:
                result = message.tool_result
                name, arguments = pending.get(result.tool_call_id, ("", {}))
                if result.is_error:
                    # First reason wins, matching what the deleted
                    # `mark_pivotal` did: a turn marked as a failed
                    # verification must not be relabelled a generic error.
                    marked = marked or "tool_error"
                    continue
                reads.update(_paths_touched(name, arguments))
                if name in WRITE_TOOLS:
                    path = _path_of(arguments)
                    if path:
                        writes.add(path)
                if name == "bash":
                    key = _command_key(arguments)
                    if key:
                        commands.add(key)
            elif message.role is Role.USER:
                if _VERIFICATION_PREFIX in (message.content or ""):
                    marked = marked or "verification_failed"
        turns.append(
            EvictedTurn(
                index=index,
                reads=frozenset(reads),
                writes=frozenset(writes),
                commands=frozenset(commands),
                marked=marked,
            )
        )
    return turns


@dataclass
class CompactionScore:
    """One compaction's evicted span, labelled."""

    turns: list[EvictedTurn] = field(default_factory=list)

    @property
    def needed_again(self) -> list[EvictedTurn]:
        return [t for t in self.turns if t.needed_again]


def _recency_kept(turns: Sequence[EvictedTurn], keep: int) -> set[int]:
    """The baseline: the last `keep` turns of the span, no heuristic at all."""
    return {t.index for t in turns[-keep:]} if keep > 0 else set()


def is_feasible(turns: Sequence[EvictedTurn], keep: int) -> bool:
    """Whether keeping `keep` turns of this span is a condensation at all.

    A condensation replaces `len(span)` messages with a summary plus what it
    keeps, so it only shrinks the view while `1 + keep < len(span)`. Spans
    here average 2.9 turns, so `keep=4` is "keep everything" on almost all of
    them and scores 100% recall for free.

    This is not a footnote. The first reading of this eval reported recency-1
    at 81% recall; 12 of its 17 true positives came from spans where that
    baseline could not have run, and on the feasible ones it is nearer half
    that. A baseline that is infeasible is not a cheaper alternative -- it is
    no alternative.
    """
    return 1 + keep < len(turns)


@dataclass
class OracleReport:
    """What the marker got right, and whether it beat doing nothing clever."""

    agent_id: str
    compactions: list[CompactionScore] = field(default_factory=list)
    #: The forced window this run used, or None for the model's own.
    window: int | None = None
    #: Turns whose commands reappeared. Reported, never scored: a test suite
    #: re-run after an edit is correct behaviour, not evidence of loss.
    repeated_commands: int = 0
    keep: int = DEFAULT_RECENCY_KEEP

    @property
    def evicted_turns(self) -> int:
        return sum(len(c.turns) for c in self.compactions)

    @property
    def lost_turns(self) -> int:
        return sum(len(c.needed_again) for c in self.compactions)

    @property
    def loss_rate(self) -> float | None:
        """Evicted turns the run demonstrably needed again.

        `None`, not zero, when nothing was evicted: "no compaction happened"
        and "compaction destroyed nothing that mattered" are different
        findings and only one of them is about the condenser.
        """
        total = self.evicted_turns
        return self.lost_turns / total if total else None

    def _confusion(self, kept: Iterable[set[int]]) -> tuple[int, int, int]:
        """(true positives, kept, needed) over every compaction."""
        tp = keptn = needed = 0
        for score, keep_set in zip(self.compactions, kept):
            lost = {t.index for t in score.needed_again}
            tp += len(lost & keep_set)
            keptn += len(keep_set)
            needed += len(lost)
        return tp, keptn, needed

    def marker_scores(self) -> tuple[float | None, float | None]:
        """What `PivotalCondenser` would actually have kept.

        Capped and most-recent-first, because that is what the strategy does.
        Scoring every marked turn measured an idealisation that keeps
        arbitrarily much, and reported its recall as the real one's.
        """
        kept = [
            {
                t.index
                for t in [x for x in c.turns if x.marked is not None][
                    -self.keep :
                ]
            }
            if self.keep > 0
            else set()
            for c in self.compactions
        ]
        return _ratios(*self._confusion(kept))

    def recency_scores(self) -> tuple[float | None, float | None]:
        kept = [_recency_kept(c.turns, self.keep) for c in self.compactions]
        return _ratios(*self._confusion(kept))

    def feasible_recency_scores(self) -> tuple[float | None, float | None]:
        """Recency, scored only where it would actually have condensed."""
        spans = [c for c in self.compactions if is_feasible(c.turns, self.keep)]
        kept = [_recency_kept(c.turns, self.keep) for c in spans]
        tp = keptn = needed = 0
        for score, keep_set in zip(spans, kept):
            lost = {t.index for t in score.needed_again}
            tp += len(lost & keep_set)
            keptn += len(keep_set)
            needed += len(lost)
        return _ratios(tp, keptn, needed)

    @property
    def feasible_spans(self) -> int:
        return sum(
            1 for c in self.compactions if is_feasible(c.turns, self.keep)
        )


def _ratios(tp: int, kept: int, needed: int) -> tuple[float | None, float | None]:
    """(precision, recall). `None` where the denominator is zero -- a
    precision of 0.0 for a strategy that kept nothing is a claim about
    accuracy, and there was no claim to be wrong about."""
    return (tp / kept if kept else None, tp / needed if needed else None)


def score_agent(
    events: Sequence[object], agent_id: str = "", keep: int = DEFAULT_RECENCY_KEEP
) -> OracleReport:
    """Label every compaction in one agent's event log.

    ``events`` are `TranscriptEvent`-shaped: `.kind` and `.payload`, in `seq`
    order. Reads `compaction` payloads for the evicted spans and `tool_call` /
    `tool_result` for what happened afterwards.
    """
    report = OracleReport(
        agent_id=agent_id, keep=keep, window=window_of(events)
    )

    # Every tool call the run made, with the sequence position it was made at,
    # so "afterwards" can be asked of a specific compaction.
    calls: list[tuple[int, str, dict]] = []
    compactions: list[tuple[int, list[Message]]] = []
    for position, event in enumerate(events):
        kind = getattr(event, "kind", None)
        payload = getattr(event, "payload", None) or {}
        if kind == "tool_call":
            arguments = payload.get("arguments")
            calls.append(
                (
                    position,
                    str(payload.get("name") or ""),
                    arguments if isinstance(arguments, dict) else {},
                )
            )
        elif kind == "compaction":
            evicted = payload.get("evicted") or []
            span = [Message.model_validate(m) for m in evicted]
            compactions.append((position, span))

    for position, span in compactions:
        turns = _turns_from_span(span)
        score = CompactionScore(turns=turns)
        later = [c for c in calls if c[0] > position]
        for turn in turns:
            # Writes *after this turn but still inside the span* count too.
            # Scanning only forward from the compaction missed them, and
            # read -> edit -> compact -> re-read is the single most common
            # agent shape there is: it was scored as a rediscovery every
            # time, when the re-read was fetching content the agent had
            # itself just changed.
            written_in_span = {
                path
                for later_turn in turns[turn.index :]
                for path in later_turn.writes
            }
            for path in sorted(turn.reads):
                if any(same_file(path, w) for w in written_in_span):
                    continue
                written = False
                for _pos, name, arguments in later:
                    target = _path_of(arguments)
                    if name in WRITE_TOOLS and target and same_file(target, path):
                        written = True
                    elif any(
                        same_file(path, t)
                        for t in _paths_touched(name, arguments)
                    ):
                        if not written:
                            turn.rediscovered_read = path
                        break
                if turn.rediscovered_read:
                    break
            for key in turn.commands:
                if any(
                    name == "bash" and _command_key(arguments) == key
                    for _pos, name, arguments in later
                ):
                    turn.repeated_command = key
                    break
        report.repeated_commands += sum(
            1 for t in score.turns if t.repeated_command
        )
        report.compactions.append(score)
    return report


def render_report(reports: Sequence[OracleReport]) -> str:
    """One block per aggregate. Prints the denominators, always.

    A precision without the count behind it reads as a measurement when it may
    be one observation.
    """
    evicted = sum(r.evicted_turns for r in reports)
    lost = sum(r.lost_turns for r in reports)
    compactions = sum(len(r.compactions) for r in reports)
    repeated = sum(r.repeated_commands for r in reports)

    keeps = {r.keep for r in reports}
    assert len(keeps) <= 1, f"mixed --keep values in one report: {keeps}"
    # `keep` has to be carried: rebuilding the merged report without it meant
    # `--keep 2` printed `recency-4` numbers computed at 4, so the one
    # comparison the eval rests on silently ignored the flag.
    merged = OracleReport(
        agent_id="(all)", keep=keeps.pop() if keeps else DEFAULT_RECENCY_KEEP
    )
    for r in reports:
        merged.compactions.extend(r.compactions)
    m_precision, m_recall = merged.marker_scores()
    r_precision, r_recall = merged.recency_scores()
    f_precision, f_recall = merged.feasible_recency_scores()

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:6.1%}"

    lines = [
        f"runs scored          : {len(reports)}",
        f"compactions          : {compactions}",
        f"evicted turns        : {evicted}",
        "",
        f"loss rate            : {pct(merged.loss_rate)}"
        f"   ({lost}/{evicted} evicted turns were read again, unmodified)",
        f"repeated commands    : {repeated}"
        "   (reported only -- a rerun after an edit is correct behaviour)",
        "",
        f"marker     precision : {pct(m_precision)}   recall: {pct(m_recall)}"
        f"   (over all {compactions} spans)",
        f"recency-{merged.keep}  precision : {pct(r_precision)}   recall: {pct(r_recall)}"
        f"   (over all {compactions} spans)",
        f"recency-{merged.keep}  precision : {pct(f_precision)}   recall: {pct(f_recall)}"
        f"   (over the {merged.feasible_spans} spans where keeping "
        f"{merged.keep} would actually condense)",
    ]
    if merged.feasible_spans < compactions:
        lines.append("")
        lines.append(
            f"{compactions - merged.feasible_spans} of {compactions} spans are "
            f"too short for recency-{merged.keep} to be a condensation at all "
            f"(1 + {merged.keep} >= span length): on those it keeps everything "
            "and scores for free. Read the feasible row."
        )
    windows = {r.window for r in reports}
    if len(windows) > 1:
        lines.append("")
        lines.append(
            "MIXED WINDOWS: "
            + ", ".join(
                f"{sum(1 for r in reports if r.window == w)} run(s) at "
                + ("the model's own window" if w is None else f"{w:,}")
                for w in sorted(windows, key=lambda w: (w is None, w or 0))
            )
            + ". Compaction only fires below ~10-12K on this workload, so "
            "pooling these puts runs that could not compact into the same "
            "denominator as runs that did."
        )

    scoreable = sum(
        len(t.reads) for r in reports for c in r.compactions for t in c.turns
    )
    if not compactions:
        lines.append("")
        lines.append(
            "NOTHING COMPACTED. Every number above is vacuous. Re-run with a "
            "smaller --max-context; without a compaction there is no retention "
            "decision to score."
        )
    elif not scoreable:
        # The failure this guard exists for actually shipped: with the label
        # reading only `read_file`, zero of 19 evicted turns carried a path,
        # the loss rate was 0.0 by arithmetic, and the report below printed
        # "compaction destroyed nothing" -- a conclusion, from an instrument
        # whose response was identically zero.
        lines.append("")
        lines.append(
            "LABEL NEVER APPLICABLE. Not one evicted turn read a file, so no "
            "run behaviour could have produced a non-zero loss rate. The 0.0% "
            "above is arithmetic, not evidence."
        )
    elif merged.loss_rate == 0:
        lines.append("")
        lines.append(
            "Loss rate is zero: compaction destroyed nothing this run went "
            "back for. There is no harm here for retention to prevent."
        )
    return "\n".join(lines)


def window_of(events: Sequence[object]) -> int | None:
    """The forced context window this run used, or None for the model's own.

    Read from the `context_window_override` event (S-404). The event existed
    for two rounds without a reader, which is the defect archetype this whole
    spec is about: it was emitted, registered, tested, and consumed by
    nothing, while `render_report` pooled every run it was meant to separate.
    """
    for event in events:
        if getattr(event, "kind", None) == WINDOW_OVERRIDE_EVENT:
            value = (getattr(event, "payload", None) or {}).get("max_context")
            if isinstance(value, int):
                return value
    return None


def lead_agents(db_path: str) -> list[tuple[str, list]]:
    """``(agent_id, events)`` for every run's lead agent in one store.

    Every run, not the first one. A suite writes eleven runs into one store,
    and scoring `runs[0]` alone would have reported one task's numbers as the
    suite's -- with a denominator small enough to look like a measurement.
    """
    from harness.persistence import RunStore

    out: list[tuple[str, list]] = []
    with RunStore(db_path) as store:
        for run in store.list_runs():
            agents = store.list_agents(run.id)
            if not agents:
                continue
            lead = next(
                (a for a in agents if a.parent_agent_id is None), agents[0]
            )
            out.append((lead.id, store.load_events(lead.id)))
    return out


def rebuild_transcript(db_path: str, agent_id: str) -> list[Message]:
    """The messages one agent saw, in order, from its event log."""
    from harness.persistence import RunStore

    messages: list[Message] = []
    with RunStore(db_path) as store:
        for event in store.load_events(agent_id):
            if event.kind == "message":
                messages.append(Message.model_validate(event.payload))
            elif event.kind == "tool_result":
                messages.append(
                    Message.model_validate(
                        {"role": "tool", "tool_result": event.payload}
                    )
                )
    return messages


async def compactions_at(messages: Sequence[Message], window: int) -> int:
    """How many times a recorded transcript would compact at ``window``.

    Drives a real `ContextManager` with the production system prompt and the
    production estimator, mirroring the loop's compact-to-fixpoint pass. This
    exists because the window/compaction table was asserted in a spec with no
    artifact behind it -- six empirical claims nobody could re-run.
    """
    from harness.adapters.base import ModelAdapter
    from harness.context import ContextManager

    # Deliberately not `tests.conformance.replay`, which is where this
    # borrowed its estimator and system prompt from. `pyproject.toml` packages
    # `harness*` only, so importing `tests.` made the CLI that generates the
    # spec's window table -- added precisely so six empirical claims would be
    # re-runnable -- runnable from exactly one directory.
    #
    # The system prompt is a fixed-size stand-in rather than the real assembled
    # one: at these windows its ~590 tokens shift the trigger, so the number
    # reported is a *lower bound* on how much a real run compacts. Stated
    # rather than hidden, because the table's shape is the claim, not its
    # third digit.
    async def summarize(messages: list[Message]) -> str:
        # Fixed-size, and the size matters: each summary becomes part of the
        # next trigger, so the counts below compound with it. Matched to the
        # conformance corpus's stand-in (`SUMMARY_CHARS`) so the two tables
        # are comparable. A real summarizer would make this non-deterministic.
        return "s" * SUMMARY_STANDIN_CHARS

    context = ContextManager(
        base_system_prompt="S" * SYSTEM_PROMPT_STANDIN_CHARS,
        count_tokens=lambda ms: ModelAdapter.count_tokens(None, ms),
        max_context=window,
        summarize=summarize,
    )
    fired = 0
    for message in messages:
        context.append(message)
        while True:
            before = context.effective_size
            if not await context.maybe_compact():
                break
            fired += 1
            if context.effective_size >= before:
                break
    return fired


def reach_table(
    db_path: str, agent_id: str, windows: Sequence[int]
) -> list[tuple[int, int]]:
    """``(window, compactions)`` for one recorded run. The spec's table."""
    import asyncio

    messages = rebuild_transcript(db_path, agent_id)
    return [(w, asyncio.run(compactions_at(messages, w))) for w in windows]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import glob

    parser = argparse.ArgumentParser(prog="condenser-oracle")
    parser.add_argument("paths", nargs="+", help="state.db files or globs")
    parser.add_argument("--keep", type=int, default=DEFAULT_RECENCY_KEEP)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--reach",
        action="store_true",
        help=(
            "Instead of scoring, replay the longest run at a range of windows "
            "and report where compaction starts firing."
        ),
    )
    args = parser.parse_args(argv)

    if args.reach:
        for pattern in args.paths:
            for path in sorted(glob.glob(pattern, recursive=True)) or [pattern]:
                agents = lead_agents(path)
                if not agents:
                    continue
                agent_id, events = max(agents, key=lambda a: len(a[1]))
                print(f"{path}  agent {agent_id[:8]}  ({len(events)} events)")
                for window, fired in reach_table(
                    path, agent_id,
                    [128_000, 32_000, 24_000, 16_000, 12_000, 10_000,
                     8_000, 6_000, 4_000, 2_000],
                ):
                    print(f"  window {window:7,}  ->  {fired:3} compactions")
        return 0

    reports = []
    for pattern in args.paths:
        for path in sorted(glob.glob(pattern, recursive=True)) or [pattern]:
            if not Path(path).is_file():
                # `RunStore` would create it, and the report would then blame
                # the window for the absence of data. For an eval whose whole
                # thesis is telling "no data" from "no effect", giving the
                # wrong reason for no data is the one unacceptable output.
                print(f"  no such database: {path}")
                continue
            try:
                agents = lead_agents(path)
            except Exception as exc:  # noqa: BLE001 - a bad db is skipped loudly
                print(f"  skipped {path}: {exc}")
                continue
            for agent_id, events in agents:
                reports.append(score_agent(events, agent_id, keep=args.keep))
    if args.json:
        print(
            json.dumps(
                {
                    "runs": len(reports),
                    "compactions": sum(len(r.compactions) for r in reports),
                    "evicted_turns": sum(r.evicted_turns for r in reports),
                    "lost_turns": sum(r.lost_turns for r in reports),
                },
                indent=2,
            )
        )
    else:
        print(render_report(reports))
    return 0
