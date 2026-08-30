"""Compaction strategies (S-105).

Compaction was the only subsystem in the harness with no seam and no
measurement: one `await self._summarize(evicted)` call inside
`ContextManager.compact`, whose output nothing inspected, compared or scored.
A 100+ step run compacts five to ten times, and each compaction summarizes the
output of the last one, so quality compounds multiplicatively — it is the
worst possible place for an unmeasured heuristic.

This module holds *what to keep and how to say it*. `ContextManager` keeps
*when to do it* — the threshold, the eviction boundary, and the bookkeeping
that makes a condensation applyable at assembly time. The split is drawn
there because the boundary calculation is what the provider APIs constrain (a
kept transcript may not start with a tool result), and that is a property of
the assembly, not of the strategy.

`DefaultCondenser` is today's behaviour, extracted without changing a byte —
the replay corpus asserts zero drift, not "±1". Everything else is a
capability the benchmark profile does not declare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from harness.types import Message, Role

__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "CondenseContext",
    "Condensation",
    "Condenser",
    "DefaultCondenser",
    "PivotalCondenser",
    "MAX_PIVOTAL_KEPT",
    "KNOWN_CONDENSERS",
    "condenser_for",
]

#: Marks the synthetic message that replaces an evicted span. Lives here
#: rather than in `context` because it is part of what a strategy emits;
#: `context` re-exports it, so the existing import site is unchanged.
COMPACTION_SUMMARY_PREFIX = "[COMPACTION SUMMARY]"

#: Ceiling on how many messages one condensation may carry forward. Pivotal
#: marking over-marks by construction -- every failing tool result is marked,
#: and a run that fails often would otherwise retain most of its own span and
#: shrink the transcript by almost nothing. The loop's fixpoint pass would
#: then spin: compaction returns a span, the assembly barely moves, and it
#: compacts again. Small enough that the shrink guard always has room.
MAX_PIVOTAL_KEPT = 4

#: Every strategy name `condenser_for` accepts. Named separately so callers
#: can reject an unknown one *before* deciding whether they would have used
#: it -- the orchestrator's profile gate ignores the config on the benchmark
#: path, which silently swallowed typos there.
KNOWN_CONDENSERS: frozenset[str] = frozenset(
    {"summarize-halve", "summarize-halve+pivotal"}
)


@dataclass(frozen=True)
class CondenseContext:
    """Everything a strategy may read about the run it is condensing.

    Deliberately not the `ContextManager` itself. A strategy that could reach
    into the manager could mutate the transcript it is being asked to
    summarize, and the whole point of the seam is that swapping a strategy
    cannot change anything but the summary and what is retained.
    """

    #: The run's goal, verbatim. Folded into the summary header by every
    #: strategy: the goal must never depend on summarizer quality.
    goal: str
    #: Event refs of the span being condensed, aligned with `span`.
    refs: tuple[int, ...]
    #: Refs the run has marked pivotal so far, with the reason each was
    #: marked. A strategy may ignore this entirely; the default does.
    pivotal: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class Condensation:
    """One strategy's answer: what the span becomes.

    `kept_refs` and `dropped_refs` partition `ctx.refs`, so the event log can
    say what survived without re-deriving it from the transcript.
    """

    summary: str
    strategy_id: str
    kept_refs: tuple[int, ...] = ()
    dropped_refs: tuple[int, ...] = ()
    #: Why each kept ref was kept, for telemetry. A retention that never
    #: retains, or always retains, is then visible in the event log rather
    #: than inferred from behaviour.
    reasons: tuple[str, ...] = ()


class Condenser(Protocol):
    """Turns an evicted span into a summary plus a retention decision."""

    strategy_id: str

    async def condense(
        self, span: list[Message], ctx: CondenseContext
    ) -> Condensation: ...


def _header(goal: str, summary: str) -> str:
    """The summary message body. Byte-identical to the pre-S-105 text."""
    return (
        f"{COMPACTION_SUMMARY_PREFIX}\n"
        f"Original goal (verbatim, never summarized):\n"
        f"{goal}\n"
        f"---\n"
        f"{summary}"
    )


@dataclass
class DefaultCondenser:
    """Today's behaviour, extracted: summarize the span, keep nothing.

    The summarizer callable is the same one `ContextManager` used to hold --
    the agent loop wires a model call, tests inject a stub. Nothing about the
    text it produces changed, which is what lets N7 assert *zero* drift rather
    than a tolerance.
    """

    summarize: Callable[[list[Message]], Awaitable[str]]
    strategy_id: str = "summarize-halve"

    async def condense(
        self, span: list[Message], ctx: CondenseContext
    ) -> Condensation:
        summary = await self.summarize(list(span))
        return Condensation(
            summary=_header(ctx.goal, summary),
            strategy_id=self.strategy_id,
            dropped_refs=ctx.refs,
        )


def _turn_bounds(span: list[Message], index: int) -> tuple[int, int]:
    """The half-open range of ``span`` that holds ``index``'s whole turn.

    A turn is one assistant message plus every tool result answering it.
    Neither half is valid alone: a `tool_result` with no preceding `tool_use`
    is rejected by the provider, and so is a `tool_use` with no result. So
    retention operates on turns, never on messages -- retaining "the failing
    tool result" would have produced a 400 on the next model call, in the
    middle of a long run, at the moment context is tightest.
    """
    start = index
    while start > 0 and span[start].role is Role.TOOL:
        start -= 1
    if span[start].role is Role.TOOL:
        # A span whose first message is a tool result -- its owning assistant
        # message was evicted by an earlier condensation. There is no whole
        # turn to keep, so keep nothing rather than an orphan the provider
        # would reject. Unreachable today (the view always starts with the
        # goal or a summary) and cheap to guarantee.
        return index, index
    end = start + 1
    while end < len(span) and span[end].role is Role.TOOL:
        end += 1
    return start, end


@dataclass
class PivotalCondenser:
    """Summarize the span, but carry its pivotal turns forward intact.

    The failure this exists to prevent: a run discovers at turn 12 that the
    test suite fails for a reason it had to dig for, compacts at turn 30, and
    the summarizer -- which is a cheap model with no idea which line mattered
    -- renders that as "ran the tests". The run then re-derives it, or does
    not.

    Retention is capped at `MAX_PIVOTAL_KEPT` *turns* and takes the most
    recent ones. Oldest-first was wrong: an early failure that has since been
    fixed is exactly the one worth summarizing away, and keeping it while
    dropping the failure the run is currently working on inverts the point.
    Uncapped was worse -- see `MAX_PIVOTAL_KEPT`.

    A retained turn is carried verbatim rather than re-summarized, because the
    reason it is retained is that the summarizer is the thing not trusted
    with it.
    """

    summarize: Callable[[list[Message]], Awaitable[str]]
    strategy_id: str = "summarize-halve+pivotal"
    max_kept: int = MAX_PIVOTAL_KEPT

    async def condense(
        self, span: list[Message], ctx: CondenseContext
    ) -> Condensation:
        summary = await self.summarize(list(span))
        by_ref = {ref: i for i, ref in enumerate(ctx.refs)}

        turns: list[tuple[int, int, str]] = []
        seen: set[int] = set()
        for ref, reason in ctx.pivotal:
            index = by_ref.get(ref)
            if index is None:
                continue  # marked outside this span; a later one may hold it
            start, end = _turn_bounds(span, index)
            # By set, not by "same as the previous one": marks arrive in ref
            # order today, but a signal that marks retroactively (the plan's
            # "a `decision` later cited") would not, and keeping one turn
            # twice puts a duplicate `tool_use` id and a duplicate
            # `tool_result` in the request, which the provider rejects.
            if start in seen:
                continue
            seen.add(start)
            turns.append((start, end, reason))
        turns.sort()

        # The condensation replaces `len(span)` messages with one summary
        # plus every message in the kept turns, so it only shrinks the view
        # while `1 + kept_messages < len(span)`. The budget is therefore in
        # *messages*, and the cap is in *turns*: a turn is an assistant
        # message plus its tool results, and the `tool_error` signal only
        # ever marks tool results, so real turns are never one message long.
        # Spending `len(span) - 2` as if it were a turn count let a
        # two-turn six-message span keep all six -- a condensation that
        # *grew* the view from 6 to 7 -- and, through the loop, wedged the
        # transcript at a floor it could never compact below while burning
        # one real summarizer call per turn for the rest of the run.
        budget = len(span) - 2
        selected: list[tuple[int, int, str]] = []
        used = 0
        for turn in reversed(turns):
            size = turn[1] - turn[0]
            if len(selected) >= self.max_kept or used + size > budget:
                break
            selected.append(turn)
            used += size
        selected.reverse()
        kept_refs: list[int] = []
        reasons: list[str] = []
        for start, end, reason in selected:
            kept_refs.extend(ctx.refs[start:end])
            reasons.append(f"{reason} (refs {ctx.refs[start]}-{ctx.refs[end - 1]})")
        kept = set(kept_refs)
        return Condensation(
            summary=_header(ctx.goal, summary),
            strategy_id=self.strategy_id,
            kept_refs=tuple(kept_refs),
            dropped_refs=tuple(r for r in ctx.refs if r not in kept),
            reasons=tuple(reasons),
        )


def condenser_for(
    strategy_id: str,
    summarize: Callable[[list[Message]], Awaitable[str]],
) -> Condenser:
    """Build the named strategy. Unknown ids raise rather than defaulting.

    Silently falling back to the default would make a typo in config look
    exactly like a working configuration -- the report would name the
    strategy the operator asked for while the run used another one.
    """
    builders: dict[str, Callable[[], Condenser]] = {
        DefaultCondenser.strategy_id: lambda: DefaultCondenser(summarize),
        PivotalCondenser.strategy_id: lambda: PivotalCondenser(summarize),
    }
    assert set(builders) == KNOWN_CONDENSERS
    if strategy_id not in KNOWN_CONDENSERS:
        known = ", ".join(sorted(KNOWN_CONDENSERS))
        raise ValueError(f"unknown condenser {strategy_id!r}; known: {known}")
    return builders[strategy_id]()
