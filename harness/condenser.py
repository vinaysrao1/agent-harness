"""Compaction strategies (S-105).

Compaction was the only subsystem in the harness with no seam and no
measurement: one `await self._summarize(evicted)` call inside
`ContextManager.compact`, whose output nothing inspected, compared or scored.

The m10 plan justified this by "a 100+ step run compacts 5-10 times". S-404
checked and that is false on this workload: compaction had fired **zero**
times across 738 recorded agents, and the eviction ladder makes it
unreachable above a ~10-12K window, because pruning is a cheaper rung that
sheds the 77% of context that is tool output. What S-404 did establish, by
forcing the window down, is that when compaction *does* fire it destroys
things runs come back for -- a 25% loss rate -- so the seam is worth having
for whatever addresses that.

This module holds *what to keep and how to say it*. `ContextManager` keeps
*when to do it* — the threshold, the eviction boundary, and the bookkeeping
that makes a condensation applyable at assembly time. The split is drawn
there because the boundary calculation is what the provider APIs constrain (a
kept transcript may not start with a tool result), and that is a property of
the assembly, not of the strategy.

`DefaultCondenser` is today's behaviour, extracted without changing a byte —
the replay corpus asserts zero drift, not "±1". It is the only strategy that
ships: `PivotalCondenser` was the second, and S-404 measured it at 0% recall
against that 25% loss rate and deleted it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from harness.types import Message

__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "CondenseContext",
    "Condensation",
    "Condenser",
    "DefaultCondenser",
    "KNOWN_CONDENSERS",
    "condenser_for",
]

#: Marks the synthetic message that replaces an evicted span. Lives here
#: rather than in `context` because it is part of what a strategy emits;
#: `context` re-exports it, so the existing import site is unchanged.
COMPACTION_SUMMARY_PREFIX = "[COMPACTION SUMMARY]"

#: Every strategy name `condenser_for` accepts. One member today. Kept as a
#: named set rather than inlined because it is what a caller checks against
#: before deciding whether it would even have used the value -- the profile
#: gate that needed that is gone with the second strategy, and returns with
#: the next one.
KNOWN_CONDENSERS: frozenset[str] = frozenset(
    {"summarize-halve"}
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
    #: Event refs of the span being condensed, aligned with `span`. A
    #: strategy cannot act on them today -- nothing applies a retention
    #: decision -- but they are what a future one names its kept turns by,
    #: and they cost nothing to pass.
    refs: tuple[int, ...]


@dataclass(frozen=True)
class Condensation:
    """One strategy's answer: what the span becomes.

    The m10 plan specified `kept_refs`/`dropped_refs` here, for a strategy
    that carries turns forward past an eviction. `PivotalCondenser` was that
    strategy; S-404 measured it at 0% recall and deleted it, and mutation
    testing then showed every line of the retention plumbing could be removed
    with the full suite still green -- it was exercised by one assertion on an
    event payload and by nothing else.

    So the fields are gone too. Keeping a documented field that no strategy
    honours and no code applies would be a lie in the type, and the honest
    version of "the seam supports retention" is that it does not, yet. The
    next retention strategy re-adds ~50 lines along with the evidence that
    earns them; S-404's recency numbers say where it should start.
    """

    summary: str
    strategy_id: str


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
            summary=_header(ctx.goal, summary), strategy_id=self.strategy_id
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
    }
    assert set(builders) == KNOWN_CONDENSERS
    if strategy_id not in KNOWN_CONDENSERS:
        known = ", ".join(sorted(KNOWN_CONDENSERS))
        raise ValueError(f"unknown condenser {strategy_id!r}; known: {known}")
    return builders[strategy_id]()
