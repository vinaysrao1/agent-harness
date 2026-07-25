"""Wall-clock deadline seam shared across the harness (wind-down plan §2a).

A :class:`Deadline` is one process-wide answer to "how much wall-clock time
is left before an external kill?" — anchored where the external clock starts
(e.g. Harbor's ``asyncio.wait_for`` around ``agent.run()``) and threaded to
every consumer (agent loops, sandbox exec caps, verification caps) so they
all count down from the same instant. ``budget=None`` means "no deadline";
every consumer must no-op in that case, preserving today's behavior.

The exec-cap constants live here rather than in ``harness.tools.builtin`` to
avoid an import cycle (the loop and the tools both need them).

A deadline also owns the *landing reserve*: the wall-clock held back from
any single sandbox exec so that when a long command returns there is still
time for one more model call — the turn in which the agent writes its
answer down. The reserve is the hard-stop floor plus an adaptive allowance
derived from this run's own observed model-call durations, so it tracks the
provider instead of guessing (see :meth:`Deadline.landing_reserve`).

On top of that reserve, :meth:`Deadline.exec_cap` adds the two bounds that
keep a single command from owning the run: a **share cap** (no exec larger
than :data:`EXEC_MAX_BUDGET_FRACTION` of the whole budget) and a **band
guarantee** (an exec started above the wind-down threshold hands control
back with enough time to actually wind down). The wind-down threshold lives
here, next to them, rather than in ``harness.loop`` — the loop re-exports it
— so the tools layer can reach it without importing the loop.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

__all__ = [
    "Deadline",
    "ExecCapDecision",
    "ExecPurpose",
    "EXEC_CAP_FLOOR_SECONDS",
    "EXEC_MAX_BUDGET_FRACTION",
    "LANDING_ALLOWANCE_DEFAULT",
    "LANDING_ALLOWANCE_MAX",
    "LANDING_ALLOWANCE_MIN",
    "LANDING_RESERVE_FRACTION",
    "MODEL_CALL_WINDOW",
    "WALL_CLOCK_STOP_FLOOR",
    "WIND_DOWN_FRACTION",
    "WIND_DOWN_MAX_REMAINING",
    "WIND_DOWN_MIN_REMAINING",
    "wind_down_threshold",
]

#: What an exec cap is *for*: an ``"exploratory"`` command is one the agent
#: issued while still working, so it is subject to both the share cap and the
#: band guarantee; a ``"verification"`` command is the completion gate the
#: loop re-runs, which runs at completion by definition and is exempt from
#: both (see :meth:`Deadline.exec_cap`).
ExecPurpose = Literal["exploratory", "verification"]

#: The smallest exec timeout the cap will ever impose: capping below this
#: would make even trivial commands (compiler start-up, test collection) fail
#: spuriously, which is worse than letting the command eat into the reserve.
EXEC_CAP_FLOOR_SECONDS = 30.0

#: Remaining wall-clock below which the agent loop refuses to start another
#: model call and pauses instead: nothing new starts inside the floor.
#:
#: This is deliberately *not* the exec reserve. When the two were the same
#: number, an exec capped at ``remaining - reserve`` returned with exactly
#: the floor left, the loop's hard-stop check fired, and the agent never got
#: the landing turn the cap was reserving time for. The reserve is now
#: ``floor + landing_allowance()`` (see :meth:`Deadline.landing_reserve`).
WALL_CLOCK_STOP_FLOOR = 60.0

#: Lower bound on the landing allowance — the time held back *on top of*
#: :data:`WALL_CLOCK_STOP_FLOOR` to fund one more model call. Calibrated to
#: the p75 observed model-call duration (15.1s). Raising it shortens every
#: capped exec, so it is the constant to touch only with exec-cap telemetry
#: in hand.
LANDING_ALLOWANCE_MIN = 15.0

#: Upper bound on the landing allowance: a provider having a very bad minute
#: must not be able to reserve away the whole budget.
LANDING_ALLOWANCE_MAX = 60.0

#: Landing allowance assumed before any model call has been observed (~p90
#: of observed call durations, 28.8s) — the first exec of a run has no
#: latency history to work from.
LANDING_ALLOWANCE_DEFAULT = 30.0

#: How many recent model-call durations feed the allowance. Small enough to
#: track a provider slowing down mid-run, large enough that one outlier call
#: cannot dominate the p75.
MODEL_CALL_WINDOW = 16

#: No single sandbox exec may own more than this share of the *whole* run.
#:
#: This is the primary bound, and it is the only one that catches a genuine
#: runaway: a command issued at the very start of a run has almost the whole
#: budget "remaining", so every reserve-based bound still permits it to eat
#: nearly all of it (one observed trial spent 1647s of its 1800s budget
#: inside a single command; the band guarantee alone would still have
#: allowed 1347s). Expressed against :attr:`Deadline.budget`, not against
#: what remains, deliberately: a cumulative counter would penalize a run for
#: one legitimate early build.
EXEC_MAX_BUDGET_FRACTION = 0.5

#: Softener on the band guarantee: an exploratory exec started above the
#: wind-down threshold always keeps at least this share of what remains.
#:
#: Without it, an exec starting just above the threshold would be chopped
#: back to the threshold itself. 0.25 rather than 0.5 because at 0.5 the
#: decisive build of a currently-passing task retains only 1.4x headroom
#: over its observed runtime, versus 2.1x at 0.25. Named so round 3 can
#: retune it from ``exec_capped`` telemetry without touching the formula.
LANDING_RESERVE_FRACTION = 0.25

#: Fraction of the wall-clock budget remaining at which the loop injects the
#: one-time wind-down reminder, clamped by the two bounds that follow — see
#: :func:`wind_down_threshold`. Chosen to leave the agent one or two turns
#: to land a best-effort answer on disk before an external deadline (e.g. a
#: benchmark harness's per-agent timeout) kills the trial mid-turn.
WIND_DOWN_FRACTION = 0.2

#: Floor on the wind-down threshold: single (slow-provider) model calls have
#: been observed to run up to ~271s, so a raw 0.2 fraction of a 900s budget
#: (180s) can vanish inside ONE call — the reminder would land with nothing
#: left to act on.
WIND_DOWN_MIN_REMAINING = 300.0

#: Ceiling on the wind-down threshold: 0.2 of a 12000s budget would wind the
#: run down with 2400s still left — disabling diligence nudges for 40
#: minutes of perfectly usable time.
WIND_DOWN_MAX_REMAINING = 600.0


def wind_down_threshold(budget: float) -> float:
    """The remaining-seconds threshold at which wind-down fires for ``budget``.

    :data:`WIND_DOWN_FRACTION` of the budget, clamped to the
    [:data:`WIND_DOWN_MIN_REMAINING`, :data:`WIND_DOWN_MAX_REMAINING`] band —
    and never more than half the budget, so degenerate tiny budgets still get
    at least half the run before the reminder lands.
    """
    return min(
        max(WIND_DOWN_FRACTION * budget, WIND_DOWN_MIN_REMAINING),
        WIND_DOWN_MAX_REMAINING,
        0.5 * budget,
    )


class ExecCapDecision(NamedTuple):
    """One exec-cap decision, including the reserve it actually applied.

    :meth:`Deadline.exec_cap` returns only ``(effective, capped, reason)``;
    this is the same decision with ``reserve`` — the number the arithmetic
    *actually* held back, i.e. :meth:`Deadline.landing_reserve` normally and
    the softened band reserve (:data:`LANDING_RESERVE_FRACTION` of what
    remains) whenever the band guarantee raised it.

    It is reported for every decision, including the ones where some other
    bound ended up binding, because that is the number round 3 retunes
    :data:`LANDING_RESERVE_FRACTION` from. A consumer cannot reconstruct it:
    ``remaining - effective`` recovers it only for ``reason`` ``"band"`` and
    ``"reserve"``, and not at all for ``"share"`` — which is precisely the
    case where "the softener bit hard" and "only the base reserve bit" would
    otherwise be indistinguishable. ``0.0`` on the no-deadline passthrough,
    where no reserve is held back at all.
    """

    effective: float
    capped: bool
    reason: str | None
    reserve: float


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of ``values`` (non-empty, unsorted ok).

    Nearest-rank rather than an interpolating definition so the result is
    always an actually-observed duration, and so it is defined for a window
    of one — the common case on the first turns of a run.
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class Deadline:
    """A wall-clock budget anchored at construction time.

    ``budget_seconds=None`` disables the deadline: :attr:`budget` is ``None``
    and :meth:`remaining` returns ``None``, so consumers treat it exactly
    like an absent deadline. ``clock`` is injectable for deterministic tests
    (same convention as ``AgentLoop``); it defaults to :func:`time.monotonic`.

    A deadline also carries a small rolling window of observed model-call
    durations (:meth:`observe_model_call`), which is what makes the landing
    reserve adaptive rather than a flat guess. The window is mutable state
    on a single-run-scoped object; it is shared by the lead agent and its
    subagents by design (they share one deadline, and a subagent's calls go
    through the same provider), but it is not thread-safe.
    """

    def __init__(
        self,
        budget_seconds: float | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        #: Total wall-clock budget in seconds; ``None`` = no deadline.
        self.budget: float | None = (
            float(budget_seconds) if budget_seconds is not None else None
        )
        self._clock = clock
        self._start = clock()
        self._call_seconds: deque[float] = deque(maxlen=MODEL_CALL_WINDOW)

    def remaining(self) -> float | None:
        """Seconds left before the deadline, clamped at zero.

        Returns ``None`` when no budget was set (no deadline; consumers
        no-op).
        """
        if self.budget is None:
            return None
        return max(0.0, self.budget - (self._clock() - self._start))

    def observe_model_call(self, seconds: float) -> None:
        """Record one completed model call's wall-clock duration.

        Called by the agent loop right after every model call returns. Only
        finite, non-negative durations are kept (a clock that jumps
        backwards must not poison the window); the window is bounded at
        :data:`MODEL_CALL_WINDOW`, oldest evicted first. Recording is done
        even when :attr:`budget` is ``None`` so a deadline-less run still
        answers :meth:`recent_call_median` truthfully.
        """
        value = float(seconds)
        if not math.isfinite(value):
            return
        self._call_seconds.append(max(0.0, value))

    def landing_allowance(self) -> float:
        """Seconds to hold back, above the stop floor, for one landing turn.

        The clamped p75 of the recent model-call window — long enough that
        three calls in four would fit, bounded by
        :data:`LANDING_ALLOWANCE_MIN`/:data:`LANDING_ALLOWANCE_MAX` so a fast
        provider still keeps a usable margin and a slow one cannot reserve
        away the run. With no observations yet, this is
        :data:`LANDING_ALLOWANCE_DEFAULT`.

        Adaptive rather than flat on purpose: observed per-call latency
        varies ~3x across providers and tasks, and a flat allowance sized
        for the slow end shortens the decisive exec of runs on the fast end.
        """
        if not self._call_seconds:
            return LANDING_ALLOWANCE_DEFAULT
        p75 = _nearest_rank(self._call_seconds, 0.75)
        return min(LANDING_ALLOWANCE_MAX, max(LANDING_ALLOWANCE_MIN, p75))

    def landing_reserve(self) -> float:
        """Seconds of wall-clock held back from any single sandbox exec.

        :data:`WALL_CLOCK_STOP_FLOOR` (below which the loop starts nothing
        new) plus :meth:`landing_allowance` (one more model call). An exec
        capped at ``remaining - landing_reserve()`` therefore returns with
        strictly more than the stop floor left, so the agent always gets a
        turn to write its answer down after a long command.

        The one documented exception: when ``remaining`` is already so small
        that ``remaining - landing_reserve()`` falls below
        :data:`EXEC_CAP_FLOOR_SECONDS`, the floor wins (a 0s timeout would
        fail even ``echo``) and the guarantee can be violated — but only
        inside the stop floor, where the loop pauses anyway.
        """
        return WALL_CLOCK_STOP_FLOOR + self.landing_allowance()

    def exec_cap(
        self,
        requested: float,
        *,
        purpose: ExecPurpose = "exploratory",
    ) -> tuple[float, bool, str | None]:
        """Bound one sandbox exec's timeout by what the run can afford.

        The ``(effective, capped, reason)`` shape every caller that only
        needs to *apply* the cap wants. :meth:`exec_decision` is the same
        decision plus the reserve it applied, for callers recording
        telemetry — see :class:`ExecCapDecision`.
        """
        decision = self.exec_decision(requested, purpose=purpose)
        return decision.effective, decision.capped, decision.reason

    def exec_decision(
        self,
        requested: float,
        *,
        purpose: ExecPurpose = "exploratory",
    ) -> ExecCapDecision:
        """Bound one sandbox exec's timeout by what the run can afford.

        Returns an :class:`ExecCapDecision`: the timeout to actually apply,
        whether it is shorter than ``requested``, which bound bit —
        ``"share"``, ``"band"`` or ``"reserve"`` (``None`` when nothing
        bit) — and the reserve the arithmetic held back, reported whichever
        bound won so telemetry never has to reconstruct it.
        ``budget is None`` is a pure passthrough:
        ``(requested, False, None, 0.0)``.

        Three bounds, in the order they matter:

        **Share cap** (``purpose="exploratory"`` only). No single exec may
        exceed :data:`EXEC_MAX_BUDGET_FRACTION` of :attr:`budget`. This is
        the primary bound — see the constant — and the only one that
        contains a long command issued while the budget is still nearly
        whole.

        **Band guarantee** (``purpose="exploratory"`` only). An exec started
        *above* :func:`wind_down_threshold` must not carry the run past that
        threshold without the agent getting a turn, so the reserve is raised
        to the threshold — softened to at least
        :data:`LANDING_RESERVE_FRACTION` of what remains, so a command
        starting just above the threshold is not chopped back to it.

        **Landing reserve.** Always applied: :meth:`landing_reserve`, so the
        agent gets one more model call after the command returns. Never
        below :data:`EXEC_CAP_FLOOR_SECONDS`, so a nearly-spent budget still
        yields a usable exec window rather than a spuriously-failing one.
        (The floor bounds only this term; the share cap still wins over it,
        which is what keeps ``effective <= 0.5 * budget`` unconditional for
        exploratory execs.)

        ``purpose="verification"`` is exempt from the share cap *and* the
        band softener, and gets the plain ``remaining - landing_reserve()``
        shape. Verification runs at completion by definition: shortening it
        would turn a legitimate check into a capped timeout, which the loop
        reads as *inconclusive* and accepts — silently converting a passing
        gate into a non-gate.
        """
        remaining = self.remaining()
        if remaining is None or self.budget is None:
            return ExecCapDecision(requested, False, None, 0.0)

        reserve = self.landing_reserve()
        softened = False
        share = float("inf")
        if purpose == "exploratory":
            threshold = wind_down_threshold(self.budget)
            if remaining > threshold:
                band = min(threshold, LANDING_RESERVE_FRACTION * remaining)
                if band > reserve:
                    reserve = band
                    softened = True
            share = EXEC_MAX_BUDGET_FRACTION * self.budget

        allowed = max(EXEC_CAP_FLOOR_SECONDS, remaining - reserve)
        effective = min(requested, share, allowed)
        if effective >= requested:
            return ExecCapDecision(requested, False, None, reserve)
        if share <= allowed:
            return ExecCapDecision(effective, True, "share", reserve)
        return ExecCapDecision(
            effective, True, "band" if softened else "reserve", reserve
        )

    def recent_call_median(self) -> float | None:
        """Median of the recent model-call window; ``None`` when empty.

        The "typical call" figure consumers use to decide whether another
        call can plausibly land, kept here so the landing reserve and every
        other latency judgement read one window.
        """
        if not self._call_seconds:
            return None
        return float(statistics.median(self._call_seconds))
