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
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Callable, Sequence

__all__ = [
    "Deadline",
    "EXEC_CAP_FLOOR_SECONDS",
    "LANDING_ALLOWANCE_DEFAULT",
    "LANDING_ALLOWANCE_MAX",
    "LANDING_ALLOWANCE_MIN",
    "MODEL_CALL_WINDOW",
    "WALL_CLOCK_STOP_FLOOR",
]

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

    def recent_call_median(self) -> float | None:
        """Median of the recent model-call window; ``None`` when empty.

        The "typical call" figure consumers use to decide whether another
        call can plausibly land, kept here so the landing reserve and every
        other latency judgement read one window.
        """
        if not self._call_seconds:
            return None
        return float(statistics.median(self._call_seconds))
