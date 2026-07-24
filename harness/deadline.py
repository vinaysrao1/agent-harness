"""Wall-clock deadline seam shared across the harness (wind-down plan §2a).

A :class:`Deadline` is one process-wide answer to "how much wall-clock time
is left before an external kill?" — anchored where the external clock starts
(e.g. Harbor's ``asyncio.wait_for`` around ``agent.run()``) and threaded to
every consumer (agent loops, sandbox exec caps, verification caps) so they
all count down from the same instant. ``budget=None`` means "no deadline";
every consumer must no-op in that case, preserving today's behavior.

The exec-cap constants live here rather than in ``harness.tools.builtin`` to
avoid an import cycle (the loop and the tools both need them).
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = [
    "Deadline",
    "EXEC_RESERVE_SECONDS",
    "EXEC_CAP_FLOOR_SECONDS",
    "WALL_CLOCK_STOP_FLOOR",
]

#: Seconds of wall-clock held back from any single sandbox exec so the agent
#: always keeps enough budget after a long command to write its answer down.
EXEC_RESERVE_SECONDS = 60.0

#: The smallest exec timeout the cap will ever impose: capping below this
#: would make even trivial commands (compiler start-up, test collection) fail
#: spuriously, which is worse than letting the command eat into the reserve.
EXEC_CAP_FLOOR_SECONDS = 30.0

#: Remaining wall-clock below which the agent loop refuses to start another
#: model call and pauses instead. Tied to :data:`EXEC_RESERVE_SECONDS` — one
#: coherent story: below the reserve, nothing new starts.
WALL_CLOCK_STOP_FLOOR = EXEC_RESERVE_SECONDS


class Deadline:
    """A wall-clock budget anchored at construction time.

    ``budget_seconds=None`` disables the deadline: :attr:`budget` is ``None``
    and :meth:`remaining` returns ``None``, so consumers treat it exactly
    like an absent deadline. ``clock`` is injectable for deterministic tests
    (same convention as ``AgentLoop``); it defaults to :func:`time.monotonic`.
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

    def remaining(self) -> float | None:
        """Seconds left before the deadline, clamped at zero.

        Returns ``None`` when no budget was set (no deadline; consumers
        no-op).
        """
        if self.budget is None:
            return None
        return max(0.0, self.budget - (self._clock() - self._start))
