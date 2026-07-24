"""Tests for harness.deadline (wind-down plan §2a).

Pure unit tests with injectable scripted clocks — no real sleeps.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.deadline import (
    EXEC_CAP_FLOOR_SECONDS,
    EXEC_RESERVE_SECONDS,
    WALL_CLOCK_STOP_FLOOR,
    Deadline,
)


def scripted_clock(values: list[float]) -> Callable[[], float]:
    """A clock returning ``values`` in order, then repeating the last."""
    it = iter(values)
    last = values[-1]

    def clock() -> float:
        nonlocal last
        last = next(it, last)
        return last

    return clock


class TestDeadline:
    def test_none_budget_means_no_deadline(self) -> None:
        deadline = Deadline(None, scripted_clock([0.0, 1e9]))
        assert deadline.budget is None
        assert deadline.remaining() is None
        # Stays None however far the clock advances.
        assert deadline.remaining() is None

    def test_anchored_at_construction_and_counts_down(self) -> None:
        # Constructed at t=100; reads at t=130 and t=175.
        deadline = Deadline(90.0, scripted_clock([100.0, 130.0, 175.0]))
        assert deadline.budget == 90.0
        assert deadline.remaining() == 60.0
        assert deadline.remaining() == 15.0

    def test_remaining_clamps_at_zero(self) -> None:
        deadline = Deadline(10.0, scripted_clock([0.0, 25.0, 9999.0]))
        assert deadline.remaining() == 0.0
        assert deadline.remaining() == 0.0

    def test_int_budget_coerced_to_float(self) -> None:
        deadline = Deadline(900, scripted_clock([0.0, 0.0]))
        assert deadline.budget == 900.0
        assert isinstance(deadline.budget, float)
        assert deadline.remaining() == 900.0

    def test_default_clock_is_monotonic(self) -> None:
        # No injected clock: remaining is sane (<= budget, > 0 immediately).
        deadline = Deadline(3600.0)
        remaining = deadline.remaining()
        assert remaining is not None
        assert 0.0 < remaining <= 3600.0


class TestConstants:
    def test_exec_reserve_and_floor(self) -> None:
        assert EXEC_RESERVE_SECONDS == 60.0
        assert EXEC_CAP_FLOOR_SECONDS == 30.0

    def test_stop_floor_shares_the_reserve(self) -> None:
        # One coherent story: below the exec reserve, nothing new starts.
        assert WALL_CLOCK_STOP_FLOOR == EXEC_RESERVE_SECONDS
