"""Tests for harness.deadline (wind-down plan §2a).

Pure unit tests with injectable scripted clocks — no real sleeps.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.deadline import (
    EXEC_CAP_FLOOR_SECONDS,
    LANDING_ALLOWANCE_DEFAULT,
    LANDING_ALLOWANCE_MAX,
    LANDING_ALLOWANCE_MIN,
    MODEL_CALL_WINDOW,
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
    def test_stop_floor_and_exec_floor(self) -> None:
        assert WALL_CLOCK_STOP_FLOOR == 60.0
        assert EXEC_CAP_FLOOR_SECONDS == 30.0

    def test_landing_allowance_bounds(self) -> None:
        assert LANDING_ALLOWANCE_MIN == 15.0
        assert LANDING_ALLOWANCE_DEFAULT == 30.0
        assert LANDING_ALLOWANCE_MAX == 60.0
        assert (
            LANDING_ALLOWANCE_MIN
            <= LANDING_ALLOWANCE_DEFAULT
            <= LANDING_ALLOWANCE_MAX
        )

    def test_reserve_is_strictly_above_the_stop_floor(self) -> None:
        # The whole point of the split: an exec capped at
        # `remaining - reserve` must not land exactly on the floor, or the
        # loop hard-stops instead of giving the agent its landing turn.
        assert LANDING_ALLOWANCE_MIN > 0.0
        assert _deadline().landing_reserve() > WALL_CLOCK_STOP_FLOOR


def _deadline(
    budget: float | None = 900.0, observations: list[float] | None = None
) -> Deadline:
    """A frozen-clock deadline, optionally pre-loaded with call durations."""
    deadline = Deadline(budget, clock=lambda: 0.0)
    for seconds in observations or []:
        deadline.observe_model_call(seconds)
    return deadline


class TestLandingAllowance:
    def test_default_before_any_observation(self) -> None:
        assert _deadline().landing_allowance() == LANDING_ALLOWANCE_DEFAULT

    def test_fast_provider_is_clamped_up_to_the_minimum(self) -> None:
        # p75 of [5,5,5,5] is 5s; a 5s allowance is not a landing turn.
        assert _deadline(observations=[5.0] * 4).landing_allowance() == (
            LANDING_ALLOWANCE_MIN
        )

    def test_very_slow_provider_is_clamped_down_to_the_maximum(self) -> None:
        # A provider having a bad minute cannot reserve away the run.
        assert _deadline(observations=[200.0] * 4).landing_allowance() == (
            LANDING_ALLOWANCE_MAX
        )

    def test_observed_p75_is_used_between_the_bounds(self) -> None:
        # Nearest-rank p75 of eight calls = the 6th smallest.
        observations = [4.8, 6.0, 9.0, 9.0, 12.0, 22.0, 28.8, 46.0]
        assert _deadline(observations=observations).landing_allowance() == 22.0

    def test_real_latency_shape_lands_just_above_the_minimum(self) -> None:
        # The observed distribution (p50 9.0s, p75 15.1s, p90 28.8s): the
        # allowance tracks p75, i.e. ~15s, not the p90/p99 tail.
        observations = [
            4.8, 5.2, 6.1, 7.0, 8.0, 9.0, 9.0, 9.4,
            10.2, 11.0, 12.5, 15.1, 15.1, 28.8, 46.0, 129.0,
        ]
        assert _deadline(observations=observations).landing_allowance() == (
            pytest.approx(15.1)
        )

    def test_single_observation_is_enough(self) -> None:
        assert _deadline(observations=[40.0]).landing_allowance() == 40.0

    def test_window_is_bounded_and_forgets_the_oldest(self) -> None:
        # One ancient 300s stall, then a window's worth of fast calls: the
        # stall is evicted and stops inflating the reserve.
        deadline = _deadline(observations=[300.0] + [5.0] * MODEL_CALL_WINDOW)
        assert deadline.landing_allowance() == LANDING_ALLOWANCE_MIN
        assert deadline.recent_call_median() == 5.0

    def test_non_finite_and_negative_observations_are_ignored(self) -> None:
        deadline = _deadline(observations=[float("nan"), float("inf")])
        assert deadline.landing_allowance() == LANDING_ALLOWANCE_DEFAULT
        deadline.observe_model_call(-3.0)  # clock ran backwards
        assert deadline.recent_call_median() == 0.0

    def test_allowance_is_recorded_even_without_a_budget(self) -> None:
        # A deadline-less run still answers latency questions truthfully;
        # its consumers no-op on `remaining() is None`, not on the window.
        deadline = _deadline(budget=None, observations=[20.0, 30.0])
        assert deadline.remaining() is None
        assert deadline.recent_call_median() == 25.0
        assert deadline.landing_allowance() == 30.0


class TestLandingReserve:
    def test_reserve_is_floor_plus_allowance(self) -> None:
        assert _deadline().landing_reserve() == 90.0  # 60 + default 30
        assert _deadline(observations=[5.0] * 4).landing_reserve() == 75.0
        assert _deadline(observations=[200.0] * 4).landing_reserve() == 120.0
        assert _deadline(observations=[40.0]).landing_reserve() == 100.0

    def test_reserve_tracks_the_allowance_exactly(self) -> None:
        deadline = _deadline(observations=[22.0, 22.0])
        assert deadline.landing_reserve() == (
            WALL_CLOCK_STOP_FLOOR + deadline.landing_allowance()
        )


class TestRecentCallMedian:
    def test_none_before_any_observation(self) -> None:
        assert _deadline().recent_call_median() is None

    def test_median_of_the_window(self) -> None:
        assert _deadline(observations=[4.0, 9.0, 100.0]).recent_call_median() == (
            9.0
        )

    def test_median_of_an_even_window_interpolates(self) -> None:
        assert _deadline(observations=[8.0, 10.0]).recent_call_median() == 9.0

    def test_outlier_does_not_move_the_median(self) -> None:
        # make-doom's shape: one 225.5s truncated call among 9s calls. The
        # median is what "a typical call costs", not the outlier.
        observations = [9.0] * 8 + [225.5]
        assert _deadline(observations=observations).recent_call_median() == 9.0


def _exec_cap(deadline: Deadline, requested: float) -> float:
    """The exec cap as applied at both call sites.

    Mirrors ``harness.tools.builtin.bash_tool`` and
    ``AgentLoop._run_verification``; kept here so the landing-window
    property below is checked against the arithmetic that actually ships.
    """
    remaining = deadline.remaining()
    if remaining is None:
        return requested
    return min(
        requested,
        max(EXEC_CAP_FLOOR_SECONDS, remaining - deadline.landing_reserve()),
    )


class TestLandingWindowProperty:
    """The contract Change 0 exists to guarantee.

    After any deadline-capped exec returns, ``remaining()`` is strictly
    greater than :data:`WALL_CLOCK_STOP_FLOOR`, so the loop's hard-stop
    check cannot fire and the agent always gets one more turn to land its
    answer. The documented exception is a run already inside the stop
    floor (``remaining <= floor + EXEC_CAP_FLOOR_SECONDS``), where the exec
    floor wins and the loop is stopping regardless.
    """

    @pytest.mark.parametrize("budget", [900.0, 1200.0, 1800.0, 2400.0])
    @pytest.mark.parametrize(
        "observations",
        [
            [],  # first exec of a run: default allowance
            [5.0] * 4,  # fast provider: allowance clamped up
            [15.1] * 8,  # the observed p75
            [200.0] * 4,  # slow provider: allowance clamped down
        ],
        ids=["no-observations", "fast", "typical", "slow"],
    )
    def test_capped_exec_always_leaves_a_landing_turn(
        self, budget: float, observations: list[float]
    ) -> None:
        elapsed = 0.0

        def clock() -> float:
            return elapsed

        deadline = Deadline(budget, clock)
        for seconds in observations:
            deadline.observe_model_call(seconds)

        remaining = budget
        while remaining > 0.0:
            elapsed = budget - remaining
            for requested in (10.0, 120.0, 300.0, 600.0, 3600.0):
                cap = _exec_cap(deadline, requested)
                assert cap >= EXEC_CAP_FLOOR_SECONDS or cap == requested
                if remaining > WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS:
                    # The exec is bounded so that what is left when it
                    # returns still buys a model call.
                    assert remaining - cap > WALL_CLOCK_STOP_FLOOR
            remaining -= 10.0

    def test_the_documented_exception_is_inside_the_stop_floor(self) -> None:
        # remaining=70: `remaining - reserve` is negative, the 30s exec
        # floor wins, and the exec can return with 40s left — below the
        # stop floor. That is fine: at remaining=70 the loop is already
        # within one landing turn of stopping.
        deadline = Deadline(900.0, scripted_clock([0.0, 830.0]))
        assert deadline.remaining() == 70.0
        assert _exec_cap(deadline, 120.0) == EXEC_CAP_FLOOR_SECONDS
        assert 70.0 - EXEC_CAP_FLOOR_SECONDS < WALL_CLOCK_STOP_FLOOR

    def test_fast_provider_keeps_its_long_decisive_exec(self) -> None:
        # The regression a flat allowance would cause: a currently-solved
        # task whose decisive 137.6s exec starts at remaining=216 of 900s.
        # Its provider is fast (median call 6.9s), so the allowance clamps
        # to the 15s minimum, the reserve is 75s and the cap is 141s — the
        # exec survives. A flat allowance sized for the slow end (p95 =
        # 60s -> reserve 120s) would have cut it to 96s and lost the task.
        deadline = Deadline(900.0, scripted_clock([0.0, 684.0]))
        for seconds in (5.0, 6.0, 6.9, 6.9, 7.2, 9.0, 11.0, 14.0):
            deadline.observe_model_call(seconds)
        assert deadline.landing_allowance() == LANDING_ALLOWANCE_MIN
        assert _exec_cap(deadline, 540.0) == 141.0
        assert 141.0 > 137.6
        assert 216.0 - 141.0 > WALL_CLOCK_STOP_FLOOR

    def test_write_compressor_shape_now_leaves_a_turn(self) -> None:
        # The proved failure: a 300s exec requested at remaining=336 ran
        # 276s and returned at remaining 59.1 — exactly under the floor, so
        # the run hard-stopped without the landing turn it was told to take.
        deadline = Deadline(900.0, scripted_clock([0.0, 564.0]))  # remaining 336
        for _ in range(8):
            deadline.observe_model_call(15.0)  # write-compressor's median
        cap = _exec_cap(deadline, 300.0)
        assert cap == 261.0  # 336 - (60 + 15)
        assert 336.0 - cap == 75.0 > WALL_CLOCK_STOP_FLOOR
