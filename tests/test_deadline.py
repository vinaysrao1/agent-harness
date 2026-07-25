"""Tests for harness.deadline (wind-down plan §2a).

Pure unit tests with injectable scripted clocks — no real sleeps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from harness.deadline import (
    EXEC_CAP_FLOOR_SECONDS,
    EXEC_MAX_BUDGET_FRACTION,
    LANDING_ALLOWANCE_DEFAULT,
    LANDING_ALLOWANCE_MAX,
    LANDING_ALLOWANCE_MIN,
    LANDING_RESERVE_FRACTION,
    MODEL_CALL_WINDOW,
    WALL_CLOCK_STOP_FLOOR,
    WIND_DOWN_FRACTION,
    WIND_DOWN_MAX_REMAINING,
    WIND_DOWN_MIN_REMAINING,
    Deadline,
    wind_down_threshold,
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


def _exec_cap(
    deadline: Deadline, requested: float, purpose: str = "exploratory"
) -> float:
    """The effective timeout :meth:`Deadline.exec_cap` would apply.

    A thin projection of the shipping method onto its first element, so the
    landing-window property below is checked against the arithmetic that
    actually runs at both call sites (``harness.tools.builtin.bash_tool``
    and ``AgentLoop._run_verification``) rather than a restatement of it.
    """
    return deadline.exec_cap(requested, purpose=purpose)[0]  # type: ignore[arg-type]


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
        # The reserve alone (verification's shape) already clears the floor:
        # 336 - (60 + 15) = 261.
        assert _exec_cap(deadline, 300.0, purpose="verification") == 261.0
        # An exploratory exec starts above the 300s wind-down threshold, so
        # the band softener holds back a further 0.25 x 336 = 84s — strictly
        # more headroom than the reserve, never less.
        cap = _exec_cap(deadline, 300.0)
        assert cap == 252.0
        assert 336.0 - cap == 84.0 > WALL_CLOCK_STOP_FLOOR


def _at(
    budget: float,
    remaining: float,
    observations: tuple[float, ...] = (5.0, 5.0, 5.0, 5.0),
) -> Deadline:
    """A deadline of ``budget`` frozen at ``remaining`` seconds left.

    ``observations`` defaults to a fast provider, whose p75 clamps up to
    :data:`LANDING_ALLOWANCE_MIN` — so the landing reserve is pinned at 75s,
    the value the simulation below was computed against.
    """
    deadline = Deadline(budget, scripted_clock([0.0, budget - remaining]))
    for seconds in observations:
        deadline.observe_model_call(seconds)
    return deadline


class _Row(NamedTuple):
    """One simulated exec: the inputs, and the cap the plan expects."""

    id: str
    budget: float
    remaining: float
    requested: float
    purpose: str
    effective: float
    capped: bool
    reason: str | None


#: The plan's simulation of :meth:`Deadline.exec_cap` against every bash
#: exec of the 18-trial benchmark run that the cap would change or that a
#: currently-solved task depends on, plus the verification-purpose rows
#: that pin the exemption. Every ``effective`` is the *exact* value the
#: constants were chosen to produce, so drift in either constant fails here
#: loudly instead of silently re-cutting a solved task's decisive command.
_SIMULATION: tuple[_Row, ...] = (
    # --- exploratory: the share cap, the primary bound --------------------
    # 1647s of an 1800s budget inside one command. The band guarantee alone
    # would still have permitted 1347s; only the share cap contains it.
    _Row("mcmc-sampling-stan", 1800, 1706.9, 1800, "exploratory", 900.0, True, "share"),
    _Row("compcert-2400", 2400, 1723.8, 2400, "exploratory", 1200.0, True, "share"),
    # --- exploratory: the band guarantee ----------------------------------
    _Row("caffe-cifar-10", 1200, 619.7, 600, "exploratory", 464.775, True, "band"),
    _Row("compcert-3600", 2400, 1241.9, 3600, "exploratory", 931.425, True, "band"),
    # The decisive `make -j 6` of a SOLVED task: it ran 208.9s and keeps
    # 435.975s (2.1x). LANDING_RESERVE_FRACTION = 0.5 would leave 1.4x.
    _Row("compcert-make-j6", 2400, 581.3, 3600, "exploratory", 435.975, True, "band"),
    _Row("compcert-1200", 2400, 713.6, 1200, "exploratory", 535.2, True, "band"),
    _Row("write-compressor", 900, 335.5, 300, "exploratory", 251.625, True, "band"),
    # --- exploratory: in-band, the landing reserve alone ------------------
    _Row("extract-moves-late", 1800, 322.7, 600, "exploratory", 247.7, True, "reserve"),
    _Row("qemu-startup", 900, 230.1, 240, "exploratory", 155.1, True, "reserve"),
    # A SOLVED task's decisive 137.6s expect driver: +3.0s of headroom, the
    # tightest margin in the table and an accepted risk.
    # LANDING_ALLOWANCE_MIN is the binding constant; raising it loses this.
    _Row("qemu-alpine-ssh", 900, 215.6, 540, "exploratory", 140.6, True, "reserve"),
    # --- exploratory: untouched -------------------------------------------
    _Row("extract-moves-926", 1800, 926.0, 600, "exploratory", 600.0, False, None),
    _Row("extract-moves-1488", 1800, 1488.0, 600, "exploratory", 600.0, False, None),
    _Row("compcert-early", 2400, 2223.0, 1200, "exploratory", 1200.0, False, None),
    _Row("gpt2-codegolf", 900, 584.0, 120, "exploratory", 120.0, False, None),
    # --- verification: exempt from the share cap and the band softener ----
    # X7: the regression. Exploratory arithmetic would shorten a legitimate
    # 300s gate; verification keeps it whole.
    _Row("verify-x7", 900, 400.0, 300, "verification", 300.0, False, None),
    # Same shapes as the exploratory rows above, minus both exemptions.
    _Row("verify-no-share-cap", 500, 500.0, 300, "verification", 300.0, False, None),
    _Row("verify-no-softener", 900, 360.0, 300, "verification", 285.0, True, "reserve"),
    # The reserve still applies to verification — it is an exemption from
    # the two exploratory-only bounds, not from the cap.
    _Row("verify-reserve", 1800, 1706.9, 1800, "verification", 1631.9, True, "reserve"),
)


class TestExecCap:
    """Change 1: share cap (primary) + band guarantee (secondary).

    The table is the plan's simulation against the real bash execs of an
    18-trial benchmark run, carried over with its exact expected values.
    """

    def test_the_table_covers_both_purposes_and_every_reason(self) -> None:
        assert len(_SIMULATION) == 18
        assert {row.reason for row in _SIMULATION} == {
            "share",
            "band",
            "reserve",
            None,
        }
        assert {row.purpose for row in _SIMULATION} == {
            "exploratory",
            "verification",
        }

    @pytest.mark.parametrize(
        "row", _SIMULATION, ids=[row.id for row in _SIMULATION]
    )
    def test_simulation_row(self, row: _Row) -> None:
        deadline = _at(row.budget, row.remaining)
        assert deadline.landing_reserve() == 75.0
        effective, capped, reason = deadline.exec_cap(
            row.requested, purpose=row.purpose  # type: ignore[arg-type]
        )
        assert effective == pytest.approx(row.effective)
        assert capped is row.capped
        assert reason == row.reason

    def test_no_budget_is_a_pure_passthrough(self) -> None:
        assert _deadline(budget=None).exec_cap(3600.0) == (3600.0, False, None)
        assert _deadline(budget=None).exec_cap(
            3600.0, purpose="verification"
        ) == (3600.0, False, None)

    def test_share_cap_wins_over_the_band_guarantee(self) -> None:
        # mcmc: the band guarantee alone permits 1346.9s — this is exactly
        # the runaway the share cap exists to contain.
        deadline = _at(1800.0, 1706.9)
        band_only = 1706.9 - wind_down_threshold(1800.0)
        assert band_only == pytest.approx(1346.9)
        assert deadline.exec_cap(1800.0)[0] == 900.0

    def test_verification_is_exempt_from_the_share_cap(self) -> None:
        # Same remaining as the mcmc row: exploratory is cut to half the
        # budget, verification is bounded only by the landing reserve.
        deadline = _at(1800.0, 1706.9)
        assert deadline.exec_cap(1800.0)[0] == 900.0
        assert deadline.exec_cap(1800.0, purpose="verification")[0] == (
            pytest.approx(1631.9)
        )

    def test_verification_is_exempt_from_the_band_softener(self) -> None:
        # remaining 360 sits above the 300s threshold, so an exploratory
        # exec is softened to 0.25 x 360 = 90s of reserve; verification
        # keeps the plain 75s reserve.
        deadline = _at(900.0, 360.0)
        assert deadline.exec_cap(600.0) == (270.0, True, "band")
        assert deadline.exec_cap(600.0, purpose="verification") == (
            285.0,
            True,
            "reserve",
        )

    def test_the_floor_still_wins_over_the_reserve(self) -> None:
        # remaining 70 of 900: `remaining - reserve` is negative, so the
        # exec floor applies rather than a 0s timeout.
        assert _at(900.0, 70.0).exec_cap(120.0) == (
            EXEC_CAP_FLOOR_SECONDS,
            True,
            "reserve",
        )

    def test_requested_below_every_bound_is_untouched(self) -> None:
        assert _at(1800.0, 1700.0).exec_cap(30.0) == (30.0, False, None)

    def test_capped_flag_tracks_the_requested_timeout(self) -> None:
        # Exactly at the bound is not "capped": nothing was taken away.
        deadline = _at(900.0, 215.6)
        assert deadline.exec_cap(140.6) == (140.6, False, None)
        assert deadline.exec_cap(140.7)[1] is True


class TestExecDecisionReserve:
    """Regression: the decision reports the reserve it *applied*.

    ``exec_cap`` returns only ``(effective, capped, reason)``, so the only
    reserve telemetry could name was the un-softened
    :meth:`Deadline.landing_reserve` — while the band softener routinely
    holds back several times that. Since
    :data:`LANDING_RESERVE_FRACTION` is the constant round 3 retunes from
    exactly this telemetry, the applied number has to be reported, not
    reconstructed: ``remaining - effective`` recovers it for reason
    ``"band"``/``"reserve"`` and not at all for ``"share"``.
    """

    def test_band_softener_reports_the_raised_reserve(self) -> None:
        # The caffe-cifar-10 row: budget 1200, remaining 619.7, requested
        # 600. Threshold 240, softened to 0.25 x 619.7 = 154.925 — over 2x
        # the 75s base reserve.
        decision = _at(1200.0, 619.7).exec_decision(600.0)
        assert decision.reason == "band"
        assert decision.reserve == pytest.approx(154.925)
        assert decision.reserve == pytest.approx(
            LANDING_RESERVE_FRACTION * 619.7
        )
        assert decision.effective == pytest.approx(464.775)

    def test_share_row_still_reports_the_softened_reserve(self) -> None:
        # The case no consumer can reconstruct: the share cap bound, so
        # remaining - effective (1706.9 - 900) says nothing about the
        # reserve. The band softener had still raised it to the threshold.
        deadline = _at(1800.0, 1706.9)
        decision = deadline.exec_decision(1800.0)
        assert decision.reason == "share"
        assert decision.reserve == wind_down_threshold(1800.0) == 360.0
        assert decision.reserve != deadline.landing_reserve()

    def test_in_band_reports_the_base_landing_reserve(self) -> None:
        # Below the threshold the softener never fires, so the applied
        # reserve is the base one — that equality is the signal.
        deadline = _at(900.0, 200.0)
        decision = deadline.exec_decision(300.0)
        assert decision.reason == "reserve"
        assert decision.reserve == deadline.landing_reserve() == 75.0

    def test_verification_reports_the_unsoftened_reserve(self) -> None:
        # Verification is exempt from the softener (and the share cap), so
        # its decisions must never report a softened reserve.
        deadline = _at(900.0, 360.0)
        assert deadline.exec_decision(600.0).reserve == 90.0  # softened
        assert (
            deadline.exec_decision(600.0, purpose="verification").reserve
            == deadline.landing_reserve()
            == 75.0
        )

    def test_uncapped_and_no_budget_still_carry_a_reserve_field(self) -> None:
        # Nothing was taken away, but the field is defined: the reserve
        # that *would* have applied, and 0.0 with no deadline at all.
        untouched = _at(1800.0, 1700.0).exec_decision(30.0)
        assert untouched.capped is False
        assert untouched.reserve == wind_down_threshold(1800.0)
        assert _deadline(budget=None).exec_decision(3600.0) == (
            3600.0,
            False,
            None,
            0.0,
        )

    def test_exec_cap_stays_the_three_tuple_facade(self) -> None:
        # The apply-the-cap callers keep their shape; the decision is a
        # superset of it, never a divergent second implementation.
        deadline = _at(1200.0, 619.7)
        decision = deadline.exec_decision(600.0)
        assert deadline.exec_cap(600.0) == (
            decision.effective,
            decision.capped,
            decision.reason,
        )


class TestExecCapProperties:
    """The two invariants the constants encode, swept rather than sampled."""

    _BUDGETS = (300.0, 900.0, 1200.0, 1800.0, 2400.0, 12000.0)
    _REQUESTS = (10.0, 120.0, 300.0, 600.0, 1200.0, 3600.0)

    @pytest.mark.parametrize("budget", _BUDGETS)
    def test_share_cap_bounds_every_exploratory_exec(self, budget: float) -> None:
        remaining = budget
        while remaining > 0.0:
            deadline = _at(budget, remaining)
            for requested in self._REQUESTS:
                cap = deadline.exec_cap(requested)[0]
                assert cap <= EXEC_MAX_BUDGET_FRACTION * budget
            remaining -= 10.0

    @pytest.mark.parametrize("budget", _BUDGETS)
    def test_band_guarantee_holds_above_the_threshold(self, budget: float) -> None:
        threshold = wind_down_threshold(budget)
        remaining = budget
        while remaining > 0.0:
            deadline = _at(budget, remaining)
            for requested in self._REQUESTS:
                cap = deadline.exec_cap(requested)[0]
                if remaining > threshold:
                    # What is left when the exec returns is enough to wind
                    # down: the threshold itself, softened to a quarter of
                    # what remained.
                    held_back = min(
                        threshold, LANDING_RESERVE_FRACTION * remaining
                    )
                    assert remaining - cap >= held_back - 1e-9
            remaining -= 10.0

    @pytest.mark.parametrize("budget", _BUDGETS)
    def test_capping_never_grows_a_timeout(self, budget: float) -> None:
        remaining = budget
        while remaining > 0.0:
            deadline = _at(budget, remaining)
            for requested in self._REQUESTS:
                exploratory = deadline.exec_cap(requested)[0]
                verification = deadline.exec_cap(
                    requested, purpose="verification"
                )[0]
                assert exploratory <= requested
                assert verification <= requested
                # The exemptions only ever loosen; exploratory is the
                # tighter of the two everywhere.
                assert exploratory <= verification
            remaining -= 10.0


class TestWindDownThresholdMovedHere:
    """Change 1 moved the wind-down band next to the cap that consumes it."""

    def test_constants_and_shape(self) -> None:
        assert WIND_DOWN_FRACTION == 0.2
        assert WIND_DOWN_MIN_REMAINING == 300.0
        assert WIND_DOWN_MAX_REMAINING == 600.0
        assert EXEC_MAX_BUDGET_FRACTION == 0.5
        assert LANDING_RESERVE_FRACTION == 0.25

    @pytest.mark.parametrize(
        "budget,expected",
        [
            (100.0, 50.0),  # half-budget clamp for degenerate budgets
            (900.0, 300.0),  # min clamp
            (1800.0, 360.0),  # the raw fraction
            (2400.0, 480.0),
            (12000.0, 600.0),  # max clamp
        ],
    )
    def test_threshold(self, budget: float, expected: float) -> None:
        assert wind_down_threshold(budget) == expected

    def test_loop_still_re_exports_it(self) -> None:
        from harness import loop

        assert loop.wind_down_threshold is wind_down_threshold
        assert loop.WIND_DOWN_FRACTION == WIND_DOWN_FRACTION
        assert loop.WIND_DOWN_MIN_REMAINING == WIND_DOWN_MIN_REMAINING
        assert loop.WIND_DOWN_MAX_REMAINING == WIND_DOWN_MAX_REMAINING
        assert "wind_down_threshold" in loop.__all__
