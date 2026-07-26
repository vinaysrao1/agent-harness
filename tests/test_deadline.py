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
    MODEL_CALL_WINDOW,
    REMAINING_RESERVE_FRACTION,
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
    """The contract Change 0 exists to guarantee, repaired in round 3.

    After any deadline-capped *exploratory* exec returns, what is left is
    at least :data:`WALL_CLOCK_STOP_FLOOR`, so the loop's hard-stop check
    cannot fire and the agent always gets one more turn to land its answer.

    This class used to assert a weaker contract with a "documented
    exception" for runs already inside the stop floor. The exception was
    stated wrongly and was therefore not a boundary case at all:
    ``allowed = max(EXEC_CAP_FLOOR_SECONDS, remaining - reserve)`` let the
    30s floor escape the reserve as soon as ``remaining < reserve + 30``
    (~105s), so the guarantee was void a full 30s *above* the floor — in a
    band where the loop happily starts turns. It is now clamped, and the
    only remaining exception is ``purpose="verification"``, which is exempt
    on purpose (nothing is landed after a completion gate).
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
                # The repaired contract, with no band exempted: an
                # exploratory exec either leaves the stop floor untouched
                # or is refused the time entirely. `cap == 0.0` is a legal
                # answer inside the floor, where nothing new should start.
                assert remaining - cap >= WALL_CLOCK_STOP_FLOOR or cap == 0.0
                # The exec floor still bounds the *reserve* term, so a
                # command with room to run never gets a spuriously tiny
                # window: below the floor only the stop-floor clamp bites.
                if remaining >= WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS:
                    assert cap >= EXEC_CAP_FLOOR_SECONDS or cap == requested
            remaining -= 10.0

    def test_the_old_documented_exception_is_gone(self) -> None:
        """Was ``test_the_documented_exception_is_inside_the_stop_floor``.

        It asserted that at remaining=70 a 120s request became a 30s exec
        returning with 40s left — below the stop floor — and called that
        acceptable because "the loop is already within one landing turn of
        stopping". Both halves were wrong: returning below the floor is
        precisely what denies the landing turn, and the violation was not
        confined to the floor at all. It began at ``WALL_CLOCK_STOP_FLOOR +
        EXEC_CAP_FLOOR_SECONDS`` = 90s, a band 30s wide in which the loop
        starts turns perfectly happily.
        """
        deadline = Deadline(900.0, scripted_clock([0.0, 830.0]))
        assert deadline.remaining() == 70.0
        assert _exec_cap(deadline, 120.0) == 10.0  # was EXEC_CAP_FLOOR_SECONDS
        assert 70.0 - 10.0 == WALL_CLOCK_STOP_FLOOR
        # The top of the band, 30s clear of the floor: the old arithmetic
        # handed out the 30s floor here too and returned at 59.
        wider = Deadline(900.0, scripted_clock([0.0, 811.0]))
        assert wider.remaining() == 89.0
        assert wider.remaining() > WALL_CLOCK_STOP_FLOOR
        assert _exec_cap(wider, 120.0) == 29.0  # was EXEC_CAP_FLOOR_SECONDS
        assert 89.0 - 29.0 == WALL_CLOCK_STOP_FLOOR

    def test_verification_keeps_the_exemption_deliberately(self) -> None:
        # The completion gate is exempt: shortening it turns a passing
        # check into a capped, inconclusive one, and there is nothing to
        # land after it. So it may still return inside the floor.
        deadline = Deadline(900.0, scripted_clock([0.0, 830.0]))
        assert deadline.remaining() == 70.0
        assert (
            _exec_cap(deadline, 120.0, purpose="verification")
            == EXEC_CAP_FLOOR_SECONDS
        )
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
    # 435.975s (2.1x). REMAINING_RESERVE_FRACTION = 0.5 would leave 1.4x.
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
        """It does not — and that was the landing guarantee's whole hole.

        Kept under its original name, with the assertion inverted. The old
        version asserted that at remaining=70 of 900 the 30s exec floor
        beats the reserve, so a 120s request became a 30s exec. That is the
        bug, not a boundary case: the exec returns at remaining 40, the
        loop's hard stop fires, and the turn the reserve was held back for
        never happens. Two trials died exactly this way —
        ``qemu-alpine-ssh`` seq 238 (30s exec issued at remaining 58.4,
        stop at 28.4) and ``make-doom-for-mips`` seq 553 (issued at 46.8,
        stop at 16.7). Neither was inside the floor when the *turn* began.

        The floor now bounds only the reserve term; it can never be paid
        out of :data:`WALL_CLOCK_STOP_FLOOR`.
        """
        assert _at(900.0, 70.0).exec_cap(120.0) == (10.0, True, "reserve")
        # Not a floor-only defect. The escape ran the whole band from
        # WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS = 89.9s down: at
        # remaining 89 the old arithmetic still handed out 30s and returned
        # at 59, one second inside the floor and one turn short.
        assert _at(900.0, 89.0).exec_cap(120.0) == (29.0, True, "reserve")
        # And the floor still wins where it legitimately can: at or above
        # 90s the clamp is slack, so a nearly-spent budget keeps a usable
        # window instead of a spuriously-failing one.
        assert _at(900.0, 90.0).exec_cap(120.0) == (
            EXEC_CAP_FLOOR_SECONDS,
            True,
            "reserve",
        )
        assert _at(900.0, 104.0).exec_cap(120.0) == (
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




# ---------------------------------------------------------------------------
# Change 1a: the frozen round-2 exec-cap corpus
# ---------------------------------------------------------------------------


class _CapRow(NamedTuple):
    """One real ``exec_capped`` event, plus what 1a does to it.

    ``live_effective`` is the timeout round 2 actually applied; ``runtime``
    and ``timed_out`` are what the command then did. Those three are
    *observations*, checked in so the counterfactual below is arithmetic
    over recorded facts rather than a story. ``effective`` and ``reason``
    are what the shipping code must produce.
    """

    task: str
    seq: int
    budget: float
    remaining: float
    requested: float
    reserve: float
    live_effective: float
    runtime: float
    timed_out: bool
    effective: float
    reason: str | None


#: Every ``exec_capped`` event of the 14-trial round-2 benchmark run: 63
#: rows across 10 tasks, verbatim from the ``state.db`` event logs (the
#: floats are rounded to 3dp; the expected outputs were computed from the
#: rounded inputs, so the table is exact).
#:
#: This is the fence for every future constant change. ``EXEC_CAP_FLOOR_
#: SECONDS``, ``LANDING_ALLOWANCE_MIN``, ``REMAINING_RESERVE_FRACTION`` and
#: ``EXEC_MAX_BUDGET_FRACTION`` are all tuned from exactly this corpus, and
#: moving any of them re-cuts real commands — including the decisive build
#: of a task that currently passes. Here that shows up as a diff, not as a
#: lost trial three days later.
_CAP_CORPUS: tuple[_CapRow, ...] = (

    # -- caffe-cifar-10 ----------------------------------------------
    _CapRow('caffe-cifar-10', 31, 1200, 1028.585, 1800, 75,
            600, 150.951, False, 600, 'share'),
    _CapRow('caffe-cifar-10', 47, 1200, 617.891, 600, 75,
            463.418, 269.944, False, 463.41825, 'band'),
    _CapRow('caffe-cifar-10', 53, 1200, 341.981, 400, 75,
            256.485, 2.302, False, 256.48575, 'band'),
    _CapRow('caffe-cifar-10', 64, 1200, 329.491, 250, 75,
            247.119, 52.671, False, 247.11825, 'band'),
    _CapRow('caffe-cifar-10', 92, 1200, 124.295, 50, 75,
            49.295, 0.98, False, 49.295, 'reserve'),
    _CapRow('caffe-cifar-10', 98, 1200, 87.269, 120, 75,
            30, 0.319, False, 27.269, 'reserve'),
    # -- compile-compcert --------------------------------------------
    _CapRow('compile-compcert', 36, 2400, 2256.432, 1800, 75,
            1200, 467.797, False, 1200, 'share'),
    _CapRow('compile-compcert', 42, 2400, 1777.764, 3600, 75,
            1200, 4.554, False, 1200, 'share'),
    _CapRow('compile-compcert', 73, 2400, 1271.209, 3000, 75,
            953.406, 476.1, False, 953.40675, 'band'),
    _CapRow('compile-compcert', 79, 2400, 790.003, 3000, 75,
            592.502, 191.303, False, 592.50225, 'band'),
    _CapRow('compile-compcert', 85, 2400, 593.641, 1200, 75,
            445.231, 1.47, False, 445.23075, 'band'),
    _CapRow('compile-compcert', 91, 2400, 584.417, 3000, 75,
            438.313, 198.91, False, 438.31275, 'band'),
    _CapRow('compile-compcert', 99, 2400, 379.652, 380, 75,
            304.652, 1.375, False, 304.652, 'reserve'),
    _CapRow('compile-compcert', 105, 2400, 372.076, 2400, 75,
            297.076, 20.493, False, 297.076, 'reserve'),
    _CapRow('compile-compcert', 116, 2400, 338.505, 300, 75,
            263.505, 0.619, False, 263.505, 'reserve'),
    # -- gpt2-codegolf -----------------------------------------------
    _CapRow('gpt2-codegolf', 32, 900, 632.219, 600, 76.896,
            450, 1.044, False, 450, 'share'),
    _CapRow('gpt2-codegolf', 88, 900, 397.432, 300, 83.657,
            298.074, 0.999, False, 298.074, 'band'),
    _CapRow('gpt2-codegolf', 119, 900, 245.493, 300, 93.707,
            151.786, 5.802, False, 151.786, 'reserve'),
    _CapRow('gpt2-codegolf', 130, 900, 227.151, 200, 84.927,
            142.224, 10.233, False, 142.224, 'reserve'),
    _CapRow('gpt2-codegolf', 136, 900, 208.799, 200, 84.927,
            123.873, 10.263, False, 123.872, 'reserve'),
    _CapRow('gpt2-codegolf', 142, 900, 150.176, 120, 84.927,
            65.249, 0.818, False, 65.249, 'reserve'),
    _CapRow('gpt2-codegolf', 148, 900, 140.846, 120, 84.927,
            55.919, 10.434, False, 55.919, 'reserve'),
    # -- make-doom-for-mips ------------------------------------------
    _CapRow('make-doom-for-mips', 451, 900, 186.015, 120, 75,
            111.015, 0.12, False, 111.015, 'reserve'),
    _CapRow('make-doom-for-mips', 457, 900, 178.38, 120, 75,
            103.38, 0.123, False, 103.38, 'reserve'),
    _CapRow('make-doom-for-mips', 463, 900, 145.259, 120, 75,
            70.259, 0.188, False, 70.259, 'reserve'),
    _CapRow('make-doom-for-mips', 469, 900, 133.578, 60, 75,
            58.578, 0.145, False, 58.578, 'reserve'),
    _CapRow('make-doom-for-mips', 475, 900, 127.255, 120, 75,
            52.255, 0.149, False, 52.255, 'reserve'),
    _CapRow('make-doom-for-mips', 481, 900, 122.49, 120, 75,
            47.49, 0.136, False, 47.49, 'reserve'),
    _CapRow('make-doom-for-mips', 487, 900, 116.788, 120, 75,
            41.788, 0.151, False, 41.788, 'reserve'),
    _CapRow('make-doom-for-mips', 493, 900, 110.879, 120, 75,
            35.879, 0.171, False, 35.879, 'reserve'),
    _CapRow('make-doom-for-mips', 499, 900, 104.971, 120, 75,
            30, 0.144, False, 30, 'reserve'),
    _CapRow('make-doom-for-mips', 505, 900, 99.162, 120, 75,
            30, 0.171, False, 30, 'reserve'),
    _CapRow('make-doom-for-mips', 511, 900, 94.347, 120, 75,
            30, 0.134, False, 30, 'reserve'),
    _CapRow('make-doom-for-mips', 517, 900, 89.702, 120, 75,
            30, 0.12, False, 29.702, 'reserve'),
    _CapRow('make-doom-for-mips', 523, 900, 84.959, 120, 75,
            30, 0.129, False, 24.959, 'reserve'),
    _CapRow('make-doom-for-mips', 529, 900, 79.555, 120, 75,
            30, 0.162, False, 19.555, 'reserve'),
    _CapRow('make-doom-for-mips', 535, 900, 74.562, 120, 75,
            30, 0.148, False, 14.562, 'reserve'),
    _CapRow('make-doom-for-mips', 541, 900, 68.444, 120, 75,
            30, 0.161, False, 8.444, 'reserve'),
    _CapRow('make-doom-for-mips', 547, 900, 62.751, 120, 75,
            30, 0.157, False, 2.751, 'reserve'),
    _CapRow('make-doom-for-mips', 553, 900, 46.758, 120, 75,
            30, 30.012, True, 0, 'reserve'),
    # -- make-mips-interpreter ---------------------------------------
    _CapRow('make-mips-interpreter', 145, 1800, 1358.576, 920, 75,
            900, 0.611, False, 900, 'share'),
    _CapRow('make-mips-interpreter', 182, 1800, 1304.933, 920, 75,
            900, 0.509, False, 900, 'share'),
    _CapRow('make-mips-interpreter', 449, 1800, 794.905, 620, 75,
            596.179, 0.484, False, 596.17875, 'band'),
    _CapRow('make-mips-interpreter', 505, 1800, 637.647, 520, 76.873,
            478.235, 0.753, False, 478.23525, 'band'),
    _CapRow('make-mips-interpreter', 549, 1800, 541.089, 520, 75,
            405.817, 0.429, False, 405.81675, 'band'),
    _CapRow('make-mips-interpreter', 585, 1800, 481.81, 620, 75,
            361.357, 0.723, False, 361.3575, 'band'),
    _CapRow('make-mips-interpreter', 623, 1800, 326.631, 260, 75,
            251.631, 0.469, False, 251.631, 'reserve'),
    _CapRow('make-mips-interpreter', 639, 1800, 191.498, 120, 84.379,
            107.119, 0.136, False, 107.119, 'reserve'),
    _CapRow('make-mips-interpreter', 650, 1800, 180.611, 100, 84.379,
            96.232, 0.461, False, 96.232, 'reserve'),
    _CapRow('make-mips-interpreter', 656, 1800, 172.995, 95, 84.379,
            88.616, 75.219, False, 88.616, 'reserve'),
    # -- mcmc-sampling-stan ------------------------------------------
    _CapRow('mcmc-sampling-stan', 21, 1800, 1772.754, 1200, 75,
            900, 406.774, False, 900, 'share'),
    # -- path-tracing ------------------------------------------------
    _CapRow('path-tracing', 248, 1800, 185.321, 120, 81.807,
            103.514, 6.338, False, 103.514, 'reserve'),
    # -- qemu-alpine-ssh ---------------------------------------------
    _CapRow('qemu-alpine-ssh', 221, 900, 207.134, 400, 75,
            132.134, 21.366, False, 132.134, 'reserve'),
    _CapRow('qemu-alpine-ssh', 227, 900, 180.546, 125, 75,
            105.546, 106.014, True, 105.546, 'reserve'),
    _CapRow('qemu-alpine-ssh', 238, 900, 58.413, 125, 75,
            30, 30.012, True, 0, 'reserve'),
    # -- qemu-startup ------------------------------------------------
    _CapRow('qemu-startup', 201, 900, 465.114, 660, 75,
            348.836, 108.416, False, 348.8355, 'band'),
    # -- write-compressor --------------------------------------------
    _CapRow('write-compressor', 43, 900, 376.353, 600, 120,
            256.353, 1, False, 256.353, 'reserve'),
    _CapRow('write-compressor', 74, 900, 302.333, 600, 88.696,
            213.637, 1.002, False, 213.637, 'reserve'),
    _CapRow('write-compressor', 92, 900, 248.954, 280, 88.696,
            160.258, 0.767, False, 160.258, 'reserve'),
    _CapRow('write-compressor', 118, 900, 192.331, 280, 75,
            117.331, 0.803, False, 117.331, 'reserve'),
    _CapRow('write-compressor', 124, 900, 173.599, 120, 75,
            98.599, 0.136, False, 98.599, 'reserve'),
    _CapRow('write-compressor', 130, 900, 160.973, 120, 75,
            85.973, 0.134, False, 85.973, 'reserve'),
    _CapRow('write-compressor', 146, 900, 111.068, 80, 77.841,
            33.227, 0.786, False, 33.227, 'reserve'),
)

#: Every ``exec_capped`` event of the round-3 rerun: 22 rows across 8
#: tasks, extracted from the same ``state.db`` shape as
#: :data:`_CAP_CORPUS` (``jobs/round3-rerun/2026-07-25__21-19-27``). Kept
#: as its own tuple rather than appended to :data:`_CAP_CORPUS` so the
#: round-2 fence above stays byte-identical: the two corpora answer
#: different questions, and only this one carries a caveat.
#:
#: **The caveat**: 8 of the 14 round-3 trials ran with a contaminated
#: clock, so ``runtime`` here (the ``exec_capped`` → next ``tool_result``
#: delta) is an upper bound, not a measurement — ``compile-compcert`` 98
#: reads 1901s against a 600.9s cap and did not time out. Nothing below
#: draws a conclusion from a round-3 runtime; the counterfactual's
#: load-bearing rows are round-2 ones, whose timings are clean.
_R3_CAP_CORPUS: tuple[_CapRow, ...] = (
    # -- caffe-cifar-10 ----------------------------------------------
    _CapRow('caffe-cifar-10', 26, 1200, 973.17, 1800, 75,
            600, 116.49, False, 600, 'share'),
    _CapRow('caffe-cifar-10', 47, 1200, 687.641, 590, 75,
            515.731, 16.931, False, 515.73075, 'band'),
    _CapRow('caffe-cifar-10', 53, 1200, 662.689, 590, 75,
            497.017, 218.818, False, 497.01675, 'band'),
    _CapRow('caffe-cifar-10', 64, 1200, 429.235, 590, 75,
            321.926, 5.144, False, 321.92625, 'band'),
    _CapRow('caffe-cifar-10', 105, 1200, 335.987, 590, 75,
            251.99, 8.347, False, 251.99025, 'band'),
    # The defect case: capped to 239.8s at 319.7s remaining, so it came
    # back with 79.9s against its own 300s wind-down threshold.
    _CapRow('caffe-cifar-10', 111, 1200, 319.684, 300, 75,
            239.763, 240.016, True, 239.763, 'band'),
    # -- compile-compcert --------------------------------------------
    _CapRow('compile-compcert', 61, 2400, 2181.666, 1800, 75,
            1200, 480.955, False, 1200, 'share'),
    _CapRow('compile-compcert', 82, 2400, 984.309, 1200, 75,
            738.232, 151.301, False, 738.23175, 'band'),
    _CapRow('compile-compcert', 98, 2400, 801.23, 700, 75,
            600.922, 1901.296, False, 600.9225, 'band'),
    # -- gpt2-codegolf -----------------------------------------------
    _CapRow('gpt2-codegolf', 31, 900, 717.083, 600, 75,
            450, 0.271, False, 450, 'share'),
    _CapRow('gpt2-codegolf', 132, 900, 196.78, 120, 83.608,
            113.172, 0.551, False, 113.172, 'reserve'),
    _CapRow('gpt2-codegolf', 140, 900, 162.999, 120, 87.279,
            75.719, 4.012, False, 75.72, 'reserve'),
    _CapRow('gpt2-codegolf', 151, 900, 132.82, 60, 87.279,
            45.54, 3.923, False, 45.541, 'reserve'),
    _CapRow('gpt2-codegolf', 157, 900, 120.279, 45, 87.279,
            33, 3.898, False, 33, 'reserve'),
    _CapRow('gpt2-codegolf', 163, 900, 103.384, 40, 87.279,
            30, 3.891, False, 30, 'reserve'),
    # -- make-doom-for-mips ------------------------------------------
    # The one row the shipping arithmetic does not reproduce on its own:
    # `bash`'s MIN_EXEC_SECONDS floor (harness.tools.builtin) turns this
    # 0.0 into the 1.0 that was actually handed out.
    _CapRow('make-doom-for-mips', 103, 900, 15.802, 240, 114.758,
            1, 1.016, True, 0, 'reserve'),
    # -- mcmc-sampling-stan ------------------------------------------
    _CapRow('mcmc-sampling-stan', 49, 1800, 1484.811, 1200, 75,
            900, 1930.706, False, 900, 'share'),
    _CapRow('mcmc-sampling-stan', 117, 1800, 583.675, 600, 75,
            437.756, 400.176, False, 437.75625, 'band'),
    # -- mteb-leaderboard --------------------------------------------
    _CapRow('mteb-leaderboard', 146, 3600, 2979.286, 3000, 75,
            1800, 961.89, False, 1800, 'share'),
    # -- qemu-alpine-ssh ---------------------------------------------
    _CapRow('qemu-alpine-ssh', 138, 900, 279.86, 260, 75,
            204.86, 170.398, False, 204.86, 'reserve'),
    _CapRow('qemu-alpine-ssh', 154, 900, 84.428, 40, 75,
            24.428, 1.49, False, 24.428, 'reserve'),
    # -- write-compressor --------------------------------------------
    _CapRow('write-compressor', 48, 900, 361.712, 300, 87.565,
            271.284, 0.177, False, 271.284, 'band'),
)

#: Both rounds: the 85 real capped execs every claim about this arithmetic
#: is measured against.
_ALL_CAP_ROWS: tuple[_CapRow, ...] = _CAP_CORPUS + _R3_CAP_CORPUS

#: The three tasks the round-2 run actually solved (grader ``reward``, not
#: the agent's own ``verification_passed``). 1a must not touch a single one
#: of their execs.
_SOLVED_TASKS = frozenset(
    {"compile-compcert", "mcmc-sampling-stan", "qemu-startup"}
)


def _at_reserve(row: _CapRow) -> Deadline:
    """A deadline reproducing ``row``'s state at the moment of the exec.

    One observation of ``reserve - WALL_CLOCK_STOP_FLOOR`` reproduces the
    recorded landing reserve exactly (nearest-rank p75 of a window of one
    is that observation, and every recorded reserve is inside the
    allowance bounds), so the row's own reserve drives the arithmetic
    rather than a stand-in.
    """
    deadline = Deadline(
        row.budget, scripted_clock([0.0, row.budget - row.remaining])
    )
    deadline.observe_model_call(row.reserve - WALL_CLOCK_STOP_FLOOR)
    return deadline


class TestFrozenExecCapCorpus:
    """Change 1a against all 63 real capped execs of the round-2 run.

    Be clear about what this buys: **zero tasks.** Nine rows change, every
    one of them already pinned at :data:`EXEC_CAP_FLOOR_SECONDS`; seven ran
    for under a third of a second and two had already timed out at 30s. No
    exec of any solved trial moves. This is a prospective repair of a
    guarantee the reserve arithmetic was claiming and not keeping — the
    tests below are what make the claim true, not a recovery story.
    """

    def test_the_corpus_is_the_whole_round_two_cap_log(self) -> None:
        assert len(_CAP_CORPUS) == 63
        assert len({(row.task, row.seq) for row in _CAP_CORPUS}) == 63
        assert len({row.task for row in _CAP_CORPUS}) == 10
        assert {row.reason for row in _CAP_CORPUS} == {
            "share",
            "band",
            "reserve",
        }
        assert _SOLVED_TASKS <= {row.task for row in _CAP_CORPUS}

    @pytest.mark.parametrize(
        "row", _CAP_CORPUS, ids=[f"{r.task}-{r.seq}" for r in _CAP_CORPUS]
    )
    def test_corpus_row(self, row: _CapRow) -> None:
        deadline = _at_reserve(row)
        assert deadline.landing_reserve() == pytest.approx(row.reserve)
        decision = deadline.exec_decision(row.requested)
        assert decision.effective == pytest.approx(row.effective, abs=1e-6)
        assert decision.reason == row.reason
        assert decision.capped is (row.effective < row.requested)

    @pytest.mark.parametrize(
        "row", _CAP_CORPUS, ids=[f"{r.task}-{r.seq}" for r in _CAP_CORPUS]
    )
    def test_every_row_leaves_the_stop_floor_alone(self, row: _CapRow) -> None:
        # The guarantee, on real inputs: the exec either returns with the
        # stop floor intact or is given no time at all.
        assert (
            row.remaining - row.effective >= WALL_CLOCK_STOP_FLOOR
            or row.effective == 0.0
        )

    def test_exactly_nine_rows_change_and_all_were_at_the_floor(self) -> None:
        changed = [
            row
            for row in _CAP_CORPUS
            if abs(row.effective - row.live_effective) > 0.01
        ]
        assert [(row.task, row.seq) for row in changed] == [
            ("caffe-cifar-10", 98),
            ("make-doom-for-mips", 517),
            ("make-doom-for-mips", 523),
            ("make-doom-for-mips", 529),
            ("make-doom-for-mips", 535),
            ("make-doom-for-mips", 541),
            ("make-doom-for-mips", 547),
            ("make-doom-for-mips", 553),
            ("qemu-alpine-ssh", 238),
        ]
        # Every one of them is the max() hole itself: round 2 handed out
        # exactly the exec floor out of the stop floor's protection.
        for row in changed:
            assert row.live_effective == EXEC_CAP_FLOOR_SECONDS
            assert row.remaining < (
                WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS
            )
            assert row.effective < row.live_effective

    def test_no_shortened_row_would_have_cut_a_running_command(self) -> None:
        """The refuted objection, checked against the recorded runtimes.

        v1 proposed refusing execs below ``MIN_USEFUL_EXEC_SECONDS = 5.0``.
        The corpus says small windows are useful: the median capped exec
        ran 0.79s. Of the nine shortened rows, seven completed in 0.12–0.32s
        and still fit; the two that do not (doom 553, qemu-alpine 238) had
        already timed out at 30s, so no completing command is cut short.
        """
        shortened = [
            row
            for row in _CAP_CORPUS
            if abs(row.effective - row.live_effective) > 0.01
        ]
        completed = [row for row in shortened if not row.timed_out]
        assert len(completed) == 7
        for row in completed:
            assert row.runtime < row.effective
            assert row.runtime <= 0.32
        already_lost = [row for row in shortened if row.timed_out]
        assert [(row.task, row.seq) for row in already_lost] == [
            ("make-doom-for-mips", 553),
            ("qemu-alpine-ssh", 238),
        ]
        for row in already_lost:
            assert row.live_effective == EXEC_CAP_FLOOR_SECONDS
            assert row.runtime >= EXEC_CAP_FLOOR_SECONDS

    def test_the_median_capped_exec_is_under_a_second(self) -> None:
        # The number that killed MIN_USEFUL_EXEC_SECONDS. An 8.4s or a
        # 2.8s window is not a token gesture on this corpus; it is ten
        # times the median command.
        runtimes = sorted(row.runtime for row in _CAP_CORPUS)
        assert runtimes[len(runtimes) // 2] == pytest.approx(0.786, abs=1e-3)

    def test_no_solved_trials_exec_is_touched(self) -> None:
        # The risk that matters: 1a must not re-cut a command a passing
        # task depended on. Every solved-trial cap already left >= 75s.
        for row in _CAP_CORPUS:
            if row.task in _SOLVED_TASKS:
                assert row.effective == pytest.approx(
                    row.live_effective, abs=1e-3
                )
                assert row.remaining - row.effective >= 75.0

    def test_the_two_hard_stopped_trials_are_the_motivating_rows(self) -> None:
        # F2: both hard-stopped trials issued their last exec *after* the
        # model call had already spent the margin, so 1a alone converts
        # them to a zero window. That is the loop's problem, not the tool's
        # — see 1c's landing band in harness.loop.
        by_key = {(row.task, row.seq): row for row in _CAP_CORPUS}
        qemu = by_key[("qemu-alpine-ssh", 238)]
        doom = by_key[("make-doom-for-mips", 553)]
        for row in (qemu, doom):
            assert row.remaining < WALL_CLOCK_STOP_FLOOR
            assert row.live_effective == EXEC_CAP_FLOOR_SECONDS
            assert row.effective == 0.0
            assert row.timed_out


def _strict_band_effective(row: _CapRow) -> float:
    """Candidate B: what a *real* band guarantee would have handed out.

    The repair this project keeps being tempted by — reserve the whole
    :func:`wind_down_threshold` whenever ``remaining`` is above it, instead
    of a quarter of what remains — with the same
    :data:`EXEC_CAP_FLOOR_SECONDS` softening and stop-floor clamp the
    shipping arithmetic uses, so the two differ only in the one term. It
    lives here, not in ``harness.deadline``, because it is a rejected
    design kept only as evidence.
    """
    deadline = _at_reserve(row)
    reserve = deadline.landing_reserve()
    threshold = wind_down_threshold(row.budget)
    if row.remaining > threshold:
        reserve = max(reserve, threshold)
    allowed = max(EXEC_CAP_FLOOR_SECONDS, row.remaining - reserve)
    allowed = min(allowed, max(0.0, row.remaining - WALL_CLOCK_STOP_FLOOR))
    return min(row.requested, EXEC_MAX_BUDGET_FRACTION * row.budget, allowed)


class TestTheBandIsNotAGuarantee:
    """The honest contract, pinned so it cannot quietly become a promise.

    ``harness.deadline`` used to document a **band guarantee**: "an exec
    started above the wind-down threshold hands control back with enough
    time to actually wind down". The arithmetic never did that. It reserves
    ``min(threshold, REMAINING_RESERVE_FRACTION * remaining)`` — a
    *proportional* bound whose ceiling happens to be the threshold — and
    since :func:`wind_down_threshold` floors at 300s, the proportion is the
    binding term throughout the wind-down region of every budget this
    harness has run (``4 * threshold >= 1200``).

    Round 5 fixed the documentation and the sentence the *model* reads
    (``harness.tools.builtin._CAP_ADVICE["band"]``) and deliberately left
    the arithmetic alone. These tests are the reason for the second half of
    that decision.
    """

    @pytest.mark.parametrize(
        "row", _ALL_CAP_ROWS, ids=[f"{r.task}-{r.seq}" for r in _ALL_CAP_ROWS]
    )
    def test_the_true_invariant_holds_on_every_corpus_row(
        self, row: _CapRow
    ) -> None:
        """What an exploratory exec really promises, on all 85 real rows.

        ``effective <= max(remaining - landing_reserve(), (1 -
        REMAINING_RESERVE_FRACTION) * remaining)`` and ``effective <= 0.5 *
        budget``. Note what is *not* asserted: nothing of the form
        ``remaining - effective >= wind_down_threshold(budget)``.
        """
        deadline = _at_reserve(row)
        decision = deadline.exec_decision(row.requested)
        # The shipping arithmetic reproduces every recorded row, so an
        # accidental change to it fails here rather than in production.
        assert decision.effective == pytest.approx(row.effective, abs=1e-6)
        assert decision.reason == row.reason

        proportional = (1.0 - REMAINING_RESERVE_FRACTION) * row.remaining
        assert decision.effective <= max(
            row.remaining - deadline.landing_reserve(), proportional
        ) + 1e-9
        assert decision.effective <= EXEC_MAX_BUDGET_FRACTION * row.budget

    def test_the_wind_down_band_is_not_guaranteed_and_we_say_so(self) -> None:
        """The negative test: the band is a ceiling, not a promise.

        caffe-cifar-10 round 3 seq 111, verbatim: budget 1200 (threshold
        300), 319.68s remaining, a 300s command. The cap hands out 239.76s
        — three quarters of what was left — and the trial came back with
        79.92s against its own 300s wind-down threshold, which is exactly
        the turn it was documented to be given and was not.

        A future change that "repairs" the guarantee will fail this test.
        Before repairing it, read
        :meth:`test_strict_band_would_have_killed_a_solved_trials_build`:
        the repair was measured and it costs a solved trial.
        """
        deadline = _at(1200.0, 319.68, observations=(15.0,))
        threshold = wind_down_threshold(1200.0)
        assert threshold == 300.0
        decision = deadline.exec_decision(300.0)

        assert decision.effective == pytest.approx(239.76, abs=0.01)
        assert decision.reason == "band"
        # Three quarters of what remained, not "everything above the
        # threshold" — the reserve is 0.25 x 319.68, well under the 300s
        # threshold it is named after.
        assert decision.reserve == pytest.approx(
            REMAINING_RESERVE_FRACTION * 319.68
        )
        assert decision.reserve < threshold
        # The claim that was false, asserted as false.
        assert 319.68 - decision.effective < threshold

    #: The counterfactual, checked in. ``(corpus, task, seq, strict,
    #: verdict)`` where ``strict`` is what :func:`_strict_band_effective`
    #: would have handed out. ``verdict`` is ``"kills"`` when the command
    #: actually ran longer than ``strict`` — i.e. a real band guarantee
    #: would have SIGKILLed a command that completed.
    _COUNTERFACTUAL: tuple[tuple[str, str, int, float, str], ...] = (
        # THE row. A `make -j 6` on a trial that graded reward=1.0: it ran
        # 198.9s and returned exit 0 inside its 438.3s cap. Strict band
        # enforcement gives it 104.4s and kills it mid-build.
        ("R2", "compile-compcert", 91, 104.417, "kills"),
        ("R2", "caffe-cifar-10", 64, 30.0, "kills"),
        # The trial the repair was proposed *for*: it would have been cut
        # from 239.8s to 30s and still not have finished.
        ("R3", "caffe-cifar-10", 111, 30.0, "shortened"),
        ("R3", "mcmc-sampling-stan", 117, 223.675, "kills"),
        ("R3", "compile-compcert", 98, 321.23, "shortened"),
        # Solved-trial execs that survive B but lose most of their margin.
        ("R2", "compile-compcert", 73, 791.209, "shortened"),
        ("R2", "compile-compcert", 79, 310.003, "shortened"),
        ("R2", "compile-compcert", 85, 113.641, "shortened"),
        ("R2", "qemu-startup", 201, 165.114, "shortened"),
        ("R3", "write-compressor", 48, 61.712, "shortened"),
    )

    @pytest.mark.parametrize(
        "corpus,task,seq,strict,verdict",
        _COUNTERFACTUAL,
        ids=[f"{c}-{t}-{s}" for c, t, s, _e, _v in _COUNTERFACTUAL],
    )
    def test_strict_band_would_have_killed_a_solved_trials_build(
        self,
        corpus: str,
        task: str,
        seq: int,
        strict: float,
        verdict: str,
    ) -> None:
        """Why the arithmetic stays proportional. Measured, not assumed.

        Over all 85 rows, reserving the whole threshold changes 26 of them
        — and on ``compile-compcert`` round 2 seq 91 it converts a command
        that completed on a *solved* trial into one that is killed. That is
        the price of the guarantee, and it buys nothing: the trial that
        motivated the repair (caffe round 3 seq 111) times out either way.
        """
        rows = _CAP_CORPUS if corpus == "R2" else _R3_CAP_CORPUS
        (row,) = [r for r in rows if r.task == task and r.seq == seq]
        assert _strict_band_effective(row) == pytest.approx(strict, abs=0.01)
        assert strict < row.effective  # every one of these is a cut
        if verdict == "kills":
            # The command ran longer than the strict cap would have
            # allowed, so it would have been killed mid-flight.
            assert row.runtime > strict
            if corpus == "R2":
                # Round-2 timings are clean, so this is the strong claim:
                # it completed inside the shipping cap and would not have
                # completed inside the strict one.
                assert not row.timed_out
                assert row.runtime < row.effective

    def test_the_repair_would_change_twenty_six_of_eighty_five_rows(
        self,
    ) -> None:
        # The scale of the change, so "it only affects a pathological case"
        # cannot be asserted without checking.
        assert len(_ALL_CAP_ROWS) == 85
        changed = [
            row
            for row in _ALL_CAP_ROWS
            if abs(_strict_band_effective(row) - row.effective) > 0.01
        ]
        assert len(changed) == 26
        # And it is one-directional: strict enforcement only ever takes
        # time away.
        for row in changed:
            assert _strict_band_effective(row) < row.effective

    def test_the_solved_trials_build_is_the_decisive_row(self) -> None:
        # Named separately from the table so `git log -S` finds it: this is
        # the row that rejected the design.
        (row,) = [
            r
            for r in _CAP_CORPUS
            if r.task == "compile-compcert" and r.seq == 91
        ]
        assert row.task in _SOLVED_TASKS
        assert row.effective == pytest.approx(438.31275)
        assert row.runtime == pytest.approx(198.91)
        assert row.timed_out is False
        strict = _strict_band_effective(row)
        assert strict == pytest.approx(104.417, abs=0.01)
        assert strict < row.runtime < row.effective


class TestStopFloorClampProperty:
    """1a's invariant, swept over a dense grid rather than sampled.

    ``hypothesis`` is not a dependency of this project, so the "property"
    is an exhaustive product over budgets, remainings, requests and
    provider latencies — deterministic, offline, and covering the whole
    band the clamp acts in at 1s resolution.
    """

    _BUDGETS = (300.0, 900.0, 1200.0, 1800.0, 2400.0, 12000.0)
    _REQUESTS = (0.0, 1.0, 30.0, 120.0, 600.0, 3600.0)
    _OBSERVATIONS = ((), (5.0,) * 4, (15.1,) * 8, (200.0,) * 4)

    @pytest.mark.parametrize("budget", _BUDGETS)
    @pytest.mark.parametrize("observations", _OBSERVATIONS)
    def test_exploratory_execs_never_spend_the_stop_floor(
        self, budget: float, observations: tuple[float, ...]
    ) -> None:
        elapsed = 0.0

        def clock() -> float:
            return elapsed

        deadline = Deadline(budget, clock)
        for seconds in observations:
            deadline.observe_model_call(seconds)

        # 1s resolution through the whole clamp band (0..150), then coarser
        # for the rest of the budget where the clamp is provably slack.
        points = [float(n) for n in range(0, 151)]
        points += [float(n) for n in range(160, int(budget) + 1, 10)]
        for remaining in points:
            elapsed = budget - remaining
            for requested in self._REQUESTS:
                effective = deadline.exec_cap(requested)[0]
                assert (
                    remaining - effective >= WALL_CLOCK_STOP_FLOOR
                    or effective == 0.0
                )
                assert effective <= requested
                assert effective >= 0.0

    @pytest.mark.parametrize("budget", _BUDGETS)
    @pytest.mark.parametrize("observations", _OBSERVATIONS)
    def test_the_clamp_only_ever_tightens(
        self, budget: float, observations: tuple[float, ...]
    ) -> None:
        """It must not manufacture time, and must change nothing above 90s.

        The oracle is the pre-change formula, spelled out here: the clamp
        is a ``min`` against the pre-change ``allowed``, so the only
        interesting claim is *where* it binds — below
        ``WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS`` and nowhere
        else. Everything above that band, including every solved trial's
        decisive command, is byte-identical to round 2.
        """
        elapsed = 0.0

        def clock() -> float:
            return elapsed

        deadline = Deadline(budget, clock)
        for seconds in observations:
            deadline.observe_model_call(seconds)

        threshold = wind_down_threshold(budget)
        for remaining in [float(n) for n in range(0, int(budget) + 1, 5)]:
            elapsed = budget - remaining
            reserve = deadline.landing_reserve()
            if remaining > threshold:
                reserve = max(
                    reserve,
                    min(threshold, REMAINING_RESERVE_FRACTION * remaining),
                )
            unclamped = max(EXEC_CAP_FLOOR_SECONDS, remaining - reserve)
            for requested in self._REQUESTS:
                before = min(
                    requested, EXEC_MAX_BUDGET_FRACTION * budget, unclamped
                )
                after = deadline.exec_cap(requested)[0]
                assert after <= before
                if remaining >= (
                    WALL_CLOCK_STOP_FLOOR + EXEC_CAP_FLOOR_SECONDS
                ):
                    assert after == pytest.approx(before)


class TestAffordableExecSeconds:
    """1b's helper: the largest exec this run can afford, right now."""

    def test_none_without_a_deadline(self) -> None:
        assert _deadline(budget=None).affordable_exec_seconds() is None

    def test_asking_for_no_more_than_it_is_never_capped(self) -> None:
        # The property the bash default relies on: pre-clamping to this
        # number means the cap cannot bite, so no cap is ever reported for
        # a timeout the agent did not choose.
        for remaining in [float(n) for n in range(0, 901, 5)]:
            deadline = _at(900.0, remaining)
            affordable = deadline.affordable_exec_seconds()
            assert affordable is not None
            decision = deadline.exec_decision(min(120.0, affordable))
            assert decision.capped is False
            assert decision.reason is None
            assert decision.effective == min(120.0, affordable)

    def test_it_is_the_unbounded_decision_not_a_second_formula(self) -> None:
        # Defined as exec_decision of an unbounded request, so the three
        # bounds can never drift out of agreement with it.
        deadline = _at(1800.0, 1706.9)
        assert deadline.affordable_exec_seconds() == pytest.approx(900.0)
        assert deadline.affordable_exec_seconds() == pytest.approx(
            deadline.exec_decision(1e9).effective
        )

    def test_it_respects_the_stop_floor_clamp(self) -> None:
        # rem 64 of 900: reserve 75, so the old floor would have said 30s.
        # What the run can actually afford is 4s — and 4s still runs a `cp`.
        assert _at(900.0, 64.0).affordable_exec_seconds() == pytest.approx(4.0)
        assert _at(900.0, 30.0).affordable_exec_seconds() == 0.0

    def test_verification_asks_for_its_own_exemption(self) -> None:
        # rem 400 of 900 is above the 300s threshold, so an exploratory
        # exec is softened to 0.25 x 400 = 100s of reserve; verification
        # keeps the plain 75s one.
        deadline = _at(900.0, 400.0)
        assert deadline.affordable_exec_seconds() == pytest.approx(300.0)
        assert deadline.affordable_exec_seconds(
            purpose="verification"
        ) == pytest.approx(325.0)
        # The exemptions show up where they bite: the share cap.
        early = _at(1800.0, 1706.9)
        assert early.affordable_exec_seconds() == pytest.approx(900.0)
        assert early.affordable_exec_seconds(
            purpose="verification"
        ) == pytest.approx(1631.9)


class TestLandingFlag:
    """1c's tool-side gate: explicit state, set by the loop, read by bash."""

    def test_off_by_default(self) -> None:
        assert _deadline().landing is False
        assert _deadline(budget=None).landing is False

    def test_begin_landing_is_idempotent(self) -> None:
        deadline = _deadline()
        assert deadline.begin_landing() is True
        assert deadline.landing is True
        # A second caller (a subagent's loop, or a re-check) must not be
        # able to re-arm it and re-inject the notice.
        assert deadline.begin_landing() is False
        assert deadline.landing is True

    def test_it_is_state_not_arithmetic(self) -> None:
        # Nothing about remaining time sets it: only the loop does. That is
        # what stops a harness-supplied default from ever being read as an
        # agent asking to run something long.
        deadline = _at(900.0, 1.0)
        assert deadline.landing is False
        assert deadline.exec_decision(120.0).effective == 0.0
        assert deadline.landing is False


class TestExecDecisionReserve:
    """Regression: the decision reports the reserve it *applied*.

    ``exec_cap`` returns only ``(effective, capped, reason)``, so the only
    reserve telemetry could name was the un-raised
    :meth:`Deadline.landing_reserve` — while the remaining-share reserve
    routinely holds back several times that. Since
    :data:`REMAINING_RESERVE_FRACTION` is the constant retuned from
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
            REMAINING_RESERVE_FRACTION * 619.7
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
    def test_the_remaining_share_reserve_holds_above_the_threshold(
        self, budget: float
    ) -> None:
        threshold = wind_down_threshold(budget)
        remaining = budget
        while remaining > 0.0:
            deadline = _at(budget, remaining)
            for requested in self._REQUESTS:
                cap = deadline.exec_cap(requested)[0]
                if remaining > threshold:
                    # What is held back is a quarter of what remained,
                    # ceilinged by the threshold — *not* enough to wind
                    # down, in general. See TestTheBandIsNotAGuarantee.
                    held_back = min(
                        threshold, REMAINING_RESERVE_FRACTION * remaining
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
        assert REMAINING_RESERVE_FRACTION == 0.25

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
