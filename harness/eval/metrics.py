"""Metrics for a PR-replay run (S-401).

Five numbers, chosen because pass rate alone cannot see the two ways an agent
"succeeds" badly.

**Diff precision** asks whether the agent changed *the files the change
touched*, not merely enough files to go green. An agent that rewrites twelve
files to fix one has not done the task well even if the tests pass.

**Regression count** asks what else it broke. A change that makes its own tests
pass by breaking three others is a net loss that pass rate scores as a win.

Two distinctions this module refuses to blur, both because collapsing them
would let a broken measurement read as a healthy one:

*Not measured* is not zero. ``regressions=None`` means no regression command
was configured; reporting that as ``0`` would claim a clean bill of health
nobody checked for. Same for ``turns_to_first_edit``: "never edited" and "we
had no telemetry to tell" are different facts.

*A tampered pass is not a pass.* The tests live in the agent's own writable
tree and deleting them is a winning move under most runners. An outcome that
is both ``passed`` and ``tampered`` is rejected at construction rather than
filtered downstream, so no code path can produce one and forget to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TaskOutcome", "SuiteReport", "diff_precision", "TamperedPassError"]


class TamperedPassError(ValueError):
    """A trial claimed a pass while having modified the grader."""


def diff_precision(touched: set[str], reference: set[str]) -> tuple[float, float]:
    """``(precision, recall)`` of the agent's file set against the change's.

    Precision falls when the agent touched files the change did not; recall
    falls when it missed files the change did. Both are reported because they
    fail in opposite directions and a single blended number would hide which.
    An empty reference or an empty touched set yields ``(0.0, 0.0)`` rather
    than a vacuous 1.0.
    """
    if not reference or not touched:
        return 0.0, 0.0
    hit = len(touched & reference)
    return hit / len(touched), hit / len(reference)


@dataclass(frozen=True)
class TaskOutcome:
    """What one trial of one task produced."""

    task_id: str
    passed: bool
    tokens: int = 0
    turns: int = 0
    #: Turn number of the first change to the work tree, ``None`` if none
    #: happened. Meaningful only when :attr:`first_edit_measured` is True.
    turns_to_first_edit: int | None = None
    #: Whether the substrate telemetry needed for the above was available.
    #: False means "unknown", which :class:`SuiteReport` keeps out of both
    #: the mean and the never-edited count.
    first_edit_measured: bool = False
    files_touched: frozenset[str] = frozenset()
    reference_files: frozenset[str] = frozenset()
    #: Tests broken outside the task's own, or ``None`` when no regression
    #: command was configured. ``None`` is not zero.
    regressions: int | None = None
    #: The agent modified, deleted or disabled the grader.
    tampered: bool = False
    refused: bool = False
    errored: bool = False
    #: The agent hit its turn or wall-clock budget rather than finishing.
    #: Still graded -- being too slow is a real failure and TB2 scores it the
    #: same way -- but reported separately, because a pass rate is not
    #: interpretable without knowing how many attempts ran out of time.
    budget_paused: bool = False

    def __post_init__(self) -> None:
        if self.passed and self.tampered:
            raise TamperedPassError(
                f"{self.task_id}: a trial that modified the grader cannot be "
                "scored as a pass; the runner must set passed=False"
            )

    @property
    def edited(self) -> bool:
        return bool(self.files_touched)

    @property
    def precision(self) -> float:
        return diff_precision(set(self.files_touched), set(self.reference_files))[0]

    @property
    def recall(self) -> float:
        return diff_precision(set(self.files_touched), set(self.reference_files))[1]


@dataclass
class SuiteReport:
    """Aggregate over N tasks x K trials.

    Every rate here carries its denominator as a sibling property, because a
    mean over a silently-filtered subset is how a metric stops measuring what
    its name says without anything looking wrong.
    """

    outcomes: list[TaskOutcome] = field(default_factory=list)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return self._mean([1.0 if o.passed else 0.0 for o in self.outcomes])

    @property
    def token_trials(self) -> int:
        """Trials the token mean is taken over: those that actually ran.

        An errored trial is constructed with ``tokens=0`` because the run
        never returned a usage figure -- not because it was free. A trial that
        burned 400k tokens and then hit a transport error would otherwise pull
        the mean toward zero, making a flaky provider look like an efficiency
        gain.
        """
        return sum(1 for o in self.outcomes if not o.errored)

    @property
    def tokens_per_task(self) -> float:
        return self._mean([float(o.tokens) for o in self.outcomes if not o.errored])

    @property
    def mean_turns(self) -> float:
        return self._mean([float(o.turns) for o in self.outcomes if not o.errored])

    # -- first edit ------------------------------------------------------

    @property
    def first_edit_trials(self) -> int:
        """Trials where first-edit timing was actually observed."""
        return sum(1 for o in self.outcomes if o.first_edit_measured)

    @property
    def mean_turns_to_first_edit(self) -> float:
        # Only over measured trials that edited at all: averaging in "never
        # edited" as zero would make a paralysed agent look decisive, and
        # averaging in "not measured" would invent data.
        seen = [
            float(o.turns_to_first_edit)
            for o in self.outcomes
            if o.first_edit_measured and o.turns_to_first_edit is not None
        ]
        return self._mean(seen)

    @property
    def never_edited(self) -> int:
        """Measured trials in which the work tree never changed."""
        return sum(
            1
            for o in self.outcomes
            if o.first_edit_measured and o.turns_to_first_edit is None
        )

    # -- diff shape ------------------------------------------------------

    @property
    def precision_trials(self) -> int:
        """Trials the precision/recall means are taken over.

        Precision is undefined for a trial that touched nothing (0/0), so
        those are excluded -- and the count is published so the mean can
        never be read as "the agent was precise" when it mostly did nothing.
        :attr:`never_edited` is the other half of that picture.
        """
        return sum(1 for o in self.outcomes if o.edited)

    @property
    def mean_precision(self) -> float:
        return self._mean([o.precision for o in self.outcomes if o.edited])

    @property
    def mean_recall(self) -> float:
        return self._mean([o.recall for o in self.outcomes if o.edited])

    # -- damage ----------------------------------------------------------

    @property
    def regression_trials(self) -> int:
        return sum(1 for o in self.outcomes if o.regressions is not None)

    @property
    def regressions(self) -> int | None:
        """Total regressions, or ``None`` if no trial measured them."""
        measured = [o.regressions for o in self.outcomes if o.regressions is not None]
        return sum(measured) if measured else None

    @property
    def tampered(self) -> int:
        return sum(1 for o in self.outcomes if o.tampered)

    @property
    def budget_paused(self) -> int:
        """Trials that ran out of turns or wall clock.

        Surfaced because it changes what the other numbers mean. On the first
        real run, three of eleven trials paused on the budget and every one of
        them did all of its editing in the final turn or two -- the eval was
        measuring how fast an agent reads, not how well it codes, and nothing
        in the report said so.
        """
        return sum(1 for o in self.outcomes if o.budget_paused)

    @property
    def refusals(self) -> int:
        return sum(1 for o in self.outcomes if o.refused)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.errored)

    def render(self) -> str:
        if not self.outcomes:
            return "no trials"
        regressions = (
            "not measured"
            if self.regressions is None
            else f"{self.regressions} (over {self.regression_trials}/{self.trials} trials)"
        )
        first_edit = (
            "not measured"
            if self.first_edit_trials == 0
            else (
                f"{self.mean_turns_to_first_edit:.1f}"
                f"  ({self.never_edited} never edited, "
                f"{self.first_edit_trials}/{self.trials} measured)"
            )
        )
        return "\n".join(
            [
                f"trials              : {self.trials}",
                f"pass rate           : {self.pass_rate:.1%}",
                f"tokens / task       : {self.tokens_per_task:,.0f}"
                f"  (over {self.token_trials}/{self.trials} trials that ran)",
                f"turns to first edit : {first_edit}",
                f"diff precision      : {self.mean_precision:.1%}"
                f"  (over {self.precision_trials}/{self.trials} trials that edited)",
                f"diff recall         : {self.mean_recall:.1%}",
                f"regressions         : {regressions}",
                f"tampered            : {self.tampered}",
                f"budget paused       : {self.budget_paused}",
                f"refusals            : {self.refusals}",
                f"errors              : {self.errors}",
            ]
        )
