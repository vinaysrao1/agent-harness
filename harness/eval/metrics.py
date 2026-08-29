"""Metrics for a PR-replay run (S-401).

The contract names five: pass rate, tokens per task, turns to first edit, diff
precision, and regression count. Two of them are the ones that make this eval
worth having over a pass/fail number.

**Diff precision** asks whether the agent changed *the files the change
touched*, not merely enough files to go green. An agent that rewrites twelve
files to fix one has not done the task well even if the tests pass, and a
pass-rate-only metric cannot say so.

**Regression count** asks what else it broke. A change that makes its own
tests pass by breaking three others is a net loss that pass rate scores as a
win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TaskOutcome", "SuiteReport", "diff_precision"]


def diff_precision(
    touched: set[str], reference: set[str]
) -> tuple[float, float]:
    """``(precision, recall)`` of the agent's file set against the change's.

    Precision falls when the agent touched files the change did not; recall
    falls when it missed files the change did. Both are reported because they
    fail in opposite directions and a single blended number would hide which.
    Empty reference yields ``(0.0, 0.0)`` rather than a misleading 1.0.
    """
    if not reference:
        return 0.0, 0.0
    if not touched:
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
    turns_to_first_edit: int | None = None
    files_touched: frozenset[str] = frozenset()
    reference_files: frozenset[str] = frozenset()
    regressions: int = 0
    refused: bool = False
    errored: bool = False

    @property
    def precision(self) -> float:
        return diff_precision(set(self.files_touched), set(self.reference_files))[0]

    @property
    def recall(self) -> float:
        return diff_precision(set(self.files_touched), set(self.reference_files))[1]


@dataclass
class SuiteReport:
    """Aggregate over N tasks × K trials."""

    outcomes: list[TaskOutcome] = field(default_factory=list)

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return self._mean([1.0 if o.passed else 0.0 for o in self.outcomes])

    @property
    def tokens_per_task(self) -> float:
        return self._mean([float(o.tokens) for o in self.outcomes])

    @property
    def mean_turns_to_first_edit(self) -> float:
        # Only over trials that edited at all: averaging in "never edited" as
        # zero would make a paralysed agent look decisive.
        seen = [float(o.turns_to_first_edit) for o in self.outcomes
                if o.turns_to_first_edit is not None]
        return self._mean(seen)

    @property
    def never_edited(self) -> int:
        return sum(1 for o in self.outcomes if o.turns_to_first_edit is None)

    @property
    def mean_precision(self) -> float:
        return self._mean([o.precision for o in self.outcomes if o.files_touched])

    @property
    def mean_recall(self) -> float:
        return self._mean([o.recall for o in self.outcomes if o.files_touched])

    @property
    def regressions(self) -> int:
        return sum(o.regressions for o in self.outcomes)

    @property
    def refusals(self) -> int:
        return sum(1 for o in self.outcomes if o.refused)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.errored)

    def render(self) -> str:
        if not self.outcomes:
            return "no trials"
        return "\n".join(
            [
                f"trials              : {self.trials}",
                f"pass rate           : {self.pass_rate:.1%}",
                f"tokens / task       : {self.tokens_per_task:,.0f}",
                f"turns to first edit : {self.mean_turns_to_first_edit:.1f}"
                f"  ({self.never_edited} never edited)",
                f"diff precision      : {self.mean_precision:.1%}",
                f"diff recall         : {self.mean_recall:.1%}",
                f"regressions         : {self.regressions}",
                f"refusals            : {self.refusals}",
                f"errors              : {self.errors}",
            ]
        )
