"""Suite definition and the held-out split (S-401).

A suite is a repository, an ordered list of revisions, and the commands used to
grade them. Everything else is derived, so the same inputs always produce the
same suite -- no sampling, no clock, no network.

The held-out split exists because eval-gated promotion is meaningless without
one: a change tuned against the tasks it is then scored on will look like an
improvement whether or not it is. The split is by a hash of the task id, so it
is stable as the suite grows -- a task never migrates between halves when new
tasks are added, which a modulo-of-index split would allow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Suite",
    "render_command",
    "HELDOUT_FRACTION",
    "is_heldout",
    "heldout_score",
    "SuiteError",
]


class SuiteError(ValueError):
    """A suite file is missing something the runner needs."""


#: Fraction of tasks reserved. A third is small enough to leave a usable
#: development set and large enough that a held-out result is not one task.
HELDOUT_FRACTION = 1 / 3

#: Size of the hash space the first four digest bytes are drawn from, so the
#: score lands in ``[0, 1)``. ``0xFFFFFFFF`` (one less) would put the maximum
#: at exactly 1.0 and leave that one id outside a ``fraction=1.0`` set that
#: should contain everything. That is a 2^-32 event and will never be seen; it
#: is written correctly anyway, and `heldout_score` exists so the choice is
#: pinned by a test rather than asserted in a comment.
_HASH_SPACE = 1 << 32


def heldout_score(task_id: str) -> float:
    """Where ``task_id`` sits in ``[0, 1)``. Stable across processes and runs.

    Exposed rather than inlined because it is the whole split: a change to it
    silently reassigns tasks between the half you tune on and the half you
    score on, which is the one thing the split exists to prevent.
    """
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / _HASH_SPACE


def is_heldout(task_id: str, fraction: float = HELDOUT_FRACTION) -> bool:
    """Whether ``task_id`` belongs to the held-out set.

    Hash-based, not index-based: an index split reshuffles every task when the
    suite grows, which would silently move tasks you had already tuned against
    into the held-out half and destroy the guarantee it exists to provide.
    """
    return heldout_score(task_id) < fraction


#: Placeholder in a suite's commands, replaced with the task's own test paths.
TESTS_PLACEHOLDER = "{tests}"


def render_command(command: str, test_paths: tuple[str, ...]) -> str:
    """Substitute ``{tests}`` in ``command`` with ``test_paths``.

    The grader is *the change's tests*, not the repository's whole suite. A
    fixed command has to name one or the other, and naming the whole suite
    makes every task's verdict hostage to any unrelated failure in the
    repository -- one flaky test and the entire benchmark reads 0%.

    A command without the placeholder is returned unchanged, so a suite that
    genuinely wants to run everything simply does not use it.
    """
    if TESTS_PLACEHOLDER not in command:
        return command
    if not test_paths:
        raise SuiteError(
            f"command uses {TESTS_PLACEHOLDER} but the task has no test paths: "
            f"{command!r}"
        )
    return command.replace(TESTS_PLACEHOLDER, " ".join(test_paths))


@dataclass(frozen=True)
class Suite:
    """A named, reproducible set of replay tasks."""

    name: str
    repo: str
    revs: tuple[str, ...]
    #: Command that grades the task's own tests. Must fail at the starting
    #: state and pass with the reference answer applied, or
    #: :func:`~harness.eval.pr_replay.validate` rejects the task.
    #:
    #: ``{tests}`` is replaced with the task's test paths -- see
    #: :func:`render_command`. Without it the command must name what to run
    #: itself, which in practice means the whole suite.
    test_command: str
    #: Optional command running the *rest* of the suite, for regression
    #: counting. Absent means regressions are reported as **not measured**
    #: rather than as zero.
    regression_command: str | None = None

    @classmethod
    def load(cls, path: Path) -> "Suite":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SuiteError(f"could not read suite {path}: {exc}") from exc
        missing = [k for k in ("name", "repo", "revs", "test_command") if k not in data]
        if missing:
            raise SuiteError(f"suite {path} is missing: {', '.join(missing)}")
        if not isinstance(data["revs"], list) or not data["revs"]:
            raise SuiteError(f"suite {path} lists no revisions")
        return cls(
            name=data["name"],
            repo=data["repo"],
            revs=tuple(data["revs"]),
            test_command=data["test_command"],
            regression_command=data.get("regression_command"),
        )

    def save(self, path: Path) -> None:
        payload = {
            "name": self.name,
            "repo": self.repo,
            "revs": list(self.revs),
            "test_command": self.test_command,
        }
        if self.regression_command is not None:
            payload["regression_command"] = self.regression_command
        Path(path).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def split(self, task_ids: list[str]) -> tuple[list[str], list[str]]:
        """``(development, heldout)`` for ``task_ids``."""
        dev = [t for t in task_ids if not is_heldout(t)]
        held = [t for t in task_ids if is_heldout(t)]
        return dev, held
