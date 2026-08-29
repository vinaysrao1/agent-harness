"""Suite definition and the held-out split (S-401).

A suite is a repository plus an ordered list of revisions. Everything else is
derived, so the same inputs always produce the same suite -- no sampling, no
clock, no network.

The held-out split exists because the plan's eval-gated promotion (§10.3 B6)
is meaningless without one: a change tuned against the tasks it is then scored
on will look like an improvement whether or not it is. The split is by a hash
of the task id, so it is stable as the suite grows -- a task never migrates
between halves when new tasks are added, which a modulo-of-index split would
allow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Suite", "HELDOUT_FRACTION", "is_heldout"]

#: Fraction of tasks reserved. A third is small enough to leave a usable
#: development set and large enough that a held-out result is not one task.
HELDOUT_FRACTION = 1 / 3


def is_heldout(task_id: str, fraction: float = HELDOUT_FRACTION) -> bool:
    """Whether ``task_id`` belongs to the held-out set.

    Hash-based, not index-based: an index split reshuffles every task when the
    suite grows, which would silently move tasks you had already tuned against
    into the held-out half and destroy the guarantee it exists to provide.
    """
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) < fraction


@dataclass(frozen=True)
class Suite:
    """A named, reproducible set of replay tasks."""

    name: str
    repo: str
    revs: tuple[str, ...]
    test_command: str

    @classmethod
    def load(cls, path: Path) -> "Suite":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            repo=data["repo"],
            revs=tuple(data["revs"]),
            test_command=data["test_command"],
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "repo": self.repo,
                    "revs": list(self.revs),
                    "test_command": self.test_command,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def split(self, task_ids: list[str]) -> tuple[list[str], list[str]]:
        """``(development, heldout)`` for ``task_ids``."""
        dev = [t for t in task_ids if not is_heldout(t)]
        held = [t for t in task_ids if is_heldout(t)]
        return dev, held
