"""Diff as the artifact (S-202).

What a repo-mode run *produced* is a diff, not a transcript. This module reads
the shadow refs S-201 wrote and turns them into something a human can look at:
a stat line, a file list, a patch, and — with the checkpoints — a way back to
any turn.

Everything here reads. ``diff`` never touches the work tree; ``rewind`` is the
one operation that writes, and it is never automatic. That asymmetry is
deliberate: a tool that silently reverts an agent's work the moment you ask to
*look* at it would be worse than no tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.repo import (
    BASELINE_REF_SUFFIX,
    CHECKPOINT_COMMAND_TIMEOUT,
    shadow_root,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    work_tree_argument,
)

__all__ = [
    "DiffStat",
    "FileChange",
    "RewindResolution",
    "ShadowReader",
    "render_report",
    "render_pr_body",
]


@dataclass(frozen=True)
class FileChange:
    """One file's change between two refs."""

    path: str
    added: int
    removed: int


@dataclass(frozen=True)
class DiffStat:
    """Aggregate change between the baseline and a checkpoint."""

    files: tuple[FileChange, ...] = ()

    @property
    def files_changed(self) -> int:
        return len(self.files)

    @property
    def added(self) -> int:
        return sum(f.added for f in self.files)

    @property
    def removed(self) -> int:
        return sum(f.removed for f in self.files)

    @property
    def empty(self) -> bool:
        return not self.files

    def summary(self) -> str:
        """``3 files changed, +40/-12`` — the line a report leads with."""
        if self.empty:
            return "no files changed"
        plural = "" if self.files_changed == 1 else "s"
        return (
            f"{self.files_changed} file{plural} changed, "
            f"+{self.added}/-{self.removed}"
        )


@dataclass(frozen=True)
class RewindResolution:
    """Which checkpoint a requested turn actually resolved to.

    Turns without tool calls write no checkpoint, so a requested turn may not
    exist. Rather than fail — the snapshot for that turn is genuinely
    identical to the one before it — the request resolves to the nearest
    earlier checkpoint and says so. Silently returning a different turn than
    was asked for would be the worse option.
    """

    requested: int
    resolved_turn: int | None
    ref: str | None
    exact: bool

    @property
    def found(self) -> bool:
        return self.ref is not None

    def describe(self) -> str:
        if not self.found:
            return f"no checkpoint at or before turn {self.requested}"
        if self.exact:
            return f"turn {self.requested}"
        return (
            f"turn {self.resolved_turn} (turn {self.requested} changed "
            "nothing, so no checkpoint was written for it)"
        )


_STAT_LINE = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
_TURN_REF = re.compile(r"refs/harness/(?P<agent>[^/]+)/turn-(?P<turn>\d+)$")


class ShadowReader:
    """Reads S-201's shadow refs. Never mutates the work tree except in
    :meth:`rewind`, which the caller must ask for explicitly."""

    def __init__(self, sandbox, run_id: str, agent_id: str) -> None:
        self._sandbox = sandbox
        self._run_id = run_id
        self._agent_id = agent_id

    def _git(self, args: str) -> str:
        # No GIT_INDEX_FILE and no --work-tree for read commands: a read that
        # carried a work tree could touch the index, and `diff` promises not
        # to. rewind() adds them back deliberately.
        return (
            f"git --git-dir={shadow_root()}/{self._run_id} {args}"
        )

    async def _run(self, command: str) -> tuple[int, str]:
        try:
            result = await self._sandbox.exec(
                command, timeout=CHECKPOINT_COMMAND_TIMEOUT
            )
        except Exception:
            return 1, ""
        return (
            getattr(result, "exit_code", 1),
            (getattr(result, "stdout", "") or "").rstrip("\n"),
        )

    @property
    def baseline_ref(self) -> str:
        return f"refs/harness/{self._agent_id}/{BASELINE_REF_SUFFIX}"

    async def turns(self) -> list[int]:
        """Turn numbers that have a checkpoint, ascending."""
        # The format string is quoted because this command is run through a
        # shell: `%(refname)` unquoted is a syntax error in sh, and `_run`
        # turns any failure into `(1, "")`, which `turns()` reports as "no
        # checkpoints". The command was therefore broken in every real shell
        # while every test passed, because the tests exec against a fake
        # sandbox that never invokes one. TestAgainstARealShell exists to stop
        # that happening again.
        code, out = await self._run(
            self._git(
                "for-each-ref --format='%(refname)' "
                f"refs/harness/{self._agent_id}"
            )
        )
        if code != 0 or not out:
            return []
        found = []
        for line in out.splitlines():
            match = _TURN_REF.match(line.strip())
            if match:
                found.append(int(match.group("turn")))
        return sorted(found)

    async def resolve(self, turn: int) -> RewindResolution:
        """Resolve ``turn`` to the checkpoint that represents it."""
        available = await self.turns()
        if turn in available:
            return RewindResolution(
                requested=turn,
                resolved_turn=turn,
                ref=f"refs/harness/{self._agent_id}/turn-{turn}",
                exact=True,
            )
        earlier = [t for t in available if t < turn]
        if not earlier:
            return RewindResolution(turn, None, None, exact=False)
        nearest = max(earlier)
        return RewindResolution(
            requested=turn,
            resolved_turn=nearest,
            ref=f"refs/harness/{self._agent_id}/turn-{nearest}",
            exact=False,
        )

    async def stat(self, ref: str | None = None) -> DiffStat:
        """Change between the baseline and ``ref`` (default: the latest turn).

        Reads only. ``--numstat`` rather than ``--stat`` because the latter is
        formatted for humans and abbreviates paths.
        """
        target = ref
        if target is None:
            available = await self.turns()
            if not available:
                return DiffStat()
            target = f"refs/harness/{self._agent_id}/turn-{max(available)}"
        code, out = await self._run(
            self._git(f"diff --numstat {self.baseline_ref} {target}")
        )
        if code != 0 or not out:
            return DiffStat()
        changes = []
        for line in out.splitlines():
            match = _STAT_LINE.match(line.strip())
            if not match:
                continue
            added, removed, path = match.groups()
            changes.append(
                FileChange(
                    path=path,
                    # "-" marks a binary file: real change, uncountable lines.
                    added=0 if added == "-" else int(added),
                    removed=0 if removed == "-" else int(removed),
                )
            )
        return DiffStat(files=tuple(changes))

    async def patch(self, ref: str | None = None) -> str:
        """The unified diff between the baseline and ``ref``. Reads only."""
        target = ref
        if target is None:
            available = await self.turns()
            if not available:
                return ""
            target = f"refs/harness/{self._agent_id}/turn-{max(available)}"
        code, out = await self._run(
            self._git(f"diff {self.baseline_ref} {target}")
        )
        return out if code == 0 else ""

    async def rewind(self, turn: int) -> RewindResolution:
        """Restore the work tree to ``turn``. **Writes.** Never automatic.

        The only mutating operation in this module, and it exists solely
        because a caller asked for it by name. It restores through the shadow
        index so the workspace's own git state (if any) is untouched.
        """
        resolution = await self.resolve(turn)
        if not resolution.found:
            return resolution
        index = f"{shadow_root()}/{self._run_id}/index-{self._agent_id}"
        shadow = (
            f"GIT_INDEX_FILE={index} "
            f"git --git-dir={shadow_root()}/{self._run_id} "
            f"{work_tree_argument()}"
        )
        # read-tree loads the snapshot into this agent's private index;
        # checkout-index then materialises it into the work tree. Two steps
        # rather than `git checkout`, which would want a branch and would
        # write to the shadow HEAD.
        code, _ = await self._run(
            f"{shadow} read-tree {resolution.ref} && {shadow} checkout-index -a -f"
        )
        if code != 0:
            return RewindResolution(
                requested=turn,
                resolved_turn=resolution.resolved_turn,
                ref=None,
                exact=resolution.exact,
            )
        return resolution


def render_report(stat: DiffStat, *, limit: int = 20) -> str:
    """The final report, leading with the stat line then the file list."""
    lines = [stat.summary()]
    for change in stat.files[:limit]:
        lines.append(f"  {change.path}  +{change.added}/-{change.removed}")
    remaining = stat.files_changed - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return "\n".join(lines)


def render_pr_body(goal: str, stat: DiffStat, evidence: list[str]) -> str:
    """Assemble a PR body from the goal, the diff stat, and ledger evidence.

    Evidence comes from the task ledger, which the harness already populates —
    this composes what exists rather than asking the model for a summary it
    would have to invent.
    """
    lines = [goal.strip(), "", f"**{stat.summary()}**", ""]
    if evidence:
        lines.append("## Evidence")
        lines.extend(f"- {item}" for item in evidence)
        lines.append("")
    if not stat.empty:
        lines.append("## Files")
        lines.extend(
            f"- `{c.path}` +{c.added}/-{c.removed}" for c in stat.files[:50]
        )
    return "\n".join(lines).rstrip() + "\n"
