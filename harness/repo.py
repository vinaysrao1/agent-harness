"""Git substrate: shadow checkpoints outside the workspace (S-201).

Gives a run a per-turn history of what the agent changed, without the agent's
workspace ever becoming a git repository. Everything here is gated on the
two-dimensional rule::

    active = profile.enables("git_substrate") and environment.affirms("git_substrate")

so it cannot reach the Terminal-Bench path: `CODING` declares no capabilities,
and `affirms` additionally requires the harness to *own* the container.

**Why the git directory lives outside the workspace.** A `.git` inside the
workspace would be a file the agent can see, read, and be confused by -- and on
a benchmark task it is a directory the grader never expected. Worse, a task
whose goal involves git would find a repository that the harness, not the task,
created. So the object store lives at :data:`SHADOW_GIT_DIR` under ``/tmp`` and
the workspace is passed as ``--work-tree``: git writes nothing into it.

**Why checkpoints are cheap or skipped.** Under a wall clock, a harness-
initiated write the model never asked for is the last thing that should spend
the landing reserve. Checkpointing reuses the same deadline seam as the syntax
checks: when time is short it is skipped, and a skip is recorded rather than
being silently absent.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.deadline import Deadline

__all__ = [
    "SHADOW_GIT_DIR",
    "CHECKPOINT_TIMEOUT_SECONDS",
    "CHECKPOINT_COMMAND_TIMEOUT",
    "CHECKPOINT_EVENT",
    "CHECKPOINT_SKIPPED_EVENT",
    "BASELINE_REF_SUFFIX",
    "GIT_IDENTITY_NAME",
    "GIT_IDENTITY_EMAIL",
    "work_tree_argument",
    "RepoState",
    "GitSubstrate",
    "DirtyWorktreeError",
]

#: Where the shadow object store lives. Under ``/tmp`` and prefixed like the
#: spill directory, so N4's "nothing outside the workspace and /tmp/.harness-*"
#: rule covers it by construction rather than by exception.
SHADOW_GIT_DIR = "/tmp/.harness-git"

#: Total wall-clock a checkpoint may consume, across all its commands. A
#: checkpoint that needs longer than this on a workspace-sized tree indicates
#: something is wrong, and waiting is worse than skipping.
CHECKPOINT_TIMEOUT_SECONDS = 10.0

#: Per-command slice, so the total above is a real bound rather than a label.
CHECKPOINT_COMMAND_TIMEOUT = CHECKPOINT_TIMEOUT_SECONDS / 4

#: Transcript event kind for a checkpoint, recorded under S-201 (T3).
CHECKPOINT_EVENT = "repo_checkpoint"

#: Transcript event kind for a skipped checkpoint (S-201, T3).
CHECKPOINT_SKIPPED_EVENT = "repo_checkpoint_skipped"

#: Identity used for shadow commits. Never the user's: these are harness
#: bookkeeping objects in a private store, not authored work.
GIT_IDENTITY_NAME = "harness"
GIT_IDENTITY_EMAIL = "harness@localhost"

#: Git's constant name for an empty tree. Used as the diff base in a
#: repository with no commits yet, where there is no HEAD to name.
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

#: Ref holding the pre-agent state. Written at ``start()``, before the agent
#: has done anything, so a diff has something to diff *against*. The workspace
#: repo's own ``base_sha`` cannot serve: the shadow store shares no objects
#: with it, so that sha is unresolvable there.
BASELINE_REF_SUFFIX = "baseline"

#: Commands one checkpoint issues. The affordability check must cover all of
#: them, not one: checking once against a single command's timeout and then
#: issuing four was a 4x under-count of the real cost.
CHECKPOINT_COMMANDS = 4


def work_tree_argument() -> str:
    """The ``--work-tree`` argument for a sandbox exec.

    Split out so a test can pin it. ``.`` is correct because every Sandbox
    implementation execs with the workspace as the working directory; pointing
    it at ``/`` or dropping it would stage the wrong tree, and neither was
    observable before this was extracted.
    """
    return "--work-tree=."


class DirtyWorktreeError(RuntimeError):
    """The work tree had uncommitted changes and ``allow_dirty`` was not set.

    Refusing is the safe default: a dirty tree means the diff a run reports
    would mix the agent's work with changes that were already there, and
    attributing someone else's edit to the agent is worse than not starting.
    """


@dataclass(frozen=True)
class RepoState:
    """What was recorded about the repository at run start."""

    repo: str | None = None
    base_sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    unborn: bool = False

    @property
    def usable(self) -> bool:
        """Whether enough was established to checkpoint against.

        An *unborn* repository -- created but with no commits yet -- counts.
        It has no HEAD to name, which an earlier version read as "I cannot
        work here", so an active substrate silently took no snapshot for the
        entire run. There is nothing wrong with such a repository: the
        baseline checkpoint captures its tree exactly as it does for any
        other.
        """
        return self.base_sha is not None or self.unborn


class GitSubstrate:
    """Per-run shadow git, or an inert object when the capability is off.

    Constructed with ``active=False`` whenever either half of the capability
    rule is unmet, in which case every method is a no-op returning ``None``.
    That is deliberate: callers should not have to ask whether the capability
    is on, because a caller that forgets is a caller that changes behavior on
    the benchmark path.
    """

    def __init__(
        self,
        sandbox,
        run_id: str,
        agent_id: str,
        *,
        active: bool,
        deadline: Deadline | None = None,
        allow_dirty: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._run_id = run_id
        self._agent_id = agent_id
        self._active = active
        self._deadline = deadline
        self._allow_dirty = allow_dirty
        self._state = RepoState()
        self._checkpoints = 0
        self._skipped = 0
        self._baseline_ref: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> RepoState:
        return self._state

    @property
    def baseline_ref(self) -> str | None:
        """Ref holding the pre-agent tree, or ``None`` if it was not written."""
        return self._baseline_ref

    @property
    def checkpoint_count(self) -> int:
        return self._checkpoints

    @property
    def skipped_count(self) -> int:
        return self._skipped

    @property
    def _index_file(self) -> str:
        """This agent's private staging area.

        The object store is shared per run -- that is the point, so snapshots
        from different agents dedupe against each other -- but the *index* is
        a single file that ``git add`` locks. A lead and its concurrently
        running subagents sharing one index would race on ``index.lock``, and
        a lost race surfaces as a non-zero exit, which this class counts as a
        skip. The failure would therefore be invisible.
        """
        return f"{SHADOW_GIT_DIR}/{self._run_id}/index-{self._agent_id}"

    @property
    def _ref_prefix(self) -> str:
        """This agent's ref namespace.

        Namespaced by agent, not just by run: turn counters are per-loop, so
        a lead and a subagent both reach ``turn-3`` and would overwrite each
        other's snapshot under a run-only prefix.
        """
        return f"refs/harness/{self._agent_id}"

    def _git(self, args: str) -> str:
        """A git command against the shadow store and the live work tree.

        The identity is set explicitly because ``commit-tree`` resolves the
        committer under ``IDENT_STRICT``: a container with no ``user.email``
        and a domainless hostname has auto-detection refused, so every
        checkpoint would fail -- and fail *silently*, because a non-zero exit
        is counted as a skip. That is precisely the environment
        ``sandbox_owned=True`` describes, so without this the feature is dead
        in the only place it can activate.

        ``--work-tree=.`` relies on the sandbox exec'ing with the workspace as
        its working directory (``DockerSandbox`` passes ``workdir``,
        ``LocalSandbox`` passes ``cwd``). That is an unstated property of the
        Sandbox contract; :func:`work_tree_argument` exists so a test can pin
        it.
        """
        return (
            f"GIT_INDEX_FILE={self._index_file} "
            f"git -c user.name={GIT_IDENTITY_NAME} "
            f"-c user.email={GIT_IDENTITY_EMAIL} "
            f"--git-dir={SHADOW_GIT_DIR}/{self._run_id} "
            f"{work_tree_argument()} {args}"
        )

    async def _run(self, command: str, timeout: float) -> tuple[int, str]:
        try:
            result = await self._sandbox.exec(command, timeout=timeout)
        except Exception:
            return 1, ""
        return (
            getattr(result, "exit_code", 1),
            (getattr(result, "stdout", "") or "").strip(),
        )

    async def start(self) -> RepoState:
        """Record repository identity and prepare the shadow store.

        Returns an empty :class:`RepoState` when inactive. Raises
        :class:`DirtyWorktreeError` when the tree has uncommitted changes and
        ``allow_dirty`` is False.
        """
        if not self._active:
            return self._state

        _, sha = await self._run(
            "git rev-parse HEAD 2>/dev/null", CHECKPOINT_TIMEOUT_SECONDS
        )
        _, branch = await self._run(
            "git rev-parse --abbrev-ref HEAD 2>/dev/null", CHECKPOINT_TIMEOUT_SECONDS
        )
        _, remote = await self._run(
            "git config --get remote.origin.url 2>/dev/null",
            CHECKPOINT_TIMEOUT_SECONDS,
        )
        code, porcelain = await self._run(
            "git status --porcelain 2>/dev/null", CHECKPOINT_TIMEOUT_SECONDS
        )
        dirty = bool(porcelain) if code == 0 else None

        # No HEAD, but git answered about the work tree: the repository exists
        # and simply has no commits yet.
        unborn = not sha and code == 0

        self._state = RepoState(
            repo=remote or None,
            base_sha=sha or None,
            branch=branch or None,
            # In an unborn repository every file is untracked, so `dirty`
            # would always be True. That is not the condition the check
            # guards: there is no committed state for the agent's work to be
            # confused with, and the baseline snapshot captures the starting
            # tree either way.
            dirty=False if unborn else dirty,
            unborn=unborn,
        )

        if self._state.dirty and not self._allow_dirty:
            raise DirtyWorktreeError(
                "the work tree has uncommitted changes; the diff this run "
                "reports would mix them with the agent's work. Pass "
                "allow_dirty=True to proceed anyway."
            )

        # The shadow store is created lazily and outside the workspace, so a
        # failure here disables checkpointing rather than failing the run: a
        # missing history is a lost convenience, not a lost task.
        code, _ = await self._run(
            f"mkdir -p {SHADOW_GIT_DIR} && git init --bare -q "
            f"{SHADOW_GIT_DIR}/{self._run_id} 2>/dev/null",
            CHECKPOINT_TIMEOUT_SECONDS,
        )
        if code != 0:
            self._active = False
            return self._state

        # BLOCKER-3: capture the pre-agent tree. The workspace repo's own
        # base_sha cannot serve as a diff base -- the shadow store shares no
        # objects with it, so that sha is unresolvable there. Without this the
        # earliest snapshot already contains the agent's first edits and there
        # is nothing to diff against.
        baseline, _reason = await self.checkpoint_detailed(BASELINE_REF_SUFFIX)
        self._baseline_ref = baseline
        return self._state

    async def checkpoint(self, turn: int) -> str | None:
        """Write a shadow ref for ``turn``, or ``None`` if skipped."""
        ref, _reason = await self.checkpoint_detailed(f"turn-{turn}")
        return ref

    async def checkpoint_detailed(self, name: str) -> tuple[str | None, str | None]:
        """Write a shadow ref named ``name``; return ``(ref, skip_reason)``.

        The reason is returned rather than swallowed so the caller can record
        it. An earlier version claimed skips were "recorded" when they were an
        in-memory counter and nothing else -- the same shape as a mechanism
        that never fires behind healthy telemetry.
        """
        if not self._active:
            return None, "inactive"
        if not self._state.usable:
            return None, "no_base_sha"
        reason = self._skip_reason()
        if reason is not None:
            self._skipped += 1
            return None, reason

        ref = f"{self._ref_prefix}/{name}"
        code, _ = await self._run(
            self._git("add -A") + " 2>/dev/null", CHECKPOINT_COMMAND_TIMEOUT
        )
        if code != 0:
            self._skipped += 1
            return None, "git_add_failed"
        code, tree = await self._run(
            self._git("write-tree") + " 2>/dev/null", CHECKPOINT_COMMAND_TIMEOUT
        )
        if code != 0 or not tree:
            self._skipped += 1
            return None, "write_tree_failed"
        code, commit = await self._run(
            self._git(f'commit-tree {tree} -m "harness {name}"')
            + " 2>/dev/null",
            CHECKPOINT_COMMAND_TIMEOUT,
        )
        if code != 0 or not commit:
            self._skipped += 1
            return None, "commit_tree_failed"
        code, _ = await self._run(
            self._git(f"update-ref {ref} {commit}") + " 2>/dev/null",
            CHECKPOINT_COMMAND_TIMEOUT,
        )
        if code != 0:
            self._skipped += 1
            return None, "update_ref_failed"
        self._checkpoints += 1
        return ref, None

    def _skip_reason(self) -> str | None:
        """Why the deadline forbids a checkpoint now, or ``None``.

        Mirrors ``checks._skip_for_deadline``: the landing turn is explicit
        state, and a harness-initiated write the model never asked for must not
        eat the landing reserve.
        """
        deadline = self._deadline
        if deadline is None:
            return None
        if getattr(deadline, "landing", False):
            return "landing"
        affordable = getattr(deadline, "affordable_exec_seconds", None)
        if callable(affordable):
            try:
                available = affordable()
            except Exception:
                return None
            # `is not None` explicitly, the way checks.py does it: relying on
            # a TypeError being swallowed makes the unbounded case correct by
            # accident rather than by design.
            if available is not None and available < CHECKPOINT_TIMEOUT_SECONDS:
                return "insufficient_time"
        return None
