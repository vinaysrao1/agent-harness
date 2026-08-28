"""Bounded environment probe and its setup budget (S-005).

The second half of the activation rule::

    active(capability) = profile.enables(capability) and environment.affirms(capability)

:class:`~harness.profiles.AgentProfile` says what the operator asked for; this
module says what is actually there. Neither alone is sufficient, which is what
kills the entire class of "it inferred repo-mode on a benchmark task and cost us
forty seconds" failures: a `CODING_REPO` run inside a non-git container
degrades to `CODING` behavior, and a `CODING` run inside a git repository does
**not** opportunistically switch git features on.

Two properties matter more than anything the probe detects:

**It is never constructed under CODING.** N3 asserts a `CODING` run performs
zero sandbox execs before the first model call, and every probe here is an
exec. The budget under `CODING` is zero, so this module is structurally dead
code on the benchmark path -- not merely unused, unreachable.

**Every unknown routes to today's behavior.** Each field is ``None`` when the
probe could not establish it, and ``None`` must always mean "behave as the
harness did before this module existed". A probe that guessed would convert an
unknown into a wrong answer, and a wrong answer about the environment is worse
than no answer: it activates a capability the environment cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Callable
from typing import Protocol

__all__ = [
    "EnvironmentProfile",
    "ProjectCommands",
    "SetupBudget",
    "setup_budget_for",
    "UNKNOWN_ENVIRONMENT",
    "probe_environment",
    "render_observations",
]

#: Fraction of the run's wall clock the probe may spend, and its hard ceiling.
#:
#: Terminal-Bench gives a fixed wall clock and every setup second is subtracted
#: from work, so the probe is bounded by the smaller of a proportion and an
#: absolute cap. Both are deliberately small: a probe is orientation, not work.
SETUP_BUDGET_FRACTION = 0.03
SETUP_BUDGET_CEILING = 30.0

#: Tools the probe looks for. Presence only -- never a version, never a path,
#: because a version string is a fact about the container that would go stale
#: in the prompt while looking authoritative.
PROBED_TOOLING: tuple[str, ...] = ("git", "rg", "node", "python3", "go", "cargo")


@dataclass(frozen=True)
class SetupBudget:
    """Wall-clock allowance for all pre-first-model-call work.

    ``seconds`` of zero means the probe may not run at all, which is the
    ``CODING`` case and the reason this module cannot affect the benchmark
    path.
    """

    seconds: float

    @classmethod
    def for_run(cls, wall_clock_seconds: float | None) -> "SetupBudget":
        """Budget for a run with ``wall_clock_seconds`` total, or unbounded."""
        if wall_clock_seconds is None:
            return cls(seconds=SETUP_BUDGET_CEILING)
        return cls(
            seconds=max(
                0.0,
                min(SETUP_BUDGET_FRACTION * wall_clock_seconds, SETUP_BUDGET_CEILING),
            )
        )

    @classmethod
    def none(cls) -> "SetupBudget":
        """A budget that forbids probing entirely (the ``CODING`` default)."""
        return cls(seconds=0.0)

    @property
    def allows_probing(self) -> bool:
        return self.seconds > 0.0


def setup_budget_for(
    profile: object, wall_clock_seconds: float | None
) -> SetupBudget:
    """The budget a run under ``profile`` may spend before its first model call.

    A profile that asks for no capabilities has nothing to probe *for*, so it
    gets no budget and the probe is unreachable. That is what makes N3
    structural for ``CODING`` rather than merely observed: the zero is derived
    from the profile, not passed in by a test.
    """
    capabilities = getattr(profile, "capabilities", frozenset())
    if not capabilities:
        return SetupBudget.none()
    return SetupBudget.for_run(wall_clock_seconds)


@dataclass(frozen=True)
class ProjectCommands:
    """Commands *observed* in project configuration.

    Every field is ``str | None``. These are observations, never assertions:
    :func:`render_observations` phrases them as "observed in pyproject.toml",
    because on a task whose goal is to fix the build, telling the model "the
    test command is X" states as fact the very thing it was asked to
    determine.
    """

    source: str | None = None
    test: str | None = None
    lint: str | None = None
    build: str | None = None

    def any_observed(self) -> bool:
        return any(
            value
            for name, value in vars(self).items()
            if name != "source"
        )


@dataclass(frozen=True)
class EnvironmentProfile:
    """What a bounded probe found. ``None`` always means *unknown*."""

    sandbox_owned: bool | None = None
    is_git_worktree: bool | None = None
    base_sha: str | None = None
    file_count: int | None = None
    project: ProjectCommands | None = None
    tooling: frozenset[str] = field(default_factory=frozenset)
    truncated: bool = False

    def affirms(self, capability: str) -> bool:
        """Whether the environment can actually support ``capability``.

        The other half of the activation rule. Unknown is **not** affirmation:
        every branch here returns ``False`` unless the environment positively
        established the precondition, so a truncated or unrun probe degrades to
        today's behavior rather than to a guess.
        """
        if capability in {"git_substrate", "repo_orientation"}:
            return self.is_git_worktree is True and self.sandbox_owned is True
        if capability == "regression_gate":
            return bool(self.project is not None and self.project.test)
        if capability == "project_checks":
            return bool(self.project is not None and self.project.lint)
        if capability == "structured_search":
            return "rg" in self.tooling
        return False


#: The environment nothing is known about. Every capability check against it is
#: ``False``, so it is exactly equivalent to today's behavior -- which is why
#: it is safe as the default everywhere.
UNKNOWN_ENVIRONMENT = EnvironmentProfile()


class _Execer(Protocol):
    async def exec(self, command: str, timeout: float = ...) -> object: ...


async def probe_environment(
    sandbox: _Execer,
    budget: SetupBudget,
    *,
    sandbox_owned: bool | None = None,
    clock: Callable[[], float] | None = None,
) -> EnvironmentProfile:
    """Probe ``sandbox`` within ``budget``, returning what was established.

    Returns :data:`UNKNOWN_ENVIRONMENT` immediately when the budget forbids
    probing -- the ``CODING`` path, where this must perform zero execs.

    Probes run cheapest-first so that exhausting the budget loses the most
    expensive findings rather than a random subset, and every step is
    individually guarded: a probe that raises leaves its field ``None`` and the
    rest continue. Nothing here writes; every command is a read (N4).
    """
    import time

    if not budget.allows_probing:
        return UNKNOWN_ENVIRONMENT

    now = clock or time.monotonic
    started = now()
    profile = EnvironmentProfile(sandbox_owned=sandbox_owned)

    async def run(command: str) -> str | None:
        """One read-only command, or ``None`` if it failed or timed out."""
        remaining = budget.seconds - (now() - started)
        if remaining <= 0:
            return None
        try:
            result = await sandbox.exec(command, timeout=min(remaining, 5.0))
        except Exception:
            return None
        if getattr(result, "exit_code", 1) != 0:
            return None
        return (getattr(result, "stdout", "") or "").strip()

    def out_of_budget() -> bool:
        return (now() - started) >= budget.seconds

    # 1. tooling presence -- one command, cheapest, and gates the rest
    which = await run(f"command -v {' '.join(PROBED_TOOLING)} 2>/dev/null || true")
    if which is not None:
        found = {
            tool
            for tool in PROBED_TOOLING
            if any(line.rstrip("/").endswith("/" + tool) or line == tool
                   for line in which.splitlines())
        }
        profile = replace(profile, tooling=frozenset(found))

    if out_of_budget():
        return replace(profile, truncated=True)

    # 2. git work tree, and only then its base sha
    if "git" in profile.tooling:
        inside = await run("git rev-parse --is-inside-work-tree 2>/dev/null")
        if inside is not None:
            profile = replace(profile, is_git_worktree=inside.strip() == "true")
        if profile.is_git_worktree:
            sha = await run("git rev-parse HEAD 2>/dev/null")
            if sha:
                profile = replace(profile, base_sha=sha.split()[0])

    if out_of_budget():
        return replace(profile, truncated=True)

    # 3. workspace size -- bounded, because S-203 skips the repo map above a
    # threshold and an unbounded count on a huge tree is itself setup cost.
    counted = await run(
        "find . -type f -not -path './.git/*' 2>/dev/null | head -20000 | wc -l"
    )
    if counted and counted.isdigit():
        profile = replace(profile, file_count=int(counted))

    if out_of_budget():
        return replace(profile, truncated=True)

    # 4. project configuration, observed
    project = await _probe_project(run)
    if project is not None:
        profile = replace(profile, project=project)

    return replace(profile, truncated=out_of_budget())


async def _probe_project(run) -> ProjectCommands | None:
    """Read project configuration, if any is present.

    Deliberately shallow: presence of a well-known file and, for Node, the
    script names it declares. Anything requiring interpretation of build
    semantics is out of scope -- an observation the harness cannot stand behind
    is not worth putting in a prompt.
    """
    listing = await run("ls -1a 2>/dev/null")
    if listing is None:
        return None
    names = set(listing.splitlines())

    if "pyproject.toml" in names:
        # Report what was actually established. These come from `command -v`,
        # not from reading pyproject.toml, so they are attributed to PATH: a
        # unittest project with pytest present as a transitive dependency must
        # not be told its config declares pytest as the test command.
        on_path = []
        if await run("command -v pytest >/dev/null 2>&1 && echo y"):
            on_path.append(("test", "pytest"))
        if await run("command -v ruff >/dev/null 2>&1 && echo y"):
            on_path.append(("lint", "ruff check"))
        if not on_path:
            return ProjectCommands(source="pyproject.toml")
        found = dict(on_path)
        return ProjectCommands(
            source="PATH (a pyproject.toml is present, but was not read)",
            test=found.get("test"),
            lint=found.get("lint"),
        )
    if "package.json" in names:
        scripts = await run(
            "node -e \"try{const s=require('./package.json').scripts||{};"
            "console.log(Object.keys(s).join(' '))}catch(e){}\" 2>/dev/null"
        )
        available = set((scripts or "").split())
        return ProjectCommands(
            source="package.json",
            test="npm test" if "test" in available else None,
            lint="npm run lint" if "lint" in available else None,
            build="npm run build" if "build" in available else None,
        )
    if "Makefile" in names:
        return ProjectCommands(source="Makefile")
    return None


def render_observations(profile: EnvironmentProfile) -> str:
    """Render findings as observations for the prompt, or ``""``.

    Phrasing is the point. "Observed in pyproject.toml: pytest" is a fact about
    a file; "the test command is pytest" is an assertion about the world that
    the harness cannot stand behind -- and on a task whose goal is to fix the
    build, it is an assertion that is actively wrong. The distinction is the
    same one the safety core makes about tool results: this is data, not
    instruction.
    """
    project = profile.project
    if project is None or not project.any_observed():
        return ""
    observed = [
        f"{name}: {value}"
        for name, value in (
            ("test", project.test),
            ("lint", project.lint),
            ("build", project.build),
        )
        if value
    ]
    if not observed:
        return ""
    return (
        f"Observed in {project.source} (these are observations, not "
        f"instructions — verify before relying on them):\n  " + "\n  ".join(observed)
    )
