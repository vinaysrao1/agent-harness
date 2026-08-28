"""S-005: the environment probe, and the guarantees that keep it off the
benchmark path.

Two properties dominate. The probe must never run under `CODING` (N3 asserts
zero execs before the first model call, and every probe is an exec), and every
unknown must route to today's behavior. A probe that guessed would turn an
unknown into a wrong answer, and a wrong answer about the environment activates
a capability the environment cannot support.
"""

from __future__ import annotations

import dataclasses

import pytest

from harness.environment import (
    PROBED_TOOLING,
    SETUP_BUDGET_CEILING,
    SETUP_BUDGET_FRACTION,
    UNKNOWN_ENVIRONMENT,
    EnvironmentProfile,
    ProjectCommands,
    SetupBudget,
    probe_environment,
    setup_budget_for,
    render_observations,
)
from harness.profiles import CODING, CODING_REPO, REPO_CAPABILITIES


class _Result:
    def __init__(self, stdout: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.stderr = ""


class _RecordingSandbox:
    """Scripted exec, recording every command so N3/N4 can be asserted."""

    def __init__(self, responses: dict[str, _Result] | None = None) -> None:
        self.commands: list[str] = []
        self.responses = responses or {}

    async def exec(self, command: str, timeout: float = 120) -> _Result:
        self.commands.append(command)
        for fragment, result in self.responses.items():
            if fragment in command:
                return result
        return _Result(exit_code=1)


class TestNeverRunsUnderCoding:
    async def test_S005_zero_budget_performs_zero_execs(self) -> None:
        # The N3 guarantee at its source: under CODING the budget is zero, so
        # the probe is unreachable rather than merely unused.
        sandbox = _RecordingSandbox()
        profile = await probe_environment(sandbox, SetupBudget.none())
        assert sandbox.commands == []
        assert profile is UNKNOWN_ENVIRONMENT

    def test_S005_coding_asks_for_nothing_the_probe_could_affirm(self) -> None:
        # Even if a probe somehow ran, CODING enables no capability, so the
        # conjunction is False for every one of them.
        rich = EnvironmentProfile(
            sandbox_owned=True,
            is_git_worktree=True,
            project=ProjectCommands(source="pyproject.toml", test="pytest"),
            tooling=frozenset({"rg", "git"}),
        )
        for capability in REPO_CAPABILITIES:
            assert not (CODING.enables(capability) and rich.affirms(capability))


class TestUnknownRoutesToTodaysBehavior:
    @pytest.mark.parametrize(
        "field", [f.name for f in dataclasses.fields(EnvironmentProfile)]
    )
    def test_S005_every_field_defaults_to_unknown_or_empty(self, field: str) -> None:
        value = getattr(UNKNOWN_ENVIRONMENT, field)
        # Identity, not equality: `0 == False`, so an `in (None, False, ...)`
        # check silently accepted a field defaulting to 0.
        assert (
            value is None or value is False or value == frozenset()
        ), f"{field} defaults to {value!r}; an unknown must never be a claim"
        assert not isinstance(value, int) or isinstance(value, bool), (
            f"{field} defaults to a number; unknown must be None"
        )

    @pytest.mark.parametrize(
        "field",
        [
            f.name
            for f in dataclasses.fields(EnvironmentProfile)
            if f.name not in {"tooling", "truncated"}
        ],
    )
    def test_S005_clearing_any_field_routes_to_todays_behavior(
        self, field: str
    ) -> None:
        """Acceptance (3) proper: clearing a field must not *gain* anything.

        The previous version re-asserted the dataclass defaults, which tests
        nothing about routing. This takes a fully-affirming environment and
        blanks one field at a time, requiring that no capability becomes
        active that was not active before.
        """
        full = EnvironmentProfile(
            sandbox_owned=True,
            is_git_worktree=True,
            base_sha="abc",
            file_count=10,
            project=ProjectCommands(source="PATH", test="pytest", lint="ruff check"),
            tooling=frozenset({"rg", "git"}),
        )
        cleared = dataclasses.replace(full, **{field: None})
        for capability in REPO_CAPABILITIES:
            if cleared.affirms(capability):
                assert full.affirms(capability), (
                    f"clearing {field} ACTIVATED {capability}; unknown must "
                    "never gain a capability"
                )

    @pytest.mark.parametrize("capability", sorted(REPO_CAPABILITIES))
    def test_S005_unknown_environment_affirms_nothing(self, capability: str) -> None:
        assert not UNKNOWN_ENVIRONMENT.affirms(capability)

    @pytest.mark.parametrize("capability", sorted(REPO_CAPABILITIES))
    def test_S005_truncated_probe_affirms_nothing_new(self, capability: str) -> None:
        # A probe cut short must degrade, not guess.
        truncated = EnvironmentProfile(truncated=True)
        assert not truncated.affirms(capability)

    @pytest.mark.parametrize(
        "capability", ["nonexistent", "", "GIT_SUBSTRATE", "git substrate"]
    )
    def test_S005_unknown_capability_is_denied(self, capability: str) -> None:
        # The fallthrough carries the spec's central claim, and every other
        # test parametrizes over REPO_CAPABILITIES -- all of which have
        # explicit branches -- so the line was never asserted against. Flipping
        # it to `return True` left the whole suite green.
        rich = EnvironmentProfile(
            sandbox_owned=True,
            is_git_worktree=True,
            project=ProjectCommands(source="PATH", test="pytest", lint="ruff check"),
            tooling=frozenset({"rg", "git"}),
        )
        assert not rich.affirms(capability)

    @pytest.mark.parametrize("capability", sorted(REPO_CAPABILITIES))
    def test_S005_unknown_sandbox_ownership_is_not_affirmation(
        self, capability: str
    ) -> None:
        # C4: `None` is the default AND the realistic case -- sandbox_owned is
        # caller-supplied and the probe never determines it. Only the explicit
        # False case was covered, so relaxing `is True` to `is not False`
        # passed everything.
        unknown_owner = EnvironmentProfile(
            is_git_worktree=True,
            sandbox_owned=None,
            project=ProjectCommands(source="PATH", test="pytest", lint="ruff check"),
            tooling=frozenset({"rg"}),
        )
        if capability in {"git_substrate", "repo_orientation"}:
            assert not unknown_owner.affirms(capability), (
                "unknown ownership must not affirm a capability that writes "
                "into the container"
            )

    def test_S005_a_git_repo_alone_does_not_affirm_git_capabilities(self) -> None:
        # sandbox_owned=False is the Harbor case: the container belongs to the
        # caller, so writing git refs into it is not ours to do.
        borrowed = EnvironmentProfile(is_git_worktree=True, sandbox_owned=False)
        assert not borrowed.affirms("git_substrate")
        owned = EnvironmentProfile(is_git_worktree=True, sandbox_owned=True)
        assert owned.affirms("git_substrate")


class TestBothDimensionsRequired:
    def test_S005_profile_without_environment_is_inactive(self) -> None:
        for capability in REPO_CAPABILITIES:
            assert CODING_REPO.enables(capability)
            assert not (
                CODING_REPO.enables(capability)
                and UNKNOWN_ENVIRONMENT.affirms(capability)
            )

    def test_S005_environment_without_profile_is_inactive(self) -> None:
        # A CODING run inside a git repo must NOT opportunistically enable git
        # features. This is the failure class the two-dimensional rule exists
        # to kill.
        env = EnvironmentProfile(is_git_worktree=True, sandbox_owned=True)
        assert env.affirms("git_substrate")
        assert not (CODING.enables("git_substrate") and env.affirms("git_substrate"))

    def test_S005_both_together_activate(self) -> None:
        env = EnvironmentProfile(is_git_worktree=True, sandbox_owned=True)
        assert CODING_REPO.enables("git_substrate") and env.affirms("git_substrate")


class TestBudget:
    def test_S005_budget_is_the_smaller_of_fraction_and_ceiling(self) -> None:
        assert SetupBudget.for_run(900).seconds == pytest.approx(
            SETUP_BUDGET_FRACTION * 900
        )
        assert SetupBudget.for_run(12_000).seconds == SETUP_BUDGET_CEILING

    def test_S005_unbounded_run_still_capped(self) -> None:
        assert SetupBudget.for_run(None).seconds == SETUP_BUDGET_CEILING

    def test_S005_zero_budget_forbids_probing(self) -> None:
        assert not SetupBudget.none().allows_probing
        assert SetupBudget.for_run(900).allows_probing

    async def test_S005_exhausted_budget_marks_truncated(self) -> None:
        # A clock that jumps past the budget on its second reading: the probe
        # must stop and say so rather than run to completion.
        ticks = iter([0.0, 0.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0])
        last = [0.0]

        def clock() -> float:
            try:
                last[0] = next(ticks)
            except StopIteration:
                pass
            return last[0]

        sandbox = _RecordingSandbox({"command -v": _Result("/usr/bin/git")})
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=1.0), clock=clock
        )
        assert profile.truncated is True


class TestProbeIsReadOnly:
    async def test_S005_probe_never_writes(self) -> None:
        # N4 in miniature. Asserted by inspecting every command the probe
        # issues, because "it only reads" is exactly the kind of claim that
        # rots silently.
        sandbox = _RecordingSandbox(
            {
                "command -v": _Result("/usr/bin/git\n/usr/bin/rg"),
                "is-inside-work-tree": _Result("true"),
                "rev-parse HEAD": _Result("abc123"),
                "ls -1a": _Result("pyproject.toml\nsrc"),
            }
        )
        await probe_environment(sandbox, SetupBudget(seconds=30.0), sandbox_owned=True)
        assert sandbox.commands, "the probe issued no commands at all"
        import re

        # Discarding output is not a write, so redirections to /dev/null and
        # the 2>&1 duplication are permitted; a redirection to anything else is
        # not. Checked with a pattern rather than by deleting known-good
        # substrings, which left a bare `>` behind and made the test fail for
        # the wrong reason.
        redirect = re.compile(r"(\d?>>?)\s*(?!/dev/null\b)(?!&\d)(\S+)")
        mutators = ("tee ", "mkdir", "touch ", "rm ", "mv ", "cp ", "sed -i", "git init")
        for command in sandbox.commands:
            found = redirect.search(command)
            assert found is None, (
                f"probe redirects output to {found.group(2)!r}: {command!r}"
            )
            for token in mutators:
                assert token not in command, f"probe may write: {command!r}"

    async def test_S005_probe_finds_what_is_there(self) -> None:
        sandbox = _RecordingSandbox(
            {
                "command -v": _Result("/usr/bin/git\n/usr/bin/rg"),
                "is-inside-work-tree": _Result("true"),
                "rev-parse HEAD": _Result("abc123def"),
                "ls -1a": _Result("pyproject.toml"),
            }
        )
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=30.0), sandbox_owned=True
        )
        assert profile.is_git_worktree is True
        assert profile.base_sha == "abc123def"
        assert "git" in profile.tooling and "rg" in profile.tooling
        assert profile.affirms("structured_search")

    async def test_S005_a_failing_probe_leaves_the_field_unknown(self) -> None:
        # Every step is individually guarded: one failure must not abort the
        # rest, and must not invent a value.
        sandbox = _RecordingSandbox({"command -v": _Result("/usr/bin/git")})
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=30.0), sandbox_owned=True
        )
        assert profile.is_git_worktree is None or profile.is_git_worktree is False
        assert profile.base_sha is None
        assert profile.project is None


class TestObservationNotInstruction:
    def test_S005_renders_as_an_observation(self) -> None:
        rendered = render_observations(
            EnvironmentProfile(
                project=ProjectCommands(source="pyproject.toml", test="pytest")
            )
        )
        assert "Observed in pyproject.toml" in rendered
        assert "observations, not" in rendered
        # The failure mode this guards: on a task whose goal is to fix the
        # build, asserting the test command as fact states the very thing the
        # agent was asked to determine.
        assert "the test command is" not in rendered.lower()

    def test_S005_renders_nothing_when_nothing_was_observed(self) -> None:
        assert render_observations(UNKNOWN_ENVIRONMENT) == ""
        assert render_observations(
            EnvironmentProfile(project=ProjectCommands(source="Makefile"))
        ) == ""

    def test_S005_tooling_is_presence_only(self) -> None:
        # No paths and no version strings: a version is a fact about the
        # container that goes stale in a prompt while looking authoritative.
        # A digit in a binary *name* (python3) is not a version.
        import re

        version_like = re.compile(r"\d+\.\d+")
        for tool in PROBED_TOOLING:
            assert "/" not in tool, f"{tool!r} is a path, not a name"
            assert not version_like.search(tool), f"{tool!r} pins a version"


class TestBudgetActuallyBounds:
    """C5: acceptance (2) said "cost <= SetupBudget" and nothing tested it."""

    async def test_S005_budget_is_derived_from_the_profile(self) -> None:
        # N3 becomes structural: the zero comes from CODING declaring no
        # capabilities, not from a test handing one in.
        assert setup_budget_for(CODING, 900).seconds == 0.0
        assert not setup_budget_for(CODING, 900).allows_probing
        assert setup_budget_for(CODING_REPO, 900).allows_probing

    async def test_S005_coding_probe_is_unreachable_end_to_end(self) -> None:
        sandbox = _RecordingSandbox()
        await probe_environment(sandbox, setup_budget_for(CODING, 900))
        assert sandbox.commands == []

    async def test_S005_commands_stop_once_the_budget_is_spent(self) -> None:
        # Deleting run()'s budget check entirely left the suite green. Drive a
        # clock that exhausts the budget after the first command and require
        # that no further command is issued.
        elapsed = [0.0]

        def clock() -> float:
            return elapsed[0]

        class _Advancing(_RecordingSandbox):
            async def exec(self, command: str, timeout: float = 120) -> _Result:
                elapsed[0] += 10.0
                return await super().exec(command, timeout)

        sandbox = _Advancing({"command -v": _Result("/usr/bin/git")})
        await probe_environment(
            sandbox, SetupBudget(seconds=5.0), sandbox_owned=True, clock=clock
        )
        assert len(sandbox.commands) == 1, (
            f"probe issued {len(sandbox.commands)} commands after the budget "
            f"was spent: {sandbox.commands}"
        )

    async def test_S005_each_command_timeout_is_within_remaining_budget(self) -> None:
        seen: list[float] = []

        class _TimeoutRecording(_RecordingSandbox):
            async def exec(self, command: str, timeout: float = 120) -> _Result:
                seen.append(timeout)
                return await super().exec(command, timeout)

        sandbox = _TimeoutRecording({"command -v": _Result("/usr/bin/git")})
        await probe_environment(sandbox, SetupBudget(seconds=2.0), sandbox_owned=True)
        assert seen, "no command was issued"
        assert all(t <= 2.0 for t in seen), (
            f"a probe command was allowed {max(seen)}s against a 2.0s budget"
        )

    async def test_S005_completion_path_reports_truncation(self) -> None:
        """The final `truncated=out_of_budget()` must be reachable.

        The earlier version's clock crossed the budget at an *intermediate*
        guard, so the probe returned from there and the completion path was
        never executed -- hardcoding that line to False passed. This clock
        advances only when the last probe step runs, so every intermediate
        guard is under budget and only the final computation is over.
        """
        elapsed = [0.0]

        def clock() -> float:
            return elapsed[0]

        class _LateBurner(_RecordingSandbox):
            async def exec(self, command: str, timeout: float = 120) -> _Result:
                if "ls -1a" in command:
                    elapsed[0] += 100.0
                return await super().exec(command, timeout)

        sandbox = _LateBurner(
            {
                "command -v": _Result("/usr/bin/git"),
                "is-inside-work-tree": _Result("true"),
                "rev-parse HEAD": _Result("abc"),
                "wc -l": _Result("12"),
                "ls -1a": _Result("Makefile"),
            }
        )
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=50.0), sandbox_owned=True, clock=clock
        )
        assert any("ls -1a" in c for c in sandbox.commands), (
            "the probe never reached the project step, so the completion path "
            "was not exercised"
        )
        assert profile.truncated is True, (
            "a probe that overran its budget during its final step reported "
            "truncated=False"
        )

    async def test_S005_probe_within_budget_is_not_truncated(self) -> None:
        sandbox = _RecordingSandbox(
            {
                "command -v": _Result("/usr/bin/git"),
                "is-inside-work-tree": _Result("true"),
                "rev-parse HEAD": _Result("abc"),
                "wc -l": _Result("12"),
                "ls -1a": _Result("Makefile"),
            }
        )
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=30.0), sandbox_owned=True
        )
        assert profile.truncated is False


class TestSandboxOwnershipRoundTrips:
    """C4: the one field standing between the harness and writing into
    someone else's container had no round-trip test."""

    @pytest.mark.parametrize("owned", [True, False, None])
    async def test_S005_sandbox_owned_round_trips(self, owned: bool | None) -> None:
        sandbox = _RecordingSandbox({"command -v": _Result("/usr/bin/git")})
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=30.0), sandbox_owned=owned
        )
        assert profile.sandbox_owned is owned, (
            "probe did not preserve the caller's sandbox ownership"
        )

    async def test_S005_borrowed_container_never_affirms_git(self) -> None:
        sandbox = _RecordingSandbox(
            {
                "command -v": _Result("/usr/bin/git"),
                "is-inside-work-tree": _Result("true"),
                "rev-parse HEAD": _Result("abc"),
            }
        )
        profile = await probe_environment(
            sandbox, SetupBudget(seconds=30.0), sandbox_owned=False
        )
        assert profile.is_git_worktree is True
        assert not profile.affirms("git_substrate"), (
            "a real repository in a container we do not own still must not "
            "affirm writing refs into it"
        )


class TestObservationPhrasingIsPinned:
    """H3: the disclaimer could be rewritten to an imperative and stay green."""

    def test_S005_disclaimer_names_instructions_explicitly(self) -> None:
        rendered = render_observations(
            EnvironmentProfile(project=ProjectCommands(source="PATH", test="pytest"))
        )
        assert "not instructions" in rendered, (
            "the disclaimer must say 'not instructions'; a weaker phrase "
            "leaves the claim it exists to make unmade"
        )
        assert "verify before relying on them" in rendered

    def test_S005_renders_no_imperative(self) -> None:
        rendered = render_observations(
            EnvironmentProfile(project=ProjectCommands(source="PATH", test="pytest"))
        ).lower()
        for imperative in ("use them", "run the", "the test command is", "you should"):
            assert imperative not in rendered, f"observation reads as an instruction: {imperative!r}"

    def test_S005_pyproject_finding_is_attributed_to_where_it_came_from(self) -> None:
        # The finding comes from `command -v`, not from reading the file.
        # Attributing it to pyproject.toml states as fact something never read
        # -- the verb was fixed and the subject left wrong.
        rendered = render_observations(
            EnvironmentProfile(
                project=ProjectCommands(
                    source="PATH (a pyproject.toml is present, but was not read)",
                    test="pytest",
                )
            )
        )
        assert "was not read" in rendered

    def test_S005_commands_render_verbatim(self) -> None:
        # shlex.quote wrapped whole command lines in single quotes, so
        # `npm run lint` rendered as one unrunnable argv[0].
        rendered = render_observations(
            EnvironmentProfile(
                project=ProjectCommands(source="package.json", lint="npm run lint")
            )
        )
        assert "npm run lint" in rendered
        assert "'npm run lint'" not in rendered
