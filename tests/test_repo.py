"""S-201: the git substrate, and the guarantees keeping it off the TB2 path."""

from __future__ import annotations

import pytest

from harness.environment import EnvironmentProfile
from harness.profiles import CODING, CODING_REPO
from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig
from harness.orchestrator import Orchestrator
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.types import Message, ModelResponse, Role, StopReason, ToolCall, Usage
from harness.repo import (
    BASELINE_REF_SUFFIX,
    CHECKPOINT_COMMAND_TIMEOUT,
    CHECKPOINT_EVENT,
    CHECKPOINT_SKIPPED_EVENT,
    CHECKPOINT_TIMEOUT_SECONDS,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    work_tree_argument,
    SHADOW_GIT_DIR,
    DirtyWorktreeError,
    GitSubstrate,
    RepoState,
)


GOAL = "Write hello.txt containing hi."
CLEAN_FINISH = "Task complete. Wrote the file; contents verified."

pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")


@pytest.fixture(autouse=True)
def _no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerSandbox, "availability", classmethod(lambda cls: False))


@pytest.fixture
def orchestrator(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return Orchestrator(HarnessConfig(home=home), RunStore(tmp_path / "state.db"))


def _write_then_finish() -> list[ModelResponse]:
    return [
        ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "hello.txt", "content": "hi"},
                    )
                ],
            ),
            usage=Usage(),
            stop_reason=StopReason.TOOL_USE,
        ),
        ModelResponse(
            message=Message(role=Role.ASSISTANT, content=CLEAN_FINISH),
            usage=Usage(),
            stop_reason=StopReason.END_TURN,
        ),
    ]


class _Result:
    def __init__(self, stdout: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.stderr = ""


class _Sandbox:
    def __init__(self, responses: dict[str, _Result] | None = None) -> None:
        self.commands: list[str] = []
        self.responses = responses or {}

    async def exec(self, command: str, timeout: float = 120) -> _Result:
        self.commands.append(command)
        for fragment, result in self.responses.items():
            if fragment in command:
                return result
        return _Result()


def _clean_repo() -> dict[str, _Result]:
    return {
        "rev-parse HEAD": _Result("deadbeef"),
        "abbrev-ref": _Result("main"),
        "remote.origin.url": _Result("git@example.com:org/repo.git"),
        "status --porcelain": _Result(""),
        "init --bare": _Result(""),
        "write-tree": _Result("treesha"),
        "commit-tree": _Result("commitsha"),
    }


class TestInertWhenTheCapabilityIsOff:
    async def test_S201_inactive_issues_no_commands(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=False)
        assert await substrate.start() == RepoState()
        assert await substrate.checkpoint(1) is None
        assert sandbox.commands == [], (
            "an inactive substrate executed commands; on the benchmark path "
            "that would break N3"
        )

    def test_S201_coding_can_never_activate_it(self) -> None:
        # Both halves are required. CODING asks for nothing, so no environment
        # can switch this on.
        rich = EnvironmentProfile(is_git_worktree=True, sandbox_owned=True)
        assert rich.affirms("git_substrate")
        assert not (CODING.enables("git_substrate") and rich.affirms("git_substrate"))

    def test_S201_borrowed_container_can_never_activate_it(self) -> None:
        # Acceptance (2): sandbox_owned=False makes it inert even under
        # CODING_REPO in a real repository. Writing refs into someone else's
        # container is not ours to do.
        borrowed = EnvironmentProfile(is_git_worktree=True, sandbox_owned=False)
        assert CODING_REPO.enables("git_substrate")
        assert not borrowed.affirms("git_substrate")

    def test_S201_no_git_makes_it_inert(self) -> None:
        no_git = EnvironmentProfile(is_git_worktree=None, sandbox_owned=True)
        assert not no_git.affirms("git_substrate")


class TestNoGitInsideTheWorkspace:
    async def test_S201_every_command_targets_the_shadow_store(self) -> None:
        # Acceptance (1). A .git in the workspace is a directory the agent can
        # see and the grader never expected.
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        await substrate.checkpoint(1)
        writing = [
            c
            for c in sandbox.commands
            if any(verb in c for verb in ("add -A", "write-tree", "commit-tree", "update-ref", "init --bare"))
        ]
        assert writing, "no write commands were issued at all"
        for command in writing:
            assert SHADOW_GIT_DIR in command, (
                f"a git write did not target the shadow store: {command!r}"
            )

    async def test_S201_shadow_dir_is_under_the_harness_tmp_prefix(self) -> None:
        # N4 permits the workspace and /tmp/.harness-*; the shadow store is
        # covered by construction rather than by exception.
        assert SHADOW_GIT_DIR.startswith("/tmp/.harness-")

    async def test_S201_checkpoint_never_runs_git_init_in_the_worktree(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        for command in sandbox.commands:
            if "init" in command:
                assert "--bare" in command and SHADOW_GIT_DIR in command


class TestDirtyWorktree:
    async def test_S201_refuses_a_dirty_tree(self) -> None:
        # Acceptance (4). A dirty tree means the reported diff mixes the
        # agent's work with changes already there.
        responses = _clean_repo() | {"status --porcelain": _Result(" M src/a.py")}
        substrate = GitSubstrate(_Sandbox(responses), "run1", active=True)
        with pytest.raises(DirtyWorktreeError):
            await substrate.start()

    async def test_S201_allow_dirty_proceeds(self) -> None:
        responses = _clean_repo() | {"status --porcelain": _Result(" M src/a.py")}
        substrate = GitSubstrate(
            _Sandbox(responses), "run1", active=True, allow_dirty=True
        )
        state = await substrate.start()
        assert state.dirty is True
        assert state.base_sha == "deadbeef"

    async def test_S201_clean_tree_proceeds(self) -> None:
        substrate = GitSubstrate(_Sandbox(_clean_repo()), "run1", active=True)
        state = await substrate.start()
        assert state.dirty is False
        assert state.branch == "main"
        assert state.repo == "git@example.com:org/repo.git"


class TestDeadlinePressure:
    class _Deadline:
        def __init__(self, landing: bool = False, affordable: float = 999.0) -> None:
            self.landing = landing
            self._affordable = affordable

        def affordable_exec_seconds(self) -> float:
            return self._affordable

    async def test_S201_landing_turn_skips_checkpointing(self) -> None:
        # Acceptance (3). A harness-initiated write the model never asked for
        # must not eat the landing reserve.
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(
            sandbox, "run1", active=True, deadline=self._Deadline(landing=True)
        )
        await substrate.start()
        before_commands = len(sandbox.commands)
        before_skips = substrate.skipped_count
        assert await substrate.checkpoint(1) is None
        assert len(sandbox.commands) == before_commands, "checkpoint ran during landing"
        assert substrate.skipped_count == before_skips + 1

    async def test_S201_insufficient_time_skips_checkpointing(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(
            sandbox,
            "run1",
            active=True,
            deadline=self._Deadline(affordable=CHECKPOINT_TIMEOUT_SECONDS - 1),
        )
        await substrate.start()
        before_commands = len(sandbox.commands)
        before_skips = substrate.skipped_count
        assert await substrate.checkpoint(1) is None
        assert len(sandbox.commands) == before_commands
        assert substrate.skipped_count == before_skips + 1

    async def test_S201_ample_time_checkpoints(self) -> None:
        # The control: without it the two skip tests could pass because
        # checkpointing never works at all.
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(
            sandbox, "run1", active=True, deadline=self._Deadline(affordable=999.0)
        )
        await substrate.start()
        before = substrate.checkpoint_count
        ref = await substrate.checkpoint(3)
        assert ref == "refs/harness/run1/turn-3"
        assert substrate.checkpoint_count == before + 1
        assert substrate.skipped_count == 0
        # start() must have captured the pre-agent tree, or there is nothing
        # to diff the first turn against.
        assert substrate.baseline_ref == f"refs/harness/run1/{BASELINE_REF_SUFFIX}"

    async def test_S201_no_deadline_checkpoints(self) -> None:
        substrate = GitSubstrate(_Sandbox(_clean_repo()), "run1", active=True)
        await substrate.start()
        assert await substrate.checkpoint(1) is not None


class TestFailureIsNotFatal:
    async def test_S201_failed_shadow_init_disables_rather_than_raises(self) -> None:
        responses = _clean_repo() | {"init --bare": _Result("", exit_code=1)}
        substrate = GitSubstrate(_Sandbox(responses), "run1", active=True)
        await substrate.start()
        assert substrate.active is False
        assert await substrate.checkpoint(1) is None

    async def test_S201_failed_checkpoint_is_counted_not_raised(self) -> None:
        responses = _clean_repo() | {"write-tree": _Result("", exit_code=1)}
        substrate = GitSubstrate(_Sandbox(responses), "run1", active=True)
        await substrate.start()
        before = substrate.skipped_count
        assert await substrate.checkpoint(1) is None
        assert substrate.skipped_count == before + 1

    async def test_S201_unknown_head_prevents_checkpointing(self) -> None:
        responses = _clean_repo() | {"rev-parse HEAD": _Result("", exit_code=1)}
        substrate = GitSubstrate(_Sandbox(responses), "run1", active=True)
        state = await substrate.start()
        assert not state.usable
        assert await substrate.checkpoint(1) is None

    async def test_S201_a_raising_sandbox_does_not_propagate(self) -> None:
        class _Exploding:
            async def exec(self, command: str, timeout: float = 120):
                raise RuntimeError("container gone")

        substrate = GitSubstrate(_Exploding(), "run1", active=True)
        state = await substrate.start()
        assert state == RepoState()
        assert await substrate.checkpoint(1) is None


class TestLoopWiring:
    """The loop must actually call the substrate, and must not when inactive.

    The first version of this class imported FakeAdapter, Orchestrator,
    AgentLoop and six types, used none of them, and called
    ``substrate.checkpoint()`` directly -- a verbatim duplicate of another
    test wearing an integration test's imports. Deleting the loop's entire
    checkpoint block left the whole suite green. These drive a real
    ``run_task`` so that deletion fails.
    """

    async def test_S201_loop_checkpoints_every_tool_turn(
        self, orchestrator, tmp_path, monkeypatch
    ) -> None:
        from harness.loop import AgentLoop

        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()

        # Inject the substrate into whatever loop the orchestrator builds.
        real_init = AgentLoop.__init__

        def patched(self, *args, **kwargs):
            kwargs.setdefault("repo", substrate)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(AgentLoop, "__init__", patched)

        before = substrate.checkpoint_count
        run_id, result = await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=FakeAdapter(_write_then_finish()),
            workspace=tmp_path / "ws",
        )
        assert result.status == "completed"
        assert substrate.checkpoint_count > before, (
            "the loop completed a tool turn without checkpointing; the "
            "checkpoint block is not being reached"
        )

        kinds = [
            event.kind
            for event in orchestrator.store.load_events(
                orchestrator.store.list_agents(run_id)[0].id
            )
        ]
        assert CHECKPOINT_EVENT in kinds, (
            f"no {CHECKPOINT_EVENT} event was recorded; got {sorted(set(kinds))}"
        )

    async def test_S201_loop_records_a_skip(
        self, orchestrator, tmp_path, monkeypatch
    ) -> None:
        # A skipped checkpoint must be visible in the log: absent telemetry is
        # indistinguishable from a checkpoint never attempted.
        from harness.loop import AgentLoop

        class _AlwaysLanding:
            landing = True

            def affordable_exec_seconds(self) -> float:
                return 0.0

        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        substrate._deadline = _AlwaysLanding()

        real_init = AgentLoop.__init__

        def patched(self, *args, **kwargs):
            kwargs.setdefault("repo", substrate)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(AgentLoop, "__init__", patched)

        run_id, _ = await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=FakeAdapter(_write_then_finish()),
            workspace=tmp_path / "ws-skip",
        )
        kinds = [
            event.kind
            for event in orchestrator.store.load_events(
                orchestrator.store.list_agents(run_id)[0].id
            )
        ]
        assert CHECKPOINT_SKIPPED_EVENT in kinds, (
            f"a skipped checkpoint was not recorded; got {sorted(set(kinds))}"
        )

    async def test_S201_no_substrate_means_no_repo_events(
        self, orchestrator, tmp_path
    ) -> None:
        # The CODING path: repo is None, so the loop's guard short-circuits and
        # nothing repo-shaped reaches the log.
        run_id, result = await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=FakeAdapter(_write_then_finish()),
            workspace=tmp_path / "ws-none",
        )
        assert result.status == "completed"
        kinds = {
            event.kind
            for event in orchestrator.store.load_events(
                orchestrator.store.list_agents(run_id)[0].id
            )
        }
        assert CHECKPOINT_EVENT not in kinds
        assert CHECKPOINT_SKIPPED_EVENT not in kinds

    async def test_S201_inactive_substrate_costs_nothing_per_turn(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=False)
        for turn in range(50):
            assert await substrate.checkpoint(turn) is None
        assert sandbox.commands == [], (
            "50 turns of an inactive substrate issued commands; the CODING "
            "path would pay for a capability it never asked for"
        )

    def test_S201_loop_accepts_a_substrate_and_defaults_to_none(self) -> None:
        import inspect

        from harness.loop import AgentLoop

        signature = inspect.signature(AgentLoop.__init__)
        assert "repo" in signature.parameters
        assert signature.parameters["repo"].default is None


class TestGitCommandShape:
    """HIGH-5: no test could distinguish a valid git command from an invalid
    one, which is how a command that cannot succeed in a container shipped."""

    async def test_S201_commit_tree_carries_an_identity(self) -> None:
        # BLOCKER-1: `commit-tree` resolves the committer under IDENT_STRICT.
        # A container with no user.email and a domainless hostname has
        # auto-detection refused, so without an explicit identity every
        # checkpoint fails -- silently, since a non-zero exit counts as a skip.
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        await substrate.checkpoint(1)
        commits = [c for c in sandbox.commands if "commit-tree" in c]
        assert commits, "no commit-tree command was issued"
        for command in commits:
            assert f"user.email={GIT_IDENTITY_EMAIL}" in command
            assert f"user.name={GIT_IDENTITY_NAME}" in command

    async def test_S201_commit_tree_receives_the_tree(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        await substrate.checkpoint(1)
        commits = [c for c in sandbox.commands if "commit-tree" in c]
        assert any("treesha" in c for c in commits), (
            f"commit-tree was issued without the tree it must commit: {commits}"
        )

    def test_S201_work_tree_is_the_exec_working_directory(self) -> None:
        # `--work-tree=.` is only correct because every Sandbox execs with the
        # workspace as its cwd (DockerSandbox passes workdir, LocalSandbox
        # passes cwd). Pointing it at `/` or dropping it was invisible before.
        assert work_tree_argument() == "--work-tree=."

    async def test_S201_every_checkpoint_command_uses_the_work_tree(self) -> None:
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        await substrate.checkpoint(1)
        for command in sandbox.commands:
            if "--git-dir" in command:
                assert work_tree_argument() in command, (
                    f"a shadow command has no work tree: {command!r}"
                )

    def test_S201_total_cost_is_bounded_by_the_stated_budget(self) -> None:
        # Acceptance (3) said "cost is bounded", but one affordability check
        # against a single command's timeout was followed by four commands each
        # allowed that timeout -- a 4x under-count.
        assert CHECKPOINT_COMMAND_TIMEOUT * 4 <= CHECKPOINT_TIMEOUT_SECONDS

    async def test_S201_no_command_exceeds_its_slice(self) -> None:
        seen: list[float] = []

        class _TimeoutRecording(_Sandbox):
            async def exec(self, command: str, timeout: float = 120) -> _Result:
                if "--git-dir" in command and "init" not in command:
                    seen.append(timeout)
                return await super().exec(command, timeout)

        sandbox = _TimeoutRecording(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        await substrate.checkpoint(1)
        assert seen, "no checkpoint command was issued"
        assert all(t <= CHECKPOINT_COMMAND_TIMEOUT for t in seen), (
            f"a checkpoint command was allowed {max(seen)}s against a "
            f"{CHECKPOINT_COMMAND_TIMEOUT}s slice"
        )


class TestBaseline:
    async def test_S201_start_captures_the_pre_agent_tree(self) -> None:
        # BLOCKER-3: the shadow store shares no objects with the workspace
        # repo, so the recorded base_sha is unresolvable there. Without a
        # baseline ref the earliest snapshot already contains the agent's
        # first edits and there is nothing to diff against.
        sandbox = _Sandbox(_clean_repo())
        substrate = GitSubstrate(sandbox, "run1", active=True)
        await substrate.start()
        assert substrate.baseline_ref == f"refs/harness/run1/{BASELINE_REF_SUFFIX}"
        assert any(BASELINE_REF_SUFFIX in c for c in sandbox.commands)

    async def test_S201_no_baseline_when_inactive(self) -> None:
        substrate = GitSubstrate(_Sandbox(_clean_repo()), "run1", active=False)
        await substrate.start()
        assert substrate.baseline_ref is None
