"""S-104: background execution.

A build that outruns the exec cap returns nothing today — the command is
killed at the timeout and the agent learns only that it took too long. That is
a whole class of Terminal-Bench task where the right move is to start the
build, do something else, and come back.

A started job is a *promise*: nothing else in the harness knows the process is
there, so nothing else can notice a run ending with it still going, or an agent
that started it and never looked.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from harness.jobs import (
    ABANDONED_EVENT,
    Job,
    JobRegistry,
    kill_command,
    parse_poll,
    poll_command,
    start_command,
)
from harness.sandbox.local import LocalSandbox
from harness.tools.builtin import bash_output_tool, bash_tool, kill_tool


@pytest.fixture
def workspace() -> Path:
    return Path(tempfile.mkdtemp())


async def _sandbox(directory: Path) -> LocalSandbox:
    sandbox = LocalSandbox(directory)
    await sandbox.start()
    return sandbox


class TestTheBenchmarkSchemaIsUntouched:
    """N2 pins the tool surface. An unconditional `run_in_background` argument
    would change `CODING`'s digest and make a Lane A spec a Lane B one."""

    def test_S104_bash_without_a_registry_has_two_arguments(self) -> None:
        properties = bash_tool(None).spec.input_schema["properties"]
        assert sorted(properties) == ["command", "timeout"]

    def test_S104_bash_with_a_registry_offers_background(self) -> None:
        properties = bash_tool(None, jobs=JobRegistry()).spec.input_schema["properties"]
        assert "run_in_background" in properties

    def test_S104_the_benchmark_profile_does_not_enable_it(self) -> None:
        from harness.profiles import CODING, CODING_REPO

        assert not CODING.enables("background_execution")
        assert CODING_REPO.enables("background_execution")

    def test_S104_the_two_bash_factories_are_distinct(self) -> None:
        # Repo mode substitutes its bash by reference. Selecting it by list
        # position would silently pick up whatever ended up first.
        from harness.orchestrator import coding_bash_factory, repo_bash_factory
        from harness.profiles import CODING_REPO

        assert coding_bash_factory not in CODING_REPO.tool_factories
        assert repo_bash_factory in CODING_REPO.tool_factories


class TestARealBackgroundJob:
    async def test_S104_a_job_outlives_the_call_that_started_it(
        self, workspace
    ) -> None:
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        result = await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 0.3; echo finished > out.txt", "run_in_background": True}
        )
        assert result.startswith("started job-")
        # The tool returned before the command did: that is the whole point.
        assert not (workspace / "out.txt").exists()

        await asyncio.sleep(1.0)
        assert (workspace / "out.txt").read_text().strip() == "finished"

    async def test_S104_output_is_readable_while_it_runs_and_after(
        self, workspace
    ) -> None:
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo first; sleep 0.4; echo second", "run_in_background": True}
        )
        reader = bash_output_tool(sandbox, jobs)

        await asyncio.sleep(0.2)
        during = await reader.handler({"handle": jobs.all()[0].handle})
        assert "still running" in during
        assert "first" in during

        await asyncio.sleep(1.0)
        after = await reader.handler({"handle": jobs.all()[0].handle})
        assert "finished with exit code 0" in after
        assert "second" in after

    async def test_S104_a_poll_returns_only_what_is_new(self, workspace) -> None:
        # Re-sending a growing build log every poll would spend the context
        # window on text the model has already read.
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo alpha; sleep 0.4; echo beta", "run_in_background": True}
        )
        reader = bash_output_tool(sandbox, jobs)
        await asyncio.sleep(0.2)
        first = await reader.handler({"handle": jobs.all()[0].handle})
        await asyncio.sleep(1.0)
        second = await reader.handler({"handle": jobs.all()[0].handle})
        assert "alpha" in first
        assert "alpha" not in second, second
        assert "beta" in second

    async def test_S104_a_failing_job_reports_its_exit_code(
        self, workspace
    ) -> None:
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo oops >&2; exit 3", "run_in_background": True}
        )
        await asyncio.sleep(0.8)
        result = await bash_output_tool(sandbox, jobs).handler(
            {"handle": jobs.all()[0].handle}
        )
        assert "exit code 3" in result
        assert "oops" in result, "stderr was not captured"

    async def test_S104_kill_stops_a_running_job(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 30; echo should_not_appear > out.txt",
             "run_in_background": True}
        )
        await asyncio.sleep(0.2)
        assert "killed" in await kill_tool(sandbox, jobs).handler(
            {"handle": jobs.all()[0].handle}
        )
        await asyncio.sleep(0.5)
        assert not (workspace / "out.txt").exists()

    async def test_S104_kill_reaches_spawned_children(self, workspace) -> None:
        # A build spawns compilers. Killing only the shell leaves them running
        # and the sandbox busy, which the next trial inherits.
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        # The child's sleep must be short enough that the test outlives it:
        # waiting 0.6s while the child sleeps 30s proves nothing, because the
        # file would be absent whether or not the kill reached it. It sleeps
        # 1.2s and the assertion happens at ~2.2s.
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "(sleep 1.2; echo child > child.txt) & sleep 30",
             "run_in_background": True}
        )
        await asyncio.sleep(0.3)
        await kill_tool(sandbox, jobs).handler({"handle": jobs.all()[0].handle})
        await asyncio.sleep(1.9)
        assert not (workspace / "child.txt").exists(), (
            "the child outlived the kill; only the job's own shell was signalled"
        )

    async def test_S104_the_child_would_have_written_without_the_kill(
        self, workspace
    ) -> None:
        # The control for the test above. Without it, a child that never runs
        # at all would make the kill look effective.
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "(sleep 1.2; echo child > child.txt) & sleep 30",
             "run_in_background": True}
        )
        await asyncio.sleep(2.2)
        assert (workspace / "child.txt").exists(), "the child never ran"
        await kill_tool(sandbox, jobs).handler({"handle": jobs.all()[0].handle})

    async def test_S104_a_payload_containing_quotes_survives(
        self, workspace
    ) -> None:
        # The payload crosses two shells. The earlier test for this asserted
        # only that the output path appeared in the generated script, which
        # deleting both `shlex.quote` calls also satisfies. This one runs it.
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        payload = "echo \"it's a 'quoted' \\$PATH\" > out.txt"
        result = await bash_tool(sandbox, jobs=jobs).handler(
            {"command": payload, "run_in_background": True}
        )
        assert result.startswith("started ")
        await asyncio.sleep(0.8)
        assert (workspace / "out.txt").read_text().strip() == "it's a 'quoted' $PATH"

    async def test_S104_an_unknown_handle_says_what_exists(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        jobs = JobRegistry()
        with pytest.raises(ValueError, match="started jobs: none"):
            await bash_output_tool(sandbox, jobs).handler({"handle": "job-9"})


class TestForegroundIsUnaffected:
    """Acceptance (4): existing exec semantics are preserved — N6."""

    async def test_S104_a_foreground_command_still_blocks(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        result = await bash_tool(sandbox, jobs=JobRegistry()).handler(
            {"command": "echo done > out.txt"}
        )
        assert (workspace / "out.txt").read_text().strip() == "done"
        assert "started job" not in result

    async def test_S104_background_false_is_foreground(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        result = await bash_tool(sandbox, jobs=JobRegistry()).handler(
            {"command": "echo done > out.txt", "run_in_background": False}
        )
        assert (workspace / "out.txt").exists()
        assert "started job" not in result


class TestTheRegistry:
    def test_S104_handles_are_unique(self) -> None:
        jobs = JobRegistry()
        assert jobs.next_handle() != jobs.next_handle()

    def test_S104_live_excludes_killed(self) -> None:
        jobs = JobRegistry()
        jobs.add(Job(handle="a", command="x", pid="1"))
        jobs.add(Job(handle="b", command="y", pid="2", killed=True))
        assert [job.handle for job in jobs.live()] == ["a"]

    def test_S104_abandoned_is_started_but_never_polled(self) -> None:
        jobs = JobRegistry()
        jobs.add(Job(handle="a", command="x", pid="1", polled=True))
        jobs.add(Job(handle="b", command="y", pid="2"))
        assert [job.handle for job in jobs.abandoned()] == ["b"]


class TestPollParsing:
    def test_S104_a_finished_job_is_distinguished_from_a_running_one(self) -> None:
        # Parsed from a sentinel file, not the process table: a finished job
        # leaves no process, and "no such pid" is indistinguishable from
        # "never started".
        assert parse_poll("__HARNESS_EXIT__0\n__HARNESS_OFFSET__3\nout") == (
            "0", "out", 3
        )
        assert parse_poll("__HARNESS_RUNNING__\n__HARNESS_OFFSET__7\npartial") == (
            None, "partial", 7
        )

    def test_S104_a_missing_exit_code_is_not_read_as_success(self) -> None:
        assert parse_poll("__HARNESS_EXIT__\n__HARNESS_OFFSET__0\n") == ("?", "", 0)

    def test_S104_an_unreadable_offset_is_not_invented(self) -> None:
        # The offset must come from the shell's own `wc -c`. If that could not
        # be read, the previous value is kept rather than a guess derived from
        # the decoded text -- which is exactly how it desynchronised before.
        assert parse_poll("__HARNESS_RUNNING__\n__HARNESS_OFFSET__\nx")[2] is None

    def test_S104_kill_targets_the_group_and_the_process(self) -> None:
        command = kill_command("1234")
        assert "-1234" in command and "TERM" in command and "KILL" in command


class TestNoRunEndsHoldingAJob:
    """Acceptance (1) and (3). A detached command is deliberately not bounded
    by the exec cap, so nothing else in the harness would ever stop it — and a
    sandbox left compiling is one the next trial inherits."""

    async def _loop(self, workspace, jobs):
        from harness.adapters.fake import FakeAdapter
        from harness.context import ContextManager
        from harness.loop import AgentLoop, Budgets
        from harness.permissions import Policy
        from harness.persistence import RunStore
        from harness.config import PermissionMode
        from harness.tools.registry import ToolRegistry

        store = RunStore(workspace / "state.db")
        run_id = store.create_run("goal", "m", "auto")
        agent_id = store.create_agent(run_id, "goal")
        registry = ToolRegistry(jobs=jobs)
        sandbox = await _sandbox(workspace)
        loop = AgentLoop(
            adapter=FakeAdapter([]),
            registry=registry,
            policy=Policy(mode=PermissionMode.AUTO),
            store=store,
            run_id=run_id,
            agent_id=agent_id,
            context=ContextManager(
                base_system_prompt="x", count_tokens=lambda m: 0,
                max_context=1000, summarize=None,
            ),
            budgets=Budgets(),
            ask=None,
            sandbox=sandbox,
        )
        return loop, store, agent_id, sandbox

    async def test_S104_reaping_kills_every_live_job(self, workspace) -> None:
        # The job must finish inside the test's own window, or the assertion
        # holds whether or not the kill landed. `job.killed` proves nothing
        # either -- it is set unconditionally after the kill attempt.
        jobs = JobRegistry()
        loop, store, agent_id, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 1.2; echo leaked > leaked.txt",
             "run_in_background": True}
        )
        await asyncio.sleep(0.3)
        await loop._reap_background_jobs("landing")
        await asyncio.sleep(1.9)
        assert not (workspace / "leaked.txt").exists(), "the job outlived the reap"

    async def test_S104_the_job_would_have_written_without_the_reap(
        self, workspace
    ) -> None:
        # The control for the test above.
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 1.2; echo leaked > leaked.txt",
             "run_in_background": True}
        )
        await asyncio.sleep(2.2)
        assert (workspace / "leaked.txt").exists(), "the job never ran"
        await loop._reap_background_jobs("teardown")

    async def test_S104_a_finished_job_is_not_reported_as_live(
        self, workspace
    ) -> None:
        # "Not killed" was the definition of live, and nothing marked natural
        # completion -- so a job polled to its exit code stayed live forever.
        # It was announced at wind-down as still running, and signalled at
        # landing on a pid the kernel had long since reused.
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo done", "run_in_background": True}
        )
        await asyncio.sleep(0.6)
        await bash_output_tool(sandbox, jobs).handler(
            {"handle": jobs.all()[0].handle}
        )
        assert jobs.live() == [], "a completed job is still reported as running"
        assert await loop._background_job_notice() is None

    async def test_S104_an_unpolled_job_is_recorded_as_abandoned(
        self, workspace
    ) -> None:
        # Whether the model actually comes back for its build is the question
        # this feature exists to answer. A mechanism nobody uses should be
        # visible as unused rather than counted as shipped.
        jobs = JobRegistry()
        loop, store, agent_id, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo hi", "run_in_background": True}
        )
        await loop._reap_background_jobs("landing")
        kinds = [event.kind for event in store.load_events(agent_id)]
        assert ABANDONED_EVENT in kinds, kinds

    async def test_S104_a_polled_job_is_not_abandoned(self, workspace) -> None:
        # The control. If every job were reported abandoned, the signal would
        # carry no information.
        jobs = JobRegistry()
        loop, store, agent_id, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo hi", "run_in_background": True}
        )
        await asyncio.sleep(0.4)
        await bash_output_tool(sandbox, jobs).handler(
            {"handle": jobs.all()[0].handle}
        )
        await loop._reap_background_jobs("landing")
        kinds = [event.kind for event in store.load_events(agent_id)]
        assert ABANDONED_EVENT not in kinds, kinds

    async def test_S104_reaping_without_a_registry_is_a_no_op(
        self, workspace
    ) -> None:
        # The benchmark path: no registry, nothing to reap, no events.
        loop, store, agent_id, sandbox = await self._loop(workspace, None)
        await loop._reap_background_jobs("landing")
        assert store.load_events(agent_id) == []

    async def test_S104_the_wind_down_notice_names_live_jobs(
        self, workspace
    ) -> None:
        # Named while there is still time to act on it. Killing at wind-down
        # would be wrong -- the build may be the work.
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        assert await loop._background_job_notice() is None

        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 5", "run_in_background": True}
        )
        notice = await loop._background_job_notice()
        assert notice and "still running" in notice and "sleep 5" in notice
        await loop._reap_background_jobs("teardown")

    async def test_S104_a_killed_job_is_not_named_again(self, workspace) -> None:
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 5", "run_in_background": True}
        )
        await kill_tool(sandbox, jobs).handler({"handle": jobs.all()[0].handle})
        assert await loop._background_job_notice() is None


class TestReapingHappensOnEveryExit:
    """Reaping only at landing left a run that errors, pauses on budget, or is
    cancelled holding live jobs — and a subagent that finishes normally never
    arms landing at all. On the benchmark path there is no backstop:
    `HarborSandbox.stop` is a no-op by design, so a leaked build keeps
    compiling into the grading phase."""

    async def test_S104_a_normal_finish_reaps(self, workspace) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.types import Message, ModelResponse, Role, StopReason, Usage

        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        loop.adapter = FakeAdapter([
            ModelResponse(
                message=Message(role=Role.ASSISTANT,
                                content="Task complete. Started the build and read it."),
                usage=Usage(), stop_reason=StopReason.END_TURN,
            )
        ])
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 1.2; echo leaked > leaked.txt",
             "run_in_background": True}
        )
        await loop.run("do the thing")
        await asyncio.sleep(1.9)
        assert not (workspace / "leaked.txt").exists(), (
            "a normally-finished run left its job running"
        )

    async def test_S104_a_crashing_run_still_reaps(self, workspace) -> None:
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)

        async def explode(goal):
            raise RuntimeError("boom")

        loop._run = explode
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 1.2; echo leaked > leaked.txt",
             "run_in_background": True}
        )
        with pytest.raises(RuntimeError):
            await loop.run("do the thing")
        await asyncio.sleep(1.9)
        assert not (workspace / "leaked.txt").exists(), (
            "a crashed run left its job running"
        )

    _loop = TestNoRunEndsHoldingAJob._loop


class TestJobFilesDoNotCollide:
    async def test_S104_two_registries_do_not_share_output_files(
        self, workspace
    ) -> None:
        # Handles restart at 1 per registry and job output lives at a path
        # derived from the handle, so two agents sharing a sandbox both wrote
        # to `job-1.out` and read each other's exit codes.
        sandbox = await _sandbox(workspace)
        first, second = JobRegistry(), JobRegistry()
        await bash_tool(sandbox, jobs=first).handler(
            {"command": "echo from_first", "run_in_background": True}
        )
        await bash_tool(sandbox, jobs=second).handler(
            {"command": "echo from_second", "run_in_background": True}
        )
        await asyncio.sleep(0.6)
        one = await bash_output_tool(sandbox, first).handler(
            {"handle": first.all()[0].handle}
        )
        two = await bash_output_tool(sandbox, second).handler(
            {"handle": second.all()[0].handle}
        )
        assert "from_first" in one and "from_second" not in one, one
        assert "from_second" in two and "from_first" not in two, two


def _docker_ready() -> bool:
    import subprocess

    try:
        from harness.sandbox.docker import DockerSandbox

        if not DockerSandbox.availability():
            return False
    except Exception:  # noqa: BLE001
        return False
    return subprocess.run(
        ["docker", "image", "inspect", "harness-sandbox:latest"], capture_output=True
    ).returncode == 0


@pytest.mark.skipif(
    not _docker_ready(),
    reason=(
        "needs harness-sandbox:latest, where /bin/sh is dash — the shell that "
        "broke two previous attempts at this. `docker build -t "
        "harness-sandbox:latest .`"
    ),
)
class TestUnderTheProductionShell:
    """`/bin/sh` in the image is dash, and `docker exec` has no tty.

    That combination is what defeated both earlier designs: `setsid` was
    absent on the *host* so its fallback hid the failure there, and `set -m`
    is silently disabled by dash without a controlling terminal — it prints
    "can't access tty; job control turned off" and carries on, leaving the job
    in the sandbox exec's process group where a group kill is ESRCH.

    Every test above runs on macOS, where /bin/sh is bash and `set -m` works.
    They were green throughout both bugs.
    """

    async def _docker(self, directory: Path):
        from harness.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox(directory, image="harness-sandbox:latest")
        await sandbox.start()
        return sandbox

    async def test_S104_a_job_gets_its_own_process_group(self, workspace) -> None:
        sandbox = await self._docker(workspace)
        try:
            jobs = JobRegistry()
            await bash_tool(sandbox, jobs=jobs).handler(
                {"command": "sleep 20", "run_in_background": True}
            )
            job = jobs.all()[0]
            assert job.pgid == job.pid, (
                f"pid {job.pid} is in group {job.pgid}; a group kill cannot "
                "reach its children"
            )
            await kill_tool(sandbox, jobs).handler({"handle": job.handle})
        finally:
            await sandbox.stop()

    async def test_S104_the_pgid_probe_reads_the_kernel_under_linux(
        self, workspace
    ) -> None:
        # The macOS twin of this test exercises the `ps` fallback, because the
        # host has no /proc. Field 5 of /proc/<pid>/stat is the pgrp; field 1
        # is the pid, and reading the wrong one is invisible to any assertion
        # of the form `pgid == pid`.
        from harness.jobs import _READ_PGID

        sandbox = await self._docker(workspace)
        try:
            probe = _READ_PGID.format(pid="$pid")
            result = await sandbox.exec(
                f"sleep 5 & pid=$!\npgid=$({probe})\necho \"$pid $pgid\"\n"
                f"kill -KILL $pid 2>/dev/null; true\n",
                timeout=20,
            )
            pid, pgid = result.stdout.strip().split("\n")[-1].split()
            assert pgid.isdigit(), result.stdout
            assert pgid != pid, (
                "the probe reported the pid as the group id: "
                f"{result.stdout!r}"
            )
        finally:
            await sandbox.stop()

    async def test_S104_kill_reaches_children_under_dash(self, workspace) -> None:
        sandbox = await self._docker(workspace)
        try:
            jobs = JobRegistry()
            await bash_tool(sandbox, jobs=jobs).handler(
                {"command": "(sleep 1.5; echo child > /tmp/child.txt) & sleep 20",
                 "run_in_background": True}
            )
            await asyncio.sleep(0.4)
            await kill_tool(sandbox, jobs).handler({"handle": jobs.all()[0].handle})
            await asyncio.sleep(2.0)
            probe = await sandbox.exec("cat /tmp/child.txt 2>/dev/null || echo ABSENT")
            assert "ABSENT" in probe.stdout, (
                "the child outlived the kill under dash — the process group "
                "was never established"
            )
        finally:
            await sandbox.stop()

    async def test_S104_a_heredoc_command_actually_runs(self, workspace) -> None:
        # `( cmd )` on one line makes `EOF )` not a terminator, so an
        # unterminated heredoc killed the inner shell before it ran anything —
        # and the job reported started, then polled as running forever.
        # `cat <<EOF > Makefile` is a construct the model uses constantly.
        sandbox = await self._docker(workspace)
        try:
            jobs = JobRegistry()
            await bash_tool(sandbox, jobs=jobs).handler({
                "command": "cat <<'EOF' > /tmp/hd.txt\nhello heredoc\nEOF",
                "run_in_background": True,
            })
            await asyncio.sleep(1.2)
            probe = await sandbox.exec("cat /tmp/hd.txt 2>/dev/null || echo ABSENT")
            assert "hello heredoc" in probe.stdout, probe.stdout
        finally:
            await sandbox.stop()

    async def test_S104_a_trailing_comment_does_not_break_the_job(
        self, workspace
    ) -> None:
        sandbox = await self._docker(workspace)
        try:
            jobs = JobRegistry()
            await bash_tool(sandbox, jobs=jobs).handler(
                {"command": "echo built > /tmp/c.txt  # build it",
                 "run_in_background": True}
            )
            await asyncio.sleep(1.2)
            probe = await sandbox.exec("cat /tmp/c.txt 2>/dev/null || echo ABSENT")
            assert "built" in probe.stdout, probe.stdout
        finally:
            await sandbox.stop()


class _ScriptedSandbox:
    """A sandbox that returns whatever the shell was told to say.

    The refusal path and the measurement path cannot be reached from a real
    shell on any host we have: `setsid` works, so `pgid` always equals `pid`.
    Both were shipped untested for exactly that reason -- and both survived
    every mutation, including replacing the whole measurement with `echo $pid`.
    """

    def __init__(self, stdout: str, exit_code: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.stderr = stderr
        self.commands: list[str] = []

    async def exec(self, command: str, timeout: float = 120):
        from harness.sandbox.base import ExecResult

        self.commands.append(command)
        if command.lstrip().startswith("kill "):
            # Not `"kill " in command`: `start_command` itself contains a
            # `kill -0` liveness probe, so that matcher answered the start
            # with an empty stdout and the refusal never fired.
            return ExecResult(exit_code=0, stdout="", stderr="")
        return ExecResult(
            exit_code=self.exit_code, stdout=self.stdout, stderr=self.stderr
        )


class TestTheProcessGroupIsMeasuredNotAssumed:
    """The central claim of this feature is that the group is *read back*.
    Every test of it compared `job.pgid` to `job.pid`, which is what `echo
    $pid` also produces -- so the entire measurement could be deleted and the
    suite stayed green. This is the project's recurring archetype: a mechanism
    that never fires, behind healthy telemetry."""

    async def test_S104_the_pgid_probe_reads_the_kernel(self, workspace) -> None:
        # A child started *without* `setsid` inherits its parent's group, so
        # its true pgid differs from its pid. If the probe echoed the pid, or
        # read the wrong field of /proc/<pid>/stat (field 1 is the pid), the
        # two would be equal.
        from harness.jobs import _READ_PGID

        sandbox = await _sandbox(workspace)
        probe = _READ_PGID.format(pid="$pid")
        result = await sandbox.exec(
            f"sleep 5 & pid=$!\npgid=$({probe})\necho \"$pid $pgid\"\n"
            f"kill -KILL $pid 2>/dev/null; true\n",
            timeout=15,
        )
        pid, pgid = result.stdout.strip().split("\n")[-1].split()
        assert pgid.isdigit(), result.stdout
        assert pgid != pid, (
            "the probe reported the pid as the group id, so it is not "
            f"measuring anything: {result.stdout!r}"
        )

    async def test_S104_a_job_outside_its_group_is_refused(self) -> None:
        sandbox = _ScriptedSandbox("4242 7")
        jobs = JobRegistry()
        with pytest.raises(ValueError, match="isolate"):
            await bash_tool(sandbox, jobs=jobs).handler(
                {"command": "make -j8", "run_in_background": True}
            )
        assert jobs.all() == [], "a refused job must not be tracked"

    async def test_S104_the_refused_process_is_stopped_not_its_group(self) -> None:
        # `kill -TERM -4242` on a process that is not a group leader signals
        # group 4242 -- which, on the path where this fires, is somebody
        # else's group. On a `docker exec` that is the sandbox's own shell.
        sandbox = _ScriptedSandbox("4242 7")
        with pytest.raises(ValueError):
            await bash_tool(sandbox, jobs=JobRegistry()).handler(
                {"command": "make -j8", "run_in_background": True}
            )
        # `"kill" in c` would match `start_command`'s own `kill -0` liveness
        # probe, so deleting the kill entirely left this green.
        kills = [c for c in sandbox.commands if c.lstrip().startswith("kill ")]
        assert kills, "the refused job's shell was left running"
        assert "-4242" not in "".join(kills), kills

    async def test_S104_an_isolated_job_is_accepted(self) -> None:
        sandbox = _ScriptedSandbox("4242 4242")
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "make -j8", "run_in_background": True}
        )
        assert [j.pgid for j in jobs.all()] == ["4242"]
        assert jobs.live(), "an isolated job is live until polled or killed"

    async def test_S104_an_already_finished_job_is_accepted_but_not_live(
        self,
    ) -> None:
        # "gone" means the process ended before its group could be read.
        # Registering it live meant the reap signalled a pid the kernel was
        # already free to reuse, and wind-down announced it as still running.
        sandbox = _ScriptedSandbox("4242 gone")
        jobs = JobRegistry()
        message = await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "echo hi", "run_in_background": True}
        )
        assert "already finished" in message
        assert len(jobs.all()) == 1
        assert jobs.live() == []

    async def test_S104_a_warning_line_is_not_read_as_the_pid(self) -> None:
        # dash prints `can't access tty; job control turned off` before the
        # echo when job control is unavailable. Taking the first token of the
        # whole stream read `can't` as the pid.
        sandbox = _ScriptedSandbox("sh: can't access tty\n4242 4242")
        jobs = JobRegistry()
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "make", "run_in_background": True}
        )
        assert [j.pid for j in jobs.all()] == ["4242"]


class TestTeardownCannotMakeThingsWorse:
    """The reap runs in `run`'s `finally`, where anything it raises replaces
    the run's real outcome, and where a second cancellation can land mid-kill."""

    _loop = TestNoRunEndsHoldingAJob._loop

    async def test_S104_abandonment_is_reported_once_not_once_per_reap(
        self, workspace
    ) -> None:
        # The reap runs at landing *and* at teardown. Counting raw events
        # would have said abandonment is twice as common on runs that reach
        # landing as on runs that error -- signal-shaped noise in the one
        # number this feature exists to produce.
        jobs = JobRegistry()
        loop, store, agent_id, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 0.4", "run_in_background": True}
        )
        await loop._reap_background_jobs("landing")
        await loop._reap_background_jobs("teardown")
        events = [
            e for e in store.load_events(agent_id) if e.kind == ABANDONED_EVENT
        ]
        assert len(events) == 1, [e.payload for e in events]

    async def test_S104_a_failing_store_does_not_replace_the_runs_error(
        self, workspace
    ) -> None:
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        await bash_tool(sandbox, jobs=jobs).handler(
            {"command": "sleep 0.4", "run_in_background": True}
        )

        def explode(*_args, **_kwargs):
            raise RuntimeError("store unavailable at teardown")

        loop.store.append_event = explode

        async def fail(_goal):
            raise RuntimeError("THE REAL FAILURE")

        loop._run = fail
        with pytest.raises(RuntimeError, match="THE REAL FAILURE"):
            await loop.run("do the thing")
        assert all(job.killed for job in jobs.all()), (
            "the kills were skipped because telemetry threw first"
        )

    async def test_S104_a_cancellation_mid_reap_still_reaps_the_rest(
        self, workspace
    ) -> None:
        jobs = JobRegistry()
        loop, _, _, sandbox = await self._loop(workspace, jobs)
        for _ in range(3):
            await bash_tool(sandbox, jobs=jobs).handler(
                {"command": "sleep 1.2; echo leaked >> leaked.txt",
                 "run_in_background": True}
            )
        real_exec = sandbox.exec
        kills = {"n": 0}

        async def flaky(command: str, timeout: float = 120):
            if command.lstrip().startswith("kill "):
                kills["n"] += 1
                if kills["n"] == 1:
                    raise asyncio.CancelledError()
            return await real_exec(command, timeout=timeout)

        sandbox.exec = flaky
        with pytest.raises(asyncio.CancelledError):
            await loop._reap_background_jobs("teardown")
        await asyncio.sleep(1.9)
        # The job whose kill was cancelled leaks -- that one is unavoidable.
        # The other two must not: letting `CancelledError` out of the per-job
        # `except Exception` aborted the loop and leaked all three.
        leaked = (workspace / "leaked.txt")
        lines = leaked.read_text().splitlines() if leaked.exists() else []
        assert len(lines) == 1, (
            f"a cancellation during the reap leaked {len(lines)} of 3 jobs"
        )


class TestTheToolsWithoutARegistry:
    """`REPO_TOOL_FACTORIES` pass `deps.jobs` straight through, and `deps.jobs`
    is None unless the profile enables `background_execution`. A repo profile
    whose environment declines the capability therefore reaches these handlers
    with None — and without the guard, `jobs.get(...)` is an AttributeError
    the model reads as a harness bug rather than as a disabled feature."""

    async def test_S104_bash_output_says_the_feature_is_off(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        with pytest.raises(ValueError, match="not enabled"):
            await bash_output_tool(sandbox, None).handler({"handle": "job-1"})

    async def test_S104_kill_says_the_feature_is_off(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        with pytest.raises(ValueError, match="not enabled"):
            await kill_tool(sandbox, None).handler({"handle": "job-1"})
