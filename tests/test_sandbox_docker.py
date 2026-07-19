"""Unit tests for harness.sandbox.docker.

Everything that needs a live Docker daemon is gated by
``pytest.mark.skipif(not DockerSandbox.availability(), ...)`` so this file
skips cleanly (not errors) when Docker/Colima isn't running -- which is the
expected state on plain CI and on a laptop without Docker Desktop open.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import docker
import pytest

from harness.config import NetworkMode
from harness.sandbox.base import SandboxError, SandboxPathError
from harness.sandbox.docker import DockerSandbox

DOCKER_UP = DockerSandbox.availability()
requires_docker = pytest.mark.skipif(
    not DOCKER_UP, reason="Docker daemon not reachable"
)


class TestAvailability:
    def test_availability_returns_bool_without_raising(self):
        # Must never raise regardless of whether a daemon is running.
        assert isinstance(DockerSandbox.availability(), bool)


class _FakeExecAPI:
    """Minimal stand-in for ``docker.APIClient``'s ``exec_*`` methods.

    Lets the timeout/kill control flow in ``DockerSandbox.exec`` be
    exercised deterministically, without a live daemon: ``exec_start``
    (called with ``stream=True``, matching the real streaming call) blocks
    on the first item pulled from its generator (simulating a long-running
    in-container command) until either the fake's own safety timeout
    elapses or ``_kill_exec`` "kills" it by setting the event -- exactly
    what a real ``kill -9 <pid>`` inside the container would cause the
    blocked SDK call to do.
    """

    def __init__(self, unblock_after: float = 3.0) -> None:
        self._unblock = threading.Event()
        self._unblock_after = unblock_after
        self._next_id = 0
        self.kill_calls: list[str] = []
        #: Raw ``exec_run`` command lists the fake container was invoked
        #: with -- lets tests assert on the *shape* of the kill command,
        #: not just that some kill happened.
        self.exec_run_calls: list[list[str]] = []

    def exec_create(self, container: str, cmd: Any, workdir: str | None = None) -> dict:
        self._next_id += 1
        return {"Id": f"exec-{self._next_id}"}

    def exec_start(
        self, exec_id: str, demux: bool = True, stream: bool = False
    ) -> Any:
        if not stream:
            # The non-streaming form must not be used any more (it forces
            # the SDK to fully buffer output before returning) -- fail
            # loudly rather than silently falling back to old behavior.
            raise AssertionError(
                "exec_start called without stream=True -- output is no "
                "longer streamed/capped incrementally"
            )
        return self._stream()

    def _stream(self):
        # Bounded even if the fix regresses and nothing ever unblocks us,
        # so a broken implementation fails the test instead of hanging the
        # suite forever.
        self._unblock.wait(timeout=self._unblock_after)
        return
        yield  # pragma: no cover - makes this a generator function

    def exec_inspect(self, exec_id: str) -> dict:
        return {"ExitCode": 0, "Pid": 4242}


class _FakeContainer:
    """Minimal stand-in for a ``docker.models.containers.Container``."""

    def __init__(self, api: _FakeExecAPI) -> None:
        self.id = "fake-container"
        self.client = SimpleNamespace(api=api)

    def exec_run(self, cmd: list[str], **kwargs: Any) -> Any:
        self.client.api.exec_run_calls.append(cmd)
        if len(cmd) == 3 and cmd[0] == "/bin/sh" and cmd[1] == "-c" and cmd[2].startswith(
            "kill -9 "
        ):
            pid = cmd[2].removeprefix("kill -9 ").strip()
            self.client.api.kill_calls.append(pid)
            self.client.api._unblock.set()
        return SimpleNamespace(exit_code=0, output=(b"", b""))


class TestDockerSandboxTimeoutKillFlow:
    """Regression coverage for the timeout/kill control flow -- doesn't need
    a live Docker daemon since it swaps in a fake exec API."""

    async def test_timeout_kills_blocked_exec_by_pid(self, tmp_path: Path):
        api = _FakeExecAPI(unblock_after=3.0)
        container = _FakeContainer(api)
        sb = DockerSandbox(tmp_path / "ws")
        sb._container = container
        sb._client = SimpleNamespace(api=api)

        start = time.monotonic()
        result = await sb.exec("sleep 999", timeout=0.2)
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert result.exit_code == -1
        # The exec's in-container process must be killed by the PID
        # reported by exec_inspect so the worker thread blocked in
        # exec_start() unblocks promptly.
        assert api.kill_calls == ["4242"]
        # The kill must be routed through a shell (`/bin/sh -c "kill -9
        # <pid>"`), not issued as a bare argv (`["kill", "-9", pid]`):
        # exec_run's list form doesn't go through a shell, so it needs a
        # standalone `kill` binary on PATH, which slim images (including
        # the DockerSandbox default, python:3.12-slim) don't ship -- that
        # form silently fails there and never unblocks the worker thread.
        assert api.exec_run_calls == [["/bin/sh", "-c", "kill -9 4242"]]
        # Well under the fake's 3s worst-case block, proving the kill (not
        # the fake's own safety timeout) is what unblocked the thread.
        assert elapsed < 2.0

    async def test_no_timeout_does_not_kill(self, tmp_path: Path):
        api = _FakeExecAPI(unblock_after=0.0)  # returns immediately
        container = _FakeContainer(api)
        sb = DockerSandbox(tmp_path / "ws")
        sb._container = container
        sb._client = SimpleNamespace(api=api)

        result = await sb.exec("echo hi", timeout=5)
        assert result.timed_out is False
        assert api.kill_calls == []


class _FakeStreamExecAPI:
    """Stand-in ``exec_start`` that yields demuxed chunks incrementally
    instead of handing back the fully-buffered output up front -- lets the
    streaming-cap regression test assert output is drained (and capped) as
    it arrives, not buffered in full by the SDK before truncation.
    """

    def __init__(
        self, stdout_total: int, stderr_total: int, chunk_size: int = 4096
    ) -> None:
        self.stdout_total = stdout_total
        self.stderr_total = stderr_total
        self.chunk_size = chunk_size
        self.calls: list[dict[str, bool]] = []

    def exec_create(self, container: str, cmd: Any, workdir: str | None = None) -> dict:
        return {"Id": "exec-stream"}

    def exec_start(self, exec_id: str, demux: bool = True, stream: bool = False) -> Any:
        self.calls.append({"demux": demux, "stream": stream})
        if not stream:
            # The old, non-streaming call shape: hands back the *entire*
            # output in one go. If DockerSandbox regresses to calling this
            # instead of the streaming form, the test below still passes
            # functionally (truncate_output caps it) but the call-shape
            # assertion below catches the regression directly.
            return (
                b"o" * self.stdout_total,
                b"e" * self.stderr_total,
            )
        return self._gen()

    def _gen(self):
        remaining_out = self.stdout_total
        remaining_err = self.stderr_total
        while remaining_out > 0 or remaining_err > 0:
            out_chunk = None
            err_chunk = None
            if remaining_out > 0:
                n = min(self.chunk_size, remaining_out)
                out_chunk = b"o" * n
                remaining_out -= n
            if remaining_err > 0:
                n = min(self.chunk_size, remaining_err)
                err_chunk = b"e" * n
                remaining_err -= n
            yield (out_chunk, err_chunk)

    def exec_inspect(self, exec_id: str) -> dict:
        return {"ExitCode": 0, "Pid": 1}


class TestDockerSandboxStreamingOutputCap:
    """Regression coverage: exec output must be streamed and capped as it
    arrives, not fully buffered by the SDK before truncate_output runs."""

    async def test_output_streamed_and_capped_incrementally(self, tmp_path: Path):
        from harness.sandbox.base import MAX_OUTPUT_BYTES

        stdout_total = MAX_OUTPUT_BYTES * 3
        stderr_total = MAX_OUTPUT_BYTES + 10
        api = _FakeStreamExecAPI(stdout_total, stderr_total)
        container = _FakeContainer(api)
        sb = DockerSandbox(tmp_path / "ws")
        sb._container = container
        sb._client = SimpleNamespace(api=api)

        result = await sb.exec("produce-lots-of-output", timeout=5)

        assert result.exit_code == 0
        assert result.timed_out is False
        # Must be invoked in streaming mode -- a regression back to the
        # fully-buffered call (stream=False) would fail this.
        assert api.calls == [{"demux": True, "stream": True}]
        # Truncation marker must report the *true* total, even though only
        # a capped head was ever retained in memory.
        assert f"{stdout_total} bytes total" in result.stdout
        assert f"{stderr_total} bytes total" in result.stderr
        # Only the capped head bytes (plus a short marker) are
        # retained/returned per stream -- not the full multi-hundred-KB
        # stream.
        assert len(result.stdout) < MAX_OUTPUT_BYTES + 200
        assert len(result.stderr) < MAX_OUTPUT_BYTES + 200
        assert result.stdout.startswith("o" * 100)
        assert result.stderr.startswith("e" * 100)

    async def test_output_under_cap_is_not_truncated(self, tmp_path: Path):
        api = _FakeStreamExecAPI(stdout_total=5, stderr_total=0)
        container = _FakeContainer(api)
        sb = DockerSandbox(tmp_path / "ws")
        sb._container = container
        sb._client = SimpleNamespace(api=api)

        result = await sb.exec("echo ooooo", timeout=5)

        assert result.stdout == "ooooo"
        assert result.stderr == ""
        assert "truncated" not in result.stdout


class _RunFailsContainers:
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom: no such image")


class _FakeDockerClient:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.closed = False
        self.delay = delay
        if fail:
            self.containers = _RunFailsContainers()
        else:
            self.containers = SimpleNamespace(run=self._run)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        if self.delay:
            time.sleep(self.delay)
        return _FakeContainer(_FakeExecAPI())

    def close(self) -> None:
        self.closed = True


class TestDockerSandboxLifecycle:
    """Regression coverage for start()/stop() lifecycle races -- mocks
    ``docker.from_env`` rather than requiring a live daemon."""

    async def test_concurrent_start_creates_only_one_container(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        created_clients: list[_FakeDockerClient] = []

        def fake_from_env(*args: Any, **kwargs: Any) -> _FakeDockerClient:
            client = _FakeDockerClient(delay=0.05)
            created_clients.append(client)
            return client

        monkeypatch.setattr(docker, "from_env", fake_from_env)

        sb = DockerSandbox(tmp_path / "ws")
        await asyncio.gather(*(sb.start() for _ in range(5)))

        # Without the start()/stop() lock, each of the 5 concurrent callers
        # could pass the "no container yet" check before any of them
        # finished creating one, each calling docker.from_env()/
        # containers.run() and leaking all but the last container/client.
        assert len(created_clients) == 1
        assert sb._container is not None

    async def test_failed_create_does_not_leak_client(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        fake_client = _FakeDockerClient(fail=True)
        monkeypatch.setattr(docker, "from_env", lambda *a, **kw: fake_client)

        sb = DockerSandbox(tmp_path / "ws")
        with pytest.raises(RuntimeError):
            await sb.start()

        # self._client must not be left pointing at a client whose
        # containers.run() never succeeded -- and that client should have
        # been closed rather than leaked.
        assert sb._client is None
        assert sb._container is None
        assert fake_client.closed is True

    async def test_stop_during_inflight_exec_does_not_raise_bare_assertion(
        self, tmp_path: Path
    ):
        api = _FakeExecAPI(unblock_after=0.0)
        container = _FakeContainer(api)
        sb = DockerSandbox(tmp_path / "ws")
        sb._container = container
        sb._client = SimpleNamespace(api=api, close=lambda: None)

        exec_task = asyncio.ensure_future(sb.exec("echo hi", timeout=5))
        # Let exec() actually start running and capture the container into
        # its local variable before stop() nulls self._container out from
        # under it -- otherwise this test would just race stop() against
        # exec()'s very first line and not exercise the mid-flight case.
        await asyncio.sleep(0)
        await sb.stop()
        # Must complete via the captured-container-as-parameter path
        # rather than re-reading self._container (now None) from a worker
        # thread and hitting a bare AssertionError.
        result = await exec_task
        assert result.exit_code == 0


@requires_docker
class TestDockerSandboxExec:
    @pytest.fixture
    async def sandbox(self, tmp_path: Path):
        sb = DockerSandbox(tmp_path / "workspace", network=NetworkMode.NONE)
        try:
            yield sb
        finally:
            await sb.stop()

    async def test_echo_stdout(self, sandbox: DockerSandbox):
        result = await sandbox.exec("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.timed_out is False

    async def test_exit_code_nonzero(self, sandbox: DockerSandbox):
        result = await sandbox.exec("exit 5")
        assert result.exit_code == 5

    async def test_timeout_reports_timed_out(self, sandbox: DockerSandbox):
        result = await sandbox.exec("sleep 30", timeout=0.5)
        assert result.timed_out is True

    async def test_start_is_idempotent(self, sandbox: DockerSandbox):
        await sandbox.start()
        container_id = sandbox._container.id  # type: ignore[attr-defined]
        await sandbox.start()
        assert sandbox._container.id == container_id  # type: ignore[attr-defined]

    async def test_context_manager(self, tmp_path: Path):
        async with DockerSandbox(tmp_path / "ws2") as sb:
            result = await sb.exec("echo hi")
            assert result.exit_code == 0

    async def test_network_none_blocks_egress(self, sandbox: DockerSandbox):
        # network_mode="none" -> no interfaces to reach the outside world.
        result = await sandbox.exec(
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 2)\""
        )
        assert result.exit_code != 0

    async def test_allowlist_falls_back_to_none_with_warning(self, tmp_path: Path):
        sb = DockerSandbox(tmp_path / "ws3", network=NetworkMode.ALLOWLIST)
        try:
            with pytest.warns(UserWarning, match="not enforced"):
                await sb.start()
        finally:
            await sb.stop()


@requires_docker
class TestDockerSandboxFiles:
    @pytest.fixture
    async def sandbox(self, tmp_path: Path):
        sb = DockerSandbox(tmp_path / "workspace")
        try:
            yield sb
        finally:
            await sb.stop()

    async def test_write_then_read(self, sandbox: DockerSandbox):
        await sandbox.write_file("notes.txt", "hello from host")
        assert await sandbox.read_file("notes.txt") == "hello from host"

    async def test_write_visible_inside_container(self, sandbox: DockerSandbox):
        await sandbox.write_file("seen.txt", "container-visible")
        result = await sandbox.exec("cat seen.txt")
        assert "container-visible" in result.stdout

    async def test_edit_file_round_trip(self, sandbox: DockerSandbox):
        await sandbox.write_file("f.py", "x = 1\n")
        await sandbox.edit_file("f.py", "x = 1", "x = 2")
        assert await sandbox.read_file("f.py") == "x = 2\n"

    async def test_read_missing_file_raises(self, sandbox: DockerSandbox):
        await sandbox.start()
        with pytest.raises(SandboxError):
            await sandbox.read_file("missing.txt")

    async def test_path_traversal_rejected(self, sandbox: DockerSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("../outside.txt")

    async def test_absolute_path_rejected(self, sandbox: DockerSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.write_file("/etc/passwd", "pwned")
