"""Local subprocess sandbox (DESIGN.md §4.7) — the no-Docker fallback.

``LocalSandbox`` runs commands as ordinary host subprocesses, scoped only to
a working directory and the path-containment checks in
:mod:`harness.sandbox.base`. **This provides no real isolation**: commands
run with the invoking user's own privileges, full filesystem visibility
(outside the workspace-root file-op checks, which do not constrain
``exec``), and full network access. It exists purely so the harness has a
working sandbox when the Docker daemon isn't running (DESIGN.md §4.7 calls
Docker "the lightest thing with full bash and good-enough isolation"; this
is the fallback for when even that isn't available). Real isolation is
:class:`~harness.sandbox.docker.DockerSandbox` — prefer it for anything
untrusted.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path

from harness.sandbox.base import (
    MAX_OUTPUT_BYTES,
    ExecResult,
    Sandbox,
    read_workspace_file,
    truncate_output,
    write_workspace_file,
)

__all__ = ["LocalSandbox"]

#: Chunk size for incrementally draining a command's stdout/stderr pipes.
_READ_CHUNK_BYTES = 65536

#: Bound on how long `exec` waits, once a timed-out command's process group
#: has been SIGKILLed, for its stdout/stderr pipes to report EOF. Needed
#: because a grandchild that detached into its own session (e.g. via
#: ``os.setsid()``) can survive the SIGKILL to the original process group
#: and keep the inherited pipe write ends open indefinitely -- without this
#: bound, exec() would hang forever waiting for EOF that never comes.
_DRAIN_GRACE_SECONDS = 5.0


async def _drain_capped(
    stream: asyncio.StreamReader | None, cap: int
) -> tuple[bytes, int]:
    """Read ``stream`` to EOF, retaining at most ``cap`` bytes of output.

    Reading is incremental and anything past ``cap`` is discarded as it's
    read rather than buffered in full and sliced afterward -- otherwise a
    runaway command (``yes``, catting a multi-GB file) could make the
    harness process hold gigabytes in memory during its up-to-``timeout``
    run even though only ``cap`` bytes are ever returned. Returns
    ``(head, total_len)``: ``head`` is at most ``cap`` bytes, ``total_len``
    is the true number of bytes seen (for an accurate truncation marker).
    """
    if stream is None:
        return b"", 0
    chunks: list[bytes] = []
    buffered = 0
    total = 0
    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if buffered < cap:
            take = chunk[: cap - buffered]
            chunks.append(take)
            buffered += len(take)
    return b"".join(chunks), total


def _drain_result(task: "asyncio.Task[tuple[bytes, int]]") -> tuple[bytes, int]:
    """Best-effort result of a :func:`_drain_capped` task.

    Returns ``(b"", 0)`` if the task never finished (i.e. it was cancelled
    after the post-timeout grace period expired because a lingering
    grandchild kept its pipe open) rather than raising.
    """
    if task.done() and not task.cancelled() and task.exception() is None:
        return task.result()
    return b"", 0


class LocalSandbox(Sandbox):
    """Runs commands and file operations directly on the host.

    Commands run via ``asyncio.create_subprocess_shell`` with
    ``cwd=workspace``. Each command is started in its own OS process group
    (``start_new_session=True``); on timeout the *whole group* is sent
    ``SIGKILL`` so shell children (e.g. a backgrounded pipeline) die too,
    not just the top-level shell.

    File operations (:meth:`read_file`/:meth:`write_file`) use plain
    synchronous :mod:`pathlib` calls run off the event loop via
    ``asyncio.to_thread`` rather than an ``aiofiles`` dependency — see
    :mod:`harness.sandbox.base` for why that tradeoff is made here.
    """

    def __init__(self, workspace: Path) -> None:
        """Create a sandbox rooted at ``workspace`` (created on `start()`)."""
        self._workspace = Path(workspace)

    @property
    def workspace(self) -> Path:
        """The host directory this sandbox's workspace root is rooted at."""
        return self._workspace

    async def start(self) -> None:
        """Create the workspace directory if it doesn't already exist."""
        await asyncio.to_thread(self._workspace.mkdir, parents=True, exist_ok=True)

    async def stop(self) -> None:
        """No-op: there is no persistent process or container to tear down."""
        return None

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        """Run ``command`` via the shell, killing its process group on timeout.

        stdout/stderr are drained incrementally and capped in memory as
        they're read (see :func:`_drain_capped`) rather than buffered in
        full via ``proc.communicate()``. Waiting for the process to exit
        and for both streams to reach EOF is bounded by ``timeout``
        together, matching ``communicate()``'s semantics. If that bound is
        hit, the whole process group is SIGKILLed and a short, separate
        grace period (:data:`_DRAIN_GRACE_SECONDS`) is given for the
        streams to drain -- bounded because a grandchild that detached into
        its own session can survive the SIGKILL and hold its pipe open
        forever, which would otherwise hang this method indefinitely.
        """
        await self.start()
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self._workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        wait_task = asyncio.ensure_future(proc.wait())
        stdout_task = asyncio.ensure_future(_drain_capped(proc.stdout, MAX_OUTPUT_BYTES))
        stderr_task = asyncio.ensure_future(_drain_capped(proc.stderr, MAX_OUTPUT_BYTES))

        timed_out = False
        _, pending = await asyncio.wait(
            {wait_task, stdout_task, stderr_task}, timeout=timeout
        )
        if pending:
            timed_out = True
            self._kill_process_group(proc)
            _, still_pending = await asyncio.wait(
                pending, timeout=_DRAIN_GRACE_SECONDS
            )
            for task in still_pending:
                task.cancel()
            for task in still_pending:
                with contextlib.suppress(BaseException):
                    await task
            if still_pending:
                # A stream still hasn't hit EOF after the grace period --
                # almost certainly a grandchild that detached into its own
                # session and is still holding the pipe open. Force our
                # side of the transport closed now (rather than leaving it
                # to a later GC pass, possibly after the event loop that
                # owns it has already closed) so we stop waiting on it and
                # don't leak the pipe file descriptors.
                with contextlib.suppress(BaseException):
                    proc._transport.close()  # type: ignore[attr-defined]

        exit_code = -1 if timed_out else (proc.returncode or 0)
        stdout_b, stdout_total = _drain_result(stdout_task)
        stderr_b, stderr_total = _drain_result(stderr_task)
        return ExecResult(
            exit_code=exit_code,
            stdout=truncate_output(
                stdout_b, stream_name="stdout", total_len=stdout_total
            ),
            stderr=truncate_output(
                stderr_b, stream_name="stderr", total_len=stderr_total
            ),
            timed_out=timed_out,
        )

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        """Send SIGKILL to ``proc``'s whole process group, ignoring races.

        ``ProcessLookupError`` means the group is already gone (the command
        finished between the timeout firing and the kill landing) — not an
        error worth surfacing.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def read_file(self, path: str) -> str:
        """Read a text file at ``path`` relative to the workspace root."""
        return await read_workspace_file(self._workspace, path)

    async def write_file(self, path: str, content: str) -> None:
        """Write a text file at ``path`` relative to the workspace root."""
        await write_workspace_file(self._workspace, path, content)
