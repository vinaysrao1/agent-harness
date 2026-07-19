"""Docker-backed sandbox (DESIGN.md §4.7) — the v1 isolated execution backend.

``DockerSandbox`` lazily creates one long-lived container per instance on
:meth:`~DockerSandbox.start`, bind-mounting the host ``workspace`` directory
at ``/workspace`` (also the container's working directory) so artifacts
survive the container and are inspectable from the host. All ``docker`` SDK
calls are blocking, so every one is dispatched via ``asyncio.to_thread``
rather than blocking the event loop.

File operations (:meth:`~DockerSandbox.read_file` /
:meth:`~DockerSandbox.write_file`) go straight through the host side of the
bind mount (via the same helpers :mod:`~harness.sandbox.local` uses) rather
than through ``docker exec`` — once mounted, the file is byte-identical
from both sides, and it avoids a container round trip for what's usually a
small config/code file.

Network policy (DESIGN.md §4.7): ``NetworkMode.NONE`` maps to Docker's
``network_mode="none"`` (no egress at all — the default for untrusted
work); ``NetworkMode.OPEN`` maps to the default bridge network.
``NetworkMode.ALLOWLIST`` is **not enforced in v1** — there is no
per-destination egress filter yet — so it falls back to ``"none"`` with a
``UserWarning``, the safe direction to fail in.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any

from harness.config import NetworkMode
from harness.sandbox.base import (
    MAX_OUTPUT_BYTES,
    ExecResult,
    Sandbox,
    SandboxError,
    read_workspace_file,
    truncate_output,
    write_workspace_file,
)

logger = logging.getLogger(__name__)

__all__ = ["DockerSandbox"]

#: Network modes with unambiguous Docker equivalents. ``ALLOWLIST`` is
#: deliberately absent: it falls back to "none" (see module docstring).
_DOCKER_NETWORK_MODE: dict[NetworkMode, str] = {
    NetworkMode.NONE: "none",
    NetworkMode.OPEN: "bridge",
}


class DockerSandbox(Sandbox):
    """One Docker container per instance, created lazily on `start()`.

    Timeout note: unlike :class:`~harness.sandbox.local.LocalSandbox`,
    which SIGKILLs a whole OS process group on timeout, Docker's exec API
    has no primitive to cancel an in-flight exec short of stopping the
    entire container. On timeout, :meth:`exec` stops *waiting* on the
    blocking SDK call and issues a best-effort follow-up kill of the
    timed-out process *by PID* (looked up via ``exec_inspect``) so the
    worker thread blocked inside the Docker SDK's ``exec_start`` unblocks
    promptly, rather than staying parked in that call -- and pinning a slot
    in the shared ``asyncio.to_thread`` executor -- until the command
    finishes on its own (which, for a server or ``tail -f``-style command,
    is never; enough concurrently timed-out execs would otherwise exhaust
    the default executor and stall every other ``to_thread`` call, e.g.
    file ops and start/stop, across the whole process). The kill is a
    chase, not atomic with the timeout, so a command that's already
    exiting or unresponsive to SIGKILL for its own reasons can still hold
    the thread briefly longer. This is a known v1 limitation of the Docker
    backend.
    """

    def __init__(
        self,
        workspace: Path,
        image: str = "python:3.12-slim",
        network: NetworkMode = NetworkMode.NONE,
    ) -> None:
        """Create a sandbox for ``workspace``, bound to ``image``/``network``.

        No container is created until :meth:`start` (or first :meth:`exec`)
        is called.
        """
        self._workspace = Path(workspace)
        self._image = image
        self._network = network
        self._client: Any | None = None
        self._container: Any | None = None
        #: Guards start()/stop() so concurrent callers can't both pass the
        #: "no container yet" check and race to create two containers (see
        #: start()).
        self._lifecycle_lock = asyncio.Lock()

    @property
    def workspace(self) -> Path:
        """The host directory bind-mounted at ``/workspace`` in the container."""
        return self._workspace

    @classmethod
    def availability(cls) -> bool:
        """Return True iff a Docker daemon is reachable right now.

        Used to skip Docker-dependent tests cleanly when the daemon isn't
        running, and could equally gate a harness-level "fall back to
        LocalSandbox" policy. Never raises: any failure (SDK not installed,
        daemon not running, socket unreachable) is treated as unavailable.
        """
        try:
            import docker
        except ImportError:
            return False
        try:
            client = docker.from_env(timeout=2)
        except Exception:
            return False
        try:
            return bool(client.ping())
        except Exception:
            return False
        finally:
            client.close()

    async def start(self) -> None:
        """Create the container if it doesn't already exist (idempotent).

        Guarded by :attr:`_lifecycle_lock`: without it, two concurrent
        callers could each observe ``self._container is None``, both
        proceed to create a container, and leave the loser's container
        running forever (never stopped/removed) with its client leaked.
        """
        async with self._lifecycle_lock:
            if self._container is not None:
                return
            await asyncio.to_thread(self._workspace.mkdir, parents=True, exist_ok=True)
            client, container = await asyncio.to_thread(self._create_container)
            self._client = client
            self._container = container

    def _create_container(self) -> tuple[Any, Any]:
        """Blocking: create and start a container. Runs in a worker thread.

        Returns ``(client, container)`` rather than assigning
        ``self._client`` as a side effect: the caller only commits the
        client once this has *succeeded*, so a ``containers.run`` failure
        (bad image, daemon error) can't leak a half-initialized client that
        a later retried ``start()`` would silently overwrite (and thus
        never close).
        """
        import docker

        client = docker.from_env()
        network_mode = _DOCKER_NETWORK_MODE.get(self._network)
        if network_mode is None:
            warnings.warn(
                f"sandbox network mode {self._network.value!r} is not "
                "enforced in v1 (no per-destination egress allowlist yet); "
                "falling back to 'none'",
                UserWarning,
                stacklevel=2,
            )
            network_mode = "none"
        try:
            container = client.containers.run(
                self._image,
                command="sleep infinity",
                detach=True,
                working_dir="/workspace",
                volumes={
                    str(self._workspace.resolve()): {"bind": "/workspace", "mode": "rw"}
                },
                network_mode=network_mode,
                tty=False,
            )
        except Exception:
            client.close()
            raise
        return client, container

    async def stop(self) -> None:
        """Force-remove the container, if one was created (idempotent)."""
        async with self._lifecycle_lock:
            container, self._container = self._container, None
            client, self._client = self._client, None
        if container is not None:
            await asyncio.to_thread(self._remove_container, container)
        if client is not None:
            await asyncio.to_thread(client.close)

    @staticmethod
    def _remove_container(container: Any) -> None:
        """Blocking: stop+remove ``container``, swallowing already-gone races."""
        try:
            container.remove(force=True)
        except Exception:
            pass

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        """Run ``command`` in the container via the low-level exec API.

        The container object is captured into a local before being handed
        to worker threads, rather than having those threads re-read
        ``self._container`` -- so a concurrent :meth:`stop` nulling that
        attribute mid-exec can't make a helper crash on a ``None``
        container. See the class docstring for the timeout/kill behavior.
        """
        if self._container is None:
            await self.start()
        container = self._container
        if container is None:
            raise SandboxError("sandbox container is not available")

        exec_id = await asyncio.to_thread(self._exec_create, container, command)
        try:
            (
                exit_code,
                stdout_b,
                stderr_b,
                stdout_total,
                stderr_total,
            ) = await asyncio.wait_for(
                asyncio.to_thread(self._exec_start, container, exec_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            await asyncio.to_thread(self._kill_exec, container, exec_id)
            return ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        return ExecResult(
            exit_code=exit_code,
            stdout=truncate_output(
                stdout_b, stream_name="stdout", total_len=stdout_total
            ),
            stderr=truncate_output(
                stderr_b, stream_name="stderr", total_len=stderr_total
            ),
            timed_out=False,
        )

    @staticmethod
    def _exec_create(container: Any, command: str) -> str:
        """Blocking: create (but don't yet run) an exec instance.

        Just an API round trip -- fast regardless of how long ``command``
        itself ends up taking, so it's safe to await directly rather than
        folding it into the timeout-bounded call.
        """
        resp = container.client.api.exec_create(
            container.id, ["/bin/sh", "-c", command], workdir="/workspace"
        )
        return resp["Id"]

    @staticmethod
    def _exec_start(container: Any, exec_id: str) -> tuple[int, bytes, bytes, int, int]:
        """Blocking: run the previously-created exec to completion, streamed.

        Uses ``exec_start(..., demux=True, stream=True)`` and caps what's
        *retained* per stream at :data:`MAX_OUTPUT_BYTES` as chunks arrive,
        mirroring :func:`harness.sandbox.local._drain_capped` -- rather than
        the non-streaming form, which has the SDK buffer the exec's entire
        stdout/stderr in memory before returning, letting a runaway
        in-container command (e.g. ``yes``) make the harness process hold
        gigabytes of output over the run's timeout even though
        :func:`~harness.sandbox.base.truncate_output` only ever returns
        :data:`MAX_OUTPUT_BYTES` of it. Returns ``(exit_code, stdout_head,
        stderr_head, stdout_total, stderr_total)`` -- the ``_total`` counts
        are the true number of bytes each stream produced, for an accurate
        truncation marker (see :func:`~harness.sandbox.base.truncate_output`).
        """
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_buffered = 0
        stderr_buffered = 0
        stdout_total = 0
        stderr_total = 0
        for stdout_chunk, stderr_chunk in container.client.api.exec_start(
            exec_id, demux=True, stream=True
        ):
            if stdout_chunk:
                stdout_total += len(stdout_chunk)
                if stdout_buffered < MAX_OUTPUT_BYTES:
                    take = stdout_chunk[: MAX_OUTPUT_BYTES - stdout_buffered]
                    stdout_chunks.append(take)
                    stdout_buffered += len(take)
            if stderr_chunk:
                stderr_total += len(stderr_chunk)
                if stderr_buffered < MAX_OUTPUT_BYTES:
                    take = stderr_chunk[: MAX_OUTPUT_BYTES - stderr_buffered]
                    stderr_chunks.append(take)
                    stderr_buffered += len(take)
        exit_code = container.client.api.exec_inspect(exec_id).get("ExitCode")
        if exit_code is None:
            exit_code = -1
        return (
            exit_code,
            b"".join(stdout_chunks),
            b"".join(stderr_chunks),
            stdout_total,
            stderr_total,
        )

    @staticmethod
    def _kill_exec(container: Any, exec_id: str) -> None:
        """Blocking: best-effort SIGKILL of the process backing ``exec_id``.

        Called after :meth:`exec` has already given up waiting on
        :meth:`_exec_start` for timeout purposes -- that call is still
        running in another worker thread, blocked reading the exec's
        output. Killing the underlying PID (found via ``exec_inspect``)
        makes that blocking read return almost immediately instead of
        occupying the shared executor thread until the command finishes on
        its own. Never raises: the exec may have already finished (a
        natural exit racing the timeout) or the container may already be
        gone, in which case there's nothing left to kill.

        The kill is issued through ``/bin/sh -c`` rather than as a direct
        argv (``["kill", "-9", pid]``): ``exec_run`` with a list-form
        command does not go through a shell, so it requires a standalone
        ``kill`` executable on ``PATH`` inside the container. Debian
        provides one via the ``procps`` package, which slim images
        (including the ``python:3.12-slim`` default) don't install --
        against those images the list form fails with "executable file not
        found in $PATH", silently swallowed below, leaving the worker
        thread parked in :meth:`_exec_start` exactly as if no kill had been
        attempted at all. ``kill`` is a shell builtin, so routing through
        ``/bin/sh -c`` (already assumed elsewhere in this class, e.g.
        :meth:`_exec_create`) works regardless of what's on ``PATH``.
        """
        pid = None
        try:
            pid = container.client.api.exec_inspect(exec_id).get("Pid")
            if pid:
                container.exec_run(["/bin/sh", "-c", f"kill -9 {pid}"])
        except Exception:
            logger.debug(
                "best-effort kill of exec %s (pid %s) failed", exec_id, pid, exc_info=True
            )

    async def read_file(self, path: str) -> str:
        """Read a text file at ``path`` via the host side of the bind mount."""
        return await read_workspace_file(self._workspace, path)

    async def write_file(self, path: str, content: str) -> None:
        """Write a text file at ``path`` via the host side of the bind mount."""
        await write_workspace_file(self._workspace, path, content)
