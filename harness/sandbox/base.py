"""Sandbox abstraction for agent-authored code execution (DESIGN.md §4.7).

A :class:`Sandbox` is where the agent loop's ``bash``/``read_file``/
``write_file``/``edit_file`` tools actually run. The interface is
backend-agnostic on purpose (DESIGN.md §4.7: "the sandbox tool interface is
backend-agnostic so swapping the runtime doesn't touch the agent loop") —
:mod:`harness.sandbox.local` is a same-host fallback with no real isolation,
:mod:`harness.sandbox.docker` is the v1 isolated backend, and a future
microVM backend (Apple `container` / Firecracker) would implement the same
contract.

All file-path arguments across the interface are relative to the sandbox's
workspace root; :func:`resolve_workspace_path` is the one place that turns a
relative path into a validated absolute host path, rejecting any attempt to
escape the workspace (absolute paths, ``..`` traversal, or symlinks that
resolve outside the root) by raising :class:`SandboxPathError`.
"""

from __future__ import annotations

import abc
import asyncio
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

__all__ = [
    "MAX_OUTPUT_BYTES",
    "SandboxError",
    "SandboxPathError",
    "ExecResult",
    "Sandbox",
    "apply_edit",
    "resolve_workspace_path",
    "truncate_output",
    "read_workspace_file",
    "write_workspace_file",
]

#: Per-stream truncation limit for `exec` output (DESIGN.md §4.7).
MAX_OUTPUT_BYTES: Final[int] = 100_000


class SandboxError(Exception):
    """Base class for sandbox failures (bad file ops, edit conflicts, ...)."""


class SandboxPathError(SandboxError):
    """Raised when a requested path escapes the sandbox workspace root.

    Covers absolute paths, ``..`` traversal that resolves outside the root,
    and symlinks (inside the workspace) that resolve to a target outside it.
    """


class ExecResult(BaseModel):
    """Outcome of one :meth:`Sandbox.exec` command.

    ``stdout``/``stderr`` are already truncated (see :func:`truncate_output`)
    if they exceeded :data:`MAX_OUTPUT_BYTES`. ``timed_out`` is True iff the
    command was killed for exceeding its timeout; in that case ``exit_code``
    has no meaningful process exit status and is set to ``-1``.
    """

    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def truncate_output(
    data: bytes, *, stream_name: str, total_len: int | None = None
) -> str:
    """Decode ``data`` as UTF-8 and truncate at :data:`MAX_OUTPUT_BYTES`.

    Decoding replaces undecodable bytes rather than raising, since command
    output is not guaranteed to be valid UTF-8. When truncated, a clear
    marker naming the stream, the limit, and the original size is appended
    so the agent knows data was dropped rather than mistaking a short read
    for the whole output.

    ``total_len`` is the true number of bytes the stream produced, for
    reporting in the truncation marker. It defaults to ``len(data)`` for
    callers that pass the full stream contents, but a caller that already
    capped what it *retained* while incrementally draining a stream (to
    avoid buffering unbounded output in memory — see
    :mod:`harness.sandbox.local`) can pass the true count separately, since
    ``data`` alone may already be no larger than :data:`MAX_OUTPUT_BYTES`.
    """
    if total_len is None:
        total_len = len(data)
    if total_len <= MAX_OUTPUT_BYTES:
        return data.decode("utf-8", errors="replace")
    head = data[:MAX_OUTPUT_BYTES]
    marker = (
        f"\n...[{stream_name} truncated at {MAX_OUTPUT_BYTES} bytes; "
        f"{total_len} bytes total]...\n"
    )
    return head.decode("utf-8", errors="replace") + marker


def resolve_workspace_path(workspace: Path, path: str) -> Path:
    """Resolve ``path`` (relative to ``workspace``) to a validated host path.

    Raises :class:`SandboxPathError` if ``path`` is empty, absolute, or
    resolves (after following any symlinks, via :meth:`Path.resolve`) to
    somewhere outside ``workspace``. Neither ``workspace`` nor ``path`` is
    required to already exist: :meth:`Path.resolve` is non-strict by
    default, so it resolves a path lexically/through any symlinks that do
    exist without requiring the final component (or ``workspace`` itself)
    to be present on disk. This function never creates anything; callers
    that need the workspace root to exist (e.g. before running a command in
    it) are responsible for creating it themselves.
    """
    if not path:
        raise SandboxPathError("path must not be empty")
    if Path(path).is_absolute():
        raise SandboxPathError(
            f"path must be relative to the sandbox workspace root, got "
            f"absolute path {path!r}"
        )
    workspace_root = workspace.resolve()
    candidate = workspace_root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise SandboxPathError(
            f"path {path!r} escapes the sandbox workspace root "
            f"({workspace_root})"
        )
    return resolved


async def read_workspace_file(workspace: Path, path: str) -> str:
    """Read the text file at ``path`` inside ``workspace``.

    Shared by :class:`~harness.sandbox.local.LocalSandbox` and
    :class:`~harness.sandbox.docker.DockerSandbox`: both ultimately expose
    the same bind-mounted (or identical, for local) host directory, so plain
    synchronous :mod:`pathlib` calls run off the event loop via
    ``asyncio.to_thread`` are sufficient — no need for an ``aiofiles``
    dependency for what is, for a personal tool, small config/code files.
    """
    resolved = resolve_workspace_path(workspace, path)

    def _read() -> str:
        try:
            return resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SandboxError(f"file not found: {path}") from None
        except IsADirectoryError:
            raise SandboxError(f"path is a directory, not a file: {path}") from None

    return await asyncio.to_thread(_read)


async def write_workspace_file(workspace: Path, path: str, content: str) -> None:
    """Write ``content`` to the text file at ``path`` inside ``workspace``.

    Creates parent directories as needed; overwrites an existing file.
    """
    resolved = resolve_workspace_path(workspace, path)

    def _write() -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write)


def apply_edit(
    content: str, old_string: str, new_string: str, *, replace_all: bool = False
) -> str:
    """Apply one ``old_string`` -> ``new_string`` replacement to ``content``.

    Mirrors the harness's standard file-edit tool semantics (DESIGN.md §8:
    "copy Claude Code's tool conventions closely", specifically old/new
    string edit semantics): ``old_string`` must match ``content`` exactly.
    Unless ``replace_all`` is set, it must also match *uniquely* — ambiguous
    edits are rejected rather than guessed at. Raises :class:`SandboxError`
    describing exactly what went wrong (not found / not unique) so the
    caller (typically an agent) can correct its next attempt.
    """
    if old_string == "":
        raise SandboxError("old_string must be non-empty")
    count = content.count(old_string)
    if count == 0:
        raise SandboxError("old_string not found in file")
    if not replace_all and count > 1:
        raise SandboxError(
            f"old_string is not unique in file ({count} occurrences); pass "
            "replace_all=True to replace all of them, or include more "
            "surrounding context to make it unique"
        )
    if replace_all:
        return content.replace(old_string, new_string)
    return content.replace(old_string, new_string, 1)


class Sandbox(abc.ABC):
    """Abstract execution sandbox for agent-authored code (DESIGN.md §4.7).

    All paths passed to :meth:`read_file`, :meth:`write_file`, and
    :meth:`edit_file` are relative to the sandbox's workspace root;
    implementations must route them through :func:`resolve_workspace_path`
    (directly or via :func:`read_workspace_file`/:func:`write_workspace_file`)
    so traversal outside the root raises :class:`SandboxPathError`.

    Supports the async context-manager protocol as sugar for
    ``start()``/``stop()``.
    """

    @abc.abstractmethod
    async def start(self) -> None:
        """Bring the sandbox up (create workspace / container as needed).

        Idempotent: calling ``start()`` on an already-started sandbox is a
        no-op.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Tear the sandbox down (remove container, if any).

        Idempotent: calling ``stop()`` on an already-stopped (or
        never-started) sandbox is a no-op.
        """

    @abc.abstractmethod
    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        """Run ``command`` as a shell command in the sandbox workspace.

        Blocks until the command exits or ``timeout`` seconds elapse, in
        which case the command (and, where the backend supports it, its
        child processes) is killed and the result has ``timed_out=True``.
        ``stdout``/``stderr`` are truncated per :data:`MAX_OUTPUT_BYTES`.
        """

    @abc.abstractmethod
    async def read_file(self, path: str) -> str:
        """Return the text content of the file at ``path``.

        Raises :class:`SandboxError` if the path does not exist or is a
        directory, and :class:`SandboxPathError` if it escapes the
        workspace root.
        """

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to the file at ``path``, creating it if needed.

        Creates any missing parent directories. Raises
        :class:`SandboxPathError` if ``path`` escapes the workspace root.
        """

    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> None:
        """Replace ``old_string`` with ``new_string`` in the file at ``path``.

        Default implementation: read the file, apply :func:`apply_edit`
        (which enforces the exact-match / uniqueness contract — see its
        docstring for the exact error conditions), and write the result
        back. Expressed purely in terms of :meth:`read_file` and
        :meth:`write_file`, so backends never need to override it.
        """
        content = await self.read_file(path)
        new_content = apply_edit(
            content, old_string, new_string, replace_all=replace_all
        )
        await self.write_file(path, new_content)

    async def __aenter__(self) -> "Sandbox":
        """Enter the async context manager: calls :meth:`start`."""
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager: calls :meth:`stop`."""
        await self.stop()
