"""Harbor-environment sandbox adapter (DESIGN.md §4.13, external benchmarks).

``HarborSandbox`` lets the agent loop run inside a Harbor (Terminal-Bench
2.0) task container by wrapping the ``environment`` object Harbor hands a
custom agent. The wrap is **duck-typed on purpose**: this module never
imports ``harbor`` — it only calls ``environment.exec(command, cwd=None,
env=None, timeout_sec=None, user=None)`` and reads ``stdout``/``stderr``/
``return_code`` off the result — so it stays importable (and testable) in
our own venv, where Harbor is not installed. All Harbor imports live in
:mod:`harness.integrations.harbor_agent` instead.

Lifecycle: **Harbor owns the container.** The trial runner starts the task
environment before our agent's ``run()`` is called and tears it down after
grading, so :meth:`HarborSandbox.start` only detects the workspace root
(no container is created) and :meth:`HarborSandbox.stop` is a no-op —
stopping the environment from here would yank the container out from under
Harbor's verifier.

File operations are implemented over ``exec``, shipping content as base64
in both directions so arbitrary text (newlines, quotes, unicode) survives
the shell without any quoting games; see :meth:`HarborSandbox.read_file` /
:meth:`HarborSandbox.write_file`. :meth:`~harness.sandbox.base.Sandbox.edit_file`
is inherited from the base class, so the exact-match/uniqueness edit
semantics are identical to every other backend.

Path jail note: unlike the host-side backends, this adapter cannot resolve
symlinks before validating a path (the filesystem lives in the container),
so the workspace jail here is **lexical** — absolute paths outside the
workspace root and any ``..`` component are rejected with
:class:`~harness.sandbox.base.SandboxPathError` before any command is sent,
but a symlink inside the workspace pointing outside it is not caught. That
is acceptable here because the Harbor task container is itself the
isolation boundary: everything the agent can reach through a symlink is
still inside the benchmark's own sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import shlex
import warnings
from pathlib import PurePosixPath
from typing import Any

from harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxPathError,
    truncate_output,
)

__all__ = ["DEFAULT_WORKSPACE_ROOT", "HarborSandbox"]

#: Fallback workspace root when ``pwd`` detection fails. Harbor task images
#: conventionally set their workdir to ``/app``.
DEFAULT_WORKSPACE_ROOT = "/app"

#: Timeout (seconds) for internal plumbing commands (``pwd``, file ops).
_INTERNAL_TIMEOUT = 120

#: Distinctive exit codes used by the read-command probe so shell failures
#: (which favor small codes like 1/2) can't be mistaken for our sentinels.
_EXIT_IS_DIRECTORY = 65
_EXIT_NOT_FOUND = 66

#: GNU ``timeout``'s exit code for a killed command. Some Harbor
#: environment providers signal an exec timeout through the return code
#: instead of raising; 124 is the only recognizable convention for that.
_TIMEOUT_EXIT_CODE = 124


def _read_command(resolved: str) -> str:
    """Shell command that base64-encodes the file at ``resolved``.

    Exits :data:`_EXIT_IS_DIRECTORY` for a directory and
    :data:`_EXIT_NOT_FOUND` for a missing path, so :meth:`HarborSandbox.read_file`
    can raise the same error shapes as the host-side backends.
    """
    quoted = shlex.quote(resolved)
    return (
        f"if [ -d {quoted} ]; then exit {_EXIT_IS_DIRECTORY}; "
        f"elif [ ! -e {quoted} ]; then exit {_EXIT_NOT_FOUND}; "
        f"else base64 < {quoted}; fi"
    )


def _write_command(resolved: str, parent: str, encoded: str) -> str:
    """Shell command that decodes ``encoded`` (base64) into ``resolved``.

    The content travels base64-encoded inside ordinary shell quoting — the
    base64 alphabet contains no quote characters, so ``shlex.quote`` around
    it is unconditionally safe regardless of what the original text held.
    ``mkdir -p`` mirrors the other backends' create-parents contract. The
    encoded payload rides the command line itself, which bounds single
    writes at the container's ARG_MAX (typically ≥2 MB) — plenty for the
    config/code files these tools move.
    """
    return (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(resolved)}"
    )


def _looks_like_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` is an exec-timeout signal from a Harbor environment.

    Harbor 0.20.0's Docker provider raises ``RuntimeError("Command timed
    out after N seconds")``; other providers may let ``asyncio.TimeoutError``
    escape. Matching on the message keeps this defensive across providers
    without importing any Harbor exception types.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


class HarborSandbox(Sandbox):
    """A :class:`~harness.sandbox.base.Sandbox` over a Harbor environment.

    Parameters
    ----------
    environment:
        A Harbor ``BaseEnvironment``-shaped object (duck-typed; see module
        docstring). Only its ``exec`` coroutine is used.
    workspace_root:
        Absolute in-container path all relative file paths are rooted at.
        When ``None`` (the default), :meth:`start` detects it by running
        ``pwd`` in the container, falling back to
        :data:`DEFAULT_WORKSPACE_ROOT` with a :class:`UserWarning`.
    """

    def __init__(self, environment: Any, workspace_root: str | None = None) -> None:
        self._environment = environment
        self._workspace_root = workspace_root
        self._started = workspace_root is not None

    @property
    def workspace_root(self) -> str | None:
        """The in-container workspace root (``None`` until :meth:`start`)."""
        return self._workspace_root

    async def start(self) -> None:
        """Detect the workspace root; the container itself is Harbor's job.

        Idempotent. When no ``workspace_root`` was given, runs ``pwd`` in
        the environment (whose working directory is the task's workdir) and
        uses its output; any failure — nonzero exit, empty output, or an
        exception from ``exec`` — falls back to
        :data:`DEFAULT_WORKSPACE_ROOT` with a :class:`UserWarning` rather
        than aborting the trial.
        """
        if self._started:
            return
        detected = ""
        try:
            result = await self._environment.exec(
                "pwd", timeout_sec=_INTERNAL_TIMEOUT
            )
            if result.return_code == 0:
                detected = (result.stdout or "").strip()
        except Exception:
            detected = ""
        if not detected:
            warnings.warn(
                "could not detect the Harbor environment's working directory "
                f"via 'pwd'; falling back to {DEFAULT_WORKSPACE_ROOT!r}",
                UserWarning,
                stacklevel=2,
            )
            detected = DEFAULT_WORKSPACE_ROOT
        self._workspace_root = detected
        self._started = True

    async def stop(self) -> None:
        """No-op: Harbor owns the container lifecycle.

        The trial runner started the task environment before our agent ran
        and must keep it alive afterwards for the verifier; stopping it
        here would break grading. Idempotent by construction.
        """
        return None

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        """Run ``command`` in the Harbor environment.

        Delegates to ``environment.exec(command, timeout_sec=...)`` where
        ``timeout_sec`` is ``max(1, ceil(timeout))`` — Harbor's exec takes
        whole seconds and its Docker provider treats ``timeout_sec=0`` as
        *no timeout at all* (``if timeout_sec:`` guard), so a fractional
        request like ``0.5`` must round **up** to 1 rather than truncate
        to an unbounded command; rounding up preserves "at least this
        long" semantics for every positive requested timeout. The result
        maps onto our :class:`~harness.sandbox.base.ExecResult`:
        ``None`` streams become ``""``, both streams get the standard
        :data:`~harness.sandbox.base.MAX_OUTPUT_BYTES` truncation, and
        timeouts are detected defensively on **both** paths Harbor
        providers use — an exception (Docker raises ``RuntimeError`` with a
        "timed out" message; others may raise ``asyncio.TimeoutError``) and
        a returned exit code of 124 (the GNU ``timeout`` convention). In
        either case the result has ``timed_out=True`` and ``exit_code=-1``
        per the base contract; a command that genuinely exits 124 on its
        own is indistinguishable from a provider-signaled timeout, which is
        the safe direction to be wrong in.
        """
        await self.start()
        timeout_sec = max(1, math.ceil(timeout))
        try:
            result = await self._environment.exec(
                command, timeout_sec=timeout_sec
            )
        except Exception as exc:
            if _looks_like_timeout(exc):
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"command timed out after {timeout_sec} seconds",
                    timed_out=True,
                )
            raise
        timed_out = result.return_code == _TIMEOUT_EXIT_CODE
        return ExecResult(
            exit_code=-1 if timed_out else result.return_code,
            stdout=truncate_output(
                (result.stdout or "").encode("utf-8"), stream_name="stdout"
            ),
            stderr=truncate_output(
                (result.stderr or "").encode("utf-8"), stream_name="stderr"
            ),
            timed_out=timed_out,
        )

    async def read_file(self, path: str) -> str:
        """Read the text file at ``path`` (relative to the workspace root).

        The content is base64-encoded *in the container* and decoded host
        side, so arbitrary bytes survive the shell round trip; this
        deliberately bypasses :meth:`exec`'s output truncation (that
        contract is for command output shown to the model, not file
        contents). Raises the same :class:`~harness.sandbox.base.SandboxError`
        shapes as :class:`~harness.sandbox.local.LocalSandbox` for a
        missing path or a directory.
        """
        await self.start()
        resolved = self._resolve(path)
        result = await self._raw_exec(_read_command(resolved))
        if result.return_code == _EXIT_NOT_FOUND:
            raise SandboxError(f"file not found: {path}")
        if result.return_code == _EXIT_IS_DIRECTORY:
            raise SandboxError(f"path is a directory, not a file: {path}")
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SandboxError(
                f"failed to read {path}: exit {result.return_code}"
                + (f": {detail}" if detail else "")
            )
        encoded = "".join((result.stdout or "").split())
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SandboxError(
                f"failed to read {path}: environment returned invalid "
                "base64 output"
            ) from exc
        return data.decode("utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path``, creating parent directories.

        The content travels base64-encoded through the shell (see
        :func:`_write_command`); no raw text ever needs quoting.
        """
        await self.start()
        resolved = self._resolve(path)
        parent = str(PurePosixPath(resolved).parent)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        result = await self._raw_exec(_write_command(resolved, parent, encoded))
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SandboxError(
                f"failed to write {path}: exit {result.return_code}"
                + (f": {detail}" if detail else "")
            )

    # -- internals -----------------------------------------------------------

    async def _raw_exec(self, command: str) -> Any:
        """``environment.exec`` for internal file-op plumbing, untruncated.

        A timeout here (exception path, see :func:`_looks_like_timeout`) is
        surfaced as a :class:`~harness.sandbox.base.SandboxError` — file
        operations have no ``timed_out`` channel in their contract.
        """
        try:
            return await self._environment.exec(
                command, timeout_sec=_INTERNAL_TIMEOUT
            )
        except Exception as exc:
            if _looks_like_timeout(exc):
                raise SandboxError(
                    f"file operation timed out after {_INTERNAL_TIMEOUT} seconds"
                ) from exc
            raise

    def _resolve(self, path: str) -> str:
        """Lexically resolve ``path`` under the workspace root.

        Pure-posix join; raises :class:`~harness.sandbox.base.SandboxPathError`
        for an empty path, any ``..`` component, or an absolute path
        outside the workspace root — all **before** any command is sent to
        the container. Absolute paths *inside* the root are accepted
        (Harbor task instructions frequently name files by absolute
        in-container path). Symlinks cannot be resolved host-side, so this
        jail is lexical only — see the module docstring for why that is
        acceptable here.
        """
        if not path:
            raise SandboxPathError("path must not be empty")
        root = PurePosixPath(self._workspace_root or DEFAULT_WORKSPACE_ROOT)
        candidate = PurePosixPath(path)
        if ".." in candidate.parts:
            raise SandboxPathError(
                f"path {path!r} contains '..' components; the Harbor "
                "sandbox jail is lexical, so traversal is rejected outright"
            )
        if candidate.is_absolute():
            if not candidate.is_relative_to(root):
                raise SandboxPathError(
                    f"path {path!r} escapes the sandbox workspace root ({root})"
                )
            return str(candidate)
        return str(root / candidate)
