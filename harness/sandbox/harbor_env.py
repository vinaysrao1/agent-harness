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

Timeout note: Harbor's exec timeout kills only the **host-side** client
(``docker compose exec``), and Docker does not propagate that disconnect to
the process tree running inside the container — the command keeps running
and keeps writing files under the agent. Because this harness *manufactures*
timeouts (the deadline caps exec windows), that orphan is routine, not
exceptional. Every agent command is therefore wrapped so the container-side
shell records its own pid, and on a timeout a follow-up exec kills that
process group; see :func:`_wrap_with_sentinel` and :func:`_cleanup_command`.

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
import logging
import math
import shlex
import uuid
import warnings
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxPathError,
    WriteMode,
    truncate_output,
)

logger = logging.getLogger(__name__)

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

#: Directory the per-exec pid sentinel is written to. ``/tmp`` is writable
#: in every Harbor task image we have seen; when it is not, the write is
#: silenced and the cleanup simply finds nothing (see
#: :func:`_wrap_with_sentinel`).
_CLEANUP_SENTINEL_DIR = "/tmp"

#: Seconds between the process group's ``SIGTERM`` and its ``SIGKILL``.
#: Long enough for a shell to flush and reap, short enough that a timed-out
#: exec doesn't cost the agent a noticeable extra pause.
_CLEANUP_GRACE_SECONDS = 1

#: Upper bound on the cleanup exec itself. It sends two signals and sleeps
#: :data:`_CLEANUP_GRACE_SECONDS`; anything beyond that means the
#: environment is unhealthy, and a timeout must not be allowed to block on
#: it.
#:
#: The bound is not free-floating. :meth:`HarborSandbox.exec` awaits the
#: cleanup *before* returning the timeout result, so every second it spends
#: is taken out of the wall-clock the deadline held back for the landing
#: turn — ``harness.deadline.LANDING_ALLOWANCE_MIN`` (15s) on a run whose
#: observed call latency sits at the floor. At the previous value of 10 a
#: wedged environment could eat ~11 of those 15 seconds; at 5 the worst
#: case is ~6, leaving the landing call the majority of its reserve. It is
#: not imported from :mod:`harness.deadline` (the sandbox layer knows
#: nothing about deadlines); ``tests/test_harbor_sandbox.py`` pins the
#: relation between the two constants instead.
_CLEANUP_TIMEOUT = 5

#: Marker the cleanup command prints when it actually found a pid and
#: signalled it, so :meth:`HarborSandbox._cleanup` can tell "the kill
#: landed" from "there was nothing to kill" (unwritable ``/tmp``, container
#: already gone). Never seen by the agent: it rides the cleanup exec's own
#: stdout, which is discarded.
_CLEANUP_KILLED_MARKER = "harness-cleanup-killed"

#: Most stale sentinels one exec's prefix sweeps (see
#: :func:`_wrap_with_sentinel`). Steady state is one — each exec retires
#: exactly one token — so this only bounds the prefix when a burst of
#: concurrent execs retires several at once.
_SENTINEL_SWEEP_MAX = 8


def _sentinel_path(token: str) -> str:
    """Path of the pid sentinel for one exec, keyed by ``token``."""
    return f"{_CLEANUP_SENTINEL_DIR}/.harness-exec-{token}.pid"


def _wrap_with_sentinel(
    command: str, token: str, retired: Sequence[str] = ()
) -> str:
    """``command`` prefixed with a statement recording the shell's own pid.

    The prefix is a **separate statement**, never chained with ``&&``: the
    agent's command runs whatever the sentinel write did, and the command
    list's exit status is still the agent command's own. That keeps
    ``set -e``, heredocs, pipelines, trailing comments and the exit code
    byte-identical to running the command unwrapped — the only observable
    difference is one small file in :data:`_CLEANUP_SENTINEL_DIR`.

    ``2>/dev/null`` comes **before** ``>`` and not after it, which is the
    whole reason the write is actually silent. A shell applies redirections
    left to right and reports a failed *setup* of one on whatever stderr it
    has at that instant: with ``> FILE 2>/dev/null`` the ``> FILE`` failure
    (read-only ``/tmp``, ``ENOSPC`` — precisely the disk-filling task that
    motivated this wrapper) is announced before stderr has been redirected,
    so a line naming an internal harness file lands in the agent's
    ``stderr`` on every exec. With the order below the same failure is
    silent, and the exit status and stdout are unchanged either way.

    The recorded ``$$`` is the exec'd shell's pid, which in a Harbor
    container exec is also its process-group and session id, so
    :func:`_cleanup_command` can signal the whole tree with ``kill -- -PGID``
    without needing ``ps`` or ``pkill`` (neither of which is installed in
    the slim images these tasks are built on; ``kill`` is a bash builtin and
    is always available).

    A sentinel outlives the exec that wrote it: the timeout path removes it
    in :func:`_cleanup_command`, and every other exec is retired to
    :attr:`HarborSandbox._retired` and swept by the ``rm -f`` in a *later*
    exec's prefix — never by a statement appended after the agent's command,
    which a trailing comment or an open heredoc would swallow, and never by
    a glob, which would delete the sentinel of a concurrently running exec.
    Sweeping in the prefix costs no extra round trip and keeps a trial's
    footprint in the task's ``/tmp`` at roughly one file rather than the one
    per exec (hundreds, over a trial) that a harness which must stay
    task-agnostic should not be leaving in a graded filesystem.
    """
    prefix = f"echo $$ 2>/dev/null > {shlex.quote(_sentinel_path(token))}"
    if retired:
        stale = " ".join(shlex.quote(_sentinel_path(t)) for t in retired)
        prefix += f"; rm -f {stale} 2>/dev/null"
    return f"{prefix}; {command}"


def _cleanup_command(token: str) -> str:
    """Shell command that kills the process group recorded by ``token``.

    ``SIGTERM`` first, then ``SIGKILL`` after :data:`_CLEANUP_GRACE_SECONDS`,
    addressed to the *negated* pid so the whole process group dies with the
    shell rather than just the shell. Every step is guarded: a missing or
    empty sentinel (``/tmp`` unwritable, provider killed the tree itself)
    short-circuits, and signalling an already-dead group is silenced. The
    sentinel is removed either way.

    On the signalling path it prints :data:`_CLEANUP_KILLED_MARKER`, which
    is what makes the repair falsifiable: without it every outcome — group
    killed, sentinel never written, container already gone — returns 0 and
    prints nothing, and the run's logs cannot say whether the kill ever
    landed. The marker goes to the cleanup exec's own stdout, which the
    caller discards after reading it; the agent never sees it.

    Known limitation, shared with :mod:`harness.sandbox.local`: a child that
    calls ``setsid`` for itself leaves the group and survives this.
    """
    quoted = shlex.quote(_sentinel_path(token))
    return (
        f"P=$(cat {quoted} 2>/dev/null); "
        f'[ -n "$P" ] && {{ kill -TERM -"$P" 2>/dev/null; '
        f"sleep {_CLEANUP_GRACE_SECONDS}; "
        f'kill -KILL -"$P" 2>/dev/null; '
        f"echo {_CLEANUP_KILLED_MARKER}; }}; "
        f"rm -f {quoted}"
    )


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


def _write_command(
    resolved: str, parent: str, encoded: str, *, mode: WriteMode = "overwrite"
) -> str:
    """Shell command that decodes ``encoded`` (base64) into ``resolved``.

    The content travels base64-encoded inside ordinary shell quoting — the
    base64 alphabet contains no quote characters, so ``shlex.quote`` around
    it is unconditionally safe regardless of what the original text held.
    ``mkdir -p`` mirrors the other backends' create-parents contract. The
    encoded payload rides the command line itself, which bounds single
    writes at the container's ARG_MAX (typically ≥2 MB) — plenty for the
    config/code files these tools move.

    ``mode="overwrite"`` (the default) redirects with ``>``, replacing
    ``resolved``'s contents; ``mode="append"`` redirects with ``>>``,
    creating the file if it doesn't exist and adding to it otherwise (see
    :data:`~harness.sandbox.base.WriteMode`).
    """
    redirect = ">>" if mode == "append" else ">"
    return (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d {redirect} {shlex.quote(resolved)}"
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
        # Tokens whose exec has returned and whose sentinel is therefore
        # dead: the next exec's prefix sweeps them (see
        # :func:`_wrap_with_sentinel`). Only *retired* tokens go in here,
        # never in-flight ones, so a concurrent subagent exec can't have its
        # live sentinel deleted out from under it.
        self._retired: list[str] = []

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

        A timeout here kills only Harbor's host-side exec client; the
        in-container process tree survives it and keeps writing. So the
        command is dispatched wrapped by :func:`_wrap_with_sentinel`, and
        **both** timeout paths issue the follow-up kill in :meth:`_cleanup`
        before returning, making ``timed_out=True`` mean what the base
        contract says it means — the command was killed — on this backend
        too. The returned :class:`~harness.sandbox.base.ExecResult` is
        unaffected by whether that cleanup succeeded.

        The same wrapper also sweeps the sentinels of already-returned
        execs, so the harness leaves ~one dotfile in the task container's
        ``/tmp`` rather than one per exec; see :func:`_wrap_with_sentinel`.
        """
        await self.start()
        timeout_sec = max(1, math.ceil(timeout))
        # uuid4 per exec, never per sandbox: subagents share the lead's
        # environment, so two execs can be in flight at once and a reused
        # sentinel would have one command kill the other's process group.
        token = uuid.uuid4().hex
        # Claim the stale sentinels this exec will sweep. Claiming (rather
        # than reading) is what makes it concurrency-safe: a second exec
        # starting before this one returns sees an empty head and sweeps
        # nothing, instead of both emitting the same `rm`.
        sweep = self._retired[:_SENTINEL_SWEEP_MAX]
        del self._retired[: len(sweep)]
        try:
            result = await self._environment.exec(
                _wrap_with_sentinel(command, token, sweep),
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            if _looks_like_timeout(exc):
                # The prefix ran (the command was launched and outlived its
                # window), so the sweep landed; the cleanup removes this
                # exec's own sentinel.
                await self._cleanup(token)
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"command timed out after {timeout_sec} seconds",
                    timed_out=True,
                )
            # Transport-level failure: whether the prefix ever ran is
            # unknown, so the claimed sweep goes back on the queue for a
            # later exec. This exec's own sentinel joins it — nothing will
            # clean up after a command whose result we never saw.
            self._retired[:0] = [*sweep, token]
            raise
        timed_out = result.return_code == _TIMEOUT_EXIT_CODE
        if timed_out:
            # 124 usually means the provider ran an in-container `timeout`,
            # which already killed the tree; the cleanup then finds a dead
            # group and no-ops. Cheap enough to do unconditionally rather
            # than guess which provider we are talking to.
            await self._cleanup(token)
        else:
            self._retired.append(token)
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

    async def write_file(
        self, path: str, content: str, *, mode: WriteMode = "overwrite"
    ) -> None:
        """Write ``content`` to ``path``, creating parent directories.

        The content travels base64-encoded through the shell (see
        :func:`_write_command`); no raw text ever needs quoting.
        ``mode="append"`` adds to an existing file (creating it if absent)
        instead of replacing its contents; see
        :data:`~harness.sandbox.base.WriteMode`.
        """
        await self.start()
        resolved = self._resolve(path)
        parent = str(PurePosixPath(resolved).parent)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        result = await self._raw_exec(
            _write_command(resolved, parent, encoded, mode=mode)
        )
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SandboxError(
                f"failed to write {path}: exit {result.return_code}"
                + (f": {detail}" if detail else "")
            )

    # -- internals -----------------------------------------------------------

    async def _cleanup(self, token: str) -> None:
        """Best-effort kill of the process group left behind by a timeout.

        Deliberately swallows **everything**: the caller is already
        returning a timeout result, and a failed cleanup (dead environment,
        cleanup exec itself timing out) must never turn that timeout into an
        exception the agent loop has to handle. The orphan is a
        best-effort repair, not a guarantee — which is why the ``bash`` tool
        also tells the model its output may be incomplete.

        Swallowed is not the same as unobserved. "Best effort" is only an
        honest description if a run can afterwards say how often the effort
        worked, so each of the three outcomes is logged: the group was
        signalled (:data:`_CLEANUP_KILLED_MARKER` on the cleanup's stdout),
        there was no pid to signal (unwritable ``/tmp``, tree already gone),
        or the cleanup exec itself failed. The two anomalies log at
        ``WARNING`` so they surface without any logging configuration; the
        healthy path logs at ``DEBUG``. Logging never changes what the
        caller returns.
        """
        try:
            result = await self._environment.exec(
                _cleanup_command(token), timeout_sec=_CLEANUP_TIMEOUT
            )
        except Exception as exc:
            logger.warning(
                "harbor exec cleanup for %s failed; the timed-out command "
                "may still be running in the container: %r",
                token,
                exc,
            )
            return
        if _CLEANUP_KILLED_MARKER in (getattr(result, "stdout", "") or ""):
            logger.debug(
                "harbor exec cleanup for %s signalled the process group "
                "(exit %s)",
                token,
                getattr(result, "return_code", None),
            )
        else:
            logger.warning(
                "harbor exec cleanup for %s found no pid sentinel (exit "
                "%s); nothing was killed",
                token,
                getattr(result, "return_code", None),
            )

    async def _raw_exec(self, command: str) -> Any:
        """``environment.exec`` for internal file-op plumbing, untruncated.

        Deliberately **not** wrapped by :func:`_wrap_with_sentinel`: these
        are our own base64 file-op commands, bounded by
        :data:`_INTERNAL_TIMEOUT` and parsed by the round-trip protocol —
        not agent commands that can spawn a long-lived tree.

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
