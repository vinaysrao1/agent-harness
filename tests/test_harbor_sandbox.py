"""Unit tests for harness.sandbox.harbor_env (the Harbor bridge sandbox).

No ``harbor`` imports anywhere (Harbor lives only in its own uv-tool venv):
:class:`StubEnvironment` duck-types the one method
:class:`~harness.sandbox.harbor_env.HarborSandbox` uses —
``exec(command, cwd=None, env=None, timeout_sec=None, user=None)`` returning
an object with ``stdout``/``stderr``/``return_code`` — and simulates an
in-container filesystem by *actually parsing* the base64 read/write
commands the sandbox emits and round-tripping content through the same
encoding, so the stub can't accidentally vouch for a broken protocol.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shlex
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from posixpath import dirname

import pytest

from harness.deadline import LANDING_ALLOWANCE_MIN
from harness.sandbox import harbor_env
from harness.sandbox.base import (
    MAX_OUTPUT_BYTES,
    SandboxError,
    SandboxPathError,
)
from harness.sandbox.harbor_env import (
    DEFAULT_WORKSPACE_ROOT,
    HarborSandbox,
    _CLEANUP_GRACE_SECONDS,
    _CLEANUP_TIMEOUT,
    _SENTINEL_SWEEP_MAX,
    _cleanup_command,
    _wrap_with_sentinel,
)


@dataclass
class StubExecResult:
    """Duck-type of Harbor's ``ExecResult`` (stdout/stderr may be None)."""

    return_code: int
    stdout: str | None = None
    stderr: str | None = None


#: Matches the sandbox's read command; all three path occurrences must be
#: the identical quoted string (backreference), mirroring the template.
_READ_RE = re.compile(
    r"^if \[ -d (?P<q>.+) \]; then exit 65; "
    r"elif \[ ! -e (?P=q) \]; then exit 66; "
    r"else base64 < (?P=q); fi$"
)

#: Matches the pid-sentinel prefix the sandbox wraps agent commands with,
#: including the optional sweep of already-retired sentinels. The stub
#: parses it rather than assuming it, so a malformed wrapper shows up as an
#: unscripted command instead of being silently tolerated. Note the
#: redirection order: ``2>/dev/null`` must precede ``>`` or a failed
#: sentinel write leaks onto the agent's stderr.
_SENTINEL_RE = re.compile(
    r"^echo \$\$ 2>/dev/null > /tmp/\.harness-exec-(?P<token>[0-9a-f]+)\.pid"
    r"(?:; rm -f (?P<sweep>[^;]*) 2>/dev/null)?; (?P<command>.*)$",
    re.DOTALL,
)

#: Matches the timeout cleanup exec, capturing the token whose process
#: group it signals.
_CLEANUP_RE = re.compile(
    r"^P=\$\(cat /tmp/\.harness-exec-(?P<token>[0-9a-f]+)\.pid 2>/dev/null\); "
    r'\[ -n "\$P" \] && \{ kill -TERM -"\$P" 2>/dev/null; '
    r'sleep \d+; kill -KILL -"\$P" 2>/dev/null; '
    r"echo harness-cleanup-killed; \}; "
    r"rm -f /tmp/\.harness-exec-(?P=token)\.pid$"
)

#: Pulls the token out of a sentinel path, for reading a prefix's sweep.
_TOKEN_RE = re.compile(r"/tmp/\.harness-exec-([0-9a-f]+)\.pid")


def unwrap(command: str) -> tuple[str | None, str]:
    """Split a dispatched command into ``(sentinel token, agent command)``.

    Returns ``(None, command)`` for anything not wrapped (file-op plumbing,
    ``pwd``, the cleanup exec itself).
    """
    match = _SENTINEL_RE.match(command)
    if match is None:
        return None, command
    return match.group("token"), match.group("command")


def swept(command: str) -> list[str]:
    """Tokens whose stale sentinels ``command``'s prefix removes."""
    match = _SENTINEL_RE.match(command)
    if match is None or not match.group("sweep"):
        return []
    return _TOKEN_RE.findall(match.group("sweep"))


class StubEnvironment:
    """Scriptable, recording stand-in for a Harbor environment.

    ``calls`` records every ``exec`` invocation *verbatim* — including the
    pid-sentinel wrapper — while ``script()`` pins the outcome (result or
    exception) for an exact **agent** command string, i.e. the wrapper is
    stripped before the lookup so scripting reads the way the caller thinks
    about it. Unscripted commands are interpreted: ``pwd`` reports
    ``workdir``; the sandbox's write command decodes its base64 payload into
    ``files``; the read command re-encodes from ``files`` (wrapped at 76
    columns, like real ``base64``, so the decoder must handle embedded
    newlines); a timeout cleanup appends its token to ``killed``; anything
    else succeeds silently with ``None`` streams.
    """

    def __init__(self, workdir: str = "/app") -> None:
        self.workdir = workdir
        self.calls: list[tuple[str, dict]] = []
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {workdir}
        self.killed: list[str] = []
        self._scripted: dict[str, StubExecResult | BaseException] = {}

    def script(self, command: str, outcome: StubExecResult | BaseException) -> None:
        """Pin the outcome of one exact command string (wrapper-agnostic)."""
        self._scripted[command] = outcome

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        timeout_sec: int | None = None,
        user: object = None,
    ) -> StubExecResult:
        self.calls.append(
            (
                command,
                {"cwd": cwd, "env": env, "timeout_sec": timeout_sec, "user": user},
            )
        )
        _, command = unwrap(command)
        if command in self._scripted:
            outcome = self._scripted[command]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        if command == "pwd":
            return StubExecResult(return_code=0, stdout=self.workdir + "\n")
        cleanup = _CLEANUP_RE.match(command)
        if cleanup is not None:
            self.killed.append(cleanup.group("token"))
            # A real cleanup that finds a live pid prints the marker the
            # sandbox reads to tell "killed" from "nothing to kill".
            return StubExecResult(
                return_code=0, stdout="harness-cleanup-killed\n"
            )
        write = self._try_write(command)
        if write is not None:
            return write
        read = self._try_read(command)
        if read is not None:
            return read
        return StubExecResult(return_code=0)

    def _try_write(self, command: str) -> StubExecResult | None:
        """Interpret the sandbox's mkdir-and-base64-decode write command.

        The redirect operator distinguishes overwrite (``>``) from append
        (``>>``); the stub honors it exactly like a real shell would, so it
        can't accidentally vouch for a broken ``mode`` plumbing.
        """
        tokens = shlex.split(command)
        shape = (
            len(tokens) == 12
            and tokens[0:2] == ["mkdir", "-p"]
            and tokens[3] == "&&"
            and tokens[4:6] == ["printf", "%s"]
            and tokens[7] == "|"
            and tokens[8:10] == ["base64", "-d"]
            and tokens[10] in (">", ">>")
        )
        if not shape:
            return None
        parent, encoded, redirect, path = tokens[2], tokens[6], tokens[10], tokens[11]
        # Honest round trip: decode exactly what would hit `base64 -d`.
        decoded = base64.b64decode(encoded)
        if redirect == ">>":
            self.files[path] = self.files.get(path, b"") + decoded
        else:
            self.files[path] = decoded
        while parent and parent not in self.dirs:
            self.dirs.add(parent)
            parent = dirname(parent)
        return StubExecResult(return_code=0)

    def _try_read(self, command: str) -> StubExecResult | None:
        """Interpret the sandbox's probing base64 read command."""
        match = _READ_RE.match(command)
        if match is None:
            return None
        path = shlex.split(match.group("q"))[0]
        if path in self.dirs:
            return StubExecResult(return_code=65)
        if path not in self.files:
            return StubExecResult(return_code=66)
        encoded = base64.b64encode(self.files[path]).decode("ascii")
        # Real `base64` wraps output at 76 columns; the decoder must cope.
        wrapped = "\n".join(textwrap.wrap(encoded, 76)) + "\n" if encoded else "\n"
        return StubExecResult(return_code=0, stdout=wrapped)


@pytest.fixture
def env() -> StubEnvironment:
    return StubEnvironment()


@pytest.fixture
async def sandbox(env: StubEnvironment) -> HarborSandbox:
    box = HarborSandbox(env)
    await box.start()
    return box


class TestWorkspaceDetection:
    async def test_detects_root_via_pwd(self, env: StubEnvironment):
        box = HarborSandbox(env)
        await box.start()
        assert box.workspace_root == "/app"
        assert env.calls[0][0] == "pwd"

    async def test_start_is_idempotent(self, env: StubEnvironment):
        box = HarborSandbox(env)
        await box.start()
        await box.start()
        assert [command for command, _ in env.calls] == ["pwd"]

    async def test_pwd_failure_falls_back_with_warning(self):
        env = StubEnvironment(workdir="/work")
        env.script("pwd", StubExecResult(return_code=1, stderr="boom"))
        box = HarborSandbox(env)
        with pytest.warns(UserWarning, match="falling back to '/app'"):
            await box.start()
        assert box.workspace_root == DEFAULT_WORKSPACE_ROOT

    async def test_pwd_exception_falls_back_with_warning(self, env: StubEnvironment):
        env.script("pwd", RuntimeError("connection lost"))
        box = HarborSandbox(env)
        with pytest.warns(UserWarning, match="could not detect"):
            await box.start()
        assert box.workspace_root == DEFAULT_WORKSPACE_ROOT

    async def test_explicit_root_skips_detection(self, env: StubEnvironment):
        box = HarborSandbox(env, workspace_root="/custom")
        await box.start()
        assert box.workspace_root == "/custom"
        assert env.calls == []

    async def test_stop_is_a_noop(self, sandbox: HarborSandbox, env: StubEnvironment):
        # Harbor owns the container: stop must not touch the environment.
        before = list(env.calls)
        await sandbox.stop()
        await sandbox.stop()
        assert env.calls == before


class TestExecMapping:
    async def test_delegates_with_ceil_timeout(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Fractional timeouts round UP (Harbor takes whole seconds); a
        # request of 5.9s must not be truncated down to 5.
        env.script(
            "echo hi", StubExecResult(return_code=0, stdout="hi\n", stderr="")
        )
        result = await sandbox.exec("echo hi", timeout=5.9)
        assert result.exit_code == 0
        assert result.stdout == "hi\n"
        assert result.stderr == ""
        assert result.timed_out is False
        command, kwargs = env.calls[-1]
        assert unwrap(command)[1] == "echo hi"
        assert kwargs["timeout_sec"] == 6

    @pytest.mark.parametrize("requested", [0.1, 0.5, 0.999, 1.0])
    async def test_subsecond_timeout_is_still_enforced(
        self, sandbox: HarborSandbox, env: StubEnvironment, requested: float
    ):
        # Regression: int(0.5) == 0, and Harbor's Docker provider treats
        # timeout_sec=0 as NO timeout ('if timeout_sec:'), turning a tight
        # sub-second timeout into an unbounded command. Every positive
        # request must map to at least 1 second.
        env.script("sleep 5", StubExecResult(return_code=0, stdout="ok"))
        await sandbox.exec("sleep 5", timeout=requested)
        _, kwargs = env.calls[-1]
        assert kwargs["timeout_sec"] == 1

    async def test_timeout_message_reports_enforced_seconds(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("sleep 60", RuntimeError("Command timed out after 1 seconds"))
        result = await sandbox.exec("sleep 60", timeout=0.5)
        assert result.timed_out is True
        assert "timed out after 1 seconds" in result.stderr

    async def test_none_streams_become_empty_strings(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("true", StubExecResult(return_code=0, stdout=None, stderr=None))
        result = await sandbox.exec("true")
        assert result.stdout == ""
        assert result.stderr == ""

    async def test_nonzero_exit_passes_through(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("exit 7", StubExecResult(return_code=7, stderr="oops\n"))
        result = await sandbox.exec("exit 7")
        assert result.exit_code == 7
        assert result.stderr == "oops\n"
        assert result.timed_out is False

    async def test_timeout_runtimeerror_maps_to_timed_out(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Harbor 0.20.0's Docker provider raises exactly this shape.
        env.script("sleep 60", RuntimeError("Command timed out after 5 seconds"))
        result = await sandbox.exec("sleep 60", timeout=5)
        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr

    async def test_timeout_asyncio_error_maps_to_timed_out(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("sleep 60", asyncio.TimeoutError())
        result = await sandbox.exec("sleep 60", timeout=5)
        assert result.timed_out is True
        assert result.exit_code == -1

    async def test_return_code_124_maps_to_timed_out(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Defensive: a provider that signals timeout via the GNU timeout
        # exit-code convention instead of raising.
        env.script("sleep 60", StubExecResult(return_code=124))
        result = await sandbox.exec("sleep 60", timeout=5)
        assert result.timed_out is True
        assert result.exit_code == -1

    async def test_non_timeout_exception_propagates(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("boom", ValueError("kaboom"))
        with pytest.raises(ValueError, match="kaboom"):
            await sandbox.exec("boom")

    async def test_output_truncated_at_limit(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        big = "x" * (MAX_OUTPUT_BYTES + 5000)
        env.script("yes", StubExecResult(return_code=0, stdout=big, stderr=big))
        result = await sandbox.exec("yes")
        assert f"stdout truncated at {MAX_OUTPUT_BYTES} bytes" in result.stdout
        assert f"stderr truncated at {MAX_OUTPUT_BYTES} bytes" in result.stderr
        assert len(result.stdout) < len(big)


class ExplodingCleanupEnvironment(StubEnvironment):
    """A stub whose *cleanup* exec fails, everything else behaving normally."""

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        timeout_sec: int | None = None,
        user: object = None,
    ) -> StubExecResult:
        if _CLEANUP_RE.match(command):
            self.calls.append((command, {"timeout_sec": timeout_sec}))
            raise RuntimeError("environment is gone")
        return await super().exec(command, cwd, env, timeout_sec, user)


class TestTimeoutKillsTheContainerProcessTree:
    """A timed-out exec must not leave the command running in the container.

    Harbor's timeout kills the host-side ``docker compose exec`` client, and
    Docker does **not** propagate that to the exec'd process tree: measured
    on this host, a killed client left its command running and still
    appending to its output file eleven seconds later. Since the deadline
    manufactures timeouts routinely, that orphan keeps mutating files under
    the agent — the observed shape being a download killed mid-flight and
    then untarred into ``gzip: stdin: unexpected end of file``.
    """

    async def test_exec_wraps_the_command_with_a_sentinel(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("make -j 6", StubExecResult(return_code=0))
        await sandbox.exec("make -j 6")
        command, _ = env.calls[-1]
        token, inner = unwrap(command)
        assert token is not None
        # The agent's command is carried verbatim, as the *last* statement.
        assert inner == "make -j 6"
        assert command.endswith("; make -j 6")
        assert command.count(token) == 1

    def test_the_prefix_is_a_separate_statement_not_a_conjunction(self):
        # `echo $$ > f && cmd` would make the whole command conditional on a
        # writable /tmp and would change the exit code; `;` cannot.
        wrapped = _wrap_with_sentinel("false", "deadbeef")
        assert "&&" not in wrapped
        assert wrapped.startswith("echo $$ 2>/dev/null > ")
        assert wrapped.endswith("; false")

    def test_the_wrapper_does_not_disturb_multi_line_commands(self):
        heredoc = "cat <<'EOF' > f.txt\nline one\nEOF"
        wrapped = _wrap_with_sentinel(heredoc, "cafe1234")
        assert wrapped.endswith("; " + heredoc)

    def test_cleanup_command_signals_the_group_then_kills_it(self):
        command = _cleanup_command("abc123")
        # Negated pid = the process group, which is what reaches children;
        # `ps`/`pkill` are absent from the slim task images, `kill` is a
        # bash builtin and always present.
        assert 'kill -TERM -"$P"' in command
        assert 'kill -KILL -"$P"' in command
        assert "ps " not in command and "pkill" not in command
        assert command.endswith("rm -f /tmp/.harness-exec-abc123.pid")
        # Only the signalling branch prints the marker, so its presence
        # distinguishes "the kill landed" from "there was nothing to kill".
        assert "; echo harness-cleanup-killed; }" in command

    async def test_timeout_issues_an_in_container_cleanup(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script(
            "./get_cifar10.sh",
            RuntimeError("Command timed out after 240 seconds"),
        )
        before = len(env.calls)
        result = await sandbox.exec("./get_cifar10.sh", timeout=240)
        # The timeout result itself is untouched by the repair.
        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out after 240 seconds" in result.stderr
        # ... and a second exec went out to kill what was left behind.
        assert len(env.calls) == before + 2
        launched, cleanup = env.calls[-2][0], env.calls[-1][0]
        token = unwrap(launched)[0]
        assert 'kill -TERM -"$P"' in cleanup
        assert 'kill -KILL -"$P"' in cleanup
        assert token in cleanup
        assert env.killed == [token]
        # Bounded: a wedged environment must not stall the timeout path.
        assert env.calls[-1][1]["timeout_sec"] == _CLEANUP_TIMEOUT == 5

    async def test_timeout_by_exit_code_124_also_cleans_up(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # The other detection path: a provider signalling timeout through
        # GNU timeout's exit code rather than raising.
        env.script("sleep 600", StubExecResult(return_code=124))
        result = await sandbox.exec("sleep 600", timeout=5)
        assert result.timed_out is True
        token = unwrap(env.calls[-2][0])[0]
        assert env.killed == [token]

    async def test_cleanup_failure_never_masks_the_timeout(self):
        # A dead environment must not turn a timeout into an exception the
        # agent loop has to handle: the repair is best effort, the result
        # is not.
        env = ExplodingCleanupEnvironment()
        box = HarborSandbox(env)
        await box.start()
        env.script("sleep 600", asyncio.TimeoutError())
        result = await box.exec("sleep 600", timeout=30)
        assert result.timed_out is True
        assert result.exit_code == -1
        assert env.killed == []  # the cleanup really did fail
        assert _CLEANUP_RE.match(env.calls[-1][0]) is not None

    async def test_successful_exec_issues_no_cleanup(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("echo hi", StubExecResult(return_code=0, stdout="hi\n"))
        before = len(env.calls)
        await sandbox.exec("echo hi")
        assert len(env.calls) == before + 1
        assert env.killed == []

    async def test_nonzero_exit_issues_no_cleanup(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Failure is not a timeout: nothing was killed, nothing to clean.
        env.script("exit 7", StubExecResult(return_code=7))
        before = len(env.calls)
        await sandbox.exec("exit 7")
        assert len(env.calls) == before + 1

    async def test_file_ops_are_not_wrapped(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # The file-op plumbing carries our own base64 protocol commands,
        # not agent commands: wrapping them would break the round trip the
        # stub parses, and there is no long-lived tree to orphan.
        await sandbox.write_file("notes.txt", "hello")
        await sandbox.read_file("notes.txt")
        assert env.calls  # guard against vacuous truth
        for command, _ in env.calls:
            assert _SENTINEL_RE.match(command) is None

    async def test_concurrent_execs_get_distinct_tokens(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Subagents share the lead's environment, so two execs really can be
        # in flight at once; a shared sentinel would have one command's
        # timeout kill the other's process group.
        env.script("first", StubExecResult(return_code=0))
        env.script("second", StubExecResult(return_code=0))
        await asyncio.gather(sandbox.exec("first"), sandbox.exec("second"))
        tokens = {
            unwrap(command)[0]
            for command, _ in env.calls
            if _SENTINEL_RE.match(command)
        }
        assert len(tokens) == 2

    def test_the_cleanup_fits_inside_the_reserved_landing_allowance(self):
        # The cleanup is awaited *before* exec() returns the timeout, so it
        # is spent out of the wall-clock the deadline held back for the
        # landing model call. LANDING_ALLOWANCE_MIN is the smallest that
        # reserve ever gets; a cleanup bound at or above it would let a
        # wedged environment consume the whole landing turn.
        assert _CLEANUP_TIMEOUT < LANDING_ALLOWANCE_MIN / 2
        # ... and still comfortably longer than the work it must do: two
        # signals around one sleep.
        assert _CLEANUP_TIMEOUT > _CLEANUP_GRACE_SECONDS


#: Real shells to exercise the wrapper against. ``sh`` is what Harbor's
#: Docker provider runs; ``bash`` is what the task images usually link it
#: to. Both are checked because redirection-error reporting is exactly the
#: kind of behaviour that differs between them.
#: ``or ["/bin/sh"]`` so a host missing both fails loudly instead of
#: collecting zero tests.
_SHELLS = [p for p in (shutil.which("sh"), shutil.which("bash")) if p] or ["/bin/sh"]


class TestTheSentinelWriteIsSilent:
    """A failed sentinel write must not appear on the agent's stderr.

    :func:`_wrap_with_sentinel` documents that when the sentinel directory
    is not writable "the write is silenced" and that "the only observable
    difference is one small file". Redirection *setup* failures are
    reported on the stderr the shell holds at that instant, so with
    ``> FILE 2>/dev/null`` the failure is announced before stderr has been
    redirected — one line naming an internal harness file, on every single
    exec, in exactly the ENOSPC/read-only-``/tmp`` conditions the wrapper
    exists to survive. These run the emitted string through a real shell.
    """

    @staticmethod
    def _unwritable_sentinel(tmp_path: Path, monkeypatch, token: str) -> str:
        """Point the sentinel at a path no user (root included) can write.

        A read-only *directory* would not stop root, which would make these
        tests silently vacuous in a container; a path that is itself a
        directory fails the redirection for everyone.
        """
        monkeypatch.setattr(harbor_env, "_CLEANUP_SENTINEL_DIR", str(tmp_path))
        path = harbor_env._sentinel_path(token)
        Path(path).mkdir()
        return path

    @pytest.mark.parametrize("shell", _SHELLS)
    @pytest.mark.parametrize(
        ("command", "rc", "out"),
        [
            ("echo RAN", 0, "RAN"),
            ("echo RAN; exit 7", 7, "RAN"),
            ("set -e; false", 1, ""),
        ],
    )
    def test_a_failed_write_leaks_nothing_and_changes_nothing(
        self, tmp_path: Path, monkeypatch, shell: str, command: str, rc: int, out: str
    ):
        path = self._unwritable_sentinel(tmp_path, monkeypatch, "abc123")
        # Guard against a vacuous test: the old order really does leak, so
        # this environment really does fail the write.
        leaky = f"echo $$ > {shlex.quote(path)} 2>/dev/null; {command}"
        control = subprocess.run(
            [shell, "-c", leaky], capture_output=True, text=True
        )
        assert control.stderr != ""

        proc = subprocess.run(
            [shell, "-c", _wrap_with_sentinel(command, "abc123")],
            capture_output=True,
            text=True,
        )
        assert proc.stderr == ""
        # The agent's command is untouched by the wrapper either way.
        assert proc.returncode == rc == control.returncode
        assert proc.stdout.strip() == out == control.stdout.strip()

    @pytest.mark.parametrize("shell", _SHELLS)
    def test_a_writable_sentinel_records_the_shells_own_pid(
        self, tmp_path: Path, monkeypatch, shell: str
    ):
        monkeypatch.setattr(harbor_env, "_CLEANUP_SENTINEL_DIR", str(tmp_path))
        path = Path(harbor_env._sentinel_path("cafe01"))
        proc = subprocess.run(
            [shell, "-c", _wrap_with_sentinel("echo RAN", "cafe01")],
            capture_output=True,
            text=True,
        )
        assert (proc.returncode, proc.stdout.strip(), proc.stderr) == (0, "RAN", "")
        assert int(path.read_text().strip()) > 0


class TestSentinelsDoNotAccumulate:
    """The harness must not leave one dotfile per exec in the task's /tmp.

    A trial issues hundreds of execs, and the container's filesystem is the
    graded artifact: any task or grader that enumerates ``/tmp`` would see
    harness state no other backend leaves behind. The sweep rides the next
    exec's prefix, so it costs no extra round trip, and it names the exact
    stale tokens rather than globbing — a glob would delete the live
    sentinel of a concurrently running subagent exec.
    """

    @pytest.mark.parametrize("shell", _SHELLS)
    def test_the_prefix_really_removes_stale_sentinels(
        self, tmp_path: Path, monkeypatch, shell: str
    ):
        monkeypatch.setattr(harbor_env, "_CLEANUP_SENTINEL_DIR", str(tmp_path))
        stale = [Path(harbor_env._sentinel_path(t)) for t in ("aa11", "bb22")]
        for path in stale:
            path.write_text("4242\n")
        live = Path(harbor_env._sentinel_path("cc33"))
        proc = subprocess.run(
            [shell, "-c", _wrap_with_sentinel("echo RAN", "cc33", ["aa11", "bb22"])],
            capture_output=True,
            text=True,
        )
        assert (proc.returncode, proc.stdout.strip(), proc.stderr) == (0, "RAN", "")
        assert [path.exists() for path in stale] == [False, False]
        assert int(live.read_text().strip()) > 0

    @pytest.mark.parametrize("shell", _SHELLS)
    def test_sweeping_a_missing_sentinel_is_silent(
        self, tmp_path: Path, monkeypatch, shell: str
    ):
        # The timeout path already removed its own sentinel; a sweep that
        # raced it must not print anything or change the exit code.
        monkeypatch.setattr(harbor_env, "_CLEANUP_SENTINEL_DIR", str(tmp_path))
        proc = subprocess.run(
            [shell, "-c", _wrap_with_sentinel("exit 3", "cc33", ["gone99"])],
            capture_output=True,
            text=True,
        )
        assert (proc.returncode, proc.stderr) == (3, "")

    async def test_the_next_exec_sweeps_the_previous_ones_sentinel(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        env.script("first", StubExecResult(return_code=0))
        env.script("second", StubExecResult(return_code=0))
        await sandbox.exec("first")
        first_token = unwrap(env.calls[-1][0])[0]
        assert swept(env.calls[-1][0]) == []  # nothing retired yet
        await sandbox.exec("second")
        assert swept(env.calls[-1][0]) == [first_token]
        # ... and it is swept once, not on every subsequent exec.
        env.script("third", StubExecResult(return_code=0))
        await sandbox.exec("third")
        second_token = unwrap(env.calls[-2][0])[0]
        assert swept(env.calls[-1][0]) == [second_token]

    async def test_a_timed_out_execs_sentinel_is_not_swept_twice(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # The cleanup exec removes it, so re-listing it in a later prefix
        # would be a pointless (and confusing) rm of a file already gone.
        env.script("sleep 600", StubExecResult(return_code=124))
        await sandbox.exec("sleep 600", timeout=5)
        timed_out_token = unwrap(env.calls[-2][0])[0]
        env.script("after", StubExecResult(return_code=0))
        await sandbox.exec("after")
        assert swept(env.calls[-1][0]) == []
        assert env.killed == [timed_out_token]

    async def test_a_live_sentinel_is_never_swept(self):
        # Subagents share the lead's environment: an exec that is still
        # running must never have its pid file deleted by a sibling, or the
        # sibling's timeout cleanup would find nothing to kill.
        environment = SlowEnvironment()
        box = HarborSandbox(environment, workspace_root="/app")
        await asyncio.gather(*(box.exec(f"cmd{i}") for i in range(3)))
        for call, _ in environment.calls:
            assert swept(call) == []  # all three were live the whole time
        assert len({unwrap(call)[0] for call, _ in environment.calls}) == 3

    async def test_the_sweep_is_bounded(self):
        environment = SlowEnvironment()
        box = HarborSandbox(environment, workspace_root="/app")
        count = _SENTINEL_SWEEP_MAX + 4
        await asyncio.gather(*(box.exec(f"cmd{i}") for i in range(count)))
        await box.exec("next")
        first = swept(environment.calls[-1][0])
        assert len(first) == _SENTINEL_SWEEP_MAX
        # The remainder is not dropped: it drains on the following exec,
        # together with the token "next" itself retired.
        await box.exec("later")
        second = swept(environment.calls[-1][0])
        assert len(second) == count - _SENTINEL_SWEEP_MAX + 1
        # Every sentinel is swept exactly once, and none twice.
        assert len(set(first + second)) == count + 1

    async def test_a_transport_failure_requeues_the_claimed_sweep(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # A command whose result we never saw may not even have run its
        # prefix, so the sentinels it was going to sweep must stay queued
        # rather than leak for the rest of the trial.
        env.script("first", StubExecResult(return_code=0))
        await sandbox.exec("first")
        first_token = unwrap(env.calls[-1][0])[0]
        env.script("boom", RuntimeError("connection reset"))
        with pytest.raises(RuntimeError):
            await sandbox.exec("boom")
        failed_token = unwrap(env.calls[-1][0])[0]
        env.script("after", StubExecResult(return_code=0))
        await sandbox.exec("after")
        assert swept(env.calls[-1][0]) == [first_token, failed_token]


class SlowEnvironment(StubEnvironment):
    """A stub whose ``exec`` yields, so gathered execs really interleave."""

    async def exec(self, command: str, *args, **kwargs) -> StubExecResult:
        self.calls.append((command, kwargs))
        await asyncio.sleep(0)
        return StubExecResult(return_code=0)


class SilentCleanupEnvironment(StubEnvironment):
    """A stub whose cleanup finds no sentinel (unwritable /tmp, dead tree)."""

    async def exec(self, command: str, *args, **kwargs) -> StubExecResult:
        if _CLEANUP_RE.match(command):
            self.calls.append((command, kwargs))
            return StubExecResult(return_code=0, stdout="")
        return await super().exec(command, *args, **kwargs)


class TestCleanupOutcomeIsObservable:
    """"Best effort" is only honest if a run can say how the effort went.

    The cleanup swallows every exception by design — a failed repair must
    never turn a timeout into an exception the loop has to handle — but
    swallowing silently left no way to tell "the kill landed" from "the
    environment was already gone", which is precisely the question the
    change's hypothesis needs answered from a run's logs.
    """

    async def test_a_landed_kill_is_logged(
        self, sandbox: HarborSandbox, env: StubEnvironment, caplog
    ):
        env.script("sleep 600", StubExecResult(return_code=124))
        with caplog.at_level(logging.DEBUG, logger="harness.sandbox.harbor_env"):
            await sandbox.exec("sleep 600", timeout=5)
        token = unwrap(env.calls[-2][0])[0]
        records = [r for r in caplog.records if token in r.getMessage()]
        assert [r.levelno for r in records] == [logging.DEBUG]
        assert "signalled the process group" in records[0].getMessage()

    async def test_a_cleanup_that_found_nothing_warns(self, caplog):
        env = SilentCleanupEnvironment()
        box = HarborSandbox(env, workspace_root="/app")
        env.script("sleep 600", StubExecResult(return_code=124))
        with caplog.at_level(logging.DEBUG, logger="harness.sandbox.harbor_env"):
            result = await box.exec("sleep 600", timeout=5)
        assert result.timed_out is True
        warnings_logged = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings_logged) == 1
        assert "no pid sentinel" in warnings_logged[0]

    async def test_a_failed_cleanup_warns_and_still_returns_the_timeout(
        self, caplog
    ):
        env = ExplodingCleanupEnvironment()
        box = HarborSandbox(env, workspace_root="/app")
        env.script("sleep 600", asyncio.TimeoutError())
        with caplog.at_level(logging.DEBUG, logger="harness.sandbox.harbor_env"):
            result = await box.exec("sleep 600", timeout=30)
        assert result.timed_out is True and result.exit_code == -1
        warnings_logged = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings_logged) == 1
        assert "may still be running" in warnings_logged[0]
        assert "environment is gone" in warnings_logged[0]

    async def test_a_clean_run_logs_nothing(
        self, sandbox: HarborSandbox, env: StubEnvironment, caplog
    ):
        env.script("echo hi", StubExecResult(return_code=0, stdout="hi\n"))
        with caplog.at_level(logging.DEBUG, logger="harness.sandbox.harbor_env"):
            await sandbox.exec("echo hi")
        assert caplog.records == []


class TestFileOps:
    async def test_write_read_round_trip(self, sandbox: HarborSandbox):
        content = (
            "line1\nline2 with 'single' and \"double\" quotes\n"
            "unicode: é 漢字 🎉\nshell hazards: $HOME `pwd` \\ && | > <\n"
        )
        await sandbox.write_file("notes.txt", content)
        assert await sandbox.read_file("notes.txt") == content

    async def test_empty_file_round_trip(self, sandbox: HarborSandbox):
        await sandbox.write_file("empty.txt", "")
        assert await sandbox.read_file("empty.txt") == ""

    async def test_write_creates_parent_dirs(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        await sandbox.write_file("sub/dir/f.txt", "hi")
        assert env.files["/app/sub/dir/f.txt"] == b"hi"
        assert "/app/sub/dir" in env.dirs

    async def test_absolute_path_inside_root_allowed(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        await sandbox.write_file("/app/abs.txt", "ok")
        assert env.files["/app/abs.txt"] == b"ok"
        assert await sandbox.read_file("/app/abs.txt") == "ok"

    async def test_read_missing_file(self, sandbox: HarborSandbox):
        with pytest.raises(SandboxError, match="file not found: missing.txt"):
            await sandbox.read_file("missing.txt")

    async def test_read_directory(self, sandbox: HarborSandbox, env: StubEnvironment):
        env.dirs.add("/app/somedir")
        with pytest.raises(
            SandboxError, match="path is a directory, not a file: somedir"
        ):
            await sandbox.read_file("somedir")

    async def test_read_unexpected_failure(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        from harness.sandbox.harbor_env import _read_command

        env.script(
            _read_command("/app/f.txt"),
            StubExecResult(return_code=1, stderr="base64: not found"),
        )
        with pytest.raises(SandboxError, match="failed to read f.txt"):
            await sandbox.read_file("f.txt")

    async def test_file_op_timeout_raises_sandbox_error(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        from harness.sandbox.harbor_env import _read_command

        env.script(
            _read_command("/app/slow.txt"),
            RuntimeError("Command timed out after 120 seconds"),
        )
        with pytest.raises(SandboxError, match="timed out"):
            await sandbox.read_file("slow.txt")


class TestWriteFileMode:
    """`mode` is a capability addition (write a large file in pieces), not
    a defect repair -- see harness/sandbox/base.py's `WriteMode` docstring."""

    async def test_default_mode_is_overwrite(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "old")
        await sandbox.write_file("f.txt", "new")
        assert await sandbox.read_file("f.txt") == "new"

    async def test_append_mode_concatenates(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "piece1-")
        await sandbox.write_file("f.txt", "piece2-", mode="append")
        await sandbox.write_file("f.txt", "piece3", mode="append")
        assert await sandbox.read_file("f.txt") == "piece1-piece2-piece3"

    async def test_append_to_missing_file_creates_it(self, sandbox: HarborSandbox):
        await sandbox.write_file("new.txt", "first chunk", mode="append")
        assert await sandbox.read_file("new.txt") == "first chunk"

    async def test_append_emits_double_redirect_operator(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        # Protects the wire protocol directly: the dispatched command must
        # use `>>`, not just have the stub's filesystem happen to agree.
        await sandbox.write_file("f.txt", "x", mode="append")
        command, _ = env.calls[-1]
        _, agent_command = unwrap(command)
        assert " base64 -d >> " in agent_command
        assert " base64 -d > " not in agent_command

    async def test_overwrite_emits_single_redirect_operator(
        self, sandbox: HarborSandbox, env: StubEnvironment
    ):
        await sandbox.write_file("f.txt", "x")
        command, _ = env.calls[-1]
        _, agent_command = unwrap(command)
        assert " base64 -d > " in agent_command
        assert ">>" not in agent_command


class TestWriteCommandUnit:
    """Direct unit tests of the pure `_write_command` helper."""

    def test_default_mode_uses_single_redirect(self):
        from harness.sandbox.harbor_env import _write_command

        command = _write_command("/app/f.txt", "/app", "aGk=")
        assert " > /app/f.txt" in command
        assert " >> " not in command

    def test_overwrite_mode_uses_single_redirect(self):
        from harness.sandbox.harbor_env import _write_command

        command = _write_command("/app/f.txt", "/app", "aGk=", mode="overwrite")
        assert " > /app/f.txt" in command
        assert " >> " not in command

    def test_append_mode_uses_double_redirect(self):
        from harness.sandbox.harbor_env import _write_command

        command = _write_command("/app/f.txt", "/app", "aGk=", mode="append")
        assert " >> /app/f.txt" in command


class TestEditSemantics:
    """Parity with LocalSandbox: the shared apply_edit contract, driven
    end-to-end through the base64 read/write protocol."""

    async def test_unique_replace(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "alpha beta gamma")
        await sandbox.edit_file("f.txt", "beta", "BETA")
        assert await sandbox.read_file("f.txt") == "alpha BETA gamma"

    async def test_not_found(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "alpha")
        with pytest.raises(SandboxError, match="old_string not found in file"):
            await sandbox.edit_file("f.txt", "zeta", "ZETA")

    async def test_not_unique(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "dup dup")
        with pytest.raises(
            SandboxError, match=r"not unique in file \(2 occurrences\)"
        ):
            await sandbox.edit_file("f.txt", "dup", "DUP")

    async def test_replace_all(self, sandbox: HarborSandbox):
        await sandbox.write_file("f.txt", "dup dup dup")
        await sandbox.edit_file("f.txt", "dup", "DUP", replace_all=True)
        assert await sandbox.read_file("f.txt") == "DUP DUP DUP"

    async def test_edit_missing_file(self, sandbox: HarborSandbox):
        with pytest.raises(SandboxError, match="file not found: nope.txt"):
            await sandbox.edit_file("nope.txt", "a", "b")


class TestPathJail:
    """Lexical containment: rejected BEFORE any command reaches the
    container (the stub records every exec call, so 'no calls' is
    checkable)."""

    @pytest.fixture
    def jailed(self, env: StubEnvironment) -> HarborSandbox:
        # Explicit root: construction issues no exec, so any call the
        # tests observe would have come from the rejected file op.
        return HarborSandbox(env, workspace_root="/app")

    @pytest.mark.parametrize(
        "path",
        [
            "",
            "../escape.txt",
            "a/../../escape.txt",
            "a/../b.txt",  # lexical jail: any '..' is rejected outright
            "/etc/passwd",
            "/apples/f.txt",  # sibling that merely shares the root prefix
        ],
    )
    async def test_rejected_paths_never_reach_exec(
        self, jailed: HarborSandbox, env: StubEnvironment, path: str
    ):
        with pytest.raises(SandboxPathError):
            await jailed.read_file(path)
        with pytest.raises(SandboxPathError):
            await jailed.write_file(path, "x")
        assert env.calls == []
