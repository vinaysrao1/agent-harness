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
import re
import shlex
import textwrap
from dataclasses import dataclass
from posixpath import dirname

import pytest

from harness.sandbox.base import (
    MAX_OUTPUT_BYTES,
    SandboxError,
    SandboxPathError,
)
from harness.sandbox.harbor_env import (
    DEFAULT_WORKSPACE_ROOT,
    HarborSandbox,
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


class StubEnvironment:
    """Scriptable, recording stand-in for a Harbor environment.

    ``calls`` records every ``exec`` invocation. ``script()`` pins the
    outcome (result or exception) for an exact command string. Unscripted
    commands are interpreted: ``pwd`` reports ``workdir``; the sandbox's
    write command decodes its base64 payload into ``files``; the read
    command re-encodes from ``files`` (wrapped at 76 columns, like real
    ``base64``, so the decoder must handle embedded newlines); anything
    else succeeds silently with ``None`` streams.
    """

    def __init__(self, workdir: str = "/app") -> None:
        self.workdir = workdir
        self.calls: list[tuple[str, dict]] = []
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {workdir}
        self._scripted: dict[str, StubExecResult | BaseException] = {}

    def script(self, command: str, outcome: StubExecResult | BaseException) -> None:
        """Pin the outcome of one exact command string."""
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
        if command in self._scripted:
            outcome = self._scripted[command]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        if command == "pwd":
            return StubExecResult(return_code=0, stdout=self.workdir + "\n")
        write = self._try_write(command)
        if write is not None:
            return write
        read = self._try_read(command)
        if read is not None:
            return read
        return StubExecResult(return_code=0)

    def _try_write(self, command: str) -> StubExecResult | None:
        """Interpret the sandbox's mkdir-and-base64-decode write command."""
        tokens = shlex.split(command)
        shape = (
            len(tokens) == 12
            and tokens[0:2] == ["mkdir", "-p"]
            and tokens[3] == "&&"
            and tokens[4:6] == ["printf", "%s"]
            and tokens[7] == "|"
            and tokens[8:10] == ["base64", "-d"]
            and tokens[10] == ">"
        )
        if not shape:
            return None
        parent, encoded, path = tokens[2], tokens[6], tokens[11]
        # Honest round trip: decode exactly what would hit `base64 -d`.
        self.files[path] = base64.b64decode(encoded)
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
        assert command == "echo hi"
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
