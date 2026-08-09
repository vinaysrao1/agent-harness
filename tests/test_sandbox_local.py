"""Unit tests for harness.sandbox.local and the shared harness.sandbox.base
plumbing (path resolution, output truncation, edit semantics) exercised
through it."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from harness.sandbox.base import (
    MAX_OUTPUT_BYTES,
    SPILL_DIR,
    ExecResult,
    SandboxError,
    SandboxPathError,
    apply_edit,
    spill_tool_output,
)
from harness.sandbox.local import LocalSandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(tmp_path / "workspace")


@pytest.fixture
def spills():
    """Collects spill paths written during a test and removes them after.

    Spills deliberately land in the real ``/tmp`` (that is the mechanism
    under test — see :data:`harness.sandbox.base.SPILL_DIR`), so the test
    owns the cleanup rather than ``tmp_path``.
    """
    created: list[str] = []
    yield created
    for path in created:
        Path(path).unlink(missing_ok=True)


class TestExec:
    async def test_echo_stdout(self, sandbox: LocalSandbox):
        result = await sandbox.exec("echo hello")
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.timed_out is False

    async def test_exit_code_nonzero(self, sandbox: LocalSandbox):
        result = await sandbox.exec("exit 7")
        assert result.exit_code == 7
        assert result.timed_out is False

    async def test_stderr_captured(self, sandbox: LocalSandbox):
        result = await sandbox.exec("echo oops 1>&2")
        assert result.stderr == "oops\n"
        assert result.exit_code == 0

    async def test_runs_with_workspace_as_cwd(self, sandbox: LocalSandbox):
        await sandbox.start()
        (sandbox.workspace / "marker.txt").write_text("x")
        result = await sandbox.exec("ls")
        assert "marker.txt" in result.stdout

    async def test_timeout_kills_sleep(self, sandbox: LocalSandbox):
        start = time.monotonic()
        result = await sandbox.exec("sleep 30", timeout=0.3)
        elapsed = time.monotonic() - start
        assert result.timed_out is True
        assert result.exit_code == -1
        # Should return promptly after the timeout, not wait out the sleep.
        assert elapsed < 10

    async def test_timeout_kills_process_group_children(self, sandbox: LocalSandbox):
        # A backgrounded child of the shell should die too, not just the
        # top-level `sh -c` process -- verified by checking the child never
        # gets to run its post-sleep side effect.
        await sandbox.start()
        marker = sandbox.workspace / "child_alive.txt"
        result = await sandbox.exec(
            f"(sleep 30; touch {marker.name}) & wait", timeout=0.3
        )
        assert result.timed_out is True
        # Give a killed-but-not-yet-reaped child a moment, then confirm it
        # never got to touch the marker file (i.e. it was actually killed,
        # not merely detached from the parent's wait).
        time.sleep(0.5)
        assert not marker.exists()

    async def test_stdout_truncated_with_marker(self, sandbox: LocalSandbox):
        # Print more than MAX_OUTPUT_BYTES bytes of 'a'.
        n = MAX_OUTPUT_BYTES + 5000
        result = await sandbox.exec(f"python3 -c \"print('a' * {n})\"")
        assert result.exit_code == 0
        assert len(result.stdout.encode("utf-8")) < n + 200
        assert "truncated" in result.stdout
        assert "stdout" in result.stdout
        # The marker must report the *true* total size even though only
        # MAX_OUTPUT_BYTES were retained -- regression for the incremental,
        # capped drain (exec no longer buffers the full output before
        # truncating it).
        assert str(n + 1) in result.stdout  # + 1 for the trailing newline

    async def test_runaway_output_capped_incrementally(self, sandbox: LocalSandbox):
        # A command producing far more than MAX_OUTPUT_BYTES quickly must
        # still come back with output capped at MAX_OUTPUT_BYTES per stream
        # and an accurate total in the marker -- regression for exec()
        # buffering the *entire* stream before truncating (unbounded
        # memory use for a runaway command).
        n = MAX_OUTPUT_BYTES * 5
        result = await sandbox.exec(f"python3 -c \"print('b' * {n})\"", timeout=30)
        assert result.exit_code == 0
        assert len(result.stdout.encode("utf-8")) < MAX_OUTPUT_BYTES + 200
        assert f"{n + 1} bytes total" in result.stdout

    async def test_timeout_does_not_hang_on_detached_grandchild(
        self, sandbox: LocalSandbox
    ):
        # A grandchild that calls os.setsid() moves to a new session and
        # survives the SIGKILL sent to the timed-out command's process
        # group, while still holding the inherited stdout pipe open. This
        # must not hang exec() until the grandchild itself exits.
        await sandbox.start()
        cmd = (
            "python3 -c \"import os,time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    time.sleep(25)\n"
            "else:\n"
            "    time.sleep(0.2)\n\""
        )
        start = time.monotonic()
        result = await asyncio.wait_for(sandbox.exec(cmd, timeout=1.0), timeout=15)
        elapsed = time.monotonic() - start
        assert result.timed_out is True
        # Bounded by timeout + the post-kill drain grace period, not by
        # the 25s the detached grandchild actually sleeps for.
        assert elapsed < 10

    async def test_short_output_not_truncated(self, sandbox: LocalSandbox):
        result = await sandbox.exec("echo short")
        assert "truncated" not in result.stdout

    async def test_start_is_idempotent_and_creates_workspace(
        self, sandbox: LocalSandbox
    ):
        await sandbox.start()
        await sandbox.start()
        assert sandbox.workspace.is_dir()

    async def test_stop_is_noop(self, sandbox: LocalSandbox):
        await sandbox.start()
        await sandbox.stop()  # must not raise

    async def test_context_manager(self, tmp_path: Path):
        ws = tmp_path / "cm-workspace"
        async with LocalSandbox(ws) as sb:
            assert ws.is_dir()
            result = await sb.exec("echo hi")
            assert result.stdout == "hi\n"


class TestFileRoundTrip:
    async def test_write_then_read(self, sandbox: LocalSandbox):
        await sandbox.write_file("notes.txt", "hello world")
        content = await sandbox.read_file("notes.txt")
        assert content == "hello world"

    async def test_write_creates_parent_dirs(self, sandbox: LocalSandbox):
        await sandbox.write_file("a/b/c.txt", "nested")
        content = await sandbox.read_file("a/b/c.txt")
        assert content == "nested"
        assert (sandbox.workspace / "a" / "b" / "c.txt").is_file()

    async def test_write_overwrites_existing(self, sandbox: LocalSandbox):
        await sandbox.write_file("f.txt", "old")
        await sandbox.write_file("f.txt", "new")
        assert await sandbox.read_file("f.txt") == "new"

    async def test_read_missing_file_raises(self, sandbox: LocalSandbox):
        await sandbox.start()
        with pytest.raises(SandboxError):
            await sandbox.read_file("nope.txt")

    async def test_read_directory_raises(self, sandbox: LocalSandbox):
        await sandbox.start()
        (sandbox.workspace / "adir").mkdir()
        with pytest.raises(SandboxError):
            await sandbox.read_file("adir")

    async def test_exec_can_see_written_file(self, sandbox: LocalSandbox):
        await sandbox.write_file("seen.txt", "content-here")
        result = await sandbox.exec("cat seen.txt")
        assert result.stdout == "content-here"


class TestWriteFileMode:
    """`mode` is a capability addition (write a large file in pieces), not
    a defect repair -- see harness/sandbox/base.py's `WriteMode` docstring."""

    async def test_default_mode_is_overwrite(self, sandbox: LocalSandbox):
        # Backward-compat pin: omitting `mode` entirely must behave exactly
        # like before this change -- every existing call site stays valid.
        await sandbox.write_file("f.txt", "old")
        await sandbox.write_file("f.txt", "new")
        assert await sandbox.read_file("f.txt") == "new"

    async def test_explicit_overwrite_mode_replaces_contents(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("f.txt", "old")
        await sandbox.write_file("f.txt", "new", mode="overwrite")
        assert await sandbox.read_file("f.txt") == "new"

    async def test_append_mode_concatenates(self, sandbox: LocalSandbox):
        await sandbox.write_file("f.txt", "piece1-")
        await sandbox.write_file("f.txt", "piece2-", mode="append")
        await sandbox.write_file("f.txt", "piece3", mode="append")
        assert await sandbox.read_file("f.txt") == "piece1-piece2-piece3"

    async def test_append_to_missing_file_creates_it(self, sandbox: LocalSandbox):
        await sandbox.write_file("new.txt", "first chunk", mode="append")
        assert await sandbox.read_file("new.txt") == "first chunk"

    async def test_append_creates_parent_dirs(self, sandbox: LocalSandbox):
        await sandbox.write_file("a/b/c.txt", "nested", mode="append")
        assert await sandbox.read_file("a/b/c.txt") == "nested"

    async def test_edit_file_after_append_reads_and_overwrites_normally(
        self, sandbox: LocalSandbox
    ):
        # edit_file's default implementation must always write back with
        # mode="overwrite" regardless of how the file got its content.
        await sandbox.write_file("f.py", "x = 1\n")
        await sandbox.write_file("f.py", "y = 2\n", mode="append")
        await sandbox.edit_file("f.py", "x = 1", "x = 100")
        assert await sandbox.read_file("f.py") == "x = 100\ny = 2\n"


class TestEditFile:
    async def test_edit_replaces_unique_match(self, sandbox: LocalSandbox):
        await sandbox.write_file("f.py", "x = 1\ny = 2\n")
        await sandbox.edit_file("f.py", "x = 1", "x = 100")
        assert await sandbox.read_file("f.py") == "x = 100\ny = 2\n"

    async def test_edit_not_found_raises_with_clear_message(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("f.py", "x = 1\n")
        with pytest.raises(SandboxError, match="not found"):
            await sandbox.edit_file("f.py", "z = 9", "z = 10")

    async def test_edit_not_unique_raises_with_clear_message(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("f.py", "dup\ndup\n")
        with pytest.raises(SandboxError, match="not unique"):
            await sandbox.edit_file("f.py", "dup", "single")

    async def test_edit_replace_all(self, sandbox: LocalSandbox):
        await sandbox.write_file("f.py", "dup\ndup\ndup\n")
        await sandbox.edit_file("f.py", "dup", "one", replace_all=True)
        assert await sandbox.read_file("f.py") == "one\none\none\n"

    async def test_edit_empty_old_string_raises(self, sandbox: LocalSandbox):
        await sandbox.write_file("f.py", "content\n")
        with pytest.raises(SandboxError):
            await sandbox.edit_file("f.py", "", "x")


class TestApplyEditUnit:
    """Direct unit tests of the pure apply_edit helper (base.py)."""

    def test_unique_replace(self):
        assert apply_edit("abc def", "abc", "xyz") == "xyz def"

    def test_not_found(self):
        with pytest.raises(SandboxError, match="not found"):
            apply_edit("abc", "zzz", "yyy")

    def test_ambiguous_without_replace_all(self):
        with pytest.raises(SandboxError, match="not unique"):
            apply_edit("aa aa", "aa", "bb")

    def test_ambiguous_with_replace_all(self):
        assert apply_edit("aa aa", "aa", "bb", replace_all=True) == "bb bb"

    def test_empty_old_string_rejected(self):
        with pytest.raises(SandboxError, match="non-empty"):
            apply_edit("abc", "", "x")


class TestPathTraversal:
    async def test_absolute_path_rejected_read(self, sandbox: LocalSandbox):
        await sandbox.start()
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("/etc/passwd")

    async def test_absolute_path_rejected_write(self, sandbox: LocalSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.write_file("/tmp/evil.txt", "pwned")

    async def test_dotdot_traversal_rejected(self, sandbox: LocalSandbox):
        await sandbox.start()
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("../outside.txt")

    async def test_dotdot_traversal_rejected_deep(self, sandbox: LocalSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.write_file("a/../../outside.txt", "x")

    async def test_dotdot_that_stays_inside_is_allowed(self, sandbox: LocalSandbox):
        await sandbox.write_file("a/b.txt", "hi")
        content = await sandbox.read_file("a/../a/b.txt")
        assert content == "hi"

    async def test_empty_path_rejected(self, sandbox: LocalSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("")

    async def test_symlink_escape_rejected(self, sandbox: LocalSandbox, tmp_path: Path):
        await sandbox.start()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("top secret")
        link = sandbox.workspace / "escape"
        os.symlink(outside, link)
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("escape/secret.txt")

    async def test_edit_file_path_traversal_rejected(self, sandbox: LocalSandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.edit_file("../outside.txt", "a", "b")


class _StubSandbox:
    """A Sandbox stand-in whose ``exec`` returns a scripted result.

    Used for the spill failure paths, which must be exercised without
    depending on a real filesystem error being reproducible.
    """

    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.commands: list[str] = []

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        self.commands.append(command)
        return self.result


class TestSpillToolOutput:
    """`spill_tool_output` — the retrieval affordance behind the truncation
    marker. Two things are load-bearing: the file must be readable back
    verbatim, and it must never land in the workspace."""

    async def test_writes_content_and_returns_tmp_path(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        await sandbox.start()
        content = "alpha\nbeta\ngamma\n"
        path = await spill_tool_output(sandbox, content)
        spills.append(path)
        assert path.startswith(SPILL_DIR + "/")
        assert path.startswith("/tmp/")
        assert Path(path).read_text(encoding="utf-8") == content

    async def test_never_writes_into_the_workspace(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        # The non-negotiable one: a stray file next to the deliverable has
        # already cost a graded task, and graders inspect the workspace.
        await sandbox.start()
        path = await spill_tool_output(sandbox, "some output\n")
        spills.append(path)
        workspace_root = str(sandbox.workspace.resolve())
        assert not path.startswith(workspace_root)
        assert not Path(path).resolve().is_relative_to(sandbox.workspace.resolve())
        assert list(sandbox.workspace.iterdir()) == []

    async def test_large_content_round_trips_across_chunks(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        # Bigger than one heredoc chunk, so the append path is exercised.
        await sandbox.start()
        content = "".join(f"line {i} " + "z" * 60 + "\n" for i in range(4000))
        assert len(content.encode("utf-8")) > 200_000
        path = await spill_tool_output(sandbox, content)
        spills.append(path)
        assert Path(path).read_text(encoding="utf-8") == content

    async def test_shell_metacharacters_are_written_verbatim(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        # The heredoc is quoted, so nothing in the content is expanded or
        # executed, and no plausible line can close the uuid delimiter.
        await sandbox.start()
        content = "EOF\n$(touch /tmp/pwned-by-spill)\n`id`\n${HOME}\n'\"\\\n"
        path = await spill_tool_output(sandbox, content)
        spills.append(path)
        assert Path(path).read_text(encoding="utf-8") == content
        assert not Path("/tmp/pwned-by-spill").exists()

    async def test_content_without_trailing_newline_gains_exactly_one(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        await sandbox.start()
        path = await spill_tool_output(sandbox, "no newline at end")
        spills.append(path)
        assert Path(path).read_text(encoding="utf-8") == "no newline at end\n"

    async def test_each_spill_gets_a_fresh_path(
        self, sandbox: LocalSandbox, spills: list[str]
    ):
        await sandbox.start()
        first = await spill_tool_output(sandbox, "one\n")
        second = await spill_tool_output(sandbox, "two\n")
        spills.extend([first, second])
        assert first != second
        assert Path(first).read_text(encoding="utf-8") == "one\n"
        assert Path(second).read_text(encoding="utf-8") == "two\n"

    async def test_nonzero_exit_raises_sandbox_error(self):
        stub = _StubSandbox(
            ExecResult(exit_code=1, stdout="", stderr="No space left on device")
        )
        with pytest.raises(SandboxError) as excinfo:
            await spill_tool_output(stub, "content\n")
        assert "No space left on device" in str(excinfo.value)

    async def test_timeout_raises_sandbox_error(self):
        stub = _StubSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        with pytest.raises(SandboxError) as excinfo:
            await spill_tool_output(stub, "content\n")
        assert "timed out" in str(excinfo.value)
