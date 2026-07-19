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
    SandboxError,
    SandboxPathError,
    apply_edit,
)
from harness.sandbox.local import LocalSandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(tmp_path / "workspace")


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
