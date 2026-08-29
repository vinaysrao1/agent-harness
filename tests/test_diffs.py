"""S-202: the diff is the artifact.

What a repo-mode run produced is a diff, not a transcript. The two properties
that matter most are asymmetric and both are asserted directly: `diff` never
touches the work tree, and `rewind` -- the one operation that writes -- is
never automatic.
"""

from __future__ import annotations

import pytest

from harness.diffs import (
    DiffStat,
    FileChange,
    RewindResolution,
    ShadowReader,
    render_pr_body,
    render_report,
)


class _Result:
    def __init__(self, stdout: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.stderr = ""


class _Sandbox:
    def __init__(self, responses: dict[str, _Result] | None = None) -> None:
        self.commands: list[str] = []
        self.responses = responses or {}

    async def exec(self, command: str, timeout: float = 120) -> _Result:
        self.commands.append(command)
        for fragment, result in self.responses.items():
            if fragment in command:
                return result
        return _Result(exit_code=1)


def _refs(*turns: int) -> str:
    lines = ["refs/harness/agent1/baseline"]
    lines += [f"refs/harness/agent1/turn-{t}" for t in turns]
    return "\n".join(lines)


def _reader(responses: dict[str, _Result] | None = None) -> tuple[ShadowReader, _Sandbox]:
    sandbox = _Sandbox(responses or {})
    return ShadowReader(sandbox, "run1", "agent1"), sandbox


class TestDiffNeverMutates:
    """Acceptance (1). A tool that silently reverted the agent's work the
    moment you asked to *look* at it would be worse than no tool."""

    async def test_S202_stat_issues_only_read_commands(self) -> None:
        reader, sandbox = _reader(
            {
                "for-each-ref": _Result(_refs(1, 2)),
                "diff --numstat": _Result("10\t2\tsrc/a.py\n0\t5\tsrc/b.py"),
            }
        )
        await reader.stat()
        assert sandbox.commands
        for command in sandbox.commands:
            for mutator in (
                "checkout-index",
                "read-tree",
                "add -A",
                "commit-tree",
                "update-ref",
                "reset",
            ):
                assert mutator not in command, f"diff may write: {command!r}"

    async def test_S202_read_commands_carry_no_work_tree(self) -> None:
        # A read that carried a work tree could touch the index; diff promises
        # not to, so the argument is simply absent from every read.
        reader, sandbox = _reader(
            {"for-each-ref": _Result(_refs(1)), "diff --numstat": _Result("")}
        )
        await reader.stat()
        for command in sandbox.commands:
            assert "--work-tree" not in command, (
                f"a read command carried a work tree: {command!r}"
            )

    async def test_S202_patch_is_read_only(self) -> None:
        reader, sandbox = _reader(
            {"for-each-ref": _Result(_refs(1)), "diff ": _Result("--- a\n+++ b")}
        )
        await reader.patch()
        for command in sandbox.commands:
            assert "checkout-index" not in command and "read-tree" not in command


class TestRewindIsExplicit:
    """Acceptance (2)."""

    async def test_S202_rewind_writes_only_when_called(self) -> None:
        reader, sandbox = _reader(
            {"for-each-ref": _Result(_refs(1, 2)), "read-tree": _Result("")}
        )
        resolution = await reader.rewind(2)
        assert resolution.found
        assert any("checkout-index" in c for c in sandbox.commands), (
            "rewind did not materialise the snapshot"
        )

    async def test_S202_nothing_else_calls_rewind(self) -> None:
        # The guarantee is structural: no other method in the module may
        # invoke it, or "never automatic" is a convention rather than a fact.
        import ast
        import inspect

        import harness.diffs as module

        tree = ast.parse(inspect.getsource(module))
        callers = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "rewind"
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "rewind"
                for inner in ast.walk(node)
            )
        ]
        assert not callers, f"rewind is called automatically by: {callers}"

    async def test_S202_loop_never_calls_rewind(self) -> None:
        from pathlib import Path

        source = Path("harness/loop.py").read_text(encoding="utf-8")
        assert "rewind" not in source, (
            "the agent loop references rewind; reverting an agent's work must "
            "never be automatic"
        )


class TestRewindResolvesTurnsWithoutCheckpoints:
    """Amendment #3: turns with no tool calls write no checkpoint."""

    async def test_S202_exact_turn_resolves_exactly(self) -> None:
        reader, _ = _reader({"for-each-ref": _Result(_refs(1, 3, 5))})
        resolution = await reader.resolve(3)
        assert resolution.exact and resolution.resolved_turn == 3
        assert "turn 3" in resolution.describe()

    async def test_S202_missing_turn_resolves_to_nearest_earlier(self) -> None:
        reader, _ = _reader({"for-each-ref": _Result(_refs(1, 3, 5))})
        resolution = await reader.resolve(4)
        assert not resolution.exact
        assert resolution.resolved_turn == 3
        # It must *say* it did this. Silently returning a different turn than
        # was asked for is the failure mode.
        assert "changed nothing" in resolution.describe()
        assert "turn 3" in resolution.describe()

    async def test_S202_turn_before_any_checkpoint_is_not_found(self) -> None:
        reader, _ = _reader({"for-each-ref": _Result(_refs(5, 6))})
        resolution = await reader.resolve(2)
        assert not resolution.found
        assert "no checkpoint" in resolution.describe()

    async def test_S202_no_checkpoints_at_all(self) -> None:
        reader, _ = _reader({"for-each-ref": _Result("")})
        assert await reader.turns() == []
        assert not (await reader.resolve(1)).found

    async def test_S202_only_this_agents_refs_are_listed(self) -> None:
        # Refs are per-agent since the S-201 amendment; a reader must not pick
        # up a sibling subagent's snapshots.
        reader, sandbox = _reader({"for-each-ref": _Result(_refs(1, 2))})
        await reader.turns()
        assert any("refs/harness/agent1" in c for c in sandbox.commands)


class TestStatParsing:
    async def test_S202_numstat_is_parsed(self) -> None:
        reader, _ = _reader(
            {
                "for-each-ref": _Result(_refs(1)),
                "diff --numstat": _Result("10\t2\tsrc/a.py\n0\t5\tsrc/b.py"),
            }
        )
        stat = await reader.stat()
        assert stat.files_changed == 2
        assert stat.added == 10 and stat.removed == 7
        assert stat.summary() == "2 files changed, +10/-7"

    async def test_S202_binary_files_count_as_changed(self) -> None:
        # git writes "-" for a binary file: a real change with uncountable
        # lines. Dropping the row would under-report files changed.
        reader, _ = _reader(
            {
                "for-each-ref": _Result(_refs(1)),
                "diff --numstat": _Result("-\t-\tassets/logo.png"),
            }
        )
        stat = await reader.stat()
        assert stat.files_changed == 1
        assert stat.added == 0 and stat.removed == 0

    async def test_S202_paths_with_spaces_survive(self) -> None:
        reader, _ = _reader(
            {
                "for-each-ref": _Result(_refs(1)),
                "diff --numstat": _Result("1\t0\tsrc/my file.py"),
            }
        )
        stat = await reader.stat()
        assert stat.files[0].path == "src/my file.py"

    async def test_S202_empty_diff_is_empty_not_absent(self) -> None:
        reader, _ = _reader(
            {"for-each-ref": _Result(_refs(1)), "diff --numstat": _Result("")}
        )
        stat = await reader.stat()
        assert stat.empty
        assert stat.summary() == "no files changed"

    async def test_S202_a_failing_git_degrades_to_empty(self) -> None:
        reader, _ = _reader({"for-each-ref": _Result(_refs(1))})
        assert (await reader.stat()).empty


class TestReport:
    """Acceptance (3): the report leads with the stat line, then the files."""

    def test_S202_report_leads_with_the_summary(self) -> None:
        stat = DiffStat(files=(FileChange("a.py", 10, 2), FileChange("b.py", 0, 5)))
        report = render_report(stat)
        assert report.splitlines()[0] == "2 files changed, +10/-7"
        assert "  a.py  +10/-2" in report

    def test_S202_report_truncates_long_lists_visibly(self) -> None:
        stat = DiffStat(files=tuple(FileChange(f"f{i}.py", 1, 0) for i in range(30)))
        report = render_report(stat, limit=5)
        assert "... and 25 more" in report

    def test_S202_singular_file_reads_correctly(self) -> None:
        assert DiffStat(files=(FileChange("a", 1, 0),)).summary() == (
            "1 file changed, +1/-0"
        )


class TestPrBody:
    """Acceptance (4): assembled from ledger evidence, not invented."""

    def test_S202_pr_body_uses_existing_evidence(self) -> None:
        stat = DiffStat(files=(FileChange("a.py", 3, 1),))
        body = render_pr_body("Fix the parser", stat, ["tests pass", "lint clean"])
        assert body.startswith("Fix the parser")
        assert "1 file changed, +3/-1" in body
        assert "- tests pass" in body

    def test_S202_pr_body_without_evidence_omits_the_section(self) -> None:
        body = render_pr_body("Goal", DiffStat(), [])
        assert "## Evidence" not in body
        assert "no files changed" in body


class TestAgentResultCarriesTheStat:
    def test_S202_diff_stat_defaults_to_none(self) -> None:
        # None, not "": "no diff was computed" and "computed an empty diff"
        # are different facts, and the benchmark path is always the former.
        from harness.loop import AgentResult
        from harness.types import Usage

        result = AgentResult(
            status="completed", final_text="done", usage=Usage(), turns=1
        )
        assert result.diff_stat is None

    def test_S202_diff_stat_round_trips(self) -> None:
        from harness.loop import AgentResult
        from harness.types import Usage

        result = AgentResult(
            status="completed",
            final_text="done",
            usage=Usage(),
            turns=1,
            diff_stat="2 files changed, +10/-7",
        )
        assert result.diff_stat == "2 files changed, +10/-7"


class TestAgainstARealShell:
    """Every other test here execs against a fake sandbox that stores the
    command string and never runs it. That is the right shape for asserting
    *which* commands are issued -- and it is blind to whether they work.

    It stayed blind to a real one: ``for-each-ref --format=%(refname)`` is a
    syntax error in ``sh``, because the parentheses are shell metacharacters.
    :meth:`ShadowReader._run` turns any failure into ``(1, "")`` and
    :meth:`turns` turns that into ``[]``, so in every real shell the reader
    reported "no checkpoints" for a store that was full of them -- while the
    suite stayed green. These tests run the commands the reader actually
    builds, through a real shell, against a real shadow store.
    """

    @pytest.fixture
    def shadow(self, tmp_path, monkeypatch):
        """A real bare shadow store holding a baseline and two turns."""
        import subprocess

        work = tmp_path / "work"
        work.mkdir()
        (work / "kept.txt").write_text("unchanged\n")

        root = tmp_path / "shadow"
        root.mkdir()
        monkeypatch.setattr("harness.repo.SHADOW_GIT_DIR", str(root))
        git_dir = root / "run1"
        subprocess.run(["git", "init", "--bare", "-q", str(git_dir)], check=True)

        index = git_dir / "index-agent1"

        def snapshot(ref: str) -> None:
            env = {"GIT_INDEX_FILE": str(index)}
            base = [
                "git", "-c", "user.name=t", "-c", "user.email=t@localhost",
                f"--git-dir={git_dir}", f"--work-tree={work}",
            ]
            subprocess.run(base + ["add", "-A"], check=True, env={**_environ(), **env})
            tree = subprocess.run(
                base + ["write-tree"], check=True, capture_output=True,
                text=True, env={**_environ(), **env},
            ).stdout.strip()
            commit = subprocess.run(
                base + ["commit-tree", tree, "-m", ref], check=True,
                capture_output=True, text=True, env={**_environ(), **env},
            ).stdout.strip()
            subprocess.run(
                base + ["update-ref", f"refs/harness/agent1/{ref}", commit],
                check=True, env={**_environ(), **env},
            )

        snapshot("baseline")
        snapshot("turn-1")                      # no edits yet
        (work / "added.py").write_text("x = 1\ny = 2\n")
        snapshot("turn-2")                      # the first real edit
        return work

    class _RealShell:
        """The one method ShadowReader calls, backed by an actual shell.

        Deliberately ``shell=True``: the sandbox contract runs these commands
        through a shell, and it is the shell that made
        ``--format=%(refname)`` a syntax error. A shim that used ``exec``
        directly would tokenize the command itself and reproduce the very
        blindness these tests exist to remove.
        """

        def __init__(self, cwd) -> None:
            self._cwd = cwd

        async def exec(self, command: str, timeout: float = 30.0):
            import subprocess

            proc = subprocess.run(
                command, shell=True, cwd=str(self._cwd),
                capture_output=True, text=True, timeout=timeout,
            )

            class _Out:
                exit_code = proc.returncode
                stdout = proc.stdout or ""

            return _Out()

    def _reader(self, work):
        return ShadowReader(self._RealShell(work), "run1", "agent1")

    async def test_S202_turns_are_found_through_a_real_shell(self, shadow) -> None:
        assert await self._reader(shadow).turns() == [1, 2]

    async def test_S202_stat_defaults_to_the_latest_turn(self, shadow) -> None:
        # stat() with no ref goes through turns(); when that silently returned
        # [] this reported "no files changed" for a real change.
        stat = await self._reader(shadow).stat()
        assert not stat.empty, "the default-ref path found nothing to diff"
        assert [f.path for f in stat.files] == ["added.py"]
        assert stat.summary() == "1 file changed, +2/-0"

    async def test_S202_a_turn_that_changed_nothing_diffs_empty(self, shadow) -> None:
        assert (await self._reader(shadow).stat("refs/harness/agent1/turn-1")).empty

    async def test_S202_resolve_finds_a_real_checkpoint(self, shadow) -> None:
        resolution = await self._reader(shadow).resolve(2)
        assert resolution.found and resolution.exact

    async def test_S202_patch_returns_a_real_diff(self, shadow) -> None:
        patch = await self._reader(shadow).patch()
        assert "added.py" in patch and "+x = 1" in patch


def _environ() -> dict:
    import os

    return dict(os.environ)
