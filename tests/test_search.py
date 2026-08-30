"""S-101: structured search as tools, not shell pipelines.

An agent looking for a symbol runs `bash("grep -rn ... | head -50")` today.
That costs a shell round-trip, an unpredictable amount of output, and a
permission decision on a general-purpose execution tool for what is a read —
and it fails the way pipelines fail: `head` closes the pipe, `grep` dies of
SIGPIPE, and the recorded exit status describes the last stage.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from harness.sandbox.local import LocalSandbox
from harness.search import (
    DEFAULT_HEAD_LIMIT,
    MAX_HEAD_LIMIT,
    OUTPUT_MODES,
    SearchRequest,
    fallback_program,
    render_results,
    ripgrep_command,
)
from harness.tools.builtin import glob_tool, grep_tool


@pytest.fixture
def corpus() -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "pkg").mkdir()
    (directory / "a.py").write_text("def alpha():\n    return 1\n")
    (directory / "pkg" / "b.py").write_text("from a import alpha\n\n\ndef beta():\n    return alpha()\n")
    (directory / "notes.txt").write_text("alpha is mentioned here\n")
    (directory / "binary.bin").write_bytes(b"\x00\x01alpha\x02\xff")
    return directory


async def _sandbox(directory: Path) -> LocalSandbox:
    sandbox = LocalSandbox(directory)
    await sandbox.start()
    return sandbox


class TestTheRequestIsValidated:
    def test_S101_an_unknown_output_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="output_mode"):
            SearchRequest(pattern="x", output_mode="everything")

    def test_S101_an_empty_pattern_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="pattern"):
            SearchRequest(pattern="")

    def test_S101_the_default_mode_is_paths_only(self) -> None:
        # The cheapest useful answer: the agent almost always wants to know
        # *where* before it wants to read anything, and returning content by
        # default spends context on files it is about to discard.
        assert SearchRequest(pattern="x").output_mode == "files_with_matches"
        assert OUTPUT_MODES[0] == "files_with_matches"

    def test_S101_the_limit_cannot_be_raised_past_the_cap(self) -> None:
        # The bound exists to stop a search consuming the context window.
        # Letting the model opt out would make it advisory.
        assert SearchRequest(pattern="x", head_limit=10_000).effective_limit == (
            MAX_HEAD_LIMIT
        )


class TestBoundedBeforeTruncationNotBy:
    """Acceptance (4). A pipeline computes every match and throws the tail
    away; the cost is already paid by then, in both time and the sandbox's
    memory."""

    def test_S101_ripgrep_is_told_the_limit(self) -> None:
        command = ripgrep_command(SearchRequest(pattern="x", head_limit=7))
        assert "--max-count 7" in command, command

    def test_S101_the_fallback_is_told_the_limit(self) -> None:
        _, spec = fallback_program(SearchRequest(pattern="x", head_limit=7))
        assert '"limit": 7' in spec

    async def test_S101_a_thousand_matches_return_the_limit(self, tmp_path) -> None:
        for index in range(1_000):
            (tmp_path / f"f{index}.py").write_text("needle\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler(
            {"pattern": "needle", "head_limit": 5}
        )
        paths = [line for line in result.splitlines() if line.endswith(".py")]
        assert len(paths) == 5, result

    async def test_S101_the_search_itself_stops_at_the_limit(
        self, tmp_path
    ) -> None:
        # The distinction acceptance (4) is actually about. Going through the
        # tool cannot see it: `render_results` slices to the limit either way,
        # so a search that scans a thousand files and throws 995 away produces
        # an identical answer to one that stopped at five. This asserts on the
        # engine's own stdout, before any rendering.
        for index in range(1_000):
            (tmp_path / f"f{index}.py").write_text("needle\n")
        sandbox = await _sandbox(tmp_path)
        command, _ = fallback_program(
            SearchRequest(pattern="needle", head_limit=5)
        )
        result = await sandbox.exec(command, timeout=60)
        produced = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(produced) == 5, (
            f"the search produced {len(produced)} results for a limit of 5; "
            "it is scanning everything and truncating afterwards"
        )

    async def test_S101_a_bounded_result_says_so(self, tmp_path) -> None:
        # A truncated result that does not announce itself is the worst
        # outcome: the model concludes the symbol appears in five files when
        # it appears in a thousand, and narrows on a false premise.
        for index in range(20):
            (tmp_path / f"f{index}.py").write_text("needle\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler(
            {"pattern": "needle", "head_limit": 5}
        )
        assert "this is a bound, not the total" in result

    async def test_S101_an_unbounded_result_does_not_claim_to_be_bounded(
        self, corpus
    ) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler({"pattern": "alpha"})
        assert "bound, not the total" not in result


class TestTheFallback:
    """The engine that runs when `rg` is absent — which is most containers."""

    async def test_S101_files_with_matches_returns_paths(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler({"pattern": "alpha"})
        assert "a.py" in result and "b.py" in result
        assert "def alpha" not in result, "paths-only mode returned content"

    async def test_S101_content_mode_returns_lines_with_numbers(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler(
            {"pattern": "def alpha", "output_mode": "content"}
        )
        assert ":1:def alpha():" in result

    async def test_S101_count_mode_returns_totals(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler(
            {"pattern": "alpha", "output_mode": "count", "glob": "*.py"}
        )
        assert ":1" in result

    async def test_S101_count_counts_matches_not_matching_lines(
        self, tmp_path
    ) -> None:
        # `rg --count-matches` counts matches. Counting matching *lines* made
        # a file with three hits on one line report 1 here and 3 there -- a
        # silent disagreement the path-set equivalence test cannot see.
        (tmp_path / "multi.txt").write_text("alpha alpha alpha\nalpha\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler(
            {"pattern": "alpha", "output_mode": "count"}
        )
        assert result.strip().endswith(":4"), result

    async def test_S101_a_glob_narrows_the_search(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler(
            {"pattern": "alpha", "glob": "*.txt"}
        )
        assert "notes.txt" in result
        assert "a.py" not in result

    async def test_S101_binary_files_are_skipped(self, corpus) -> None:
        # Exactly as ripgrep skips them. Returning a line of a binary file is
        # noise the model then has to spend a turn discarding.
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler({"pattern": "alpha"})
        assert "binary.bin" not in result

    async def test_S101_noise_directories_are_skipped(self, tmp_path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "real.py").write_text("needle\n")
        for noise in (".git", "node_modules", "__pycache__", ".venv"):
            (tmp_path / noise).mkdir()
            (tmp_path / noise / "junk.py").write_text("needle\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler({"pattern": "needle"})
        assert "real.py" in result
        for noise in (".git", "node_modules", "__pycache__", ".venv"):
            assert noise not in result, result

    async def test_S101_context_lines_are_returned(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler(
            {"pattern": "def beta", "output_mode": "content", "context": 1}
        )
        assert "return alpha()" in result

    async def test_S101_case_insensitive_matching(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        assert "a.py" in await grep_tool(sandbox).handler(
            {"pattern": "ALPHA", "case_insensitive": True}
        )
        assert "no matches" in await grep_tool(sandbox).handler({"pattern": "ALPHA"})

    async def test_S101_a_regex_with_shell_metacharacters_is_not_mangled(
        self, tmp_path
    ) -> None:
        # The fallback takes its arguments as JSON on stdin precisely so a
        # pattern never has to survive two levels of shell quoting. Getting
        # that wrong turns a search into a syntax error, or -- worse -- into a
        # different search that silently succeeds.
        (tmp_path / "x.py").write_text("value = $(compute) && other || 'quoted'\n")
        sandbox = await _sandbox(tmp_path)
        for pattern in [r"\$\(compute\)", r"&& other", r"'quoted'", r"a|b"]:
            result = await grep_tool(sandbox).handler(
                {"pattern": pattern, "output_mode": "content"}
            )
            assert "search failed" not in result, (pattern, result)


class TestGlob:
    async def test_S101_glob_finds_files_by_name(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await glob_tool(sandbox).handler({"pattern": "*.py"})
        assert "a.py" in result
        assert "notes.txt" not in result

    async def test_S101_glob_returns_paths_only(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        result = await glob_tool(sandbox).handler({"pattern": "*.py"})
        assert "def alpha" not in result

    @pytest.mark.parametrize("pattern", ["pkg/*.py", "*/pkg/*.py", "pkg/**/*.py"])
    async def test_S101_a_path_pattern_matches_the_whole_path(
        self, corpus, pattern: str
    ) -> None:
        # `-path` matches the path *as find prints it*, which begins "./", so
        # the bare pattern matched nothing. The earlier version of this test
        # used `*/pkg/*.py` -- the one form that happened to work -- and so
        # certified the broken behaviour as correct while both documented
        # examples ('pkg/*.py', 'src/**/*.ts') returned "no matches".
        sandbox = await _sandbox(corpus)
        result = await glob_tool(sandbox).handler({"pattern": pattern})
        assert "b.py" in result, (pattern, result)

    async def test_S101_glob_skips_the_same_noise_as_grep(self, tmp_path) -> None:
        # `rg --files` honours .gitignore and skips hidden files; the `find`
        # fallback had no pruning at all, so in any real repo the two engines
        # returned unrelated answers -- fifty vendored files and a "stopped at
        # 50" notice from one, the sources from the other.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "real.py").write_text("x\n")
        for noise in (".git", "node_modules", ".venv", "__pycache__"):
            (tmp_path / noise).mkdir()
            (tmp_path / noise / "junk.py").write_text("x\n")
        sandbox = await _sandbox(tmp_path)
        result = await glob_tool(sandbox).handler({"pattern": "*.py"})
        assert "real.py" in result
        for noise in (".git", "node_modules", ".venv", "__pycache__"):
            assert noise not in result, result


class TestAFailedSearchIsNotNoMatches:
    """The archetype's natural home.

    `Sandbox.exec` does not raise on a non-zero exit — it returns the code,
    stderr and a timeout flag. An earlier version discarded all three and
    rendered empty stdout as "no matches", so an invalid regex, an unknown
    `--type`, a missing `python3` and a backtracking timeout all reported the
    repository as not containing the pattern. That is a confident wrong answer
    the model then acts on.
    """

    async def test_S101_an_invalid_regex_is_reported_as_one(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        with pytest.raises(ValueError, match="invalid regular expression"):
            await grep_tool(sandbox).handler({"pattern": "alpha("})

    async def test_S101_a_missing_backend_is_reported(self, corpus, monkeypatch) -> None:
        # An image without python3 and without rg. The spec previously claimed
        # this "fails rather than degrading further"; it reported the whole
        # repository as empty.
        from harness.search import fallback_program

        def broken(request):
            return "exit 127", ""

        monkeypatch.setattr("harness.tools.builtin.fallback_program", broken)
        sandbox = await _sandbox(corpus)
        with pytest.raises(ValueError, match="not available in this sandbox"):
            await grep_tool(sandbox).handler({"pattern": "alpha"})

    async def test_S101_a_timeout_is_reported_as_one(self, corpus, monkeypatch) -> None:
        class _TimedOut:
            exit_code = -1
            stdout = ""
            stderr = ""
            timed_out = True

        async def fake_exec(self, command, **kwargs):
            if "command -v rg" in command:
                return type("R", (), {"exit_code": 1, "stdout": "", "stderr": ""})()
            return _TimedOut()

        monkeypatch.setattr(LocalSandbox, "exec", fake_exec)
        sandbox = await _sandbox(corpus)
        with pytest.raises(ValueError, match="timed out"):
            await grep_tool(sandbox).handler({"pattern": "alpha"})

    @pytest.mark.parametrize(
        "engine,exit_code,expect_error",
        [
            ("rg", 0, False),          # matches
            ("rg", 1, False),          # genuinely no matches
            ("rg", 2, True),           # bad pattern, unknown --type, bad path
            ("python fallback", 0, False),
            ("python fallback", 2, True),
            ("find", 1, True),
        ],
    )
    def test_S101_exit_codes_are_interpreted_per_engine(
        self, engine: str, exit_code: int, expect_error: bool
    ) -> None:
        # rg alone distinguishes "no matches" (1) from "error" (2). Treating 2
        # as absence is the whole defect: an unknown `--type`, an unreadable
        # path or a pattern rejected by Rust's regex crate all exit 2 and would
        # report the repository as not containing the pattern. Tested here
        # rather than end-to-end because rg does not exist on this host.
        from harness.search import describe_failure

        failure = describe_failure(
            SearchRequest(pattern="x"),
            engine=engine, exit_code=exit_code, stderr="boom", timed_out=False,
        )
        assert (failure is not None) is expect_error, (engine, exit_code, failure)

    async def test_S101_a_genuine_absence_is_still_no_matches(self, corpus) -> None:
        # The control. Broadening error detection must not turn every empty
        # result into an error.
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler({"pattern": "zzz_definitely_absent"})
        assert "no matches" in result


class TestThePathIsContained:
    """Every other file-touching tool resolves against the workspace root.
    Search took `path` verbatim, so `/etc` and `../` both returned real files."""

    @pytest.mark.parametrize("path", ["/etc", "~/", "../", "a/../../b"])
    async def test_S101_an_escaping_path_is_rejected(self, corpus, path: str) -> None:
        sandbox = await _sandbox(corpus)
        with pytest.raises(ValueError, match="workspace|\\.\\."):
            await grep_tool(sandbox).handler({"pattern": "x", "path": path})

    async def test_S101_glob_rejects_the_same_paths(self, corpus) -> None:
        sandbox = await _sandbox(corpus)
        with pytest.raises(ValueError, match="workspace"):
            await glob_tool(sandbox).handler({"pattern": "*.py", "path": "/etc"})

    async def test_S101_a_path_naming_a_file_is_searched(self, corpus) -> None:
        # os.walk yields nothing for a regular file, so the most natural
        # narrowing after a files_with_matches search -- "now search just that
        # file" -- returned "no matches" every time in the fallback engine.
        sandbox = await _sandbox(corpus)
        result = await grep_tool(sandbox).handler(
            {"pattern": "alpha", "path": "a.py"}
        )
        assert "a.py" in result and "no matches" not in result


class TestVolumeIsBoundedNotJustCount:
    async def test_S101_context_lines_do_not_consume_the_match_budget(
        self, tmp_path
    ) -> None:
        # The engines stop after `limit` matches; the renderer used to cut
        # after `limit` *lines*, so context=2 delivered a third of the stated
        # bound and could end on a bare context line -- which reads as a match
        # on the wrong text.
        for index in range(30):
            (tmp_path / f"f{index}.py").write_text("pad\nneedle\npad2\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler({
            "pattern": "needle", "output_mode": "content",
            "context": 1, "head_limit": 10,
        })
        matches = [line for line in result.splitlines() if ":needle" in line]
        assert len(matches) == 10, result

    async def test_S101_one_enormous_line_cannot_flood_the_result(
        self, tmp_path
    ) -> None:
        # A minified file is a single 280,000-character line. Rendering it
        # whole produced a 100,000-character result under a stated bound of
        # one: the count was bounded and the volume, which is what costs
        # context, was not.
        (tmp_path / "min.js").write_text("var x=1;" + "a" * 280_000 + "needle\n")
        sandbox = await _sandbox(tmp_path)
        result = await grep_tool(sandbox).handler({
            "pattern": "needle", "output_mode": "content", "head_limit": 1,
        })
        assert len(result) < 2_000, len(result)
        assert "…" in result


class TestNothingIsEverInstalled:
    """Acceptance (3). An attached sandbox belongs to whoever attached it;
    mutating it to make a search faster would be the harness modifying a
    benchmark's environment."""

    def test_S101_the_search_module_issues_no_install_command(self) -> None:
        source = Path("harness/search.py").read_text()
        builtin = Path("harness/tools/builtin.py").read_text()
        for forbidden in ("apt-get", "apt install", "pip install", "npm install",
                          "brew install", "curl -", "wget ", "chmod +x"):
            assert forbidden not in source, forbidden
            assert forbidden not in builtin, forbidden

    async def test_S101_an_absent_ripgrep_is_detected_not_installed(
        self, corpus, monkeypatch
    ) -> None:
        commands: list[str] = []
        real = LocalSandbox.exec

        async def spy(self, command, **kwargs):
            commands.append(command)
            return await real(self, command, **kwargs)

        monkeypatch.setattr(LocalSandbox, "exec", spy)
        sandbox = await _sandbox(corpus)
        await grep_tool(sandbox).handler({"pattern": "alpha"})
        assert any("command -v rg" in c for c in commands), commands
        assert not any("install" in c for c in commands), commands


class TestTheEngineIsProbedOnce:
    async def test_S101_the_probe_is_cached(self, corpus, monkeypatch) -> None:
        # A registry is built before the environment probe runs, so the engine
        # is decided at first use. Doing it per call would add a round-trip to
        # every search.
        probes: list[str] = []
        real = LocalSandbox.exec

        async def spy(self, command, **kwargs):
            if "command -v rg" in command:
                probes.append(command)
            return await real(self, command, **kwargs)

        monkeypatch.setattr(LocalSandbox, "exec", spy)
        sandbox = await _sandbox(corpus)
        tool = grep_tool(sandbox)
        await tool.handler({"pattern": "alpha"})
        await tool.handler({"pattern": "beta"})
        await tool.handler({"pattern": "gamma"})
        assert len(probes) == 1, probes


class TestToolSurface:
    def test_S101_search_is_repo_mode_only(self) -> None:
        from harness.profiles import CODING, CODING_REPO

        def names(profile):
            from tests.test_edits import _FakeDeps

            deps = _FakeDeps()
            return {factory(deps).spec.name for factory in profile.tool_factories}

        repo, coding = names(CODING_REPO), names(CODING)
        assert {"grep", "glob"} <= repo
        assert not ({"grep", "glob"} & coding), (
            "promoting search into CODING is Lane B and needs a TB2 run"
        )


def _docker_available() -> bool:
    try:
        from harness.sandbox.docker import DockerSandbox

        return DockerSandbox.availability()
    except Exception:  # noqa: BLE001
        return False


def _image_present() -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", "harness-sandbox:latest"],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not (_docker_available() and _image_present()),
    reason=(
        "needs the harness-sandbox image, which is the only place ripgrep "
        "exists here — `docker build -t harness-sandbox:latest .`"
    ),
)
class TestBothEnginesAgree:
    """Acceptance (2): with no `rg`, results are identical modulo ordering.

    This is the criterion that cannot be tested on the host, because `rg` is
    not installed there — so it runs inside the sandbox image, which has it.
    A skipped test is a weak test, and this one is marked with exactly what to
    run to make it execute rather than a bare `skipif`.
    """

    @staticmethod
    async def _both(directory: Path, arguments: dict) -> tuple[str, str]:
        from harness.sandbox.docker import DockerSandbox
        from harness.tools.builtin import _Engine

        sandbox = DockerSandbox(directory, image="harness-sandbox:latest")
        await sandbox.start()
        try:
            probe = await sandbox.exec("command -v rg", timeout=20)
            assert probe.exit_code == 0, "precondition: the image must have rg"

            with_rg = grep_tool(sandbox)

            # Force the fallback rather than uninstalling anything: the engine
            # choice is the thing under test, and removing rg from the image
            # would make this test depend on mutating a container.
            import harness.tools.builtin as builtin

            engine = _Engine(sandbox)
            engine._has_ripgrep = False
            original = builtin._Engine
            try:
                builtin._Engine = lambda _sandbox: engine
                fallback_tool = grep_tool(sandbox)
                return (
                    await with_rg.handler(dict(arguments)),
                    await fallback_tool.handler(dict(arguments)),
                )
            finally:
                builtin._Engine = original
        finally:
            await sandbox.stop()

    @pytest.mark.parametrize(
        "arguments",
        [
            {"pattern": "alpha"},
            {"pattern": "alpha", "glob": "*.py"},
            {"pattern": "def alpha", "output_mode": "content"},
            {"pattern": "alpha", "output_mode": "count"},
            {"pattern": "ALPHA", "case_insensitive": True},
        ],
    )
    async def test_S101_the_two_engines_return_the_same_paths(
        self, corpus: Path, arguments: dict
    ) -> None:
        rg_output, fallback_output = await self._both(corpus, arguments)

        def paths(text: str) -> set[str]:
            found = set()
            for line in text.splitlines():
                if not line.strip() or line.startswith("["):
                    continue
                name = line.split(":")[0].lstrip("./")
                if name:
                    found.add(name)
            return found

        assert paths(rg_output) == paths(fallback_output), (
            f"\nrg:\n{rg_output}\n\nfallback:\n{fallback_output}"
        )
