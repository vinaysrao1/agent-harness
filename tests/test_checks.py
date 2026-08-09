"""Unit tests for harness.checks (harness-run post-write syntax checks).

Every test here is offline and deterministic: the mapping tests are pure,
the runner tests drive a stub sandbox with scripted :class:`ExecResult`s,
and the few end-to-end tests use :class:`LocalSandbox` with an injected,
frozen :class:`Deadline`. No Docker, no network, no real clock.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from harness.checks import (
    CHECK_OUTPUT_LIMIT,
    CHECK_TIMEOUT_SECONDS,
    SYNTAX_CHECK_EVENT,
    SYNTAX_CHECKERS,
    SyntaxCheckOutcome,
    format_syntax_failure,
    run_syntax_check,
    syntax_check_command,
    syntax_check_language,
    truncate_check_output,
)
from harness.deadline import Deadline
from harness.diligence import looks_unfinished
from harness.sandbox.base import ExecResult
from harness.sandbox.local import LocalSandbox

# ---------------------------------------------------------------------------
# Shared fixtures / stubs
# ---------------------------------------------------------------------------


class StubCheckSandbox:
    """A ``Sandbox`` stub that scripts one ``exec`` result.

    Records every command and timeout it was handed, so a test can assert
    both *what* ran and that nothing ran at all when the check was skipped.
    ``boom=True`` makes ``exec`` raise, which is the fail-open path.
    """

    def __init__(self, result: ExecResult | None = None, boom: bool = False) -> None:
        self.calls: list[tuple[str, float]] = []
        self._result = result or ExecResult(exit_code=0, stdout="", stderr="")
        self._boom = boom

    async def exec(self, command: str, timeout: float = 120) -> ExecResult:
        self.calls.append((command, timeout))
        if self._boom:
            raise RuntimeError("sandbox is gone")
        return self._result


class SpyStore:
    """A ``RunStore`` stand-in recording ``append_event`` calls."""

    def __init__(self, boom: bool = False) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self._boom = boom

    def append_event(self, agent_id: str, kind: str, payload: dict) -> int:
        if self._boom:
            raise RuntimeError("transcript write failed")
        self.events.append((agent_id, kind, payload))
        return len(self.events)


def _fixed_deadline(budget: float, remaining: float | None = None) -> Deadline:
    """A ``Deadline`` of ``budget`` frozen at ``remaining`` seconds left.

    Same construction as ``tests/test_tools.py``'s helper: the clock returns
    0.0 once (at construction) and the elapsed value forever after, so
    ``remaining()`` is exact and never races real time.
    """
    elapsed = 0.0 if remaining is None else budget - remaining
    calls = iter([0.0])
    return Deadline(budget, clock=lambda: next(calls, elapsed))


async def _run(sandbox, path: str, **kwargs) -> str | None:
    return await run_syntax_check(sandbox, path, **kwargs)


def _only_event(store: SpyStore) -> dict:
    assert len(store.events) == 1, store.events
    agent_id, kind, payload = store.events[0]
    assert agent_id == "agent-1"
    assert kind == SYNTAX_CHECK_EVENT
    return payload


# ---------------------------------------------------------------------------
# The pure mapping
# ---------------------------------------------------------------------------


class TestSyntaxCheckCommand:
    """`syntax_check_command` is pure: no sandbox, no clock, no I/O."""

    @pytest.mark.parametrize(
        ("path", "language", "fragment"),
        [
            ("solve.py", "python", "python3 -c "),
            ("run.sh", "bash", "bash -n "),
            ("run.bash", "bash", "bash -n "),
            ("data.json", "json", "python3 -m json.tool "),
            ("app.js", "javascript", "node --check "),
            ("app.mjs", "javascript", "node --check "),
            ("app.cjs", "javascript", "node --check "),
            ("conf.yaml", "yaml", "yaml.compose_all"),
            ("conf.yml", "yaml", "yaml.compose_all"),
        ],
    )
    def test_every_registered_suffix_maps_to_its_checker(
        self, path: str, language: str, fragment: str
    ):
        command = syntax_check_command(path)
        assert command is not None
        assert fragment in command
        assert shlex.quote(path) in command
        assert syntax_check_language(path) == language

    @pytest.mark.parametrize(
        "path",
        [
            "notes.txt",
            "README.md",
            "main.c",
            "Makefile",
            "solve",
            "archive.tar.gz",
            "",
            ".hidden",
        ],
    )
    def test_unknown_suffix_has_no_check(self, path: str):
        # Silence is the default: the overwhelming majority of writes are
        # to files no checker covers, and they must cost nothing.
        assert syntax_check_command(path) is None
        assert syntax_check_language(path) is None

    def test_suffix_match_is_case_insensitive(self):
        assert syntax_check_command("SOLVE.PY") is not None
        assert syntax_check_language("Conf.YAML") == "yaml"

    def test_registry_covers_exactly_the_documented_suffixes(self):
        assert set(SYNTAX_CHECKERS) == {
            ".py",
            ".sh",
            ".bash",
            ".json",
            ".js",
            ".mjs",
            ".cjs",
            ".yaml",
            ".yml",
        }

    def test_path_in_subdirectory_is_matched_by_suffix(self):
        assert syntax_check_language("src/pkg/mod.py") == "python"

    def test_path_with_spaces_is_quoted(self):
        command = syntax_check_command("my dir/my file.py")
        assert command is not None
        assert "'my dir/my file.py'" in command

    def test_shell_metacharacters_in_path_cannot_escape_the_command(self):
        # The path is attacker-adjacent data (the model chose it), so the
        # quoting is a correctness requirement, not a nicety.
        command = syntax_check_command("a; rm -rf /tmp/x.py")
        assert command is not None
        assert "; rm -rf" not in command.replace("'a; rm -rf /tmp/x.py'", "")

    def test_python_check_does_not_use_py_compile(self):
        # Artifact pin. `py_compile` writes __pycache__/*.pyc next to the
        # source; an unexpected extra file in a deliverable directory has
        # already cost this project one task (polyglot-c-py).
        command = syntax_check_command("solve.py")
        assert command is not None
        assert "py_compile" not in command
        assert "compile(" in command

    def test_python_hosted_checks_suppress_tracebacks(self):
        # Without this, PyYAML's ~1200 characters of stack frames arrive
        # first and CHECK_OUTPUT_LIMIT truncates away the diagnostic the
        # check exists to deliver.
        for path in ("solve.py", "conf.yaml"):
            command = syntax_check_command(path)
            assert command is not None
            assert "sys.tracebacklimit=0" in command

    def test_tracebacklimit_is_set_before_the_import(self):
        # So a missing PyYAML reports one clean ModuleNotFoundError line,
        # which is what the "unavailable" triage matches on.
        command = syntax_check_command("conf.yaml")
        assert command is not None
        assert command.index("tracebacklimit") < command.index("import yaml")

    def test_json_check_discards_stdout(self):
        # `json.tool` pretty-prints what it validated; without the redirect
        # a passing check would dump the whole file back into the result.
        command = syntax_check_command("data.json")
        assert command is not None
        assert "> /dev/null" in command

    def test_yaml_check_composes_rather_than_loads(self):
        # `safe_load` is parse + compose-one-document + construct, and both
        # extra phases reject *valid* YAML: multi-document streams and
        # custom tags. `compose_all` is the parse alone. `list(...)` forces
        # the lazy generator -- without it the file is never read.
        for path in ("conf.yaml", "conf.yml"):
            command = syntax_check_command(path)
            assert command is not None
            assert "safe_load" not in command
            assert "list(yaml.compose_all(" in command


class TestJsoncFilenamesAreExempt:
    """JSON-with-comments filenames get no check at all.

    `json.tool` is strict RFC 8259, but tsconfig/devcontainer/.vscode files
    are read by tools that document comment support, so a `//` comment in
    one is correct work and reporting it is a false positive.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "tsconfig.json",
            "tsconfig.build.json",
            "TSConfig.json",
            "jsconfig.json",
            "devcontainer.json",
            ".eslintrc.json",
            ".vscode/settings.json",
            ".vscode/launch.json",
            "repo/.devcontainer/devcontainer.json",
            "web/.vscode/tasks.json",
        ],
    )
    def test_jsonc_filenames_have_no_check(self, path: str):
        assert syntax_check_command(path) is None
        assert syntax_check_language(path) is None

    @pytest.mark.parametrize(
        "path",
        [
            "data.json",
            "package.json",
            "src/fixtures/expected.json",
            "vsconfig.json",
            "config/my-tsconfig.json",
            "vscode/settings.json",
        ],
    )
    def test_ordinary_json_is_still_checked(self, path: str):
        assert syntax_check_language(path) == "json"

    def test_the_exemption_is_scoped_to_json(self):
        # The directory rule must not disable checking for every file that
        # happens to live under .vscode/.
        assert syntax_check_language(".vscode/hook.py") == "python"
        assert syntax_check_language(".devcontainer/setup.sh") == "bash"


# ---------------------------------------------------------------------------
# Output truncation and the appended text
# ---------------------------------------------------------------------------


class TestTruncateCheckOutput:
    def test_short_output_is_unchanged(self):
        assert truncate_check_output("boom") == "boom"

    def test_output_at_the_limit_is_unchanged(self):
        text = "x" * CHECK_OUTPUT_LIMIT
        assert truncate_check_output(text) == text

    def test_long_output_keeps_the_head_and_names_the_drop(self):
        text = "HEAD" + "x" * (CHECK_OUTPUT_LIMIT * 3)
        truncated = truncate_check_output(text)
        assert truncated.startswith("HEAD")
        assert "chars truncated" in truncated
        # The head is kept, not the tail: a parser reports the first syntax
        # error first and everything after it is cascade.
        assert len(truncated) < len(text)


class TestFormatSyntaxFailure:
    def _outcome(self, output: str = "SyntaxError: invalid syntax") -> SyntaxCheckOutcome:
        return SyntaxCheckOutcome(
            path="solve.py",
            language="python",
            ok=False,
            exit_code=1,
            skipped_reason=None,
            output=output,
        )

    def test_names_the_harness_the_path_and_the_exit_code(self):
        text = format_syntax_failure(self._outcome())
        assert text.startswith("Syntax check failed (harness-run):")
        assert "solve.py" in text
        assert "python" in text
        assert "exited 1" in text
        assert "SyntaxError: invalid syntax" in text

    def test_does_not_trip_looks_unfinished(self):
        # The model may quote this back; it must not read as a promise or a
        # question (harness.diligence._PROMISE_PATTERNS).
        text = format_syntax_failure(self._outcome())
        unfinished, reason = looks_unfinished(text, 0)
        assert unfinished is False, reason

    def test_never_ends_in_a_question_even_when_the_compiler_does(self):
        # The closing sentence follows the checker's output precisely so
        # the block never ends on punctuation a compiler chose.
        text = format_syntax_failure(self._outcome(output="what is this?"))
        assert not text.rstrip().endswith("?")
        assert looks_unfinished(text, 0)[0] is False

    def test_names_the_partial_write_case(self):
        text = format_syntax_failure(self._outcome())
        assert "written in pieces" in text


# ---------------------------------------------------------------------------
# The runner: what reaches the model
# ---------------------------------------------------------------------------


class TestRunSyntaxCheckSilence:
    async def test_unknown_suffix_runs_nothing_and_records_nothing(self):
        sandbox = StubCheckSandbox()
        store = SpyStore()
        assert await _run(sandbox, "notes.txt", store=store, agent_id="agent-1") is None
        assert sandbox.calls == []
        assert store.events == []

    async def test_clean_file_appends_nothing_and_records_ok(self):
        sandbox = StubCheckSandbox(ExecResult(exit_code=0, stdout="", stderr=""))
        store = SpyStore()
        assert await _run(sandbox, "solve.py", store=store, agent_id="agent-1") is None
        assert _only_event(store) == {
            "path": "solve.py",
            "language": "python",
            "ok": True,
            "exit_code": 0,
            "skipped_reason": None,
            "tool": None,
        }

    async def test_check_is_bounded_by_the_check_timeout(self):
        sandbox = StubCheckSandbox()
        await _run(sandbox, "solve.py")
        assert sandbox.calls == [
            (syntax_check_command("solve.py"), CHECK_TIMEOUT_SECONDS)
        ]


class TestRunSyntaxCheckFailure:
    async def test_genuine_failure_is_appended_and_recorded(self):
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr='  File "solve.py", line 3\nSyntaxError: invalid syntax',
            )
        )
        store = SpyStore()
        appended = await _run(
            sandbox, "solve.py", store=store, agent_id="agent-1", tool="write_file"
        )
        assert appended is not None
        assert appended.startswith("Syntax check failed (harness-run):")
        assert "SyntaxError: invalid syntax" in appended
        assert _only_event(store) == {
            "path": "solve.py",
            "language": "python",
            "ok": False,
            "exit_code": 1,
            "skipped_reason": None,
            "tool": "write_file",
        }

    async def test_stdout_only_diagnostics_are_still_reported(self):
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=2, stdout="parse error at line 9", stderr="")
        )
        appended = await _run(sandbox, "run.sh")
        assert appended is not None
        assert "parse error at line 9" in appended

    async def test_huge_diagnostics_are_truncated(self):
        noise = "E" * (CHECK_OUTPUT_LIMIT * 5)
        sandbox = StubCheckSandbox(ExecResult(exit_code=1, stdout="", stderr=noise))
        appended = await _run(sandbox, "solve.py")
        assert appended is not None
        assert "chars truncated" in appended
        # A pathological compiler cannot blow up the transcript: the
        # appended text is the framing plus at most the output limit.
        assert len(appended) < CHECK_OUTPUT_LIMIT + 500

    async def test_a_failing_check_still_returns_when_the_store_raises(self):
        # Telemetry for an unrequested check must never turn a successful
        # write into an error result.
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=1, stdout="", stderr="SyntaxError: bad")
        )
        appended = await _run(
            sandbox, "solve.py", store=SpyStore(boom=True), agent_id="agent-1"
        )
        assert appended is not None


class TestRunSyntaxCheckFailsOpen:
    """The pin: a missing or confused checker never reaches the model."""

    async def test_exit_127_appends_nothing(self):
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=127, stdout="", stderr="node: command not found")
        )
        store = SpyStore()
        assert await _run(sandbox, "app.js", store=store, agent_id="agent-1") is None
        payload = _only_event(store)
        assert payload["ok"] is None
        assert payload["skipped_reason"] == "unavailable"
        assert payload["exit_code"] == 127

    async def test_command_not_found_on_any_exit_code_appends_nothing(self):
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=1, stdout="", stderr="sh: node: not found")
        )
        store = SpyStore()
        assert await _run(sandbox, "app.js", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "unavailable"

    async def test_missing_python_module_appends_nothing(self):
        # PyYAML absent is "no checker", not "your yaml is broken".
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'yaml'",
            )
        )
        store = SpyStore()
        assert await _run(sandbox, "conf.yaml", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "unavailable"

    async def test_esm_dialect_complaint_appends_nothing(self):
        # `node --check` parses a bare .js as CommonJS, so a valid ES module
        # "fails". Reporting that is a false positive on correct work.
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr="SyntaxError: Cannot use import statement outside a module",
            )
        )
        store = SpyStore()
        assert await _run(sandbox, "app.js", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "inconclusive"

    async def test_node_dialect_complaint_survives_its_stack_and_banner(self):
        # The real `node --check` output is not one line: the SyntaxError is
        # followed by stack frames and a `Node.js vX` banner. Triage that
        # only inspected the last line would miss this and report a valid ES
        # module as broken.
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr=(
                    "/w/app.js:1\n"
                    "import fs from 'fs';\n"
                    "^^^^^^\n"
                    "\n"
                    "SyntaxError: Cannot use import statement outside a module\n"
                    "    at wrapSafe (node:internal/modules/cjs/loader:1464:18)\n"
                    "    at checkSyntax (node:internal/main/check_syntax:78:3)\n"
                    "\n"
                    "Node.js v20.19.6"
                ),
            )
        )
        store = SpyStore()
        assert await _run(sandbox, "app.js", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "inconclusive"

    async def test_non_zero_without_diagnostics_appends_nothing(self):
        sandbox = StubCheckSandbox(ExecResult(exit_code=3, stdout="", stderr="  \n"))
        store = SpyStore()
        assert await _run(sandbox, "solve.py", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "no_diagnostics"

    async def test_timed_out_check_appends_nothing(self):
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        )
        store = SpyStore()
        assert await _run(sandbox, "solve.py", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "timeout"

    async def test_exec_that_raises_appends_nothing_and_does_not_propagate(self):
        sandbox = StubCheckSandbox(boom=True)
        store = SpyStore()
        assert await _run(sandbox, "solve.py", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "exec_error"

    async def test_no_store_is_a_silent_no_op(self):
        sandbox = StubCheckSandbox(ExecResult(exit_code=0, stdout="", stderr=""))
        assert await _run(sandbox, "solve.py") is None
        assert await _run(sandbox, "solve.py", agent_id="agent-1") is None


class TestEchoedSourceCannotSuppressTheReport:
    """The file's own contents must not decide the file's own verdict.

    Every checker here echoes the offending source line back before saying
    what is wrong with it. Matching the "checker unavailable" markers
    against that whole blob handed the decision to whatever the agent wrote:
    a genuine syntax error in error-handling code -- `print("... not
    found")`, `raise ... "permission denied"` -- was silently swallowed
    *and* logged as a missing interpreter, so the telemetry said the
    mechanism could not run when in fact it had run and found the bug.
    """

    def _python_error(self, source_line: str, message: str) -> str:
        """The exact shape CPython prints under ``sys.tracebacklimit = 0``."""
        return (
            '  File "solve.py", line 3\n'
            f"    {source_line}\n"
            "          ^\n"
            f"SyntaxError: {message}"
        )

    @pytest.mark.parametrize(
        "source_line",
        [
            'print("error: input file not found)',
            'raise OSError("no such file or directory: %s" % p)',
            'sys.exit("permission denied)',
            'log("cannot execute the plan)',
            'ERR = "ImportError: no module named widgets"',
        ],
    )
    async def test_marker_text_inside_the_broken_line_is_still_reported(
        self, source_line: str
    ):
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr=self._python_error(source_line, "unterminated string literal"),
            )
        )
        store = SpyStore()
        appended = await _run(sandbox, "solve.py", store=store, agent_id="agent-1")
        assert appended is not None, source_line
        assert "unterminated string literal" in appended
        payload = _only_event(store)
        assert payload["ok"] is False
        assert payload["skipped_reason"] is None

    def _node_error(self, source_line: str) -> str:
        """The exact shape ``node --check`` prints.

        The difference that matters against :meth:`_python_error`: node
        echoes the offending line at **column 0**, so "echoed source is
        indented" is not a property the guard may rely on.
        """
        return (
            "/w/app.js:2\n"
            f"{source_line}\n"
            "       ^^^^^^^^^^^^^^^^^^^^^^\n"
            "\n"
            "SyntaxError: Invalid or unexpected token\n"
            "    at wrapSafe (node:internal/modules/cjs/loader:1464:18)\n"
            "    at checkSyntax (node:internal/main/check_syntax:78:3)\n"
            "\n"
            "Node.js v20.19.6"
        )

    @pytest.mark.parametrize("path", ["app.js", "app.mjs", "app.cjs"])
    @pytest.mark.parametrize(
        "source_line",
        [
            'error: "not found: bad input,',
            'ENOENT: "no such file or directory,',
            'denied: "permission denied for that path,',
            'cannot: "cannot execute the plan,',
        ],
    )
    async def test_column_zero_echo_carrying_a_marker_is_still_reported(
        self, source_line: str, path: str
    ):
        # Each of these column-0 property lines matches the verdict-line
        # shape on its `word: ` prefix, so shape alone readmits the file's
        # own contents and swallows a real JS syntax error. Only the caret
        # ruler underneath it says the line came out of the file.
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=1, stdout="", stderr=self._node_error(source_line))
        )
        store = SpyStore()
        appended = await _run(sandbox, path, store=store, agent_id="agent-1")
        assert appended is not None, source_line
        assert "Invalid or unexpected token" in appended
        payload = _only_event(store)
        assert payload["ok"] is False
        assert payload["skipped_reason"] is None

    async def test_echo_without_a_caret_line_is_also_ignored(self):
        # IndentationError prints no caret ruler, so the rule cannot be
        # "drop the line above the carets".
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr=(
                    '  File "solve.py", line 2\n'
                    '    y = handle("file not found")\n'
                    "IndentationError: unexpected indent"
                ),
            )
        )
        appended = await _run(sandbox, "solve.py")
        assert appended is not None
        assert "IndentationError: unexpected indent" in appended

    async def test_a_real_unavailable_line_still_wins_over_the_echo(self):
        # The narrowing must not cost the fail-open behaviour it protects:
        # the checker's own last line still decides.
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'yaml'",
            )
        )
        store = SpyStore()
        assert await _run(sandbox, "conf.yaml", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "unavailable"

    async def test_unreadable_file_is_unavailable_not_a_syntax_error(self):
        sandbox = StubCheckSandbox(
            ExecResult(
                exit_code=1,
                stdout="",
                stderr=(
                    "FileNotFoundError: [Errno 2] No such file or directory: 'x.py'"
                ),
            )
        )
        store = SpyStore()
        assert await _run(sandbox, "x.py", store=store, agent_id="agent-1") is None
        assert _only_event(store)["skipped_reason"] == "unavailable"


class TestRunSyntaxCheckDeadline:
    async def test_scarce_remaining_time_skips_the_check_entirely(self):
        deadline = _fixed_deadline(3600.0, remaining=65.0)
        assert deadline.affordable_exec_seconds() < CHECK_TIMEOUT_SECONDS
        sandbox = StubCheckSandbox()
        store = SpyStore()
        result = await _run(
            sandbox,
            "solve.py",
            deadline=deadline,
            store=store,
            agent_id="agent-1",
        )
        assert result is None
        # Not merely unreported -- never run. The landing reserve is not
        # this mechanism's to spend.
        assert sandbox.calls == []
        payload = _only_event(store)
        assert payload["skipped_reason"] == "deadline"
        assert payload["ok"] is None
        assert payload["exit_code"] is None

    async def test_landing_turn_skips_the_check(self):
        deadline = _fixed_deadline(3600.0)
        deadline.begin_landing()
        sandbox = StubCheckSandbox()
        store = SpyStore()
        assert (
            await _run(
                sandbox,
                "solve.py",
                deadline=deadline,
                store=store,
                agent_id="agent-1",
            )
            is None
        )
        assert sandbox.calls == []
        assert _only_event(store)["skipped_reason"] == "landing"

    async def test_ample_remaining_time_runs_the_check(self):
        deadline = _fixed_deadline(3600.0)
        sandbox = StubCheckSandbox(
            ExecResult(exit_code=1, stdout="", stderr="SyntaxError: bad")
        )
        appended = await _run(sandbox, "solve.py", deadline=deadline)
        assert appended is not None
        assert len(sandbox.calls) == 1

    async def test_deadline_without_a_budget_never_skips(self):
        deadline = Deadline(None)
        sandbox = StubCheckSandbox()
        await _run(sandbox, "solve.py", deadline=deadline)
        assert len(sandbox.calls) == 1

    async def test_caller_supplied_skip_reason_short_circuits(self):
        sandbox = StubCheckSandbox()
        store = SpyStore()
        assert (
            await _run(
                sandbox,
                "solve.py",
                store=store,
                agent_id="agent-1",
                skip_reason="append_mode",
            )
            is None
        )
        assert sandbox.calls == []
        assert _only_event(store)["skipped_reason"] == "append_mode"


# ---------------------------------------------------------------------------
# End to end against a real interpreter (still offline: no network, no Docker)
# ---------------------------------------------------------------------------


class TestRunSyntaxCheckAgainstLocalSandbox:
    @pytest.fixture
    async def sandbox(self, tmp_path: Path) -> LocalSandbox:
        box = LocalSandbox(tmp_path / "workspace")
        await box.start()
        return box

    async def test_real_syntax_error_is_caught_and_reported(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("solve.py", "def f(:\n")
        appended = await run_syntax_check(sandbox, "solve.py")
        assert appended is not None
        assert "SyntaxError" in appended

    async def test_reported_output_is_the_diagnostic_not_a_stack_trace(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("solve.py", "def f(:\n")
        appended = await run_syntax_check(sandbox, "solve.py")
        assert appended is not None
        assert "Traceback (most recent call last)" not in appended
        assert "SyntaxError: invalid syntax" in appended
        # Small enough that the truncation limit is never the binding
        # constraint on a real one-error file.
        assert len(appended) < CHECK_OUTPUT_LIMIT

    async def test_real_valid_file_says_nothing(self, sandbox: LocalSandbox):
        await sandbox.write_file("solve.py", "def f():\n    return 1\n")
        assert await run_syntax_check(sandbox, "solve.py") is None

    async def test_check_leaves_no_artifacts_in_the_workspace(
        self, sandbox: LocalSandbox
    ):
        # The polyglot-c-py pin: an extra file in the deliverable directory
        # loses the task. `compile()` writes nothing; `py_compile` would
        # have left __pycache__/solve.cpython-*.pyc right here.
        await sandbox.write_file("solve.py", "def f():\n    return 1\n")
        before = sorted(p.name for p in sandbox.workspace.rglob("*"))
        await run_syntax_check(sandbox, "solve.py")
        after = sorted(p.name for p in sandbox.workspace.rglob("*"))
        assert after == before == ["solve.py"]
        assert not (sandbox.workspace / "__pycache__").exists()

    async def test_check_leaves_no_artifacts_when_the_file_is_broken(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("solve.py", "def f(:\n")
        await run_syntax_check(sandbox, "solve.py")
        assert sorted(p.name for p in sandbox.workspace.rglob("*")) == ["solve.py"]

    async def test_real_json_failure_is_caught_and_stdout_is_discarded(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("data.json", '{"a": }')
        appended = await run_syntax_check(sandbox, "data.json")
        assert appended is not None
        assert "Expecting value" in appended

    async def test_real_valid_json_does_not_echo_the_document(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("data.json", '{"a": 1}')
        assert await run_syntax_check(sandbox, "data.json") is None

    async def test_real_shell_failure_is_caught(self, sandbox: LocalSandbox):
        await sandbox.write_file("run.sh", "if [ x ; then\n")
        appended = await run_syntax_check(sandbox, "run.sh")
        assert appended is not None
        assert "syntax error" in appended.lower()

    async def test_path_with_a_space_is_checked_not_mangled(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("my dir/my file.py", "def f(:\n")
        appended = await run_syntax_check(sandbox, "my dir/my file.py")
        assert appended is not None
        assert "SyntaxError" in appended

    async def test_real_broken_file_containing_not_found_is_reported(
        self, sandbox: LocalSandbox
    ):
        # End-to-end proof of the swallowed-report bug: a real CPython
        # SyntaxError whose echoed source line carries an "unavailable"
        # marker. Control case (`def f(:`) is covered above, so a pass here
        # cannot be explained by the check simply never running.
        await sandbox.write_file(
            "swallow.py",
            'import sys\ndef main():\n    print("error: input file not found)\n',
        )
        store = SpyStore()
        appended = await run_syntax_check(
            sandbox, "swallow.py", store=store, agent_id="agent-1"
        )
        assert appended is not None
        assert "SyntaxError" in appended
        payload = _only_event(store)
        assert payload["ok"] is False
        assert payload["skipped_reason"] is None

    async def test_real_jsonc_file_is_not_checked_at_all(
        self, sandbox: LocalSandbox
    ):
        # `json.tool` genuinely rejects this file; the point is that no
        # check runs, so a correct tsconfig never produces a report.
        await sandbox.write_file(
            "tsconfig.json", '{\n  // target\n  "compilerOptions": {}\n}\n'
        )
        store = SpyStore()
        assert (
            await run_syntax_check(
                sandbox, "tsconfig.json", store=store, agent_id="agent-1"
            )
            is None
        )
        assert store.events == []


class TestJavaScriptChecksAgainstLocalSandbox:
    """`node --check` echoes at column 0, so the guard cannot read that echo."""

    @pytest.fixture
    async def sandbox(self, tmp_path: Path) -> LocalSandbox:
        box = LocalSandbox(tmp_path / "workspace")
        await box.start()
        probe = await box.exec("node --version", timeout=30)
        if probe.exit_code != 0:
            pytest.skip("node is not on the sandbox's PATH")
        return box

    @pytest.mark.parametrize("name", ["swallow.js", "swallow.mjs"])
    async def test_real_column_zero_broken_line_with_a_marker_is_reported(
        self, sandbox: LocalSandbox, name: str
    ):
        # End-to-end proof against the real interpreter. The indented control
        # below differs from this file *only* in the indentation of the
        # broken line, and it was always reported; this one was silently
        # swallowed and logged skipped_reason="unavailable", i.e. the log
        # called a working node a missing one.
        await sandbox.write_file(
            name, 'module.exports = {\nerror: "not found: bad input,\n}\n'
        )
        store = SpyStore()
        appended = await run_syntax_check(
            sandbox, name, store=store, agent_id="agent-1"
        )
        assert appended is not None
        assert "SyntaxError" in appended
        payload = _only_event(store)
        assert payload["ok"] is False
        assert payload["skipped_reason"] is None

    async def test_real_indented_broken_line_with_a_marker_is_reported(
        self, sandbox: LocalSandbox
    ):
        # The control for the pair above.
        await sandbox.write_file(
            "indented.js", 'module.exports = {\n  error: "not found: bad input,\n}\n'
        )
        appended = await run_syntax_check(sandbox, "indented.js")
        assert appended is not None
        assert "SyntaxError" in appended

    async def test_real_valid_commonjs_says_nothing(self, sandbox: LocalSandbox):
        await sandbox.write_file("app.js", "const a = 1;\nmodule.exports = a;\n")
        assert await run_syntax_check(sandbox, "app.js") is None

    async def test_real_valid_es_module_says_nothing(self, sandbox: LocalSandbox):
        await sandbox.write_file(
            "app.mjs", 'import fs from "fs";\nexport default fs;\n'
        )
        assert await run_syntax_check(sandbox, "app.mjs") is None

    async def test_real_es_module_named_js_is_never_reported(
        self, sandbox: LocalSandbox
    ):
        # Older node parses a bare .js as CommonJS and rejects this correct
        # file ("Cannot use import statement outside a module"); newer node
        # sniffs the module syntax and accepts it. Either way the model must
        # never be told a correct file is broken, so the assertion is on the
        # verdict rather than on which node happens to be installed.
        await sandbox.write_file("esm.js", 'import fs from "fs";\nexport default fs;\n')
        store = SpyStore()
        assert (
            await run_syntax_check(sandbox, "esm.js", store=store, agent_id="agent-1")
            is None
        )
        assert _only_event(store)["ok"] is not False


class TestYamlChecksAgainstLocalSandbox:
    """The YAML checker must not report files that are valid YAML."""

    @pytest.fixture
    async def sandbox(self, tmp_path: Path) -> LocalSandbox:
        box = LocalSandbox(tmp_path / "workspace")
        await box.start()
        probe = await box.exec("python3 -c 'import yaml'", timeout=30)
        if probe.exit_code != 0:
            pytest.skip("PyYAML is not importable by the sandbox's python3")
        return box

    async def test_multi_document_stream_says_nothing(self, sandbox: LocalSandbox):
        # `---`-separated documents are the normal form for Kubernetes
        # manifest bundles and Ansible playbooks. `safe_load` rejected them
        # with "expected a single document in the stream".
        await sandbox.write_file("manifests.yaml", "a: 1\n---\nb: 2\n")
        assert await run_syntax_check(sandbox, "manifests.yaml") is None

    async def test_custom_tags_say_nothing(self, sandbox: LocalSandbox):
        # !Ref/!GetAtt/!Sub appear in every CloudFormation template.
        # `safe_load` rejected them with "could not determine a constructor".
        await sandbox.write_file(
            "template.yml", "X: !Ref Foo\nY: !GetAtt [A, B]\n"
        )
        assert await run_syntax_check(sandbox, "template.yml") is None

    async def test_valid_yaml_says_nothing(self, sandbox: LocalSandbox):
        await sandbox.write_file("conf.yaml", "a: 1\nb:\n  - x\n  - y\n")
        assert await run_syntax_check(sandbox, "conf.yaml") is None

    async def test_malformed_yaml_is_still_reported(self, sandbox: LocalSandbox):
        # The pin against fixing the false positives by checking nothing.
        await sandbox.write_file("broken.yaml", "a: [1, 2\n")
        appended = await run_syntax_check(sandbox, "broken.yaml")
        assert appended is not None
        assert "ParserError" in appended

    async def test_the_reported_yaml_error_is_not_a_stack_trace(
        self, sandbox: LocalSandbox
    ):
        await sandbox.write_file("broken.yaml", "a: [1, 2\n")
        appended = await run_syntax_check(sandbox, "broken.yaml")
        assert appended is not None
        assert "Traceback (most recent call last)" not in appended
        assert len(appended) < CHECK_OUTPUT_LIMIT

    async def test_yaml_check_leaves_no_artifacts(self, sandbox: LocalSandbox):
        await sandbox.write_file("conf.yaml", "a: 1\n")
        await run_syntax_check(sandbox, "conf.yaml")
        assert sorted(p.name for p in sandbox.workspace.rglob("*")) == ["conf.yaml"]
