"""Tests for harness.cli.

No network, no API keys, no Docker daemon: ``DockerSandbox.availability``
is patched to False, HARNESS_HOME points at ``tmp_path``, and every model
the CLI builds is the ``fake`` adapter driven by a JSONL script — real
provider adapters are never constructed.
"""

from __future__ import annotations

import asyncio
import io
import re
import sys
from pathlib import Path

import pytest

from harness.adapters.fake import FakeAdapter
from harness.cli import main, make_ask
from tests.test_eval_pr_replay import host_run
from harness.config import HarnessConfig, load_config
from harness.loop import Budgets
from harness.orchestrator import Orchestrator
from harness.permissions import ToolMeta
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    Usage,
)

#: LocalSandbox-fallback warnings fire on every CLI run in these tests.
pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")


@pytest.fixture(autouse=True)
def no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the LocalSandbox fallback: tests must not need a Docker daemon."""
    monkeypatch.setattr(
        DockerSandbox, "availability", classmethod(lambda cls: False)
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HARNESS_HOME at a throwaway directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HARNESS_HOME", str(home))
    return home


def write_fake_config(home: Path, script_lines: list[str]) -> Path:
    """Write a JSONL FakeAdapter script plus a config.toml registering it
    as model ``fake``; return the config path."""
    script = home / "script.jsonl"
    script.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    config = home / "config.toml"
    config.write_text(
        f'[models.fake]\nadapter = "fake"\nmodel = "{script}"\n',
        encoding="utf-8",
    )
    return config


# -- run ---------------------------------------------------------------------


def test_run_end_to_end(home: Path, capsys: pytest.CaptureFixture) -> None:
    """`harness run` executes the scripted task, prints the final text, run
    id, and usage summary, and leaves the artifact in the run workspace."""
    write_fake_config(
        home,
        [
            '{"tool_calls": [{"name": "write_file", '
            '"arguments": {"path": "out.txt", "content": "cli"}}]}',
            '{"content": "Task complete. Wrote out.txt."}',
        ],
    )
    exit_code = main(["run", "Write out.txt.", "--model", "fake"])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Task complete. Wrote out.txt." in out
    assert "status: completed" in out
    assert "usage:" in out
    match = re.search(r"run id: ([0-9a-f]{32})", out)
    assert match is not None
    run_id = match.group(1)

    # Artifact landed in the default run workspace under HARNESS_HOME.
    assert (home / "runs" / run_id / "workspace" / "out.txt").read_text() == "cli"

    # `harness runs` lists it; `harness cost` aggregates it.
    assert main(["runs"]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "completed" in out
    assert "Write out.txt." in out

    assert main(["cost", run_id]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "usage:" in out


def test_run_explicit_workspace(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """`--workspace` overrides the default workspace location."""
    write_fake_config(
        home,
        [
            '{"tool_calls": [{"name": "write_file", '
            '"arguments": {"path": "out.txt", "content": "here"}}]}',
            '{"content": "Task complete."}',
        ],
    )
    workspace = tmp_path / "elsewhere"
    exit_code = main(
        ["run", "goal", "--model", "fake", "--workspace", str(workspace)]
    )
    assert exit_code == 0
    assert (workspace / "out.txt").read_text() == "here"


def test_run_missing_config_file(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """An explicitly named but missing config exits 2 with a clean message."""
    exit_code = main(
        ["run", "goal", "--model", "m", "--config", str(tmp_path / "nope.toml")]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "not found" in err
    assert "Traceback" not in err


def test_run_invalid_config(
    home: Path, capsys: pytest.CaptureFixture
) -> None:
    """Malformed TOML exits 2 with a clean message, never a traceback."""
    (home / "config.toml").write_text("[models\n", encoding="utf-8")
    exit_code = main(["run", "goal", "--model", "m"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_run_unknown_model(home: Path, capsys: pytest.CaptureFixture) -> None:
    """A model name missing from the registry exits 2 and names the model."""
    exit_code = main(["run", "goal", "--model", "nope"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "Traceback" not in err


# -- runs / cost -------------------------------------------------------------


def test_runs_empty(home: Path, capsys: pytest.CaptureFixture) -> None:
    """`harness runs` on a fresh store reports no runs."""
    assert main(["runs"]) == 0
    assert "no runs recorded" in capsys.readouterr().out


def test_runs_and_cost_seeded_store(
    home: Path, capsys: pytest.CaptureFixture
) -> None:
    """`harness runs`/`harness cost` read a store seeded out-of-band."""
    with RunStore(home / "state.db") as store:
        run_id = store.create_run(
            "Seeded goal.", "opus", "gated", status="completed"
        )
        store.record_usage(
            run_id, None, "opus", Usage(input_tokens=100, output_tokens=40)
        )
        store.record_usage(
            run_id, None, "opus", Usage(input_tokens=50, output_tokens=10)
        )

    assert main(["runs"]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "Seeded goal." in out
    assert "opus" in out

    assert main(["cost", run_id]) == 0
    out = capsys.readouterr().out
    assert "opus: 2 call(s)" in out
    assert "input=150" in out
    assert "output=50" in out


def test_cost_unknown_run(home: Path, capsys: pytest.CaptureFixture) -> None:
    """`harness cost` with an unknown run id exits 2 cleanly."""
    exit_code = main(["cost", "deadbeef"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "deadbeef" in err
    assert "Traceback" not in err


# -- ask flow ----------------------------------------------------------------


def _ask_with_stdin(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    stdin_text: str,
    tool_name: str = "bash",
) -> tuple[bool, Orchestrator]:
    """Run one make_ask() approval round with scripted stdin."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    with RunStore(home / "state.db") as store:
        orchestrator = Orchestrator(HarnessConfig(home=home), store)
        ask = make_ask(orchestrator)
        approved = asyncio.run(
            ask(tool_name, {"command": "ls"}, ToolMeta(side_effect=True))
        )
    return approved, orchestrator


def test_ask_yes(
    monkeypatch: pytest.MonkeyPatch, home: Path, capsys: pytest.CaptureFixture
) -> None:
    """'y' approves once, and the prompt shows the tool name + arguments."""
    approved, orchestrator = _ask_with_stdin(monkeypatch, home, "y\n")
    assert approved is True
    assert orchestrator._grants == []
    out = capsys.readouterr().out
    assert "bash" in out
    assert "command" in out


def test_ask_no(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """'n' denies."""
    approved, _ = _ask_with_stdin(monkeypatch, home, "n\n")
    assert approved is False


def test_ask_always_grants(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """'a' approves and grants the tool pattern for the rest of the run."""
    approved, orchestrator = _ask_with_stdin(monkeypatch, home, "a\n")
    assert approved is True
    assert orchestrator._grants == ["bash"]


def test_ask_reprompts_on_garbage(
    monkeypatch: pytest.MonkeyPatch, home: Path, capsys: pytest.CaptureFixture
) -> None:
    """Unrecognized input re-prompts until a valid choice arrives."""
    approved, _ = _ask_with_stdin(monkeypatch, home, "wat\ny\n")
    assert approved is True
    assert "unrecognized choice" in capsys.readouterr().out


def test_ask_eof_denies(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """EOF on stdin (non-interactive session) denies."""
    approved, _ = _ask_with_stdin(monkeypatch, home, "")
    assert approved is False


def test_ask_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """Regression: the approval prompt reads stdin off-loop
    (asyncio.to_thread), so an unanswered prompt must not freeze the event
    loop — concurrent coroutines (subagents, HTTP responses, timeouts) keep
    running while the prompt waits."""
    import builtins
    import threading

    release = threading.Event()
    outcomes: list[str] = []

    def blocking_input(prompt: str = "") -> str:
        # Waits until the *event loop* sets `release`. If input() ran on
        # the loop thread, the loop could never set it and this would time
        # out — the regression this test guards against.
        got = release.wait(timeout=5)
        outcomes.append("released" if got else "timed_out")
        return "y"

    monkeypatch.setattr(builtins, "input", blocking_input)

    async def main() -> bool:
        with RunStore(home / "state.db") as store:
            orchestrator = Orchestrator(HarnessConfig(home=home), store)
            ask = make_ask(orchestrator)
            task = asyncio.create_task(
                ask("bash", {"command": "ls"}, ToolMeta(side_effect=True))
            )
            # Only reachable while the prompt is pending if the loop is free.
            await asyncio.sleep(0.05)
            release.set()
            return await asyncio.wait_for(task, timeout=5)

    approved = asyncio.run(main())
    assert approved is True
    assert outcomes == ["released"]


# -- resume ------------------------------------------------------------------


def test_resume_unknown_run(home: Path, capsys: pytest.CaptureFixture) -> None:
    """`harness resume` with an unknown run id exits 2 cleanly."""
    exit_code = main(["resume", "deadbeef"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "deadbeef" in err
    assert "Traceback" not in err


def test_resume_continues_paused_run(
    home: Path, capsys: pytest.CaptureFixture
) -> None:
    """A run paused mid-task (seeded via the orchestrator) resumes through
    the CLI using the config's fake model and completes."""
    # The config's script holds only the *finishing* response — exactly what
    # the resumed run still needs.
    write_fake_config(
        home, ['{"content": "Task complete. Resumed and finished."}']
    )

    # Seed a paused run directly through the orchestrator (adapter injected,
    # budget of one turn forces a pause after the first tool call).
    seed_adapter = FakeAdapter(
        [
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "a.txt", "content": "x"},
                        )
                    ],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
        ]
    )
    with RunStore(home / "state.db") as store:
        orchestrator = Orchestrator(load_config(), store)
        run_id, paused = asyncio.run(
            orchestrator.run_task(
                "Finish the thing.",
                "fake",
                adapter_override=seed_adapter,
                budgets=Budgets(max_turns=1),
            )
        )
    assert paused.status == "paused_budget"
    capsys.readouterr()  # discard seeding output, if any

    exit_code = main(["resume", run_id])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Task complete. Resumed and finished." in out
    assert "status: completed" in out

    with RunStore(home / "state.db") as store:
        assert store.get_run(run_id).status == "completed"


# -- harness eval (S-401) -----------------------------------------------------


class TestEvalCommand:
    """The command exists to make the eval runnable; what it must not do is
    print a pass rate over a suite that quietly shrank."""

    def _suite_file(self, tmp_path, **overrides):
        import json

        payload = {
            "name": "toy",
            "repo": str(tmp_path / "repo"),
            "revs": ["HEAD"],
            "test_command": "true",
        }
        payload.update(overrides)
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(payload))
        return path

    def test_S401_a_malformed_suite_exits_cleanly(self, tmp_path, capsys) -> None:
        from harness.cli import main

        bad = tmp_path / "suite.json"
        bad.write_text('{"name": "toy"}')
        code = main(["eval", str(bad), "--model", "fake-model"])
        assert code == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "Traceback" not in err

    def test_S401_a_missing_suite_exits_cleanly(self, tmp_path, capsys) -> None:
        from harness.cli import main

        code = main(["eval", str(tmp_path / "nope.json"), "--model", "fake-model"])
        assert code == 2
        assert capsys.readouterr().err.startswith("error: ")

    def test_S401_revisions_that_produced_no_task_are_reported(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        # generate() skips a rev it cannot resolve. Silently, the suite shrinks
        # and still prints a pass rate.
        import subprocess

        from harness.cli import main

        repo = tmp_path / "repo"
        repo.mkdir()
        for args in (["init", "-q"], ["config", "user.name", "t"],
                     ["config", "user.email", "t@l"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True)
        (repo / "a.txt").write_text("a\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "root"],
                       check=True, capture_output=True)

        suite = self._suite_file(tmp_path, revs=["HEAD", "does-not-exist"])
        monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
        code = main(["eval", str(suite), "--model", "fake-model", "--split", "all"])
        out = capsys.readouterr().out
        assert "2 revision(s) produced no task" in out, out
        assert code == 2  # nothing left to run, and it says so

    def test_S401_eval_runs_a_suite_end_to_end(
        self, home: Path, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        """Acceptance (6): generate, validate, run, grade, report -- through
        the command a person would actually type."""
        import json
        import subprocess
        import sys as _sys

        from harness.repo import SHADOW_GIT_DIR  # noqa: F401  (patched below)

        monkeypatch.setattr("harness.repo.SHADOW_GIT_DIR", str(tmp_path / "shadow"))

        repo = tmp_path / "repo"
        (repo / "tests").mkdir(parents=True)

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True)

        (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        git("init", "-q")
        git("config", "user.name", "t")
        git("config", "user.email", "t@localhost")
        git("add", "-A")
        git("commit", "-q", "-m", "initial")
        solved = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
        (repo / "calc.py").write_text(solved)
        (repo / "tests" / "test_mul.py").write_text(
            "from calc import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"
        )
        git("add", "-A")
        git("commit", "-q", "-m", "Add a multiply helper")

        write_fake_config(
            home,
            [
                json.dumps({"tool_calls": [{"name": "write_file", "arguments": {
                    "path": "calc.py", "content": solved}}]}),
                json.dumps({"content": "Task complete. Added mul() and the tests pass."}),
            ],
        )
        suite = tmp_path / "suite.json"
        suite.write_text(json.dumps({
            "name": "toy",
            "repo": str(repo),
            "revs": ["HEAD"],
            "test_command": f"{_sys.executable} -m pytest tests -q",
        }))

        code = main([
            "eval", str(suite), "--model", "fake", "--split", "all",
            "--workdir", str(tmp_path / "work"), "--wall-clock", "120",
        ])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "suite toy: 1 task(s)" in out
        assert "validated: 1 usable, 0 dropped" in out
        assert "PASS" in out
        assert "pass rate           : 100.0%" in out
        assert "diff precision      : 100.0%" in out
        assert "regressions         : not measured" in out, out

    def test_S401_validation_uses_the_runner_it_will_grade_with(
        self, home: Path, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Structural, and it has to be.

        Validation on the host while grading happens in a container is not a
        difference any fixture here can observe -- these tests patch Docker
        away, so both paths collapse to `LocalSandbox`. On a real machine it
        was the default, and it scored correct solutions as model failures: a
        suite whose test_command names a host interpreter validates perfectly
        and then fails every trial inside the image. So the assertion is on
        *which runner is passed*, not on what it returns.
        """
        import json
        import subprocess

        from harness.cli import main

        monkeypatch.setattr("harness.repo.SHADOW_GIT_DIR", str(tmp_path / "shadow"))
        repo = tmp_path / "repo"
        (repo / "tests").mkdir(parents=True)

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True)

        (repo / "a.py").write_text("X = 1\n")
        git("init", "-q")
        git("config", "user.name", "t")
        git("config", "user.email", "t@localhost")
        git("add", "-A")
        git("commit", "-q", "-m", "initial")
        (repo / "a.py").write_text("X = 2\n")
        (repo / "tests" / "test_a.py").write_text(
            "from a import X\n\n\ndef test_x():\n    assert X == 2\n"
        )
        git("add", "-A")
        git("commit", "-q", "-m", "Bump X")

        sentinel = object()
        seen = {}

        import harness.eval.grading as grading_module
        import harness.eval.pr_replay as replay_module

        monkeypatch.setattr(
            grading_module, "sandboxed_runner", lambda config: sentinel
        )
        real_validate = replay_module.validate

        async def spy(task, command, workdir, run_command, timeout=300):
            seen["runner"] = run_command
            # Fall back to a real host run so the command still completes.
            return await real_validate(task, command, workdir, host_run, timeout)

        monkeypatch.setattr(replay_module, "validate", spy)

        write_fake_config(home, ['{"content": "Task complete. Nothing to do."}'])
        suite = tmp_path / "suite.json"
        suite.write_text(json.dumps({
            "name": "toy", "repo": str(repo), "revs": ["HEAD"],
            "test_command": "true",
        }))
        main([
            "eval", str(suite), "--model", "fake", "--split", "all",
            "--workdir", str(tmp_path / "work"), "--wall-clock", "60",
        ])
        assert seen.get("runner") is sentinel, (
            "validation did not use the sandboxed runner the trials grade with"
        )

    def test_S401_an_unvalidated_run_says_it_is_not_a_benchmark_result(
        self, home: Path, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        import json
        import subprocess

        monkeypatch.setattr("harness.repo.SHADOW_GIT_DIR", str(tmp_path / "shadow"))
        repo = tmp_path / "repo"
        (repo / "tests").mkdir(parents=True)

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True)

        (repo / "a.py").write_text("X = 1\n")
        git("init", "-q")
        git("config", "user.name", "t")
        git("config", "user.email", "t@localhost")
        git("add", "-A")
        git("commit", "-q", "-m", "initial")
        (repo / "a.py").write_text("X = 2\n")
        (repo / "tests" / "test_a.py").write_text("from a import X\n\n\ndef test_x():\n    assert X == 2\n")
        git("add", "-A")
        git("commit", "-q", "-m", "Bump X")

        write_fake_config(home, ['{"content": "Task complete. Nothing to do."}'])
        suite = tmp_path / "suite.json"
        suite.write_text(json.dumps({
            "name": "toy", "repo": str(repo), "revs": ["HEAD"],
            "test_command": "true",
        }))
        code = main([
            "eval", str(suite), "--model", "fake", "--split", "all",
            "--no-validate", "--workdir", str(tmp_path / "work"),
            "--wall-clock", "60",
        ])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "validation skipped" in out
        assert "not a benchmark result" in out
