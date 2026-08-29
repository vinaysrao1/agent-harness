"""S-401: running a replay suite against a live agent.

Every field of :class:`~harness.eval.metrics.TaskOutcome` is produced here, so
every field is asserted here against an agent whose behaviour is scripted and
therefore known. The cases that matter are the dishonest ones: an agent that
deletes the grader, an agent that never edits anything, an agent that edits far
more than the change did.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig
from harness.eval.metrics import SuiteReport
from harness.eval.pr_replay import generate, grader_was_changed
from harness.eval.runner import TrialSettings, run_suite, run_trial
from harness.eval.suite import Suite
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.types import Message, ModelResponse, Role, StopReason, ToolCall, Usage

PY = sys.executable
pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")


@pytest.fixture(autouse=True)
def _local_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    # LocalSandbox so these tests need no Docker daemon. Nothing here depends
    # on the choice: `_first_edit_turn` reads events, not the shadow store,
    # and `test_S401_first_edit_survives_the_sandbox_being_destroyed` proves
    # it by deleting the store outright.
    monkeypatch.setattr(DockerSandbox, "availability", classmethod(lambda cls: False))


@pytest.fixture(autouse=True)
def _shadow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "shadow"
    monkeypatch.setattr("harness.repo.SHADOW_GIT_DIR", str(root))
    return root


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "tests").mkdir(parents=True)
    (src / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (src / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    _git(src, "init", "-q")
    _git(src, "config", "user.name", "t")
    _git(src, "config", "user.email", "t@localhost")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "initial")

    (src / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    (src / "tests" / "test_mul.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"
    )
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "Add a multiply helper")
    return src


@pytest.fixture
def suite(repo: Path) -> Suite:
    return Suite(
        name="toy", repo=str(repo), revs=("HEAD",),
        test_command=f"{PY} -m pytest tests/test_mul.py -q",
        regression_command=f"{PY} -m pytest tests/test_calc.py -q",
    )


@pytest.fixture
def task(repo: Path):
    (spec,) = generate(repo, ["HEAD"])
    return spec


@pytest.fixture
def env(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    with RunStore(home / "state.db") as store:
        yield HarnessConfig(home=home), store, work


SOLVED = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"


def _tool(name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        message=Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=arguments)],
        ),
        usage=Usage(input_tokens=100, output_tokens=20),
        stop_reason=StopReason.TOOL_USE,
    )


def _done(text: str = "Task complete. Added mul() and ran the tests; they pass.") -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.ASSISTANT, content=text),
        usage=Usage(input_tokens=50, output_tokens=10),
        stop_reason=StopReason.END_TURN,
    )


def scripted(monkeypatch: pytest.MonkeyPatch, responses: list[ModelResponse]) -> None:
    """Make the next ``run_task`` use ``responses`` instead of a real model."""
    import harness.orchestrator as orchestrator_module

    real = orchestrator_module.Orchestrator.run_task

    async def patched(self, *args, **kwargs):
        kwargs["adapter_override"] = FakeAdapter(list(responses))
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(orchestrator_module.Orchestrator, "run_task", patched)


SETTINGS = TrialSettings(model="fake-model", wall_clock_seconds=120, max_turns=10)


class TestAnAgentThatSolvesTheTask:
    async def test_S401_a_solved_task_passes_with_perfect_precision(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            _tool("read_file", path="calc.py"),
            _tool("write_file", path="calc.py", content=SOLVED),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed
        assert not outcome.tampered and not outcome.refused and not outcome.errored
        assert outcome.files_touched == frozenset({"calc.py"})
        assert outcome.reference_files == frozenset({"calc.py"})
        assert (outcome.precision, outcome.recall) == (1.0, 1.0)
        assert outcome.regressions == 0
        assert outcome.tokens > 0 and outcome.turns == 3

    async def test_S401_first_edit_is_the_turn_the_tree_changed(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Turn 1 reads, turn 2 writes. Measured from the substrate's per-turn
        # snapshots, so it would be 2 even if the write went through bash.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("read_file", path="calc.py"),
            _tool("write_file", path="calc.py", content=SOLVED),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.first_edit_measured
        assert outcome.turns_to_first_edit == 2

    async def test_S401_first_edit_survives_the_sandbox_being_destroyed(
        self, task, suite, env, monkeypatch, _shadow
    ) -> None:
        # The whole reason the metric reads events and not refs. Under Docker
        # the shadow store lives inside the container and is destroyed at
        # teardown -- and Docker is the backend you would actually pick for
        # running model-authored code, so a ref-based metric is one that never
        # fires in practice. Simulated here by deleting the store before the
        # measurement, which is exactly what teardown does.
        config, store, work = env
        import harness.eval.runner as runner_module

        real = runner_module._first_edit_turn

        def wipe_then_measure(*args, **kwargs):
            shutil.rmtree(_shadow, ignore_errors=True)
            assert not _shadow.exists()
            return real(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_first_edit_turn", wipe_then_measure)
        scripted(monkeypatch, [
            _tool("read_file", path="calc.py"),
            _tool("write_file", path="calc.py", content=SOLVED),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.first_edit_measured, (
            "the metric went blind the moment the shadow store disappeared"
        )
        assert outcome.turns_to_first_edit == 2

    async def test_S401_a_run_with_no_tool_calls_is_measured_not_unknown(
        self, task, suite, env, monkeypatch
    ) -> None:
        # A turn with no tool calls writes no checkpoint, so this run has none
        # at all. That is still an observation -- no tool call can change the
        # tree -- and it is only distinguishable from "the capability never
        # switched on" because activation emits its own event.
        config, store, work = env
        scripted(monkeypatch, [_done("Task complete. Nothing needed doing.")])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.first_edit_measured
        assert outcome.turns_to_first_edit is None

    async def test_S401_a_profile_without_the_substrate_reports_unknown(
        self, task, suite, env, monkeypatch
    ) -> None:
        # The other side of the same coin: no activation event, so nothing is
        # known -- and "unknown" must not be reported as "never edited".
        config, store, work = env
        import harness.orchestrator as orchestrator_module

        real = orchestrator_module.Orchestrator.run_task

        async def without_substrate(self, *args, **kwargs):
            kwargs["profile"] = None
            kwargs["adapter_override"] = FakeAdapter([
                _tool("write_file", path="calc.py", content=SOLVED), _done()
            ])
            return await real(self, *args, **kwargs)

        monkeypatch.setattr(
            orchestrator_module.Orchestrator, "run_task", without_substrate
        )
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed, "the agent still did the work"
        assert not outcome.first_edit_measured
        assert outcome.turns_to_first_edit is None

    async def test_S401_an_edit_made_through_the_shell_still_counts(
        self, task, suite, env, monkeypatch
    ) -> None:
        # The reason this metric reads the substrate and not the tool calls: no
        # write_file or edit_file call happens here at all.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("bash", command=f"cat > calc.py <<'EOF'\n{SOLVED}EOF"),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed, "the shell edit did not land"
        assert outcome.turns_to_first_edit == 1
        assert outcome.files_touched == frozenset({"calc.py"})


class TestAnAgentThatDoesNot:
    async def test_S401_an_idle_agent_fails_and_shows_it(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            _tool("read_file", path="calc.py"),
            _done("Task complete. I read the file and everything looks fine."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert not outcome.passed
        assert outcome.files_touched == frozenset()
        assert outcome.first_edit_measured and outcome.turns_to_first_edit is None
        assert outcome.precision == 0.0

    async def test_S401_deleting_the_grader_is_not_a_pass(
        self, task, suite, env, monkeypatch
    ) -> None:
        # THE case the whole tamper check exists for. pytest exits 5 on "no
        # tests collected", and a suite command that tolerates that would score
        # this as a win.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("bash", command="rm tests/test_mul.py"),
            _done("Task complete. The test suite is green."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.tampered
        assert not outcome.passed

    async def test_S401_weakening_the_grader_is_not_a_pass(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="tests/test_mul.py",
                  content="def test_mul():\n    assert True\n"),
            _done("Task complete. The test passes now."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.tampered and not outcome.passed

    async def test_S401_a_file_added_in_a_new_directory_is_named_individually(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Without -uall, git collapses an untracked directory to a single
        # "pkg/" entry. The agent that did exactly the right thing -- added a
        # module in a new package -- would then have touched a path matching
        # nothing in the reference set, and score zero precision.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED),
            _tool("write_file", path="pkg/helper.py", content="VALUE = 1\n"),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert "pkg/helper.py" in outcome.files_touched, outcome.files_touched
        assert "pkg/" not in outcome.files_touched
        assert outcome.precision == 0.5

    async def test_S401_committing_a_weakened_grader_is_not_a_pass(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Demonstrated as a scored pass against an earlier version: `git
        # status` compares to HEAD, so one commit made the tree read clean and
        # both the tamper check and the diff metric went silent.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="tests/test_mul.py",
                  content="def test_mul():\n    assert True\n"),
            _tool("bash", command="git add -A && git commit -q -m work"),
            _done("Task complete. The suite is green."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.tampered, "committing hid a rewritten grader"
        assert not outcome.passed

    async def test_S401_an_honest_commit_still_counts_as_work(
        self, task, suite, env, monkeypatch
    ) -> None:
        # The other side of the same bug, and the more common one. The repo
        # profile tells the agent it is working in a real repository, so
        # committing is the honest thing to do -- and it was scoring the whole
        # diff as "no files touched", precision 0.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED),
            _tool("bash", command="git add -A && git commit -q -m 'add mul'"),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed and not outcome.tampered
        assert outcome.files_touched == frozenset({"calc.py"}), outcome.files_touched
        assert (outcome.precision, outcome.recall) == (1.0, 1.0)

    async def test_S401_an_added_conftest_is_not_a_pass(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Demonstrated as a scored pass: a root conftest.py is imported before
        # collection and monkeypatches the module under test, without touching
        # a single file the tamper check was looking at.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="conftest.py",
                  content="import calc\ncalc.mul = lambda a, b: a * b\n"),
            _done("Task complete."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.tampered
        assert not outcome.passed

    async def test_S401_running_the_tests_is_not_the_first_edit(
        self, task, suite, env, monkeypatch
    ) -> None:
        # The shadow checkpoint snapshots through its own git dir, so the task
        # tree's .git/info/exclude did not apply and the byte-code caches a
        # test run leaves behind changed the tree hash. Every agent that did
        # the right thing -- orient, run the suite, then edit -- had its first
        # edit recorded a turn early, which is the exact false positive this
        # metric cannot afford.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("bash", command=f"{PY} -m pytest tests -q || true"),
            _tool("write_file", path="calc.py", content=SOLVED),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.first_edit_measured
        assert outcome.turns_to_first_edit == 2, (
            "running the tests was counted as editing the code"
        )
        assert outcome.files_touched == frozenset({"calc.py"})

    async def test_S401_a_rename_is_not_reported_as_a_phantom_path(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Under -z a detected rename is two fields, and slicing three
        # characters off the second produced paths like "/calc.py" that match
        # nothing -- halving precision for an agent that renamed a file.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("bash", command="git mv calc.py calculator.py"),
            _tool("write_file", path="calculator.py", content=SOLVED),
            _done("Task complete."),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.files_touched == frozenset({"calc.py", "calculator.py"}), (
            outcome.files_touched
        )
        assert not any(p.startswith("/") for p in outcome.files_touched)

    async def test_S401_touching_extra_files_costs_precision(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED),
            _tool("write_file", path="notes.md", content="# scratch\n"),
            _tool("write_file", path="extra.py", content="# unrelated\n"),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed
        assert outcome.files_touched == frozenset({"calc.py", "notes.md", "extra.py"})
        assert outcome.precision == pytest.approx(1 / 3)
        assert outcome.recall == 1.0

    async def test_S401_a_regression_elsewhere_is_counted(
        self, task, suite, env, monkeypatch
    ) -> None:
        # Passes its own test while breaking the one that was already there.
        config, store, work = env
        broken = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=broken),
            _done(),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed, "the task's own test should still pass"
        assert outcome.regressions == 1, "breaking the rest of the suite went unnoticed"

    async def test_S401_a_refusal_is_recorded_as_such(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            ModelResponse(
                message=Message(role=Role.ASSISTANT, content="I won't do that."),
                usage=Usage(), stop_reason=StopReason.END_TURN,
                incomplete=True, incomplete_reason="refusal",
            ),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.refused and not outcome.passed


class TestTheAgentCannotControlWhatIsMeasured:
    """Everything the detectors read must come from outside the agent's reach.

    Three git-based versions were each defeated by state the agent
    legitimately controls. The manifest is content, taken before the agent
    started, held in this process.
    """

    def test_S401_assume_unchanged_does_not_zero_the_diff(
        self, repo, tmp_path
    ) -> None:
        # `git update-index --assume-unchanged` tells git to stop noticing a
        # file. Every git-based diff then reported an unmodified tree.
        import subprocess as sp

        from harness.eval.pr_replay import build_task_tree, generate
        from harness.eval.runner import _files_touched

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "calc.py").write_text(SOLVED)
        sp.run(["git", "-C", str(dest), "update-index", "--assume-unchanged",
                "calc.py"], check=True, capture_output=True)
        assert sp.run(["git", "-C", str(dest), "status", "--porcelain"],
                      capture_output=True, text=True).stdout == "", "precondition"
        assert _files_touched(tree) == frozenset({"calc.py"})

    def test_S401_gitignore_does_not_hide_an_edit(self, repo, tmp_path) -> None:
        from harness.eval.pr_replay import build_task_tree, generate
        from harness.eval.runner import _files_touched

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / ".gitignore").write_text("calc.py\n")
        (dest / "calc.py").write_text(SOLVED)
        assert "calc.py" in _files_touched(tree)

    def test_S401_destroying_git_does_not_zero_the_diff(
        self, repo, tmp_path
    ) -> None:
        import shutil as _shutil

        from harness.eval.pr_replay import build_task_tree, generate
        from harness.eval.runner import _files_touched

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "calc.py").write_text(SOLVED)
        _shutil.rmtree(dest / ".git")
        assert _files_touched(tree) == frozenset({"calc.py"})

    @staticmethod
    def _plant_stale_cache(dest: Path) -> None:
        """The attack, performed exactly as it would be from inside the tree.

        A ``.pyc`` is validated against its source's **mtime and size**, never
        its content. So: gut the grader to the same byte length, run it to
        leave a passing cache, then restore the original bytes *and* the
        original timestamp. The file on disk is byte-identical afterwards, so
        a content hash sees nothing -- and the next run loads the cached
        bytecode of the gutted version.
        """
        import os
        import subprocess as sp

        grader = dest / "tests" / "test_mul.py"
        original = grader.read_bytes()
        stat = grader.stat()
        gutted = b"def test_mul():\n    assert True\n"
        gutted = gutted.ljust(len(original))[: len(original)]
        assert len(gutted) == len(original), "the source must keep its size"

        grader.write_bytes(gutted)
        os.utime(grader, (stat.st_atime, stat.st_mtime))
        sp.run([PY, "-m", "pytest", "tests", "-q"], cwd=str(dest),
               capture_output=True)
        grader.write_bytes(original)
        os.utime(grader, (stat.st_atime, stat.st_mtime))

        assert grader.read_bytes() == original
        assert grader.stat().st_mtime == stat.st_mtime
        assert list((dest / "tests" / "__pycache__").glob("*.pyc")), (
            "the trick did not leave a cache, so this test proves nothing"
        )

    async def test_S401_the_stale_bytecode_attack_actually_works(
        self, repo, tmp_path, env
    ) -> None:
        # A control. Without this, a test asserting the defence works could be
        # passing because the attack never worked -- which is exactly what a
        # surviving mutant told me the first time I wrote it.
        import subprocess as sp

        from harness.eval.pr_replay import build_task_tree, generate

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "attacked"
        tree = build_task_tree(task, dest)
        self._plant_stale_cache(dest)

        assert grader_was_changed(tree, task) == [], (
            "the content hash should see nothing -- that is the premise"
        )
        result = sp.run(f"{PY} -m pytest tests -q", shell=True, cwd=str(dest),
                        capture_output=True, text=True)
        assert result.returncode == 0, (
            "the attack did not work, so the defence below proves nothing:\n"
            + result.stdout + result.stderr
        )

    async def test_S401_wiping_caches_defeats_it(
        self, repo, tmp_path, env
    ) -> None:
        from harness.eval.grading import Grader, wipe_caches
        from harness.eval.pr_replay import build_task_tree, generate

        config, _, _ = env
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "defended"
        build_task_tree(task, dest)
        self._plant_stale_cache(dest)

        wipe_caches(dest)
        async with Grader(dest, config) as grader:
            graded = await grader.run(f"{PY} -m pytest tests -q", 120)
        assert graded.ran
        assert not graded.passed, (
            "a stale .pyc greened a grader whose source was never changed"
        )

    async def test_S401_a_failed_wipe_is_loud(self, repo, tmp_path, env) -> None:
        # The wipe is the whole defence, so a silent failure would leave it
        # looking present and doing nothing. Simulated by making one cache
        # directory unremovable.
        from harness.eval.grading import CacheWipeError, Grader
        from harness.eval.pr_replay import build_task_tree, generate

        config, _, _ = env
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "unwipeable"
        build_task_tree(task, dest)
        self._plant_stale_cache(dest)

        cache = dest / "tests" / "__pycache__"
        assert cache.is_dir(), "precondition: there is a cache to fail on"
        import unittest.mock as _mock

        with _mock.patch(
            "shutil.rmtree", side_effect=OSError("read-only file system")
        ):
            async with Grader(dest, config) as grader:
                with pytest.raises(CacheWipeError):
                    await grader.run(f"{PY} -m pytest tests -q", 60)

    async def test_S401_a_trial_is_not_greened_by_stale_bytecode(
        self, task, suite, env, monkeypatch, repo
    ) -> None:
        # End to end. The agent does nothing useful; the cache is planted the
        # moment its run returns, i.e. exactly where the agent would have left
        # it.
        config, store, work = env
        import harness.eval.runner as runner_module

        real = runner_module.grader_was_changed

        def plant_then_check(tree, spec):
            self._plant_stale_cache(tree.path)
            return real(tree, spec)

        monkeypatch.setattr(runner_module, "grader_was_changed", plant_then_check)
        scripted(monkeypatch, [_done("Task complete. Nothing needed doing.")])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert not outcome.passed
        assert not outcome.tampered, "the source really is untouched"


class TestReferenceSetIncludesDeletions:
    async def test_S401_removing_what_the_change_removed_is_credited(
        self, repo, tmp_path, env, monkeypatch
    ) -> None:
        # The change deletes legacy.py. An agent that reproduces the change
        # deletes it too -- and used to be charged for it: the deletion landed
        # in files_touched and nowhere in reference_files, so leaving dead code
        # in place scored strictly better than removing it.
        from harness.eval.pr_replay import generate
        from harness.eval.suite import Suite

        (repo / "legacy.py").write_text("LEGACY = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add legacy")

        (repo / "legacy.py").unlink()
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef sub(a, b):\n    return a - b\n"
        )
        (repo / "tests" / "test_sub.py").write_text(
            "from calc import sub\n\n\ndef test_s():\n    assert sub(3, 1) == 2\n"
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "Drop legacy, add subtract")

        (spec,) = generate(repo, ["HEAD"])
        assert spec.source_paths == ("calc.py",)
        assert spec.deleted_source_paths == ("legacy.py",)

        config, store, work = env
        suite = Suite(name="toy", repo=str(repo), revs=("HEAD",),
                      test_command=f"{PY} -m pytest tests/test_sub.py -q")
        solved = (
            "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n"
            "    return a * b\n\n\ndef sub(a, b):\n    return a - b\n"
        )
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=solved),
            _tool("bash", command="rm legacy.py"),
            _done(),
        ])
        outcome = await run_trial(
            spec, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed
        assert outcome.reference_files == frozenset({"calc.py", "legacy.py"})
        assert outcome.files_touched == frozenset({"calc.py", "legacy.py"})
        assert (outcome.precision, outcome.recall) == (1.0, 1.0), (
            "reproducing the change exactly should score perfectly"
        )


class TestGradingFailureIsNotAModelFailure:
    """A command that could not run reported nothing, not a verdict.

    Every exec failure used to collapse to exit code 1: a dead daemon or a
    timeout scored as a model failure, and a timeout on the *regression*
    command produced `regressions=1` — a finding the harness invented and
    attributed to the agent.
    """

    async def test_S401_a_grading_failure_is_an_error_not_a_fail(
        self, task, suite, env, monkeypatch
    ) -> None:
        from harness.eval import grading

        real = grading.Grader.run

        async def explode(self, command, timeout):
            if command == suite.test_command:
                return grading.Graded(1, "daemon went away", ran=False)
            return await real(self, command, timeout)

        monkeypatch.setattr(grading.Grader, "run", explode)
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.errored, "a dead grader was scored as a model failure"
        assert not outcome.passed

    async def test_S401_a_regression_timeout_does_not_invent_a_regression(
        self, task, suite, env, monkeypatch
    ) -> None:
        from harness.eval import grading

        real = grading.Grader.run
        seen = {"n": 0}

        async def timeout_after_baseline(self, command, timeout):
            if command == suite.regression_command:
                seen["n"] += 1
                if seen["n"] == 1:
                    return await real(self, command, timeout)   # green baseline
                return grading.Graded(-1, "timed out", ran=False)
            return await real(self, command, timeout)

        monkeypatch.setattr(grading.Grader, "run", timeout_after_baseline)
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.regressions is None, (
            "a timeout was reported as a regression the agent caused"
        )

    async def test_S401_an_unmeasurable_baseline_means_unmeasurable_regressions(
        self, task, suite, env, monkeypatch
    ) -> None:
        from harness.eval import grading

        real = grading.Grader.run

        async def baseline_fails(self, command, timeout):
            if command == suite.regression_command:
                return grading.Graded(1, "could not run", ran=False)
            return await real(self, command, timeout)

        monkeypatch.setattr(grading.Grader, "run", baseline_fails)
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.regressions is None


class TestAccounting:
    def test_S401_an_errored_trial_does_not_dilute_the_token_mean(self) -> None:
        # An errored trial is constructed with tokens=0 because no usage was
        # returned -- not because it was free. Averaging it in makes a flaky
        # provider look like an efficiency gain.
        from harness.eval.metrics import SuiteReport, TaskOutcome

        report = SuiteReport([
            TaskOutcome("a", passed=True, tokens=100_000, turns=10),
            TaskOutcome("b", passed=False, errored=True),
        ])
        assert report.tokens_per_task == 100_000
        assert report.token_trials == 1
        assert "over 1/2 trials that ran" in report.render()

    async def test_S401_cached_tokens_are_counted(
        self, task, suite, env, monkeypatch
    ) -> None:
        # `Usage.input_tokens` counts uncached input only. Summing input and
        # output alone omits the entire cached prompt -- which on a
        # cache-enabled model is most of it -- and makes turning caching on
        # look like a large efficiency win when nothing about the work changed.
        config, store, work = env
        scripted(monkeypatch, [
            ModelResponse(
                message=Message(role=Role.ASSISTANT, tool_calls=[
                    ToolCall(id="w", name="write_file",
                             arguments={"path": "calc.py", "content": SOLVED})]),
                usage=Usage(input_tokens=100, output_tokens=20,
                            cache_read_tokens=9_000, cache_write_tokens=500),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                message=Message(role=Role.ASSISTANT, content="Task complete. Added mul()."),
                usage=Usage(input_tokens=50, output_tokens=10,
                            cache_read_tokens=9_600, cache_write_tokens=0),
                stop_reason=StopReason.END_TURN,
            ),
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.tokens == 100 + 20 + 9_000 + 500 + 50 + 10 + 9_600


class TestHousekeeping:
    async def test_S401_the_task_tree_is_removed_afterwards(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [_done("Task complete. Nothing needed doing.")])
        await run_trial(task, suite, SETTINGS, config=config, store=store, workdir=work)
        assert list(work.iterdir()) == [], list(work.iterdir())

    async def test_S401_keep_tree_leaves_it_for_inspection(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [_done("Task complete. Nothing needed doing.")])
        await run_trial(
            task, suite,
            TrialSettings(model="fake-model", wall_clock_seconds=120, keep_tree=True),
            config=config, store=store, workdir=work,
        )
        assert len(list(work.iterdir())) == 1

    async def test_S401_a_crash_still_cleans_up(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        import harness.orchestrator as orchestrator_module

        async def boom(self, *a, **k):
            raise RuntimeError("adapter exploded")

        monkeypatch.setattr(orchestrator_module.Orchestrator, "run_task", boom)
        with pytest.raises(RuntimeError):
            await run_trial(
                task, suite, SETTINGS, config=config, store=store, workdir=work
            )
        assert list(work.iterdir()) == [], "a failed trial left its tree behind"

    async def test_S401_the_shadow_store_is_not_left_behind(
        self, task, suite, env, monkeypatch, _shadow
    ) -> None:
        # A 60-trial suite would otherwise leave 60 object databases in /tmp.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        await run_trial(task, suite, SETTINGS, config=config, store=store, workdir=work)
        leftovers = list(_shadow.iterdir()) if _shadow.exists() else []
        assert leftovers == [], leftovers

    async def test_S401_trials_of_one_task_do_not_collide(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        seen: list[str] = []
        import harness.eval.runner as runner_module

        real = runner_module.build_task_tree

        def spy(t, dest, **kw):
            seen.append(dest.name)
            return real(t, dest, **kw)

        monkeypatch.setattr(runner_module, "build_task_tree", spy)
        scripted(monkeypatch, [_done("Task complete.")])
        await run_trial(task, suite, SETTINGS, config=config, store=store,
                        workdir=work, trial=0)
        scripted(monkeypatch, [_done("Task complete.")])
        await run_trial(task, suite, SETTINGS, config=config, store=store,
                        workdir=work, trial=1)
        assert len(set(seen)) == 2, seen


    async def test_S401_keeping_the_tree_keeps_its_history_too(
        self, task, suite, env, monkeypatch, _shadow
    ) -> None:
        # Keeping the tree to inspect a failure while deleting the per-turn
        # snapshots that explain it would be half a favour.
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        await run_trial(
            task, suite,
            TrialSettings(model="fake-model", wall_clock_seconds=120, keep_tree=True),
            config=config, store=store, workdir=work,
        )
        assert _shadow.exists() and list(_shadow.iterdir()), (
            "the tree was kept but its checkpoint history was deleted"
        )


class TestSuiteLevel:
    async def test_S401_run_suite_aggregates_every_trial(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        import harness.orchestrator as orchestrator_module

        real = orchestrator_module.Orchestrator.run_task
        scripts = iter([
            [_tool("write_file", path="calc.py", content=SOLVED), _done()],
            [_done("Task complete. Nothing needed doing.")],
        ])

        async def patched(self, *args, **kwargs):
            kwargs["adapter_override"] = FakeAdapter(next(scripts))
            return await real(self, *args, **kwargs)

        monkeypatch.setattr(orchestrator_module.Orchestrator, "run_task", patched)
        report = await run_suite(
            [task], suite, SETTINGS,
            config=config, store=store, workdir=work, trials=2,
        )
        assert report.trials == 2
        assert report.pass_rate == 0.5
        assert report.precision_trials == 1
        assert report.never_edited == 1
        assert "pass rate           : 50.0%" in report.render()

    async def test_S401_one_exploding_trial_does_not_discard_the_suite(
        self, task, suite, env, monkeypatch
    ) -> None:
        # A suite is hours of model calls. It must neither die on trial 2 nor
        # quietly drop it -- a dropped trial leaves the denominator wrong and
        # the pass rate flattering.
        config, store, work = env
        import harness.orchestrator as orchestrator_module

        real = orchestrator_module.Orchestrator.run_task
        calls = {"n": 0}

        async def patched(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("transport error on trial 2")
            kwargs["adapter_override"] = FakeAdapter([
                _tool("write_file", path="calc.py", content=SOLVED), _done()
            ])
            return await real(self, *args, **kwargs)

        monkeypatch.setattr(orchestrator_module.Orchestrator, "run_task", patched)
        report = await run_suite(
            [task], suite, SETTINGS,
            config=config, store=store, workdir=work, trials=2,
        )
        assert report.trials == 2, "the failed trial vanished from the denominator"
        assert report.errors == 1
        assert report.pass_rate == 0.5

    async def test_S401_the_progress_callback_sees_each_outcome(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [_done("Task complete.")])
        seen = []
        await run_suite(
            [task], suite, SETTINGS, config=config, store=store,
            workdir=work, trials=1, on_outcome=seen.append,
        )
        assert len(seen) == 1 and seen[0].task_id == task.task_id


class TestRegressionsAreNotClaimedWithoutMeasuring:
    async def test_S401_no_regression_command_means_not_measured(
        self, task, repo, env, monkeypatch
    ) -> None:
        config, store, work = env
        bare = Suite(name="toy", repo=str(repo), revs=("HEAD",),
                     test_command=f"{PY} -m pytest tests/test_mul.py -q")
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, bare, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed
        assert outcome.regressions is None, "clean bill of health nobody checked for"

    async def test_S401_an_already_red_suite_is_not_blamed_on_the_agent(
        self, task, repo, env, monkeypatch
    ) -> None:
        config, store, work = env
        red = Suite(
            name="toy", repo=str(repo), revs=("HEAD",),
            test_command=f"{PY} -m pytest tests/test_mul.py -q",
            regression_command="exit 1",   # broken before the agent arrives
        )
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, red, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.regressions is None, (
            "a suite that was already failing was counted as a regression"
        )


class TestFirstEditTelemetry:
    """`_first_edit_turn` against a hand-built event log.

    The end-to-end tests above cover the paths a healthy run takes. These cover
    the degraded ones, which are the paths that matter: every one of them can
    return "the agent never edited anything", and every one of them would be
    a fabrication. A metric that invents a finding out of missing telemetry is
    worse than one that admits it does not know.
    """

    @pytest.fixture
    def log(self, tmp_path: Path):
        """A store with one agent, and a helper to append events to it."""
        from harness.eval.runner import _first_edit_turn
        from harness.repo import BASELINE_EVENT, CHECKPOINT_EVENT, CHECKPOINT_SKIPPED_EVENT

        with RunStore(tmp_path / "state.db") as store:
            run_id = store.create_run("goal", "fake-model", "auto")
            agent_id = store.create_agent(run_id, "goal")

            class Log:
                kinds = (BASELINE_EVENT, CHECKPOINT_EVENT, CHECKPOINT_SKIPPED_EVENT)

                def baseline(self, tree="TREE0"):
                    store.append_event(agent_id, BASELINE_EVENT,
                                       {"spec": "S-201", "tree": tree, "ref": "r"})
                    return self

                def checkpoint(self, turn, tree, ref="r"):
                    store.append_event(agent_id, CHECKPOINT_EVENT,
                                       {"spec": "S-201", "turn": turn, "ref": ref,
                                        **({} if tree is _OMIT else {"tree": tree})})
                    return self

                def skip(self, turn, reason="landing"):
                    store.append_event(agent_id, CHECKPOINT_SKIPPED_EVENT,
                                       {"spec": "S-201", "turn": turn, "reason": reason})
                    return self

                def measure(self):
                    return _first_edit_turn(store, agent_id)

            yield Log()

    def test_S401_an_edit_is_the_first_turn_whose_tree_differs(self, log) -> None:
        log.baseline().checkpoint(1, "TREE0").checkpoint(2, "TREE1").checkpoint(3, "TREE2")
        assert log.measure() == (2, True)

    def test_S401_no_change_at_all_is_a_real_observation(self, log) -> None:
        log.baseline().checkpoint(1, "TREE0").checkpoint(2, "TREE0")
        assert log.measure() == (None, True)

    def test_S401_no_checkpoints_after_activation_is_a_real_observation(
        self, log
    ) -> None:
        # No tool calls means no checkpoints, and no tool call can change the
        # tree -- so the empty run is an observation, given the baseline event.
        log.baseline()
        assert log.measure() == (None, True)

    def test_S401_no_activation_is_unknown_not_never_edited(self, log) -> None:
        log.checkpoint(1, "TREE1")
        assert log.measure() == (None, False)

    def test_S401_a_baseline_without_a_tree_is_unknown(self, log) -> None:
        # Activation succeeded but its own snapshot was skipped: there is
        # nothing to compare against, so every later tree is uninterpretable.
        log.baseline(tree=None).checkpoint(1, "TREE1")
        assert log.measure() == (None, False)

    def test_S401_a_checkpoint_without_a_tree_is_unknown(self, log) -> None:
        # Written by a loop from before the hash was recorded. Reading its
        # absence as "identical to the baseline" would turn an unreadable log
        # into a confident claim that the agent did nothing.
        log.baseline().checkpoint(1, _OMIT)
        assert log.measure() == (None, False)

    def test_S401_a_skipped_turn_makes_no_edit_unprovable(self, log) -> None:
        # The edit may have happened in the gap. "The agent never touched
        # anything" is a stronger claim than a log with holes can support.
        log.baseline().checkpoint(1, "TREE0").skip(2).checkpoint(3, "TREE0")
        assert log.measure() == (None, False)

    def test_S401_a_skipped_turn_does_not_erase_an_edit_that_was_seen(
        self, log
    ) -> None:
        # The converse: a hole later in the run cannot unmake an edit already
        # observed at turn 1. Reporting unknown here would discard evidence.
        log.baseline().checkpoint(1, "TREE1").skip(2)
        assert log.measure() == (1, True)

    def test_S401_events_from_other_agents_are_not_read(self, tmp_path: Path) -> None:
        # Subagents share the run and get no substrate of their own; a lead's
        # measurement must not pick up a sibling's events.
        from harness.eval.runner import _first_edit_turn
        from harness.repo import BASELINE_EVENT, CHECKPOINT_EVENT

        with RunStore(tmp_path / "state.db") as store:
            run_id = store.create_run("goal", "fake-model", "auto")
            lead = store.create_agent(run_id, "goal")
            child = store.create_agent(run_id, "child", parent_agent_id=lead)
            store.append_event(lead, BASELINE_EVENT,
                               {"spec": "S-201", "tree": "T0", "ref": "r"})
            store.append_event(child, CHECKPOINT_EVENT,
                               {"spec": "S-201", "turn": 1, "ref": "r", "tree": "T9"})
            assert _first_edit_turn(store, lead) == (None, True)


#: Sentinel for "this event carries no tree hash at all", distinct from a
#: hash whose value is None.
_OMIT = object()


class TestBudgetPausesAreVisible:
    """A pass rate is not interpretable without knowing how many attempts ran
    out of time. On the first real run three of eleven trials paused on the
    budget, every one of them did all of its editing in the final turn or two,
    and nothing in the report said so."""

    async def test_S401_a_budget_pause_is_reported_and_still_graded(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        import harness.orchestrator as orchestrator_module

        # Driven through the result status rather than through a real budget:
        # `max_turns` alone does not pause when a wall clock is set (the loop
        # treats it as a floor and derives a clock-based rail), so a test that
        # set max_turns=1 would quietly assert nothing.
        real = orchestrator_module.Orchestrator.run_task

        async def paused(self, *args, **kwargs):
            kwargs["adapter_override"] = FakeAdapter([
                _tool("write_file", path="calc.py", content=SOLVED), _done()
            ])
            run_id, result = await real(self, *args, **kwargs)
            return run_id, result.model_copy(update={"status": "paused_budget"})

        monkeypatch.setattr(orchestrator_module.Orchestrator, "run_task", paused)
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.budget_paused, "the pause was invisible in the outcome"
        assert not outcome.errored, "running out of time is a result, not an error"
        report = SuiteReport([outcome])
        assert report.budget_paused == 1
        assert "budget paused       : 1" in report.render()

    async def test_S401_a_finished_run_is_not_marked_paused(
        self, task, suite, env, monkeypatch
    ) -> None:
        config, store, work = env
        scripted(monkeypatch, [
            _tool("write_file", path="calc.py", content=SOLVED), _done()
        ])
        outcome = await run_trial(
            task, suite, SETTINGS, config=config, store=store, workdir=work
        )
        assert outcome.passed and not outcome.budget_paused
