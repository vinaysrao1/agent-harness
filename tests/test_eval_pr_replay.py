"""S-401: PR-replay eval — task generation, validation, metrics, split.

The load-bearing claim is that a generated task is a *benchmark*, not a hope.
That rests entirely on validation: a task whose tests already pass at the base
is trivial, one whose tests fail at the head is impossible, and both are
silent poison. Most of these tests are about that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.eval.metrics import SuiteReport, TaskOutcome, diff_precision
from harness.eval.pr_replay import (
    TaskSpec,
    build_worktree,
    classify_paths,
    generate,
    remove_worktree,
    validate,
)
from harness.eval.suite import HELDOUT_FRACTION, Suite, is_heldout

REPO = Path(__file__).resolve().parent.parent


def _task(**kw) -> TaskSpec:
    base = dict(
        repo=str(REPO), head_sha="h" * 40, base_sha="b" * 40,
        title="t", body="", source_paths=("src/a.py",), test_paths=("tests/test_a.py",),
    )
    base.update(kw)
    return TaskSpec(**base)


class TestPathClassification:
    @pytest.mark.parametrize(
        "path",
        ["tests/test_a.py", "test_a.py", "a_test.py", "conftest.py",
         "src/tests/test_b.py", "app/foo.spec.ts", "app/foo.test.tsx"],
    )
    def test_S401_test_paths_are_recognised(self, path: str) -> None:
        source, tests, _ = classify_paths([path])
        assert tests == [path] and not source, (
            f"{path} classified as source; the grader would be deleted"
        )

    @pytest.mark.parametrize(
        "path", ["harness/loop.py", "src/main.rs", "app/latest.py", "contest.py"]
    )
    def test_S401_source_paths_are_recognised(self, path: str) -> None:
        source, tests, _ = classify_paths([path])
        assert source == [path] and not tests, (
            f"{path} classified as a test; the agent would be handed the answer"
        )

    def test_S401_docs_are_ignored_not_restored(self) -> None:
        # Restoring a spec file would leak the answer; counting it in diff
        # precision would reward touching files the task never needed.
        _, _, ignored = classify_paths(["specs/S-001.md", "README.md", "uv.lock"])
        assert len(ignored) == 3

    def test_S401_a_markdown_file_under_tests_is_documentation(self) -> None:
        # Ordering matters: ignore is checked before the test patterns.
        _, tests, ignored = classify_paths(["tests/README.md"])
        assert not tests and ignored == ["tests/README.md"]


class TestGenerationIsReproducible:
    def test_S401_same_inputs_give_the_same_suite(self) -> None:
        a = generate(REPO, ["b4fc55f"])
        b = generate(REPO, ["b4fc55f"])
        assert a == b, "generation is not deterministic"

    def test_S401_generates_from_a_real_merge(self) -> None:
        (task,) = generate(REPO, ["b4fc55f"])
        assert task.well_formed
        assert "harness/adapters/openai_compat.py" in task.source_paths
        assert "tests/test_adapter_openai.py" in task.test_paths

    def test_S401_base_is_the_first_parent(self) -> None:
        (task,) = generate(REPO, ["b4fc55f"])
        expected = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "b4fc55f^1"],
            capture_output=True, text=True,
        ).stdout.strip()
        assert task.base_sha == expected

    def test_S401_unknown_rev_is_skipped_not_fabricated(self) -> None:
        assert generate(REPO, ["definitely-not-a-rev"]) == []

    def test_S401_ill_formed_tasks_are_marked(self) -> None:
        assert not _task(test_paths=()).well_formed, "no grader"
        assert not _task(source_paths=()).well_formed, "nothing to implement"


class TestThePromptDoesNotLeakTheAnswer:
    def test_S401_prompt_names_the_tests_but_not_the_diff(self) -> None:
        prompt = _task(title="Fix the parser", body="It mishandles nesting.").prompt()
        assert "Fix the parser" in prompt
        assert "tests/test_a.py" in prompt
        # The reference implementation must never appear.
        assert "src/a.py" not in prompt, "the prompt names the file to change"

    def test_S401_prompt_forbids_editing_the_grader(self) -> None:
        assert "without modifying them" in _task().prompt()


class TestWorktreeConstruction:
    def test_S401_starting_state_has_tests_but_not_the_implementation(
        self, tmp_path: Path
    ) -> None:
        (task,) = generate(REPO, ["5f54410"])
        dest = tmp_path / "wt"
        assert build_worktree(task, dest)
        try:
            # The change's test file is present...
            assert (dest / "tests/test_specs.py").exists()
            # ...and its implementation is not.
            assert not (dest / "harness/specs.py").exists()
        finally:
            remove_worktree(task, dest)

    def test_S401_generation_never_mutates_the_source_repo(
        self, tmp_path: Path
    ) -> None:
        before = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        (task,) = generate(REPO, ["5f54410"])
        dest = tmp_path / "wt2"
        build_worktree(task, dest)
        remove_worktree(task, dest)
        after = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        assert before == after, "the generator moved the source repository's HEAD"


class TestValidationIsWhatMakesThisABenchmark:
    def test_S401_a_real_task_validates(self, tmp_path: Path) -> None:
        (task,) = generate(REPO, ["b4fc55f"])
        cmd = (
            f"{REPO}/.venv/bin/python -m pytest tests/test_adapter_openai.py "
            "-q --no-header -p no:cacheprovider"
        )
        v = validate(task, cmd, tmp_path, timeout=300)
        assert v.usable, f"expected a usable task, got: {v.reason}"
        assert v.fails_at_base and v.passes_at_head

    def test_S401_a_trivial_task_is_rejected(self, tmp_path: Path) -> None:
        # A command that always succeeds passes at base, so the task is
        # already solved and measures nothing.
        (task,) = generate(REPO, ["b4fc55f"])
        v = validate(task, "true", tmp_path, timeout=60)
        assert not v.usable
        assert "trivial" in (v.reason or "")

    def test_S401_an_unsolvable_task_is_rejected(self, tmp_path: Path) -> None:
        # A command that always fails also fails at head, so no agent could
        # ever pass it.
        (task,) = generate(REPO, ["b4fc55f"])
        v = validate(task, "exit 1", tmp_path, timeout=60)
        assert not v.usable
        assert "not solvable" in (v.reason or "")

    def test_S401_a_broken_environment_is_not_mistaken_for_a_failing_test(
        self, tmp_path: Path
    ) -> None:
        # The bug this caught for real: a worktree holds only *tracked* files,
        # so `.venv/bin/python` does not exist there. That fails exactly like a
        # legitimately failing test, and without this distinction the suite
        # fills with tasks no agent could pass.
        (task,) = generate(REPO, ["b4fc55f"])
        v = validate(task, ".venv/bin/python -m pytest -q", tmp_path, timeout=60)
        assert v.environment_broken
        assert not v.usable
        assert "bare worktree" in (v.reason or "")

    def test_S401_ill_formed_task_is_rejected_without_running_anything(
        self, tmp_path: Path
    ) -> None:
        v = validate(_task(test_paths=()), "true", tmp_path)
        assert not v.usable and v.reason == "no tests"


class TestMetrics:
    def test_S401_precision_and_recall_fail_in_opposite_directions(self) -> None:
        # Touched a superset: precise? no. Complete? yes.
        p, r = diff_precision({"a", "b", "c"}, {"a", "b"})
        assert p < 1.0 and r == 1.0
        # Touched a subset: precise? yes. Complete? no.
        p, r = diff_precision({"a"}, {"a", "b"})
        assert p == 1.0 and r < 1.0

    def test_S401_empty_sets_do_not_score_as_perfect(self) -> None:
        assert diff_precision(set(), {"a"}) == (0.0, 0.0)
        assert diff_precision({"a"}, set()) == (0.0, 0.0)

    def test_S401_never_edited_is_not_averaged_as_zero(self) -> None:
        # Averaging "never edited" in as 0 would make a paralysed agent look
        # like the most decisive one in the suite.
        report = SuiteReport([
            TaskOutcome("a", False, turns_to_first_edit=None),
            TaskOutcome("b", True, turns_to_first_edit=4),
        ])
        assert report.mean_turns_to_first_edit == 4.0
        assert report.never_edited == 1

    def test_S401_regressions_and_refusals_are_counted(self) -> None:
        report = SuiteReport([
            TaskOutcome("a", True, regressions=3),
            TaskOutcome("b", False, refused=True),
            TaskOutcome("c", False, errored=True),
        ])
        assert report.regressions == 3
        assert report.refusals == 1 and report.errors == 1
        assert report.pass_rate == pytest.approx(1 / 3)

    def test_S401_empty_report_does_not_divide_by_zero(self) -> None:
        assert SuiteReport().render() == "no trials"


class TestHeldOutSplit:
    def test_S401_split_is_about_the_target_fraction(self) -> None:
        ids = [f"{i:012x}" for i in range(600)]
        held = [t for t in ids if is_heldout(t)]
        assert 0.25 < len(held) / len(ids) < 0.42

    def test_S401_membership_is_stable_as_the_suite_grows(self) -> None:
        # An index-based split would reshuffle every task when the suite
        # grows, silently moving tasks you had already tuned against into the
        # held-out half and destroying the guarantee it exists to give.
        small = {t for t in (f"{i:012x}" for i in range(50)) if is_heldout(t)}
        large = {t for t in (f"{i:012x}" for i in range(500)) if is_heldout(t)}
        assert small <= large

    def test_S401_split_is_deterministic(self) -> None:
        assert is_heldout("abc123") == is_heldout("abc123")

    def test_S401_dev_and_heldout_partition_the_suite(self) -> None:
        suite = Suite(name="s", repo=".", revs=("a",), test_command="true")
        ids = [f"{i:012x}" for i in range(200)]
        dev, held = suite.split(ids)
        assert set(dev) | set(held) == set(ids)
        assert not (set(dev) & set(held))

    def test_S401_suite_round_trips(self, tmp_path: Path) -> None:
        suite = Suite(name="s", repo="/r", revs=("a", "b"), test_command="pytest -q")
        path = tmp_path / "suite.json"
        suite.save(path)
        assert Suite.load(path) == suite
