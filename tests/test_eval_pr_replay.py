"""S-401: coding tasks replayed from merged changes.

The three properties that make this an eval rather than a script, each of
which was got wrong first:

1. **The agent cannot read the answer.** Task trees are built with
   ``git archive`` into a fresh single-commit repository. The first version
   used ``git worktree add``, which shares the source object database -- in a
   task generated from this repository, ``git diff HEAD..<head>`` inside the
   agent's own working directory returned 22,565 characters of the reference
   solution.
2. **A generated task is worthless until validated.** Passing at the base is
   trivial; failing at the head is impossible. Both silently poison the metric.
3. **Validation must not discard valid tasks.** The environmental check runs
   against the head state only. Against the base it deleted every task whose
   base failure mentioned a missing file -- exactly what a change that *adds*
   a file looks like.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harness.eval.metrics import SuiteReport, TamperedPassError, TaskOutcome, diff_precision
from harness.eval.pr_replay import (
    BUILD_BYPRODUCTS,
    GitError,
    TaskSpec,
    build_task_tree,
    classify_paths,
    generate,
    grader_was_changed,
    validate,
)
from harness.eval.suite import (
    HELDOUT_FRACTION,
    Suite,
    SuiteError,
    heldout_score,
    is_heldout,
)

PY = sys.executable


async def host_run(command: str, cwd: Path, timeout: float):
    """Run a graded command on the host, for tests that do not need a sandbox.

    `validate` takes its runner as a required argument precisely so that where
    commands run is always a stated choice: validating on the host while
    grading in a container made a correct solution score as a model failure.
    These tests state it here.
    """
    from harness.eval.grading import Graded

    proc = subprocess.run(
        command, shell=True, cwd=str(cwd), capture_output=True,
        text=True, timeout=timeout,
    )
    return Graded(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def _hash_changed(tree, path: str) -> bool:
    """Whether ``path``'s content differs from what the tree recorded."""
    import hashlib

    current = hashlib.sha256((tree.path / path).read_bytes()).hexdigest()
    return current != tree.grader_hashes[path]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with two commits: a helper, then a helper plus a test."""
    src = tmp_path / "src"
    (src / "tests").mkdir(parents=True)
    (src / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (src / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (src / "README.md").write_text("# calc\n")
    git(src, "init", "-q")
    git(src, "config", "user.name", "t")
    git(src, "config", "user.email", "t@localhost")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "initial")

    (src / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    (src / "tests" / "test_mul.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"
    )
    (src / "README.md").write_text("# calc\n\nNow with mul.\n")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "Add a multiply helper")
    return src


TEST_COMMAND = f"{PY} -m pytest tests -q"


# -- classification -----------------------------------------------------------


class TestPathClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "tests/test_x.py",
            "test/test_x.py",
            "src/__tests__/x.js",          # Jest
            "spec/models/user_spec.rb",    # RSpec
            "test-utils/render.tsx",
            "pkg/thing_test.go",
            "Tests/AppTests/LoginTests.swift",
            "src/Foo.Tests.cs",
            "conftest.py",
            "src/a.spec.ts",
            "src/a.test.tsx",
        ],
    )
    def test_S401_test_paths_are_recognised(self, path: str) -> None:
        source, tests, _ = classify_paths([path])
        assert tests == [path] and source == []

    @pytest.mark.parametrize(
        "path",
        [
            "harness/loop.py",
            "src/index.ts",
            "src/protest/banner.py",   # "test" as a substring of a word
            "latest/thing.go",
        ],
    )
    def test_S401_source_paths_are_recognised(self, path: str) -> None:
        source, tests, _ = classify_paths([path])
        assert source == [path], f"{path} was not treated as source"

    def test_S401_a_production_testing_package_is_not_a_grader(self) -> None:
        # `testing/` is a production package in a large fraction of Go and
        # Python projects (k8s.io/client-go/testing, django/test/client.py).
        # Misreading one as a test is not a cosmetic error: the file is
        # restored at HEAD, so part of the reference answer is handed to the
        # agent, and it is removed from the reference set so recall is scored
        # against the wrong denominator.
        source, tests, _ = classify_paths(
            ["testing/helpers.go", "internal/testing/fake.go"]
        )
        assert tests == []
        assert source == ["internal/testing/fake.go", "testing/helpers.go"]

    def test_S401_a_file_named_like_a_test_is_treated_as_one(self) -> None:
        # `scripts/test_runner.py` reads like a helper, but pytest's default
        # collection is `test_*.py` -- it would be collected and run. Treating
        # it as source would leave a file the grader executes unrestored, so
        # the convention wins over the intent its name suggests.
        _, tests, _ = classify_paths(["scripts/test_runner.py"])
        assert tests == ["scripts/test_runner.py"]

    def test_S401_rspec_directories_are_not_swallowed_by_the_ignore_list(self) -> None:
        # An earlier version ignored `specs?/` wholesale, which made the RSpec
        # pattern unreachable and produced an empty test set -- and therefore
        # an ill-formed, silently dropped task -- for every Ruby repository.
        _, tests, ignored = classify_paths(["spec/user_spec.rb"])
        assert tests == ["spec/user_spec.rb"] and ignored == []

    def test_S401_the_ignore_list_contains_no_directory_patterns(self) -> None:
        # Structural, not by example: the ignore list is consulted before the
        # test patterns, so any pattern here that can match a directory
        # segment shadows a whole test convention -- which is exactly how
        # `specs?/` made the RSpec pattern dead code. Asserted against the
        # compiled patterns so a new one cannot slip in unnoticed.
        from harness.eval.pr_replay import IGNORED_PATH_PATTERNS

        for pattern in IGNORED_PATH_PATTERNS:
            assert not pattern.pattern.endswith("/"), pattern.pattern
            for directory in ("spec", "specs", "test", "tests", "docs", "src"):
                assert not pattern.search(f"{directory}/thing.py"), (
                    f"{pattern.pattern!r} swallows everything under {directory}/"
                )

    def test_S401_lockfiles_are_not_source(self) -> None:
        # Counted as source, a lockfile inflates the reference set and drags
        # recall down for every agent that (correctly) never touches it.
        source, _, ignored = classify_paths(["package-lock.json", "poetry.lock"])
        assert source == [] and len(ignored) == 2

    def test_S401_the_licence_pattern_does_not_eat_source_files(self) -> None:
        # Unanchored, `LICENSE` matches the start of `licenses.py` and the
        # module silently stops being source -- it would be restored to
        # neither side and dropped from the reference set.
        source, _, ignored = classify_paths(
            ["licenses.py", "src/authors.py", "notices/build.py", "LICENSE.md", "LICENSE"]
        )
        assert source == ["licenses.py", "notices/build.py", "src/authors.py"]
        assert ignored == ["LICENSE", "LICENSE.md"]

    def test_S401_docs_are_ignored_not_restored(self) -> None:
        _, tests, ignored = classify_paths(["docs/guide.md", "CHANGELOG.md"])
        assert tests == [] and len(ignored) == 2

    def test_S401_a_markdown_file_under_tests_is_documentation(self) -> None:
        _, tests, ignored = classify_paths(["tests/README.md"])
        assert tests == [] and ignored == ["tests/README.md"]

    def test_S401_classification_is_case_insensitive(self) -> None:
        _, tests, _ = classify_paths(["Test/Widget.cs", "TESTS/thing.py"])
        assert len(tests) == 2


# -- generation ---------------------------------------------------------------


class TestGeneration:
    def test_S401_generates_from_a_real_commit(self, repo: Path) -> None:
        (task,) = generate(repo, ["HEAD"])
        assert task.title == "Add a multiply helper"
        assert task.source_paths == ("calc.py",)
        assert task.test_paths == ("tests/test_mul.py",)
        assert task.ignored_paths == ("README.md",)
        assert task.well_formed

    def test_S401_same_inputs_give_the_same_suite(self, repo: Path) -> None:
        assert generate(repo, ["HEAD"]) == generate(repo, ["HEAD"])

    def test_S401_base_is_the_first_parent(self, repo: Path) -> None:
        (task,) = generate(repo, ["HEAD"])
        assert task.base_sha == git(repo, "rev-parse", "HEAD~1")

    def test_S401_unknown_rev_is_skipped_not_fabricated(self, repo: Path) -> None:
        assert generate(repo, ["nope"]) == []

    def test_S401_strict_mode_reports_a_bad_rev(self, repo: Path) -> None:
        # A caller who mistyped a suite file should learn about it rather than
        # get a quietly shorter benchmark.
        with pytest.raises(GitError):
            generate(repo, ["nope"], strict=True)

    def test_S401_a_root_commit_has_no_base_state(self, repo: Path) -> None:
        root = git(repo, "rev-list", "--max-parents=0", "HEAD")
        assert generate(repo, [root]) == []

    def test_S401_deleted_paths_do_not_break_generation(self, repo: Path) -> None:
        # `git checkout head -- <deleted path>` fails, which failed the whole
        # task, so every change that removed or renamed a file was dropped.
        (repo / "tests" / "test_calc.py").unlink()
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # tidy\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Drop the old test")
        (task,) = generate(repo, ["HEAD"])
        assert "tests/test_calc.py" not in task.test_paths
        assert task.source_paths == ("calc.py",)

    def test_S401_deleted_source_is_part_of_the_reference_answer(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The change removed the file, so an agent reproducing the change
        # removes it too. Excluding deletions from the reference set charged
        # the correct behaviour as an unrequested edit -- the deletion landed
        # in files_touched and nowhere in the reference -- so leaving dead code
        # in place scored better than removing it.
        (repo / "legacy.py").write_text("LEGACY = 1\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "add legacy")
        (repo / "legacy.py").unlink()
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef sub(a, b):\n    return a - b\n"
        )
        (repo / "tests" / "test_sub.py").write_text(
            "from calc import sub\n\n\ndef test_s():\n    assert sub(3, 1) == 2\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Drop legacy, add subtract")

        (task,) = generate(repo, ["HEAD"])
        assert task.deleted_source_paths == ("legacy.py",)
        assert "legacy.py" not in task.source_paths

    def test_S401_ill_formed_tasks_are_marked(self) -> None:
        assert not TaskSpec("r", "h", "b", "t", "", (), ()).well_formed          # no tests
        assert not TaskSpec("r", "h", "b", "t", "", (), ("t/a.py",)).well_formed  # no source
        assert not TaskSpec("r", "h", "b", "", "", ("a.py",), ("t/a.py",)).well_formed


# -- the prompt ---------------------------------------------------------------


class TestThePromptDoesNotLeakTheAnswer:
    def test_S401_prompt_names_the_tests_and_forbids_editing_them(
        self, repo: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        prompt = task.prompt()
        assert "tests/test_mul.py" in prompt
        assert "do not" in prompt.lower() and "modify" in prompt.lower()

    def test_S401_the_commit_body_is_withheld_by_default(self, repo: Path) -> None:
        # A squash-merged pull request's body routinely names the functions to
        # add. Passing it through hands over the answer in prose.
        task = TaskSpec(
            repo=str(repo), head_sha="h", base_sha="b",
            title="Add a multiply helper",
            body="Adds mul(a, b) to calc.py returning a * b.",
            source_paths=("calc.py",), test_paths=("tests/test_mul.py",),
        )
        assert "mul(a, b)" not in task.prompt()
        assert "mul(a, b)" in task.prompt(include_body=True)

    def test_S401_answer_bearing_test_payloads_are_flagged(self) -> None:
        # A hash golden is non-invertible; a plaintext expected-output fixture
        # is the solution written down. Both get restored -- the grader needs
        # them -- so a suite that cares must be able to see which is which.
        task = TaskSpec(
            "r", "h", "b", "t", "", ("a.py",),
            ("tests/test_a.py", "tests/fixtures/expected.txt"),
        )
        assert task.answer_bearing_tests == ("tests/fixtures/expected.txt",)


# -- isolation: the blocker ---------------------------------------------------


class TestTheAgentCannotReadTheAnswer:
    """The property the first implementation did not have."""

    def test_S401_starting_state_has_the_tests_but_not_the_implementation(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        assert (dest / "tests" / "test_mul.py").exists(), "the grader is missing"
        assert "def mul" not in (dest / "calc.py").read_text(), (
            "the implementation was restored along with the tests"
        )

    def test_S401_the_task_repo_has_no_history_beyond_its_own_commit(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        assert len(git(dest, "log", "--all", "--oneline").splitlines()) == 1
        assert git(dest, "remote", "-v") == ""

    def test_S401_the_head_commit_is_unreachable_from_the_task_repo(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The exact query a repo-mode agent with git tooling would type first.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        probe = subprocess.run(
            ["git", "-C", str(dest), "cat-file", "-t", task.head_sha],
            capture_output=True, text=True,
        )
        assert probe.returncode != 0, "the reference solution's commit is readable"
        diff = subprocess.run(
            ["git", "-C", str(dest), "diff", f"HEAD..{task.head_sha}"],
            capture_output=True, text=True,
        )
        assert diff.returncode != 0 and "def mul" not in diff.stdout

    def test_S401_the_solution_appears_nowhere_in_the_task_tree(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # Belt and braces: grep the whole directory, .git included.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        hit = subprocess.run(
            ["grep", "-rl", "return a * b", str(dest)], capture_output=True, text=True
        )
        assert hit.stdout.strip() == "", hit.stdout

    def test_S401_an_empty_base_tree_still_gets_a_commit(
        self, tmp_path: Path
    ) -> None:
        # Without --allow-empty the commit fails, the task repo has no HEAD,
        # and `git status` then reports every file the agent did not touch as
        # new -- so `files_touched` becomes the whole tree and diff precision
        # turns to noise. The old code treated that failure as expected.
        src = tmp_path / "src"
        src.mkdir()
        git(src, "init", "-q")
        git(src, "config", "user.name", "t")
        git(src, "config", "user.email", "t@localhost")
        git(src, "commit", "-q", "--allow-empty", "-m", "empty root")
        (src / "thing.py").write_text("VALUE = 1\n")
        (src / "tests").mkdir()
        (src / "tests" / "test_thing.py").write_text(
            "from thing import VALUE\n\n\ndef test_v():\n    assert VALUE == 1\n"
        )
        git(src, "add", "-A")
        git(src, "commit", "-q", "-m", "Add thing")

        (task,) = generate(src, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        assert git(dest, "rev-parse", "HEAD"), "the task repo has no HEAD"
        # The restored grader is committed, so the tree starts clean.
        assert git(dest, "status", "--porcelain") == ""

        # And the degenerate case the flag actually exists for: no tests to
        # restore on top of an empty base, so there is genuinely nothing to
        # commit. A well-formed task can never reach this -- it always has a
        # grader -- which is exactly why the old code could tolerate a failed
        # commit and never be caught doing it.
        empty = TaskSpec(str(src), task.head_sha, task.base_sha, "t", "",
                         ("thing.py",), ())
        bare = tmp_path / "bare"
        build_task_tree(empty, bare)
        assert git(bare, "rev-parse", "HEAD"), (
            "an empty starting tree produced a repo with no HEAD; git status "
            "there reports every file as new"
        )
        assert git(bare, "log", "--oneline", "--all").count("\n") == 0

    def test_S401_at_head_the_tree_is_solved(self, repo: Path, tmp_path: Path) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "head"
        build_task_tree(task, dest, at_head=True)
        assert "def mul" in (dest / "calc.py").read_text()

    def test_S401_generation_never_mutates_the_source_repo(
        self, repo: Path, tmp_path: Path
    ) -> None:
        before = git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain")
        (task,) = generate(repo, ["HEAD"])
        build_task_tree(task, tmp_path / "a")
        build_task_tree(task, tmp_path / "b", at_head=True)
        assert (git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain")) == before
        assert git(repo, "worktree", "list").count("\n") == 0, (
            "a worktree was registered in the source repository"
        )

    def test_S401_rebuilding_over_an_existing_directory_is_clean(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "leftover.txt").write_text("from a previous trial\n")
        build_task_tree(task, dest)
        assert not (dest / "leftover.txt").exists()

    def test_S401_a_failed_init_step_raises_rather_than_limping_on(
        self, repo: Path, tmp_path: Path, monkeypatch
    ) -> None:
        # Every step here can fail for environment reasons -- no git, a
        # read-only directory, a full disk -- and each failure leaves a task
        # tree that looks buildable and grades wrongly. A tolerated failure
        # would be reported as the agent's, which is the worst direction for
        # the error to go.
        import harness.eval.pr_replay as module

        (task,) = generate(repo, ["HEAD"])
        real = module._git

        def fail_on_add(dest, *args, **kwargs):
            if args and args[0] == "add":
                return 1, "", "simulated: could not write the index"
            return real(dest, *args, **kwargs)

        monkeypatch.setattr(module, "_git", fail_on_add)
        with pytest.raises(GitError, match="add"):
            build_task_tree(task, tmp_path / "task")

    def test_S401_a_construction_failure_raises(self, tmp_path: Path) -> None:
        # Returning False here would let a harness bug be recorded as an agent
        # failure, which is the worst possible direction for the error to go.
        bogus = TaskSpec(str(tmp_path / "nope"), "h", "b", "t", "", ("a.py",), ("t/a.py",))
        with pytest.raises(GitError):
            build_task_tree(bogus, tmp_path / "dest")


# -- the grader must survive --------------------------------------------------


class TestTheGraderIsProtected:
    def test_S401_an_untouched_grader_reports_nothing(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        assert grader_was_changed(tree, task) == []

    def test_S401_a_deleted_grader_is_detected(self, repo: Path, tmp_path: Path) -> None:
        # Deleting the tests is a winning move under most runners: pytest exits
        # 5 on "no tests collected" and 0 on an emptied file with --exitfirst.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "tests" / "test_mul.py").unlink()
        assert grader_was_changed(tree, task) == ["tests/test_mul.py"]

    def test_S401_a_weakened_grader_is_detected(self, repo: Path, tmp_path: Path) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "tests" / "test_mul.py").write_text("def test_mul():\n    pass\n")
        assert grader_was_changed(tree, task) == ["tests/test_mul.py"]

    def test_S401_editing_the_source_is_not_tampering(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "calc.py").write_text("def add(a, b):\n    return a + b\n\ndef mul(a,b):\n    return a*b\n")
        assert grader_was_changed(tree, task) == []

    def test_S401_running_the_tests_is_not_tampering(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # Byte-code caches land inside tests/ the moment pytest runs. Counting
        # those would flag every agent that did the obviously right thing.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        subprocess.run(TEST_COMMAND, shell=True, cwd=dest, capture_output=True)
        assert (dest / "tests" / "__pycache__").exists(), "no byproducts were created"
        assert grader_was_changed(tree, task) == []

    def test_S401_committing_does_not_hide_a_rewritten_grader(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # THE attack. `git status` compares to HEAD, and HEAD is the agent's
        # to move: rewrite the grader, commit, and the tree reads as clean.
        # The content hash is taken at build time and lives in this process,
        # so no amount of git surgery inside the tree reaches it.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "tests" / "test_mul.py").write_text("def test_mul():\n    assert True\n")
        git(dest, "add", "-A")
        git(dest, "commit", "-q", "-m", "work")
        assert git(dest, "status", "--porcelain") == "", "precondition: tree is clean"
        assert grader_was_changed(tree, task) == ["tests/test_mul.py"]

    def test_S401_deleting_git_does_not_hide_a_rewritten_grader(
        self, repo: Path, tmp_path: Path
    ) -> None:
        import shutil as _shutil

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "tests" / "test_mul.py").write_text("def test_mul():\n    assert True\n")
        _shutil.rmtree(dest / ".git")
        # The hash check needs no git at all, so the tampering is still
        # reported even though the follow-up scan for added files cannot run.
        assert grader_was_changed(tree, task) == ["tests/test_mul.py"]

    def test_S401_a_destroyed_repository_does_not_blind_the_check(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # An earlier version had to *raise* here, because its scan for added
        # files went through git and a missing repository meant it had
        # gathered no evidence. The manifest needs no git, so a destroyed
        # repository costs nothing: the graders are intact and it says so, and
        # an added conftest.py is still caught.
        import shutil as _shutil

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        _shutil.rmtree(dest / ".git")
        assert grader_was_changed(tree, task) == []
        (dest / "conftest.py").write_text("import calc\n")
        assert grader_was_changed(tree, task) == ["conftest.py"]

    def test_S401_gitignore_cannot_hide_a_grader_config_file(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The tree's .gitignore lives in the agent's own workspace. While the
        # scan used `git ls-files --exclude-standard`, appending a filename to
        # it made that file invisible to every check -- so the agent could add
        # a monkeypatching conftest.py and have it disappear.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / ".gitignore").write_text("conftest.py\ntests/test_zz.py\n")
        (dest / "conftest.py").write_text("import calc\ncalc.mul = lambda a, b: a * b\n")
        (dest / "tests" / "test_zz.py").write_text("def test_zz():\n    pass\n")
        assert grader_was_changed(tree, task) == ["conftest.py", "tests/test_zz.py"]

    def test_S401_deleting_a_pre_existing_test_is_tampering(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The grading command runs the whole suite, so every test file is the
        # grader -- but only the change's own tests were hashed. An agent
        # could ship a regression and delete the test that proved it.
        (task,) = generate(repo, ["HEAD"])
        assert "tests/test_calc.py" not in task.test_paths, "precondition"
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "tests" / "test_calc.py").unlink()
        assert grader_was_changed(tree, task) == ["tests/test_calc.py"]

    def test_S401_editing_a_pre_existing_grader_config_is_tampering(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # GRADER_CONFIG_NAMES used to be consulted only for files the agent
        # *added*, so every name on the list was unguarded whenever the
        # repository already had one -- which is the common case.
        (repo / "conftest.py").write_text("# fixtures\n")
        (repo / "Makefile").write_text("test:\n\t@pytest tests\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "add project config")
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef pow2(a):\n    return a * a\n"
        )
        (repo / "tests" / "test_pow.py").write_text(
            "from calc import pow2\n\n\ndef test_p():\n    assert pow2(3) == 9\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Add a square helper")

        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        assert (dest / "conftest.py").exists(), "precondition: it was already there"
        (dest / "conftest.py").write_text("import calc\ncalc.pow2 = lambda a: a * a\n")
        (dest / "Makefile").write_text("test:\n\t@true\n")
        assert grader_was_changed(tree, task) == ["Makefile", "conftest.py"]

    def test_S401_an_added_conftest_is_tampering(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # Verified against an earlier version as a scored pass: a root
        # conftest.py is imported before collection and can monkeypatch the
        # module under test, without touching a single test file.
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "conftest.py").write_text(
            "import calc\ncalc.mul = lambda a, b: a * b\n"
        )
        assert grader_was_changed(tree, task) == ["conftest.py"]

    @pytest.mark.parametrize(
        "name", ["pytest.ini", "tox.ini", "sitecustomize.py", "Makefile", "hack.pth"]
    )
    def test_S401_added_grader_config_is_tampering(
        self, repo: Path, tmp_path: Path, name: str
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / name).write_text("# whatever\n")
        assert grader_was_changed(tree, task) == [name]

    def test_S401_editing_a_config_file_the_change_itself_added_is_the_task(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The converse. If the reference commit created pyproject.toml, then
        # creating it is the work, not tampering -- otherwise every packaging
        # task would be unpassable by construction.
        (task,) = generate(repo, ["HEAD"])
        task = TaskSpec(
            task.repo, task.head_sha, task.base_sha, task.title, task.body,
            source_paths=task.source_paths + ("pyproject.toml",),
            test_paths=task.test_paths,
        )
        dest = tmp_path / "task"
        tree = build_task_tree(task, dest)
        (dest / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
        assert grader_was_changed(tree, task) == []

    def test_S401_build_byproducts_are_excluded_by_git_itself(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        dest = tmp_path / "task"
        build_task_tree(task, dest)
        exclude = (dest / ".git" / "info" / "exclude").read_text()
        assert "__pycache__/" in exclude
        assert set(BUILD_BYPRODUCTS) <= set(exclude.split())
        assert not (dest / ".gitignore").exists(), (
            "the harness wrote into the tree under test"
        )


# -- validation ---------------------------------------------------------------


class TestValidationIsWhatMakesThisABenchmark:
    async def test_S401_a_real_task_validates(self, repo: Path, tmp_path: Path) -> None:
        (task,) = generate(repo, ["HEAD"])
        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert result.usable, result.reason
        assert result.fails_at_base and result.passes_at_head

    async def test_S401_a_trivial_task_is_rejected(self, repo: Path, tmp_path: Path) -> None:
        # Its test passes without the change, so it measures nothing.
        (repo / "tests" / "test_trivial.py").write_text("def test_ok():\n    assert True\n")
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef unused():\n    return None\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "A test that never needed the change")
        (task,) = generate(repo, ["HEAD"])
        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert not result.usable and "trivial" in (result.reason or "")

    async def test_S401_an_unsolvable_task_is_rejected(self, repo: Path, tmp_path: Path) -> None:
        (repo / "tests" / "test_broken.py").write_text("def test_x():\n    assert False\n")
        (repo / "calc.py").write_text((repo / "calc.py").read_text() + "\n# touched\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "A test that fails at head too")
        (task,) = generate(repo, ["HEAD"])
        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert not result.usable and "not solvable" in (result.reason or "")

    async def test_S401_a_broken_environment_is_not_mistaken_for_a_failing_test(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (task,) = generate(repo, ["HEAD"])
        result = await validate(
            task, "definitely-not-a-command --run", tmp_path / "work", host_run
        )
        assert result.environment_broken and not result.usable

    async def test_S401_a_feature_addition_is_not_called_environmental(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # THE regression this class exists for. A change that adds a module
        # fails at base with "No module named ...", which the environmental
        # check matched -- deleting the most valuable tasks in the suite.
        (repo / "widget.py").write_text("def spin():\n    return 'spun'\n")
        (repo / "tests" / "test_widget.py").write_text(
            "from widget import spin\n\n\ndef test_spin():\n    assert spin() == 'spun'\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Add the widget module")
        (task,) = generate(repo, ["HEAD"])
        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert result.usable, f"a feature-addition task was discarded: {result.reason}"
        assert not result.environment_broken

    async def test_S401_the_environmental_check_is_never_applied_to_the_base(
        self, repo: Path, tmp_path: Path, monkeypatch
    ) -> None:
        # Directly, not via its consequences: the base state's output must
        # never be consulted. A task that adds a module fails at base with
        # "No module named ...", which is on the environmental list, so any
        # reading of base output deletes the most valuable tasks in the suite.
        import harness.eval.pr_replay as module

        (repo / "widget.py").write_text("def spin():\n    return 'spun'\n")
        (repo / "tests" / "test_widget.py").write_text(
            "from widget import spin\n\n\ndef test_spin():\n    assert spin() == 'spun'\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Add the widget module")
        (task,) = generate(repo, ["HEAD"])

        base_outputs: list[str] = []
        real = module._looks_environmental

        def spy(output: str) -> bool:
            base_outputs.append(output)
            return real(output)

        monkeypatch.setattr(module, "_looks_environmental", spy)
        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert result.usable, result.reason
        assert len(base_outputs) == 1, (
            "the environmental check ran more than once; the base state's "
            "output is being consulted"
        )
        assert "No module named" not in base_outputs[0], base_outputs[0]

    async def test_S401_a_renamed_test_stays_solvable(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # A change that replaces one test file with another. The old test
        # asserts the OLD behaviour, so leaving it in the task tree makes the
        # task unsolvable by the reference answer itself -- while validating
        # as usable, because validation was comparing against the pristine
        # head tree where the old file does not exist. The only route to a
        # pass was deleting a test the tamper check did not guard: the eval
        # rewarded exactly the behaviour it exists to catch.
        (repo / "api.py").write_text("def greet():\n    return 'hi'\n")
        (repo / "tests" / "test_old.py").write_text(
            "from api import greet\n\n\ndef test_g():\n    assert greet() == 'hi'\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "add api")

        (repo / "api.py").write_text("def greet():\n    return 'hello'\n")
        (repo / "tests" / "test_old.py").unlink()
        (repo / "tests" / "test_new.py").write_text(
            "from api import greet\n\n\ndef test_g():\n    assert greet() == 'hello'\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Rename the greeting test")

        (task,) = generate(repo, ["HEAD"])
        assert task.test_paths == ("tests/test_new.py",)
        assert task.deleted_test_paths == ("tests/test_old.py",)

        dest = tmp_path / "task"
        build_task_tree(task, dest)
        assert not (dest / "tests" / "test_old.py").exists(), (
            "the change deleted this test, so it is not part of the grader"
        )
        assert (dest / "tests" / "test_new.py").exists()

        result = await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert result.usable, result.reason

    async def test_S401_validation_checks_the_tree_the_agent_actually_gets(
        self, repo: Path, tmp_path: Path, monkeypatch
    ) -> None:
        # Structural: the head state must be built from the base tree with the
        # change applied, never from the pristine head tree. Asserted on the
        # arguments, because the two agree on a well-behaved fixture and
        # differ exactly when a file was deleted.
        import harness.eval.pr_replay as module

        (task,) = generate(repo, ["HEAD"])
        seen = []
        real = module.build_task_tree

        def spy(t, dest, **kw):
            seen.append(kw.get("at_head", False))
            return real(t, dest, **kw)

        monkeypatch.setattr(module, "build_task_tree", spy)
        await validate(task, TEST_COMMAND, tmp_path / "work", host_run)
        assert seen == [False, True]

        # `at_head` must mean "the agent's tree, solved" -- base, plus the
        # change layered on -- and not "the head tree". The two differ by
        # every file the change deleted, and a file the change removed is one
        # the agent was never asked to remove, so it is still there when the
        # agent finishes. Validating against a tree without it asks a question
        # nobody will ever be graded on.
        (repo / "legacy.py").write_text("LEGACY = 1\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "add legacy")
        (repo / "legacy.py").unlink()
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef sub(a, b):\n    return a - b\n"
        )
        (repo / "tests" / "test_sub.py").write_text(
            "from calc import sub\n\n\ndef test_sub():\n    assert sub(3, 1) == 2\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Drop legacy, add subtract")

        (later,) = generate(repo, ["HEAD"])
        starting = tmp_path / "start"
        solved = tmp_path / "solved"
        real(later, starting)
        real(later, solved, at_head=True)
        assert (starting / "legacy.py").exists(), (
            "the change deleted a source file the agent is not asked to delete"
        )
        assert (solved / "legacy.py").exists(), (
            "the head state was built from the head tree, not from the tree "
            "the agent is actually given"
        )
        assert "def sub" in (solved / "calc.py").read_text()
        assert "def sub" not in (starting / "calc.py").read_text()

    def test_S401_an_export_ignored_grader_blames_the_harness_not_the_task(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # `git archive` honours export-ignore and exits 0 having written
        # nothing. Without a check the task ships with no grader, and
        # validation reports "tests fail at head" -- pointing at the task when
        # the fault is entirely ours. `/tests export-ignore` is a near-
        # universal convention in some ecosystems, so this would drop whole
        # suites with a misleading reason.
        # The attribute must exist at the revision being archived, so it goes
        # in first and the change under test comes after it.
        (repo / ".gitattributes").write_text("/tests export-ignore\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Add gitattributes")
        (repo / "calc.py").write_text(
            (repo / "calc.py").read_text() + "\n\ndef div(a, b):\n    return a / b\n"
        )
        (repo / "tests" / "test_div.py").write_text(
            "from calc import div\n\n\ndef test_div():\n    assert div(6, 3) == 2\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "Add a divide helper")
        (task,) = generate(repo, ["HEAD"])
        assert task.test_paths == ("tests/test_div.py",), task.test_paths
        with pytest.raises(GitError, match="export-ignore"):
            build_task_tree(task, tmp_path / "task")

    async def test_S401_ill_formed_task_is_rejected_without_running_anything(
        self, tmp_path: Path
    ) -> None:
        task = TaskSpec("/nonexistent", "h", "b", "t", "", ("a.py",), ())
        result = await validate(task, "exit 0", tmp_path, host_run)
        assert not result.usable and result.reason == "no tests"
        assert list(tmp_path.iterdir()) == [], "a tree was built for an unusable task"

    async def test_S401_validation_leaves_no_directories_behind(
        self, repo: Path, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (task,) = generate(repo, ["HEAD"])
        await validate(task, TEST_COMMAND, work, host_run)
        assert list(work.iterdir()) == [], list(work.iterdir())

    async def test_S401_concurrent_trials_do_not_share_a_directory(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # Named by task id alone, the second trial's construction failed and
        # was reported as a defect in the task.
        work = tmp_path / "work"
        work.mkdir()
        (task,) = generate(repo, ["HEAD"])
        seen = []
        import harness.eval.pr_replay as module

        real = module.build_task_tree

        def spy(t, dest, **kw):
            seen.append(dest.name)
            return real(t, dest, **kw)

        module.build_task_tree = spy
        try:
            await validate(task, TEST_COMMAND, work, host_run)
            await validate(task, TEST_COMMAND, work, host_run)
        finally:
            module.build_task_tree = real
        assert len(set(seen)) == len(seen) == 4, seen


# -- metrics ------------------------------------------------------------------


class TestMetrics:
    def test_S401_precision_and_recall_fail_in_opposite_directions(self) -> None:
        # Touched twice as many files as needed: precision 0.5, recall 1.0.
        assert diff_precision({"a", "b"}, {"a"}) == (0.5, 1.0)
        # Touched half of what was needed: precision 1.0, recall 0.5.
        assert diff_precision({"a"}, {"a", "b"}) == (1.0, 0.5)

    def test_S401_empty_sets_do_not_score_as_perfect(self) -> None:
        assert diff_precision(set(), {"a"}) == (0.0, 0.0)
        assert diff_precision({"a"}, set()) == (0.0, 0.0)

    def test_S401_a_tampered_trial_cannot_be_constructed_as_a_pass(self) -> None:
        with pytest.raises(TamperedPassError):
            TaskOutcome("t", passed=True, tampered=True)
        assert TaskOutcome("t", passed=False, tampered=True).tampered

    def test_S401_not_measured_is_not_zero(self) -> None:
        report = SuiteReport([TaskOutcome("a", passed=True), TaskOutcome("b", passed=False)])
        assert report.regressions is None, "unmeasured regressions reported as clean"
        assert "not measured" in report.render()
        assert report.first_edit_trials == 0
        assert report.never_edited == 0, "'not measured' counted as 'never edited'"

    def test_S401_measured_regressions_carry_their_denominator(self) -> None:
        report = SuiteReport(
            [
                TaskOutcome("a", passed=True, regressions=1),
                TaskOutcome("b", passed=True, regressions=0),
                TaskOutcome("c", passed=True),  # not measured
            ]
        )
        assert report.regressions == 1 and report.regression_trials == 2
        assert "over 2/3 trials" in report.render()

    def test_S401_never_edited_is_not_averaged_as_zero(self) -> None:
        report = SuiteReport(
            [
                TaskOutcome("a", passed=True, turns_to_first_edit=4, first_edit_measured=True),
                TaskOutcome("b", passed=False, turns_to_first_edit=None, first_edit_measured=True),
            ]
        )
        assert report.mean_turns_to_first_edit == 4.0
        assert report.never_edited == 1
        assert report.first_edit_trials == 2

    def test_S401_precision_publishes_the_trials_it_averaged_over(self) -> None:
        # An agent that does nothing on most tasks must not read as precise.
        report = SuiteReport(
            [
                TaskOutcome("a", passed=True, files_touched=frozenset({"x"}),
                            reference_files=frozenset({"x"})),
                TaskOutcome("b", passed=False),
                TaskOutcome("c", passed=False),
            ]
        )
        assert report.mean_precision == 1.0
        assert report.precision_trials == 1
        assert "over 1/3 trials that edited" in report.render()

    def test_S401_counts_are_reported(self) -> None:
        report = SuiteReport(
            [
                TaskOutcome("a", passed=False, refused=True),
                TaskOutcome("b", passed=False, errored=True),
                TaskOutcome("c", passed=False, tampered=True),
            ]
        )
        assert (report.refusals, report.errors, report.tampered) == (1, 1, 1)
        assert report.pass_rate == 0.0

    def test_S401_empty_report_does_not_divide_by_zero(self) -> None:
        assert SuiteReport().render() == "no trials"
        assert SuiteReport().pass_rate == 0.0


# -- the held-out split -------------------------------------------------------


class TestHeldOutSplit:
    def test_S401_split_is_about_the_target_fraction(self) -> None:
        ids = [f"task-{i:05d}" for i in range(4000)]
        observed = sum(is_heldout(t) for t in ids) / len(ids)
        # Tight enough that 1/4 or 1/2 would fail: at n=4000 the standard
        # error is ~0.0075, so 0.02 is under three sigma from a real 1/3 and
        # nowhere near either neighbour.
        assert abs(observed - HELDOUT_FRACTION) < 0.02, observed

    def test_S401_membership_is_stable_as_the_suite_grows(self) -> None:
        first = [f"task-{i:05d}" for i in range(50)]
        grown = first + [f"task-{i:05d}" for i in range(50, 500)]
        suite = Suite("s", "r", ("x",), "pytest")
        _, held_before = suite.split(first)
        _, held_after = suite.split(grown)
        assert set(held_before) == set(held_after) & set(first)
        assert held_before, "the fixture happened to hold out nothing"

    def test_S401_split_is_deterministic_across_processes(self) -> None:
        # Pinned values, not self-consistency: `is_heldout(x) == is_heldout(x)`
        # holds for any implementation, including a broken one. These are the
        # sha256-derived answers, so swapping in Python's salted hash() -- or
        # any other function -- fails here.
        assert [is_heldout(f"task-{i:05d}") for i in range(10)] == [
            False, True, True, True, False, False, False, False, False, False,
        ]

    def test_S401_the_score_itself_is_pinned(self) -> None:
        # The booleans above cannot distinguish a denominator of 2**32 from
        # 0xFFFFFFFF -- the two differ by about 2^-32, so no sample of ids
        # would ever disagree. The score does, to full precision.
        # Exact equality, not approx: the two candidate denominators differ by
        # a factor of 2**32/(2**32-1), about 2e-10 on these values, so any
        # tolerance loose enough to be comfortable is loose enough to miss it.
        assert heldout_score("task-00000") == 0.7840683846734464
        assert heldout_score("task-00001") == 0.15210718451999128
        assert all(0.0 <= heldout_score(f"t{i}") < 1.0 for i in range(500))

    def test_S401_dev_and_heldout_partition_the_suite(self) -> None:
        ids = [f"task-{i:05d}" for i in range(200)]
        dev, held = Suite("s", "r", ("x",), "pytest").split(ids)
        assert sorted(dev + held) == sorted(ids)
        assert set(dev).isdisjoint(held)
        assert dev and held

    def test_S401_a_fraction_of_one_holds_out_everything(self) -> None:
        # The endpoint an off-by-one in the hash denominator breaks.
        assert all(is_heldout(f"t{i}", fraction=1.0) for i in range(500))
        assert not any(is_heldout(f"t{i}", fraction=0.0) for i in range(500))


class TestSuiteFile:
    def test_S401_suite_round_trips(self, tmp_path: Path) -> None:
        suite = Suite("s", "/repo", ("a", "b"), "pytest -q", regression_command="pytest")
        path = tmp_path / "suite.json"
        suite.save(path)
        assert Suite.load(path) == suite

    def test_S401_an_absent_regression_command_stays_absent(self, tmp_path: Path) -> None:
        suite = Suite("s", "/repo", ("a",), "pytest -q")
        path = tmp_path / "suite.json"
        suite.save(path)
        assert Suite.load(path).regression_command is None

    @pytest.mark.parametrize(
        "payload",
        ['{"name": "s"}', '{"name":"s","repo":"r","revs":[],"test_command":"x"}', "not json"],
    )
    def test_S401_a_malformed_suite_is_rejected_clearly(
        self, tmp_path: Path, payload: str
    ) -> None:
        path = tmp_path / "suite.json"
        path.write_text(payload)
        with pytest.raises(SuiteError):
            Suite.load(path)
