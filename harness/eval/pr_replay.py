"""Generate coding tasks by replaying merged changes (S-401).

Terminal-Bench measures terminal work, is ±5 points noisy at n=89, and costs
hours per run. None of that makes it a good instrument for "is the harness
getting better at coding". This builds a second one out of history that
already exists: take a merged change, put the tree back to its parent, restore
**only its tests**, and ask the agent to make them pass. The change's own tests
are the grader, and the real diff is the reference answer.

The construction is deliberately blunt -- ``git checkout base`` then
``git checkout head -- <test paths>`` -- because anything cleverer is a place
for the generator to disagree with git about what the change was.

**A generated task is worthless until it is validated.** A task whose tests
already pass at the base is trivial; one whose tests fail even at the head is
impossible. Either silently poisons the metric, and both are common enough
that generation without validation produces a suite that measures nothing.
:func:`validate` is therefore not optional polish -- it is what makes the
output a benchmark rather than a list of hopes.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "TaskSpec",
    "Validation",
    "classify_paths",
    "generate",
    "validate",
    "TEST_PATH_PATTERNS",
]

#: A path is a *test* if any of these match. Deliberately conservative and
#: explicit: misclassifying a source file as a test would hand the agent the
#: implementation, and misclassifying a test as source would delete the grader.
TEST_PATH_PATTERNS = (
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)test_[^/]+$"),
    re.compile(r"[^/]+_test\.[a-z]+$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"\.spec\.[jt]sx?$"),
    re.compile(r"\.test\.[jt]sx?$"),
)

#: Paths that are neither source nor grader -- documentation, lockfiles,
#: generated indexes. Excluded from *both* sides: restoring them would leak
#: the answer, and counting them in diff precision would reward the agent for
#: touching files the task never needed.
_IGNORED_PATTERNS = (
    re.compile(r"\.md$"),
    re.compile(r"(^|/)specs?/"),
    re.compile(r"(^|/)docs?/"),
    re.compile(r"\.lock$"),
    re.compile(r"(^|/)CHANGELOG"),
)


def _is_test(path: str) -> bool:
    return any(p.search(path) for p in TEST_PATH_PATTERNS)


def _is_ignored(path: str) -> bool:
    return any(p.search(path) for p in _IGNORED_PATTERNS)


def classify_paths(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split changed paths into ``(source, tests, ignored)``.

    Order matters: a file under ``tests/`` that also ends in ``.md`` is
    documentation, not a grader, so the ignore check runs first.
    """
    source, tests, ignored = [], [], []
    for path in paths:
        if _is_ignored(path):
            ignored.append(path)
        elif _is_test(path):
            tests.append(path)
        else:
            source.append(path)
    return sorted(source), sorted(tests), sorted(ignored)


@dataclass(frozen=True)
class TaskSpec:
    """One replayable task, fully described by two commits and a split."""

    repo: str
    head_sha: str
    base_sha: str
    title: str
    body: str
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    ignored_paths: tuple[str, ...] = ()

    @property
    def task_id(self) -> str:
        return self.head_sha[:12]

    @property
    def well_formed(self) -> bool:
        """Whether this could be a task at all, before running anything.

        A change with no tests has no grader; a change with no source is
        nothing for the agent to write.
        """
        return bool(self.test_paths) and bool(self.source_paths)

    def prompt(self) -> str:
        """What the agent is told. The change's own description, plus the
        tests it must satisfy -- never the diff."""
        lines = [self.title.strip()]
        if self.body.strip():
            lines += ["", self.body.strip()]
        lines += [
            "",
            "The following tests describe the expected behaviour and are "
            "already present in the working tree. Make them pass without "
            "modifying them:",
        ]
        lines += [f"  {p}" for p in self.test_paths]
        return "\n".join(lines)


@dataclass(frozen=True)
class Validation:
    """Whether a generated task is usable, and why not if it isn't."""

    task_id: str
    fails_at_base: bool | None = None
    passes_at_head: bool | None = None
    reason: str | None = None
    detail: str = ""
    environment_broken: bool = False

    @property
    def usable(self) -> bool:
        return (
            self.fails_at_base is True
            and self.passes_at_head is True
            and not self.environment_broken
        )


def _git(repo: Path, *args: str, timeout: float = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def _changed_paths(repo: Path, base: str, head: str) -> list[str]:
    code, out = _git(repo, "diff", "--name-only", f"{base}..{head}")
    return [line.strip() for line in out.splitlines() if line.strip()] if code == 0 else []


def generate(repo: Path, revs: list[str]) -> list[TaskSpec]:
    """Build a task per revision in ``revs``.

    ``revs`` are commit-ish; each task's base is that commit's **first
    parent**, which for a merge is the mainline before the change landed.
    Reproducible by construction: the output is a function of the repository
    and the revision list, with no sampling and no clock.
    """
    tasks: list[TaskSpec] = []
    for rev in revs:
        code, head = _git(repo, "rev-parse", rev)
        if code != 0:
            continue
        code, base = _git(repo, "rev-parse", f"{rev}^1")
        if code != 0:
            continue
        code, subject = _git(repo, "log", "-1", "--format=%s", head)
        _, body = _git(repo, "log", "-1", "--format=%b", head)
        source, tests, ignored = classify_paths(_changed_paths(repo, base, head))
        tasks.append(
            TaskSpec(
                repo=str(repo),
                head_sha=head,
                base_sha=base,
                title=subject if code == 0 else "",
                body=body,
                source_paths=tuple(source),
                test_paths=tuple(tests),
                ignored_paths=tuple(ignored),
            )
        )
    return tasks


def build_worktree(task: TaskSpec, dest: Path) -> bool:
    """Materialise the task's starting state at ``dest``.

    Base tree, then the change's tests restored on top: the grader exists, the
    implementation does not. Uses a detached worktree so the source repository
    is never modified -- a generator that mutated the repo it reads from would
    be unusable on anything you care about.
    """
    repo = Path(task.repo)
    code, _ = _git(repo, "worktree", "add", "--detach", str(dest), task.base_sha)
    if code != 0:
        return False
    code, _ = _git(dest, "checkout", task.head_sha, "--", *task.test_paths)
    return code == 0


def remove_worktree(task: TaskSpec, dest: Path) -> None:
    """Drop a worktree created by :func:`build_worktree`."""
    _git(Path(task.repo), "worktree", "remove", "--force", str(dest))


def validate(
    task: TaskSpec, test_command: str, workdir: Path, timeout: float = 300
) -> Validation:
    """Check the task is solvable and not already solved.

    Runs ``test_command`` twice: at the constructed starting state, where it
    must **fail**, and at the change's head, where it must **pass**. A task
    that passes at base is trivial; one that fails at head is impossible.
    Both are silent poison in an eval suite, which is why this exists.
    """
    if not task.well_formed:
        return Validation(
            task.task_id,
            reason="no tests" if not task.test_paths else "no source changes",
        )

    base_dir = workdir / f"{task.task_id}-base"
    if not build_worktree(task, base_dir):
        return Validation(task.task_id, reason="could not build base worktree")
    try:
        base_code, base_out = _run(test_command, base_dir, timeout)
    finally:
        remove_worktree(task, base_dir)

    head_dir = workdir / f"{task.task_id}-head"
    code, _ = _git(Path(task.repo), "worktree", "add", "--detach", str(head_dir), task.head_sha)
    if code != 0:
        return Validation(
            task.task_id,
            fails_at_base=base_code != 0,
            reason="could not build head worktree",
        )
    try:
        head_code, head_out = _run(test_command, head_dir, timeout)
    finally:
        _git(Path(task.repo), "worktree", "remove", "--force", str(head_dir))

    # A command that cannot run in a bare worktree is a broken *task*, not a
    # failing one, and must never be counted as "fails at base".
    if _looks_environmental(base_out) or _looks_environmental(head_out):
        return Validation(
            task.task_id,
            fails_at_base=None,
            passes_at_head=None,
            environment_broken=True,
            reason=(
                "the test command could not run in a bare worktree (a "
                "worktree has only tracked files: no virtualenv, no installed "
                "dependencies). Use absolute interpreter paths or add a setup "
                "step."
            ),
            detail=(head_out or base_out)[-400:],
        )

    fails_at_base = base_code != 0
    passes_at_head = head_code == 0
    reason = None
    if not fails_at_base:
        reason = "tests already pass at base (task is trivial)"
    elif not passes_at_head:
        reason = "tests fail at head (task is not solvable as generated)"
    return Validation(
        task.task_id,
        fails_at_base=fails_at_base,
        passes_at_head=passes_at_head,
        reason=reason,
        detail=(head_out or base_out)[-400:],
    )


#: Output signatures meaning the command could not run at all, as opposed to
#: running and reporting failures. A fresh worktree contains only *tracked*
#: files -- no virtualenv, no node_modules, no build output -- so a test
#: command referencing them fails identically to a genuinely failing test.
#: Without this distinction a broken command reads as "tests fail at base",
#: which is exactly what a valid task looks like, and the suite fills up with
#: tasks no agent could ever pass.
_ENVIRONMENT_BROKEN = (
    "no such file or directory",
    "command not found",
    "modulenotfounderror: no module named 'pytest'",
    "no module named pytest",
    "permission denied",
    "cannot execute",
)


def _looks_environmental(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _ENVIRONMENT_BROKEN)


def _run(command: str, cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or ""))
