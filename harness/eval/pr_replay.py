"""Generate coding tasks by replaying merged changes (S-401).

Terminal-Bench measures terminal work, is ±5 points noisy at n=89, and costs
hours per run. None of that makes it a good instrument for "is the harness
getting better at coding". This builds a second one out of history that
already exists: take a merged change, put the tree back to its parent, restore
**only its tests**, and ask the agent to make them pass. The change's own tests
are the grader; the real diff is the reference answer.

Three properties this has to get right, each learned by getting it wrong:

**The agent must not be able to read the answer.** An earlier version used
``git worktree add``, which shares the source repository's object database — so
``git log --all`` and ``git diff HEAD..<head>`` in the agent's own working
directory returned the complete reference solution (22,565 characters of it, on
a task from this repo). Against a repo-mode agent with git tooling that is the
first thing it would type. The task tree is now built with ``git archive`` into
a **fresh repository with a single commit and no remotes**, so there is no
history to mine.

**A generated task is worthless until it is validated.** A task whose tests
already pass at the base is trivial; one whose tests fail at the head is
impossible. Both silently poison the metric.

**Validation must not discard valid tasks.** The environmental check runs only
against the *head* state, where any failure genuinely is environmental. Running
it against the base state deleted every task whose base failure mentioned a
missing file — which is exactly what a PR that *adds* a file looks like, the
most valuable class of task in the suite.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:  # pragma: no cover
    from harness.eval.grading import Graded

#: How a graded command is executed. Supplied by the caller so validation and
#: trial grading provably share an environment.
RunCommand = Callable[[str, Path, float], Awaitable["Graded"]]

__all__ = [
    "TaskSpec",
    "Validation",
    "GitError",
    "classify_paths",
    "generate",
    "build_task_tree",
    "validate",
    "grader_was_changed",
    "TaskTree",
    "tree_manifest",
    "GRADER_CONFIG_NAMES",
    "TEST_PATH_PATTERNS",
    "IGNORED_PATH_PATTERNS",
    "BUILD_BYPRODUCTS",
    "RunCommand",
]


class GitError(RuntimeError):
    """A git command failed in a way the caller must not paper over."""


#: A path is a *test* if any of these match.
#:
#: Path-based classification cannot be exact, and the failure mode is
#: asymmetric: a production file misread as a test is **restored at head**,
#: which hands the agent part of the answer. ``(^|/)testing/`` was dropped for
#: exactly that reason — it is a production package in a large fraction of Go
#: and Python projects. ``(^|/)tests?/`` has the same theoretical problem
#: (Django ships ``django/test/``) and is kept because the convention is
#: overwhelming; the bound on the damage is validation, which rejects any task
#: whose restored files make the tests pass at base.
#:
#: Case-insensitive: Swift's ``Tests/MyAppTests/`` and .NET's ``Test/`` are as
#: real as ``tests/``.
TEST_PATH_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(^|/)tests?/",            # tests/ test/
        r"(^|/)__tests__/",         # Jest
        r"(^|/)spec/",              # RSpec
        r"(^|/)test[-_]?utils?/",   # shared test helpers
        r"(^|/)test_[^/]+$",
        r"[^/]+_test\.[a-z]+$",
        r"[^/]+Tests?\.(swift|cs|java|kt)$",
        r"(^|/)conftest\.py$",
        r"\.spec\.[jt]sx?$",
        r"\.test\.[jt]sx?$",
    )
)

#: Neither source nor grader: documentation, lockfiles, generated indexes.
#: Excluded from both sides — restoring them could leak the answer, and
#: counting them in diff precision would reward touching files the task never
#: needed.
#:
#: **Every pattern here matches a file, never a directory segment**, and that
#: is the invariant that keeps :data:`TEST_PATH_PATTERNS` reachable: this list
#: is consulted first, so a directory pattern here would shadow a whole test
#: convention. An earlier version ignored all of ``specs?/`` and ``docs?/``,
#: which swallowed Ruby's ``spec/`` suite entirely — the RSpec pattern below
#: became dead code and every Ruby repository produced an empty suite.
#: `test_S401_the_ignore_list_contains_no_directory_patterns` pins it.
IGNORED_PATH_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\.(md|rst|txt|adoc)$",
        r"(^|/)(package|poetry|pnpm|yarn|cargo|composer|gemfile)[-.]?lock",
        # The extension alternation is deliberately the documentation set and
        # not `[a-z]+`: `AUTHORS.md` is a notice file, `authors.py` is source,
        # and the looser pattern cannot tell them apart -- it would drop the
        # module from the reference set entirely.
        r"(^|/)(CHANGELOG|LICENSE|NOTICE|AUTHORS)(\.(md|rst|txt|adoc))?$",
        r"\.lock$",
    )
)

#: Written into the task repo's ``.git/info/exclude``. Running a test suite
#: creates byte-code caches, coverage data and build directories; counting
#: those as files the agent "touched" would sink diff precision for every
#: agent that did the obviously right thing and ran the tests.
#:
#: Covers more than Python and Node deliberately: a Java agent running
#: ``mvn test`` puts hundreds of files under ``target/classes/`` into the
#: touched set, and ``precision_trials`` does **not** catch it — the trial did
#: edit, so the denominator stays healthy while the numerator becomes noise.
#: The list is a heuristic and always will be; a repository whose real source
#: lives in ``bin/`` or ``dist/`` is misread, which is why the task tree's own
#: ``.gitignore`` is left untouched and this goes in ``.git/info/exclude``
#: where it is additive rather than authoritative.
BUILD_BYPRODUCTS = (
    # Python
    "__pycache__/", "*.py[cod]", ".pytest_cache/", ".mypy_cache/",
    ".ruff_cache/", ".coverage", ".coverage.*", "htmlcov/", "*.egg-info/",
    ".tox/", ".nox/", ".hypothesis/",
    # JavaScript / TypeScript
    "node_modules/", ".next/", ".nuxt/", ".svelte-kit/", ".turbo/",
    "dist/", "out/", "coverage/",
    # JVM
    "target/", "build/", ".gradle/", ".m2/", "*.class",
    # Go, Rust, .NET, PHP, Ruby
    "vendor/", "bin/", "obj/", "*.o", "*.so", "*.dll",
    ".phpunit.result.cache", ".bundle/", "tmp/",
    # Generic
    ".cache/", "*.log",
)


#: Test payloads that are *answers* rather than graders. A hash golden is
#: non-invertible and safe to restore; a plaintext expected-output fixture is
#: the solution written down. Restored anyway (the grader needs them) but
#: recorded so a suite can exclude such tasks.
ANSWER_BEARING_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (r"(^|/)fixtures?/", r"(^|/)golden/", r"(^|/)snapshots?/", r"\.snap$")
)


def _is_test(path: str) -> bool:
    return any(p.search(path) for p in TEST_PATH_PATTERNS)


def _is_ignored(path: str) -> bool:
    return any(p.search(path) for p in IGNORED_PATH_PATTERNS)


def _is_answer_bearing(path: str) -> bool:
    return any(p.search(path) for p in ANSWER_BEARING_PATTERNS)


def classify_paths(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split changed paths into ``(source, tests, ignored)``.

    The ignore list is consulted first, so a ``.md`` under ``tests/`` is
    documentation rather than a grader. That ordering is only safe because
    every ignore pattern matches a *file* — see
    :data:`IGNORED_PATH_PATTERNS` — which is what leaves ``spec/`` free to
    mean "Ruby tests".
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
    #: Test files the change *removed*. Part of the grader change just as much
    #: as the files it added: the grader is "the tests as of head", and a test
    #: the change deleted is not one of them. Leaving them in place made tasks
    #: that rename a test unsolvable **even by the reference answer**, while
    #: still validating as usable -- and the only route to a pass was deleting
    #: a file the tamper check did not guard, so the eval rewarded exactly the
    #: behaviour it exists to catch.
    deleted_test_paths: tuple[str, ...] = ()
    #: Source files the change *removed*. Part of the reference answer: an
    #: agent that reproduces the change deletes them. Excluding them charged an
    #: agent for doing the right thing — the deletion landed in
    #: ``files_touched`` and nowhere in the reference set, so leaving dead code
    #: in place scored better than removing it.
    deleted_source_paths: tuple[str, ...] = ()

    @property
    def task_id(self) -> str:
        return self.head_sha[:12]

    @property
    def answer_bearing_tests(self) -> tuple[str, ...]:
        """Restored test payloads that may contain the answer verbatim."""
        return tuple(p for p in self.test_paths if _is_answer_bearing(p))

    @property
    def well_formed(self) -> bool:
        """Whether this could be a task at all, before running anything."""
        return bool(self.test_paths) and bool(self.source_paths) and bool(self.title)

    def prompt(self, *, include_body: bool = False) -> str:
        """What the agent is told.

        ``include_body`` defaults to **False**. A squash-merged pull request's
        body routinely names the functions and files to add — on this
        repository one commit body lists every symbol its change introduced —
        so passing it through would hand over the answer in prose. The title
        alone states the goal; the tests state the contract.
        """
        lines = [self.title.strip()]
        if include_body and self.body.strip():
            lines += ["", self.body.strip()]
        lines += [
            "",
            "The tests listed below are already present in the working tree "
            "and describe the expected behaviour. Make them pass. Do not "
            "modify, delete or disable them — a run that changes them does "
            "not count as a pass.",
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


def _git(
    repo: Path, *args: str, timeout: float = 120, strip: bool = True
) -> tuple[int, str, str]:
    """Run git, returning ``(code, stdout, stderr)``.

    stderr is returned rather than discarded: an earlier version collapsed
    every failure into ``(1, str(exc))``, so a root commit, a shallow clone and
    a typo'd revision all vanished identically with no diagnostic.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)
    out = proc.stdout or ""
    return proc.returncode, out.strip() if strip else out, (proc.stderr or "").strip()


def _changed_paths(repo: Path, base: str, head: str) -> list[str]:
    """Paths changed between base and head that **exist at head**.

    ``--diff-filter=d`` excludes deletions, because they cannot be restored.
    They are not discarded, though — :func:`_deleted_paths` collects them
    separately, and the deleted *tests* among them have to be removed from the
    task tree or the grader is not the change's grader.

    ``--no-renames`` throughout. With rename detection on, moving
    ``test_old.py`` to ``test_new.py`` is a single ``R`` entry and appears in
    neither filter, so the old file was never removed from the task tree and
    the task became unsolvable. A rename is a delete plus an add, and that is
    exactly the accounting the task tree needs.
    """
    code, out, err = _git(
        repo, "diff", "--name-only", "--no-renames", "--diff-filter=d",
        f"{base}..{head}",
    )
    if code != 0:
        raise GitError(f"could not diff {base}..{head}: {err}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _deleted_paths(repo: Path, base: str, head: str) -> list[str]:
    """Paths the change removed: present at base, absent at head.

    ``--no-renames`` for the reason given in :func:`_changed_paths`: a renamed
    file is a deletion, and rename detection makes it invisible to this query.
    """
    code, out, err = _git(
        repo, "diff", "--name-only", "--no-renames", "--diff-filter=D",
        f"{base}..{head}",
    )
    if code != 0:
        raise GitError(f"could not diff {base}..{head}: {err}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def generate(repo: Path, revs: list[str], *, strict: bool = False) -> list[TaskSpec]:
    """Build a task per revision in ``revs``.

    Each task's base is the revision's **first parent** — for a merge, the
    mainline before the change landed. Reproducible by construction: a pure
    function of the repository and the revision list, with no sampling and no
    clock.

    ``strict`` raises on a revision that cannot be resolved instead of skipping
    it, so a caller who supplied a bad list learns about it. It governs only
    *resolution*: a failure after both endpoints resolved is a broken
    repository, and :class:`GitError` propagates from here either way.
    """
    tasks: list[TaskSpec] = []
    for rev in revs:
        code, head, err = _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")
        if code != 0:
            if strict:
                raise GitError(f"unknown revision {rev!r}: {err}")
            continue
        code, base, err = _git(repo, "rev-parse", "--verify", f"{rev}^1")
        if code != 0:
            # A root commit has no parent, so there is no "before" state.
            if strict:
                raise GitError(f"{rev!r} has no first parent (root commit?): {err}")
            continue
        _, subject, _ = _git(repo, "log", "-1", "--format=%s", head)
        _, body, _ = _git(repo, "log", "-1", "--format=%b", head)
        # Deliberately not guarded. An unresolvable revision above is a user
        # error and is skipped; a diff that fails *between two commits that
        # both just resolved* is a broken repository, and swallowing it would
        # turn a corrupt object store into a quietly shorter benchmark.
        source, tests, ignored = classify_paths(_changed_paths(repo, base, head))
        deleted_source, deleted_tests, _ = classify_paths(
            _deleted_paths(repo, base, head)
        )
        tasks.append(
            TaskSpec(
                repo=str(repo), head_sha=head, base_sha=base,
                title=subject, body=body,
                source_paths=tuple(source), test_paths=tuple(tests),
                ignored_paths=tuple(ignored),
                deleted_test_paths=tuple(deleted_tests),
                deleted_source_paths=tuple(deleted_source),
            )
        )
    return tasks


def _extract(repo: Path, rev: str, dest: Path, paths: tuple[str, ...] = ()) -> None:
    """Extract ``rev``'s tree (or ``paths`` within it) into ``dest``.

    ``git archive`` copies content without any repository metadata, which is
    what makes the result isolated: no objects, no refs, no remotes, nothing to
    mine for the answer.
    """
    dest.mkdir(parents=True, exist_ok=True)
    args = ["archive", "--format=tar", rev, *paths]
    try:
        archive = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, timeout=180
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise GitError(f"git archive failed: {exc}") from exc
    if archive.returncode != 0:
        raise GitError(
            f"git archive {rev} failed: {archive.stderr.decode(errors='replace')[:300]}"
        )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True
    )
    if extract.returncode != 0:
        raise GitError(f"tar extract failed: {extract.stderr.decode(errors='replace')[:300]}")


def _is_byproduct(path: str) -> bool:
    """Whether ``path`` is a build or test artefact rather than content.

    Matched in Python against :data:`BUILD_BYPRODUCTS` rather than delegated to
    ``git check-ignore``. That is deliberate: ``--exclude-standard`` reads the
    tree's own ``.gitignore``, which lives **inside the agent's writable
    workspace** — appending ``conftest.py`` to it made an added grader-config
    file invisible to every check that used it. Ignore rules that decide
    whether the harness can see something must come from somewhere the agent
    cannot write.

    Every pattern is a single path segment (asserted by a test), so a pattern
    matches if it matches any segment: ``__pycache__/`` catches the directory
    at any depth, ``*.pyc`` catches the file.
    """
    segments = path.split("/")
    return any(
        fnmatch.fnmatch(segment, pattern)
        for segment in segments
        for pattern in _BYPRODUCT_PATTERNS
    )


#: Trailing slashes stripped once, at import.
_BYPRODUCT_PATTERNS = tuple(p.rstrip("/") for p in BUILD_BYPRODUCTS)


def tree_manifest(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every content file under ``root``.

    Excludes ``.git`` (the agent's repository state is not content) and
    anything :func:`_is_byproduct` matches.
    """
    manifest: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() and not path.exists():
            continue
        relative = str(path.relative_to(root))
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if _is_byproduct(relative):
            continue
        digest = _hash_file(path)
        if digest is not None:
            manifest[relative] = digest
    return manifest


@dataclass(frozen=True)
class TaskTree:
    """A materialised task tree, and the content it held when it was built.

    ``manifest`` is a sha256 of every file at build time. Everything the
    harness later wants to know — what the agent changed, whether the grader is
    still the grader — is a comparison against it.

    It is content, and it is held in this process, and that is the entire
    design. Three previous versions asked git instead, and each was defeated by
    something the agent legitimately controls:

    - ``git status`` compares to ``HEAD``, and one ``git commit`` makes the
      tree read as clean;
    - ``--exclude-standard`` reads the tree's ``.gitignore``, so appending a
      filename to it hid that file from every check;
    - ``git update-index --assume-unchanged`` makes a modified file report as
      unmodified.

    None of those reach a dictionary of hashes taken before the agent started.
    """

    path: Path
    manifest: dict[str, str]

    def diff(self) -> tuple[set[str], set[str], set[str]]:
        """``(added, modified, deleted)`` against the recorded manifest."""
        current = tree_manifest(self.path)
        added = set(current) - set(self.manifest)
        deleted = set(self.manifest) - set(current)
        modified = {
            path
            for path in set(current) & set(self.manifest)
            if current[path] != self.manifest[path]
        }
        return added, modified, deleted

    def changed_paths(self) -> frozenset[str]:
        """Every path whose content differs from the starting state."""
        added, modified, deleted = self.diff()
        return frozenset(added | modified | deleted)


def _hash_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_task_tree(
    task: TaskSpec, dest: Path, *, at_head: bool = False
) -> TaskTree:
    """Materialise the task's starting state at ``dest``, isolated.

    Base tree, then the change's tests restored on top and the tests it
    *deleted* removed: the grader is the change's grader, the implementation is
    not there. The result is a **fresh git repository with one commit** — the
    agent gets working git tooling (S-201 needs a repository) with no history,
    no remotes and no objects reachable from the source. It cannot read the
    answer because the answer is not there.

    ``at_head=True`` additionally applies the change's source files, producing
    the agent's tree *with the reference answer applied*. That, and not the
    pristine head tree, is what validation must check: the two differ by every
    file the change deleted, so validating against the head tree passed tasks
    that the reference answer itself cannot solve.

    Raises :class:`GitError` rather than returning a bool: a caller that
    silently treated a construction failure as a task defect would report
    harness bugs as agent failures.
    """
    if dest.exists():
        shutil.rmtree(dest)
    repo = Path(task.repo)
    _extract(repo, task.base_sha, dest)
    if task.test_paths:
        _extract(repo, task.head_sha, dest, task.test_paths)
        missing = [p for p in task.test_paths if not (dest / p).exists()]
        if missing:
            # `git archive` honours `export-ignore` gitattributes, and
            # `/tests export-ignore` is a near-universal convention in some
            # ecosystems. It exits 0 having written nothing, so without this
            # the task ships with no grader and validation blames the task.
            raise GitError(
                "these test files were not extracted, most likely an "
                f"export-ignore gitattribute: {', '.join(missing)}"
            )
    for path in task.deleted_test_paths:
        (dest / path).unlink(missing_ok=True)
    if at_head and task.source_paths:
        _extract(repo, task.head_sha, dest, task.source_paths)

    code, _, err = _git(dest, "init", "-q")
    if code != 0:
        raise GitError(f"could not initialise task repo (init): {err}")
    # The agent gets a working repository, so it gets sensible ignore rules
    # too. Nothing the harness measures depends on them any more -- the
    # manifest does not consult git at all -- but an agent whose `git status`
    # is full of byte-code noise is being handed a worse workspace than a
    # person would have. In .git/info/exclude, not .gitignore: the tree under
    # test is not ours to edit.
    exclude = dest / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("\n".join(BUILD_BYPRODUCTS) + "\n", encoding="utf-8")

    for cmd in (
        ["config", "user.name", "harness-eval"],
        ["config", "user.email", "harness-eval@localhost"],
        ["add", "-A"],
        # --allow-empty so an empty starting tree still produces a HEAD, which
        # the agent's own git commands need.
        ["commit", "-q", "--allow-empty", "-m", "task starting state"],
    ):
        code, _, err = _git(dest, *cmd)
        if code != 0:
            raise GitError(f"could not initialise task repo ({cmd[0]}): {err}")

    return TaskTree(path=dest, manifest=tree_manifest(dest))


#: Files that decide what the grader collects and how it runs, none of which
#: are test files. An agent that adds **or edits** one has changed the grader's
#: behaviour without modifying a single test.
#:
#: A blocklist, and honest about being one. Two entries were demonstrated as
#: scored passes against earlier versions: a root ``conftest.py`` is imported
#: before collection and can monkeypatch the module under test, and
#: ``addopts = "--collect-only"`` in ``pyproject.toml`` makes pytest exit 0
#: having run nothing.
GRADER_CONFIG_NAMES = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "pyproject.toml",
        "setup.py",
        "sitecustomize.py",
        "usercustomize.py",
        "Makefile",
        "makefile",
        "GNUmakefile",
        "package.json",
        "jest.config.js",
        "jest.config.ts",
        "jest.config.mjs",
        "vitest.config.js",
        "vitest.config.ts",
        ".rspec",
        "Rakefile",
        "phpunit.xml",
        "phpunit.xml.dist",
        "build.gradle",
        "build.gradle.kts",
        "pom.xml",
    }
)


def _is_grader_config(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in GRADER_CONFIG_NAMES or name.endswith(".pth")


def grader_was_changed(tree: TaskTree, task: TaskSpec) -> list[str]:
    """What the agent did to the grader. Non-empty means the run does not count.

    One question asked of the manifest — *what content differs from the
    starting state?* — and then a classification of the answer. The previous
    version asked several different questions of git, and each one was routed
    around separately:

    - it hashed only ``task.test_paths``, so **deleting a pre-existing test**
      that the change never touched was invisible, even though the grading
      command runs the whole suite;
    - it applied :data:`GRADER_CONFIG_NAMES` only to files the agent *added*,
      so **editing an existing** ``conftest.py`` or ``Makefile`` — the common
      case, since most repositories already have one — was unguarded;
    - it asked git which files were new, so a line appended to ``.gitignore``
      hid them.

    Now: any change to a test file is tampering, whether the change under test
    introduced that test or it was already there, and whether the agent added,
    edited or deleted it. Any change to a grader-config file is tampering
    unless the reference commit touched that file too — otherwise a task whose
    work *is* editing ``pyproject.toml`` would be unpassable by construction.

    What this still cannot see: an edit to an ordinary source file that the
    grader imports, made so the test passes by a side effect rather than by
    the intended change. That is indistinguishable from a legitimate fix
    without understanding the code, and nothing here attempts it.
    """
    added, modified, deleted = tree.diff()
    reference = set(task.source_paths) | set(task.test_paths)
    changed: set[str] = set()
    for path in added | modified | deleted:
        if path in task.test_paths or _is_test(path):
            changed.add(path)
        elif _is_grader_config(path) and path not in reference:
            changed.add(path)
    return sorted(changed)


#: Output signatures meaning the command could not run at all.
#:
#: Applied **only to the head state**, where the tests pass by construction, so
#: any failure there really is environmental. Applying it to the base state
#: deleted every task whose base failure mentioned a missing file — which is
#: precisely what a pull request that *adds* a file looks like, and the most
#: valuable class of task in the suite.
_ENVIRONMENT_BROKEN = (
    "no such file or directory",
    "command not found",
    "no module named pytest",
    "modulenotfounderror: no module named 'pytest'",
    "permission denied",
    "cannot execute",
)


def _looks_environmental(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _ENVIRONMENT_BROKEN)


async def validate(
    task: TaskSpec,
    test_command: str,
    workdir: Path,
    run_command: "RunCommand",
    timeout: float = 300,
) -> Validation:
    """Check the task is solvable and not already solved.

    Runs ``test_command`` at the constructed starting state, where it must
    **fail**, and at that same state with the change's source files applied,
    where it must **pass**.

    "That same state" is one correction. An earlier version compared against
    the pristine head tree, which differs from the agent's tree by every file
    the change deleted — so a change that renamed a test validated as usable
    while being unsolvable *by the reference answer itself*, and the only route
    to a pass was deleting a test file.

    ``run_command`` is the other, and it has no default on purpose. Validation
    used to run on the host while trials were graded inside a container, so a
    suite whose ``test_command`` names a host interpreter validated perfectly
    and then failed every trial — a correct solution scored as a model failure,
    and that was the *default* outcome on any machine with a Docker daemon. The
    caller must say where commands run, and pass the same runner it will grade
    with.
    """
    if not task.well_formed:
        missing = (
            "no tests" if not task.test_paths
            else "no source changes" if not task.source_paths
            else "no title"
        )
        return Validation(task.task_id, reason=missing)

    unique = uuid.uuid4().hex[:8]
    base_dir = workdir / f"{task.task_id}-base-{unique}"
    head_dir = workdir / f"{task.task_id}-head-{unique}"
    try:
        try:
            build_task_tree(task, base_dir)
            build_task_tree(task, head_dir, at_head=True)
        except GitError as exc:
            return Validation(task.task_id, reason=f"could not build task tree: {exc}")

        base = await run_command(test_command, base_dir, timeout)
        head = await run_command(test_command, head_dir, timeout)
    finally:
        # Guaranteed on every path, including construction failure.
        for path in (base_dir, head_dir):
            shutil.rmtree(path, ignore_errors=True)

    if not head.ran:
        return Validation(
            task.task_id, environment_broken=True,
            reason="the test command did not run to completion in the task tree",
            detail=head.output[-400:],
        )
    if _looks_environmental(head.output) and head.exit_code != 0:
        return Validation(
            task.task_id, environment_broken=True,
            reason=(
                "the test command could not run in the task tree (it contains "
                "only committed files: no virtualenv, no installed "
                "dependencies). Use absolute interpreter paths, an image that "
                "has them, or a setup step."
            ),
            detail=head.output[-400:],
        )

    fails_at_base = not base.passed
    passes_at_head = head.passed
    reason = None
    if not fails_at_base:
        reason = "tests already pass at base (task is trivial)"
    elif not passes_at_head:
        reason = (
            "tests fail with the reference answer applied (task is not "
            "solvable as generated)"
        )
    return Validation(
        task.task_id,
        fails_at_base=fails_at_base,
        passes_at_head=passes_at_head,
        reason=reason,
        detail=(head.output if not passes_at_head else base.output)[-400:],
    )
