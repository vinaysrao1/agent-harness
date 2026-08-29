"""Running a PR-replay suite against a live agent (S-401).

This is the part that makes the metrics real. Every field of
:class:`~harness.eval.metrics.TaskOutcome` is produced here, and the module's
job is to keep its four sources honest:

- **The graders.** ``passed`` is the task's own test command, run in the tree
  the agent left behind -- and forced to False if the agent touched the tests.
- **The work tree.** ``files_touched`` is what the tree actually differs by,
  not what the agent said it did.
- **The substrate.** ``turns_to_first_edit`` comes from S-201's per-turn
  checkpoints rather than from tool calls, because an agent that edits through
  a bash heredoc never calls ``edit_file`` and a tool-call-derived metric
  would score it as never having edited anything. When those checkpoints are
  unreadable -- see :func:`_first_edit_turn` -- the field is reported as *not
  measured* rather than as zero.
- **The agent result.** ``tokens``, ``turns``, ``refused``, ``errored``.

``regressions`` is measured only when the suite supplies a regression command
*and* that command was green in the starting tree. A task whose wider suite was
already red cannot attribute a later failure to the agent, so it reports
``None`` -- not zero.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness.config import HarnessConfig
from harness.eval.metrics import SuiteReport, TaskOutcome
from harness.eval.grading import Grader
from harness.eval.pr_replay import (
    TaskSpec,
    TaskTree,
    build_task_tree,
    grader_was_changed,
    validate,
)
from harness.eval.suite import Suite, render_command
from harness.loop import Budgets
from harness.orchestrator import Orchestrator
from harness.persistence import RunStore
from harness.profiles import CODING_REPO
from harness.repo import (
    BASELINE_EVENT,
    CHECKPOINT_EVENT,
    CHECKPOINT_SKIPPED_EVENT,
    shadow_root,
)

__all__ = ["TrialSettings", "run_trial", "run_suite"]


@dataclass(frozen=True)
class TrialSettings:
    """Everything a trial needs that is not the task itself."""

    model: str
    #: Wall clock for the agent, in seconds. Also the budget the loop's
    #: wind-down and landing logic reason about.
    wall_clock_seconds: float = 900.0
    max_turns: int = 60
    #: Timeout for one invocation of a grading command.
    grade_timeout: float = 300.0
    #: Whether to hand the agent the change's commit body as well as its
    #: subject. Off by default: on a squash-merged pull request the body
    #: routinely names the functions to add.
    include_commit_body: bool = False
    #: Keep the task tree after the trial (for inspecting a failure).
    keep_tree: bool = False


def _git(repo: Path, *args: str, strip: bool = True) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)
    out = proc.stdout or ""
    return proc.returncode, out.strip() if strip else out, (proc.stderr or "").strip()


def _total_tokens(usage) -> int:
    """Every token the trial actually cost.

    ``Usage.input_tokens`` counts *uncached* input only -- cache reads and
    writes are tracked in their own fields. Summing input+output therefore
    omits the entire cached prompt, which on a cache-enabled model is most of
    the input: the metric would make prompt caching look like a free
    efficiency win when in fact it is the measurement losing sight of the
    tokens. ``reasoning_tokens`` is deliberately not added -- it is a subset of
    ``output_tokens``, not a sibling bucket.
    """
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens
    )


def _files_touched(tree: TaskTree) -> frozenset[str]:
    """Repo-relative paths whose content differs from the starting state.

    A content comparison against the manifest recorded at build time — no git
    at all. Three git-based versions were each defeated by state the agent
    legitimately controls: ``git status`` compares to ``HEAD`` (one commit and
    the tree reads clean), ``--exclude-standard`` reads the tree's own
    ``.gitignore`` (append a filename and it disappears), and
    ``git update-index --assume-unchanged`` makes a modified file report as
    unmodified. None of those reach a dictionary of hashes.
    """
    return tree.changed_paths()


def _first_edit_turn(store: RunStore, agent_id: str) -> tuple[int | None, bool]:
    """``(turn, measured)`` for the agent's first change to the work tree.

    Read from S-201's checkpoint **events**, not from the agent's tool calls
    and not from the shadow refs.

    Not from tool calls, because an agent that edits through a bash heredoc
    changes the tree without ever calling ``edit_file``; a tool-call-derived
    metric scores it as never having edited anything, and "the agent went eight
    turns without touching the code" is precisely the finding this metric
    exists to surface, so a false positive on it is worse than no metric.

    Not from the shadow refs, because the shadow store lives inside the
    sandbox: under :class:`~harness.sandbox.docker.DockerSandbox` -- the
    backend one would actually choose for running model-authored code -- it is
    destroyed with the container. A metric that only works when isolation is
    switched off is a metric that never fires in practice. Each checkpoint
    event carries the tree hash it computed, so the comparison is between two
    strings recorded at the time and survives teardown on every backend.

    ``measured`` is False whenever the answer would be a guess. Those cases are
    kept apart from the one real negative:

    - no ``repo_baseline`` event: the capability never activated;
    - a baseline with no tree: activation succeeded but its snapshot was
      skipped, so there is nothing to compare against;
    - a checkpoint with no tree hash: written by an older loop;
    - no edit found, but some turn's checkpoint was skipped: the edit may have
      happened in that gap, and "the agent never touched anything" would be a
      stronger claim than the evidence supports.

    The real negative -- activated, nothing skipped, every checkpointed tree
    identical to the baseline -- returns ``(None, True)``. So does a run whose
    agent made no tool calls at all: no tool call can change the tree, so the
    absence of checkpoints is itself the observation, provided the baseline
    event proves the substrate was watching.
    """
    events = store.load_events(agent_id)
    baseline_event = next((e for e in events if e.kind == BASELINE_EVENT), None)
    if baseline_event is None:
        return None, False
    baseline = baseline_event.payload.get("tree")
    if baseline is None:
        return None, False

    skipped = any(e.kind == CHECKPOINT_SKIPPED_EVENT for e in events)
    for event in (e for e in events if e.kind == CHECKPOINT_EVENT):
        tree = event.payload.get("tree")
        if tree is None:
            return None, False
        if tree != baseline:
            return int(event.payload["turn"]), True

    return None, not skipped


def _cleanup_shadow(run_id: str) -> None:
    """Remove this run's shadow store. A suite of 60 trials would otherwise
    leave 60 object databases in ``/tmp`` for nobody."""
    shutil.rmtree(Path(shadow_root()) / run_id, ignore_errors=True)


async def run_trial(
    task: TaskSpec,
    suite: Suite,
    settings: TrialSettings,
    *,
    config: HarnessConfig,
    store: RunStore,
    workdir: Path,
    trial: int = 0,
) -> TaskOutcome:
    """Run one agent attempt at one task and grade it.

    The task tree is built fresh under a per-trial unique directory, so two
    trials of the same task -- concurrent or sequential -- never share state.
    It is removed afterwards unless ``settings.keep_tree``.
    """
    dest = workdir / f"{task.task_id}-t{trial}-{uuid.uuid4().hex[:8]}"
    run_id: str | None = None
    try:
        tree = build_task_tree(task, dest)

        # Regression baseline, before the agent touches anything. A suite that
        # is already red here cannot attribute a later failure to the agent.
        baseline_green: bool | None = None
        if suite.regression_command:
            async with Grader(dest, config) as grader:
                baseline = await grader.run(
                    suite.regression_command, settings.grade_timeout
                )
            # `ran is False` means we never learned anything, so a later
            # failure cannot be attributed to the agent either.
            baseline_green = baseline.passed if baseline.ran else None

        orchestrator = Orchestrator(config, store)
        run_id, result = await orchestrator.run_task(
            task.prompt(include_body=settings.include_commit_body),
            settings.model,
            workspace=dest,
            profile=CODING_REPO,
            budgets=Budgets(
                max_turns=settings.max_turns,
                wall_clock_seconds=settings.wall_clock_seconds,
            ),
        )
        agent_id = next(
            agent.id
            for agent in store.list_agents(run_id)
            if agent.parent_agent_id is None
        )

        # Measured before grading, because a test run writes byte-code caches
        # and coverage files into the tree and those are the agent's changes
        # only in the sense that it was standing nearby. (No cache wipe here:
        # `Grader.run` wipes before every command, which covers it. A second
        # call would be a defence nothing exercises.)
        touched = _files_touched(tree)
        tampered = bool(grader_was_changed(tree, task))

        regressions: int | None = None
        grading_failed = False
        async with Grader(dest, config) as grader:
            graded = await grader.run(
                render_command(suite.test_command, task.test_paths),
                settings.grade_timeout,
            )
            # A tampered trial is never a pass, whatever the grader says: the
            # grader is one of the files the agent could have rewritten. A
            # trial whose grader never ran is not a fail either -- it is an
            # error, and scoring it as a model failure would blame the agent
            # for a dead Docker daemon.
            grading_failed = not graded.ran
            passed = graded.passed and not tampered
            if suite.regression_command and baseline_green:
                after = await grader.run(
                    suite.regression_command, settings.grade_timeout
                )
                # Only a command that ran can report a regression. Counting a
                # timeout as one invents a finding and attributes it to the
                # agent.
                regressions = (0 if after.passed else 1) if after.ran else None

        first_edit, measured = _first_edit_turn(store, agent_id)
        return TaskOutcome(
            task_id=task.task_id,
            passed=passed,
            tokens=_total_tokens(result.usage),
            turns=result.turns,
            turns_to_first_edit=first_edit,
            first_edit_measured=measured,
            files_touched=touched,
            # Deletions included: the change removed those files, so an agent
            # that reproduces the change removes them too. Excluding them
            # charged the correct behaviour as an unrequested edit, and left
            # dead code scoring better than removing it.
            reference_files=frozenset(task.source_paths)
            | frozenset(task.deleted_source_paths),
            regressions=regressions,
            tampered=tampered,
            refused=result.refused,
            errored=result.status == "error" or grading_failed,
            budget_paused=result.status == "paused_budget",
        )
    finally:
        # Both or neither: keeping the tree to inspect a failure and deleting
        # the per-turn history that explains it would be half a favour.
        if not settings.keep_tree:
            if run_id is not None:
                _cleanup_shadow(run_id)
            shutil.rmtree(dest, ignore_errors=True)


async def run_suite(
    tasks: list[TaskSpec],
    suite: Suite,
    settings: TrialSettings,
    *,
    config: HarnessConfig,
    store: RunStore,
    workdir: Path,
    trials: int = 1,
    on_outcome=None,
) -> SuiteReport:
    """Run every task ``trials`` times and aggregate.

    Sequential on purpose. These trials each start a sandbox, run a real model
    and execute a test suite; running them concurrently would make wall-clock
    measurements meaningless and put several agents on the same machine's CPU
    while one of them is being timed.
    """
    report = SuiteReport()
    for trial in range(trials):
        for task in tasks:
            try:
                outcome = await run_trial(
                    task, suite, settings,
                    config=config, store=store, workdir=workdir, trial=trial,
                )
            except Exception:  # noqa: BLE001
                # A suite is hours of model calls. Losing all of it because
                # trial 47 hit a transport error would be the wrong trade --
                # and silently dropping that trial would be worse still, so it
                # is recorded as an error and counted in the denominator.
                outcome = TaskOutcome(task_id=task.task_id, passed=False, errored=True)
            report.outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome)
    return report
