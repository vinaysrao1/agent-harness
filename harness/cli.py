"""Command-line entry point (DESIGN.md §2 non-goals: "v1 is a CLI").

Subcommands:

- ``harness run GOAL``    — run one task end-to-end and print the result.
- ``harness runs``        — table of every persisted run.
- ``harness cost RUN_ID`` — aggregated token usage for one run.
- ``harness resume RUN_ID`` — continue an interrupted run (v1 limits below).
- ``harness eval SUITE`` — run a PR-replay suite and print its report.

The approval prompt (gated mode) renders the exact tool name and arguments
and reads one of ``y`` (allow once), ``n`` (deny), or ``a`` (allow and grant
this tool for the rest of the run, via
:meth:`harness.orchestrator.Orchestrator.grant` /
:meth:`harness.permissions.Policy.with_grant`).

Error contract: config load errors, unknown model names, and unknown run
ids print a clean one-line message to stderr and exit ``2`` — never a
traceback. (Exit ``2`` is also what argparse itself uses for usage errors.)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from harness.config import ConfigError, PermissionMode, load_config
from harness.eval.pr_replay import GitError
from harness.eval.suite import SuiteError
from harness.loop import AskCallable
from harness.orchestrator import Orchestrator, UnknownModelError, UnknownRunError
from harness.permissions import ToolMeta
from harness.persistence import RunStore

__all__ = ["main", "make_ask"]

#: v1 resume limitations, shown in ``harness resume --help`` (honesty per
#: the module spec: state exactly what resume does and does not restore).
_RESUME_EPILOG = """\
v1 limitations (stated honestly):
  - only the lead agent's transcript is reconstructed (from its persisted
    events, with compacted spans replayed as their summaries); subagents
    are not resumed — the lead may spawn new ones
  - budgets restart from the defaults minus the turns/tokens the lead
    agent itself already consumed
  - a custom --workspace from the original run is not remembered; the
    default run workspace <home>/runs/<RUN_ID>/workspace is used
  - 'a' (always-for-this-run) approval grants from the original session
    are persisted with the run and restored on resume
"""


def make_ask(orchestrator: Orchestrator) -> AskCallable:
    """Build the terminal approval callback for one run.

    Renders the tool name and arguments, then reads a choice from stdin:
    ``y`` allows once, ``n`` denies, ``a`` allows *and* grants the tool's
    name as an allow pattern for the rest of the run via
    :meth:`Orchestrator.grant`. Anything else re-prompts. EOF on stdin
    denies (the safe default for a non-interactive session).

    The blocking ``input()`` runs off-loop via :func:`asyncio.to_thread` —
    an unanswered prompt (which can sit for hours in a long-horizon run)
    must never freeze the event loop, or every concurrently running
    subagent, HTTP response, and timeout would stall with it (§4.12: ASKs
    from all agents share one approval queue). An :class:`asyncio.Lock`
    serializes concurrent asks so prompts from different agents do not
    interleave on the terminal.
    """

    prompt_lock = asyncio.Lock()

    async def ask(tool_name: str, arguments: dict, meta: ToolMeta) -> bool:
        async with prompt_lock:
            print(f"\napproval required: {tool_name} {arguments!r}")
            while True:
                try:
                    choice = await asyncio.to_thread(
                        input, "allow? [y]es once / [n]o / [a]lways for this run: "
                    )
                except EOFError:
                    print("(stdin closed; denying)")
                    return False
                choice = choice.strip().lower()
                if choice == "y":
                    return True
                if choice == "n":
                    return False
                if choice == "a":
                    orchestrator.grant(tool_name)
                    return True
                print(f"unrecognized choice {choice!r}; expected y, n, or a")

    return ask


def _print_result(run_id: str, result_status: str, final_text: str | None) -> None:
    """Print the standard run/resume footer: final text, id, status."""
    print(final_text or "(no final text)")
    print(f"\nrun id: {run_id}")
    print(f"status: {result_status}")


def _print_usage_summary(store: RunStore, run_id: str) -> None:
    """Print one aggregated-usage line for ``run_id``."""
    totals = store.total_usage(run_id)
    print(
        "usage: "
        f"input={totals['input_tokens']} "
        f"output={totals['output_tokens']} "
        f"cache_read={totals['cache_read_tokens']} "
        f"cache_write={totals['cache_write_tokens']}"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """``harness run``: execute one task and print its outcome."""
    config = load_config(args.config)
    mode = PermissionMode(args.mode) if args.mode else None
    workspace = Path(args.workspace).expanduser() if args.workspace else None
    with RunStore(config.home / "state.db") as store:
        orchestrator = Orchestrator(config, store)
        run_id, result = asyncio.run(
            orchestrator.run_task(
                args.goal,
                args.model,
                mode=mode,
                workspace=workspace,
                ask=make_ask(orchestrator),
            )
        )
        _print_result(run_id, result.status, result.final_text)
        _print_usage_summary(store, run_id)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    """``harness runs``: print a table of every persisted run."""
    config = load_config(args.config)
    with RunStore(config.home / "state.db") as store:
        runs = store.list_runs()
        if not runs:
            print("no runs recorded")
            return 0
        header = f"{'RUN ID':<34} {'CREATED':<27} {'STATUS':<14} {'MODEL':<12} {'MODE':<6} GOAL"
        print(header)
        for run in runs:
            goal = run.goal if len(run.goal) <= 50 else run.goal[:47] + "..."
            print(
                f"{run.id:<34} {run.created_at:<27} {run.status:<14} "
                f"{run.model:<12} {run.permission_mode:<6} {goal}"
            )
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    """``harness cost``: aggregated token usage for one run."""
    config = load_config(args.config)
    with RunStore(config.home / "state.db") as store:
        run = store.get_run(args.run_id)
        if run is None:
            raise UnknownRunError(f"no such run: {args.run_id!r}")
        print(f"run {run.id} ({run.status}): {run.goal}")
        per_model: dict[str, dict[str, int]] = {}
        for record in store.list_usage(args.run_id):
            bucket = per_model.setdefault(
                record.model,
                {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            )
            bucket["calls"] += 1
            bucket["input"] += record.usage.input_tokens
            bucket["output"] += record.usage.output_tokens
            bucket["cache_read"] += record.usage.cache_read_tokens
            bucket["cache_write"] += record.usage.cache_write_tokens
        for model, bucket in sorted(per_model.items()):
            print(
                f"  {model}: {bucket['calls']} call(s), "
                f"input={bucket['input']} output={bucket['output']} "
                f"cache_read={bucket['cache_read']} "
                f"cache_write={bucket['cache_write']}"
            )
        _print_usage_summary(store, args.run_id)
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """``harness resume``: continue an interrupted run (v1 limits in --help)."""
    config = load_config(args.config)
    with RunStore(config.home / "state.db") as store:
        orchestrator = Orchestrator(config, store)
        result = asyncio.run(
            orchestrator.resume_task(args.run_id, ask=make_ask(orchestrator))
        )
        _print_result(args.run_id, result.status, result.final_text)
        _print_usage_summary(store, args.run_id)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """``harness eval``: run a PR-replay suite (S-401) and print its report.

    Every task that is dropped is reported with its reason. A benchmark that
    quietly shrinks is worse than one that fails loudly: a suite whose task
    count fell from 40 to 6 still prints a pass rate, and it looks fine.
    """
    import asyncio as _asyncio

    from harness.eval.grading import sandboxed_runner
    from harness.eval.pr_replay import generate, validate
    from harness.eval.runner import TrialSettings, run_suite
    from harness.eval.suite import Suite, is_heldout

    config = load_config(args.config)
    suite = Suite.load(Path(args.suite).expanduser())
    # The suite name comes out of a JSON file, so it is data, not a path
    # component: "../.." in it would place task trees anywhere on the disk.
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", suite.name).strip("-.") or "suite"
    workdir = Path(args.workdir).expanduser() if args.workdir else (
        config.home / "eval" / safe_name
    )
    workdir.mkdir(parents=True, exist_ok=True)

    tasks = generate(Path(suite.repo).expanduser(), list(suite.revs))
    print(f"suite {suite.name}: {len(tasks)} task(s) from {len(suite.revs)} revision(s)")
    if len(tasks) < len(suite.revs):
        print(
            f"  {len(suite.revs) - len(tasks)} revision(s) produced no task "
            "(unresolvable, or a root commit with no parent to diff against)"
        )

    if args.split != "all":
        want_heldout = args.split == "heldout"
        kept = [t for t in tasks if is_heldout(t.task_id) == want_heldout]
        print(f"  {args.split} split: {len(kept)} of {len(tasks)}")
        tasks = kept

    # Before validation, not after: validating a task costs two full runs of
    # the test command, and `--limit 3` on a 40-revision suite was paying for
    # all 40.
    if args.limit and len(tasks) > args.limit:
        print(f"  limited to the first {args.limit} of {len(tasks)} tasks")
        tasks = tasks[: args.limit]

    if args.validate:
        # Validated through the same sandbox that will grade, so "fails at
        # base, passes with the reference answer" is established where the
        # verdict will actually be reached. Validating on the host while
        # grading in a container made a correct solution score as a model
        # failure -- and that was the default on any machine with a daemon.
        async def _validate_all():
            run_command = sandboxed_runner(config)
            usable, rejected = [], []
            for task in tasks:
                outcome = await validate(
                    task, suite.test_command, workdir,
                    run_command, args.grade_timeout,
                )
                (usable if outcome.usable else rejected).append((task, outcome))
            return usable, rejected

        usable, rejected = _asyncio.run(_validate_all())
        for task, outcome in rejected:
            print(f"  dropped {task.task_id}: {outcome.reason}")
        print(f"  validated: {len(usable)} usable, {len(rejected)} dropped")
        tasks = [task for task, _ in usable]
    else:
        print(
            "  validation skipped: tasks were not checked for failing at the "
            "base and passing at the head, so the number below is not a "
            "benchmark result"
        )

    if not tasks:
        print("no usable tasks; nothing to run", file=sys.stderr)
        return 2

    settings = TrialSettings(
        model=args.model,
        wall_clock_seconds=args.wall_clock,
        max_turns=args.max_turns,
        grade_timeout=args.grade_timeout,
        keep_tree=args.keep_trees,
    )
    with RunStore(config.home / "state.db") as store:
        report = _asyncio.run(
            run_suite(
                tasks, suite, settings,
                config=config, store=store, workdir=workdir,
                trials=args.trials,
                on_outcome=lambda o: print(
                    f"  {o.task_id}: {'PASS' if o.passed else 'fail'}"
                    + ("  [TAMPERED]" if o.tampered else "")
                ),
            )
        )
    print()
    print(report.render())
    if not args.validate:
        print("(unvalidated suite - not a benchmark result)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Personal model-agnostic agentic harness (see DESIGN.md).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--config",
            default=None,
            help="Path to config.toml (default: $HARNESS_HOME/config.toml).",
        )

    run = subparsers.add_parser("run", help="Run one task end-to-end.")
    run.add_argument("goal", help="The task the agent should pursue.")
    run.add_argument(
        "--model",
        required=True,
        help="Model name from the config registry ([models.NAME]).",
    )
    run.add_argument(
        "--mode",
        choices=[m.value for m in PermissionMode],
        default=None,
        help="Permission mode for this run (default: config's setting).",
    )
    run.add_argument(
        "--workspace",
        default=None,
        help="Sandbox workspace directory (default: <home>/runs/<run_id>/workspace).",
    )
    add_config(run)
    run.set_defaults(func=_cmd_run)

    runs = subparsers.add_parser("runs", help="List every persisted run.")
    add_config(runs)
    runs.set_defaults(func=_cmd_runs)

    cost = subparsers.add_parser(
        "cost", help="Aggregated token usage for one run."
    )
    cost.add_argument("run_id", help="The run id (see `harness runs`).")
    add_config(cost)
    cost.set_defaults(func=_cmd_cost)

    resume = subparsers.add_parser(
        "resume",
        help="Continue an interrupted run (lead agent only in v1).",
        epilog=_RESUME_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    resume.add_argument("run_id", help="The run id to resume.")
    add_config(resume)
    resume.set_defaults(func=_cmd_resume)


    ev = subparsers.add_parser(
        "eval", help="Run a PR-replay suite (S-401) and print its report."
    )
    ev.add_argument("suite", help="Path to the suite JSON file.")
    ev.add_argument(
        "--model", required=True, help="Model name from the config registry."
    )
    ev.add_argument(
        "--split",
        choices=["dev", "heldout", "all"],
        default="dev",
        help=(
            "Which half of the hash split to run (default: dev). Score a "
            "change you tuned on 'heldout', once."
        ),
    )
    ev.add_argument("--trials", type=int, default=1, help="Attempts per task.")
    ev.add_argument(
        "--limit", type=int, default=None, help="Run at most this many tasks."
    )
    ev.add_argument(
        "--wall-clock", type=float, default=900.0,
        help="Seconds each attempt gets (default: 900).",
    )
    ev.add_argument("--max-turns", type=int, default=60)
    ev.add_argument(
        "--grade-timeout", type=float, default=300.0,
        help=(
            "Seconds one grading command gets (default: 300). Raise it for a "
            "suite whose own tests take longer than that, or every task fails "
            "on a timeout that has nothing to do with the agent."
        ),
    )
    ev.add_argument("--workdir", default=None, help="Where task trees are built.")
    ev.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help=(
            "Skip the fails-at-base / passes-at-head check. Faster, and the "
            "resulting number is not a benchmark result."
        ),
    )
    ev.add_argument(
        "--keep-trees",
        action="store_true",
        help="Leave each task tree on disk for inspection.",
    )
    add_config(ev)
    ev.set_defaults(func=_cmd_eval, validate=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Known operational errors (bad config, unknown model, unknown run) are
    printed as one clean line on stderr with exit code 2 — never a
    traceback.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        UnknownModelError,
        UnknownRunError,
        SuiteError,
        GitError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
