---
id: S-201
title: Git substrate
status: Implemented
lane: A
depends: S-005
effort: M
---

# S-201 — Git substrate

## Contract
Record `repo`, `base_sha` and `branch` on the run. Write per-turn shadow
checkpoints to `refs/harness/<run_id>/turn-<n>` using a git directory outside
the workspace.

## Invariants
Lane A by construction: gated on `profile.enables("git_substrate") and
environment.affirms("git_substrate")`, and `CODING` enables nothing. N3 and N4
are the ones at risk — an inactive substrate must issue no commands and write
nothing — and both are asserted directly.

## Acceptance
1. No `.git` inside the workspace. ✅ — every write command is asserted to
   target `/tmp/.harness-git`, and `git init` is only ever `--bare` there.
2. Absent `git`, or `sandbox_owned=False`, the capability is inert. ✅
3. Checkpoint cost is bounded and skipped under deadline pressure. ✅ — but
   only after correction. The first version checked affordability **once**
   against a single command's timeout and then issued **four** commands each
   allowed that timeout: a 4× under-count, so "bounded" named a budget the
   code did not honor. Each command now gets a quarter-slice, and a test
   asserts `4 × slice ≤ total`. The skip half is real and fires against the
   live `Deadline` API, with a control test so the skip tests cannot pass by
   checkpointing never working at all.
4. Refuse to start on a dirty tree unless `allow_dirty`. ✅

## Telemetry
`repo_checkpoint` and `repo_checkpoint_skipped`, both registered in
`harness.specs.EVENT_KIND_SPECS` — the first kinds to go through T3.

Registering them exposed an asymmetry in T3 itself: `unowned_event_kinds`
caught emitted-but-unregistered, and `stale_legacy_kinds` guarded the reverse
for the legacy set, but nothing caught **registered-but-never-emitted**. A spec
could declare telemetry it never produced and T3 would stay green forever.
`unemitted_spec_kinds` closes it, and it belongs here because this is the spec
that first used the mechanism.

## Rollback
`git revert`. `AgentLoop(repo=...)` defaults to `None`, so removal is inert.

## Neutrality argument
`AgentLoop.repo` defaults to `None` and the loop guards on it, so an existing
caller gains nothing. When a substrate *is* passed but inactive, `checkpoint()`
returns before issuing any command — asserted over 50 turns, because "it is a
no-op" is exactly the sort of claim that rots. Conformance stayed at 48 passed
with the wiring in place; no golden moved.

## Why the git directory is outside the workspace
A `.git` inside the workspace is a directory the agent can see and be confused
by, and one the grader never expected. Worse, a task whose goal involves git
would find a repository the *harness* created rather than the task. So the
object store lives under `/tmp/.harness-git` — already inside N4's permitted
prefix, so it is covered by construction rather than by exception — and the
workspace is passed as `--work-tree`.

## Why failure is never fatal
Every failure path disables checkpointing rather than raising: a failed shadow
init, a failed `write-tree`, an unknown HEAD, even a sandbox that raises. A
missing history is a lost convenience; a failed run is a lost task. The one
deliberate exception is the dirty work tree, which raises — because a diff that
mixes the agent's work with changes that were already there attributes
someone else's edit to the agent, and that is worse than not starting.

## Defects review found in this spec
- **`commit-tree` had no identity, so the feature was dead in the only
  environment it can activate in.** `commit-tree` resolves the committer under
  `IDENT_STRICT`; a container with no `user.email` and a domainless hostname
  has auto-detection refused. Every checkpoint would have failed — and failed
  *silently*, because a non-zero exit is counted as a skip. It passed locally
  only because a global `user.email` exists, and `sandbox_owned=True` means a
  harness-owned container, which is exactly where it dies.
- **The loop wiring was untested, and the test claiming to cover it did not.**
  `test_S201_loop_checkpoints_each_turn_when_active` imported `FakeAdapter`,
  `Orchestrator`, `AgentLoop` and six types, used none of them, and duplicated
  another test's body. Deleting the loop's entire checkpoint block left all
  2019 tests green. The conformance suite could not see it either, since
  `build_loop` never passes `repo=`. Replaced with tests that drive `run_task`.
- **There was no baseline to diff against.** The shadow store shares no objects
  with the workspace repo, so the recorded `base_sha` is unresolvable there —
  and the first checkpoint is taken *after* turn 1's edits land. `start()` now
  writes a `baseline` ref capturing the pre-agent tree.
- **A skip was claimed as "recorded" when it was an in-memory counter.**
  `checkpoint_detailed` now returns the reason and the loop emits it.
- **No test could tell a valid git command from an invalid one.** Removing
  `commit-tree`'s tree argument left the suite green, because the fake sandbox
  matches by substring and returns exit 0 for anything.
- `--work-tree=.` depends on the sandbox exec'ing with the workspace as its
  cwd, which is true of every implementation but stated nowhere. Extracted to
  `work_tree_argument()` so a test pins it.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **An unborn HEAD is a silent dead end.** A fresh `git init` with no commits
  makes `rev-parse HEAD` fail, so `RepoState.usable` is False and an *active*
  substrate never checkpoints for the whole run — plausible for a repo task,
  and invisible except as a missing history.
- **Turn numbering has holes.** The checkpoint sits inside the tool-call
  branch, so a turn with no tool calls writes no ref. S-202's `rewind <turn>`
  will meet missing refs.
- **Subagent concurrency is a hazard at the seam S-202 will use.**
  `build_loop` is shared by the lead and every subagent, and subagents run
  concurrently. One substrate passed there means concurrent `git add -A`
  against a single `$GIT_DIR/index` — `index.lock` races, silently counted as
  skips — and ref collisions, since `run_id` is shared while turn counters are
  per-loop.
- **Nothing constructs a `GitSubstrate` yet.** `AgentLoop` accepts and calls
  one, but the orchestrator never builds it, so no run has a substrate. Wiring
  that requires calling the S-005 probe during a run, which is a behavior
  change for non-`CODING` profiles and belongs with S-202 (`harness diff`),
  the first spec that needs the checkpoints to exist. The consequence is real:
  the substrate has never run against a live git repository, only against a
  scripted sandbox.
- `RepoState` is returned but not persisted to the run row, so the contract's
  "record on the run" is half-met — the state is available to the caller and
  written nowhere. S-202 owns the run-row column, since it is the spec that
  reads it back.
- **No cleanup.** One bare repo per `run_id` under `/tmp/.harness-git`, holding
  full-tree snapshots, never removed. The `SPILL_DIR` precedent is also never
  cleaned, but that stores kilobytes of text rather than object storage of the
  user's tree.
- Checkpoints are `add -A` snapshots of the whole work tree. On a large
  repository that is a real cost even inside the timeout; there is no
  path-scoping yet.
- The deadline seam is duck-typed (`getattr(deadline, "landing", ...)`) rather
  than importing the concrete `Deadline`, to avoid a circular import. A
  `Deadline` that renamed those members would silently stop skipping.
