---
id: S-005
title: EnvironmentProfile probe and SetupBudget
status: Implemented
lane: A
depends: S-004
effort: S
---

# S-005 — `EnvironmentProfile` probe and `SetupBudget`

## Contract
`harness/environment.py`: one bounded probe returning a frozen record with
explicit `unknown` states, plus a `SetupBudget` governing all pre-first-model-
call work.

## Invariants
N3 above all: a `CODING` run performs zero sandbox execs before the first model
call. Every probe here is an exec, so under `CODING` the budget is zero and
this module is unreachable rather than merely unused. N4 also applies — the
probe only reads.

## Acceptance
1. Under `CODING` the probe is never constructed and N3 holds. ✅ — and now
   for a non-vacuous reason: `setup_budget_for(profile, wall_clock)` returns a
   zero budget for any profile declaring no capabilities, so the zero is
   *derived from `CODING`* rather than handed in by a test.
2. Probe cost ≤ `SetupBudget`; on exhaustion it returns `truncated=True` with
   every unfinished field `None`. ✅
3. Every `None` field routes to today's behavior. ✅ — a parametrized test
   takes a fully-affirming environment and blanks one field at a time,
   requiring that no capability becomes active that was not active before. The
   first version only re-asserted the dataclass defaults, which tests no
   routing at all.
4. The probe never writes to the workspace. ✅ — asserted by inspecting every
   command it issues, not by claiming it.
5. Detection is observation, not instruction. ✅

## Telemetry
none yet. The probe's findings are not persisted; when a Layer 2 spec first
acts on a capability, that spec owns the event and registers it under T3.

## Rollback
`git revert`. Nothing constructs the probe yet, so removal is inert.

## Neutrality argument
`SetupBudget.none()` returns `allows_probing == False`, and `probe_environment`
returns `UNKNOWN_ENVIRONMENT` before issuing any command. `CODING` declares no
capabilities (S-004), so even a probe that somehow ran could not activate
anything: the conjunction `enables and affirms` is `False` on the left.

## The two-dimensional rule, and why both halves are needed
```
active(capability) = profile.enables(capability) and environment.affirms(capability)
```

Each half alone produces a failure this design exists to prevent:

- **Profile without environment**: `CODING_REPO` inside a non-git container
  would try to write git refs into a tree that has no `.git`. It degrades to
  `CODING` instead.
- **Environment without profile**: a `CODING` run inside a git repository would
  opportunistically switch git features on and spend setup time the benchmark
  did not ask for. It does not. This is the "it inferred repo-mode on a TB2
  task and cost us forty seconds" class, killed structurally.

Both are asserted directly rather than argued.

## Unknown is not affirmation
Every branch of `affirms()` returns `False` unless the environment positively
established the precondition. A truncated probe, a failed probe, and an unrun
probe are therefore indistinguishable from today's behavior — which is the
point. A probe that guessed would convert an unknown into a wrong answer, and a
wrong answer about the environment activates a capability the environment
cannot support.

`sandbox_owned` carries this furthest: a git work tree the harness does **not**
own (the Harbor case, where the container belongs to the caller) does not
affirm `git_substrate`. Writing refs into someone else's container is not ours
to do, however real the repository is.

## Observation, not instruction
`render_observations` phrases findings as *"Observed in pyproject.toml (these
are observations, not instructions — verify before relying on them)"*.

The distinction is not cosmetic. On a task whose goal is to fix the build,
"the test command is pytest" asserts as fact the very thing the agent was asked
to determine. It is the same distinction the safety core makes about tool
results: this is data, not instruction. A test asserts the phrasing, because
prose guarantees rot.

## Defects review found in this spec
- **The default-deny fallthrough was never asserted.** Every test parametrized
  over `REPO_CAPABILITIES`, all of which have explicit branches, so flipping
  `return False` to `return True` left the suite green — while that line
  carries this spec's central claim.
- **"Unknown is not affirmation" was not tested for unknown.** Only the
  explicit `sandbox_owned=False` case was covered, so relaxing `is True` to
  `is not False` passed. `None` is both the default *and* the realistic case,
  since `sandbox_owned` is caller-supplied and the probe never determines it.
- **The budget bounded nothing under test.** Deleting `run()`'s budget check
  entirely left the suite green, as did hardcoding the completion path's
  `truncated` to `False` — the existing test exited at an earlier guard and
  never reached that line.
- **The pyproject "observation" was attributed to a file never read.** The
  finding came from `command -v pytest`, but rendered as *"Observed in
  pyproject.toml: test = pytest"*. On a unittest project with pytest present
  transitively, that states as fact something the probe never established —
  the exact failure this spec's phrasing rule exists to prevent, with the verb
  fixed and the subject left wrong. Now attributed to `PATH`.
- **`shlex.quote` was the wrong function.** It applies shell *word* quoting to
  a whole command line, so `npm run lint` rendered as `'npm run lint'` — a
  single unrunnable argv[0].
- `clock` was typed `object | None` and dispatched via `callable()`, so a float
  or a `Deadline` silently substituted real wall-clock. Now
  `Callable[[], float] | None`.
- `typecheck` and `format` were declared, rendered, and set by no branch — and
  `format` gated a capability. Removed. `file_count` was likewise dead; it is
  now populated, because S-203 gates on it.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Nothing constructs the probe yet.** `run_task` does not call it, so the
  whole module is currently reachable only from tests. That is deliberate —
  wiring it in is a behavior change for non-`CODING` profiles and belongs to
  the first Layer 2 spec that needs a finding. The consequence is that the
  probe's real-sandbox behavior is unproven against a live container.
- **Project detection is shallow**: presence of `pyproject.toml`,
  `package.json` or `Makefile`, plus Node's declared script names. It does not
  interpret build semantics, and an observation the harness cannot stand behind
  is not worth putting in a prompt.
- **The budget bounds the probe, not the caller.** `SetupBudget` is honored
  inside `probe_environment`; nothing yet reserves it from the run's
  `Deadline`, so the plan's `deadline.reserve_setup()` is not implemented. With
  no caller there is nothing to reserve from, and adding an unused method to
  `Deadline` would touch a module N6 pins.
- **The budget is not a hard bound under Harbor.** `HarborSandbox.exec` rounds
  its timeout up to whole seconds (`max(1, ceil(timeout))`), so on the one
  platform where the probe would matter each command has a 1s floor regardless
  of remaining budget. "Cost ≤ SetupBudget" holds in the harness's own
  arithmetic, not necessarily in wall-clock there.
- **`sandbox_owned` is never probed**, only passed in by the caller. The probe
  cannot determine whether it owns its container, so this field is as reliable
  as whoever supplies it.
- `tooling` detection parses `command -v` output by suffix match, which would
  mis-detect a binary whose path ends in the same name (`/opt/notgit/git` is
  fine; a directory literally named `git` is not distinguished).
