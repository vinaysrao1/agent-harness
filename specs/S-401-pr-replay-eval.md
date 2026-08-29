---
id: S-401
title: PR-replay coding eval
status: Implemented
lane: A
depends: S-201
effort: L
---

# S-401 — PR-replay coding eval

## Contract
A second benchmark, built out of history that already exists. Take a merged
change, put the tree back to its first parent, restore **only its tests**, and
ask the agent to make them pass. The change's tests are the grader; its real
diff is the reference answer.

Introduces `harness/eval/`: `pr_replay` (generate, build, validate, tamper
check), `suite` (a suite file and the held-out split), `metrics`
(`TaskOutcome`, `SuiteReport`), and `runner` (run a trial, run a suite). One
command: `harness eval SUITE --model NAME [--split dev|heldout|all]
[--trials N] [--limit N] [--wall-clock S] [--max-turns N] [--workdir DIR]
[--no-validate] [--keep-trees]`.

Why this and not more Terminal-Bench: TB2 is ±5 points noisy at n=89, costs
hours per run, and measures terminal work rather than repository work. This
generates as many tasks as there are merged changes, runs on any repository,
and grades against tests a human already wrote and merged.

## Invariants
Lane A. `harness/eval/` has no importer inside the agent loop; nothing on the
benchmark path reaches it. Conformance stayed at 48 passed; no golden moved.

This spec **does** change the orchestrator — see *S-201 activation* below — and
that change is where the neutrality argument has to be made, not here.

## Acceptance
1. A task generated from a real merged change validates: its tests fail at the
   starting state and pass with the reference answer applied. ✅ —
   `TestValidationIsWhatMakesThisABenchmark`. Both states are built the same
   way; comparing against the pristine head tree instead passed tasks the
   reference answer itself cannot solve.
2. The reference solution is not present in, or reachable from, the task tree.
   ✅ — `TestTheAgentCannotReadTheAnswer`: single-commit history, no remotes,
   the head commit unreachable, and a recursive grep of the whole directory
   (`.git` included) for the solution's text. This is a claim about the
   **tree**, not about the agent's reach: the source repository sits on the
   same host at a path the suite file names, and `LocalSandbox` provides no
   filesystem isolation, so under that backend the answer is one
   `git -C <source> log` away. Isolation from the host is the sandbox's job and
   is a separate property from this one.
3. The attacks on the grader that have been demonstrated are caught, and a
   detected tamper can never be scored as a pass. ✅ —
   `TestTheGraderIsProtected` and `TamperedPassError`. **Not** "no code path
   can produce a tampered pass": that would be a claim about detector
   completeness, and this detector is not complete. See *What the tamper check
   does and does not cover*.
4. Every field of `TaskOutcome` has a producer, asserted against an agent whose
   behaviour is scripted. ✅ — `tests/test_eval_runner.py`.
5. The held-out split is stable as the suite grows and identical across
   processes. ✅ — pinned sha256-derived values, not self-consistency.
6. `harness eval` runs a suite end to end and reports what it dropped. ✅

## Telemetry
none of its own. The runner reads the events S-201 already writes.

## Rollback
`git revert`. Nothing outside `harness/eval/` and the `eval` subcommand depends
on it; the orchestrator change is separately revertible (see below).

## S-201 activation (the part that touches the agent path)
Before this spec, `GitSubstrate` was constructed **nowhere but in tests**, and
`probe_environment` was called nowhere at all. Both were the archetype this
project keeps hitting: a mechanism that never fires, behind telemetry that
looks healthy because the tests exercising it pass.

`Orchestrator._activate_repo_substrate` is now their first production caller,
applying the two-dimensional rule once per run:

```
active(c) = profile.enables(c) ∧ environment.affirms(c)
```

The profile half is checked **first and cheaply, before any exec**. A run whose
profile did not name `git_substrate` issues zero commands — that is what keeps
N3/N4 structural rather than merely observed. The environment half is the one
that says no for real reasons: a non-git workspace has nothing to checkpoint,
and a container the harness does not own must not be written to.

Neutrality argument: `CODING` names no capabilities, so the gate returns before
`setup_budget_for` is even consulted. `tests/test_repo.py::TestProductionActivation`
asserts this on the *commands actually executed* — a substrate that probed and
then declined would emit the same events, and asserting on events would miss
it. `test_S201_a_profile_without_the_capability_never_probes` is the one that
makes the gate load-bearing: with a profile that asks for a *different*
capability the budget is non-zero, so deleting the gate really does issue git
commands, and every other test in the file still passes.

Failures never fail the run: a probe that raises, or a substrate that cannot
start, disables the feature. `allow_dirty=True` is passed deliberately — a real
repository is often dirty, and aborting is the right default for a caller who
can go and commit first, the wrong one for a run already under way. The tree's
state lands on `substrate.state.dirty` and **nothing persists it**, so today a
dirty baseline is knowable and not reported; that is S-202's open gap, not a
claim this spec gets to make.

## What the tamper check does and does not cover
The tests live in the agent's writable tree and deleting them wins under most
runners, so this check is what makes the grade mean anything. Four versions of
it have been defeated, and the pattern in the failures is the useful part:
**every one of them asked git, and git state is what the agent controls.**

- `git status` compares to `HEAD`, and one `git commit` makes the tree read
  clean.
- `git ls-files --exclude-standard` reads the tree's own `.gitignore`, so
  appending `conftest.py` to it hid that file from every check.
- `git update-index --assume-unchanged` makes a modified file report as
  unmodified, zeroing the diff metric.
- Hashing only `task.test_paths` missed **deleting a pre-existing test** the
  change never touched — and the grading command runs the whole suite, so
  every test file is the grader. An agent could ship a regression and delete
  the test that proved it.
- Applying `GRADER_CONFIG_NAMES` only to files the agent *added* left every
  name on the list unguarded whenever the repository already had one, which is
  the common case. Editing an existing `conftest.py` or `Makefile` was a
  scored pass.

The answer is to stop asking git. `TaskTree.manifest` is a sha256 of every file
at build time, held in this process; `diff()` compares content. Then:

- **any** change to a file matching the test patterns is tampering — added,
  edited or deleted, whether or not the change under test introduced it;
- **any** change to a `GRADER_CONFIG_NAMES` file is tampering unless the
  reference commit touched that file too, so a task whose work *is* editing
  `pyproject.toml` stays passable.

One further hole was not about detection at all. A `.pyc` is validated against
its source's **mtime and size**, never its content, so a grader can be gutted
to the same byte length, executed to leave a passing cache, then restored
byte-for-byte with its original timestamp — content hash clean, grader green.
Reproduced end to end: `1 passed` against a module that does not define the
function under test.

`wipe_caches` runs before every graded command, and it **raises** rather than
swallowing a failure, because it is the only thing between a planted `.pyc` and
a green grade and a silent failure would leave the defence looking present and
doing nothing.

A first version also set `PYTHONDONTWRITEBYTECODE=1`, described in this spec as
part of the defence. It is not: that variable stops Python *writing* bytecode
and has no effect on *reading* a cache that already exists, which is the whole
attack. It was removed — a second defence that does nothing is worse than one,
because it makes the first look optional — and with it went an unrequested
mutation of the operator's own test command.

The test carries a **control**. `test_S401_the_stale_bytecode_attack_actually_works`
asserts the attack succeeds before its sibling asserts the wipe stops it. That
is there because the first version of the defence test passed with the defence
removed, and two attempts to reproduce the attack by hand also failed — for
unrelated setup reasons. Without the control, "the defence works" and "the
attack never worked" are the same green test.

What it still cannot see: an edit to an ordinary source file the grader
imports, made so the test passes by a side effect rather than by the intended
change. That is indistinguishable from a legitimate fix without understanding
the code, and nothing here attempts it.

## Grading happens where validation happened
Two separate mistakes, one fix.

Grading originally ran `subprocess.run(shell=True)` in the harness process
while the agent ran in a container. That undoes the isolation completely: a
`conftest.py` the agent added is arbitrary code and pytest imports it before
collection, so choosing Docker to keep model-authored code off the host and
then grading on the host executes exactly what the container existed to
contain.

Moving grading into a sandbox then created a *correctness* bug that was worse
than the security one it fixed: validation still ran on the host. A suite whose
`test_command` names a host interpreter — which any real suite file does —
validated perfectly and then failed every trial inside the image, so a correct
solution scored as a plain model failure. That was the default outcome on any
machine with a Docker daemon. `validate` now takes its runner as a **required**
argument, so where commands run is always a stated choice, and the CLI passes
the same sandboxed runner it grades with.

The third piece: `Graded.ran` distinguishes "the tests failed" from "the
command never ran". Every exec failure used to collapse to exit code 1, so a
dead daemon or a timeout scored as a model failure — and a timeout on the
*regression* command produced `regressions=1`, a finding the harness invented
and attributed to the agent.

## Defects this found in shipped code## Defects this found in shipped code
All the same shape, and collectively the reason this spec is larger than it
looks.

**`ShadowReader.turns()` never worked.** The command is
`for-each-ref --format=%(refname) …`, run through a shell, where `%(refname)`
unquoted is a syntax error. `_run` maps any failure to `(1, "")` and `turns()`
maps that to `[]`, so in every real shell the reader reported "no checkpoints"
for a store full of them. Every S-202 test passed, because they exec against a
fake sandbox that records command strings and never runs one.
`tests/test_diffs.py::TestAgainstARealShell` now runs the reader's own commands
through a real shell against a real shadow store; reintroducing the missing
quotes fails four tests.

**Porcelain paths lost their first character.** `git status --porcelain` pads
the status field with a leading space for an unstaged modification (`" M path"`).
Stripping the output as a whole ate it, so `line[3:]` returned `alc.py` for
`calc.py` — which meant the tamper check compared a name that matched nothing
and **passed for a grader the agent had rewritten**. Both readers now use `-z`
and never strip.

## Reading the substrate after the sandbox is gone
The first version of this runner read `turns_to_first_edit` out of S-201's
shadow refs after the run. That worked in tests and would have worked never in
practice: the shadow store lives at `/tmp/.harness-git` **inside the sandbox**,
so under `DockerSandbox` — the backend anyone would pick for running
model-authored code — it is destroyed at teardown. The metric would have
reported "not measured" on every real run, honestly and uselessly, which is the
project's recurring failure wearing a clean shirt.

The fix is not to keep the store alive; it is to stop needing it. A checkpoint
already computes a `write-tree` hash, so `Checkpoint` now carries it and the
loop records it in the `repo_checkpoint` event. "Did this turn change the
tree?" becomes a comparison of two strings that were written to the log at the
moment they were true, and it survives teardown on every backend. A host-side
mount would also have worked and was rejected: it gives the agent a writable
host directory outside its workspace, which is a security regression traded for
telemetry.

Activation gets its own event (`repo_baseline`) for a related reason. A run
whose agent makes no tool calls writes no checkpoints at all — identical in the
log to a run where the capability never switched on. Without a record of
activation, a consumer must report one of those as the other.

## What "not measured" means, and why it is not zero
Two fields are tri-state on purpose:

- `regressions` is `None` unless the suite supplies a regression command *and*
  that command was green in the starting tree. Reporting `0` for a check nobody
  ran is a clean bill of health nobody issued; and a task whose wider suite was
  already red cannot attribute a later failure to the agent.
- `turns_to_first_edit` is paired with `first_edit_measured`. "The agent never
  edited anything" and "we had no telemetry to tell" are different findings,
  and `SuiteReport` keeps the second out of both the mean and the never-edited
  count. Four distinct situations resolve to *unknown* — no activation event, a
  baseline with no tree, a checkpoint written by an older loop, and an edit
  that could have happened inside a skipped turn — and each is listed in
  `_first_edit_turn`'s docstring rather than collapsed into a bare `False`.

`SuiteReport` publishes the denominator beside every mean (`precision_trials`,
`regression_trials`, `first_edit_trials`) for the same reason: a mean over a
silently-filtered subset is how a metric stops measuring what its name says
while nothing looks wrong.

## Why the first edit is read from checkpoints, not from tool calls
An agent that edits through a bash heredoc changes the tree without ever
calling `edit_file`. A tool-call-derived metric scores it as never having
edited anything — and "the agent went eight turns without touching the code" is
exactly the finding this metric exists to surface, so a false positive on it is
worse than no metric. `test_S401_an_edit_made_through_the_shell_still_counts` pins it, and
`test_S401_first_edit_survives_the_sandbox_being_destroyed` deletes the shadow
store outright before measuring.

## What the first live run changed
Everything above was verified against scripted agents. The first run against a
real model on a real repository — GLM-5.3-flash over 11 merged `pallets/click`
commits — found four things that no fixture had.

**`harness-sandbox:latest` did not exist.** DESIGN.md has named it as the
default sandbox image since the sandbox was written and no Dockerfile existed
anywhere in the tree, so every Docker-backed run failed on an image pull that
could not succeed. There is now a `Dockerfile`.

**The grader could not be *this task's* tests.** `test_command` was a fixed
string, so it had to name either the change's tests or the whole suite — and
naming the whole suite makes every verdict hostage to any unrelated failure in
the repository. One flaky test and the benchmark reads 0% for reasons that have
nothing to do with the agent. `{tests}` now substitutes the task's own paths.

**The environmental check only knew about `pytest`.** A task whose tests import
`typing_extensions` did not match, so a missing dependency was reported as
"tests fail with the reference answer applied (task is not solvable as
generated)" — blaming the task for a gap in the image. Those are opposite
findings: one is fixed by editing a Dockerfile, the other by dropping the task,
and telling them apart is the entire purpose of the check. The markers are now
broad, which is safe *only* because they are applied to the head state alone:
there the reference answer is applied, so every module the change introduces
exists and an import that still fails is necessarily a third-party dependency.

**Three of eleven trials paused on the budget, invisibly.** The report showed a
pass rate and said nothing about attempts that ran out of time. Worse, the
first-edit denominator was the only clue anything was odd: 8/11 measured. The
three unmeasured trials turned out to have done *all* of their editing in the
final one or two turns —

```
writes on turns : [12, 12, 13, 13, 13]
checkpointed    : [1..11]
SKIPPED         : (12, insufficient_time), (13, landing)
```

— exactly the turns where S-201 stops checkpointing to protect the landing
reserve. So `_first_edit_turn` reported *unknown*, correctly, on live data. Had
it reported "never edited", three runs in which the agent worked hard and
shipped a fix would have been recorded as runs where it never touched the code.
This is the tri-state earning its place; it is also the reason `budget_paused`
is now a field and a reported line, because a pass rate is not interpretable
without it.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list. Most
of what follows was found by hostile review after a version I had called
finished, twice.

- **The tamper check cannot see semantic cheating.** An edit to an ordinary
  source file the grader imports, made so the test passes by a side effect
  rather than by the intended change, is indistinguishable from a legitimate
  fix without understanding the code. `GRADER_CONFIG_NAMES` is also a
  blocklist and will miss names — though it is now applied to edits as well as
  additions, which was the larger hole.
- **`BUILD_BYPRODUCTS` is excluded from the manifest, so the manifest does not
  see everything.** That is the price of not charging an agent for running the
  tests, and it is what made the stale-`.pyc` trick possible in the first
  place. `wipe_caches` closes the Python case; an equivalent for another
  ecosystem's compiled cache is not written. A repository whose real source
  lives in `bin/`, `dist/` or `target/` is also misread.
- **Diff precision has no notion of a partial edit.** Touching the right file
  for the wrong reason scores the same as fixing it.
- **The diff artifact still does not survive a Docker run.** The first-edit
  metric does — a tree hash fits in an event log — but reconstructing a patch
  needs the objects, and those die with the container. S-202's reader is
  LocalSandbox-only. `git bundle` before teardown is the direction.
- **The first-edit metric needs `git` in the sandbox image.** It survives
  container teardown, which is what the earlier version could not do; it still
  requires the substrate to have activated, and `probe_environment` will not
  affirm `git_substrate` in an image without git. `python:3.12-slim` has none,
  so a run against that image reports `first_edit_measured=False` — honest,
  and easy to mistake for the feature being broken.
- **Isolation from the host is the sandbox's, not this module's.** The task
  tree contains no route to the answer. The *machine* does: the source
  repository is at a path the suite names, and `LocalSandbox` — the fallback
  whenever no Docker daemon is present — has full filesystem access. A suite
  run without Docker is not a trustworthy measurement, and nothing refuses to
  run one; it only warns.
- **Path classification is a heuristic with an asymmetric failure.** A
  production file misread as a test is *restored at head*, handing the agent
  part of the answer. `testing/` was dropped for this reason; `tests?/` is kept
  because the convention is overwhelming, and Django's `django/test/` is a real
  counterexample. The bound is validation, which rejects a task whose restored
  files make the tests pass at base — that catches the total case, not a
  partial leak.
- **Deleted source files are left in the task tree.** The change removed them;
  the agent is not asked to. They are counted in `reference_files`, so removing
  them is credited and leaving them costs recall — but an orphan that breaks
  the new tests makes the task unsolvable, and validation then drops it with a
  reason that blames the task rather than the construction.
- **`regressions` is a boolean wearing an integer's name.** 0 or 1: the rest of
  the suite still passes, or it does not. Per-test counting needs a
  per-framework result parser.
- **A trial starts up to three containers.** Regression baseline before the
  agent, then grading and the regression re-run after, plus the agent's own.
  Validation starts two more per task. Correct for isolation, expensive for a
  60-trial suite, and trials are sequential on top of that.
- **Activation is lead-only.** Subagents get no substrate. S-201 namespaces the
  index and refs per agent, but nothing has exercised two agents checkpointing
  concurrently.
- **Restored test payloads may contain the answer.** A hash golden is
  non-invertible; a plaintext expected-output fixture is the solution written
  down. `TaskSpec.answer_bearing_tests` flags them and nothing acts on the flag.
- **No suite is committed.** The machinery runs any suite file; this ships
  none, because a suite pinned to this repository's own history would be a
  benchmark whose author has read every line of the answers. The first real
  suite was generated from `pallets/click` and lives outside the repo.
- **Suite curation is manual and necessary.** Of twelve generated tasks, one
  was correctly dropped because `tests/typing/typing_edit.py` is a *static
  typing* fixture, not an executable test — running it under pytest launches
  an editor. Nothing distinguishes a type-check fixture from a runnable test by
  path, and the reason reported ("the test command could not run") is close
  but not exact.
- **The wall clock is a load-bearing parameter with no guidance.** At 420s,
  three of eleven trials ran out of time and every one of them wrote its fix in
  the final turn under a one-second command cap; the eval was measuring reading
  speed. `budget_paused` makes that visible, and nothing recommends a value.
- **`--no-validate` exists.** The help text and the report both say the number
  is not a benchmark result. Nothing enforces it.
- **The environmental check is a substring list.** A head-state test whose own
  assertion message contains "command not found" is misread as a broken
  environment. Applying it to the head state only bounds the damage.
- **No statistical machinery.** The report is point estimates. Comparing two
  harness versions needs the McNemar/Wilson treatment used for TB2.
