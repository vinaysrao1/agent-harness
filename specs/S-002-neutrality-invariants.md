---
id: S-002
title: Neutrality invariants and the TB2 conformance suite
status: Implemented
lane: A
depends: S-001
effort: S
---

# S-002 — Neutrality invariants and the TB2 conformance suite

## Contract
`tests/conformance/` implementing **N1–N6** and `tests/golden/` holding the
frozen artifacts plus `CHANGELOG.md`, one entry per deliberate change to the
`CODING` path, each naming the TB2 run id that justified it.

N7–N8 are defined against the replay corpus and are therefore S-003's to
deliver. N6 is **not** deferred with them: the plan's own N6 row says "already
exists — promote it to the conformance suite", and the frozen `exec_capped`
corpus has been in `tests/test_deadline.py` since round 3.

## Invariants
This spec *is* the invariants. It establishes the baseline; it does not change
behavior.

## Acceptance
1. N1–N6 pass at **`b4fc55f` + S-001**, with goldens generated from that tree.

   The baseline is that tree and not `b4fc55f` alone, because N5's golden
   hashes `COMPLETION_GATES` / `NUDGE_SOURCES` — constants this spec itself
   introduces into `harness/loop.py`. Pinning them "at `b4fc55f`" is not a
   stricter reading of the plan, it is an unsatisfiable one. The prompt, tool
   and deadline goldens *are* reproducible at `b4fc55f` and were verified so.
2. Negative tests are the deliverable: an injected extra tool fails N2, an
   injected extra system-prompt line fails N1, an injected startup `exec` fails
   N3. An invariant never seen to fail is not known to work.
3. N1–N5 complete in under 5 seconds.

## Telemetry
none

## Rollback
`git revert`; the goldens are additive files.

## Neutrality argument
Lane A — tests only.

## Note
N7–N8 are defined against the replay corpus and land with S-003. **N6 lands
here**, not with them: the plan's own N6 row says "already exists — promote it
to the conformance suite", and the frozen `exec_capped` corpus has been in
`tests/test_deadline.py` since round 3. Deferring it would have been deferring
work already done.

## What landed
`tests/conformance/` — `fixture.py` (the fixed inputs every golden is computed
over), `test_neutrality.py` (N1, N2, N5), `test_neutrality_run.py` (N3, N4, via
a scripted `FakeAdapter` on `LocalSandbox`), and `test_n6_deadline.py` (N6).
`tests/golden/` holds five digests plus `CHANGELOG.md`. Re-freeze with
`python -m tests.conformance.freeze [golden-name ...]`.

**N1 pins `context.assemble()[0]`** — the string a turn actually sends: base
rules, skill bodies, the memory block and the instruction ledger. An earlier
draft hashed only `Orchestrator._system_prompt`, which left the entire
context-assembly layer free: a new always-in-context section reached every turn
of every run and neither the conformance suite nor the other 1912 tests
noticed. A second, narrower digest of the base rules is kept so a failure
localizes. The fixture holds one fixed skill and a fixed memory index, because
with an empty `SkillLibrary` the skills-index branch never rendered and its
header text was unpinned.

**N2 drives the real `Orchestrator._build_registry`**, not the
`CODING_TOOL_FACTORIES` tuple. Iterating the tuple pinned the tuple: a tool
registered directly in `_build_registry` reached the model on every run and the
invariant never saw it. Both surfaces are pinned — the 13-tool subagent
registry and the 15-tool lead registry, the latter being what Terminal-Bench
presents — along with the tool cap. Switching to the real registry reproduced
both digests byte-for-byte, so this was Lane A with no re-freeze.

**N4 baselines a pristine workspace.** Baselining after a run has already
executed bakes that run's residue into the baseline, so an identical second run
reproduces it and the check passes. That is not hypothetical: the first draft
accepted a run writing a fixed-name bookkeeping file on every startup. Residue
in the real world is deterministic — `.git`, a staged binary, a cache index —
so a check that only catches *non-deterministic* residue catches the case that
does not happen.

N4's scope is now asserted rather than implied. It pins the workspace tree. It
does **not** police writes outside the workspace: `LocalSandbox.write_file`
routes through `resolve_workspace_path`, which raises on any escaping path, so
asserting that recorded write paths are relative is a tautology enforced by the
code it claims to check — and the one deliberate outside-workspace write, the
spill path, goes through `exec` rather than `write_file`, so a write recorder
cannot see it at all. The recorder is kept, with a test proving it is
non-vacuous, and the structural refusal is asserted directly.

**N5 has two halves.** The declared `COMPLETION_GATES` / `NUDGE_SOURCES` tuples
plus the tuned budgets (`MAX_NUDGES`, `MAX_TRUNCATION_CONTINUES`) are pinned
against a golden — the tuples alone left the *values* free, and raising
`MAX_NUDGES` from 2 to 5 is a real change to how a run terminates. The
structural counts of `nudges += 1` sites and `_finish("completed")` returns are
derived from `loop.py`'s AST, so a gate added without declaring itself fails
too; `status` is matched positionally *or* by keyword, since the positional-only
form was evadable. The limit is stated in `loop.py`: a gate that neither nudges
nor adds a completion return would still slip past.

## Negative tests
Every invariant is driven with a real violation, not merely observed passing:
an added tool, a tool registered outside the factory tuple, a description edit,
a reordering, a schema change, an injected prompt line, a new always-in-context
section, a raised nudge budget, a third nudge site, a keyword-form completion
return, an exec issued before the first model call, and fixed-name residue
written into the workspace during a run.

## Known gaps
This list is meant to be exhaustive; if you find something not on it, that is a
defect in the list, not an accepted limitation.

- The `Orchestrator.__new__` construction in the fixture raises if
  `_system_prompt` or `_build_registry` starts reading instance state, but not
  if either starts reading a *class* attribute.
- The fixture pins `PermissionMode.GATED` and a `/workspace` label; Terminal-
  Bench runs `AUTO` with a Harbor-supplied label. The fixture is a **drift
  detector**, not a byte-reproduction of a benchmark turn — shared rules text
  is still pinned, which is what N1 is for.
- A qualifying §2.4 baseline (kimi-k3, k=3, at the frozen commit) is owed
  before this spec moves to `Verified`. See `tests/golden/CHANGELOG.md`.
- **N3 and N4 measure `LocalSandbox` through `run_task`; Terminal-Bench enters
  through `harbor_agent` on `HarborSandbox`.** The unqualified claim "zero
  execs before the first model call" is false on the benchmark path by exactly
  one: `HarborSandbox.start` execs `pwd` to detect the workspace root, before
  the orchestrator exists. That exec is pinned rather than wished away — it may
  be one, a second fails — so growth is caught either side of the boundary.
  N4 likewise tree-hashes a local workspace, not a Harbor container.
- The lead surface digest re-assembles `spawn_agent`/`await_agents` rather than
  driving `_execute`. A third lead-only tool would not move that digest, so the
  registration site is pinned separately by an AST count over `_execute`.
- N1 pins the assembled system string *and* the trailing reminder, but not
  every model-facing string the harness can emit (tool-result truncation
  markers, nudge texts). Those are covered by ordinary unit tests, which a
  future author could update without a re-freeze.
