---
id: S-004
title: AgentProfile promotion
status: Implemented
lane: A
depends: S-002
effort: S
---

# S-004 — `AgentProfile` promotion

## Contract
Fold `domain_rules`, `tool_factories`, `sandbox_spec` and
`capabilities: frozenset[str]` into `AgentProfile`. Ship `CODING` (identical to
today), `CODING_READONLY`, and `CODING_REPO`.

## Invariants
N1 and N2 must hold for `CODING` **with no golden change**. That is the whole
spec: promoting the struct must be invisible on the benchmark path. Every other
assertion here is secondary to that one.

## Acceptance
1. N1/N2 green with no golden change. ✅ — verified; the digests did not move.
2. Precedence: explicit run-level arguments override profile defaults, and a
   profile's `allow` globs can never weaken the operator's mode. ✅ — the
   override path is covered end-to-end by the pre-existing
   `tests/test_profiles.py::test_explicit_args_override_profile`, and the
   `allow` merge by two tests that drive `run_task` (below).

   Narrowing recorded: m10 states this as "explicit run-level
   `mode`/`model`/`budgets` override profile defaults". `AgentProfile` carries
   none of those three, so that half is vacuous by construction; what is
   actually testable is that explicit `domain_rules`/`tool_factories` win.
3. Every profile's assembled prompt still carries the data-not-instructions
   clause. ✅ — by `tests/test_profiles.py`, which already parametrizes over
   `ALL_PROFILES` and so covered `CODING_REPO` the moment it was added.
4. `spawn_agent` accepts an optional profile; default is inherit.
   **Deferred to S-304** — see below. Not ticked.

## Telemetry
none

## Rollback
`git revert`. `Profile` remains an alias of `AgentProfile`, so no call site
depends on the new name.

## Neutrality argument
`CODING` declares no capabilities, no sandbox spec and no permission patterns,
so every new field is inert on the benchmark path. The promotion is a rename
plus defaulted fields; N1/N2 passing unchanged is the proof, not the claim.

## Capabilities are asked for, not switched on
`capabilities` is half of the two-dimensional rule
`active(c) = profile.enables(c) and environment.affirms(c)`. `enables()` is
named for what it answers — *was this asked for* — never *is this on*. The
other half arrives with S-005. Until then a declared capability does nothing,
which is deliberate: promoting the struct and adding the features are separate
changes, and only the first is provably neutral.

`CODING_REPO` therefore ships with today's tool set. It names five capabilities
(`git_substrate`, `repo_orientation`, `project_checks`, `regression_gate`,
`structured_search`) that Layer 2 will bind, so those specs can gate on
`profile.enables(...)` without inventing a vocabulary each time.

## A profile cannot weaken the operator
`permission_allow` is additive into the same `Policy.allow` list the config
uses. Two properties make it safe, both asserted rather than argued:

- `evaluate()` consults `HARD_DENY_CATEGORIES` **before** allow patterns, so a
  profile asking for `"*"` still cannot reach a hard-denied tool, in either
  mode.
- A profile has no mode field at all. The strongest form of "it cannot change
  the mode" is having nowhere to put one.

## Decision: `spawn_agent`'s schema is unchanged
Acceptance (4) is satisfied at the **handler**, not in the JSON schema. Adding
a `profile` argument to the tool's `input_schema` would change the tool surface
N2 freezes — a Lane B change owing a re-freeze and a benchmark run.

There is no evidence yet that letting the model choose a profile helps, and
S-304 (heterogeneous subagents) is the spec that would produce that evidence.
So the capability is wired for callers now and not paid for on the benchmark
path. When S-304 wants it model-facing, that is its re-freeze to justify.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- `sandbox_spec` carries a single `network` field and nothing consumes it yet.
  It exists so S-005 and Layer 2 have somewhere to put sandbox requirements;
  today it is a typed placeholder, and a profile that set it would be ignored.
- `permission_allow` is wired into `build_policy` and now covered by two tests
  that drive `run_task`, but **no shipped profile sets it**, so the path has no
  production exercise.
- **Every builtin is `side_effect=False`**, so in `gated` mode no builtin ever
  reaches the ask callback — the sandbox is the trust boundary. The
  `permission_allow` tests therefore need a custom side-effecting tool. Worth
  knowing independently: `--mode gated` currently gates nothing among the
  builtins.
- `resume_task` does not accept or forward a profile, so `permission_allow` is
  absent on resume. A live-vs-resume permission asymmetry, unfixed: giving
  `resume_task` a profile argument is a behavior change that belongs to
  whichever spec first ships a profile that sets patterns.
- Subagents inherit the lead's profile in full; per-spawn profiles are S-304's.
