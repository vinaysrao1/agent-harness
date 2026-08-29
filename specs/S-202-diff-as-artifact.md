---
id: S-202
title: Diff as the artifact
status: Implemented
lane: A
depends: S-201
effort: S
---

# S-202 — Diff as the artifact

## Contract
Read S-201's shadow refs into something a human can look at: a stat line, a
file list, a patch, `rewind` to any turn, `AgentResult.diff_stat`, and a PR
body assembled from ledger evidence.

## Invariants
Lane A. `harness/diffs.py` is imported lazily inside `_record_diff_stat`, which
returns immediately without an active substrate — so the benchmark path neither
imports nor executes any of it. Conformance stayed at 48 passed; no golden
moved.

## Acceptance
1. `diff` works from the shadow ref with no workspace mutation. ✅ — asserted
   by inspecting every command issued, and by the absence of `--work-tree` from
   reads.
2. `rewind` is explicit and never automatic. ✅ — asserted structurally: an AST
   check that no other function in the module calls it, and a source check that
   the agent loop never mentions it.
3. The final report leads with `N files changed, +X/−Y` and the file list. ✅
4. A PR body assembles from `task_ledger.evidence`. ✅

## Telemetry
none. The diff is computed on demand from refs S-201 already wrote; adding an
event would record a fact already recoverable from the refs themselves.

## Rollback
`git revert`. `AgentResult.diff_stat` defaults to `None`, so a caller that
ignores it is unaffected.

## Why reads carry no work tree
`ShadowReader._git` deliberately omits both `GIT_INDEX_FILE` and
`--work-tree`. A read that carried a work tree could touch the index, and
`diff` promises not to. `rewind` adds them back explicitly — it is the one
operation that writes, and it exists solely because a caller asked for it by
name. A tool that silently reverted the agent's work the moment you asked to
*look* at it would be worse than no tool, so the asymmetry is enforced rather
than documented.

## Turns without checkpoints
S-201 writes a checkpoint only on turns with tool calls, so a requested turn
may have none. `rewind` resolves to the nearest earlier checkpoint **and says
so** — "turn 3 (turn 4 changed nothing, so no checkpoint was written for it)".
Failing would be unhelpful, since the tree at turn 4 genuinely is the tree at
turn 3; returning turn 3 silently would be worse than failing.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **No CLI.** The contract names `harness diff <run_id>` and
  `harness rewind <run_id> <turn>`; this ships the reader they would call, not
  the commands. Wiring `harness/cli.py` means resolving a run to its agent and
  a live sandbox after the run has ended — the sandbox is torn down in a
  `finally`, so a post-hoc CLI needs a way to re-attach that does not exist
  yet. Naming that here rather than shipping a command that cannot run.
- **`RepoState` is still not persisted to the run row.** S-201 deferred it to
  this spec on the grounds that this is the spec that reads it back — but with
  no CLI there is no post-hoc reader, so the column would have no consumer. It
  moves to whichever spec first needs to resolve a run without a live loop.
- **`diff_stat` is computed only on the completion path.** A run that errors or
  pauses on budget reports `None`, even though its checkpoints exist and the
  diff is perfectly computable. Deliberate for now — the failure paths already
  carry enough that adds risk — but it means the artifact is missing exactly
  when a human most wants to see what the agent had done.
- **`rewind` restores the tree but not the agent's context.** Files go back;
  the transcript, ledger and memory do not. It is a filesystem operation
  presented as one, not a time machine.
- ~~The reader is exercised against a scripted sandbox only.~~ **Closed by
  S-401, which found the defect this gap was describing.** `turns()` builds
  `for-each-ref --format=%(refname) …` and runs it through a shell, where the
  unquoted parentheses are a syntax error; `_run` maps the failure to
  `(1, "")` and `turns()` maps that to `[]`, so the reader reported "no
  checkpoints" for a store full of them — in every real shell, while every
  test here passed. `TestAgainstARealShell` now runs the reader's own commands
  against a real shadow store.
- **The artifact does not survive a Docker run.** The shadow store lives at
  `/tmp/.harness-git` *inside the sandbox*. Under `DockerSandbox` it is
  destroyed with the container, so after the run — the only time anyone wants
  to look at the diff — there is nothing to read. Only `LocalSandbox` runs
  leave a readable history.

  S-401 hit this and routed around it rather than through it: the facts a
  metric needs (which turn changed the tree) are small enough to put in the
  event log, so `Checkpoint` now carries its `write-tree` hash and the loop
  records it. That does nothing for *this* module, whose whole output is the
  objects. The remaining options are a host-side mount — which hands the agent
  a writable host directory outside its workspace, a security regression traded
  for convenience — or `git bundle` before teardown, which copies only what the
  run produced and is the direction to prefer. Neither is built.
