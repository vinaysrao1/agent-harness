---
id: S-106
title: Model routing
status: Implemented
lane: A
depends: S-105
effort: S
---

# S-106 — Model routing

## Contract
A `Router` resolving a model per *call purpose*, defaulting to the run's model
for every purpose. New module `harness/routing.py`; `HarnessConfig.routing`;
a `purpose` column on `usage`.

## The half of this that is not routing
The plan calls this "a one-line change" and says the spec exists to make it
measurable rather than merely done. Measuring it turned out to be the whole
job:

**`RunStore.record_usage` has exactly one caller** — `harness/loop.py`, the
main model call. The compaction summarizer calls `adapter.complete` directly
and its tokens are recorded **nowhere**. Every cost number this harness
produces — `harness run`'s usage summary, the eval's `tokens / task`, the
per-trial cost arithmetic in `agent-eval-methodology` — excludes them.

Today that error is zero, because S-404 established compaction never fires on
this workload. It stops being zero the moment it does, and it would have been
wrong silently: a summarizer routed to a cheap model would have shown up as a
*cost reduction* with no line item to explain it, because the expensive calls
it replaced were never counted either.

So routing without per-purpose accounting is worse than no routing. Both land
together.

## Two of the plan's four purposes have no call site
The plan names `main`, `summarize`, `classify`, `lint`. Only the first two are
model calls in this harness:

- `main` — `AgentLoop`'s model call.
- `summarize` — the compaction summarizer.
- `classify` — no such call exists.
- `lint` — `diligence.lint_verification` is a **pure function**, five
  deterministic detectors over a command string. It makes no model call and
  has no reason to.

Naming a purpose nothing can route is the defect archetype this project keeps
catching: a mechanism that never fires, with a config surface implying it
does. `CallPurpose` has two members. The next real model call adds the third,
along with the call site that justifies it.

## The config surface was dead on arrival
The first version of this shipped the archetype it is written about.
`load_config` builds `HarnessConfig` from an **explicit keyword list**, and
`routing` was not in it — so `[routing]` in a `config.toml` was parsed,
discarded, and never seen. Acceptance 2 was false (the only way to set it was
to construct `HarnessConfig` in Python, which is what the test did), and
acceptance 4 was *inverted*: a typo'd model name did not raise, it silently
used the main model. Verbatim the failure the raise exists to prevent.

`condenser` (S-105) had the identical hole, so it is a pattern in that
function, not a slip. Both are read now, and
`test_S106_every_declared_field_is_read_from_toml` reflects over the model
rather than naming fields, so it fails on the *next* one dropped.

## Invariants
Lane A. Default `routing = {}` resolves every purpose to the run's own model,
which is the same adapter *object* the run already shares — so the calls are
byte-identical and N1/N7/N8 are untouched. Conformance stays at 48.

**Reported numbers change, and they are corrections, not regressions.** The
CLI's usage summary now includes summarizer tokens. The eval's `tokens / task`
did *not*, until a review caught it: it read `AgentResult.usage`, accumulated
in memory from the lead's main calls only, so it had never counted subagents
either. It now sums the usage ledger, which fixes both. Comparing a post-S-106
cost figure against a pre-S-106 one is comparing different quantities; the
older number was an undercount of unknown size. On the current workload the
difference is zero — no compaction fires, and PR-replay trials spawn no
subagents — which is the only reason this lands as Lane A.

**One number that is not a report also changed, and was put back.**
`resume_task` derives the remaining token budget by summing the lead's usage
rows. Once the summarizer wrote rows, a resumed run subtracted spend the live
loop never counted — two definitions of one budget, measured at 40% apart on a
compacting run. The sum now filters `purpose == "main"`, matching what the
loop accrues.

## Acceptance
1. Default config is behaviour-identical — the summarizer resolves to the run's
   own adapter object, not merely to an equivalent one.
2. Routing the summarizer is a config change, not a code change.
3. `usage` rows carry the purpose, and the summarizer's calls **produce a row
   at all** — asserted by count, since the pre-S-106 count was zero.
4. An unknown model in `[routing]` raises at run construction, naming the
   purpose. A typo must not silently fall back to the main model: that is the
   failure mode where a run reports the cheap model in config and bills the
   expensive one.

## Telemetry
`usage.purpose`, defaulting to `"main"` for rows written before this column
existed. That default is a guess about history, not a measurement — every
pre-S-106 row *is* a main-model call, because nothing else wrote one.

## Rollback
`git revert`. With `routing = {}` — the default — the router returns the run's
model for every purpose and the only residue is a column nobody reads.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **No routing is exercised in production.** `routing` defaults to `{}` and no
  shipped config sets it, so the routed path runs only in tests. That is the
  archetype's shape, and the reason it is acceptable here is narrow: the
  *counting* half fires unconditionally, and it is the half that was broken.
  A routed run is one config line away and the test drives it end to end
  through a real `config.toml`.
- **`[routing]` is undocumented.** It is in no README or DESIGN section, and
  the sample config does not mention it. A setting nobody can discover is
  only marginally better than one that does not work.
- **`harness cost` still buckets by model, not purpose.** `UsageRecord` now
  carries `purpose` and `list_usage` selects it, so the data is reachable; the
  CLI does not use it. Under the default unrouted config a summarizer row
  carries the run's own model name, so the line item the routing story is
  about — "a cost reduction with no line item to explain it" — still does not
  appear in the one command that would show it.
- **`purpose` is a free string in the column.** `record_usage` takes `str`,
  not `CallPurpose`, so a caller can write anything. Typing it would mean the
  persistence layer importing `routing`, which is the wrong direction; the
  column is validated by nothing. Both production callers now pass
  `CallPurpose.<X>.value`, so the enum and the column cannot drift silently.
- **The summarizer swallows its own telemetry failures.** `except Exception:
  pass` around `record_usage` means a schema mismatch at runtime reverts to
  the pre-S-106 state — recording nothing — with no event and no warning. A
  test catches it in CI; nothing catches it in production.
- **Cost reports now include summarizer tokens, and older numbers do not.**
  `tokens / task` in the eval and the CLI's usage summary both change meaning.
  On the current workload the difference is exactly zero because compaction
  never fires, which is the only reason this is Lane A — but a comparison
  against a figure recorded before this commit is comparing two different
  quantities, and nothing in the report says so.
- **A routed adapter is built once per run and shared across agents**, like
  the main one. Fine today (provider clients are stateless across calls) and
  stated because it is an assumption, not a guarantee.
- **`classify` and `lint` are not implemented**, deliberately — see above. If
  a future model call needs one, the enum member and the call site land
  together.
