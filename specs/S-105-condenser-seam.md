---
id: S-105
title: Condenser seam
status: Implemented
lane: A
depends: S-003
effort: M
---

# S-105 — Condenser seam

> **Superseded in part by S-404.** `PivotalCondenser`, the `mark_pivotal`
> machinery, **and the retention plumbing behind them** were measured and
> **deleted**. S-404 scored the retention decision against what runs actually
> did next: a **25% loss rate** (compaction really does destroy things a run
> comes back for) against **0% marker recall** — it marked 1 turn of 100 and
> not one of the 25 — while keeping the single newest turn caught 46% on the
> spans where that is a condensation at all. The marker watched *errors*; what
> a run loses is *knowledge*. That is a category mismatch, not a tuning
> problem.
>
> The plumbing went too, one round later. Mutation testing showed every
> load-bearing line of it — `_Applied.kept` in `_effective`, the prune
> protection, the resume splice — could be deleted with the full 2,562-test
> suite still green. It was exercised by one assertion on an event payload,
> and that payload logged the strategy's *request* rather than what was
> applied. Keeping it as a documented capability nothing honours would have
> been the archetype this project keeps catching. The next retention strategy
> re-adds ~50 lines with the evidence that earns them.
>
> What remains from this spec, and is live: the `Condenser` seam, the
> non-destructive transcript, and `effective_size`. Sections below describing
> pivotal retention are the record of what was built and why it went.

## Contract
Extract compaction into a `Condenser` protocol; persist condensation as an
event that is *applied* at assembly time rather than destructively rewriting
the transcript; add pivotal-event retention.

```python
class Condenser(Protocol):
    async def condense(
        self, span: list[Message], ctx: CondenseContext
    ) -> Condensation: ...
```

`Condensation{summary, strategy_id, kept_refs, dropped_refs}`.

New module `harness/condenser.py`. `ContextManager` gains `condenser=`,
`mark_pivotal(ref, reason)` and `effective_size`; `self.transcript` stops
being rewritten by compaction. New capability `pivotal_retention`, in
`REPO_CAPABILITIES` only.

## Why this ranks high
A 100+ step run compacts 5–10 times and quality compounds multiplicatively.
It is also the only subsystem in the harness with no measurement at all: today
the summarizer is one `await self._summarize(evicted)` call whose output is
never inspected, compared, or scored by anything.

## Invariants
Lane A. Two things carry it.

**N7 with zero drift.** The default condenser produces byte-identical output
to today's `compact()`, so the replay corpus fires prune and compaction at the
same turn indices with the same shed volumes and the same token totals. Not
"±1" — zero. Any drift is a bug in the extraction, not a tuning question.

**Pivotal retention is off on the benchmark path.** It changes what survives
eviction, which changes the assembly, which N7 and N8 pin. It is a
`CODING_REPO` capability, exactly as `read_staleness` (S-102) and
`background_execution` (S-104) are, and for the same reason: `CODING` is
frozen until a Lane B run says otherwise.

## The transcript stops being rewritten
Today `compact()` does `self.transcript[:half] = [summary_message]`. The
evicted messages are gone from memory the moment the summarizer returns; the
loop persists them as a `compaction` event purely as a backstop for resume.

Instead, `ContextManager` keeps the raw transcript and a list of
`Condensation`s, and `_effective()` applies them in order to produce what the
model sees. Compaction always evicts a *prefix*, so each condensation is a
prefix replacement over the running view — which makes the chain trivial to
apply and trivial to reason about, and is why this is cheap rather than a
rewrite of the assembly path.

Everything that used to index `self.transcript` now indexes the effective
view. That is not a change of meaning: today's transcript *is* the effective
view. The prune plan's `frozenset[int]` stays consistent because every reader
inside one turn derives from the same memoized `_effective()`.

**The shrink guard has to move with it.** `harness/loop.py`'s
compact-to-fixpoint pass reads `len(self.context.transcript)` to decide
whether compaction is still making progress. Under a non-destructive
transcript that length never changes, so `len >= size_before` is true on the
*first* check every turn and compact-to-fixpoint degrades into compact-once —
a heavy transcript that one halving cannot bring under the threshold goes to
the model over the window, which is the case that loop was written for. (An
earlier draft of this paragraph claimed the guard would never fire and the
loop would summarize forever. That was wrong in the other direction, and
`maybe_compact`'s own `_eviction_boundary() < 2` check bounds the loop
independently.) `effective_size` is what it must read, and
`tests/conformance/replay.py` mirrors the same pass — reverting *its* copy
produces real N7 drift, which is what pins the change.

## Acceptance
1. The default condenser reproduces today's output byte-for-byte on the
   replay corpus — N7 with zero drift.
2. Strategies are swappable by config and identified in the event payload.
3. Pivotal retention: turns marked pivotal survive eviction regardless of age.
4. Constraint-survival: a constraint planted early is still in the assembly
   after forced compaction — **closes open question §9.1**.

## What "pivotal" can actually mean here
The plan names three signals: `verification_failed`, a plan-changing non-zero
exit, and a `decision` later cited. Only messages reach the condenser, so
pivotality has to be *told* to the context rather than inferred from it:

- `verification_failed` — marked at **one** of the four sites that emit the
  event: the branch that nudges, where the loop appends a user message holding
  the command, its exit code and its output. That message is the thing worth
  keeping; the other three sites accept the failure and end the run, so there
  is no later compaction to survive. The mark is on the message, not on the
  event, because only messages reach the condenser.
- A non-zero exit — a tool result with `is_error=True`, marked by
  `ContextManager.append` itself. Marking at the loop's call site instead
  looked equivalent and was not: resume rebuilds the transcript by appending
  replayed messages, so the marks were gone by the time a resumed run next
  compacted, and it dropped exactly what retention had been carrying.
  "Plan-changing" is not derivable, so this over-marks; the cap below is what
  keeps it from swallowing the eviction.
- A `decision` later cited — **not implemented.** "Later cited" needs a
  citation graph the harness does not have, and guessing it from
  substring overlap would be a mechanism whose failures are invisible. It is
  in Known gaps, not in the code.

## Retention has to survive the *other* eviction layer
Pruning (§4.3.2) stubs the oldest tool results under pressure; compaction
(§4.3.3) is the rung above it. Retention puts the kept turn at the **front**
of the effective view, and the shed is oldest-first — so the retained failure
was the first thing pruning stubbed, on every turn after the compaction, while
`kept_refs` and `pivotal_reasons` went on saying it had survived. Compaction
fires at 0.80 of the window and pruning engages at 0.50, so the view is
normally still under pressure right after a compaction: the common path, not
an edge case.

`_prune_plan` therefore skips what a condensation kept. Keyed on
`_retained_refs()` — the union of every condensation's `kept_refs` — and not
on `_pivotal`, because marks are recorded on every profile including the
benchmark one, and protecting marks directly would change the `CODING`
assembly and break N7. `DefaultCondenser` keeps nothing, so the protected set
is empty there.

The tests missed this because every one of them read `effective_messages()`,
which is the *pre-prune* view. The one that read `assemble()` planted its
marker in a user message, which pruning structurally never touches.

## Telemetry
The `compaction` event payload gains `strategy_id`, `kept_refs` and
`pivotal_reasons`. A retention that never retains anything, or that retains
everything, is then visible in the event log rather than inferred from
behaviour.

## Rollback
`git revert`. The condenser defaults to `DefaultCondenser`, whose output is
byte-identical to the code it replaces; pivotal retention is gated on a
capability no benchmark profile declares.

## What the constraint-survival test found (§9.1)
§9.1 asked whether compaction preserves what a run must not forget. It stayed
open because nothing measured it. `TestConstraintSurvival` drives the real
`AgentLoop` for forty turns at a 1,200-token window — seven-plus compactions —
plants a constraint by three routes, and reports which survive:

| route | survives | why |
|---|---|---|
| instruction ledger (`add_instruction`) | yes | re-rendered into the system prompt every assembly; compaction cannot reach it |
| the goal | yes | folded verbatim into every summary header, never through the summarizer |
| prose in a user message, unregistered | **no** | it has no representation the harness knows to preserve |

The third row is the answer, not a bug: a constraint the model never registers
is a constraint the harness cannot protect. That is what `add_instruction`
exists for, and this is the test that says how much it matters. The question
is closed because it now has a number attached and a control that fails when
the mechanism is removed.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **The raw transcript is held for the whole run.** Compaction used to free
  it. Measured on a deliberately heavy shape — 400 turns of 4 KB prose plus
  8 KB tool output at the production 128K window, seven compactions — the raw
  transcript is 4.88 MB against a 1.02 MB effective view: **3.86 MB extra**.
  Small next to the store, which already holds every one of those messages,
  but it is not nothing and it grows with run length rather than with window
  size.
- **A `decision` later cited is not a pivotal signal.** The plan names it;
  "later cited" needs a citation graph the harness does not have, and
  approximating it by substring overlap would be a mechanism whose failures
  are invisible. Not implemented rather than faked.
- **`tool_error` over-marks.** Every failing tool result marks its turn, and
  most failures are not plan-changing. The cap is what makes this survivable,
  not the precision of the signal. No measurement yet of how often retention
  keeps the wrong turn — that needs labelled runs.
- **Marks are never cleared.** A failure that the run subsequently fixed stays
  marked, so a late compaction can retain a turn that no longer matters. The
  most-recent-first ordering mitigates this and does not solve it.
- **A resumed run keeps `tool_error` marks but loses the rest.** `append`
  re-derives `tool_error` from the replayed messages, so those come back for
  free; a `verification_failed` mark is on a plain user message and there is
  nothing in the message to re-derive it from. It would have to be persisted.
- **`_goal_text` nests across a resume.** Pre-existing, not an S-105
  regression, but §9.1's "the goal survives" row now rests on it. Resume
  replays the *condensed* view, so the first message it appends is the
  previous `[COMPACTION SUMMARY]` — and the next compaction folds that whole
  header into a new one, nesting them. The goal is still in there, further
  down each time.
- **`_turn_bounds` returns nothing for a span that opens on a tool result.**
  Correct (an orphaned `tool_result` is rejected by the provider) but it means
  a turn split across an eviction boundary is silently unretainable. The
  boundary snaps forward past tool results, so this should be unreachable.
- **Retention is capped at four turns and `len(span) - 2` messages.** Both
  numbers keep the loop's fixpoint pass shrinking; neither comes from evidence
  about how much context is worth keeping. Getting the *units* right took two
  goes. `max_kept` alone let a run where everything is pivotal stall at
  `1 + MAX_PIVOTAL_KEPT` messages. Spending `len(span) - 2` as if it were a
  turn count then let a two-turn six-message span keep all six — a
  condensation that *grew* the view from 6 to 7 — because `tool_error` only
  ever marks tool results, so a real pivotal turn is never one message long.
  The test that was supposed to catch the second one built its span from
  single-message user messages, where turn count equals message count and the
  bug is unreachable.
- **Pivotal retention is unmeasured against the thing it is for.** It has unit
  tests and a control, but no eval says a run with retention finishes more
  tasks than one without. S-401's PR-replay is where that number would come
  from, and it has not been run.
- **Strategy selection is silent.** A repo-mode config naming a strategy is
  honoured; the same config under `CODING` is ignored without a warning. That
  is deliberate — the config is global and the profile is per-run — but a run
  report does not currently say which strategy actually ran, only the event
  log does.
- **`_effective()` rebuilds the whole view on every invalidation.** O(n) per
  turn, memoized within a turn. Same order as the assembly it feeds, so it
  does not change the shape of the cost, but it is a second full pass.
