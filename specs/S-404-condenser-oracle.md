---
id: S-404
title: Condenser oracle eval
status: Implemented
lane: A
depends: S-105
effort: M
---

# S-404 — Condenser oracle eval

## Why this exists
S-105 shipped `PivotalCondenser` with unit tests, a control, and no evidence
that it helps anything. The obvious way to get that evidence — an A/B of pass
rates with and without retention — would have returned "no difference", and
the result would have been uninterpretable. Across every run this harness has
recorded:

| | |
|---|---|
| agents / model turns | 738 / 12,729 |
| `compaction` events ever emitted | **0** |
| median peak context | 11,204 |
| max peak context | 122,270 |
| runs that would cross `0.8 × 128K` | 4 / 686 |
| runs reaching even the *pruning* rung (`0.5 × 128K`) | 24 / 686 |

The first draft of that table used `usage.input_tokens` alone, which with
prompt caching is only the *uncached* fraction. The real context is
`input + cache_read + cache_write`. The median was understated 2.3× and the
prune-rung count 1.8× — every error in the direction that supported the
thesis. The conclusion survives (4 of 686 is still ~0.6%) but the numbers
were wrong and are corrected here.

Compaction has never fired on this workload. An A/B would have measured a
treatment that never applies, spent a few hundred model runs, and produced a
flat line indistinguishable from "the feature does nothing useful".

This also corrects S-105's own framing. Its "Why this ranks high" section says
a 100+ step run compacts five to ten times. That is the m10 plan's claim,
repeated without checking; it is not true of the runs being measured.

## Contract
`harness eval condenser-oracle` scores the *retention decision* offline
against a ground-truth label derived from what the run did next. No model
calls beyond the runs already recorded.

Two pieces:

1. **A window override.** `TrialSettings.max_context` and
   `Orchestrator.run_task(max_context=...)` force compaction by shrinking the
   window rather than waiting for a workload that triggers it. This makes
   compaction an independent variable, and the resulting curve answers a
   question worth having on its own: how small a window this harness
   tolerates before it degrades — which is what says whether cheap
   small-window models are viable.
2. **An oracle scorer** over a run's event log:
   `harness/eval/condenser_oracle.py`.

## The label
For each `compaction` event, the evicted span is in the payload. Walk forward
through the rest of the run and ask which evicted turns the run
**demonstrably needed again**:

- **`REDISCOVERED_READ` (primary).** The run reads a file it had already read
  inside the evicted span, with no intervening write to that path by the
  agent. This is unambiguous repeated work: the content did not change, and
  the run went back for it because it no longer had it.
- **`REPEATED_COMMAND` (secondary, weaker).** The run re-issues a command
  whose head and first argument appeared in the evicted span. Reported
  separately and never merged into the primary number, because re-running a
  test suite after an edit is correct behaviour, not evidence of loss.

Both are proxies for "the run needed this and no longer had it". They are
deterministic, derived from `tool_call` events alone, and stated as proxies
rather than as ground truth.

## What is scored
The marker (`tool_error` on a failing tool result, `verification_failed` on
the loop's reminder) is **re-derived offline** from the evicted messages
themselves — `is_error` is on the message, and the reminder is matched by its
own prefix. So the scorer needs no marks persisted and can be run against any
recorded run, including ones made before S-105.

Reported per run and in aggregate:

- **loss rate** — evicted turns needed again ÷ evicted turns. If this is
  ~0 there is no harm to prevent and `PivotalCondenser` should be deleted
  whatever else it scores.
- **recall** — needed-again turns the marker would have kept.
- **precision** — marker-kept turns that were needed again.
- **the recency baseline** — the same numbers for "keep the last N turns",
  which costs no marking heuristic at all. A marker that does not beat this
  is not earning its complexity.

## Acceptance
1. The scorer runs against a recorded run and produces the four numbers.
2. The window override actually makes runs compact — asserted, not assumed.
3. The recency baseline is computed on the same spans, so the comparison is
   paired rather than across arms.
4. A synthetic run with a planted rediscovered read is labelled, and a control
   where the file was written in between is **not** — the confound the
   primary label exists to exclude.

## What this deliberately is not
Not an A/B, and not a pass-rate measurement. Those come after, and only if
this says there is an effect to find. Standing at ~12 PR-replay tasks, a
pass-rate comparison cannot detect anything but an enormous effect, and
reporting one would be fitting a conclusion to an n that cannot carry it.

## The expected result, written down in advance
Most likely: the loss rate is low, and recall is near zero because the marker
watches **errors** while what a run actually loses is **knowledge** — a file
it read, a path it found. If that is what comes back, the honest response is
to delete `PivotalCondenser` and keep the seam, not to retune the marker until
the number moves.

Pre-registering that is the point. The alternative is looking at the output
and deciding afterwards which metric was the real one.

## Why compaction is not merely rare — it is structurally unreachable
The first attempt at forcing it failed. A live run at a 24,000-token window
produced **zero** compactions, and the reason is the eviction ladder itself:

```
prune engages at   0.50 W
prune sheds down to 0.40 W
compaction fires at 0.80 W   -- on the POST-prune size
```

Compaction reads the **post-prune** size, so it fires only when what pruning
does not shed exceeds `0.80 W`. On a real 53-turn `click` run:

| | tokens |
|---|---|
| assistant + user `.content` only | 740 |
| assistant + user **as actually counted** (tool_calls payload included) | 5,381 |
| system prompt | 591 |
| **floor that pruning can never touch** | **5,972** |
| tool results (prunable, outside the keep window) | 19,634 |

An earlier draft of this section reported the 740 and called tool output "96%
of the context". Both were wrong: the 740 omits the `tool_calls` arguments the
provider is charged for, the real share is 77%, and — the larger error — the
floor is not what triggers compaction at all. `PRUNE_KEEP_TURNS = 3` protects
the newest tool results, and at `W = 10,000` those are ~7,148 of the 8,108
post-prune tokens that cross the threshold. The trigger is dominated by the
term the earlier draft did not count.

Replaying that run's real transcript through a real `ContextManager`
(`harness condenser-oracle <db> --reach`, so the table is reproducible rather
than asserted):

| window | compactions |
|---|---|
| 128,000 / 32,000 / 24,000 / 16,000 / 12,000 | **0** |
| 10,000 | 3 |
| 8,000 | 6 |
| 6,000 | 9 |
| 4,000 | 17 |
| 2,000 | 47 |

The counts depend on the summarizer stand-in, because each summary lands in
the transcript the next compaction measures. Two independent rebuilds
disagreed (3/6/9/17/47 against 2/4/7/15/33) for exactly that reason before
`SUMMARY_STANDIN_CHARS` and `SYSTEM_PROMPT_STANDIN_CHARS` were pinned to the
conformance corpus's values. The *shape* — zero above 12K, monotone below —
is what the claim rests on; the counts are indicative.

So the eval runs at **6,000**. That is not a realistic production window and
is not presented as one: it is the forcing function that makes the retention
decision exist at all. The oracle's numbers do not depend on the task passing,
which is what makes a degenerate window usable here — loss rate, precision and
recall are all measurable on a run that failed.

The finding stands on its own, independent of anything about retention: **on a
tool-output-heavy workload the compaction rung is dead code above a ~10-12K
window.** Pruning gets there first and targets well below the trigger. That is
the ladder working as designed, and it means S-105's summarizer — and every
future condenser strategy — is inert on this workload at any plausible window.

## The result
12 `click` PR-replay tasks, GLM-flash, `--max-context 6000`. 34 compactions,
100 evicted turns. (Pass rate 16.7% — the window is deliberately crippling;
the oracle's numbers do not depend on the task passing.)

| | |
|---|---|
| **loss rate** | **25.0%** — 25 of 100 evicted turns were read again, unmodified |
| repeated commands | 15 (reported only) |
| **marker** | precision **0.0%**, recall **0.0%** |

The first reading of this gave 21.0%. A review found `_SEGMENTS` split on a
bare `|`, shredding the escaped alternation in `grep -n "pager\|PAGER" f.py`
— 20 of 127 real bash calls, including reads of the two most-read files in
the corpus. Fixing it *raised* the loss rate: the label had been blind to
those reads.

The first question is answered, and the answer is *yes*: compaction does
destroy things a run comes back for, in about a fifth of evicted turns. There
is harm here to prevent.

The second answer is that the marker does not address it. It marked **1 of 100
turns** — one `tool_error`, no `verification_failed` at all — and that one was
not among the 21. Recall is 0/21. This is precisely the failure written down in
advance: **the marker watches errors, and what a run loses is knowledge** — a
file it read, a path it found. That is a category mismatch, not a tuning
problem, and it does not depend on the window.

### The baseline, with the degenerate row excluded
The first reading of this was nearly a mistake worth recording. Spans here
average 2.9 turns, so `recency-4` keeps 89 of 100 turns — its 100% recall is
"keep almost everything", not a result. The comparison has to be read down the
whole ladder:

| strategy | over all 34 spans | over spans where it would actually condense |
|---|---|---|
| marker | P 0.0% · R **0.0%** | — |
| recency-1 | P 52.9% · R 72.0% | 15 spans · P 40.0% · R **46.2%** |
| recency-2 | P 32.4% · R 88.0% | 6 spans · P 16.7% · R 40.0% |
| recency-4 | P 28.1% · R 100.0% | 4 spans · P 25.0% · R 100.0% |

The right-hand column is the one to read, and getting there took two
corrections. A condensation replaces the span with a summary plus what it
keeps, so keeping `k` turns only shrinks the view while `1 + k < len(span)`.
Spans here average 2.9 turns: `recency-4` keeps 89 of 100 turns, and even
`recency-1` is not a condensation on 19 of the 34 spans. The first reading of
this reported recency-1 at 81% recall; 12 of its 17 true positives came from
spans where the baseline could not have run at all.

`render_report` now prints both columns and names the infeasible spans, so
the degenerate row cannot be quoted by accident.

Even at 46%, the cheapest thing that could possibly work beats the marking
heuristic by 46 points of recall while the marker scores zero.

### The decision, as pre-registered — carried out
Recall below ~0.3 was the stated kill line. It is 0.0. **`PivotalCondenser` and
the `mark_pivotal` machinery are deleted.** Removed: `PivotalCondenser`,
`_turn_bounds`, `MAX_PIVOTAL_KEPT`, `CondenseContext.pivotal`,
`Condensation.reasons`, `ContextManager.mark_pivotal` and `_pivotal`, the
auto-marking in `append`, the loop's one marking site, the `pivotal_retention`
capability, and `select_condenser_strategy` — the profile gate had nothing left
to gate once one strategy remained, and it comes back with the second one. The seam stays — it is what made
this measurable at all, and S-106 and S-109 depend on it.

The recency numbers say where a future retention strategy should start, but
they come from the same run that killed the marker, so they are a hypothesis
and not a mandate: shipping recency retention on this evidence would be the
same mistake in the other direction. It needs its own spec and its own run, at
a window where spans are not three turns long.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **`REDISCOVERED_READ` reads shell commands with a regex, not a parser.**
  `paths_read_by` splits on `&&`/`;`/`|`, keeps segments starting with a read
  verb, and takes tokens that look like paths. It will miss a read behind a
  variable, a subshell, or `xargs`, and it does not resolve `cd` prefixes — a
  `cd sub && cat x.py` records `x.py` while a `read_file` of `sub/x.py`
  records the full path, and the two will not match.
- **Path matching is suffix-based.** The agent reads one file two ways: a
  tool call gets the workspace-relative path, a shell command routinely
  carries the absolute one. Measured on the corpus, **35 of 51** distinct
  shell-read paths never matched a tool path by string equality — every one a
  rediscovery the label could not see. `same_file` therefore matches on a
  path-component suffix, which conflates two files with the same trailing
  components in different subtrees. Inside one checkout that is rare, and a
  missed rediscovery is the worse error for an eval whose pre-registered
  conclusion is "there is nothing here".
- **Writes are read from the span's messages, not from the event log.** A
  write recorded only as a pre-compaction `tool_call` outside the evicted span
  is invisible to the confound check. In production every write before a
  compaction is in the span, because compaction evicts a prefix.
- **A turn is credited with a read even if the file was already in context.**
  The label cannot tell "went back because compaction destroyed it" from
  "read it twice for its own reasons". The control for that is the write
  check, which is narrower.
- **"No intervening write" is checked per path, not per byte range.** An agent
  that edits line 400 and re-reads line 12 is doing genuine recovery work, and
  this scores it as legitimate. Conservative in the same direction as above.
- **A turn is scored, not a fact.** If a turn read five files and the run went
  back for one, the whole turn counts as needed-again — which is right for a
  retention decision that operates on turns, and wrong if the question were
  how much information was lost.
- **The 6,000-token window changes the runs, not just the compaction.** At
  that size the agent's behaviour is not the behaviour it has at 128K, so the
  loss rate measured here is a statement about compaction under duress, not
  about production. It is the only regime where there is anything to measure.
- **No A/B.** Deliberately: see "What this deliberately is not". If the oracle
  says there is an effect, the A/B is the next spec, with the random-retention
  arm as its control.
