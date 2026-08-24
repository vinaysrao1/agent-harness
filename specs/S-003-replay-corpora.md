---
id: S-003
title: Replay corpora and the context-timing invariants
status: Implemented
lane: A
depends: S-002
effort: S
---

# S-003 — Replay corpora and the context-timing invariants

## Contract
A frozen corpus of real transcripts, replayable through `ContextManager`
without a model, and N7/N8 defined against it. Plus the existing `exec_capped`
corpus, already promoted as N6 by S-002.

## Invariants
Lane A. Nothing in `harness/` changes; this is test infrastructure and frozen
data.

## Acceptance
1. `replay_corpus()` reconstructs prune and compaction turn indices
   deterministically.
2. N7 and N8 are implemented against it.
3. Corpus regeneration is a documented, reproducible command — see
   `tests/golden/README.md`.

## Telemetry
none

## Rollback
`git revert`; the corpus and trace are additive files.

## Neutrality argument
Lane A — no `harness/` module imports anything under `tests/`.

## Why this matters more than its size suggests
**Compaction has fired zero times across 645 real trials.** A Terminal-Bench
run therefore cannot tell you whether a change to the condenser broke it: the
mechanism never runs. The same is true of any retuning of the context
constants — the benchmark exercises pruning but not compaction, and it exercises
neither in a way whose *timing* is observable in the score.

That makes this corpus the only instrument for S-105 (condenser seam), S-109
(real token counting) and S-102 (file cache). S-109 is the sharpest case: every
threshold in `harness/context.py` was tuned against the `chars / 4` estimator,
so replacing the estimator retimes every run. N7 turns that from a footnote
into a failing test — verified, by swapping the estimator to `chars / 3` and
watching N7 fail.

## Design: shapes, not content
The corpus stores per message its role, exact character counts, and whether it
carries a tool result. Content is reconstructed as filler of the same length.

Sufficient, because every decision N7 and N8 pin depends on roles, ages,
tool-result-ness and token counts, and the estimator counts characters — so
filler of equal length reproduces the counts exactly. Nothing in the decision
path reads what the text says.

Necessary, because real transcripts contain agent solutions to benchmark
tasks. `jobs/` is gitignored for that reason, and a frozen artifact has to be
committed to be frozen. An integrity test asserts the corpus contains no free
text.

## Design: scaled windows
Each trial is replayed at 128K (production) and at 64K/32K/16K.

At the production window the replay agrees with production on the two facts
that are observable: pruning fires (on 13 of 25 trials) and compaction does
not — matching the direct observation of zero compaction events in 645 trials.
That agreement is asserted as a test.

It is agreement on those two facts, **not** a reproduction. Production turn
indices are not recorded anywhere, so nothing checks that pruning fires at the
*same* turns; and the replay's assembled prompt, though real, comes from the
conformance fixture rather than from each trial's own run. Calling this
"reproduces production exactly" would overstate it.

The smaller windows exist because compaction has to be exercised by something:
a corpus on which a mechanism never fires pins nothing about it. Scaling the
window rather than inventing transcripts keeps the message-size distribution
real.

## Defects review found in this spec
- **The first corpus contained no tool-role messages at all.** Tool results are
  persisted under their own `tool_result` event kind, not inside `message`, so
  reading only `message` events produced a transcript with nothing prunable —
  pruning could not fire at *any* window, and N7 would have frozen a trace of a
  mechanism that structurally could not run. `Orchestrator._replay_lead_transcript`
  replays both kinds for exactly this reason; the extractor now mirrors it.
- **`freeze` silently reset the trace.** The replay trace guards N7 *and* N8,
  yet it alone was rewritten unconditionally — no diff, no changed-count, and
  therefore no CHANGELOG reminder — so a documented, zero-argument
  `python -m tests.conformance.freeze` reset a failing invariant and exited 0.
  An invariant any freeze invocation quietly resets is not an invariant. It now
  reports which keys moved and is a nameable re-freeze target.
- **N8 could not see the prompt.** The replay used a 6-character placeholder
  system prompt where production sends ~2,400 characters every turn, so the
  one thing m10 §2.2 explicitly assigns N8 — "guards silent prompt growth" —
  was the one thing it could not do.
- **The tool-call length round-trip was not exact.** The extractor stored
  `len(json.dumps(args))` while the estimator charges
  `len(name) + len(repr(args))`. Since "filler reproduces the counts exactly"
  is the entire justification for storing shapes rather than content, an
  inexact round-trip undermined the design's premise.
- **N7 initially pinned only *when* pruning fires, not *how much* it sheds.**
  `PRUNE_TARGET_FRACTION` changes the shed volume while leaving the firing
  turns untouched, and shedding more only makes the assembly smaller — which N8
  deliberately permits, since cheaper is normally an improvement. Destroying
  more context is not an improvement. The trace now records shed volume.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Content-dependent behavior is not pinned.** A future condenser that decides
  what to keep by reading the text would not be covered by a shape-only corpus.
  S-105's pivotal-retention criteria are content-dependent and will need their
  own fixture.
- The corpus cannot be rebuilt from a clean checkout — it derives from `jobs/`,
  which is deliberately not in the repository. The frozen artifact is the
  contract; regeneration requires local runs.
- N8 is one-sided by design (cheaper passes). A change that reduces tokens
  while degrading quality is caught only insofar as it moves the N7 trace.
- **Compaction is exercised only at scaled-down windows.** No trial compacts at
  128K or 64K; the 10 and 15 compacting trials live at 32K and 16K. So S-105 is
  covered at 4–8× shrunk windows with a fixed 2,000-char stand-in summary, not
  under production conditions. The corpus cannot fix this — the harness has
  genuinely never compacted in production.
- **The summarizer is a constant, not a condenser.** S-105's acceptance
  criterion (1) — "the default condenser reproduces today's output
  byte-for-byte on the replay corpus" — is not satisfiable against N7 as built,
  because the replay injects a fixed string in place of the condenser and never
  runs it. S-105 will need to extend this harness, not merely consult it.
- **N7 is exact-equality; m10 §2.2 and S-109 both say "±1".** Exact is stricter
  and, since the replay is deterministic, achievable — but S-109 cannot satisfy
  its own acceptance criterion against this test without a re-freeze. The
  documents disagree; this one is deliberately the stricter, and S-109 should
  expect to re-freeze rather than expect tolerance.
- The corpus records no provenance for *which* runs produced it beyond the
  trial names, and `jobs/` is not in the repository, so that provenance is not
  recoverable. See `tests/golden/CHANGELOG.md`.
