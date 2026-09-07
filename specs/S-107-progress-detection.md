---
id: S-107
title: Progress and stuck detection
status: Implemented
lane: A
depends: S-003
effort: M
---

# S-107 — Progress and stuck detection

## Contract
A deterministic detector over live loop state, emitting `stuck_signal` with its
evidence. New module `harness/progress.py`. **No nudge, no behaviour change** —
see "Why this does not nudge yet".

## The measurement had to be done twice
Acceptance 4 asks for precision/recall on hand-labelled stuck runs *before
enabling this anywhere*. There is something better than hand labels: 727
recorded Terminal-Bench trials carrying the verifier's own verdict, which
cannot be fitted to the detector because it already exists.

The first pass got two things wrong, and both were the same mistake — an
instrument that could not see what it was measuring:

**Lift was quoted against the wrong base rate.** Long runs fail more (32.7%
under 20 calls, 77.8% over 80). A detector that needs a long run to fire
inherits that as apparent skill. The null "detector" `len(calls) >= 25` — no
logic whatsoever — scores **1.31×** against the global base rate.

**A failing command is not `is_error`.** That flag means the *tool* failed; a
command exiting non-zero is a perfectly good tool result carrying bad news.
`is_error` is set 8 times across the corpus, a non-zero exit appears 299
times, **and the two never coincide**.

Corrected, quoting lift against the base rate of runs long enough for each
detector to fire:

| detector | fires | precision | global lift | **matched lift** |
|---|---|---|---|---|
| `REPEATED_CALL` (identical call ×4) | 16 | 75.0% | 1.65× | **1.31×** |
| `CONSECUTIVE_FAILURE` (same head fails ×3) | 34 | 64.7% | 1.43× | **1.22×** |
| *no progress in 25 calls* | 141 | 60.3% | 1.33× | **1.01×** |
| **[null] `len(calls) >= 25`** | 230 | 59.6% | 1.31× | 1.00× |
| **[null] `len(calls) >= 50`** | 70 | 72.9% | 1.60× | 1.00× |

Two decisions reverse:

- **`NO_PROGRESS` is deleted.** At a matched base rate it is 1.01× —
  indistinguishable from the null. It shipped on its global lift, which was
  run length wearing a detector's clothes.
- **`CONSECUTIVE_FAILURE` is built.** It shipped *rejected* on "3 fires,
  33.3% precision, anti-predictive". Implemented as the plan actually
  describes it — non-zero exit, command *head* rather than the whole line —
  it fires 34 times at 1.22× matched, above what `NO_PROGRESS` ever managed.
  And 33.3% of n=3 was one failure in three; a 95% interval on that spans
  roughly 1%–91%. "A hypothesis the data refutes" was not a conclusion n=3
  could carry.

The remaining plan detector, *no `task_ledger` change across M turns*, is
still not built: 60% of runs never touch the ledger, so the condition is
permanently true for most of them.

`harness progress-audit` prints all of this including the null rows, so the
confound cannot be read as signal again.

## Why this does not nudge yet
The plan's acceptance 2 describes a single advisory nudge sharing
`MAX_NUDGES`. Two reasons that is not in this changeset:

**It is Lane B, structurally.** N5 hashes `NUDGE_SOURCES` *and* the count of
`nudges += 1` sites in `harness/loop.py`. A third nudge source breaks the
golden by construction — which is N5 working, not N5 being inconvenient. It
owes a TB2 run.

**The evidence does not justify it.** The best detector is **1.31× against a
length-matched base rate, on sixteen firings** — twelve failures and four
successes. That is a weak signal on a small n, and a nudge costs the run a
turn plus the context the reminder occupies. Whether that trade is worth it is
an empirical question about nudge cost, and nothing here measures nudge cost.

Worth stating plainly: on the corrected numbers, neither detector is strong.
They are shipped as telemetry because knowing how often they fire is cheap and
the event costs nothing; they are not shipped as a nudge because 1.31× on
n=16 is not a mandate.

So this ships the detector and the telemetry, and the nudge waits on evidence.
That is what acceptance 4 asks for in the order it asks for it. The event
carries everything a later Lane B change needs to decide.

## Invariants
Lane A. `stuck_signal` is an event; events do not reach the model. N1 (prompt),
N2 (tool surface), N5 (nudge sources), N7 (context timing) and N8 (cost) are
all untouched, and no golden is re-frozen. The detector is O(1) per turn
against a bounded window — it must not be possible for stuck detection to be
what makes a run slow.

## Acceptance
1. No model call; pure functions, like `diligence`.
2. `stuck_signal` carries the detector, its evidence, and the spec id.
3. Precision/recall reported against real outcomes before anything is enabled
   — done above, and reproducible from a command.
4. The detector costs O(1) per turn, asserted rather than assumed.
5. `CODING`'s behaviour is byte-identical: conformance stays at 48.

## Telemetry
`stuck_signal`, at most once per detector per run — a signal that re-fires
every turn once tripped would make "how often does this happen" unanswerable
from the log.

## Rollback
`git revert`. Nothing reads the event and nothing acts on it.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Neither detector is strong.** 1.31x and 1.22x against a length-matched
  base rate, on n=16 and n=34. They are worth their cost as telemetry and are
  not worth a nudge on this evidence; that is the whole argument, and it is
  thin on both sides.
- **The confound control is coarse.** "Runs at least as long as the shortest
  firing run" is a crude match. A proper control would stratify by length or
  regress it out; this catches the gross case (`no_progress` at 1.01x) and
  would miss a subtler one.
- **`REPEATED_CALL`'s stated mechanism is not what fires it.** The docstring
  says "the model re-issuing a call it has forgotten it made". Of its 16
  firings, 14 have file writes interleaved between the repeats, and three of
  the most-repeated are healthy edit-test loops on runs that **succeeded** --
  `cd /app && python solve.py` six times with three writes between. For
  `bash`, identical arguments are stateful: re-running the build after an edit
  is how iteration works. Whatever the 1.31x is measuring, it is not the
  described mechanism.
- **There is no window on `REPEATED_CALL`.** Four occurrences spread over 500
  calls fire identically to four in a row. In `CODING_REPO`, where runs are
  longer, `pytest -q` four times across 300 turns is near-certain, so the fire
  rate there should tend to 1.0 and the detector becomes another length proxy.
  Unmeasured: no repo-mode corpus with outcomes exists.
- **Nothing consumes `stuck_signal`.** By design — see "Why this does not
  nudge yet" — but it is the archetype's shape, and the only thing separating
  it is that the spec says so and names the evidence that would change it.
  If no Lane B change follows, this is a detector that fires into a log
  nobody reads, and it should be deleted rather than left as furniture.
- **The precision numbers are Terminal-Bench's, not repo mode's.** The corpus
  is 619 TB2 trials; the detectors are also live in `CODING_REPO`, where the
  work is longer and more edit-heavy and both thresholds may be wrong. Nothing
  measures that yet, because no repo-mode corpus with outcomes exists.
- **Thresholds are tuned on the corpus they are measured against.** 4 and 25
  were chosen by scanning that data, so the reported precision is optimistic
  by however much that overfits. The held-out discipline S-401 applies to
  tasks is not applied here.
- **`NO_PROGRESS` cannot see a write made through `bash`.** `bash cat > f` or
  `sed -i` counts as no progress. Measured direction: it inflates firing, so
  precision is a floor rather than a ceiling on that axis.
- **Neither detector survives a resume.** `ProgressMonitor` is loop state and
  is not persisted, so a resumed run starts blind — the same trade
  `written_data` makes, and worse here because a resumed run is by definition
  a long one.
- **`_seen` grows with distinct calls.** O(1) per call but O(n) memory in
  distinct `(tool, arguments)` pairs. A 500-turn run holds 500 keys of a few
  hundred bytes; nothing bounds it, and nothing needs to yet.
- **The audit scores whole runs, not the turns after firing.** "This run
  failed and the detector fired at 60%" is weaker than "the detector fired and
  the run made no progress thereafter". The latter is what would justify a
  nudge, and it is not measured.
