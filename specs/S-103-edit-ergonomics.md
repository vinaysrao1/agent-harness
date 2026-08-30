---
id: S-103
title: Edit ergonomics
status: Implemented
lane: A
depends: -
effort: S
---

# S-103 — Edit ergonomics

## Contract
(a) On an `old_string` mismatch, name the difference and show the nearest
candidates with line numbers. (b) `multi_edit(path, edits[])` — several edits to
one file, applied in order, atomically.

New module `harness/edits.py`; new tool `multi_edit`, in `CODING_REPO` only.

## Invariants
Lane A. `CODING`'s tool set is byte-identical — `multi_edit` ships in
`REPO_TOOL_FACTORIES`, which is `CODING_TOOL_FACTORIES` plus one. The mismatch
diagnostic is appended to an existing error message whose opening text is
unchanged, so N1/N2 hold and every caller matching on it still matches.
Conformance stayed at 48; no golden moved.

## Acceptance
1. Whitespace-only and indent-width mismatches are always identified as such by
   name. ✅ — `TestTheMismatchIsNamed`, covering indent width, tabs-vs-spaces,
   trailing whitespace and CRLF, plus a control that a genuine content
   difference is *not* blamed on whitespace.
2. A failing edit in `multi_edit` leaves the file byte-identical. ✅ —
   asserted on the bytes, through the real tool and a real sandbox.
3. One `syntax_check` runs after the whole `multi_edit`, not per edit. ✅ —
   asserted on the calls, plus a control that a *failed* batch checks nothing.
4. Mismatch diagnostics are bounded to a fixed character budget. ✅ —
   `DIAGNOSTIC_BUDGET`, asserted against a 4,000-line file.

## Telemetry
none of its own. `multi_edit` emits `syntax_check` through the existing path,
tagged `tool="multi_edit"`.

## Rollback
`git revert`. The diagnostic is additive to an error string; removing
`multi_edit` from `REPO_TOOL_FACTORIES` restores the previous tool set exactly.

## Why "not found" was the wrong answer
A rejected edit costs a turn, and under a wall clock that is expensive. The
information needed to fix it is already in hand — the harness *just read the
file* to attempt the edit — so answering "old_string not found in file" and
stopping throws it away and invites a blind retry with the same whitespace.

The common failure is not "that text isn't there". It is "that text is there
with a different indent width", or a tab, or a trailing space, or CRLF. Each is
a one-line correction if named and a guessing game if not, which is why
`classify_mismatch` returns the *specific* difference and only falls back to a
character diff. Whitespace is rendered visible (`·`, `\t`) **only** when
whitespace is the difference: doing it always would make ordinary diffs harder
to read for no gain.

Silence is a valid answer. Below a similarity floor the diagnostic is empty —
pointing at an unrelated stretch of code implies it was meant, which is worse
than saying nothing.

## Why atomic, and why one check
`multi_edit` applies every edit to an in-memory copy and writes once. A failure
part way through leaves the file untouched *by construction*: there is no
partial state to roll back because nothing was written. The alternative — a
model reasoning about which half of a three-hunk change landed — is a model
that has lost the thread.

One syntax check per batch, not per edit, because the intermediate states are
*expected* to be broken: a rename touching three call sites is invalid after
the first. Checking per edit would report three failures that are not true of
the result, and spend the deadline three times to do it.

## Tool-count discipline
Layer 1 caps `CODING` at 15 tool specs (13 factories + 2 lead-only) and it is
already at 15. `multi_edit` therefore ships in repo mode only. Promoting it is
a Lane B change requiring a TB2 run *and* the removal or merging of an existing
tool — tool-surface growth degrades selection quality measurably on
non-Anthropic models, and the scored model is one.
`test_S103_coding_stays_at_its_cap` fails if that budget moves.

## What the review found
Four defects, three of them the same shape as the ones this codebase keeps
producing. Recorded because the pattern is more useful than the fixes.

**The CRLF branch could never fire.** `nearest_candidates` built windows with
`"\n".join(splitlines())`, which strips `\r`, so the `found` side was always
LF-normalised. A CRLF file was therefore reported as *"the text itself
differs"* at **100% similar**, and the block shown back to the model was
`\r`-stripped — so copying it into a retry failed again. A confident wrong
answer, strictly worse than the bare "not found" this spec exists to replace.
The test that should have caught it called `classify_mismatch` with a `found`
containing CRLF: a value the real path cannot produce. Green throughout. That
is the archetype, inside the spec whose rationale cites the archetype.

Windows now keep their terminators, the message names *which side* has CRLF
(the single constant was backwards half the time), and the test goes through
`describe_mismatch`.

**It was catastrophically slow on the benchmark path.** `edit_file` is in
`CODING`. Scoring every window with a full `SequenceMatcher.ratio` measured
**424 seconds** on a 5,000-line file and 200 seconds on one minified line —
synchronous on the event loop, with no `await` for the deadline to fire
through, and producing a one-sentence diagnostic because every candidate
exceeded the budget. One failed edit could consume an entire benchmark trial.
Now 0.08 seconds.

The fix is the *shape* of the search — anchor by dictionary lookup, rank with a
linear `quick_ratio`, pay the quadratic `ratio` only for the three survivors
over a sampled prefix. File-size and `old_string`-size caps were written first
and then **deleted**: measured with them disabled the timings did not move, so
they were guards that never fire dressed as safety.

**`multi_edit` was invisible to the diligence detectors.** `record_written_data`
branched on three tool names; the fourth fell through and recorded nothing, so
`_lint_tautology`, `_lint_no_execution` and `_lint_self_authored_checker` went
silent for anything written through it — while `multi_edit`'s own description
tells the model to *prefer* it over `edit_file`. An empty finding list is
indistinguishable from a clean one.

**Two of the tests written to pin the fixes were themselves defective**, and
both failure modes generalise:

- A *"does not hang"* test hangs when the property breaks. Its failure took
  minutes instead of milliseconds, which made the whole mutation suite
  unrunnable. Replaced by asserting the **input length** to the quadratic step,
  with the comparison stubbed so it is never actually run.
- Two tests imported the constant they were pinning. Raising
  `_RATIO_SAMPLE_CHARS` to 10^9 left one asserting `90,000 <= 10^9` and
  passing. Bounds in tests are now literals.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **`apply_patch` is deliberately absent.** A third edit format dilutes tool
  choice against a model post-trained on string-replace. Revisit only with
  S-401 evidence.
- **The budget can be exceeded by up to 3 characters.** The accounting charges
  one character per joining newline, and that is correct — but no test pins it,
  because the overshoot is only visible when a rendered block's length lands
  *exactly* on the remaining budget, and a sweep steps over it. Contriving that
  input would be more complexity than a 3-character overshoot on a 1,200
  character bound is worth. Stated rather than tested.
- **The classifier is single-cause.** It names the first difference it
  recognises; a mismatch that is both an indent change and a content change
  reports the indent. Deliberate — whitespace is the more common and more
  actionable cause — but it can point at the smaller problem.
- **Anchoring requires the first line to match.** A mismatch whose *first* line
  differs in content finds no anchor and produces no diagnostic at all. That is
  the cost of making the search cheap, and it is the honest silence rather than
  a wrong answer, but it means the feature is quietest exactly when the model
  has mistyped the opening line.
- **`multi_edit` is one file per call.** A rename spanning three files is still
  three calls. Multi-file atomicity needs a different rollback story than
  "nothing was written yet".
- **Atomicity is about *edit* failures, not *write* failures.**
  `write_workspace_file` opens `"w"` — truncate in place, no temp-and-rename —
  so `ENOSPC` after truncation leaves a partial file and no "nothing was
  written" note. Shared with `edit_file`, so not a regression, but the module
  docstring's "written once, or not at all" is true only of the tested path.
- **`resume_task` strips `multi_edit`.** It does not thread profiles, so a
  resumed `CODING_REPO` run replays `multi_edit` calls into a registry that
  lacks the tool. Pre-existing mechanism; S-103 adds a new instance of it.
- **`edit_file`'s behaviour on `CODING` did change**, even though its tool set
  did not. A failed edit now returns up to ~1,200 extra characters. N1 pins the
  system prompt and N2 the tool specs; neither observes tool *results*, so the
  invariants passing is not evidence about this. The change is believed
  harmless — strictly more information on a path that previously returned one
  sentence — but it is a behaviour change on the scored path and is recorded as
  one.
- **No measurement.** The claim that better diagnostics reduce wasted turns is
  untested. S-401 can measure it — turns-to-first-edit on a fixed suite, with
  and without — and has not.
