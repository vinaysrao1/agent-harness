---
id: S-110
title: Report a model refusal as a refusal
status: Implemented
lane: A
depends: S-001
effort: XS
---

# S-110 — Report a model refusal as a refusal

## Contract
A turn that comes back as a refusal is recorded as one: its own
`IncompleteReason`, its own placeholder text, a `model_refusal` event, and
`AgentResult.refused`.

## Invariants
Lane A. No completion path and no nudge source is added, so N5's frozen counts
hold; the prompt and tool surface are untouched (N1, N2); nothing new executes
(N3, N4). Conformance stayed at 48 passed with no golden moved.

## Acceptance
1. A refusal produces `incomplete_reason == "refusal"` and text that does not
   read as an answer. ✅
2. A provider returning nothing for no stated reason keeps its own, different
   text. ✅
3. `AgentResult.refused` is False by default and True after a refusing turn,
   and is distinct from `error`. ✅
4. The harness does not re-prompt around the refusal. ✅ — asserted
   structurally against the source, not just described.

## Telemetry
`model_refusal` (`{"spec": "S-110", "turn", "provider_stop_reason"}`),
registered in `EVENT_KIND_SPECS`.

## Rollback
`git revert`. `refused` defaults to False, so a caller ignoring it is
unaffected.

## The defect
On the Opus 5 benchmark run, three security-flavoured tasks —
`crack-7z-hash`, `vulnerable-secret`, `break-filter-js-from-html` — returned
at turn 1, in about 1.5 seconds, with `finish_reason: content_filter`, no
content and no tool calls.

An empty assistant turn is translated with a placeholder so a transcript
containing one stays replayable. The placeholder used was *"(provider returned
an empty assistant message)"* — and `looks_unfinished` accepts a non-empty
sentence with no open tasks as a finished run. So the loop finished
`completed`, with that sentence as the agent's final answer, and the trial was
scored 0.

Two of the three were solved by other models on the same harness, so this is a
real scored loss (~2 tasks). But the score is the smaller problem. The
harness told its caller *the agent finished and produced this text* when what
actually happened was *the model declined and no work was attempted*. Those
are different facts, and reporting one as the other is wrong independently of
any benchmark.

Verified by replaying all three recorded refusals through the fix.

## What this deliberately does not do
It does not re-prompt, reframe, or retry. The bug is that a decline was
**disguised as an answer**, not that the model declined. Building something to
talk a model out of a safety decision would be a different thing entirely, and
not one worth building here. A test asserts the refusal branch contains no
retry, reminder or continue, so this stays true by construction rather than by
intention.

The right escalation for a refusal is to the caller — which is what
`AgentResult.refused` and the event now make possible. In headless benchmark
mode there is no one to escalate to, so it is simply reported honestly and
scored as the failure it is.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Only the OpenAI-compatible adapter is covered.** The native Anthropic
  adapter has its own translation path and its own refusal shape
  (`stop_reason: "refusal"` with `stop_details`), which is untouched. No
  scored run has used that adapter, but the asymmetry is real.
- **A refusal mid-run is recorded but not acted on.** If turn 1 succeeds and
  turn 7 is refused, the flag is set and the loop continues as before. Whether
  a late refusal should end the run is a behavioural question this spec does
  not answer.
- **`refused` is not persisted to the run row**, so it is visible to the
  caller in-process and in the event log, but not to a later reader of the
  database without walking events.
- The Harbor bridge does not surface `refused`, so a benchmark trial still
  reports only a reward of 0 — correct, but it means the distinction is
  invisible in benchmark reporting.
