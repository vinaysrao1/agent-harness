---
id: S-000
title: One line, imperative, names the capability not the change
status: Draft
lane: A
depends: -
effort: XS
---

# S-000 — <title>

## Contract
What this introduces, in interface terms. One paragraph. If it adds a tool, a
config key, or an event kind, name it here exactly as it will appear.

## Invariants
Which of N1–N8 this preserves, and why. A Lane A spec asserts all eight hold
untouched. A Lane B spec names the ones it breaks and carries a `REFREEZE:`
block below.

## Acceptance
Numbered, executable criteria. Each one must be runnable as a command, because
these become the tests before the implementation exists — and, under the
dogfooding loop, the command the agent declares to `declare_verification`.

1. ...
2. ...

## Telemetry
The event kinds this emits, each registered in `harness.specs.EVENT_KIND_SPECS`.
Write `none` if it emits nothing.

## Rollback
How to disable this without reverting the commit — a config default, a profile
gate, or an explicit statement that rollback is `git revert`.

## Neutrality argument
Why a `CODING` run cannot reach this code, or — for Lane B — what changes on the
benchmark path and in which direction.
