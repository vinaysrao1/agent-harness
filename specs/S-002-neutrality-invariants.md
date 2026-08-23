---
id: S-002
title: Neutrality invariants and the TB2 conformance suite
status: Draft
lane: A
depends: S-001
effort: S
---

# S-002 — Neutrality invariants and the TB2 conformance suite

## Contract
`tests/conformance/` implementing N1–N8 and `tests/golden/` holding the frozen
artifacts plus `CHANGELOG.md`, one entry per deliberate change to the `CODING`
path, each naming the TB2 run id that justified it.

## Invariants
This spec *is* the invariants. It establishes the baseline; it does not change
behavior.

## Acceptance
1. All eight invariants pass at `b4fc55f`, with goldens generated from that
   commit.
2. Negative tests are the deliverable: an injected extra tool fails N2, an
   injected extra system-prompt line fails N1, an injected startup `exec` fails
   N3. An invariant never seen to fail is not known to work.
3. N1–N5 complete in under 5 seconds.

## Telemetry
none

## Rollback
`git revert`; the goldens are additive files.

## Neutrality argument
Lane A — tests only.

## Note
N6–N8 are defined against the replay corpus and therefore land with S-003.
N1–N5 are self-contained and land here.
