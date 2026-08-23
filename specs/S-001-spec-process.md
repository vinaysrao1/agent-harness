---
id: S-001
title: Spec process and traceability
status: Implemented
lane: A
depends: -
effort: XS
---

# S-001 — Spec process and traceability

## Contract
`specs/` holding one file per spec, a six-key front-matter schema
(`id, title, status, lane, depends, effort`), the lifecycle
`Draft → Accepted → Implemented → Verified` with `Rejected`/`Superseded` as
terminal states, a generated `specs/_index.md`, and the three traceability
rules T1–T3 as tests. `harness/specs.py` is the machine-readable half;
`specs/_template.md` is the human half.

## Invariants
All of N1–N8 hold trivially: nothing here is imported by the agent loop, no
tool is registered, and `harness/__init__.py` is empty so importing the package
does not import this module.

## Acceptance
1. `pytest tests/test_specs.py` fails if any spec at `Accepted` or later lacks
   a test naming it.
2. `python -m harness.specs` regenerates `specs/_index.md` deterministically —
   running it twice with no spec change produces no diff.
3. `specs/_template.md` exists and carries the sections a `harness run --spec`
   invocation would seed into the instruction ledger.

## Telemetry
none

## Rollback
`git revert`. Nothing depends on this module at runtime.

## Neutrality argument
Lane A by construction — a `CODING` run never imports `harness.specs`.

## Scope note on T3
The plan states T3 as "every new transcript event kind declares its spec id in
its payload". That is a runtime property of a payload and cannot be asserted
statically. What is enforced here is the checkable half: every event kind
emitted anywhere in the package is either grandfathered in
`LEGACY_EVENT_KINDS` (24 kinds, frozen at `b4fc55f`) or registered in
`EVENT_KIND_SPECS` against the spec that introduced it. The payload-level
assertion belongs to the first spec that emits a new kind, where a real payload
exists to assert against.

The scan resolves the `kind` argument whether it is passed positionally or by
keyword, whether `append_event` is reached through an attribute or a bare name,
and whether the value is a literal or a constant imported from elsewhere in the
package. Any shape it cannot resolve — `*args`, `**kwargs`, a computed
expression, an unresolvable name — raises rather than being skipped. That
distinction is the whole point: an earlier version read the second positional
argument unconditionally, so `append_event(agent_id, kind="x", ...)` was passed
over in silence and a new event kind could enter the package unaudited. A check
that fails to look is worse than no check, because it reports success.
