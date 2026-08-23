# Golden changelog

Every entry records a **deliberate** change to the `CODING` execution path —
the path Terminal-Bench scores. A golden moves only via the Lane B re-freeze
protocol: a `REFREEZE:` block in the spec naming the invariant and the expected
direction of effect, a TB2 run at the §2.4 statistical bar, then regeneration
with that run's id recorded here.

This file is the ledger that makes a year of incremental change auditable: for
any behavioral difference on the benchmark path, there should be a row here
naming the spec that caused it and the benchmark number that justified it.

Regenerate with `python -m tests.conformance.freeze`. Doing so without an entry
below defeats the point of the ledger.

| date | spec | commit | invariant(s) | reference run | provenance | reason |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-22 | S-002 | `b4fc55f` + S-001 | N1–N6 (initial freeze) | `jobs/pass1-opus5-v2/2026-08-21__16-26-32` — 62/89 = 69.7% | **Not a §2.4-qualifying measurement.** See note below. | Initial freeze. No behavior change: the goldens record the path as it already stood. |

## Note on the initial freeze

The reference run above is recorded for orientation, **not** as the frozen
baseline §12 measures against. It does not qualify under §2.4 on three counts,
each verified from the run's own artifacts:

- **It predates the commit.** `result.json` gives `started_at`
  `2026-08-21T16:26:33`; `b4fc55f` is dated `2026-08-22`. The run cannot have
  measured the path these goldens freeze — and `b4fc55f` is itself a
  behavioral change to the run path (the retry-ladder fix).
- **Wrong model.** `config.json` records `opus-5`; §2.4 pins the scored model
  to `openrouter/moonshotai/kimi-k3`.
- **Wrong k.** `n_total_trials: 89` with `max_retries: 0` is k=1; §2.4 fixes
  k=3 in advance.

A §2.4-qualifying baseline — kimi-k3, k=3, at the frozen commit — is owed
before S-002 can move from `Implemented` to `Verified`. Recording a
non-qualifying number as though it were the baseline would have made row 1 of
an attribution ledger unattributable, which is worse than having no row.

## Re-freezes

| date | spec | commit | golden(s) | reason |
| --- | --- | --- | --- | --- |
| 2026-08-22 | S-002 | `b4fc55f` + S-001 | `coding_assembled_system`, `coding_trailing_reminder` | **Fixture widening, not a behavior change.** Review found the assembled golden covered the skills *index* but not skill *bodies*, and did not cover the trailing system-reminder at all — both model-facing, both freely editable without any test noticing. The fixture now loads a skill body and the reminder is pinned as its own digest. No `harness/` behavior changed; verified by the tool and control-flow goldens staying byte-identical across the same edit. Lane A, no TB2 run required. |
