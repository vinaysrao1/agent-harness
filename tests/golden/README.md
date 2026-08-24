# Frozen artifacts

Goldens pinning the `CODING` execution path — the path Terminal-Bench scores.
A change here is a change to what the benchmark measures. See `CHANGELOG.md`
for the ledger and the re-freeze protocol.

| file | invariant | what it pins |
| --- | --- | --- |
| `coding_base_prompt.sha256` | N1 | the base rules string (narrow digest, so failures localize) |
| `coding_assembled_system.sha256` | N1 | `context.assemble()[0]` — rules + skill bodies + memory block + ledger |
| `coding_trailing_reminder.sha256` | N1 | the trailing system-reminder, which rides in `messages`, not the system string |
| `coding_tools_subagent.sha256` | N2 | the 13-tool subagent registry: names, order, schemas |
| `coding_tools_lead.sha256` | N2 | the 15-tool lead registry (what a benchmark run presents) |
| `coding_control_flow.sha256` | N5 | declared gates, nudge sources, and the tuned budgets |
| `replay_corpus.jsonl` | N7, N8 | 25 real trials as **shapes** — roles and sizes, no content |
| `replay_trace.json` | N7, N8 | when pruning and compaction fire, and tokens per turn |

## Regenerating

Digests, all or by name:

```bash
python -m tests.conformance.freeze
python -m tests.conformance.freeze coding_assembled_system.sha256
```

Only files whose value actually changed are rewritten, and each change prints
`old -> new`. Regenerating everything for a change that affects one hides
unrelated drift under an approved change.

The replay corpus is rebuilt from run data, which is **not** in the repository
(`jobs/` is gitignored — it contains agent solutions to benchmark tasks, and
publishing those risks contaminating the benchmark). Rebuild it only when you
have runs locally:

```bash
python -m tests.conformance.build_replay_corpus jobs -o tests/golden/replay_corpus.jsonl
python -m tests.conformance.freeze          # re-traces and rewrites replay_trace.json
```

The corpus stores roles and character counts, never text. That is sufficient
because every decision N7 and N8 pin depends on message roles, ages,
tool-result-ness and token counts — and the estimator is `chars / 4`, so
filler of the same length reproduces the counts exactly.

**What this does not pin:** content-dependent behavior. A future condenser
that decides what to keep by reading the text would not be covered.
