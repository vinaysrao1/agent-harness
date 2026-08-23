# Judge evaluation

Offline evaluation of an **independent verifier** for the harness's completion
gate. It exists because the current gate is circular: the model declares a shell
command that proves its own work, the loop re-runs that command, and exit 0 is
accepted. Agents routinely write both the solution and the check that blesses
it, so the gate passes on work the real grader scores 0.

Everything here runs against recorded runs — no benchmark execution required.

## The corpus

One record per trial whose self-verification gate **passed**, labeled by what
the grader said afterwards:

| label | meaning | count |
|---|---|---|
| `true_pass` | gate passed, grader agreed | 108 |
| `false_pass` | gate passed, grader scored 0 | 33 |

Pooled from two full Terminal-Bench 2.0 runs using different models, which is
what makes it possible to check that a result generalises rather than fitting
one model's quirks.

**The corpus and cached verdicts are deliberately not committed** (see
`.gitignore`): they are derived run data containing agent solutions to benchmark
tasks, and publishing them risks contaminating the benchmark.

### Regenerate

```bash
python tools_eval/judge_eval/build_corpus.py <jobdir> [<jobdir> ...] \
    -o tools_eval/judge_eval/corpus.jsonl
```

Workspaces are not preserved (containers are deleted after a trial), so final
files are reconstructed by replaying `write_file`/`edit_file` calls **and bash
heredocs** — heredocs are the majority (188 files vs 136), so omitting them
halves the evidence. Files written *programmatically* by a script cannot be
recovered at all; this is the main reason offline results are directional
rather than conclusive.

## Running a sweep

```bash
export OPENROUTER_API_KEY=...
python tools_eval/judge_eval/run_judge.py --only K_grader_actions
python tools_eval/judge_eval/analyze.py K_grader_actions      # held-out + CIs
python tools_eval/judge_eval/sweep_filters.py K_grader_actions # free re-scoring
```

Verdicts are cached in `verdicts.json` keyed by config, so re-running is cheap.
**The cache key does not include config parameters** — change a config's
`extra_body` or model and you must delete its cached rows, or you will silently
score stale verdicts.

## What was measured

Pre-registered bar, fixed before any result was inspected: **recall ≥ 40% on the
33 false passes, false-rejection ≤ 5% on the 108 true passes.** Nothing has
cleared it yet.

| config | recall | false-reject | precision |
|---|---|---|---|
| adversarial framing | 90% | 90% | — |
| calibrated, verdict-first | 51.5% | 13.0% | 54.8% |
| calibrated, defect-first, derived verdict | 51.5% | 19.4% | — |
| grader framing (`K`) | 48.5% | 8.3% | 64.0% |
| `K` + confidence ≥ 0.6 | 39.4% | 4.6% | 72.2% |
| `K` + `evidence_sufficient` | 21.2% | 0.9% | 87.5% |

Findings worth not rediscovering:

- **Framing dominates.** "Predict what a hidden grader would test" beats "audit
  this work for defects". Telling the judge to default to *not verified* when
  uncertain makes it reject 60–90% of **correct** work — uncertainty is its
  normal state, so that instruction collapses into always saying no.
- **Let it abstain.** `evidence_sufficient` lets the judge decline cases needing
  execution, and removes nearly all false rejections.
- **State the defect before the verdict**, but keep the model's *own* verdict.
  Deriving the verdict from the defect text costs ~11 points of false-rejection:
  it removes the model's ability to note a concern while still voting verified.
- **Ensembling does not work** (tried three times: cross-model, unions, 2-of-4
  and 3-of-4 votes). Errors are correlated — a second model family caught a
  strict subset and found nothing new. 86% of false rejections recur across
  completely different prompts, so they are systematic, not noise.
- **The ceiling is inspectability, not judgement.** Defects visible in artifacts
  are caught 75% of the time; defects that are runtime properties (does it
  compile both ways, does it hit an accuracy threshold, does it converge) only
  15%. More prompting will not move that — the judge needs to execute code.
- A deterministic "agent left its own test failing" detector was built and
  **underperformed the model judge** (18% recall at 16% false-rejection). It was
  generalised from two hand-read examples; measuring first is the only thing
  that stopped it shipping.

`stale_failure.py` / `eval_stale.py` are retained as the record of that negative
result. Note that the shell exit code cannot be trusted for this:
`cmd; echo EXIT=$?; grep ...` exits with the grep's status, so a failing run
reports `exit code: 0` while its stdout says `EXIT=1`.
