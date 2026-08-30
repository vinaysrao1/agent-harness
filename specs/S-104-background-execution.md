---
id: S-104
title: Background execution
status: Implemented
lane: A
depends: S-005
effort: M
---

# S-104 — Background execution

## Contract
`bash(..., run_in_background=true) -> handle`, `bash_output(handle)`,
`kill(handle)`. Jobs are harness-tracked, deadline-aware and reaped. New module
`harness/jobs.py`; `ToolDeps.jobs` and `ToolRegistry.jobs`, both per agent; new
event kind `background_job_abandoned`. All three tools in `CODING_REPO` only.

## Invariants
Lane A, and the argument is structural rather than a default. `bash`'s
`run_in_background` parameter appears in the schema **only when a job registry
is supplied**, because N2 pins the tool surface — an unconditional argument
would change `CODING`'s digest and make this Lane B. Repo mode substitutes its
own `bash` factory *by reference* (`repo_bash_factory`), not by list position,
which would silently pick up whatever ended up first. Conformance stayed at 48.

## Acceptance
1. Wind-down names unreaped jobs; the landing turn kills them, and so does
   `AgentLoop.run`'s `finally`. ✅ — `TestNoRunEndsHoldingAJob`,
   `TestReapingHappensOnEveryExit`. Wind-down deliberately does not kill: the
   build may be the work, so it tells the model while there is still time to
   act. Landing alone was not enough — a run that errors, pauses on budget, or
   is cancelled never reaches it, and a subagent that finishes normally never
   arms it at all. There is no backstop underneath: `HarborSandbox.stop` is a
   documented no-op, so on the benchmark path a leaked build keeps compiling
   *into the grading phase*, competing with the verifier for the container.
2. Background output goes through the same spill path as foreground. ✅ — it
   is an ordinary tool result, so `ToolRegistry.dispatch` truncates and spills
   it exactly as it does any other.
3. A job started and never polled is surfaced as `background_job_abandoned`. ✅
   — with a control asserting a polled job is *not* reported, so the signal
   carries information.
4. Foreground exec semantics are preserved — N6. ✅ — `TestForegroundIsUnaffected`.

## Telemetry
`background_job_abandoned` (registered in `EVENT_KIND_SPECS`). Whether the model
actually comes back for its build is the question this feature exists to answer,
and a mechanism nobody uses should be visible as unused rather than counted as
shipped.

## Rollback
`git revert`. Every integration point takes `jobs=None`; without a registry the
schema, the handler and the reaping are all exactly as before.

## Why this is built on `exec` alone
A background primitive on the `Sandbox` ABC would need a correct implementation
in every backend and a new contract for handles that outlive a call.
Redirecting to a file under `/tmp/.harness-jobs` and polling it needs neither,
behaves identically under Docker and Local, and keeps the process tree inside
the sandbox where the existing orphan-kill machinery already reaches it.

`tail -c +N` resumes each poll from where the last one stopped: re-sending a
growing build log every time would spend the context window on text the model
has already read.

## The shell defects worth recording
Every one of these was a silent degradation: the job started, the tool reported
success, and the harness's own telemetry said the feature worked.

**Isolating the process group cannot be asserted, only measured.** A build
spawns compilers; signalling only the job's own shell leaves them running. Two
mechanisms were tried and both failed silently. `setsid` is a Linux tool,
absent on macOS, and a plain-subshell fallback meant every host-side run took
the fallback. `set -m` (POSIX job control) looked like the portable answer, but
**dash turns it off** when there is no controlling terminal — which is every
`docker exec` without `-t`, i.e. the production path — printing `can't access
tty; job control turned off` to stderr and carrying on. So `start_command` now
uses whichever mechanism is available, **reads the real group id back**, and
`_start_background` **refuses the job** unless `pgid == pid`. A job that
finished before its group could be read reports `gone` and is accepted; one
that is alive without its own group is rejected rather than tracked as killable.
`TestUnderTheProductionShell` runs five of these in the Docker image, where
`/bin/sh` is dash; with `setsid` disabled, all five fail — though as one
observation, not five: they fail at the same refusal, so the class is a canary
rather than five independent checks. The measurement itself is pinned
separately, on both the `/proc` and the `ps` branch, by starting a child that
is *not* isolated and asserting its group id differs from its pid. Every
earlier test compared `pgid` to `pid`, which `echo $pid` also satisfies, so the
whole mechanism could be deleted and the suite stayed green.

**`{ cmd; } > out; echo $? > rc` never records the exit code.** An `exit 3`
inside a brace group terminates the *outer* shell, so the sentinel is never
written and a failing job polls as "still running" forever — or inherits a
stale sentinel and reports a previous job's success. A subshell contains it.

**The closing parenthesis needs its own line.** The command's last line may be
a heredoc terminator or a `#` comment; appending `)` to it is a syntax error,
which presented as a job that started, produced nothing, and polled as running
forever.

**`mkdir -p ... && ...` is not a guard.** A `;` after an `&&` list ends it, so
a failed mkdir still launched the job — with no output file, no sentinel, and a
poll that said "still running" forever. It is `|| exit 1`.

**The poll offset must come from the shell, not from the decoded string.** A
non-UTF-8 byte decodes to one U+FFFD and re-encodes to three, and the harness's
own truncation marker was being counted as job output. Either desynchronised
the offset permanently; the symptom was a job that "finished with nothing to
show" while the rest of its log sat in a file nobody would read again. The
offset is now `wc -c` on the file, carried back in a `__HARNESS_OFFSET__`
sentinel.

**"Live" meant "not killed".** Nothing marked natural completion, so a job
polled all the way to its exit code stayed live forever: announced at wind-down
as still running, and signalled at landing on a pid the kernel had long since
reused. `Job.exit_code` is recorded on the poll that observes it and `live()`
excludes it.

Most of these were found by a surviving mutation rather than by review, and
several only after a test was tightened enough to observe the property — the
kill test asserted after 0.6s about a child that slept 30, which was true
either way.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Job output files are never cleaned up.** `/tmp/.harness-jobs` accumulates
  one `.out` and one `.rc` per job for the life of the sandbox, exactly as
  S-201's shadow store does. Deleting them at reap would break a final
  `bash_output` in the landing turn, which is the moment the model is most
  likely to want them.
- **Not promoted to `CODING`, and the plan says it matters there.**
  `build-pov-ray` is a Terminal-Bench task where a build longer than the exec
  cap currently returns nothing. Promotion is Lane B, needs a TB2 run, and
  `CODING` is at its 15-tool cap — so it also needs two tools removed or
  merged. No evidence gathered either way.
- **Nothing bounds how many jobs an agent may start.** A loop that starts a job
  per turn will fill the sandbox with processes; the registry tracks them but
  imposes no limit.
- **The wind-down notice fires once.** It is emitted on the single turn the
  band arms, so a job started *after* wind-down is never announced — only
  killed at landing.
- **`kill` sends TERM then KILL 0.1s apart.** A job that traps TERM and needs
  longer to flush is killed mid-write. The delay is arbitrary and unmeasured.
- **A job that cannot be isolated is refused, not degraded.** If the shell
  reports a `pgid` that is neither the pid nor `gone`, `_start_background`
  returns an error instead of a handle. This is the right failure, but it means
  a host whose `/bin/sh` supports neither mechanism has no background
  execution at all rather than a partial one. No such host is known.
- **The reap is best-effort and unverified.** `_reap_background_jobs` swallows
  every exception from `sandbox.exec` and marks the job killed regardless; it
  does not confirm the process is gone. A sandbox already torn down when the
  `finally` runs will silently reap nothing. It must swallow them: it runs
  inside `run`'s `finally`, where anything raised would replace the run's real
  outcome.
- **A finished job the model never polled is still reaped as if running.**
  `finished` is set by the poll that observes the sentinel, or at start for a
  job that was already over. Everything in between is unobservable without
  polling, so the reap signals a pid that may have been recycled. The signal
  is `kill -TERM -<pid>`, so a recycled *group leader* would be hit; nothing
  bounds that beyond the container's pid space.
- **The teardown reap can extend a run that has already blown its deadline.**
  Each live job costs up to 10s of `sandbox.exec`, sequentially, inside the
  `finally` that Harbor's `wait_for` is waiting on. A run killed for lateness
  gets later.
- **A poll can return the same bytes twice.** `poll_command` reads `wc -c`
  before `tail`, so anything written between the two is counted in the new
  offset *and* re-sent on the next poll. Duplication, not loss — the opposite
  failure to the one the offset exists to prevent.
- **Polling is pull-only.** A job that finishes and is never polled again has
  its exit code recorded nowhere except the sentinel file; the abandonment
  event says it was never read, not what it produced.
