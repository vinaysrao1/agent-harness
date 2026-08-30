---
id: S-102
title: File cache and read staleness
status: Implemented
lane: A
depends: S-101
effort: S
---

# S-102 — File cache and read staleness

## Contract
A per-agent `ReadLedger` holding a bounded LRU content cache and a record of
what was read at what version. `read_file` serves from the cache; `edit_file`
and `multi_edit` **warn** when editing a file that was never read, or was read
before something else changed it. New module `harness/reads.py`;
`ToolDeps.reads`, constructed once per agent in `_build_registry`.

## Invariants
Lane A, and the argument is **not** "the default is `None`". Every production
caller passes a ledger, so a `reads=None` default proves nothing about the
benchmark path — a first version rested on exactly that and was not neutral.

The real argument is that the two halves are separated. The **cache** changes
no bytes and runs everywhere. The **advisory** changes tool-result bytes and is
gated on `profile.enables("read_staleness")`, which `CODING` does not.
`TestTheProfileDerivationInsideExecute` runs a real `run_task` under `CODING`,
`profile=None` and `CODING_REPO` and asserts the first two produce
`"edited solve.py"` exactly. Conformance stayed at 48; no golden moved.

## Acceptance
1. Repeat `read_file` on an unchanged file performs no sandbox call. ✅ —
   asserted on the *calls* the sandbox receives, because elapsed time cannot
   distinguish a cache hit from a fast filesystem.
2. Staleness never rejects an edit in any profile. ✅ — every staleness test
   asserts the file's new content as well as the advisory, so a version that
   warned *instead of* editing would fail.
3. The advisory contains no promise pattern and does not end in `?`. ✅ —
   asserted directly against `looks_unfinished`, not by inspecting the string.
4. N7 holds: caching does not change prune timing. ✅ for the cache, which
   alters only how often the *sandbox* is called. **Not evidenced by
   conformance**, though: the replay corpus rebuilds every tool message from a
   stored character count and dispatches no tools, so N7 and N8 would stay
   green with an advisory appended to every result. The advisory's absence on
   the benchmark path is asserted directly instead, by running a task.

## Telemetry
none. A staleness advisory is already visible in the tool result it is appended
to, and a cache hit is the absence of an event rather than one.

## Rollback
`git revert`. Passing `reads=None` — the default everywhere — restores the
previous behaviour exactly.

## Why the cache does not revalidate
Checking a file's mtime before serving it would be the sandbox call the cache
exists to avoid, so correctness comes from invalidation instead:

- a write to a path drops that path;
- an **append** drops it too, because the harness knows a change happened but
  not the resulting whole, and claiming to know it would be worse than
  admitting it does not;
- **any `bash` command drops everything.**

That last one is deliberately blunt. A shell command can rewrite any path
without the harness learning which, so after one every cached file is a belief
rather than knowledge. It costs re-reads — agents run `bash` constantly, so in
practice the cache mostly serves read-then-read-again within a turn — but a
cache that survives an unobserved write hands the model a file that no longer
exists in that form, and it will edit against it.

## Why the ledger only warns
Rejecting a stale edit would add a new failure mode under a wall clock, and the
harness's own belief about staleness is approximate for exactly the reason
above. The order here is the one this codebase used for the syntax check: warn,
measure the false-positive rate, and only then consider enforcement as its own
Lane B spec.

The advisory carries `format_syntax_failure`'s two constraints. It **names the
harness as the author**, so the model does not debug a message it never
produced. And it contains no promise pattern and never ends in a question mark,
because a model quoting it back into its final answer would otherwise trip
`looks_unfinished` and have a finished run refused as incomplete.

A harness write counts as knowledge: `note_write` records both the content and
its version. Without the version half, writing a file and then editing it warns
that the file "was edited without being read first" — on content the agent
authored one call earlier. An advisory that fires on the most ordinary sequence
there is becomes noise, and then becomes ignored.

## What the review found
Nine defects. Two were the same failure this codebase keeps producing, and one
of them broke neutrality outright.

**The benchmark path was not byte-identical.** A file created by a `bash`
heredoc has never been "read", so every later `edit_file` carried a 178-byte
advisory — and heredoc-create-then-edit is the *modal* write path on
Terminal-Bench. The spec's neutrality argument was that `reads` defaults to
`None`; every production caller passed a ledger, so the default described a
caller that does not exist. The cache and the advisory are now separated: the
cache changes no bytes and runs everywhere, the advisory is gated on a profile
capability that `CODING` does not have.

**The cache had three deterministic ways to serve content that was not on
disk**, none needing concurrency or a subagent:

- keyed on the raw argument, `a.py` and `./a.py` were two entries, so writing
  through one spelling left the other stale until the next `bash`;
- `edit_file` invalidated the path but never recorded its own result, so a
  second edit warned that the file "changed since it was last read" — blaming
  an external change for the harness's own edit one call earlier;
- a subagent shares the lead's sandbox but had its own ledger, so a child's
  write left the parent serving pre-spawn bytes.

The last one was the design error: the cache is a belief about the
*filesystem*, which is shared, while the versions are what *one agent* has
seen. They now have different lifetimes — `FileCache` per run, `ReadLedger`
per agent.

Also fixed: a read suspends at an `await` before it can store, so a concurrent
`bash` invalidation could land in the window and the read would reinsert
pre-command bytes nothing would evict again (a generation counter now rejects
those stores); `note_write` was `record_read` minus the eviction loop, so the
write path was unbounded; `_versions` grew without limit; the read-only
profile — the one place an un-revalidating cache can never be wrong — was the
only profile not wired to one; and `bash` invalidated on the success path only,
so a command that raised after writing left the cache intact.

Four of the mutations that survived the first round did so for one reason: the
tests built tools by hand and never went through `Orchestrator._build_registry`.
The components were right and the wiring was wrong, which is the same blind
spot that hid S-108's dead half.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Symlinks are still two keys.** `normalise_path` collapses `./a.py` and
  `pkg/../a.py`, but resolving a symlink needs a sandbox call — which is the
  thing the cache exists to avoid. A file reached by two names is a stale entry
  waiting to happen.
- **`bash` invalidation makes the cache mostly inert.** A real run interleaves
  `bash` constantly, so the realistic hit rate is read-then-read-again within a
  turn. Unmeasured.
- **`edit_file` reads the file twice** when a ledger is present: once for the
  staleness check, once inside `sandbox.edit_file`. Threading the content
  through would remove both the second read and the window between them.
- **The staleness check and the edit are not atomic.** The check reads at one
  moment and `sandbox.edit_file` re-reads at another, so with concurrent tool
  calls the check can report "fresh" on content already replaced.
- **No false-positive measurement.** The rationale for warning rather than
  rejecting was "measure first". The replay corpus can carry it and does not.
- **N7/N8 cannot see this feature at all.** The corpus rebuilds tool messages
  from character counts, so the invariants would stay green through any change
  to tool-result text. The benchmark-path assertion is a separate test.
- **`MAX_CACHED_FILES = 64` and `MAX_TRACKED_VERSIONS = 512` have no basis.**
  They bound memory; nothing measured what hit rate they buy.
