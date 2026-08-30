---
id: S-101
title: Structured search tier
status: Implemented
lane: A
depends: S-005
effort: M
---

# S-101 — Structured search tier

## Contract
`glob(pattern, path?, head_limit?)` and `grep(pattern, path?, glob?, type?,
output_mode, context?, head_limit?, case_insensitive?)` with
`output_mode ∈ {files_with_matches (default), content, count}`, backed by `rg`
when it is present and a self-contained Python fallback when it is not. New
module `harness/search.py`; both tools in `CODING_REPO` only.

## Invariants
Lane A. `CODING`'s tool set is unchanged — both tools ship in
`REPO_TOOL_FACTORIES`, and `test_S101_search_is_repo_mode_only` fails if either
reaches the benchmark profile. Conformance stayed at 48; no golden moved.

`harness/search.py` **is** imported on the benchmark path — `tools/builtin.py`
imports it at module level and `loop.py` imports `builtin`. An earlier version
of this section claimed otherwise. Neutrality holds because the *tool specs* are
unchanged, which is what N1/N2 pin; the import is not the thing that matters,
and saying it was would have been an argument that did not apply.

## Acceptance
1. Default mode returns paths only. ✅ — `test_S101_files_with_matches_returns_paths`,
   which asserts content is *absent*, not merely that paths are present.
2. With no `rg`, results are identical modulo ordering. ✅ —
   `TestBothEnginesAgree`, five query shapes, run inside the sandbox image
   because that is the only place `rg` exists here.
3. `rg` is never installed into an attached sandbox. ✅ — a source scan for
   install verbs, plus a spy asserting the only probe issued is `command -v rg`.
4. Result volume for a large query is bounded before truncation, not by it. ✅ —
   `test_S101_the_search_itself_stops_at_the_limit` asserts on the **engine's
   own stdout**: 1,000 matching files, limit 5, exactly 5 lines produced.

## Telemetry
none of its own. A search is a read; recording every query would duplicate the
`tool_call` event that already exists.

## Rollback
`git revert`. Removing the two factories from `REPO_TOOL_FACTORIES` restores
the previous tool set exactly.

## Why not just `bash("grep -rn … | head")`
That is what the agent does today, and it works. What it cannot do is bound the
work: the pipeline computes every match and `head` throws the tail away, so the
cost is already paid in time and in the sandbox's memory. It also fails the way
pipelines fail — `head` closes the pipe, `grep` dies of SIGPIPE, and the exit
status the harness records describes the last stage rather than the search. And
it spends a permission decision on a general-purpose execution tool for what is
a read.

The bound here is inside the engine: `--max-count` for ripgrep, an early
`break` in the fallback's walk. Acceptance (4) is specifically about that
distinction, and it is not observable through the tool — `render_results` slices
to the limit either way, so a search that scans a thousand files and discards
995 returns an identical answer to one that stopped at five. The test asserts on
the engine's raw output for that reason.

A bounded result **says so**. A truncation that does not announce itself is
worse than no result: the model concludes a symbol appears in five files when it
appears in a thousand, and narrows its next search on a false premise.

## Why the fallback takes JSON on stdin
A regex reaching the shell through `argv` has to survive two levels of quoting.
Getting that wrong turns a search into a syntax error — or, worse, into a
*different search that silently succeeds*. The fallback is a single `python3 -c`
program reading a JSON blob from stdin, so the pattern is never re-parsed by a
shell. `test_S101_a_regex_with_shell_metacharacters_is_not_mangled` covers
`$(…)`, `&&`, quotes and alternation.

One exec does the whole walk rather than the harness listing files and reading
each back: a repository has thousands of files, and a round-trip each would take
longer than the search is worth.

## Why the engine is probed at first use
A registry is built *before* the environment probe runs, so a tool that read
`EnvironmentProfile.tooling` at construction time would capture "no rg" and
keep believing it for the whole run. One `command -v rg` per sandbox, cached,
is cheaper than being wrong — `test_S101_the_probe_is_cached` pins that it
happens once.

## What the review found
Thirteen defects. The critical one is this project's archetype, and this spec
had *written it down as a design decision*:

> `It is the outermost stage, so its status is the one the shell reports --
> which is why rg's own exit code is not consulted anywhere.`

That comment sat above a `| head` pipeline — the exact construction this spec
spends a section explaining is unacceptable in `bash("grep … | head")`.
`Sandbox.exec` does not raise on a non-zero exit; it *returns* the code, stderr
and a timeout flag, and all three were discarded. Every failure therefore
reached the model as **"no matches"**:

| input | reported |
|---|---|
| invalid regex `alpha(` | no matches |
| catastrophic backtracking (30s timeout) | no matches |
| `type="python"` (rg exits 2) | no matches |
| **no `python3` in the sandbox** | no matches — the whole repository reads as empty |

The last row is the worst, and this spec's Known gaps asserted the opposite
("fails rather than degrading further"). The `| head` is gone — it bounded
transfer, never work, since `rg` must walk the tree regardless — and
`describe_failure` distinguishes rg's exit 1 (no matches) from exit 2 (error)
and from 127 (missing backend).

The rest, in descending order of how badly they misled:

- **`path` naming a file returned nothing.** `os.walk` yields nothing for a
  regular file, so the most natural narrowing an agent makes after a
  `files_with_matches` search — "now search just that file" — was guaranteed
  empty in the engine that runs in most containers.
- **`path` escaped the workspace.** `/etc` and `../` both returned real files.
  Every other file-touching tool resolves against the workspace root.
- **`content` mode's bound was truncation-after-the-fact** — the exact thing
  acceptance (4) is about. The engines stop after `limit` *matches*; the
  renderer cut after `limit` *lines*, so `context=2` delivered a third of the
  stated bound and could end on a bare context line, which reads as a match on
  the wrong text. And a single 280,000-character minified line produced a
  100,000-character result under a stated bound of one: the *count* was bounded
  and the *volume* was not.
- **`glob` patterns containing `/` matched nothing.** `-path` matches the path
  as `find` prints it, beginning `./`. Both documented examples — `pkg/*.py`,
  `src/**/*.ts` — returned "no matches", which reads as "no such file". The
  test covering this used `*/pkg/*.py`, the one form that happened to work, and
  so **certified the broken behaviour as correct**.
- **`glob`'s fallback pruned nothing**, while `rg --files` honours `.gitignore`
  — so in any repo with a `.venv` the two engines returned unrelated answers.
- **`count` counted matching lines, not matches**, where `rg --count-matches`
  counts matches. A file with three hits on one line reported 1 against 3.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **Not promoted to `CODING`, which is the point of the spec.** The plan calls
  this "the single most likely net-positive Lane B change" and requires a TB2
  run. `CODING` is at its 15-tool cap, so promotion means removing two tools —
  most plausibly narrowing `bash`'s role, a larger change than this. **No
  evidence has been gathered either way.** S-401 can measure it and has not.
- **`structured_search` is a declared capability nothing gates on, defined
  backwards.** `affirms("structured_search")` is `"rg" in tooling`, so wiring
  it to the documented `enables() ∧ affirms()` rule would remove `grep`/`glob`
  from exactly the sandboxes the Python fallback exists for. It is dead in
  `REPO_CAPABILITIES` and its definition contradicts this implementation.
  Pre-existing from S-004; not fixed here, and it should be either redefined or
  removed.
- **The fallback needs `python3`.** Absent both `rg` and `python3` the tool now
  *reports* that rather than claiming no matches, but it still cannot search.
- **`type` is ripgrep-only** and the fallback ignores it silently, so
  `type="py"` returns more than asked for. Silent over-matching is the wrong
  direction for a filter.
- **Ordering differs between engines**, and the equivalence test compares path
  sets. An agent relying on the first result being most relevant gets different
  answers.
- **The equivalence test is skipped without Docker** — it is the only coverage
  of acceptance (2), and it compares paths only, so a divergence in line
  numbers, counts or context markers is invisible to it.
- **`SKIP_DIRS` is a fixed list and does not read `.gitignore`**; `rg` does. The
  two engines genuinely disagree on a repository that ignores something
  unusual, and the equivalence corpus is one where they agree.
- **The fallback reads whole files into memory** where `rg` streams, so a very
  large file is an allocation the sandbox may not survive.
- **`--max-count` is per file.** With the `| head` removed, `rg`'s total bound
  is `render_results`, which is truncation-after-the-fact for that engine — the
  thing the fallback avoids. `rg` must walk the tree for `files_with_matches`
  regardless, so no work is saved by bounding earlier, but `content` mode over
  a huge file still transfers more than the limit before it is cut.
