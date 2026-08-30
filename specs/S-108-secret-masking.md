---
id: S-108
title: Secret masking at the tool boundary
status: Implemented
lane: A
depends: -
effort: S
---

# S-108 — Secret masking at the tool boundary

## Contract
A `SecretRegistry` bound into `ToolDeps`; known secret values are masked in
every tool result and every persisted event. New module `harness/secrets.py`;
optional `secrets=` on `ToolRegistry`, `RunStore` and `Orchestrator`;
`build_secret_registry(config)` collects the API keys the run already resolved.

## Invariants
Lane A, and by identity rather than by argument: an empty registry's `mask`
returns **the same string object** it was given, and `mask_payload` returns the
same object. `build_secret_registry` yields an empty registry for any config
without resolvable keys, which is every fake-adapter test and the entire
benchmark path. Conformance stayed at 48; no golden moved.

## Acceptance
1. A command that echoes a configured secret returns a mask. ✅ —
   `TestTheToolBoundary`, through the real `ToolRegistry.dispatch`.
2. Masking applies before persistence, so the event log is clean too. ✅ —
   `TestThePersistedLog`, including events no tool produced (a model message, a
   `run_error` carrying a credentialed URL).
3. Masking is O(result size) and does not change result-length semantics for
   truncation accounting (N8). ✅ — one `str.replace` per registered secret;
   on `CODING` the registry is empty and the identity path runs, so N8 cannot
   observe it.

## Telemetry
none. Recording *that* a secret was masked would put the fact it appeared into
the log this spec exists to keep clean, and recording where would point at it.

## Rollback
`git revert`. Every integration point takes `secrets=None` and defaults to it,
so removing the argument restores the previous behaviour exactly.

## Why mask before the spill, not after
An oversized result takes three paths at once: truncated into the model's
context, written **in full** into the sandbox by `_spill_full_result`, and
persisted as an event. Masking where the result is returned would clean the
model's copy and leave the secret in a file on disk with a path the model was
just told. So the mask is applied the moment the handler returns — before the
byte count, before the spill, before truncation.

Truncation accounting therefore measures masked text. That is safe here only
because the benchmark registry is empty; if it were not, replacing a 40-char
key with a 24-char label would shift truncation boundaries, and N8 exists to
catch exactly that kind of drift.

## Why short values are refused, loudly
A four-character "secret" occurs inside ordinary output constantly, and
replacing it corrupts tool results far more than the leak it prevents. Anything
below `MIN_MASKABLE_LENGTH` raises at registration rather than being quietly
dropped: silently declining to mask something the caller asked to have masked
is precisely the failure this module exists to prevent, and it would be
invisible. `build_secret_registry` catches that error and skips the key,
because a run must not die because redaction could not be set up — but the
distinction is made at the boundary, not hidden inside the registry.

Secrets are stored longest-first so a value containing another value masks as
one unit. Registered the other way round, the shorter secret redacts the inside
of the longer one and leaves a partially-redacted fragment — worse than no
masking, because it looks safe.

## What the review found
Seven defects. The top two meant the feature was largely imaginary, and both
are the archetype.

**The persistence half never fired in production.** `RunStore` accepted a
`secrets=` keyword and *no real caller passed it* — the CLI, the eval runner
and the Harbor bridge all construct the store before a registry can exist. The
only call site passing it was this spec's own test, which built a shape
production never builds. So the event log was unmasked in every actual run
while a green test asserted otherwise, and thirteen mutations all passed
because none of them could be "delete the argument from the real call sites":
there were none. `Orchestrator.__init__` now calls `store.bind_secrets()`,
because it is the one object holding both.

**The most common credential setup produced an empty registry.** When config
omits `api_key` the adapters deliberately let the provider SDK read
`ANTHROPIC_API_KEY` itself — documented, supported, and probably the modal
setup. `build_secret_registry` read only `ModelConfig`, so the registry came
back empty while the live key sat in `os.environ`, inherited by every
`LocalSandbox` subprocess. An agent running `env` — the opening scenario of
this module's own docstring — leaked it with masking a silent no-op.
`CREDENTIAL_ENV_VARS` now registers those variables directly.

The rest:

- **`dispatch`'s two `is_error` returns bypassed masking**, and an exception
  message routinely carries the thing that failed: a `CalledProcessError` with
  the command, an HTTP client raising with the request URL.
- **`SecretRegistry.__repr__` printed every value in plaintext**, undoing the
  discipline `ModelConfig.api_key` already had as a `SecretStr`. `repr=False`
  on the field, and a `__repr__` that reports a count.
- **`mask_payload` skipped dict keys and tuples.** Model-supplied
  `tool_call.arguments` is an arbitrary object; a tuple serialises into the log
  as an array.
- **Overlapping (not containing) secrets left a fragment** — exactly what this
  spec claimed the longest-first ordering eliminated. Ordering fixes
  containment only. Masking is now span-based: locate every occurrence, merge
  overlapping spans, replace right to left.
- **Every configured model's key was resolved at every `Orchestrator`
  construction**, and the eval runner builds one per trial — so a `keychain:`
  reference meant a keyring read per model per trial. The runner now builds one
  registry per suite.

**`ToolDeps.secrets` was deleted.** This spec had already flagged it as a seam
with no user; the review showed it was worse — never populated, so the first
caller would have got an `AttributeError`. Removed rather than documented.

Three of this spec's own tests were environment-dependent: they asserted an
empty registry and passed only on a machine without `ANTHROPIC_API_KEY`
exported. Now explicitly isolated by a fixture.

## Known gaps
Exhaustive as far as is known; anything missing is a defect in this list.

- **It only knows what it was told.** The registry holds credentials from a
  fixed list of environment variables plus any key a configured model resolves.
  A secret the agent discovers — a token in a `.env` file it reads, a password
  in a config — is not masked. This is not a scanner and has no entropy
  heuristic, deliberately: a false positive corrupts a tool result, and a false
  negative is the status quo.
- **`CREDENTIAL_ENV_VARS` is a fixed list and will fall behind.** A provider
  whose variable is not on it is unmasked. Matching `*_API_KEY` by shape was
  rejected because it sweeps in variables whose values are short or structural.
- **Only exact byte matches.** A base64, URL-encoded, or shell-escaped form of
  a secret passes through untouched, as does one split across a line break.
- **Nothing outside tool results and event payloads is masked.** Adapter
  request/response bodies, the `runs`/`agents` goal text, log output, and
  tracebacks propagating past `dispatch` are all untouched.
- **Masking is O(secrets × result).** Fine for the handful a config holds; the
  span scan is a `str.find` loop per secret.
- **N8 is not exercised against a populated registry.** The invariant holds on
  the benchmark path because the registry is empty there — masking a 40-char
  key to a 24-char label *would* shift truncation boundaries, and nothing
  tests what that does.
- **Key resolution is still eager.** `build_secret_registry` resolves every
  configured model's key, not just the one in use. Callers that build many
  Orchestrators should pass a prebuilt registry; the eval runner now does, and
  nothing enforces it for anyone else.
