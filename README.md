# agent-harness

A personal, model-agnostic agentic harness. One CLI runs LLM agents that pursue
a goal with sandboxed code execution, persistent memory, skills, permission
gating, and crash-resumable runs. Design rationale: [DESIGN.md](DESIGN.md).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q        # 542 passed, 13 docker-skipped
```

## Configure

`~/.harness/config.toml` (or `HARNESS_HOME` to relocate):

```toml
[models.opus]
adapter = "anthropic"
model   = "claude-opus-4-8"
api_key = "env:ANTHROPIC_API_KEY"

[models.kimi]
adapter  = "openai"
base_url = "https://api.moonshot.ai/v1"
model    = "kimi-k3"
api_key  = "env:MOONSHOT_API_KEY"

[sandbox]
network = "none"            # none | allowlist | open

[permissions]
default = "gated"           # gated (approval prompts) | auto
```

API keys are references (`env:`/`keychain:`), never literals in context or logs.

## Use

```bash
harness run "refactor the parser and make the tests pass" --model opus
harness run "..." --model kimi --mode auto     # fully autonomous (hard-denies still apply)
harness runs                                    # list persisted runs
harness cost <run_id>                           # token/usage accounting
harness resume <run_id>                         # continue an interrupted run
```

Docker daemon running → agents execute in a container (workspace bind-mounted
under `~/.harness/runs/<id>/workspace`). Daemon down → subprocess fallback with
a warning (path-jailed but not isolated).

## Layout

| Layer | Files |
|---|---|
| Core types / adapters | `harness/types.py`, `harness/adapters/` (anthropic, openai_compat, fake) |
| Context & instruction adherence | `harness/context.py` (layered compaction, ledgers, reminders) |
| Agent loop & diligence | `harness/loop.py`, `harness/diligence.py` |
| Multi-agent & CLI | `harness/orchestrator.py` (spawn/await, depth ≤ 1), `harness/cli.py` |
| Sandbox | `harness/sandbox/` (docker, local fallback) |
| Permissions | `harness/permissions.py` (ALLOW/DENY/ASK, hard-deny categories) |
| Memory & skills | `harness/memory/store.py`, `harness/skills.py` |
| Persistence | `harness/persistence.py` (SQLite event log, resume) |

## PR-replay eval (S-401)

A second benchmark built from history that already exists: reset a merged
commit's tree to its parent, restore only that commit's tests, and ask the
agent to make them pass. The commit's tests are the grader; its diff is the
reference answer.

```bash
docker build -t harness-sandbox:latest .      # once; graded commands run in it
harness eval suite.json --model glm-flash --split dev
```

A suite file names a repository, the revisions to replay, and how to grade:

```json
{
  "name": "click",
  "repo": "/path/to/click",
  "revs": ["a6256bfb...", "1f9cd54f..."],
  "test_command": "PYTHONPATH=src python3 -m pytest {tests} -q -p no:cacheprovider"
}
```

`{tests}` is replaced with the task's own test paths. Without it the command
must name what to run itself, which in practice means the whole suite — and
then any unrelated failure in the repository sinks every task.

Two things worth knowing before reading a number from it:

- **Set the wall clock generously.** At 420s, three of eleven trials ran out of
  time and wrote their fix in the final turn. Watch the `budget paused` line;
  if it is not zero, the run is measuring reading speed as much as coding.
- **Everything a task needs must be in the image.** Graded commands run inside
  the sandbox with no network. A missing dependency is reported as an
  environment failure rather than a task defect, so read those reasons — they
  are usually a Dockerfile edit, not a bad task.

## TB2 run playbook

When running against Harbor trials:

**Pre-build images** — Pre-pull images before a scored run to avoid EnvStart timeouts:
```bash
harbor run <config> ... --install-only   # Install/build, skip agent & verifier
harbor run <config> ... [normal run]     # Reuses cached images
```

**Verifier and build headroom** — On contended hosts, add timeout multipliers:
```bash
--timeout-multiplier 1.2                        # Global (all timeouts)
--verifier-timeout-multiplier 1.5               # Verifier only
--environment-build-timeout-multiplier 1.5      # Docker build only
```
Multipliers override `--timeout-multiplier`. Do not bump `--agent-timeout-multiplier`
on scored runs (changes the benchmark condition; if used, Fix 1's derivation reads
it from `config.json` automatically).

**Manual deadline overrides** — Set per-invocation:
```bash
export HARNESS_WALL_CLOCK_SECONDS=<seconds>     # Env override (highest priority)
--agent-kwarg agent_timeout_sec=<seconds>       # Alternative: via agent kwarg
```

## Status

Milestones M1–M4 of DESIGN.md §7 are implemented (loop, durability, permissions,
two adapters, memory, skills, subagents). Not yet built: MCP connectors + OAuth
(M5), the eval runner and consolidation loop (M6–M7).

## License

[Apache-2.0](LICENSE)
