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

## Status

Milestones M1–M4 of DESIGN.md §7 are implemented (loop, durability, permissions,
two adapters, memory, skills, subagents). Not yet built: MCP connectors + OAuth
(M5), the eval runner and consolidation loop (M6–M7).
