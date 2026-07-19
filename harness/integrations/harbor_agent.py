"""Harbor (Terminal-Bench 2.0) custom-agent entry point (DESIGN.md §4.13).

This is the bridge that produces the project's benchmark numbers:
:class:`HarnessAgent` implements Harbor's ``BaseAgent`` contract by running
our own :class:`~harness.orchestrator.Orchestrator` loop **host-side**
against the Harbor task container, wrapped as a
:class:`~harness.sandbox.harbor_env.HarborSandbox`.

Deployment model: ``harbor run --agent
harness.integrations.harbor_agent:HarnessAgent`` imports this module inside
Harbor's own uv-tool venv (not our project venv), so our package must be
installed there editable::

    uv tool install --force harbor --with-editable /Users/vinay/vinaysrao1/harness
    harbor run --agent harness.integrations.harbor_agent:HarnessAgent \\
        --model openrouter/moonshotai/kimi-k2 ...

Every ``import harbor...`` statement in the whole harness lives in this one
module, guarded below so the failure mode outside Harbor's venv is a clear
instruction rather than a bare ``ModuleNotFoundError``. The sandbox wrapper
itself (:mod:`harness.sandbox.harbor_env`) is duck-typed and imports no
Harbor code, so our test suite never needs Harbor installed.
"""

from __future__ import annotations

import os
import warnings
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import Any

try:
    from harbor.agents.base import BaseAgent
except ImportError as exc:  # pragma: no cover - exercised via stubs in tests
    raise ImportError(
        "harness.integrations.harbor_agent requires the 'harbor' package, "
        "which is not importable here. This module is meant to be loaded by "
        "'harbor run --agent harness.integrations.harbor_agent:HarnessAgent' "
        "inside Harbor's own uv-tool venv; install this project into that "
        "venv with:\n"
        "  uv tool install --force harbor "
        "--with-editable /Users/vinay/vinaysrao1/harness"
    ) from exc

from harness.adapters import get_adapter
from harness.config import (
    HarnessConfig,
    ModelConfig,
    PermissionMode,
    load_config,
)
from harness.loop import AgentResult, Budgets
from harness.orchestrator import Orchestrator
from harness.permissions import ToolMeta
from harness.persistence import RunStore
from harness.sandbox.harbor_env import HarborSandbox

__all__ = ["HarnessAgent", "resolve_model"]

#: Default per-trial budgets, overridable via HARNESS_MAX_TURNS /
#: HARNESS_MAX_TOKENS in the agent's ``extra_env`` (``harbor run
#: --agent-kwarg``/config) or the process environment.
_DEFAULT_MAX_TURNS = 80
_DEFAULT_MAX_TOKENS = 2_000_000

#: Cap on the ``final_text`` echoed into Harbor's context metadata.
_FINAL_TEXT_LIMIT = 2000


def _harness_version() -> str:
    """Our installed package version (best-effort, for Harbor's records)."""
    try:
        return _package_version("agent-harness")
    except PackageNotFoundError:  # pragma: no cover - editable installs have it
        return "unknown"


async def _never_ask(tool_name: str, arguments: dict, meta: ToolMeta) -> bool:
    """Belt-and-braces approval callback: always deny.

    Benchmark runs are headless and use :attr:`PermissionMode.AUTO`, where
    nothing should reach ASK in the first place; if something does anyway,
    denying is the only sane headless answer.
    """
    return False


def resolve_model(model_name: str | None, config: HarnessConfig) -> ModelConfig:
    """Resolve Harbor's ``--model`` string to a harness :class:`ModelConfig`.

    Pure function of its arguments (separately testable, no I/O). Tried in
    order:

    1. A name in ``config.models`` — the user's own registry entry wins, so
       short names like ``kimi-or`` keep working under Harbor.
    2. A litellm-style provider-prefixed string:

       - ``openrouter/<model>`` -> ``openai`` adapter against
         ``https://openrouter.ai/api/v1`` with ``env:OPENROUTER_API_KEY``
       - ``anthropic/<model>``  -> ``anthropic`` adapter with
         ``env:ANTHROPIC_API_KEY``
       - ``openai/<model>``     -> ``openai`` adapter (default base URL)
         with ``env:OPENAI_API_KEY``

    Anything else raises :class:`ValueError` listing the accepted forms and
    the available registry names.
    """
    if model_name and model_name in config.models:
        return config.models[model_name]
    if model_name and "/" in model_name:
        provider, _, rest = model_name.partition("/")
        if rest:
            if provider == "openrouter":
                return ModelConfig(
                    adapter="openai",
                    base_url="https://openrouter.ai/api/v1",
                    model=rest,
                    api_key="env:OPENROUTER_API_KEY",
                )
            if provider == "anthropic":
                return ModelConfig(
                    adapter="anthropic",
                    model=rest,
                    api_key="env:ANTHROPIC_API_KEY",
                )
            if provider == "openai":
                return ModelConfig(
                    adapter="openai",
                    model=rest,
                    api_key="env:OPENAI_API_KEY",
                )
    available = ", ".join(sorted(config.models)) or "(none)"
    raise ValueError(
        f"cannot resolve model {model_name!r}: expected either a name from "
        f"the harness config registry (available: {available}) or a "
        "litellm-style string of the form 'openrouter/<model>', "
        "'anthropic/<model>', or 'openai/<model>'"
    )


class HarnessAgent(BaseAgent):
    """Runs the agent harness as a Harbor custom agent.

    Harbor constructs one instance per trial with ``logs_dir`` /
    ``model_name`` / assorted extras (``task_dir``, ``trial_paths``,
    ``agent_timeout_sec``, ...); the base-class ``__init__`` accepts and
    absorbs those, so no override is needed here. :meth:`run` drives one
    :meth:`~harness.orchestrator.Orchestrator.run_task` inside the Harbor
    task container and reports token usage back through Harbor's
    ``AgentContext``.
    """

    @staticmethod
    def name() -> str:
        """Harbor agent name (``--agent`` display / results key)."""
        return "agent-harness"

    def version(self) -> str | None:
        """The installed harness package version."""
        return _harness_version()

    async def setup(self, environment: Any) -> None:
        """No-op: our loop runs host-side, nothing to install in-container.

        The model adapters, orchestrator, and stores all live in this
        (host) process; the container only ever sees ``exec`` calls from
        :class:`~harness.sandbox.harbor_env.HarborSandbox`.
        """
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        """Run one benchmark trial: ``instruction`` is the goal.

        Per-trial isolation: all harness state (RunStore SQLite, memory,
        skills, workspaces) is rooted at a fresh ``harness-home`` directory
        under this trial's ``logs_dir``. The RunStore is single-writer and
        Harbor runs trials concurrently, so the user's real ``~/.harness``
        is never touched — except read-only, to load their model registry
        (see :meth:`_load_user_config`).

        ``context`` is populated (tokens + metadata) before returning *and*
        on exception paths, so Harbor records partial usage even for a
        failed trial; the exception is re-raised so Harbor marks the
        failure.
        """
        harness_home = Path(self.logs_dir) / "harness-home"
        harness_home.mkdir(parents=True, exist_ok=True)
        # Deliberately NOT exported as $HARNESS_HOME: Harbor runs trials
        # concurrently in one process, so a process-global env mutation is
        # last-writer-wins — trial A could observe trial B's home. Every
        # run-path consumer takes the home explicitly (HarnessConfig(home=
        # ...), RunStore path, _load_user_config's explicit path); nothing
        # here may rely on the environment variable.

        model_config = resolve_model(self.model_name, self._load_user_config())
        adapter = get_adapter(model_config)
        budgets = Budgets(
            max_turns=self._int_setting("HARNESS_MAX_TURNS", _DEFAULT_MAX_TURNS),
            max_tokens=self._int_setting(
                "HARNESS_MAX_TOKENS", _DEFAULT_MAX_TOKENS
            ),
        )

        sandbox = HarborSandbox(environment)
        await sandbox.start()

        run_id: str | None = None
        result: AgentResult | None = None
        error: BaseException | None = None
        with RunStore(harness_home / "state.db") as store:
            orchestrator = Orchestrator(HarnessConfig(home=harness_home), store)
            try:
                run_id, result = await orchestrator.run_task(
                    instruction,
                    self.model_name or "harbor-model",
                    mode=PermissionMode.AUTO,
                    ask=_never_ask,
                    adapter_override=adapter,
                    budgets=budgets,
                    sandbox=sandbox,
                )
            except BaseException as exc:
                error = exc
                raise
            finally:
                self._populate_context(
                    context,
                    harness_home=harness_home,
                    run_id=run_id,
                    result=result,
                    error=error,
                )

    # -- helpers -------------------------------------------------------------

    def _load_user_config(self) -> HarnessConfig:
        """Load the *user's* real config for model-registry resolution.

        The registry the user actually maintains lives at
        ``~/.harness/config.toml``, so that explicit path is passed here
        (read-only) to keep names like ``kimi-or`` resolvable under
        Harbor — never :func:`~harness.config.load_config`'s no-path form,
        which consults ``$HARNESS_HOME`` (unreliable under Harbor's
        concurrent trials; see the note in :meth:`run`). A missing file
        yields an empty registry, leaving only the litellm-style forms of
        :func:`resolve_model`.
        """
        user_config_path = Path.home() / ".harness" / "config.toml"
        if user_config_path.is_file():
            return load_config(user_config_path)
        return HarnessConfig()

    def _int_setting(self, name: str, default: int) -> int:
        """Read an integer setting from ``extra_env`` then ``os.environ``.

        A present-but-unparseable value falls back to ``default`` with a
        :class:`UserWarning` rather than failing the whole trial.
        """
        raw = self.extra_env.get(name, os.environ.get(name))
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            warnings.warn(
                f"{name}={raw!r} is not an integer; using default {default}",
                UserWarning,
                stacklevel=2,
            )
            return default

    @staticmethod
    def _populate_context(
        context: Any,
        *,
        harness_home: Path,
        run_id: str | None,
        result: AgentResult | None,
        error: BaseException | None,
    ) -> None:
        """Fill Harbor's ``AgentContext`` from the run outcome.

        Token fields map from :class:`~harness.types.Usage`, normalized to
        Harbor's convention at this reporting boundary:
        ``n_input_tokens`` is documented by Harbor as input tokens
        *including cache* (its own claude_code agent sums input +
        cache reads + cache writes), while the harness ``Usage``
        convention (see :class:`harness.types.Usage`) is that
        ``input_tokens`` *excludes* cache traffic for every adapter —
        so the three fields are summed here; ``cache_read_tokens``
        also maps to ``n_cache_tokens``. ``cost_usd`` is deliberately left ``None``
        (the harness tracks tokens, not provider pricing). ``metadata``
        carries enough to find the full trace in the per-trial harness
        home. On a crash before any result exists, the metadata still
        lands (status ``error`` + the exception text).
        """
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "status": result.status if result is not None else "error",
            "turns": result.turns if result is not None else 0,
            "final_text": (
                (result.final_text or "")[:_FINAL_TEXT_LIMIT]
                if result is not None
                else ""
            ),
            "harness_home": str(harness_home),
        }
        if error is not None:
            metadata["error"] = f"{type(error).__name__}: {error}"
        if result is not None:
            usage = result.usage
            context.n_input_tokens = (
                usage.input_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens
            )
            context.n_output_tokens = usage.output_tokens
            context.n_cache_tokens = usage.cache_read_tokens
        context.metadata = metadata
