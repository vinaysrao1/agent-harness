"""Tests for harness.integrations.harbor_agent and the orchestrator's
caller-provided-sandbox path.

Harbor is never imported (it lives only in its own uv-tool venv): the
module under test is imported against duck-typed stub ``harbor`` modules
injected into ``sys.modules``, the environment is the same
:class:`StubEnvironment` the sandbox tests use, and Harbor's
``AgentContext`` is a plain attribute bag. The dedicated ImportError test
goes the other way and *blocks* ``harbor`` imports to pin the guidance
message deterministically.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig, PermissionMode, load_config
from harness.orchestrator import Orchestrator
from harness.persistence import RunStore
from harness.sandbox.harbor_env import HarborSandbox
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from tests.test_harbor_sandbox import StubEnvironment

_MODULE = "harness.integrations.harbor_agent"

#: A final message the diligence check accepts as finished.
CLEAN_FINISH = "Task complete. Ran the command; output verified."


def resp(
    content: str | None = None,
    calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> ModelResponse:
    """Build one scripted assistant response."""
    tool_calls = calls or []
    return ModelResponse(
        message=Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls),
        usage=usage or Usage(),
        stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
    )


def bash_script() -> list[ModelResponse]:
    """Script: run one bash command, then finish cleanly."""
    return [
        resp(
            calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo hello"})],
            usage=Usage(input_tokens=10, output_tokens=2, cache_read_tokens=3),
        ),
        resp(
            CLEAN_FINISH,
            usage=Usage(input_tokens=5, output_tokens=1, cache_write_tokens=4),
        ),
    ]


# -- stub harbor plumbing ----------------------------------------------------


class _StubBaseAgent:
    """Duck-type of harbor.agents.base.BaseAgent's constructor contract."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: object = None,
        mcp_servers: list | None = None,
        skills_dir: str | None = None,
        *args: object,
        extra_env: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        self.logs_dir = logs_dir
        self.model_name = model_name
        self.logger = logger
        self.mcp_servers = mcp_servers or []
        self.skills_dir = skills_dir
        self._extra_env: dict[str, str] = dict(extra_env) if extra_env else {}

    @property
    def extra_env(self) -> dict[str, str]:
        return dict(self._extra_env)


class _StubTrialAgentConfig(BaseModel):
    """Duck-type of harbor.models.trial.config.AgentConfig (timeout fields).

    ``extra="allow"`` absorbs the fields real config.json files carry
    (``name``, ``model_name``, ...) without modeling them.
    """

    model_config = ConfigDict(extra="allow")

    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = None


class _StubTaskConfig(BaseModel):
    """Duck-type of harbor.models.trial.config.TaskConfig.

    ``get_task_id()`` mirrors Harbor's dispatch: git fields present yields
    a GitTaskId-alike whose ``get_local_path()`` lands under the
    test-controlled task cache dir (Harbor hashes the id into
    ``~/.cache/harbor/tasks/<uuid>/<task-name>``); otherwise a
    LocalTaskId-alike resolving ``path`` directly.
    """

    model_config = ConfigDict(extra="allow")

    path: Path | None = None
    git_url: str | None = None
    git_commit_id: str | None = None

    #: Test hook standing in for Harbor's task cache root; monkeypatched
    #: per-test by the ``trial_layout`` fixture.
    git_cache_dir: ClassVar[Path | None] = None

    def get_task_id(self) -> SimpleNamespace:
        if self.git_url is not None:
            cache = type(self).git_cache_dir
            assert cache is not None, (
                "git-shaped fixture requires _StubTaskConfig.git_cache_dir"
            )
            assert self.path is not None
            task_dir = cache / self.path.name
            return SimpleNamespace(get_local_path=lambda: task_dir)
        assert self.path is not None
        local = self.path.expanduser().resolve()
        return SimpleNamespace(get_local_path=lambda: local)


class _StubTrialConfig(BaseModel):
    """Duck-type of harbor.models.trial.config.TrialConfig."""

    model_config = ConfigDict(extra="allow")

    task: _StubTaskConfig
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    agent: _StubTrialAgentConfig = Field(default_factory=_StubTrialAgentConfig)


@pytest.fixture
def harbor_agent(monkeypatch: pytest.MonkeyPatch):
    """Import the module under test against stub ``harbor`` modules.

    The stub packages are injected via monkeypatch (auto-restored); the
    module itself is imported fresh and evicted afterwards so no other
    test sees a harbor_agent bound to the stubs.
    """
    harbor = types.ModuleType("harbor")
    agents = types.ModuleType("harbor.agents")
    agents_base = types.ModuleType("harbor.agents.base")
    agents_base.BaseAgent = _StubBaseAgent
    harbor.agents = agents
    agents.base = agents_base
    models = types.ModuleType("harbor.models")
    models_trial = types.ModuleType("harbor.models.trial")
    models_trial_config = types.ModuleType("harbor.models.trial.config")
    models_trial_config.TrialConfig = _StubTrialConfig
    harbor.models = models
    models.trial = models_trial
    models_trial.config = models_trial_config
    monkeypatch.setitem(sys.modules, "harbor", harbor)
    monkeypatch.setitem(sys.modules, "harbor.agents", agents)
    monkeypatch.setitem(sys.modules, "harbor.agents.base", agents_base)
    monkeypatch.setitem(sys.modules, "harbor.models", models)
    monkeypatch.setitem(sys.modules, "harbor.models.trial", models_trial)
    monkeypatch.setitem(
        sys.modules, "harbor.models.trial.config", models_trial_config
    )
    sys.modules.pop(_MODULE, None)
    try:
        yield importlib.import_module(_MODULE)
    finally:
        sys.modules.pop(_MODULE, None)


class _HarborImportBlocker:
    """Meta-path finder that makes any ``harbor`` import fail, even if some
    future dev venv happens to have Harbor installed."""

    def find_spec(self, name: str, path=None, target=None):
        if name == "harbor" or name.startswith("harbor."):
            raise ModuleNotFoundError(f"import of {name} blocked by test")
        return None


def test_import_without_harbor_raises_clear_error():
    """Importing the entry point outside Harbor's venv names the fix."""
    blocker = _HarborImportBlocker()
    sys.meta_path.insert(0, blocker)
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == _MODULE or name == "harbor" or name.startswith("harbor.")
    }
    try:
        with pytest.raises(
            ImportError,
            match=r"uv tool install --force harbor --with-editable",
        ):
            importlib.import_module(_MODULE)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop(_MODULE, None)
        sys.modules.update(saved)


# -- resolve_model -----------------------------------------------------------


class TestResolveModel:
    @pytest.fixture
    def registry_config(self, tmp_path: Path) -> HarnessConfig:
        """A config loaded from a real TOML file with one registry entry."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[models.kimi-or]\n"
            'adapter = "openai"\n'
            'base_url = "https://openrouter.ai/api/v1"\n'
            'model = "moonshotai/kimi-k2"\n'
            'api_key = "env:OPENROUTER_API_KEY"\n'
        )
        return load_config(config_path)

    def test_registry_name_wins(self, harbor_agent, registry_config: HarnessConfig):
        model_config = harbor_agent.resolve_model("kimi-or", registry_config)
        assert model_config is registry_config.models["kimi-or"]
        assert model_config.model == "moonshotai/kimi-k2"

    def test_openrouter_form(self, harbor_agent):
        model_config = harbor_agent.resolve_model(
            "openrouter/moonshotai/kimi-k2", HarnessConfig()
        )
        assert model_config.adapter == "openai"
        assert model_config.base_url == "https://openrouter.ai/api/v1"
        assert model_config.model == "moonshotai/kimi-k2"
        assert model_config.api_key.get_secret_value() == "env:OPENROUTER_API_KEY"

    def test_anthropic_form(self, harbor_agent):
        model_config = harbor_agent.resolve_model(
            "anthropic/claude-opus-4-8", HarnessConfig()
        )
        assert model_config.adapter == "anthropic"
        assert model_config.base_url is None
        assert model_config.model == "claude-opus-4-8"
        assert model_config.api_key.get_secret_value() == "env:ANTHROPIC_API_KEY"

    def test_openai_form(self, harbor_agent):
        model_config = harbor_agent.resolve_model("openai/gpt-5.2", HarnessConfig())
        assert model_config.adapter == "openai"
        assert model_config.base_url is None
        assert model_config.model == "gpt-5.2"
        assert model_config.api_key.get_secret_value() == "env:OPENAI_API_KEY"

    @pytest.mark.parametrize(
        "bad", [None, "", "gibberish", "mistral/mistral-large", "openrouter/"]
    )
    def test_unresolvable_lists_forms_and_registry(
        self, harbor_agent, registry_config: HarnessConfig, bad: str | None
    ):
        with pytest.raises(ValueError) as excinfo:
            harbor_agent.resolve_model(bad, registry_config)
        message = str(excinfo.value)
        assert "kimi-or" in message  # available registry names
        for form in ("openrouter/<model>", "anthropic/<model>", "openai/<model>"):
            assert form in message


# -- orchestrator sandbox-override path --------------------------------------


async def test_orchestrator_runs_on_harbor_sandbox(tmp_path: Path):
    """End-to-end: run_task(sandbox=...) drives every bash call through the
    Harbor environment's exec, persists events, and never builds a
    Docker/Local sandbox."""
    env = StubEnvironment()
    sandbox = HarborSandbox(env)
    await sandbox.start()
    adapter = FakeAdapter(bash_script())

    with RunStore(tmp_path / "state.db") as store:
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        run_id, result = await orchestrator.run_task(
            "Echo hello in the container.",
            "fake-model",
            mode=PermissionMode.AUTO,
            adapter_override=adapter,
            sandbox=sandbox,
        )

        assert result.status == "completed"
        assert result.final_text == CLEAN_FINISH

        # The bash tool call reached the Harbor environment.
        commands = [command for command, _ in env.calls]
        assert "echo hello" in commands

        # The system prompt renders the container root, not a host path.
        assert "/app (managed by caller)" in adapter.calls[0].system

        # Events persisted through the normal store path.
        run = store.get_run(run_id)
        assert run is not None and run.status == "completed"
        (lead,) = store.list_agents(run_id)
        kinds = {event.kind for event in store.load_events(lead.id)}
        assert {"message", "tool_call", "tool_result", "decision"} <= kinds


async def test_isolated_spawn_forced_onto_override_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression: under run_task(sandbox=...), spawn_agent(share_sandbox=
    false) must NOT build a host-side Docker/Local sandbox — that workspace
    would be empty, invisible to the benchmark container, and (with no
    Docker daemon) an unsandboxed LocalSandbox on the host. The subagent is
    coerced onto the shared override and the tool result says why."""
    env = StubEnvironment()
    sandbox = HarborSandbox(env)
    await sandbox.start()

    lead_adapter = FakeAdapter(
        [
            resp(
                calls=[
                    ToolCall(
                        id="s1",
                        name="spawn_agent",
                        arguments={
                            "prompt": "Run the child command.",
                            "share_sandbox": False,
                        },
                    )
                ]
            ),
            resp(calls=[ToolCall(id="a1", name="await_agents", arguments={})]),
            resp(CLEAN_FINISH),
        ]
    )
    child_adapter = FakeAdapter(
        [
            resp(
                calls=[
                    ToolCall(
                        id="c1", name="bash", arguments={"command": "echo child"}
                    )
                ]
            ),
            resp("Child task complete. Ran the command; output verified."),
        ]
    )
    adapters = iter([lead_adapter, child_adapter])

    def forbidden_create_sandbox(self, workspace, *, warn=True):
        raise AssertionError(
            "sandbox override in effect: _create_sandbox must not be called "
            "for an isolated subagent spawn"
        )

    monkeypatch.setattr(Orchestrator, "_create_sandbox", forbidden_create_sandbox)

    with RunStore(tmp_path / "state.db") as store:
        orchestrator = Orchestrator(HarnessConfig(home=tmp_path / "home"), store)
        run_id, result = await orchestrator.run_task(
            "Fan out inside the container.",
            "fake-model",
            mode=PermissionMode.AUTO,
            adapter_override=lambda: next(adapters),
            sandbox=sandbox,
        )

        assert result.status == "completed"

        # The child's bash call ran inside the Harbor environment (the
        # shared override), not a fresh host sandbox.
        assert "echo child" in [command for command, _ in env.calls]

        lead, child = store.list_agents(run_id)
        assert child.status == "completed"

        # The spawn tool result told the model isolation was unavailable.
        spawn_results = [
            event.payload["content"]
            for event in store.load_events(lead.id)
            if event.kind == "tool_result"
            and event.payload["tool_call_id"] == "s1"
        ]
        assert len(spawn_results) == 1
        assert child.id in spawn_results[0]
        assert "share_sandbox=false was ignored" in spawn_results[0]


# -- HarnessAgent.run --------------------------------------------------------


def _agent_context() -> SimpleNamespace:
    """Duck-type of harbor.models.agent.context.AgentContext."""
    return SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata=None,
    )


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HOME (and any pre-set HARNESS_HOME) away from the real user."""
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "pre-existing"))
    return tmp_path


#: These end-to-end tests construct agents without an on-disk Harbor trial
#: layout (no <trial_dir>/config.json), so the wall-clock derivation
#: legitimately warns and disables wind-down; only the dedicated deadline
#: tests below assert on that warning.
@pytest.mark.filterwarnings("ignore:could not derive")
class TestHarnessAgentRun:
    async def test_run_end_to_end(
        self,
        harbor_agent,
        isolated_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A full trial against the stub environment: the goal's bash call
        lands in environment.exec, files/state live under logs_dir, and the
        AgentContext is populated from the AgentResult."""
        adapter = FakeAdapter(bash_script())
        monkeypatch.setattr(harbor_agent, "get_adapter", lambda config: adapter)

        logs_dir = isolated_home / "logs"
        agent = harbor_agent.HarnessAgent(
            logs_dir=logs_dir, model_name="openai/gpt-5.2"
        )
        env = StubEnvironment()
        context = _agent_context()

        await agent.run("Echo hello in the container.", env, context)

        # The bash call went through the Harbor environment.
        assert "echo hello" in [command for command, _ in env.calls]

        # Per-trial isolation: state landed under logs_dir, not ~/.harness.
        harness_home = logs_dir / "harness-home"
        assert (harness_home / "state.db").is_file()

        # Context populated from the AgentResult usage. Harbor defines
        # n_input_tokens as input tokens INCLUDING cache, so the bridge
        # sums input (15) + cache reads (3) + cache writes (4).
        assert context.n_input_tokens == 22
        assert context.n_output_tokens == 3
        assert context.n_cache_tokens == 3
        assert context.cost_usd is None
        assert context.metadata["status"] == "completed"
        assert context.metadata["turns"] == 2
        assert context.metadata["final_text"] == CLEAN_FINISH
        assert context.metadata["harness_home"] == str(harness_home)
        assert context.metadata["run_id"]
        assert "error" not in context.metadata

    async def test_run_does_not_mutate_process_environment(
        self,
        harbor_agent,
        isolated_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Regression: run() must not set os.environ['HARNESS_HOME'] —
        Harbor runs trials concurrently in one process, so a process-global
        assignment is last-writer-wins and can point trial A's
        env-consulting code at trial B's home. Explicit config carries the
        per-trial home instead."""
        import os

        adapter = FakeAdapter(bash_script())
        monkeypatch.setattr(harbor_agent, "get_adapter", lambda config: adapter)

        preexisting = str(isolated_home / "pre-existing")
        assert os.environ["HARNESS_HOME"] == preexisting  # fixture setup

        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs", model_name="openai/gpt-5.2"
        )
        await agent.run("Echo hello in the container.", StubEnvironment(), _agent_context())

        assert os.environ["HARNESS_HOME"] == preexisting

    def test_populate_context_counts_cache_in_input_tokens(self, harbor_agent):
        """Regression: Harbor's AgentContext.n_input_tokens is 'input
        tokens including cache' (its claude_code agent sums input +
        cache reads + cache writes), but the harness Usage convention
        is that input_tokens excludes cache traffic (for every adapter)
        — the bridge must normalize by summing all three at the
        reporting boundary."""
        from harness.loop import AgentResult

        context = _agent_context()
        result = AgentResult(
            status="completed",
            final_text="done",
            usage=Usage(
                input_tokens=100,
                output_tokens=7,
                cache_read_tokens=9000,
                cache_write_tokens=250,
            ),
            turns=3,
        )
        harbor_agent.HarnessAgent._populate_context(
            context,
            harness_home=Path("/tmp/h"),
            run_id="r1",
            result=result,
            error=None,
        )
        assert context.n_input_tokens == 100 + 9000 + 250
        assert context.n_output_tokens == 7
        assert context.n_cache_tokens == 9000

    def test_populate_context_does_not_double_count_openai_cached_tokens(
        self, harbor_agent
    ):
        """Regression: the OpenAI API's prompt_tokens INCLUDES cached
        tokens (prompt_tokens_details.cached_tokens is a subset). The
        openai_compat adapter normalizes to the cache-exclusive Usage
        convention, so the boundary sum in _populate_context must yield
        exactly the provider's prompt total — not prompt + cached (the
        double-count this guards against)."""
        from types import SimpleNamespace as NS

        from harness.adapters.openai_compat import from_openai_response
        from harness.loop import AgentResult

        prompt_tokens = 5000
        cached_tokens = 4200  # subset of prompt_tokens per the OpenAI API
        usage = from_openai_response(
            NS(
                choices=[
                    NS(
                        message=NS(content="done", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=NS(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=9,
                    prompt_tokens_details=NS(cached_tokens=cached_tokens),
                ),
            )
        ).usage

        context = _agent_context()
        result = AgentResult(
            status="completed", final_text="done", usage=usage, turns=1
        )
        harbor_agent.HarnessAgent._populate_context(
            context,
            harness_home=Path("/tmp/h"),
            run_id="r1",
            result=result,
            error=None,
        )
        assert context.n_input_tokens == prompt_tokens
        assert context.n_input_tokens != prompt_tokens + cached_tokens
        assert context.n_cache_tokens == cached_tokens
        assert context.n_output_tokens == 9

    async def test_run_populates_context_and_reraises_on_crash(
        self,
        harbor_agent,
        isolated_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A crash mid-run still lands metadata (with the error) before the
        exception propagates to Harbor."""
        adapter = FakeAdapter([])
        monkeypatch.setattr(harbor_agent, "get_adapter", lambda config: adapter)

        async def explode(self, *args, **kwargs):
            raise RuntimeError("provider melted")

        monkeypatch.setattr(Orchestrator, "run_task", explode)

        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs", model_name="openai/gpt-5.2"
        )
        context = _agent_context()
        with pytest.raises(RuntimeError, match="provider melted"):
            await agent.run("goal", StubEnvironment(), context)

        assert context.metadata["status"] == "error"
        assert context.metadata["error"] == "RuntimeError: provider melted"
        assert context.n_input_tokens is None  # no result to report

    async def test_budgets_read_from_extra_env(
        self, harbor_agent, isolated_home: Path
    ):
        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs",
            model_name="openai/gpt-5.2",
            extra_env={"HARNESS_MAX_TURNS": "7"},
        )
        assert agent._int_setting("HARNESS_MAX_TURNS", 80) == 7
        assert agent._int_setting("HARNESS_MAX_TOKENS", 2_000_000) == 2_000_000

    async def test_budgets_fall_back_from_environ(
        self, harbor_agent, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HARNESS_MAX_TOKENS", "12345")
        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs", model_name="openai/gpt-5.2"
        )
        assert agent._int_setting("HARNESS_MAX_TOKENS", 2_000_000) == 12345

    async def test_bad_budget_value_warns_and_defaults(
        self, harbor_agent, isolated_home: Path
    ):
        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs",
            model_name="openai/gpt-5.2",
            extra_env={"HARNESS_MAX_TURNS": "lots"},
        )
        with pytest.warns(UserWarning, match="not an integer"):
            assert agent._int_setting("HARNESS_MAX_TURNS", 80) == 80

    def test_registry_resolution_reads_users_real_config(
        self, harbor_agent, isolated_home: Path
    ):
        """_load_user_config reads ~/.harness/config.toml explicitly, so
        registry names survive the per-trial HARNESS_HOME redirect."""
        user_harness = isolated_home / "fakehome" / ".harness"
        user_harness.mkdir(parents=True)
        (user_harness / "config.toml").write_text(
            '[models.mine]\nadapter = "fake"\nmodel = "whatever"\n'
        )
        agent = harbor_agent.HarnessAgent(
            logs_dir=isolated_home / "logs", model_name="mine"
        )
        config = agent._load_user_config()
        assert "mine" in config.models
        resolved = harbor_agent.resolve_model("mine", config)
        assert resolved.adapter == "fake"

    def test_name_and_version(self, harbor_agent):
        assert harbor_agent.HarnessAgent.name() == "agent-harness"
        agent = harbor_agent.HarnessAgent(logs_dir=Path("/tmp/x"))
        assert isinstance(agent.version(), str)


# -- wall-clock deadline derivation ------------------------------------------

#: Verbatim GitTaskId-shaped trial config.json from the scored run
#: jobs/full-tb2-kimi-m8a/2026-07-23__16-31-46/adaptive-rejection-sampler__WinxnwH.
_GIT_CONFIG_JSON = """{
    "task": {
        "path": "adaptive-rejection-sampler",
        "git_url": "https://github.com/laude-institute/terminal-bench-2.git",
        "git_commit_id": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
        "source": "terminal-bench"
    },
    "trial_name": "adaptive-rejection-sampler__WinxnwH",
    "trials_dir": "jobs/full-tb2-kimi-m8a/2026-07-23__16-31-46",
    "agent": {
        "name": "harness.integrations.harbor_agent:HarnessAgent",
        "model_name": "kimi-or"
    },
    "job_id": "94cec2ee-c444-4812-b972-a335a74b67ad"
}"""


def _task_toml(timeout_sec: float | None = 2400.0, extra: str = "") -> str:
    """A real-shaped TB2 task.toml (modeled on the cached compile-compcert).

    ``timeout_sec=None`` omits the ``[agent]`` table entirely; ``extra`` is
    appended verbatim (e.g. ``[[steps]]`` blocks).
    """
    agent_table = (
        f"[agent]\ntimeout_sec = {timeout_sec}\n\n"
        if timeout_sec is not None
        else ""
    )
    return (
        'version = "1.0"\n\n'
        "[metadata]\n"
        'author_name = "Test Author"\n'
        'author_email = "test@example.com"\n'
        'difficulty = "medium"\n'
        'category = "system-administration"\n\n'
        "[verifier]\n"
        "timeout_sec = 2400.0\n\n"
        f"{agent_table}"
        "[environment]\n"
        "build_timeout_sec = 600.0\n"
        'docker_image = "example/task-image:20251031"\n'
        "cpus = 2\n"
        'memory = "4G"\n'
        f"{extra}"
    )


@pytest.fixture
def trial_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Builder for an on-disk Harbor trial directory.

    Lays out ``<trial_dir>/config.json`` and ``<trial_dir>/agent`` (the
    ``logs_dir`` Harbor hands the agent), plus a task-cache dir the stub
    ``get_local_path()`` resolves git-shaped configs into. Returns the
    ``logs_dir`` to construct the agent with.
    """

    def build(
        config_text: str = _GIT_CONFIG_JSON,
        task_toml: str | None = None,
        *,
        write_config: bool = True,
        create_task_dir: bool = True,
    ) -> Path:
        trial_dir = tmp_path / "trials" / "adaptive-rejection-sampler__WinxnwH"
        logs_dir = trial_dir / "agent"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if write_config:
            (trial_dir / "config.json").write_text(config_text)
        cache_dir = tmp_path / "task-cache"
        monkeypatch.setattr(_StubTaskConfig, "git_cache_dir", cache_dir)
        if create_task_dir:
            task_dir = cache_dir / "adaptive-rejection-sampler"
            task_dir.mkdir(parents=True, exist_ok=True)
            if task_toml is not None:
                (task_dir / "task.toml").write_text(task_toml)
        return logs_dir

    return build


class TestResolveDeadlineMath:
    """The pure replication of Harbor's Trial._compute_agent_timeout_sec +
    _resolve_timeout_sec math. Budgets vary across the real TB2 range
    (600–12000s) — never a single hardcoded value."""

    @pytest.mark.parametrize(
        ("base", "override", "max_sec", "agent_mult", "global_mult", "want"),
        [
            # Base only: passes through untouched.
            (2400.0, None, None, None, 1.0, 2400.0),
            # Config-level override wins over the task's base...
            (2400.0, 600.0, None, None, 1.0, 600.0),
            # ...even upward (it replaces the base, it is not a clamp).
            (600.0, 12000.0, None, None, 1.0, 12000.0),
            # An override also covers a task with no timeout of its own.
            (None, 3600.0, None, None, 1.0, 3600.0),
            # max_timeout_sec clamps.
            (12000.0, None, 3600.0, None, 1.0, 3600.0),
            # Global multiplier applies after the clamp.
            (12000.0, None, 2400.0, None, 1.5, 3600.0),
            # Global multiplier alone.
            (600.0, None, None, None, 2.0, 1200.0),
            # Agent multiplier beats the global one.
            (2400.0, None, None, 2.0, 0.5, 4800.0),
        ],
    )
    def test_math_table(
        self, harbor_agent, base, override, max_sec, agent_mult, global_mult, want
    ):
        got = harbor_agent._resolve_deadline(
            base=base,
            override=override,
            max_sec=max_sec,
            agent_multiplier=agent_mult,
            global_multiplier=global_mult,
        )
        assert got == want

    def test_no_timeout_anywhere_is_none(self, harbor_agent):
        """Neither the task nor the config sets a timeout: Harbor enforces
        no deadline, so the derivation must yield None, not 0."""
        assert (
            harbor_agent._resolve_deadline(
                base=None,
                override=None,
                max_sec=3600.0,
                agent_multiplier=2.0,
                global_multiplier=1.5,
            )
            is None
        )


class TestParseTimeoutSeconds:
    @pytest.mark.parametrize(
        ("raw", "want"),
        [(2400, 2400.0), (600.5, 600.5), ("1800", 1800.0), (" 3600 ", 3600.0)],
    )
    def test_usable_values_parse(self, harbor_agent, raw, want):
        assert (
            harbor_agent._parse_timeout_seconds(raw, source="agent_timeout_sec")
            == want
        )

    def test_none_passes_through_silently(self, harbor_agent):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                harbor_agent._parse_timeout_seconds(
                    None, source="agent_timeout_sec"
                )
                is None
            )

    @pytest.mark.parametrize("raw", ["soon", "", -5, 0, True, [1800]])
    def test_garbage_warns_and_yields_none(self, harbor_agent, raw):
        with pytest.warns(UserWarning, match="not a positive number"):
            assert (
                harbor_agent._parse_timeout_seconds(
                    raw, source="agent_timeout_sec"
                )
                is None
            )


class TestWallClockPrecedence:
    """_wall_clock_budget resolution order: env override > constructor
    kwarg > on-disk derivation > None."""

    @pytest.fixture(autouse=True)
    def no_ambient_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARNESS_WALL_CLOCK_SECONDS", raising=False)

    def test_constructor_kwarg_accepted_and_stored(self, harbor_agent):
        agent = harbor_agent.HarnessAgent(
            logs_dir=Path("/tmp/x"), agent_timeout_sec="2400"
        )
        assert agent.agent_timeout_sec == 2400.0
        assert agent._wall_clock_budget() == 2400.0

    def test_constructor_garbage_warns_and_is_ignored(self, harbor_agent):
        with pytest.warns(UserWarning, match="agent_timeout_sec"):
            agent = harbor_agent.HarnessAgent(
                logs_dir=Path("/tmp/x"), agent_timeout_sec="soon"
            )
        assert agent.agent_timeout_sec is None

    def test_env_override_beats_kwarg(self, harbor_agent):
        agent = harbor_agent.HarnessAgent(
            logs_dir=Path("/tmp/x"),
            agent_timeout_sec=2400,
            extra_env={"HARNESS_WALL_CLOCK_SECONDS": "1234"},
        )
        assert agent._wall_clock_budget() == 1234.0

    @pytest.mark.parametrize("raw", ["0", "-5", "nan", "soon"])
    def test_env_override_non_positive_or_garbage_falls_through_to_kwarg(
        self, harbor_agent, raw
    ):
        """A non-positive/garbage env override must not become the budget:
        Deadline(0).remaining() sits below the stop floor and would hard-stop
        the trial on turn 1 with zero adapter calls. It warns and falls
        through to the kwarg path instead."""
        agent = harbor_agent.HarnessAgent(
            logs_dir=Path("/tmp/x"),
            agent_timeout_sec=2400,
            extra_env={"HARNESS_WALL_CLOCK_SECONDS": raw},
        )
        with pytest.warns(
            UserWarning, match="HARNESS_WALL_CLOCK_SECONDS.*not a positive"
        ):
            assert agent._wall_clock_budget() == 2400.0

    def test_env_override_non_positive_falls_through_to_derivation(
        self, harbor_agent, trial_layout
    ):
        logs_dir = trial_layout(task_toml=_task_toml(2400.0))
        agent = harbor_agent.HarnessAgent(
            logs_dir=logs_dir,
            extra_env={"HARNESS_WALL_CLOCK_SECONDS": "0"},
        )
        with pytest.warns(
            UserWarning, match="HARNESS_WALL_CLOCK_SECONDS.*not a positive"
        ):
            assert agent._wall_clock_budget() == 2400.0

    def test_kwarg_beats_derivation(self, harbor_agent, trial_layout):
        logs_dir = trial_layout(task_toml=_task_toml(2400.0))
        agent = harbor_agent.HarnessAgent(
            logs_dir=logs_dir, agent_timeout_sec=600
        )
        assert agent._wall_clock_budget() == 600.0

    def test_derivation_when_nothing_explicit(self, harbor_agent, trial_layout):
        logs_dir = trial_layout(task_toml=_task_toml(2400.0))
        agent = harbor_agent.HarnessAgent(logs_dir=logs_dir)
        assert agent._wall_clock_budget() == 2400.0

    def test_no_source_at_all_is_none(self, harbor_agent, tmp_path: Path):
        agent = harbor_agent.HarnessAgent(logs_dir=tmp_path / "agent")
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._wall_clock_budget() is None


class TestDeriveHarborDeadline:
    """The on-disk derivation against real-shaped trial state."""

    def _agent(self, harbor_agent, logs_dir: Path):
        return harbor_agent.HarnessAgent(logs_dir=logs_dir)

    def test_git_shaped_real_config(self, harbor_agent, trial_layout):
        """The scored run's actual GitTaskId-shaped config.json resolves
        through the task cache to task.toml's agent timeout."""
        logs_dir = trial_layout(_GIT_CONFIG_JSON, _task_toml(2400.0))
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 2400.0

    def test_local_task_id_shaped_config(self, harbor_agent, tmp_path: Path):
        """LocalTaskId shape: task.path set, no git fields — resolved
        directly, no task cache involved."""
        task_dir = tmp_path / "local-task"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(_task_toml(12000.0))
        trial_dir = tmp_path / "trials" / "local-task__abc1234"
        logs_dir = trial_dir / "agent"
        logs_dir.mkdir(parents=True)
        (trial_dir / "config.json").write_text(
            json.dumps(
                {
                    "task": {"path": str(task_dir)},
                    "trial_name": "local-task__abc1234",
                    "agent": {
                        "name": (
                            "harness.integrations.harbor_agent:HarnessAgent"
                        ),
                        "model_name": "kimi-or",
                    },
                }
            )
        )
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 12000.0

    def test_global_multiplier_applies(self, harbor_agent, trial_layout):
        config = json.loads(_GIT_CONFIG_JSON)
        config["timeout_multiplier"] = 1.5
        logs_dir = trial_layout(json.dumps(config), _task_toml(2400.0))
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 3600.0

    def test_agent_multiplier_beats_global(self, harbor_agent, trial_layout):
        config = json.loads(_GIT_CONFIG_JSON)
        config["timeout_multiplier"] = 0.5
        config["agent_timeout_multiplier"] = 2.0
        logs_dir = trial_layout(json.dumps(config), _task_toml(600.0))
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 1200.0

    def test_config_override_timeout_wins(self, harbor_agent, trial_layout):
        config = json.loads(_GIT_CONFIG_JSON)
        config["agent"]["override_timeout_sec"] = 600.0
        logs_dir = trial_layout(json.dumps(config), _task_toml(2400.0))
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 600.0

    def test_max_timeout_clamps(self, harbor_agent, trial_layout):
        config = json.loads(_GIT_CONFIG_JSON)
        config["agent"]["max_timeout_sec"] = 3600.0
        logs_dir = trial_layout(json.dumps(config), _task_toml(12000.0))
        agent = self._agent(harbor_agent, logs_dir)
        assert agent._derive_harbor_deadline() == 3600.0

    def test_missing_config_json_warns_none(self, harbor_agent, trial_layout):
        logs_dir = trial_layout(task_toml=_task_toml(), write_config=False)
        agent = self._agent(harbor_agent, logs_dir)
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._derive_harbor_deadline() is None

    def test_missing_task_toml_warns_none(self, harbor_agent, trial_layout):
        logs_dir = trial_layout(task_toml=None)  # task dir exists, no toml
        agent = self._agent(harbor_agent, logs_dir)
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._derive_harbor_deadline() is None

    def test_missing_cache_dir_warns_none(self, harbor_agent, trial_layout):
        """The task was never materialized into the cache at all."""
        logs_dir = trial_layout(create_task_dir=False)
        agent = self._agent(harbor_agent, logs_dir)
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._derive_harbor_deadline() is None

    def test_malformed_toml_warns_none(self, harbor_agent, trial_layout):
        logs_dir = trial_layout(task_toml="[agent\ntimeout_sec = oops")
        agent = self._agent(harbor_agent, logs_dir)
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._derive_harbor_deadline() is None

    def test_trial_config_validation_error_warns_none(
        self, harbor_agent, trial_layout
    ):
        logs_dir = trial_layout('{"task": 42}', _task_toml())
        agent = self._agent(harbor_agent, logs_dir)
        with pytest.warns(UserWarning, match="could not derive"):
            assert agent._derive_harbor_deadline() is None

    def test_multi_step_task_is_none_without_warning(
        self, harbor_agent, trial_layout
    ):
        """[[steps]] means per-step timeouts Harbor does not sum into one
        agent deadline — an expected shape, so silent None, not a warning."""
        steps = (
            "[[steps]]\n"
            'instruction_path = "steps/one/instruction.md"\n\n'
            "[[steps]]\n"
            'instruction_path = "steps/two/instruction.md"\n'
        )
        logs_dir = trial_layout(task_toml=_task_toml(2400.0, extra=steps))
        agent = self._agent(harbor_agent, logs_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert agent._derive_harbor_deadline() is None

    def test_no_agent_timeout_in_task_is_none_without_warning(
        self, harbor_agent, trial_layout
    ):
        """A task.toml with no [agent] timeout means Harbor enforces no
        deadline — legitimate, so silent None."""
        logs_dir = trial_layout(task_toml=_task_toml(timeout_sec=None))
        agent = self._agent(harbor_agent, logs_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert agent._derive_harbor_deadline() is None


class TestRunUsesDerivedDeadline:
    async def test_run_threads_derived_budget_into_run_task(
        self,
        harbor_agent,
        isolated_home: Path,
        trial_layout,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """End-to-end wiring: run() derives the wall clock first and hands
        Orchestrator.run_task a Deadline anchored at run() entry — the
        single source of truth on the Harbor path, so
        Budgets.wall_clock_seconds is deliberately left unset (it would
        re-anchor inside the loop and miss the pre-loop setup time)."""
        from harness.loop import AgentResult

        monkeypatch.delenv("HARNESS_WALL_CLOCK_SECONDS", raising=False)
        adapter = FakeAdapter([])
        monkeypatch.setattr(harbor_agent, "get_adapter", lambda config: adapter)

        captured: dict[str, object] = {}

        async def fake_run_task(self, goal, model_name, **kwargs):
            captured["budgets"] = kwargs["budgets"]
            captured["deadline"] = kwargs["deadline"]
            return "r1", AgentResult(
                status="completed", final_text="done", usage=Usage(), turns=1
            )

        monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

        logs_dir = trial_layout(task_toml=_task_toml(2400.0))
        agent = harbor_agent.HarnessAgent(
            logs_dir=logs_dir, model_name="openai/gpt-5.2"
        )
        context = _agent_context()
        await agent.run("goal", StubEnvironment(), context)

        deadline = captured["deadline"]
        assert deadline.budget == 2400.0
        assert deadline.remaining() <= 2400.0
        assert captured["budgets"].wall_clock_seconds is None
        assert context.metadata["status"] == "completed"
