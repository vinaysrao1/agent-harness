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
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    monkeypatch.setitem(sys.modules, "harbor", harbor)
    monkeypatch.setitem(sys.modules, "harbor.agents", agents)
    monkeypatch.setitem(sys.modules, "harbor.agents.base", agents_base)
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
