"""S-108: masking known secrets at the tool boundary.

A secret reaches the transcript by accident — the agent runs `env`, a build
script echoes its config, a stack trace prints a URL with credentials in it.
From there it goes three places at once: the model's context, the persisted
event log, and (for a large result) a spill file on disk. All three outlive
the run, and the event log is the copy nobody thinks to check.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness.persistence import RunStore
from harness.secrets import MIN_MASKABLE_LENGTH, SecretRegistry, SecretTooShortError
from harness.permissions import ToolMeta
from harness.tools.registry import Tool, ToolRegistry
from harness.types import ToolCall, ToolSpec

SECRET = "sk-ant-api03-0123456789abcdefghij"


def echo_tool(text: str) -> Tool:
    """A tool that returns whatever it is told to — an `env` dump, say."""

    async def handler(arguments: dict) -> str:
        return text

    return Tool(
        spec=ToolSpec(name="echo", description="echo", input_schema={"type": "object"}),
        meta=ToolMeta(side_effect=False),
        handler=handler,
    )


class TestAnEmptyRegistryIsTheIdentity:
    """What makes this Lane A. The benchmark path registers nothing, so it
    cannot observe that masking exists — not "the output happens to match",
    but the same string object comes back."""

    def test_S108_mask_returns_the_same_object(self) -> None:
        registry = SecretRegistry()
        text = "nothing to see here"
        assert registry.mask(text) is text

    def test_S108_mask_payload_returns_the_same_object(self) -> None:
        registry = SecretRegistry()
        payload = {"command": "ls", "nested": [{"x": "y"}]}
        assert registry.mask_payload(payload) is payload

    def test_S108_empty_is_reported(self) -> None:
        assert SecretRegistry().empty
        registry = SecretRegistry()
        registry.register("k", SECRET)
        assert not registry.empty


class TestShortValuesAreRefused:
    """A four-character 'secret' appears inside ordinary output constantly.
    Replacing it corrupts results far worse than the leak it prevents."""

    def test_S108_a_short_value_raises_rather_than_being_ignored(self) -> None:
        # Loudly, not silently: quietly declining to mask something the caller
        # asked to have masked is the exact failure this module exists to stop.
        with pytest.raises(SecretTooShortError, match="would corrupt"):
            SecretRegistry().register("tiny", "abc")

    def test_S108_the_boundary_is_where_it_says_it_is(self) -> None:
        registry = SecretRegistry()
        with pytest.raises(SecretTooShortError):
            registry.register("just-under", "x" * (MIN_MASKABLE_LENGTH - 1))
        assert registry.register("exactly", "y" * MIN_MASKABLE_LENGTH)

    def test_S108_an_absent_key_is_not_an_error(self) -> None:
        # An unset API key is the common case, not a failure.
        registry = SecretRegistry()
        assert registry.register("unset", None) is False
        assert registry.register("blank", "") is False
        assert registry.empty


class TestMasking:
    def test_S108_a_secret_is_replaced_by_its_label(self) -> None:
        registry = SecretRegistry()
        registry.register("anthropic-api-key", SECRET)
        masked = registry.mask(f"export KEY={SECRET} && curl ...")
        assert SECRET not in masked
        assert "[redacted:anthropic-api-key]" in masked

    def test_S108_every_occurrence_is_replaced(self) -> None:
        registry = SecretRegistry()
        registry.register("k", SECRET)
        assert SECRET not in registry.mask(f"{SECRET} and again {SECRET}")

    def test_S108_a_containing_secret_masks_as_one_value(self) -> None:
        # Registered short-first, the shorter value would mask the inside of
        # the longer one and leave a partially-redacted fragment — which is
        # worse than no masking, because it looks safe.
        outer = SECRET + "-with-suffix-1234"
        registry = SecretRegistry()
        registry.register("inner", SECRET)
        registry.register("outer", outer)
        masked = registry.mask(f"value={outer}")
        assert masked == "value=[redacted:outer]"

    def test_S108_registering_the_same_value_twice_is_idempotent(self) -> None:
        registry = SecretRegistry()
        assert registry.register("first", SECRET)
        assert registry.register("second", SECRET) is False

    def test_S108_nested_payloads_are_masked(self) -> None:
        registry = SecretRegistry()
        registry.register("k", SECRET)
        payload = {"cmd": f"run {SECRET}", "items": [{"deep": SECRET}], "n": 3}
        masked = registry.mask_payload(payload)
        assert SECRET not in json.dumps(masked)
        assert masked["n"] == 3


class TestTheToolBoundary:
    """Acceptance (1): a command that echoes a configured secret returns a
    mask."""

    async def test_S108_a_tool_result_is_masked(self) -> None:
        registry = SecretRegistry()
        registry.register("api-key", SECRET)
        tools = ToolRegistry(secrets=registry)
        tools.register(echo_tool(f"API_KEY={SECRET}"))
        result = await tools.dispatch(ToolCall(id="1", name="echo", arguments={}))
        assert SECRET not in result.content
        assert "[redacted:api-key]" in result.content

    async def test_S108_without_a_registry_nothing_changes(self) -> None:
        tools = ToolRegistry()
        tools.register(echo_tool(f"API_KEY={SECRET}"))
        result = await tools.dispatch(ToolCall(id="1", name="echo", arguments={}))
        assert result.content == f"API_KEY={SECRET}"

    async def test_S108_the_spill_file_is_masked_too(self) -> None:
        # The masking has to happen before the spill, not just before the
        # model sees it: `_spill_full_result` writes the FULL result into the
        # sandbox, so masking afterwards keeps the model's copy clean and
        # leaves the secret sitting on disk.
        from harness.tools.registry import MAX_RESULT_BYTES

        spilled: dict[str, str] = {}

        async def spill(content: str) -> str:
            spilled["content"] = content
            return "/tmp/spilled.txt"

        registry = SecretRegistry()
        registry.register("api-key", SECRET)
        tools = ToolRegistry(spill=spill, secrets=registry)
        huge = f"{SECRET}\n" + ("x" * (MAX_RESULT_BYTES + 100))
        tools.register(echo_tool(huge))
        await tools.dispatch(ToolCall(id="1", name="echo", arguments={}))
        assert spilled, "precondition: the result must actually spill"
        assert SECRET not in spilled["content"]


class TestThePersistedLog:
    """Acceptance (2): masking applies before persistence, so the event log is
    clean too. It outlives the run and nobody re-reads it."""

    def _store(self, secrets: SecretRegistry | None):
        directory = Path(tempfile.mkdtemp())
        store = RunStore(directory / "state.db", secrets=secrets)
        run_id = store.create_run("goal", "model", "auto")
        agent_id = store.create_agent(run_id, "goal")
        return store, agent_id

    def test_S108_a_persisted_event_is_masked(self) -> None:
        registry = SecretRegistry()
        registry.register("api-key", SECRET)
        store, agent_id = self._store(registry)
        store.append_event(agent_id, "tool_result", {"content": f"KEY={SECRET}"})
        (event,) = store.load_events(agent_id)
        assert SECRET not in json.dumps(event.payload)

    def test_S108_events_a_tool_never_produced_are_masked(self) -> None:
        # Masking at the tool boundary alone would miss the model's own
        # message, a nudge quoting a command, a run_error carrying a URL with
        # credentials in it.
        registry = SecretRegistry()
        registry.register("api-key", SECRET)
        store, agent_id = self._store(registry)
        store.append_event(agent_id, "message", {"content": f"I saw {SECRET}"})
        store.append_event(agent_id, "run_error", {"error": f"https://u:{SECRET}@h"})
        for event in store.load_events(agent_id):
            assert SECRET not in json.dumps(event.payload)

    def test_S108_without_a_registry_the_log_is_unchanged(self) -> None:
        store, agent_id = self._store(None)
        store.append_event(agent_id, "tool_result", {"content": f"KEY={SECRET}"})
        (event,) = store.load_events(agent_id)
        assert event.payload["content"] == f"KEY={SECRET}"


@pytest.fixture
def no_env_credentials(monkeypatch):
    """Clear the provider credential variables.

    These tests exercise the *config* path. Without this they pass or fail
    depending on whether the developer happens to have ANTHROPIC_API_KEY
    exported, which is the kind of hidden dependency that makes a suite
    green on one machine and red on another.
    """
    from harness.secrets import CREDENTIAL_ENV_VARS

    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestEnvironmentCredentials:
    """The setup that produced an empty registry while the key sat in
    os.environ: config omits `api_key`, the provider SDK reads its own
    variable, and every LocalSandbox subprocess inherits it. An agent running
    `env` -- the opening scenario of the module docstring -- leaked it with
    masking a silent no-op."""

    def test_S108_an_sdk_fallback_variable_is_registered(
        self, monkeypatch, no_env_credentials
    ) -> None:
        from harness.config import HarnessConfig
        from harness.orchestrator import build_secret_registry

        monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
        registry = build_secret_registry(HarnessConfig(home=Path("/tmp/x")))
        assert not registry.empty
        assert SECRET not in registry.mask(f"ANTHROPIC_API_KEY={SECRET}")

    def test_S108_a_short_environment_value_is_skipped_not_raised(
        self, monkeypatch, no_env_credentials
    ) -> None:
        from harness.config import HarnessConfig
        from harness.orchestrator import build_secret_registry

        monkeypatch.setenv("OPENAI_API_KEY", "short")
        assert build_secret_registry(HarnessConfig(home=Path("/tmp/x"))).empty

    def test_S108_the_variable_list_is_explicit_not_a_pattern(self) -> None:
        # Matching on name shape (*_API_KEY) would sweep in variables whose
        # values are short or structural, and a false positive corrupts tool
        # output everywhere the value appears.
        from harness.secrets import CREDENTIAL_ENV_VARS

        assert all(name.isupper() for name in CREDENTIAL_ENV_VARS)
        assert "ANTHROPIC_API_KEY" in CREDENTIAL_ENV_VARS


class TestBuiltFromConfig:
    def test_S108_a_config_without_keys_yields_an_empty_registry(self, no_env_credentials) -> None:
        from harness.config import HarnessConfig
        from harness.orchestrator import build_secret_registry

        assert build_secret_registry(HarnessConfig(home=Path("/tmp/x"))).empty

    def test_S108_a_resolvable_key_is_registered(self, monkeypatch, no_env_credentials) -> None:
        from harness.config import HarnessConfig, ModelConfig
        from harness.orchestrator import build_secret_registry

        monkeypatch.setenv("FAKE_KEY_FOR_TEST", SECRET)
        config = HarnessConfig(
            home=Path("/tmp/x"),
            models={"m": ModelConfig(adapter="openai", model="x",
                                     api_key="env:FAKE_KEY_FOR_TEST")},
        )
        registry = build_secret_registry(config)
        assert not registry.empty
        assert "[redacted:m-api-key]" in registry.mask(f"key={SECRET}")

    def test_S108_an_unresolvable_key_does_not_fail_the_run(self, no_env_credentials) -> None:
        # A run must not die because redaction could not be set up.
        from harness.config import HarnessConfig, ModelConfig
        from harness.orchestrator import build_secret_registry

        config = HarnessConfig(
            home=Path("/tmp/x"),
            models={"m": ModelConfig(adapter="openai", model="x",
                                     api_key="env:DEFINITELY_NOT_SET_ANYWHERE")},
        )
        assert build_secret_registry(config).empty

    def test_S108_a_too_short_key_is_skipped_not_raised(self, monkeypatch, no_env_credentials) -> None:
        from harness.config import HarnessConfig, ModelConfig
        from harness.orchestrator import build_secret_registry

        monkeypatch.setenv("SHORT_KEY_FOR_TEST", "abc")
        config = HarnessConfig(
            home=Path("/tmp/x"),
            models={"m": ModelConfig(adapter="openai", model="x",
                                     api_key="env:SHORT_KEY_FOR_TEST")},
        )
        assert build_secret_registry(config).empty


class TestTheWiringProductionActuallyUses:
    """D1: `RunStore(path, secrets=...)` is a shape no production code builds.

    Every entry point — the CLI, the eval runner, the Harbor bridge — builds
    the store *before* a registry can exist, so the keyword was passed by
    exactly one caller: a test. The persistence half of this spec was real,
    tested, and dead in every actual run. These tests go through
    `Orchestrator(config, store)`, which is what those callers do.
    """

    def _store(self, tmp_path: Path) -> RunStore:
        return RunStore(tmp_path / "state.db")

    def test_S108_constructing_an_orchestrator_masks_the_store(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator

        monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
        store = self._store(tmp_path)
        assert store._secrets is None, "precondition: the store starts unbound"

        Orchestrator(HarnessConfig(home=tmp_path), store)

        run_id = store.create_run("goal", "model", "auto")
        agent_id = store.create_agent(run_id, "goal")
        store.append_event(agent_id, "tool_call", {"arguments": {
            "command": f'curl -H "Authorization: Bearer {SECRET}"'}})
        (event,) = store.load_events(agent_id)
        assert SECRET not in json.dumps(event.payload), (
            "the event log is unmasked through the wiring production uses"
        )

    def test_S108_a_supplied_registry_reaches_the_store(
        self, tmp_path: Path, no_env_credentials
    ) -> None:
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator

        registry = SecretRegistry()
        registry.register("caller-supplied", SECRET)
        store = self._store(tmp_path)
        Orchestrator(HarnessConfig(home=tmp_path), store, secrets=registry)

        run_id = store.create_run("goal", "model", "auto")
        agent_id = store.create_agent(run_id, "goal")
        store.append_event(agent_id, "message", {"content": SECRET})
        (event,) = store.load_events(agent_id)
        assert SECRET not in json.dumps(event.payload)

    def test_S108_an_empty_registry_leaves_the_store_a_no_op(
        self, tmp_path: Path, no_env_credentials
    ) -> None:
        # Neutrality: on the benchmark path there is nothing to redact, and
        # binding an empty registry must not change what is stored.
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator

        store = self._store(tmp_path)
        Orchestrator(HarnessConfig(home=tmp_path), store)
        run_id = store.create_run("goal", "model", "auto")
        agent_id = store.create_agent(run_id, "goal")
        payload = {"content": f"KEY={SECRET}"}
        store.append_event(agent_id, "message", payload)
        (event,) = store.load_events(agent_id)
        assert event.payload == payload


class TestErrorResultsAreMaskedToo:
    """D3: masking sat after `dispatch`'s two `is_error` early returns.

    An exception message routinely carries the thing that failed — a
    CalledProcessError with the command, an HTTP client raising with the
    request URL — and that return reaches the model's context and the event
    log exactly as a successful one does.
    """

    async def test_S108_an_exception_message_is_masked(self) -> None:
        registry = SecretRegistry()
        registry.register("api-key", SECRET)

        async def handler(arguments: dict) -> str:
            raise RuntimeError(f"curl failed: Authorization: Bearer {SECRET}")

        tools = ToolRegistry(secrets=registry)
        tools.register(Tool(
            spec=ToolSpec(name="boom", description="", input_schema={"type": "object"}),
            meta=ToolMeta(side_effect=False),
            handler=handler,
        ))
        result = await tools.dispatch(ToolCall(id="1", name="boom", arguments={}))
        assert result.is_error
        assert SECRET not in result.content

    async def test_S108_the_unknown_tool_message_is_unaffected(self) -> None:
        # It contains only the tool name, so there is nothing to mask; the
        # assertion is that adding masking did not change it.
        tools = ToolRegistry(secrets=SecretRegistry())
        result = await tools.dispatch(ToolCall(id="1", name="nope", arguments={}))
        assert result.content == "unknown tool: 'nope'"


class TestOverlappingSecrets:
    def test_S108_an_overlap_leaves_no_fragment(self) -> None:
        # Sorting longest-first fixes containment only. Two secrets that
        # overlap without containing leave a partially-redacted fragment,
        # which looks safe and is not.
        registry = SecretRegistry()
        registry.register("a", "ABCDEFGHIJKL")
        registry.register("b", "GHIJKLMNOPQR")
        # Exact equality, not "the fragment is absent". Without merging, the
        # spans are replaced right-to-left against indices that the first
        # replacement already invalidated, producing `tok=[redacted:a]d:b]` --
        # which contains neither fragment and would satisfy a weaker
        # assertion while being visibly corrupt.
        assert registry.mask("tok=ABCDEFGHIJKLMNOPQR") == "tok=[redacted:a]"

    def test_S108_repeated_secrets_survive_reverse_replacement(self) -> None:
        # Several occurrences means several spans, and replacing them in the
        # wrong order shifts every index after the first.
        registry = SecretRegistry()
        registry.register("k", SECRET)
        assert registry.mask(f"a={SECRET} b={SECRET} c={SECRET}") == (
            "a=[redacted:k] b=[redacted:k] c=[redacted:k]"
        )

    def test_S108_a_dict_key_is_masked(self) -> None:
        # Model-supplied `tool_call.arguments` is an arbitrary object, so a
        # secret can be a key as easily as a value -- and a key rebuilt
        # verbatim lands in the log in full.
        registry = SecretRegistry()
        registry.register("k", SECRET)
        masked = registry.mask_payload({SECRET: "value", "outer": {SECRET: 1}})
        assert SECRET not in json.dumps(masked), masked
        assert masked["[redacted:k]"] == "value"

    def test_S108_a_tuple_is_masked(self) -> None:
        # A tuple falls through a str/dict/list chain and serialises into the
        # event log as an array, carrying the value with it.
        registry = SecretRegistry()
        registry.register("k", SECRET)
        masked = registry.mask_payload({"args": (SECRET, "x")})
        assert SECRET not in json.dumps(masked)

    def test_S108_non_string_leaves_are_untouched(self) -> None:
        registry = SecretRegistry()
        registry.register("k", SECRET)
        assert registry.mask_payload({"n": 3, "b": True, "z": None}) == (
            {"n": 3, "b": True, "z": None}
        )

    def test_S108_the_registry_never_prints_its_values(self) -> None:
        # ToolDeps-style dataclasses are captured in closures and rendered by
        # pytest --showlocals, rich tracebacks and logging.exception. A
        # default dataclass repr would print live credentials into exactly the
        # places this module exists to keep clean.
        registry = SecretRegistry()
        registry.register("api-key", SECRET)
        assert SECRET not in repr(registry)
        assert "1 value(s)" in repr(registry)
