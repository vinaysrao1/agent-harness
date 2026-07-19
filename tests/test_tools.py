"""Unit tests for harness.tools.registry and harness.tools.builtin."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.memory.store import FactNotFoundError, MemoryStore
from harness.permissions import ToolMeta
from harness.persistence import RunStore
from harness.sandbox.base import SandboxError
from harness.sandbox.local import LocalSandbox
from harness.skills import SkillLibrary
from harness.context import ContextManager
from harness.tools.builtin import (
    MissingArgumentError,
    add_instruction_tool,
    bash_tool,
    edit_file_tool,
    load_skill_tool,
    memory_read_fact_tool,
    memory_search_tool,
    memory_write_fact_tool,
    read_file_tool,
    search_history_tool,
    task_list_tool,
    task_update_tool,
    write_file_tool,
)
from harness.tools.registry import (
    MAX_RESULT_BYTES,
    DuplicateToolError,
    Tool,
    ToolRegistry,
)
from harness.types import Message, Role, ToolCall, ToolSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(tmp_path / "workspace")


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def run_store(tmp_path: Path):
    store = RunStore(tmp_path / "state.db")
    yield store
    store.close()


@pytest.fixture
def run_id(run_store: RunStore) -> str:
    return run_store.create_run(
        goal="test goal", model="fake", permission_mode="gated"
    )


@pytest.fixture
def skill_library(tmp_path: Path) -> SkillLibrary:
    root = tmp_path / "skills"
    skill_dir = root / "greet"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Says hello politely\n---\n"
        "Say hello and ask how you can help.\n",
        encoding="utf-8",
    )
    return SkillLibrary(root)


def _call(name: str, arguments: dict | None = None, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


def _fake_tool(name: str, handler=None, *, side_effect: bool = False) -> Tool:
    async def default_handler(arguments: dict) -> str:
        return "ok"

    return Tool(
        spec=ToolSpec(
            name=name,
            description="a fake tool for registry tests",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        meta=ToolMeta(side_effect=side_effect),
        handler=handler or default_handler,
    )


def _assert_valid_object_schema(schema: dict) -> None:
    """Minimal structural check that ``schema`` is a JSON Schema object shape."""
    assert schema.get("type") == "object"
    assert isinstance(schema.get("properties"), dict)
    assert isinstance(schema.get("required"), list)
    for key in schema["required"]:
        assert key in schema["properties"], f"required key {key!r} not in properties"
    for prop_name, prop_schema in schema["properties"].items():
        assert isinstance(prop_schema, dict)
        assert "type" in prop_schema, f"property {prop_name!r} missing 'type'"


def _make_context(reminder_interval: int = 1) -> ContextManager:
    """A ContextManager for context-binding tests (stub counter/summarizer)."""

    async def summarize(messages):
        return "stub"

    return ContextManager(
        base_system_prompt="base prompt",
        count_tokens=lambda messages: 0,
        max_context=1_000_000,
        summarize=summarize,
        reminder_interval=reminder_interval,
    )


# ---------------------------------------------------------------------------
# ToolRegistry: register/get/specs
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_register_then_get_returns_same_tool(self):
        registry = ToolRegistry()
        tool = _fake_tool("noop")
        registry.register(tool)
        assert registry.get("noop") is tool

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        with pytest.raises(DuplicateToolError, match="noop"):
            registry.register(_fake_tool("noop"))

    def test_get_unknown_raises_keyerror(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="nope"):
            registry.get("nope")

    def test_specs_returns_every_registered_spec(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("a"))
        registry.register(_fake_tool("b"))
        names = {spec.name for spec in registry.specs()}
        assert names == {"a", "b"}
        assert all(isinstance(spec, ToolSpec) for spec in registry.specs())

    def test_specs_empty_for_fresh_registry(self):
        assert ToolRegistry().specs() == []


# ---------------------------------------------------------------------------
# ToolRegistry.dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_success_returns_content_untouched(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        result = await registry.dispatch(_call("noop"))
        assert result.tool_call_id == "call-1"
        assert result.content == "ok"
        assert result.is_error is False

    async def test_unknown_tool_is_error_not_exception(self):
        registry = ToolRegistry()
        result = await registry.dispatch(_call("does-not-exist", call_id="call-9"))
        assert result.tool_call_id == "call-9"
        assert result.is_error is True
        assert "does-not-exist" in result.content

    async def test_handler_exception_becomes_error_result(self):
        async def boom(arguments: dict) -> str:
            raise RuntimeError("kaboom")

        registry = ToolRegistry()
        registry.register(_fake_tool("boom", handler=boom))
        result = await registry.dispatch(_call("boom"))
        assert result.is_error is True
        assert "kaboom" in result.content
        assert "boom" in result.content
        assert "RuntimeError" in result.content

    async def test_handler_exception_does_not_propagate(self):
        # The whole point of dispatch: a crashing handler must not raise
        # out of dispatch() and take the agent loop down with it.
        async def boom(arguments: dict) -> str:
            raise ValueError("nope")

        registry = ToolRegistry()
        registry.register(_fake_tool("boom", handler=boom))
        result = await registry.dispatch(_call("boom"))  # must not raise
        assert result.is_error is True

    async def test_oversized_result_is_truncated_with_marker(self):
        original_size = MAX_RESULT_BYTES + 5000

        async def huge(arguments: dict) -> str:
            return "x" * original_size

        registry = ToolRegistry()
        registry.register(_fake_tool("huge", handler=huge))
        result = await registry.dispatch(_call("huge"))
        assert result.is_error is False
        assert len(result.content.encode("utf-8")) < MAX_RESULT_BYTES + 200
        assert "truncated" in result.content
        assert str(original_size) in result.content

    async def test_small_result_not_truncated(self):
        registry = ToolRegistry()
        registry.register(_fake_tool("noop"))
        result = await registry.dispatch(_call("noop"))
        assert "truncated" not in result.content

    async def test_result_at_exact_limit_not_truncated(self):
        async def exact(arguments: dict) -> str:
            return "y" * MAX_RESULT_BYTES

        registry = ToolRegistry()
        registry.register(_fake_tool("exact", handler=exact))
        result = await registry.dispatch(_call("exact"))
        assert "truncated" not in result.content
        assert len(result.content) == MAX_RESULT_BYTES


# ---------------------------------------------------------------------------
# Sandbox tools: bash / read_file / write_file / edit_file
# ---------------------------------------------------------------------------


class TestBashTool:
    def test_schema_is_valid(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        assert tool.spec.name == "bash"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert "command" in tool.spec.input_schema["required"]

    def test_side_effect_false(self, sandbox: LocalSandbox):
        assert bash_tool(sandbox).meta.side_effect is False

    async def test_runs_command_and_formats_output(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        output = await tool.handler({"command": "echo hello"})
        assert "exit code: 0" in output
        assert "hello" in output
        assert "stdout" in output
        assert "stderr" in output

    async def test_captures_nonzero_exit_and_stderr(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        output = await tool.handler({"command": "echo problem 1>&2; exit 3"})
        assert "exit code: 3" in output
        assert "problem" in output

    async def test_missing_command_raises(self, sandbox: LocalSandbox):
        tool = bash_tool(sandbox)
        with pytest.raises(MissingArgumentError):
            await tool.handler({})

    async def test_end_to_end_via_registry_dispatch(self, sandbox: LocalSandbox):
        registry = ToolRegistry()
        registry.register(bash_tool(sandbox))
        result = await registry.dispatch(_call("bash", {"command": "echo via-dispatch"}))
        assert result.is_error is False
        assert "via-dispatch" in result.content


class TestReadWriteEditFileTools:
    def test_read_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = read_file_tool(sandbox)
        assert tool.spec.name == "read_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    def test_write_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = write_file_tool(sandbox)
        assert tool.spec.name == "write_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {"path", "content"}
        assert tool.meta.side_effect is False

    def test_edit_file_schema_and_meta(self, sandbox: LocalSandbox):
        tool = edit_file_tool(sandbox)
        assert tool.spec.name == "edit_file"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "path",
            "old_string",
            "new_string",
        }
        assert tool.meta.side_effect is False

    async def test_write_then_read_round_trip(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        read = read_file_tool(sandbox)
        write_result = await write.handler({"path": "notes.txt", "content": "hello world"})
        assert "notes.txt" in write_result
        assert await read.handler({"path": "notes.txt"}) == "hello world"

    async def test_read_missing_file_raises_sandbox_error(self, sandbox: LocalSandbox):
        await sandbox.start()
        tool = read_file_tool(sandbox)
        with pytest.raises(SandboxError):
            await tool.handler({"path": "nope.txt"})

    async def test_edit_file_applies_unique_replacement(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        read = read_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "x = 1\ny = 2\n"})
        await edit.handler({"path": "f.py", "old_string": "x = 1", "new_string": "x = 100"})
        assert await read.handler({"path": "f.py"}) == "x = 100\ny = 2\n"

    async def test_edit_file_replace_all(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        read = read_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "dup\ndup\n"})
        await edit.handler(
            {"path": "f.py", "old_string": "dup", "new_string": "one", "replace_all": True}
        )
        assert await read.handler({"path": "f.py"}) == "one\none\n"

    async def test_edit_file_ambiguous_match_raises(self, sandbox: LocalSandbox):
        write = write_file_tool(sandbox)
        edit = edit_file_tool(sandbox)
        await write.handler({"path": "f.py", "content": "dup\ndup\n"})
        with pytest.raises(SandboxError, match="not unique"):
            await edit.handler({"path": "f.py", "old_string": "dup", "new_string": "one"})

    async def test_write_file_missing_content_raises(self, sandbox: LocalSandbox):
        tool = write_file_tool(sandbox)
        with pytest.raises(MissingArgumentError):
            await tool.handler({"path": "f.txt"})

    async def test_dispatch_error_on_path_traversal(self, sandbox: LocalSandbox):
        registry = ToolRegistry()
        registry.register(read_file_tool(sandbox))
        result = await registry.dispatch(
            _call("read_file", {"path": "../outside.txt"})
        )
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


class TestMemoryTools:
    def test_read_fact_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_read_fact_tool(memory_store)
        assert tool.spec.name == "memory_read_fact"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    def test_write_fact_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_write_fact_tool(memory_store)
        assert tool.spec.name == "memory_write_fact"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "name",
            "description",
            "type",
            "body",
        }
        assert tool.meta.side_effect is False

    def test_search_schema_and_meta(self, memory_store: MemoryStore):
        tool = memory_search_tool(memory_store)
        assert tool.spec.name == "memory_search"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.meta.side_effect is False

    async def test_write_then_read_round_trip(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        read = memory_read_fact_tool(memory_store)
        write_result = await write.handler(
            {
                "name": "prefers-dark-mode",
                "description": "User prefers dark mode",
                "type": "user",
                "body": "Observed across sessions.",
                "sources": ["2026-07-19-onboarding.md"],
            }
        )
        assert "prefers-dark-mode" in write_result
        read_result = await read.handler({"name": "prefers-dark-mode"})
        assert "prefers-dark-mode" in read_result
        assert "Observed across sessions." in read_result

    async def test_read_unknown_fact_raises(self, memory_store: MemoryStore):
        tool = memory_read_fact_tool(memory_store)
        with pytest.raises(FactNotFoundError):
            await tool.handler({"name": "no-such-fact"})

    async def test_search_finds_written_fact(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        search = memory_search_tool(memory_store)
        await write.handler(
            {
                "name": "likes-tea",
                "description": "User likes tea",
                "type": "user",
                "body": "Drinks green tea every morning.",
            }
        )
        result = await search.handler({"query": "green tea"})
        assert "likes-tea" in result
        assert "green tea" in result

    async def test_search_no_matches(self, memory_store: MemoryStore):
        search = memory_search_tool(memory_store)
        result = await search.handler({"query": "nonexistent-topic"})
        assert "no matches" in result

    async def test_write_fact_invalid_type_raises(self, memory_store: MemoryStore):
        write = memory_write_fact_tool(memory_store)
        with pytest.raises(Exception):
            await write.handler(
                {
                    "name": "bad-fact",
                    "description": "desc",
                    "type": "not-a-real-type",
                    "body": "body",
                }
            )

    async def test_dispatch_end_to_end(self, memory_store: MemoryStore):
        registry = ToolRegistry()
        registry.register(memory_write_fact_tool(memory_store))
        registry.register(memory_read_fact_tool(memory_store))
        await registry.dispatch(
            _call(
                "memory_write_fact",
                {
                    "name": "test-fact",
                    "description": "desc",
                    "type": "project",
                    "body": "body text",
                },
            )
        )
        result = await registry.dispatch(
            _call("memory_read_fact", {"name": "test-fact"}, call_id="call-2")
        )
        assert result.is_error is False
        assert "body text" in result.content


# ---------------------------------------------------------------------------
# Task ledger tools
# ---------------------------------------------------------------------------


class TestTaskLedgerTools:
    def test_task_update_schema_and_meta(self, run_store: RunStore, run_id: str):
        tool = task_update_tool(run_store, run_id)
        assert tool.spec.name == "task_update"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert set(tool.spec.input_schema["required"]) == {
            "item_id",
            "description",
            "status",
        }
        assert tool.meta.side_effect is False

    def test_task_list_schema_and_meta(self, run_store: RunStore, run_id: str):
        tool = task_list_tool(run_store, run_id)
        assert tool.spec.name == "task_list"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.spec.input_schema["required"] == []
        assert tool.meta.side_effect is False

    async def test_list_empty_ledger(self, run_store: RunStore, run_id: str):
        tool = task_list_tool(run_store, run_id)
        result = await tool.handler({})
        assert "empty" in result

    async def test_update_then_list_round_trip(self, run_store: RunStore, run_id: str):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {
                "item_id": "step-1",
                "description": "Write the tests",
                "status": "in_progress",
            }
        )
        result = await listing.handler({})
        assert "step-1" in result
        assert "Write the tests" in result
        assert "in_progress" in result

    async def test_update_with_evidence_shown_in_listing(
        self, run_store: RunStore, run_id: str
    ):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {
                "item_id": "step-1",
                "description": "Run tests",
                "status": "done",
                "evidence": "pytest: 42 passed",
            }
        )
        result = await listing.handler({})
        assert "pytest: 42 passed" in result

    async def test_repeated_update_overwrites_item(
        self, run_store: RunStore, run_id: str
    ):
        update = task_update_tool(run_store, run_id)
        listing = task_list_tool(run_store, run_id)
        await update.handler(
            {"item_id": "step-1", "description": "first", "status": "pending"}
        )
        await update.handler(
            {"item_id": "step-1", "description": "first", "status": "done"}
        )
        result = await listing.handler({})
        assert result.count("step-1") == 1
        assert "done" in result

    async def test_dispatch_end_to_end(self, run_store: RunStore, run_id: str):
        registry = ToolRegistry()
        registry.register(task_update_tool(run_store, run_id))
        registry.register(task_list_tool(run_store, run_id))
        await registry.dispatch(
            _call(
                "task_update",
                {"item_id": "a", "description": "d", "status": "pending"},
            )
        )
        result = await registry.dispatch(_call("task_list", {}, call_id="call-2"))
        assert result.is_error is False
        assert "a" in result.content


# ---------------------------------------------------------------------------
# Skill tool
# ---------------------------------------------------------------------------


class TestLoadSkillTool:
    def test_schema_and_meta(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        assert tool.spec.name == "load_skill"
        _assert_valid_object_schema(tool.spec.input_schema)
        assert tool.spec.input_schema["required"] == ["name"]
        assert tool.meta.side_effect is False

    async def test_loads_full_body(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        body = await tool.handler({"name": "greet"})
        assert body == "Say hello and ask how you can help."

    async def test_unknown_skill_raises_keyerror(self, skill_library: SkillLibrary):
        tool = load_skill_tool(skill_library)
        with pytest.raises(KeyError):
            await tool.handler({"name": "does-not-exist"})

    async def test_dispatch_unknown_skill_is_error_result(
        self, skill_library: SkillLibrary
    ):
        registry = ToolRegistry()
        registry.register(load_skill_tool(skill_library))
        result = await registry.dispatch(
            _call("load_skill", {"name": "does-not-exist"})
        )
        assert result.is_error is True
        assert "does-not-exist" in result.content


# ---------------------------------------------------------------------------
# All builtins together
# ---------------------------------------------------------------------------


class TestAllBuiltinsTogether:
    def test_register_all_builtins_without_name_collisions(
        self,
        sandbox: LocalSandbox,
        memory_store: MemoryStore,
        run_store: RunStore,
        run_id: str,
        skill_library: SkillLibrary,
    ):
        registry = ToolRegistry()
        tools = [
            bash_tool(sandbox),
            read_file_tool(sandbox),
            write_file_tool(sandbox),
            edit_file_tool(sandbox),
            memory_read_fact_tool(memory_store),
            memory_write_fact_tool(memory_store),
            memory_search_tool(memory_store),
            task_update_tool(run_store, run_id),
            task_list_tool(run_store, run_id),
            load_skill_tool(skill_library),
            add_instruction_tool(run_store, run_id),
            search_history_tool(run_store, run_id),
        ]
        for tool in tools:
            registry.register(tool)
        specs = registry.specs()
        names = [spec.name for spec in specs]
        assert len(names) == len(set(names)) == 12
        for spec in specs:
            _assert_valid_object_schema(spec.input_schema)
        for tool in tools:
            assert tool.meta.side_effect is False


# ---------------------------------------------------------------------------
# Context binding: skill splicing, task snapshot, instruction ledger
# ---------------------------------------------------------------------------


class TestContextBoundTools:
    async def test_load_skill_with_context_splices_body_into_system_prompt(
        self, skill_library: SkillLibrary
    ):
        """Regression (DESIGN.md §4.6): with a bound context the skill body
        rides the system prompt — exempt from tool-result pruning — and the
        tool result is a short acknowledgment, not the body."""
        context = _make_context()
        tool = load_skill_tool(skill_library, context)
        ack = await tool.handler({"name": "greet"})
        assert "greet" in ack
        assert "system prompt" in ack
        assert "Say hello and ask how you can help." not in ack
        system, _ = context.assemble()
        assert "Say hello and ask how you can help." in system
        assert "## Loaded skill: greet" in system

    async def test_load_skill_without_context_returns_body(
        self, skill_library: SkillLibrary
    ):
        tool = load_skill_tool(skill_library)
        assert (
            await tool.handler({"name": "greet"})
            == "Say hello and ask how you can help."
        )

    async def test_task_update_with_context_refreshes_snapshot(
        self, run_store: RunStore, run_id: str
    ):
        """Regression (DESIGN.md §4.9): task updates mirror the live ledger
        into the context's trailing reminder."""
        context = _make_context(reminder_interval=1)
        tool = task_update_tool(run_store, run_id, context)
        await tool.handler(
            {
                "item_id": "step-1",
                "description": "Write the tests",
                "status": "in_progress",
            }
        )
        context.append(Message(role=Role.USER, content="goal"))
        context.append(Message(role=Role.ASSISTANT, content="turn 1"))
        _, messages = context.assemble()
        reminder = messages[-1].content or ""
        assert reminder.startswith("<system-reminder>")
        assert "Current task ledger:" in reminder
        assert "- [in_progress] step-1: Write the tests" in reminder

    async def test_add_instruction_persists_and_joins_context_ledger(
        self, run_store: RunStore, run_id: str
    ):
        """Regression (DESIGN.md §4.5): recorded instructions land in the
        instruction_ledger table and the context's reminder ledger."""
        context = _make_context(reminder_interval=1)
        tool = add_instruction_tool(run_store, run_id, context)
        assert tool.spec.name == "add_instruction"
        assert tool.meta.side_effect is False
        result = await tool.handler({"instruction": "never push to main"})
        assert "never push to main" in result

        items = run_store.list_instructions(run_id)
        assert len(items) == 1
        assert items[0].instruction == "never push to main"
        assert items[0].source == "user"
        assert items[0].item_id == "instr-1"

        context.append(Message(role=Role.USER, content="goal"))
        context.append(Message(role=Role.ASSISTANT, content="turn 1"))
        _, messages = context.assemble()
        assert "never push to main" in (messages[-1].content or "")

    async def test_add_instruction_auto_ids_do_not_collide(
        self, run_store: RunStore, run_id: str
    ):
        tool = add_instruction_tool(run_store, run_id)
        await tool.handler({"instruction": "first"})
        await tool.handler({"instruction": "second", "source": "task"})
        items = run_store.list_instructions(run_id)
        assert [(i.item_id, i.instruction) for i in items] == [
            ("instr-1", "first"),
            ("instr-2", "second"),
        ]
        assert items[1].source == "task"


# ---------------------------------------------------------------------------
# search_history tool (the §4.3 layer-4 retrieval backstop)
# ---------------------------------------------------------------------------


class TestSearchHistoryTool:
    @pytest.fixture
    def agent_id(self, run_store: RunStore, run_id: str) -> str:
        return run_store.create_agent(run_id, "test agent")

    async def test_finds_persisted_tool_output(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        """Regression (DESIGN.md §4.3.4): content evicted from context is
        still reachable through the run's event log."""
        run_store.append_event(
            agent_id,
            "tool_result",
            {
                "tool_call_id": "c1",
                "content": "the secret port is 54321",
                "is_error": False,
            },
        )
        tool = search_history_tool(run_store, run_id)
        assert tool.spec.name == "search_history"
        result = await tool.handler({"query": "secret port"})
        assert "54321" in result
        assert "tool_result" in result
        assert f"agent {agent_id}" in result

    async def test_search_is_case_insensitive_and_reports_no_matches(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        run_store.append_event(
            agent_id, "message", {"role": "user", "content": "Deploy THE WIDGET"}
        )
        tool = search_history_tool(run_store, run_id)
        assert "widget" in (await tool.handler({"query": "the widget"})).lower()
        assert "no matches" in await tool.handler({"query": "zebra"})

    async def test_limit_caps_matches_but_reports_total(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        for i in range(5):
            run_store.append_event(
                agent_id, "message", {"role": "user", "content": f"needle {i}"}
            )
        tool = search_history_tool(run_store, run_id)
        result = await tool.handler({"query": "needle", "limit": 2})
        assert "5 match(es)" in result
        assert "showing first 2" in result
        assert result.count("[agent") == 2

    async def test_registered_dispatch_round_trip(
        self, run_store: RunStore, run_id: str, agent_id: str
    ):
        run_store.append_event(
            agent_id, "message", {"role": "user", "content": "haystack needle"}
        )
        registry = ToolRegistry()
        registry.register(search_history_tool(run_store, run_id))
        result = await registry.dispatch(
            _call("search_history", {"query": "needle"})
        )
        assert result.is_error is False
        assert "needle" in result.content
