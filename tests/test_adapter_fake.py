"""Unit tests for harness.adapters.fake and the get_adapter factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from harness.adapters import AdapterError, get_adapter
from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.fake import FakeAdapter
from harness.adapters.openai_compat import OpenAICompatAdapter
from harness.config import ModelConfig
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)


def scripted(content: str) -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason=StopReason.END_TURN,
    )


class TestFakeAdapter:
    async def test_returns_responses_in_order(self) -> None:
        adapter = FakeAdapter([scripted("first"), scripted("second")])
        first = await adapter.complete([Message(role=Role.USER, content="a")], [])
        second = await adapter.complete([Message(role=Role.USER, content="b")], [])
        assert first.message.content == "first"
        assert second.message.content == "second"

    async def test_records_every_call(self) -> None:
        adapter = FakeAdapter([scripted("ok")])
        messages = [Message(role=Role.USER, content="hi")]
        tools = [ToolSpec(name="bash", description="run", input_schema={})]
        await adapter.complete(messages, tools, system="rules", temperature=0.1)
        (call,) = adapter.calls
        assert call.messages == messages
        assert call.tools == tools
        assert call.system == "rules"
        assert call.params == {"temperature": 0.1}

    async def test_exhausted_script_raises_clear_error(self) -> None:
        adapter = FakeAdapter([scripted("only")])
        await adapter.complete([Message(role=Role.USER, content="a")], [])
        with pytest.raises(AdapterError, match="exhausted") as excinfo:
            await adapter.complete([Message(role=Role.USER, content="b")], [])
        assert "1 scripted response" in str(excinfo.value)
        assert excinfo.value.retryable is False
        # The failing call is still recorded.
        assert len(adapter.calls) == 2

    async def test_empty_script_raises_immediately(self) -> None:
        adapter = FakeAdapter()
        with pytest.raises(AdapterError, match="exhausted"):
            await adapter.complete([Message(role=Role.USER, content="a")], [])

    async def test_responses_are_copies(self) -> None:
        adapter = FakeAdapter([scripted("ok")])
        response = await adapter.complete([Message(role=Role.USER, content="a")], [])
        response.message.content = "mutated"
        assert adapter._responses[0].message.content == "ok"

    def test_capabilities(self) -> None:
        caps = FakeAdapter().capabilities
        assert caps.supports_cache_control is False
        assert caps.max_context == 1_000_000


class TestJsonlScript:
    def write_script(self, tmp_path: Path, lines: list[dict]) -> Path:
        path = tmp_path / "script.jsonl"
        path.write_text(
            "\n".join(json.dumps(line) for line in lines), encoding="utf-8"
        )
        return path

    async def test_loads_jsonl_script(self, tmp_path: Path) -> None:
        path = self.write_script(
            tmp_path,
            [
                {
                    "content": "checking",
                    "tool_calls": [{"name": "bash", "arguments": {"cmd": "ls"}}],
                },
                {"content": "done"},
            ],
        )
        adapter = FakeAdapter(path)

        first = await adapter.complete([Message(role=Role.USER, content="go")], [])
        assert first.message.content == "checking"
        assert first.stop_reason is StopReason.TOOL_USE
        (call,) = first.message.tool_calls
        assert call.name == "bash"
        assert call.arguments == {"cmd": "ls"}
        assert call.id  # generated id

        second = await adapter.complete([Message(role=Role.USER, content="on")], [])
        assert second.message.content == "done"
        assert second.stop_reason is StopReason.END_TURN
        assert second.message.tool_calls == []

    async def test_explicit_tool_call_id_honored(self, tmp_path: Path) -> None:
        path = self.write_script(
            tmp_path,
            [{"tool_calls": [{"id": "my_id", "name": "bash", "arguments": {}}]}],
        )
        adapter = FakeAdapter(path)
        response = await adapter.complete([Message(role=Role.USER, content="x")], [])
        assert response.message.tool_calls[0].id == "my_id"

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        path.write_text('{"content": "a"}\n\n{"content": "b"}\n', encoding="utf-8")
        assert len(FakeAdapter(path)._responses) == 2

    def test_malformed_line_raises_with_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        path.write_text('{"content": "ok"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            FakeAdapter(path)

    def test_non_object_line_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        path.write_text('["a", "b"]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            FakeAdapter(path)

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = self.write_script(tmp_path, [{"content": "hi"}])
        assert len(FakeAdapter(str(path))._responses) == 1

    def test_non_dict_tool_call_entry_raises_with_line_number(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "script.jsonl"
        path.write_text('{"tool_calls": ["bash"]}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            FakeAdapter(path)

    def test_tool_call_entry_missing_name_raises_with_line_number(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "script.jsonl"
        path.write_text(
            '{"content": "ok"}\n{"tool_calls": [{"arguments": {}}]}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="line 2"):
            FakeAdapter(path)


class TestGetAdapterFactory:
    def test_dispatch_fake(self) -> None:
        adapter = get_adapter(ModelConfig(adapter="fake", model="no-such-file"))
        assert isinstance(adapter, FakeAdapter)
        assert adapter._responses == []

    def test_dispatch_fake_with_script_path(self, tmp_path: Path) -> None:
        script = tmp_path / "script.jsonl"
        script.write_text('{"content": "hi"}\n', encoding="utf-8")
        adapter = get_adapter(ModelConfig(adapter="fake", model=str(script)))
        assert isinstance(adapter, FakeAdapter)
        assert len(adapter._responses) == 1

    def test_dispatch_fake_missing_jsonl_path_fails_fast(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "no-such-script.jsonl"
        with pytest.raises(FileNotFoundError, match=str(missing)):
            get_adapter(ModelConfig(adapter="fake", model=str(missing)))

    def test_dispatch_fake_missing_path_with_separator_fails_fast(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_adapter(
                ModelConfig(adapter="fake", model="scripts/does-not-exist")
            )

    def test_dispatch_fake_non_path_string_stays_empty_script(self) -> None:
        # No ".jsonl" suffix or path separator: treated as an intentionally
        # empty fake rather than a typo'd script path.
        adapter = get_adapter(ModelConfig(adapter="fake", model="no-such-file"))
        assert isinstance(adapter, FakeAdapter)
        assert adapter._responses == []

    def test_dispatch_anthropic_resolves_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "dummy-key")
        config = ModelConfig(
            adapter="anthropic",
            model="claude-opus-4-8",
            api_key=SecretStr("env:FAKE_ANTHROPIC_KEY"),
        )
        adapter = get_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._model == "claude-opus-4-8"

    def test_dispatch_anthropic_forwards_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "dummy-key")
        config = ModelConfig(
            adapter="anthropic",
            model="claude-opus-4-8",
            base_url="https://my-gateway.example/",
            api_key=SecretStr("env:FAKE_ANTHROPIC_KEY"),
        )
        adapter = get_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert str(adapter._client.base_url).startswith(
            "https://my-gateway.example/"
        )

    def test_dispatch_openai_with_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MOONSHOT_KEY", "dummy-key")
        config = ModelConfig(
            adapter="openai",
            model="kimi-k3",
            base_url="https://api.moonshot.ai/v1",
            api_key=SecretStr("env:FAKE_MOONSHOT_KEY"),
        )
        adapter = get_adapter(config)
        assert isinstance(adapter, OpenAICompatAdapter)
        assert str(adapter._client.base_url).startswith(
            "https://api.moonshot.ai/v1"
        )

    def test_unknown_adapter_lists_valid_names(self) -> None:
        config = ModelConfig(adapter="gemini", model="gemini-3")
        with pytest.raises(ValueError) as excinfo:
            get_adapter(config)
        message = str(excinfo.value)
        assert "gemini" in message
        for name in ("anthropic", "openai", "fake"):
            assert name in message
