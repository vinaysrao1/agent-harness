"""Tests for harness.context (DESIGN.md §4.3 + §4.5).

No network, no API keys: the token counter is a fake message-counting
callable and the summarizer is an async stub, exactly as the ContextManager
contract intends (the real loop injects an adapter counter and a
cheap-model call).
"""

from __future__ import annotations

from harness.context import (
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_THRESHOLD,
    MEMORY_BLOCK_BEGIN,
    MEMORY_BLOCK_END,
    ContextManager,
)
from harness.types import Message, Role, ToolCall, ToolResult

BASE_PROMPT = "You are a diligent harness agent."


def fake_count_tokens(messages: list[Message]) -> int:
    """Deterministic fake counter: 100 tokens per message."""
    return 100 * len(messages)


async def stub_summarize(messages: list[Message]) -> str:
    """Async summarizer stub."""
    return f"STUB SUMMARY of {len(messages)} messages"


def make_cm(
    *,
    max_context: int = 1_000_000,
    reminder_interval: int = 5,
    count_tokens=fake_count_tokens,
    summarize=stub_summarize,
) -> ContextManager:
    return ContextManager(
        base_system_prompt=BASE_PROMPT,
        count_tokens=count_tokens,
        max_context=max_context,
        summarize=summarize,
        reminder_interval=reminder_interval,
    )


def user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant_turn(i: int, tool: str = "bash") -> Message:
    """Assistant message carrying one tool call with id ``c<i>``."""
    return Message(
        role=Role.ASSISTANT,
        content=f"assistant turn {i}",
        tool_calls=[ToolCall(id=f"c{i}", name=tool, arguments={"cmd": "ls"})],
    )


def tool_result(i: int, content: str | None = None) -> Message:
    return Message(
        role=Role.TOOL,
        tool_result=ToolResult(
            tool_call_id=f"c{i}", content=content or f"output-{i}"
        ),
    )


def is_reminder(message: Message) -> bool:
    return (
        message.role is Role.USER
        and message.content is not None
        and message.content.startswith("<system-reminder>")
    )


# -- system prompt assembly ---------------------------------------------------


def test_system_contains_base_skills_memory_and_ledger_in_order() -> None:
    cm = make_cm()
    cm.add_skill_body("Always run pytest before declaring done.", name="tdd")
    cm.add_memory_block("User prefers French replies.")
    cm.add_instruction("never push to main", source="user")
    system, messages = cm.assemble()

    assert messages == []
    i_base = system.index(BASE_PROMPT)
    i_skill = system.index("Always run pytest")
    i_mem = system.index("User prefers French replies.")
    i_ledger = system.index("- [user] never push to main")
    assert i_base < i_skill < i_mem < i_ledger
    assert "## Loaded skill: tdd" in system


def test_memory_block_wrapped_in_data_delimiters() -> None:
    cm = make_cm()
    cm.add_memory_block("fact: the deploy script lives in ops/")
    system, _ = cm.assemble()

    begin = system.index(MEMORY_BLOCK_BEGIN)
    body = system.index("fact: the deploy script lives in ops/")
    end = system.index(MEMORY_BLOCK_END)
    assert begin < body < end
    # The delimiters must label the block as data, not instructions.
    assert "BEGIN RECALLED MEMORY" in MEMORY_BLOCK_BEGIN
    assert "END RECALLED MEMORY" in MEMORY_BLOCK_END
    assert "data" in MEMORY_BLOCK_BEGIN
    assert "instructions" in MEMORY_BLOCK_BEGIN


def test_instruction_ledger_renders_text_and_source() -> None:
    cm = make_cm()
    cm.add_instruction("never push to main", source="user")
    cm.add_instruction("reply in French", source="task")
    rendered = cm.render_instructions()
    assert rendered == "- [user] never push to main\n- [task] reply in French"


def test_empty_ledger_renders_empty_and_is_omitted_from_system() -> None:
    cm = make_cm()
    assert cm.render_instructions() == ""
    system, _ = cm.assemble()
    assert "Instruction ledger" not in system


# -- tool-output pruning ------------------------------------------------------


def test_recent_transcript_passes_through_verbatim() -> None:
    cm = make_cm()
    cm.append(user("goal"))
    cm.append(assistant_turn(1))
    cm.append(tool_result(1))
    _, messages = cm.assemble()
    assert messages == cm.transcript


def test_pruning_boundary_exactly_three_turns_kept() -> None:
    cm = make_cm()
    cm.append(user("goal"))  # ref 1
    for i in range(1, 6):  # refs 2..11
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    _, messages = cm.assemble()

    results = [m for m in messages if m.role is Role.TOOL]
    # result 1 has 4 assistant messages after it -> older than 3 turns -> pruned
    assert results[0].tool_result is not None
    assert results[0].tool_result.content == (
        f"[pruned: bash result, {len('output-1')} chars; event ref 3]"
    )
    assert results[0].tool_result.tool_call_id == "c1"
    # results 2..5 have ages 3, 2, 1, 0 -> kept verbatim
    for i, result in zip(range(2, 6), results[1:]):
        assert result.tool_result is not None
        assert result.tool_result.content == f"output-{i}"


def test_pruning_boundary_shifts_with_each_new_assistant_turn() -> None:
    cm = make_cm()
    cm.append(user("goal"))
    for i in range(1, 7):  # one more turn than the boundary test
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    _, messages = cm.assemble()
    results = [m for m in messages if m.role is Role.TOOL]
    assert "[pruned:" in (results[0].tool_result.content or "")
    assert "[pruned:" in (results[1].tool_result.content or "")
    assert results[2].tool_result.content == "output-3"


def test_pruning_does_not_mutate_transcript() -> None:
    cm = make_cm()
    cm.append(user("goal"))
    for i in range(1, 6):
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    cm.assemble()
    stored = [m for m in cm.transcript if m.role is Role.TOOL]
    assert stored[0].tool_result is not None
    assert stored[0].tool_result.content == "output-1"


def test_pruning_stub_uses_generic_name_for_unknown_call() -> None:
    cm = make_cm()
    cm.append(user("goal"))
    orphan = Message(
        role=Role.TOOL,
        tool_result=ToolResult(tool_call_id="nope", content="x" * 42),
    )
    cm.append(orphan)  # ref 2
    for i in range(1, 5):
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    _, messages = cm.assemble()
    stub = messages[1].tool_result
    assert stub is not None
    assert stub.content == "[pruned: tool result, 42 chars; event ref 2]"


# -- reminder cadence ---------------------------------------------------------


def test_reminder_every_n_assistant_turns_exactly() -> None:
    cm = make_cm(reminder_interval=2)
    cm.add_instruction("never push to main", source="user")
    cm.set_task_snapshot("1. [open] fix the bug")
    cm.append(user("goal"))

    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    _, messages = cm.assemble()
    assert not is_reminder(messages[-1])  # 1 turn: not due

    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))
    _, messages = cm.assemble()
    assert is_reminder(messages[-1])  # 2 turns: due
    assert len(messages) == len(cm.transcript) + 1
    assert "- [user] never push to main" in messages[-1].content
    assert "1. [open] fix the bug" in messages[-1].content
    assert messages[-1].content.endswith("</system-reminder>")

    cm.append(Message(role=Role.ASSISTANT, content="turn 3"))
    _, messages = cm.assemble()
    assert not is_reminder(messages[-1])  # 3 turns: not due

    cm.append(Message(role=Role.ASSISTANT, content="turn 4"))
    _, messages = cm.assemble()
    assert is_reminder(messages[-1])  # 4 turns: due again


def test_cadence_reminder_is_idempotent_across_assembles() -> None:
    cm = make_cm(reminder_interval=1)
    cm.append(user("goal"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    _, first = cm.assemble()
    _, second = cm.assemble()
    assert is_reminder(first[-1])
    assert is_reminder(second[-1])


def test_no_reminder_before_any_assistant_turn() -> None:
    cm = make_cm(reminder_interval=1)
    cm.append(user("goal"))
    _, messages = cm.assemble()
    assert messages == cm.transcript


async def test_reminder_fires_on_first_assemble_after_compaction() -> None:
    cm = make_cm(reminder_interval=50)  # never due by cadence
    cm.add_instruction("always reply in French", source="user")
    cm.append(user("goal"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    cm.append(user("progress note"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))

    await cm.compact()
    _, messages = cm.assemble()
    assert is_reminder(messages[-1])
    assert "always reply in French" in messages[-1].content

    # The post-compaction reminder fires exactly once.
    _, messages = cm.assemble()
    assert not is_reminder(messages[-1])


async def test_maybe_compact_probe_does_not_consume_reminder_flag() -> None:
    cm = make_cm(reminder_interval=50, max_context=1_000_000)
    cm.append(user("goal"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    await cm.compact()
    # A below-threshold maybe_compact (as the loop calls each turn) must not
    # eat the post-compaction reminder before the real assemble sees it.
    assert await cm.maybe_compact() is None
    _, messages = cm.assemble()
    assert is_reminder(messages[-1])


# -- compaction ---------------------------------------------------------------


async def test_maybe_compact_threshold_is_strictly_greater() -> None:
    # 4 transcript messages + 1 system message = 500 fake tokens.
    # 0.8 * 625 == 500 exactly -> NOT over threshold -> no compaction.
    cm = make_cm(max_context=625, reminder_interval=50)
    cm.append(user("goal"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    cm.append(user("note"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))
    assert await cm.maybe_compact() is None
    assert len(cm.transcript) == 4

    # One more message: 600 > 500 -> compaction triggers.
    cm.append(user("another note"))
    evicted = await cm.maybe_compact()
    assert evicted is not None
    assert len(evicted) == 2  # oldest half of 5 messages
    # 5 messages - 2 evicted + 1 summary = 4.
    assert len(cm.transcript) == 4
    assert COMPACTION_THRESHOLD == 0.8


async def test_compact_replaces_oldest_half_with_summary_message() -> None:
    cm = make_cm()
    contents = ["goal", "turn 1", "note", "turn 2", "turn 3"]
    cm.append(user("goal"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 1"))
    cm.append(user("note"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 3"))

    evicted = await cm.compact()
    assert [m.content for m in evicted] == contents[:2]
    assert len(cm.transcript) == 4  # summary + 3 survivors
    summary = cm.transcript[0]
    assert summary.role is Role.USER
    assert summary.content is not None
    assert summary.content.startswith(COMPACTION_SUMMARY_PREFIX + "\n")
    assert "STUB SUMMARY of 2 messages" in summary.content
    assert [m.content for m in cm.transcript[1:]] == contents[2:]


async def test_goal_text_preserved_verbatim_in_summary_header() -> None:
    goal = 'Refactor auth; NEVER touch `main` — deadline "Friday" (v2.1)'
    cm = make_cm()
    cm.append(user(goal))
    for i in range(1, 4):
        cm.append(Message(role=Role.ASSISTANT, content=f"turn {i}"))
    await cm.compact()
    assert goal in (cm.transcript[0].content or "")


async def test_goal_survives_repeated_compactions_verbatim() -> None:
    goal = "the one true goal: ship it"
    cm = make_cm()
    cm.append(user(goal))
    for i in range(1, 6):
        cm.append(Message(role=Role.ASSISTANT, content=f"turn {i}"))
    await cm.compact()
    for i in range(6, 10):
        cm.append(Message(role=Role.ASSISTANT, content=f"turn {i}"))
    await cm.compact()  # evicts the first summary message itself
    assert goal in (cm.transcript[0].content or "")


async def test_evicted_span_returned_intact_and_passed_to_summarizer() -> None:
    seen: list[list[Message]] = []

    async def recording_summarize(messages: list[Message]) -> str:
        seen.append(list(messages))
        return "recorded"

    cm = make_cm(summarize=recording_summarize)
    cm.append(user("goal"))
    cm.append(assistant_turn(1))
    cm.append(tool_result(1, content="precious full output"))
    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))

    # The naive halfway boundary would land on the tool result (index 2),
    # splitting it from its assistant tool call; the boundary snaps past it
    # so all three are evicted together.
    originals = list(cm.transcript[:3])
    evicted = await cm.compact()

    assert evicted == originals  # intact, field-for-field
    assert evicted[1].tool_calls[0].name == "bash"
    assert evicted[2].tool_result is not None  # evicted with its call
    assert seen == [evicted]  # summarizer saw exactly the evicted span


async def test_compaction_never_splits_tool_call_from_its_results() -> None:
    """Regression: the eviction boundary snaps forward past TOOL messages,
    so the kept transcript never starts with orphaned tool results (which
    provider APIs reject with a non-retryable 400)."""
    cm = make_cm()
    cm.append(user("goal"))
    # Assistant turn with two tool calls and two results: the halfway
    # boundary (index 2, first result) would orphan both results.
    cm.append(
        Message(
            role=Role.ASSISTANT,
            content="two calls",
            tool_calls=[
                ToolCall(id="c1", name="bash", arguments={"cmd": "ls"}),
                ToolCall(id="c1b", name="bash", arguments={"cmd": "pwd"}),
            ],
        )
    )
    cm.append(
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="c1", content="out-1"),
        )
    )
    cm.append(
        Message(
            role=Role.TOOL,
            tool_result=ToolResult(tool_call_id="c1b", content="out-1b"),
        )
    )
    cm.append(Message(role=Role.ASSISTANT, content="turn 2"))

    evicted = await cm.compact()

    # Both results travel with their assistant message.
    assert [m.role for m in evicted] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.TOOL,
    ]
    # The surviving transcript must not begin with tool results: every
    # tool_result in it is preceded by an assistant message carrying the
    # matching tool call.
    known_call_ids: set[str] = set()
    for message in cm.transcript:
        if message.role is Role.ASSISTANT:
            known_call_ids.update(call.id for call in message.tool_calls)
        if message.role is Role.TOOL and message.tool_result is not None:
            assert message.tool_result.tool_call_id in known_call_ids


async def test_compact_on_tiny_transcript_is_a_no_op() -> None:
    called = False

    async def failing_summarize(messages: list[Message]) -> str:
        nonlocal called
        called = True
        return "should not happen"

    cm = make_cm(summarize=failing_summarize)
    cm.append(user("goal"))
    assert await cm.compact() == []
    assert [m.content for m in cm.transcript] == ["goal"]
    assert not called


async def test_event_refs_stable_across_compaction_for_pruning_stubs() -> None:
    cm = make_cm()
    cm.append(user("goal"))  # ref 1
    for i in range(1, 10):  # refs 2..19
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    # 19 messages -> the halfway boundary (9) lands on turn 5's assistant
    # message, so compaction evicts refs 1..9: the goal and turns 1-4 in
    # full (each assistant message together with its tool result).
    await cm.compact()
    _, messages = cm.assemble()
    stubs = [
        m.tool_result.content
        for m in messages
        if m.role is Role.TOOL and "[pruned:" in (m.tool_result.content or "")
    ]
    # result 5 (ref 11) survives compaction with 4 assistant turns after it,
    # so it prunes — and its stub must cite the original ref, un-renumbered.
    assert any("event ref 11" in stub for stub in stubs)


def test_append_returns_monotonic_event_refs() -> None:
    cm = make_cm()
    assert cm.append(user("goal")) == 1
    assert cm.append(Message(role=Role.ASSISTANT, content="turn 1")) == 2
    assert cm.append(user("note")) == 3
