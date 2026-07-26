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
    PRUNE_KEEP_TURNS,
    PRUNE_PRESSURE_THRESHOLD,
    PRUNE_TARGET_FRACTION,
    ContextManager,
)
from harness.types import Message, Role, ToolCall, ToolResult

BASE_PROMPT = "You are a diligent harness agent."

#: A ``max_context`` small enough that the legacy pruning fixtures below (a
#: dozen-odd 100-token messages) sit above ``PRUNE_PRESSURE_THRESHOLD``.
#: Pruning is pressure-gated now, so every test that asserts a stub must
#: create the pressure that licenses it.
UNDER_PRESSURE_CONTEXT = 1_000


def fake_count_tokens(messages: list[Message]) -> int:
    """Deterministic fake counter: 100 tokens per message."""
    return 100 * len(messages)


def char_count_tokens(messages: list[Message]) -> int:
    """Chars-per-4 counter, matching the adapters' default (base.py:167).

    Used by the pressure tests, where the shed's own ``chars // 4`` budgeting
    proxy has to be commensurate with the counter for the arithmetic to mean
    anything.
    """
    chars = 0
    for message in messages:
        if message.content:
            chars += len(message.content)
        if message.tool_result is not None:
            chars += len(message.tool_result.content)
    return chars // 4


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
    """Under pressure, PRUNE_KEEP_TURNS is the floor of the shed."""
    cm = make_cm(max_context=UNDER_PRESSURE_CONTEXT)
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
    """Under pressure, the keep window slides forward with the turns."""
    cm = make_cm(max_context=UNDER_PRESSURE_CONTEXT)
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
    cm = make_cm(max_context=UNDER_PRESSURE_CONTEXT)
    cm.append(user("goal"))
    for i in range(1, 6):
        cm.append(assistant_turn(i))
        cm.append(tool_result(i))
    cm.assemble()
    stored = [m for m in cm.transcript if m.role is Role.TOOL]
    assert stored[0].tool_result is not None
    assert stored[0].tool_result.content == "output-1"


def test_pruning_stub_uses_generic_name_for_unknown_call() -> None:
    cm = make_cm(max_context=UNDER_PRESSURE_CONTEXT)
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


# -- pressure-gated pruning (G1) ----------------------------------------------


def pressure_cm(
    *,
    max_context: int,
    turns: int,
    result_chars,
    reminder_interval: int = 1_000_000,
) -> ContextManager:
    """Goal + ``turns`` × (assistant tool call, tool result) on a chars/4
    counter. ``result_chars`` is an int or a callable of the 1-based turn."""
    cm = make_cm(
        max_context=max_context,
        count_tokens=char_count_tokens,
        reminder_interval=reminder_interval,
    )
    cm.append(user("goal"))
    for i in range(1, turns + 1):
        cm.append(assistant_turn(i))
        size = result_chars(i) if callable(result_chars) else result_chars
        cm.append(tool_result(i, "x" * size))
    return cm


def stub_count(messages: list[Message]) -> int:
    return sum(
        1
        for m in messages
        if m.role is Role.TOOL
        and m.tool_result is not None
        and m.tool_result.content.startswith("[pruned:")
    )


def test_no_pruning_below_the_pressure_threshold() -> None:
    """THE REGRESSION FENCE. Before G1 this failed.

    A 25-turn transcript at ~9.5% window utilization must reach the model
    intact — including tool results twenty turns old. The pre-G1 assembler
    stubbed every tool result older than PRUNE_KEEP_TURNS unconditionally,
    with no reference to max_context, while compaction (the pressure-driven
    mechanism) never fired once across all fourteen round-2 benchmark
    trials. The harness was discarding evidence at under 13% utilization.
    """
    cm = pressure_cm(max_context=100_000, turns=25, result_chars=1500)
    raw = cm._raw_token_count()
    assert raw < 0.5 * PRUNE_PRESSURE_THRESHOLD * 100_000  # ~10% utilization

    _, messages = cm.assemble()
    assert cm._prune_plan() == frozenset()
    assert stub_count(messages) == 0
    for message in messages:
        text = message.content or (
            message.tool_result.content if message.tool_result else ""
        )
        assert "[pruned:" not in text
    # The twenty-turn-old result is still there, verbatim.
    oldest = [m for m in messages if m.role is Role.TOOL][0]
    assert oldest.tool_result is not None
    assert oldest.tool_result.content == "x" * 1500


def test_pressure_engages_shedding_oldest_first_down_to_target() -> None:
    """Above the threshold the shed engages: oldest first, never inside the
    keep window, stopping once the estimate reaches PRUNE_TARGET_FRACTION."""
    cm = pressure_cm(max_context=100_000, turns=14, result_chars=16_000)
    raw = cm._raw_token_count()
    assert raw > PRUNE_PRESSURE_THRESHOLD * 100_000

    plan = cm._prune_plan()
    assert plan  # stubs appear
    candidates = cm._tool_results_oldest_first()
    eligible = [i for i, _, age in candidates if age > PRUNE_KEEP_TURNS]
    # Exactly a prefix of the oldest eligible results, nothing newer.
    assert sorted(plan) == eligible[: len(plan)]
    assert len(plan) < len(eligible)  # graduated, not "everything old"
    # Nothing inside the keep window was touched.
    recent = {i for i, _, age in candidates if age <= PRUNE_KEEP_TURNS}
    assert not (plan & recent)
    # The shed reached target (it did not bottom out on the keep window).
    assert cm._token_count() <= PRUNE_TARGET_FRACTION * 100_000

    _, messages = cm.assemble()
    assert stub_count(messages) == len(plan)


def test_shedding_is_graduated_not_a_cliff() -> None:
    """More pressure sheds strictly more; 0.55 utilization is not 0.75."""
    low = pressure_cm(max_context=100_000, turns=14, result_chars=16_000)
    high = pressure_cm(max_context=100_000, turns=19, result_chars=16_000)
    assert 0.5 < low._raw_token_count() / 100_000 < 0.6
    assert 0.7 < high._raw_token_count() / 100_000 < 0.8
    assert len(low._prune_plan()) < len(high._prune_plan())


def test_pressure_signal_is_measured_before_pruning_not_after() -> None:
    """The second defect: pre-G1, ``_token_count`` (what compaction reads)
    was computed on an assembly that had *already* pruned, so the cheap
    destructive mechanism suppressed the trigger for the careful one.

    Now there are two distinct counts: ``_raw_token_count`` (no pruning —
    the pruning decision's input) and ``_token_count`` (post-pruning — what
    goes on the wire, and what compaction reads). Neither recurses.
    """
    seen: list[list[Message]] = []

    def recording(messages: list[Message]) -> int:
        seen.append(list(messages))
        return char_count_tokens(messages)

    cm = pressure_cm(max_context=100_000, turns=14, result_chars=16_000)
    cm._count_tokens = recording  # type: ignore[assignment]
    cm._invalidate_counts()

    assert cm._token_count() > 0
    # Exactly two assemblies were counted — no recursion, no re-entry.
    assert len(seen) == 2
    # The first (the pruning decision's signal) saw no stubs at all ...
    assert stub_count(seen[0]) == 0
    # ... the second (compaction's signal) saw the live plan.
    assert stub_count(seen[1]) == len(cm._prune_plan())
    assert cm._raw_token_count() > cm._token_count()
    # Both are memoized for the turn: no further counter calls.
    cm._token_count()
    cm._raw_token_count()
    cm._prune_plan()
    assert len(seen) == 2


def test_hysteresis_consecutive_assemblies_are_byte_identical() -> None:
    """A transcript that has not changed must assemble identically, so the
    provider's prompt-cache prefix survives. Without the 0.50/0.40 gap a
    transcript hovering at the trigger would flip stub state turn-to-turn.

    Growth only *extends* the plan — already-stubbed indices stay stubbed —
    so the shared prefix is preserved as the run goes on.
    """
    cm = pressure_cm(max_context=100_000, turns=13, result_chars=16_000)
    assert 0.5 < cm._raw_token_count() / 100_000 < 0.55

    first_plan = cm._prune_plan()
    first = cm.assemble()
    second_plan = cm._prune_plan()
    second = cm.assemble()
    assert first_plan == second_plan
    assert first == second

    cm.append(assistant_turn(14))
    cm.append(tool_result(14, "x" * 16_000))
    grown = cm._prune_plan()
    assert first_plan <= grown  # prefix-stable: only ever extends


async def test_ladder_hands_off_to_compaction_when_the_shed_cannot_reach_target(
) -> None:
    """Rung three. The keep window is inviolable, so a transcript whose
    *recent* results alone exceed the window cannot be shed to target; the
    post-prune count stays above COMPACTION_THRESHOLD and compaction takes
    over. The compact-to-fixpoint pass must still terminate."""
    cm = pressure_cm(
        max_context=100_000,
        turns=5,
        result_chars=lambda i: 40_000 if i <= 2 else 120_000,
    )
    assert cm._prune_plan()  # pruning engaged ...
    assert cm._token_count() > COMPACTION_THRESHOLD * 100_000  # ... and lost

    evicted = []
    for _ in range(20):  # fixpoint, with a fuse
        span = await cm.maybe_compact()
        if span is None:
            break
        evicted.append(span)
    else:  # pragma: no cover - the fuse blowing is the failure
        raise AssertionError("compact-to-fixpoint did not terminate")
    assert evicted


def test_single_oversized_result_is_shed_in_one_pass() -> None:
    """One result larger than the entire budget: stubbed on the first
    iteration, and the pass stops there."""
    cm = pressure_cm(
        max_context=100_000,
        turns=5,
        result_chars=lambda i: 500_000 if i == 1 else 100,
    )
    plan = cm._prune_plan()
    assert len(plan) == 1
    _, messages = cm.assemble()
    results = [m for m in messages if m.role is Role.TOOL]
    assert results[0].tool_result is not None
    assert results[0].tool_result.content.startswith("[pruned: bash result")
    for result in results[1:]:
        assert result.tool_result is not None
        assert result.tool_result.content == "x" * 100


def test_ranged_read_survives_twenty_unrelated_turns_at_a_real_window() -> None:
    """The make-doom-for-mips fence.

    In that round-2 trial 66 of 90 ranged reads (73%) re-read a byte range
    the agent had already read, and one ``sed -n`` range over /app/vm.js
    issued at seq 49 was re-read in overlapping fragments twelve times
    (seqs 157, 182, 212, 262, 307, 317, 347, 429, 497, 509, 515, 545);
    93 of the trial's 106 bash calls (88%) were pure inspection. The
    transcript had been erased under it. At the provider's real window
    (openai_compat max_context = 128,000) reading a range once must be
    enough.
    """
    range_a = "\n".join(f"{n}: line of vm.js" for n in range(1740, 1831))
    cm = make_cm(max_context=128_000, count_tokens=char_count_tokens)
    cm.append(user("port doom to mips"))
    cm.append(assistant_turn(0))
    cm.append(tool_result(0, range_a))
    for i in range(1, 21):
        cm.append(assistant_turn(i))
        cm.append(tool_result(i, f"unrelated output {i}"))

    assert cm._prune_plan() == frozenset()
    _, messages = cm.assemble()
    contents = [
        m.tool_result.content for m in messages if m.tool_result is not None
    ]
    assert range_a in contents


def test_solved_trial_sized_transcripts_never_prune() -> None:
    """Safety assertion for the three round-2 solves (compile-compcert,
    mcmc-sampling-stan, qemu-startup): all peaked under 6,000 input tokens
    against a 128,000-token window, so G1 must be a no-op for them — an
    empty plan on *every* turn, not merely at the end."""
    cm = make_cm(max_context=128_000, count_tokens=char_count_tokens)
    cm.append(user("solve the task"))
    for i in range(1, 61):
        cm.append(assistant_turn(i))
        cm.append(tool_result(i, "x" * 380))
        assert cm._prune_plan() == frozenset(), f"pruned on turn {i}"
        _, messages = cm.assemble()
        assert stub_count(messages) == 0
    assert cm._raw_token_count() < 6_000


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
    cm = make_cm(max_context=UNDER_PRESSURE_CONTEXT)
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
