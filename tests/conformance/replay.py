"""Replay the frozen corpus through ContextManager without a model (S-003).

``replay_corpus`` reconstructs each trial's messages from its stored shapes
and drives them through a real :class:`~harness.context.ContextManager`,
recording *when* pruning and compaction fire and how large each turn's
assembly is. That trace is what N7 and N8 freeze.

**Why the window is scaled.** Replaying at the production 128K window pins
nothing: across 645 real trials the largest transcript is 175K characters
against a 256K prune threshold, and compaction has fired **zero** times ever.
A corpus replayed at 128K would record "nothing happened" and stay green
through any change to the machinery -- the precise failure S-002's negative
tests exist to prevent, reappearing one layer up.

So each trial is replayed at several window sizes. The production window is
kept because "nothing fires on the real workload" is itself a fact worth
pinning; the smaller windows exist so the machinery is actually exercised.
Scaling the window rather than inventing transcripts keeps the shapes real:
these are the message-size distributions the harness genuinely produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.context import ContextManager
from harness.types import Message, Role, ToolCall, ToolResult

CORPUS_PATH = Path(__file__).resolve().parent.parent / "golden" / "replay_corpus.jsonl"

#: Windows each trial is replayed at.
#:
#: 128_000 is what a real run uses (the OpenAI-compatible adapter default).
#: At that window the replay reproduces production exactly: pruning fires on
#: 13 of 25 trials and compaction on none -- matching the direct observation
#: that across 645 real trials there is not a single ``compaction`` event.
#: That agreement is the corpus's fidelity check, asserted in N7.
#:
#: The smaller windows exist because compaction must be exercised by
#: *something*: pinning a mechanism that never runs pins nothing. Scaling the
#: window rather than inventing transcripts keeps the message-size
#: distribution real. Changing this tuple changes the golden.
REPLAY_WINDOWS: tuple[int, ...] = (128_000, 64_000, 32_000, 16_000)

#: Filler byte. Content is reconstructed as a repeated character of the exact
#: recorded length: the estimator counts characters, so this reproduces token
#: counts exactly while carrying no task content.
_FILLER = "x"


def _text(n: int) -> str:
    return _FILLER * n


#: ``len(repr({"command": ""}))`` -- the fixed overhead of the reconstructed
#: argument dict. Derived, not guessed: an earlier hard-coded 16 was wrong by
#: one against ``repr`` (it was the ``json.dumps`` width) and ignored the tool
#: name entirely.
_ARG_OVERHEAD = len(repr({"command": ""}))

#: The tool name the estimator also counts. Held constant so the arithmetic
#: below is exact.
_CALL_NAME = "bash"


def _rebuild_call(call_id: str, counted: int) -> ToolCall:
    """Rebuild a tool call whose estimator cost is exactly ``counted``.

    The estimator charges ``len(name) + len(repr(arguments))``. Solving for the
    filler length makes the reconstruction exact rather than approximately
    right, which matters because "filler reproduces the counts exactly" is the
    entire justification for storing shapes instead of content.
    """
    filler = max(0, counted - len(_CALL_NAME) - _ARG_OVERHEAD)
    return ToolCall(id=call_id, name=_CALL_NAME, arguments={"command": _text(filler)})


def _next_pending(built: list[Message]) -> str:
    """Id of the earliest tool call in ``built`` that has no result yet."""
    answered = {
        m.tool_result.tool_call_id for m in built if m.tool_result is not None
    }
    for message in built:
        for call in message.tool_calls:
            if call.id not in answered:
                return call.id
    return "call-orphan"


def messages_from_shapes(shapes: list[dict]) -> list[Message]:
    """Rebuild a transcript from stored shapes, preserving every length."""
    out: list[Message] = []
    for index, shape in enumerate(shapes):
        role = Role(shape["role"])
        if role is Role.TOOL:
            # Answer the most recent unanswered call, so the pruning stub can
            # resolve a real tool name. With unpaired ids every stub rendered
            # the literal "tool", which both mis-sized the stub and left the
            # name-resolution branch of _assemble unexercised.
            out.append(
                Message(
                    role=role,
                    tool_result=ToolResult(
                        tool_call_id=_next_pending(out),
                        content=_text(shape.get("result_chars") or 0),
                        is_error=bool(shape.get("is_error")),
                    ),
                )
            )
            continue
        calls = [
            _rebuild_call(f"call-{index}-{n}", size)
            for n, size in enumerate(shape.get("calls") or [])
        ]
        content = _text(shape["chars"]) if shape["chars"] else None
        if content is None and not calls and role is Role.ASSISTANT:
            content = ""
        out.append(Message(role=role, content=content, tool_calls=calls))
    return out


#: Length of the stand-in compaction summary.
#:
#: A 7-character summary makes compaction nearly free, and the post-compaction
#: size is precisely what decides whether compaction re-fires -- so a toy value
#: pins a caricature. 2000 is the order of a real summary; the exact figure is
#: arbitrary but must be realistic and, being frozen, must not drift.
SUMMARY_CHARS = 2000


async def _fixed_summary(messages: list[Message]) -> str:
    """Deterministic stand-in for the model summarizer.

    Fixed rather than model-generated: N7 pins compaction *timing*, and a real
    summarizer would make the trace non-deterministic and the golden
    meaningless.
    """
    return _text(SUMMARY_CHARS)


_SYSTEM_CACHE: str | None = None


def production_system_prompt() -> str:
    """The real assembled ``CODING`` system prompt.

    The replay previously used the 6-character literal ``"SYSTEM"``. The real
    string is ~2,400 characters -- about 590 tokens charged on *every* turn --
    so N8 could not see prompt growth at all, which m10 §2.2 names as N8's job:
    "guards silent prompt growth". Adding 700 characters to CODING_RULES moved
    N1's digest and left N8 bit-identical.

    Cached because it is constant and building it touches the filesystem.
    """
    global _SYSTEM_CACHE
    if _SYSTEM_CACHE is None:
        import tempfile

        from tests.conformance.fixture import coding_assembled_system, fixture_skills

        with tempfile.TemporaryDirectory() as tmp:
            _SYSTEM_CACHE = coding_assembled_system(fixture_skills(Path(tmp)))
    return _SYSTEM_CACHE


def replay_trial(shapes: list[dict], window: int) -> dict[str, Any]:
    """Drive one trial through a ContextManager and record what fired.

    Messages are appended one at a time, mirroring the loop; after each,
    compaction is offered and the assembly measured. Returns the turn indices
    at which pruning and compaction fired, plus the per-turn token counts.
    """
    import asyncio

    context = ContextManager(
        base_system_prompt=production_system_prompt(),
        count_tokens=_estimate,
        max_context=window,
        summarize=_fixed_summary,
    )
    prune_turns: list[int] = []
    prune_shed: list[int] = []
    compact_turns: list[int] = []
    tokens: list[int] = []

    async def drive() -> None:
        for index, message in enumerate(messages_from_shapes(shapes)):
            context.append(message)
            # Mirror harness/loop.py's compact-to-fixpoint pass. Calling
            # maybe_compact() once per message pinned a control flow the
            # harness does not use, and diverged from the real loop on a third
            # of the compacting traces.
            fired = False
            while True:
                size_before = len(context.transcript)
                evicted = await context.maybe_compact()
                if not evicted:
                    break
                fired = True
                if len(context.transcript) >= size_before:
                    break
            if fired:
                compact_turns.append(index)
            plan = context._prune_plan()
            if plan:
                prune_turns.append(index)
                # How much is shed, not only when. PRUNE_TARGET_FRACTION
                # changes the shed *volume* while leaving the firing turns
                # alone, and shedding more only makes the assembly smaller --
                # which N8 deliberately permits, since cheaper is normally an
                # improvement. Destroying more context is not an improvement,
                # so the volume is pinned here.
                prune_shed.append(len(plan))
            system, assembled = context.assemble()
            tokens.append(_estimate(assembled) + len(system) // 4)

    asyncio.run(drive())
    return {
        "prune_turns": prune_turns,
        "prune_shed": prune_shed,
        "compact_turns": compact_turns,
        "total_tokens": sum(tokens),
        "peak_tokens": max(tokens) if tokens else 0,
    }


def _estimate(messages: list[Message]) -> int:
    """The production estimator: total characters / 4.

    Imported behavior rather than a copy -- if the estimator changes (S-109),
    N7 must move, which is exactly the trap S-109 is written to guard.
    """
    from harness.adapters.base import ModelAdapter

    return ModelAdapter.count_tokens(None, messages)  # type: ignore[arg-type]


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Every frozen trial, in file order."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay_corpus(path: Path = CORPUS_PATH) -> dict[str, dict[str, Any]]:
    """Replay every trial at every window. Keyed ``"<trial>@<window>"``."""
    trace: dict[str, dict[str, Any]] = {}
    for record in load_corpus(path):
        for window in REPLAY_WINDOWS:
            trace[f"{record['trial']}@{window}"] = replay_trial(
                record["messages"], window
            )
    return trace
