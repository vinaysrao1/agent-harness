"""The agent loop — the harness's core primitive (DESIGN.md §4.1).

:class:`AgentLoop` wires every lower layer together for one agent: the model
adapter (§4.2, which owns the shared retry policy internally), the context
manager (§4.3/§4.5), the permission engine (§4.11), the tool
registry/router (§3), the diligence stop-condition check (§4.9), and the
SQLite run store (§4.10), which receives every message, tool call, tool
result, permission decision, compaction, and nudge as an append-only
transcript event the moment it happens — so a crash at any point loses
nothing.

Loop shape per turn:

1. Check budgets — exceeded means pause resumably (``paused_budget``), never
   a hard failure (§4.9: "budgets are pause-points, not failures").
2. ``await context.maybe_compact()`` repeatedly until the assembly is back
   under the threshold (or compaction stops shrinking the transcript) —
   each evicted span is persisted as a ``compaction`` event together with
   its summary text (the §4.3.4 retrieval backstop / resume substitution).
3. Assemble and call ``adapter.complete``. Retries live in exactly one
   layer — the adapters wrap their provider calls in
   :func:`~harness.adapters.base.retry_with_backoff` themselves — so the
   loop calls ``complete`` once; an :class:`AdapterError` that survives the
   adapter's retries (from the model call *or* from the compaction
   summarizer, which uses the same adapter) ends the run with ``error``.
4. If the response carries tool calls: gate each through
   :func:`harness.permissions.evaluate` (ASK defers to the injected ``ask``
   callable, once per call), dispatch every allowed call **concurrently**,
   and append all results in the original tool-call order — deterministic
   regardless of completion order.
5. If it carries none: run :func:`harness.diligence.looks_unfinished`; an
   unfinished-looking answer earns a continue-reminder nudge (bounded by
   :data:`~harness.diligence.MAX_NUDGES`), otherwise the run completes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harness.adapters.base import AdapterError, ModelAdapter
from harness.context import ContextManager
from harness.diligence import CONTINUE_REMINDER, MAX_NUDGES, looks_unfinished
from harness.permissions import Decision, Policy, ToolMeta, evaluate
from harness.persistence import RunStore
from harness.tools.registry import ToolRegistry
from harness.types import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "Budgets",
    "AgentResult",
    "AskCallable",
    "AgentLoop",
]

#: The approval callback for ASK decisions: called with
#: ``(tool_name, arguments, meta)``; returns whether the user approved.
#: The CLI wires this to a terminal prompt; tests inject a stub.
AskCallable = Callable[[str, dict, ToolMeta], Awaitable[bool]]

#: Task-ledger statuses that count as closed for the diligence check;
#: anything else is an open item (§4.9).
_CLOSED_TASK_STATUSES = frozenset(
    {"done", "completed", "complete", "cancelled", "canceled", "closed"}
)


class Budgets(BaseModel):
    """Per-run loop budgets (DESIGN.md §4.1). Hitting one pauses, not kills."""

    model_config = ConfigDict(frozen=True)

    max_turns: int = 50
    max_tokens: int = 1_000_000


class AgentResult(BaseModel):
    """What one :meth:`AgentLoop.run` invocation ultimately produced.

    ``status`` is ``"completed"`` for a normal finish, ``"paused_budget"``
    when a budget was hit (resumable — every event is already persisted),
    or ``"error"`` when the adapter failed after retries. ``final_text`` is
    the model's final message on completion, the adapter's error message on
    error, and ``None`` on a budget pause. ``usage`` and ``turns`` cover
    every model call made by this invocation.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "paused_budget", "error"]
    final_text: str | None
    usage: Usage
    turns: int


class AgentLoop:
    """Runs one agent to completion, budget pause, or error (DESIGN.md §4.1).

    Parameters
    ----------
    adapter:
        The run's model adapter. Adapters own the retry policy (their
        ``complete()`` wraps the provider call in
        :func:`~harness.adapters.base.retry_with_backoff` internally), so
        the loop deliberately adds no second retry layer.
    registry:
        Tool router; supplies the specs handed to the model and dispatches
        allowed calls.
    policy:
        Permission policy every tool call is evaluated against (§4.11).
    store:
        Durability layer; every event is appended as it happens (§4.10).
    run_id / agent_id:
        The store rows this loop writes under (both must already exist).
    context:
        The agent's :class:`~harness.context.ContextManager`; the loop is
        the only mutator of its transcript.
    budgets:
        Turn and token ceilings; hitting either pauses the run resumably.
    ask:
        Async approval callback for ASK decisions, awaited once per call.
    model:
        Model label recorded with each usage row (for `harness cost`).
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        registry: ToolRegistry,
        policy: Policy,
        store: RunStore,
        run_id: str,
        agent_id: str,
        context: ContextManager,
        budgets: Budgets,
        ask: AskCallable,
        *,
        model: str = "unknown",
    ) -> None:
        self.adapter = adapter
        self.registry = registry
        self.policy = policy
        self.store = store
        self.run_id = run_id
        self.agent_id = agent_id
        self.context = context
        self.budgets = budgets
        self.ask = ask
        self.model = model

    # -- persistence helpers -------------------------------------------------

    def _append_message(self, message: Message) -> None:
        """Add ``message`` to the live context and persist it as an event."""
        self.context.append(message)
        self.store.append_event(
            self.agent_id, "message", message.model_dump(mode="json")
        )

    def _record_decision(
        self, call: ToolCall, decision: Decision, decided_by: str
    ) -> None:
        """Persist one permission decision as both an event and an approval
        row (§4.11: every decision is logged, including auto-allows)."""
        self.store.append_event(
            self.agent_id,
            "decision",
            {
                "tool_call_id": call.id,
                "tool_name": call.name,
                "arguments": call.arguments,
                "decision": decision.value,
                "decided_by": decided_by,
            },
        )
        self.store.record_approval(
            self.run_id,
            self.agent_id,
            call.name,
            call.arguments,
            decision.value,
            decided_by,
        )

    def _finish(
        self,
        status: Literal["completed", "paused_budget", "error"],
        final_text: str | None,
        usage: Usage,
        turns: int,
    ) -> AgentResult:
        """Persist the terminal agent status and build the result."""
        self.store.set_agent_status(self.agent_id, status)
        return AgentResult(
            status=status, final_text=final_text, usage=usage, turns=turns
        )

    # -- tool handling -------------------------------------------------------

    def _tool_meta(self, name: str) -> ToolMeta:
        """Look up a tool's permission metadata.

        Unknown tool names get a benign default — the permission engine then
        allows them through to :meth:`ToolRegistry.dispatch`, which turns
        them into a clear ``unknown tool`` error result for the model.
        """
        try:
            return self.registry.get(name).meta
        except KeyError:
            return ToolMeta(side_effect=False)

    async def _resolve_tool_calls(
        self, calls: list[ToolCall]
    ) -> list[ToolResult]:
        """Gate, dispatch, and order the results of one turn's tool calls.

        Every call is evaluated via :func:`~harness.permissions.evaluate`;
        ASK awaits :attr:`ask` exactly once per call. All allowed calls are
        dispatched concurrently with :func:`asyncio.gather`, and the
        returned list is in the **original tool-call order** regardless of
        completion order — ``results[i]`` always answers ``calls[i]``.
        """
        results: list[ToolResult | None] = [None] * len(calls)
        allowed: list[tuple[int, ToolCall]] = []

        for index, call in enumerate(calls):
            meta = self._tool_meta(call.name)
            decision = evaluate(call.name, meta, self.policy)
            decided_by = "policy"
            if decision is Decision.ASK:
                approved = await self.ask(call.name, call.arguments, meta)
                decision = Decision.ALLOW if approved else Decision.DENY
                decided_by = "user"
            self._record_decision(call, decision, decided_by)
            if decision is Decision.ALLOW:
                allowed.append((index, call))
            else:
                results[index] = ToolResult(
                    tool_call_id=call.id,
                    content=f"denied by {decided_by}",
                    is_error=True,
                )

        if allowed:
            dispatched = await asyncio.gather(
                *(self.registry.dispatch(call) for _, call in allowed)
            )
            for (index, _), result in zip(allowed, dispatched):
                results[index] = result

        # Every slot is filled: each call was either denied above or
        # dispatched (gather preserves the order of its awaitables).
        return [result for result in results if result is not None]

    # -- diligence -----------------------------------------------------------

    def _open_task_count(self) -> int:
        """Count task-ledger items not yet closed out (§4.9)."""
        return sum(
            1
            for item in self.store.list_task_items(self.run_id)
            if item.status.lower() not in _CLOSED_TASK_STATUSES
        )

    # -- main loop -----------------------------------------------------------

    async def run(self, goal: str) -> AgentResult:
        """Pursue ``goal`` until completion, a budget pause, or an error.

        Seeds the transcript with ``goal`` as the user message, then loops:
        budget check, compaction check, model call, tool dispatch or
        diligence check — persisting every event as it happens. See the
        class and module docstrings for the full per-turn contract.
        """
        self._append_message(Message(role=Role.USER, content=goal))

        total_usage = Usage()
        turns = 0
        nudges = 0

        while True:
            # 1. Budgets: pause resumably, never fail (§4.9).
            spent = total_usage.input_tokens + total_usage.output_tokens
            if turns >= self.budgets.max_turns or spent >= self.budgets.max_tokens:
                return self._finish("paused_budget", None, total_usage, turns)

            try:
                # 2. Compaction, run to fixpoint: one halving per turn may
                # not bring a heavy transcript back under the threshold, so
                # keep compacting until the assembly fits (or the transcript
                # stops shrinking — the floor against infinite loops). Each
                # evicted span is persisted together with its summary text
                # (§4.3.4 backstop; resume substitutes the summary for the
                # span). The summarizer calls the same adapter as the model
                # call, so its AdapterError is handled identically below.
                while True:
                    size_before = len(self.context.transcript)
                    evicted = await self.context.maybe_compact()
                    if not evicted:
                        break
                    self.store.append_event(
                        self.agent_id,
                        "compaction",
                        {
                            "evicted_count": len(evicted),
                            "evicted": [
                                message.model_dump(mode="json")
                                for message in evicted
                            ],
                            "summary": self.context.last_summary,
                        },
                    )
                    if len(self.context.transcript) >= size_before:
                        break

                # 3. Model call. Retries happen inside the adapter (single
                # retry layer); a failure that survives them ends the run
                # with status 'error' — never an unhandled exception.
                system, messages = self.context.assemble()
                specs = self.registry.specs()
                response = await self.adapter.complete(messages, specs, system)
            except AdapterError as exc:
                return self._finish("error", str(exc), total_usage, turns)

            turns += 1
            total_usage = total_usage + response.usage
            self.store.record_usage(
                self.run_id, self.agent_id, self.model, response.usage
            )
            self._append_message(response.message)

            # 4. Tool calls: gate, dispatch concurrently, append in order.
            if response.message.tool_calls:
                for call in response.message.tool_calls:
                    self.store.append_event(
                        self.agent_id, "tool_call", call.model_dump(mode="json")
                    )
                results = await self._resolve_tool_calls(
                    response.message.tool_calls
                )
                for result in results:
                    self.context.append(
                        Message(role=Role.TOOL, tool_result=result)
                    )
                    self.store.append_event(
                        self.agent_id,
                        "tool_result",
                        result.model_dump(mode="json"),
                    )
                continue

            # 5. No tool calls: diligence stop-condition check (§4.9).
            final_text = response.message.content
            unfinished, reason = looks_unfinished(
                final_text, self._open_task_count()
            )
            if unfinished and nudges < MAX_NUDGES:
                nudges += 1
                reminder = CONTINUE_REMINDER.format(reason=reason)
                # Persisted as a regular 'message' event (plus the 'nudge'
                # bookkeeping event below) so resume replays the exact
                # transcript the model saw — a context mutation that is not
                # persisted would silently diverge on resume (§4.10).
                self._append_message(Message(role=Role.USER, content=reminder))
                self.store.append_event(
                    self.agent_id,
                    "nudge",
                    {"nudge_number": nudges, "reason": reason},
                )
                continue

            return self._finish("completed", final_text, total_usage, turns)
