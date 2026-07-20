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
   :data:`~harness.diligence.MAX_NUDGES`). Then, if the model declared a
   verification command (DESIGN.md §10.3 B1, via the
   ``declare_verification`` tool), the loop re-executes it in the sandbox
   before accepting ``completed`` — gated through the same permission
   engine as a model-issued ``bash`` call, so a deny glob or ASK policy
   applies to verification executions too: exit 0 finishes the run
   (``verification_passed``); anything else injects a fix-and-re-verify
   reminder and continues, consuming the same nudge budget — once nudges
   are exhausted the run completes anyway with the failure recorded
   (``verification_failed`` with ``nudges_exhausted``) so it stays
   auditable. With no declaration, the heuristic alone decides, as before.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harness.adapters.base import AdapterError, ModelAdapter
from harness.context import ContextManager
from harness.diligence import (
    CONTINUE_REMINDER,
    MAX_NUDGES,
    VERIFICATION_FAILED_REMINDER,
    VERIFICATION_TIMEOUT_SECONDS,
    VERIFICATION_TOOL_NAME,
    looks_unfinished,
    truncate_verification_output,
)
from harness.permissions import Decision, Policy, ToolMeta, evaluate
from harness.persistence import RunStore
from harness.sandbox.base import Sandbox
from harness.tools.registry import ToolRegistry
from harness.types import (
    Message,
    Role,
    StopReason,
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

#: Max consecutive-independent times the loop re-prompts a turn that hit the
#: output-token cap (``stop_reason == MAX_TOKENS``) without producing a tool
#: call. Bounds a model that spends its whole per-call budget thinking and
#: emits nothing actionable, so it cannot loop forever; after this many the
#: truncated turn is accepted like any other final answer.
MAX_TRUNCATION_CONTINUES: int = 3

#: Injected when a turn is cut off at the output-token cap with no action taken
#: (see :data:`MAX_TRUNCATION_CONTINUES`). Steers the model to act rather than
#: keep thinking, since the cap means the previous turn produced nothing usable.
TRUNCATION_REMINDER: str = (
    "<system-reminder>\n"
    "Your previous response was cut off at the output-token limit before you "
    "produced a tool call or a complete answer — you spent the whole turn "
    "thinking. Be decisive now: take the next concrete action with a tool "
    "call (write the file, run the command), keeping any prose minimal.\n"
    "</system-reminder>"
)

#: Fraction of ``wall_clock_seconds`` remaining at which the loop injects the
#: one-time wind-down reminder (below): with 900s and 0.2, the reminder lands
#: once ≤180s remain. Chosen to leave the agent one or two turns to land a
#: best-effort answer on disk before an external deadline (e.g. a benchmark
#: harness's per-agent timeout) kills the trial mid-turn.
WIND_DOWN_FRACTION = 0.2

#: Injected once as a user message when the wall-clock budget is nearly spent.
#: Unlike the diligence nudge (which pushes the agent to keep working), this
#: tells it to *stop* and finalize — a working partial answer on disk beats a
#: perfect one that is never written before the deadline. Format with
#: ``remaining=`` and ``budget=`` (whole seconds).
WIND_DOWN_REMINDER: str = (
    "<system-reminder>\n"
    "You are approaching your hard time limit ({remaining}s of {budget}s "
    "remain). Stop exploring and do not start new lines of work. Right now, "
    "make sure your best current solution is fully written to the expected "
    "output location(s) and can actually run — a working partial answer on "
    "disk beats a perfect one you never finish. Do a quick sanity check, "
    "then conclude.\n"
    "</system-reminder>"
)


class Budgets(BaseModel):
    """Per-run loop budgets (DESIGN.md §4.1). Hitting one pauses, not kills.

    ``max_output_tokens`` caps the completion tokens of a *single* model call
    (passed through as the provider ``max_tokens`` param): unlike the run-wide
    ``max_tokens`` ceiling it bounds one turn, so a pathologically long single
    generation cannot consume the whole wall-clock. ``None`` leaves it
    unset. ``wall_clock_seconds`` enables the wind-down reminder (see
    :data:`WIND_DOWN_REMINDER`) when a hard external deadline applies; ``None``
    disables it. Both default off, so nothing changes for callers that do not
    set them.
    """

    model_config = ConfigDict(frozen=True)

    max_turns: int = 50
    max_tokens: int = 1_000_000
    max_output_tokens: int | None = None
    wall_clock_seconds: float | None = None


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
    sandbox:
        Where a declared verification command is re-executed (§10.3 B1).
        ``None`` disables the verification gate entirely — a declared
        command is then ignored, like the pre-B1 loop.
    declared_command:
        A previously declared verification command to re-arm the B1 gate
        with (used by resume: the last persisted ``verification_declared``
        event is rehydrated here so an interrupted run keeps the promise
        that the check "will be re-run before your answer is accepted").
        ``None`` (the default) starts with no declaration, as before.
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
        clock: Callable[[], float] = time.monotonic,
        sandbox: Sandbox | None = None,
        declared_command: str | None = None,
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
        #: Monotonic clock for the wall-clock wind-down check and per-turn
        #: duration recording (§10.2 A5); injectable so tests can drive
        #: deadlines and durations deterministically.
        self.clock = clock
        self.sandbox = sandbox
        #: Rehydrated verification declaration (resume); ``run()`` seeds its
        #: loop-local ``declared_command`` from this.
        self.declared_command = declared_command
        #: Monotonic counter for synthetic verification-execution tool-call
        #: ids, so each execution's permission decision is auditable on its
        #: own row (§4.11: every decision is logged).
        self._verification_seq = 0

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

    async def _gate_verification(self, command: str) -> tuple[Decision, str]:
        """Gate a verification execution through the permission engine.

        The declared command is arbitrary shell that the harness is about
        to run on the model's behalf, so it must clear exactly the policy
        that would have applied had the model run it through the ``bash``
        tool (§4.11: an explicit user deny glob is the highest-precedence
        rule and must not be circumventable by declaring the command as a
        verification instead). ASK defers to :attr:`ask` like any other
        gated call, and the decision — including auto-allows — is recorded
        as a ``decision`` event plus an approval row under a synthetic
        tool-call id, keeping the "every decision is logged" invariant.

        When the registry has no ``bash`` tool (a custom profile that
        includes ``declare_verification`` but excludes ``bash``), there is
        no policy-blessed shell meta to reuse, so the execution is
        conservatively treated as side-effecting rather than silently
        allowed through the benign unknown-tool default.
        """
        try:
            meta = self.registry.get("bash").meta
        except KeyError:
            meta = ToolMeta(side_effect=True)
        decision = evaluate("bash", meta, self.policy)
        decided_by = "policy"
        if decision is Decision.ASK:
            approved = await self.ask("bash", {"command": command}, meta)
            decision = Decision.ALLOW if approved else Decision.DENY
            decided_by = "user"
        self._verification_seq += 1
        self._record_decision(
            ToolCall(
                id=f"verification-exec-{self._verification_seq}",
                name="bash",
                arguments={"command": command},
            ),
            decision,
            decided_by,
        )
        return decision, decided_by

    async def _run_verification(self, command: str) -> tuple[bool, dict]:
        """Execute the declared verification command in the sandbox (B1).

        Returns ``(passed, payload)`` where ``payload`` is the event body
        for ``verification_passed``/``verification_failed``: the command,
        its exit code (``None`` if it could not even execute), its combined
        output (tail-truncated via
        :func:`~harness.diligence.truncate_verification_output`), and
        ``timed_out`` when the :data:`~harness.diligence.VERIFICATION_TIMEOUT_SECONDS`
        bound was hit. A command that fails to *run* (sandbox error) is a
        failed verification, never an exception out of the loop.

        Execution is first gated through :meth:`_gate_verification`; a
        denied command never reaches the sandbox and yields a failed-not-run
        payload with ``denied: True``.
        """
        decision, decided_by = await self._gate_verification(command)
        if decision is not Decision.ALLOW:
            return False, {
                "command": command,
                "exit_code": None,
                "output": (
                    "verification command was not executed: "
                    f"denied by {decided_by}"
                ),
                "denied": True,
            }
        try:
            result = await self.sandbox.exec(
                command, timeout=VERIFICATION_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - any sandbox failure is a
            # verification failure, not a crashed run
            return False, {
                "command": command,
                "exit_code": None,
                "output": f"verification command failed to execute: {exc}",
            }
        parts = [part for part in (result.stdout, result.stderr) if part]
        payload: dict = {
            "command": command,
            "exit_code": result.exit_code,
            "output": truncate_verification_output("\n".join(parts).strip()),
        }
        if result.timed_out:
            payload["timed_out"] = True
        return result.exit_code == 0 and not result.timed_out, payload

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
        truncation_continues = 0
        wound_down = False
        #: The model's currently declared verification command (§10.3 B1);
        #: set/replaced by successful ``declare_verification`` tool calls.
        #: Seeded from the constructor so resume can re-arm a declaration
        #: replayed from the persisted event log.
        declared_command: str | None = self.declared_command
        start = self.clock()
        call_params: dict[str, object] = {}
        if self.budgets.max_output_tokens is not None:
            call_params["max_tokens"] = self.budgets.max_output_tokens

        while True:
            # 1. Budgets: pause resumably, never fail (§4.9).
            spent = total_usage.input_tokens + total_usage.output_tokens
            if turns >= self.budgets.max_turns or spent >= self.budgets.max_tokens:
                return self._finish("paused_budget", None, total_usage, turns)

            # 1b. Wall-clock wind-down: once the hard external deadline is near,
            # inject a one-time reminder to stop exploring and land a working
            # answer on disk (§4.9 land-early discipline). It rides the very
            # next model call, so the agent sees it before acting. Diligence
            # nudges are suppressed afterwards so it may actually conclude.
            if not wound_down and self.budgets.wall_clock_seconds is not None:
                remaining = self.budgets.wall_clock_seconds - (self.clock() - start)
                threshold = self.budgets.wall_clock_seconds * WIND_DOWN_FRACTION
                if remaining <= threshold:
                    self._append_message(
                        Message(
                            role=Role.USER,
                            content=WIND_DOWN_REMINDER.format(
                                remaining=max(0, int(remaining)),
                                budget=int(self.budgets.wall_clock_seconds),
                            ),
                        )
                    )
                    self.store.append_event(
                        self.agent_id,
                        "wind_down",
                        {"remaining_seconds": max(0.0, remaining)},
                    )
                    wound_down = True

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
                call_started = self.clock()
                response = await self.adapter.complete(
                    messages, specs, system, **call_params
                )
            except AdapterError as exc:
                return self._finish("error", str(exc), total_usage, turns)

            turns += 1
            total_usage = total_usage + response.usage
            self.store.record_usage(
                self.run_id,
                self.agent_id,
                self.model,
                response.usage,
                # Wall-clock duration of this model call (§10.2 A5),
                # measured on the same injectable monotonic clock as the
                # wind-down check so tests stay deterministic.
                duration_ms=int((self.clock() - call_started) * 1000),
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
                # Self-verification declarations (§10.3 B1): a successful
                # declare_verification call sets (or replaces) the command
                # the loop will hold the model to at completion time.
                # ``results[i]`` answers ``calls[i]`` by contract, so a
                # denied or invalid call (error result) never counts.
                for tool_call, result in zip(
                    response.message.tool_calls, results
                ):
                    if (
                        tool_call.name == VERIFICATION_TOOL_NAME
                        and not result.is_error
                    ):
                        declared_command = str(
                            tool_call.arguments.get("command", "")
                        )
                        self.store.append_event(
                            self.agent_id,
                            "verification_declared",
                            {
                                "command": declared_command,
                                "description": str(
                                    tool_call.arguments.get("description", "")
                                ),
                            },
                        )
                continue

            # 5a. Truncated turn with no action: the model hit the output-token
            # cap before emitting a tool call (observed: a reasoning model that
            # spent the whole cap thinking and returned an empty message).
            # Accepting it would bank a non-answer as "done", so re-prompt it to
            # act — bounded so a persistently-truncating turn cannot loop.
            if (
                response.stop_reason is StopReason.MAX_TOKENS
                and truncation_continues < MAX_TRUNCATION_CONTINUES
            ):
                truncation_continues += 1
                self._append_message(
                    Message(role=Role.USER, content=TRUNCATION_REMINDER)
                )
                self.store.append_event(
                    self.agent_id,
                    "truncation_continue",
                    {"count": truncation_continues},
                )
                continue

            # 5b. No tool calls: diligence stop-condition check (§4.9).
            final_text = response.message.content
            unfinished, reason = looks_unfinished(
                final_text, self._open_task_count()
            )
            # Once wound down, accept the final answer rather than nudging the
            # agent back into work it no longer has time to finish.
            if unfinished and nudges < MAX_NUDGES and not wound_down:
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

            # 5c. Self-verification gate (§10.3 B1): the model declared a
            # command that proves the goal — re-run it before accepting
            # completed. Failures share the diligence nudge budget above,
            # so a permanently-failing check cannot loop forever; once the
            # budget is spent (or the run is wound down) the answer is
            # accepted anyway, with the failure persisted for audit. With
            # no declaration (or no sandbox), behavior is unchanged.
            if declared_command and self.sandbox is not None:
                passed, payload = await self._run_verification(
                    declared_command
                )
                if passed:
                    self.store.append_event(
                        self.agent_id, "verification_passed", payload
                    )
                elif payload.get("denied"):
                    # Policy denied the execution (§4.11): the check cannot
                    # run no matter what the model changes, so nudging would
                    # only burn budget re-hitting the same deny. Accept the
                    # answer with the not-run failure persisted for audit.
                    self.store.append_event(
                        self.agent_id, "verification_failed", payload
                    )
                elif nudges < MAX_NUDGES and not wound_down:
                    nudges += 1
                    payload["nudge_number"] = nudges
                    self.store.append_event(
                        self.agent_id, "verification_failed", payload
                    )
                    # Persisted as a regular 'message' event (like the
                    # diligence nudge) so resume replays the transcript
                    # the model actually saw.
                    self._append_message(
                        Message(
                            role=Role.USER,
                            content=VERIFICATION_FAILED_REMINDER.format(
                                command=payload["command"],
                                exit_code=payload["exit_code"],
                                output=payload["output"],
                            ),
                        )
                    )
                    continue
                else:
                    # Record *why* the failure is accepted anyway — the
                    # nudge budget ran out, the run wound down, or both.
                    # Folding wind-down into "nudges_exhausted" would
                    # corrupt the failure-classification signal B2/B4 mine.
                    if nudges >= MAX_NUDGES:
                        payload["nudges_exhausted"] = True
                    if wound_down:
                        payload["wound_down"] = True
                    self.store.append_event(
                        self.agent_id, "verification_failed", payload
                    )

            return self._finish("completed", final_text, total_usage, turns)
