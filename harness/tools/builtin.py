"""Built-in tool factories (DESIGN.md §3 "Tool router", §4.7, §4.9).

Every function here *binds* a dependency (a :class:`~harness.sandbox.base.Sandbox`,
a :class:`~harness.memory.store.MemoryStore`, a
:class:`~harness.persistence.RunStore` plus a ``run_id``, or a
:class:`~harness.skills.SkillLibrary`) and returns a fully-formed
:class:`~harness.tools.registry.Tool` ready to hand to
:meth:`~harness.tools.registry.ToolRegistry.register`. None of these
functions do any I/O themselves -- they close over the dependency and defer
to it from an async handler closure.

Permission metadata (DESIGN.md §4.11):

- Sandbox tools (``bash``/``read_file``/``write_file``/``edit_file``) are
  ``side_effect=False``. This looks surprising for a tool that can write
  files, but DESIGN.md §4.11 draws the ``side_effect`` line at *external*
  state: gated mode auto-allows writes *inside* the sandbox and only asks
  for "writes outside the sandbox" -- the sandbox boundary itself is the
  isolation, so contained effects never need a human in the loop.
- Memory tools are ``side_effect=False`` -- harness-local storage
  (``~/.harness/memory``), not an external system, "part of the design"
  per the task spec.
- Task-ledger tools are ``side_effect=False`` for the same reason: the
  ledger is harness-local bookkeeping (the run's own SQLite database),
  not an external side effect a user needs to approve.
- ``load_skill`` is ``side_effect=False``: it only reads a skill body into
  context. ``add_instruction`` and ``search_history`` are likewise
  harness-local.

Context binding (DESIGN.md §4.3/§4.5/§4.6/§4.9): ``load_skill_tool``,
``task_update_tool``, and ``add_instruction_tool`` optionally bind the
agent's :class:`~harness.context.ContextManager`. When bound, a loaded
skill body is spliced into the system prompt for the rest of the run
(exempt from tool-result pruning and compaction), every task update
refreshes the task-ledger snapshot rendered into trailing reminders, and
recorded instructions join the instruction ledger that reminders re-inject.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from harness.deadline import Deadline
from harness.diligence import (
    VERIFICATION_LINT_EVENT,
    VERIFICATION_TOOL_NAME,
    WrittenData,
    format_lint_advisory,
    lint_verification,
)
from harness.memory.store import FactType, MemoryStore
from harness.permissions import ToolMeta
from harness.persistence import RunStore, TaskLedgerItem
from harness.sandbox.base import Sandbox
from harness.skills import SkillLibrary
from harness.tools.registry import Tool
from harness.types import ToolSpec

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (context is optional)
    from harness.context import ContextManager

__all__ = [
    "DEFAULT_EXEC_TIMEOUT",
    "MIN_EXEC_SECONDS",
    "LANDING_REFUSAL",
    "MissingArgumentError",
    "bash_tool",
    "read_file_tool",
    "write_file_tool",
    "edit_file_tool",
    "memory_read_fact_tool",
    "memory_write_fact_tool",
    "memory_search_tool",
    "task_update_tool",
    "task_list_tool",
    "load_skill_tool",
    "add_instruction_tool",
    "search_history_tool",
    "declare_verification_tool",
    "render_task_items",
]

#: Metadata shared by every tool in this module (see module docstring).
_NOT_SIDE_EFFECTING = ToolMeta(side_effect=False)


class MissingArgumentError(ValueError):
    """Raised when a tool call's ``arguments`` dict is missing a required key.

    A plain :class:`ValueError` subclass so :meth:`~harness.tools.registry.ToolRegistry.dispatch`'s
    generic exception handling turns it into a clear, actionable error
    `ToolResult` (naming the missing argument) without any special-casing.
    """

    def __init__(self, tool_name: str, key: str) -> None:
        super().__init__(f"{tool_name!r} call is missing required argument {key!r}")


def _require_str(tool_name: str, arguments: dict, key: str) -> str:
    """Fetch ``arguments[key]`` and require it be a non-empty string."""
    if key not in arguments:
        raise MissingArgumentError(tool_name, key)
    value = arguments[key]
    if not isinstance(value, str):
        raise ValueError(
            f"{tool_name!r} argument {key!r} must be a string, got "
            f"{type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# Sandbox tools
# ---------------------------------------------------------------------------


#: Timeout applied to a ``bash`` call that names none.
#:
#: Deliberately a named constant rather than an inline literal, because the
#: distinction matters: a timeout the *harness* chose is not agent intent.
#: 74% of observed bash calls (380 of 513) omit ``timeout`` entirely, and 24
#: of 63 exec-cap events were raised against this bare default — telling a
#: model it "asked for 120s" when it asked for nothing is a false diagnosis,
#: and keying any refusal off it would strand an agent whose one-line ``cp``
#: was going to land the answer. See the ``bash`` handler.
DEFAULT_EXEC_TIMEOUT: float = 120.0

#: Smallest timeout this tool will ever *dispatch*.
#:
#: :meth:`~harness.deadline.Deadline.exec_decision` may legitimately answer
#: ``0.0`` — inside :data:`~harness.deadline.WALL_CLOCK_STOP_FLOOR` nothing
#: new should start, and saying so honestly is the point of the clamp. But
#: handing ``0.0`` to ``sandbox.exec`` is not "nothing started": it starts
#: the command and kills it instantly, so it comes back as a *timeout* —
#: a hard failure for a one-line ``cp`` that would have taken 10ms and
#: landed the answer. That is refusal-by-arithmetic wearing an exec's
#: clothes, which is exactly what Change 1 set out to remove, so a window
#: that rounds to nothing is widened to this instead.
#:
#: 1.0s, against a measured median capped-exec runtime of 0.79s on the
#: 63-row round-2 corpus: enough for the short commands that actually land
#: answers, and a rounding error against the 60s stop floor it is borrowed
#: from. A caller that explicitly asks for less than this still gets what
#: it asked for — the floor never lengthens an agent's own choice.
MIN_EXEC_SECONDS: float = 1.0


#: What to tell the model when its exec timeout was cut, keyed by the
#: ``exec_cap`` reason. The reason changes the *correct response*: a share
#: cap early in a run means "break this work up", while a landing cap near
#: the deadline means "stop starting long work and write your answer down".
#: One generic note would teach the wrong lesson in one of the two cases.
#: ``"landing"`` is not an ``exec_cap`` reason — it is the loop's explicit
#: final-turn state — but its advice belongs in the same table so all the
#: near-deadline wording the model ever sees is written in one place.
_CAP_ADVICE: dict[str, str] = {
    "share": (
        "no single command may use more than half the run's total budget; "
        "break the work into smaller steps rather than re-running this"
    ),
    "band": (
        "time is held back so you get a turn to wind down and land your "
        "answer before the deadline; do not start another long command"
    ),
    "reserve": (
        "time is held back so you get a turn to land your answer before "
        "the deadline; do not start another long command"
    ),
    "landing": (
        "this is your final turn before the run's wall-clock deadline: "
        "there is no longer time for a command to finish and for you to act "
        "on its output. Do not retry — say where your answer is (the paths "
        "you have already written), or write it now with write_file, and "
        "then give your final answer"
    ),
}

#: What ``bash`` returns instead of running a command once the loop has
#: declared the landing turn. A *normal* result, not an error: the refusal
#: is the harness's decision, not the agent's mistake, and routing it
#: through the error path would spend nudge and truncation budget the
#: landing turn needs.
LANDING_REFUSAL: str = f"command not run — {_CAP_ADVICE['landing']}."


def bash_tool(
    sandbox: Sandbox,
    deadline: Deadline | None = None,
    store: RunStore | None = None,
    agent_id: str | None = None,
) -> Tool:
    """Build the ``bash`` tool: runs a shell command in ``sandbox``.

    Result is exit code + stdout + stderr formatted as plain text -- no
    JSON wrapping, so the model reads it the way a human reads a terminal.
    Empty stdout/stderr sections are omitted (DESIGN.md §10.2 A4: trim
    result-string boilerplate) rather than printed as empty headers -- most
    successful commands produce no stderr, and paying two boilerplate lines
    for that on every single `bash` call (the highest-volume tool) adds up.

    ``deadline`` (wind-down plan §Fix 3a), when given, caps the requested
    ``timeout`` by what wall-clock is actually left, via
    :meth:`~harness.deadline.Deadline.exec_cap`: a command allowed to run
    past the run's own external kill would never get its result back to the
    model at all, and one allowed to swallow the whole budget never hands
    control back at all. That method owns the three bounds (share cap, band
    guarantee, landing reserve) and the floor below which no cap goes;
    ``bash`` only reports the outcome. ``deadline=None`` (the default) or an
    unset deadline budget is a pure passthrough -- today's behavior,
    unchanged.

    ``store``/``agent_id``, when both given, turn every capped exec into an
    ``exec_capped`` transcript event carrying ``requested``, ``effective``,
    ``remaining``, ``budget``, ``reserve``, ``applied_reserve``, ``reason``
    and ``purpose``. It is written whenever the cap bites, including when
    the command then succeeds inside the shorter window -- the point is to
    measure the cap's own cost from real runs rather than re-deriving it
    from simulation.

    The two reserve fields are deliberately both there: ``reserve`` is the
    base landing reserve (floor + adaptive allowance), ``applied_reserve``
    is what the cap arithmetic actually held back -- larger when the band
    softener raised it. Only the pair answers the question the event exists
    to answer, since ``remaining - effective`` recovers the applied reserve
    for ``reason`` ``"band"``/``"reserve"`` but not for ``"share"``.

    Two deadline behaviours beyond the cap itself:

    - **An omitted ``timeout`` is the harness's number, not the agent's.**
      74% of observed bash calls omit it, and capping the resulting bare
      :data:`DEFAULT_EXEC_TIMEOUT` produced 24 of 63 ``exec_capped`` events
      that told the model it had "requested 120s" when it had requested
      nothing. So an omitted timeout is pre-shrunk to
      :meth:`~harness.deadline.Deadline.affordable_exec_seconds` instead:
      the command still runs, in whatever window the run can afford, and no
      cap is reported or recorded because none bit. No refusal is ever
      keyed off it -- the median capped exec on that corpus ran 0.79s, so
      small windows are useful, and refusing an agent's one-line ``cp``
      because *the harness* defaulted to 120s is exactly the failure this
      avoids.
    - **The landing turn.** Once the loop calls
      :meth:`~harness.deadline.Deadline.begin_landing`, ``bash`` stops
      executing and returns :data:`LANDING_REFUSAL` as an ordinary (not
      ``is_error``) result. That gate reads explicit loop state and nothing
      else; it never inspects the requested timeout.
    - **No zero-second exec.** The cap arithmetic may legitimately answer
      ``0.0``; dispatching that would turn a 10ms ``cp`` into a timeout,
      so the dispatched window is floored at :data:`MIN_EXEC_SECONDS`. The
      only two ways a command is not run are the landing refusal above and
      the policy layer -- never a number that rounded to nothing.
    """

    spec = ToolSpec(
        name="bash",
        description=(
            "Run a shell command in the sandbox workspace and return its "
            "exit code, stdout, and stderr. Times out after `timeout` "
            "seconds; omit it and the default (120s) adapts down to "
            "whatever the run's remaining wall-clock can afford. Near the "
            "run's wall-clock deadline, an explicit `timeout` may be capped "
            "shorter to preserve time to land a final answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 120).",
                },
            },
            "required": ["command"],
        },
    )

    async def handler(arguments: dict) -> str:
        command = _require_str("bash", arguments, "command")
        if deadline is not None and deadline.landing:
            # The loop has declared the final turn. Refuse from explicit
            # state, before any arithmetic: nothing here inspects what
            # timeout was asked for, so no default of the harness's own can
            # ever be mistaken for an agent asking to run something long.
            return LANDING_REFUSAL
        explicit = arguments.get("timeout")
        from_default = explicit is None
        if from_default:
            # An omitted timeout is the harness's number, not the agent's.
            # Shrink it to what the run can afford instead of asking for
            # 120s and reporting a cap: the cap then cannot bite, which is
            # what makes "capped from your requested Ns" always true when it
            # is printed.
            requested = DEFAULT_EXEC_TIMEOUT
            affordable = (
                deadline.affordable_exec_seconds()
                if deadline is not None
                else None
            )
            if affordable is not None:
                # Never below MIN_EXEC_SECONDS: `affordable` is 0.0 once
                # `remaining <= WALL_CLOCK_STOP_FLOOR`, and a default the
                # harness shrank to nothing would fail the agent's command
                # outright instead of shortening it.
                requested = max(
                    MIN_EXEC_SECONDS, min(DEFAULT_EXEC_TIMEOUT, affordable)
                )
        else:
            requested = float(explicit)
        remaining = deadline.remaining() if deadline is not None else None
        effective, capped, reason = requested, False, None
        if deadline is not None:
            decision = deadline.exec_decision(requested)
            effective, capped, reason = (
                decision.effective,
                decision.capped,
                decision.reason,
            )
            # A window that rounds to nothing is widened, never dispatched
            # (see MIN_EXEC_SECONDS). Reachable whenever `remaining` has
            # fallen to the stop floor without the loop's landing turn
            # arming — on a run's very first turn the call window is empty,
            # so the band is deliberately disabled, and generation can
            # spend the margin between the loop-top check and this exec.
            # The cap is still *reported*: the agent asked for 30s and is
            # getting 1s, which is true and worth saying. Only the silent
            # sub-second window is removed.
            #
            # The floor is MIN_EXEC_SECONDS, not zero: `remaining` strictly
            # between the stop floor and one second above it clamps to a
            # positive fraction of a second, which is a window the harness's
            # own arithmetic invented and no command can use. `min` with
            # `requested` keeps the one documented exemption — a caller that
            # explicitly asks for less than a second gets what it asked for,
            # never less.
            floor = min(requested, MIN_EXEC_SECONDS)
            if 0.0 < requested and effective < floor:
                effective = floor
            # By construction the pre-clamped default is never capped; the
            # suppression is belt-and-braces so the invariant "a reported
            # cap names a number the agent actually chose" is enforced here
            # rather than merely implied by the arithmetic above.
            #
            # `effective < requested` is the other half of the same
            # invariant, and it is the floor above that makes it load-
            # bearing: an explicit sub-second ask that the deadline clamped
            # is restored to exactly `requested`, so `decision.capped` is
            # True while no cap actually bit. Reporting one there would
            # tell the agent "capped to 0.25s" when it asked for 0.25s and
            # got 0.25s, and would put a row with requested == effective
            # into the `exec_capped` corpus this docstring cites for
            # calibration. A cap is reported only when it took something.
            capped = capped and not from_default and effective < requested
            if capped and store is not None and agent_id is not None:
                store.append_event(
                    agent_id,
                    "exec_capped",
                    {
                        "requested": requested,
                        "effective": effective,
                        "remaining": remaining,
                        "budget": deadline.budget,
                        "reserve": deadline.landing_reserve(),
                        "applied_reserve": decision.reserve,
                        "reason": reason,
                        "purpose": "exploratory",
                    },
                )
        advice = _CAP_ADVICE.get(reason or "", "")
        # A default the deadline shortened is worth saying — the model needs
        # to know the window was small — but it is said as what it is, not
        # as a cap on something the model asked for.
        shortened_default = from_default and effective < DEFAULT_EXEC_TIMEOUT
        short_note = (
            f"no timeout was given, so the default was shortened to "
            f"{effective}s to fit the ~{remaining}s of wall-clock remaining"
        )
        result = await sandbox.exec(command, timeout=effective)
        lines = [f"exit code: {result.exit_code}"]
        if result.timed_out:
            if capped:
                lines.append(
                    f"(command timed out after {effective}s — capped from "
                    f"your requested {requested}s with ~{remaining}s of "
                    f"wall-clock remaining: {advice})"
                )
            elif shortened_default:
                lines.append(
                    f"(command timed out after {effective}s — {short_note}: "
                    f"{_CAP_ADVICE['reserve']})"
                )
            else:
                lines.append(f"(command timed out after {effective}s)")
        elif capped:
            lines.append(
                f"(note: timeout was capped to {effective}s to fit the "
                f"remaining time budget: {advice})"
            )
        elif shortened_default:
            lines.append(f"(note: {short_note})")
        if result.stdout:
            lines.append("--- stdout ---")
            lines.append(result.stdout)
        if result.stderr:
            lines.append("--- stderr ---")
            lines.append(result.stderr)
        if not result.stdout and not result.stderr:
            lines.append("(no output)")
        return "\n".join(lines)

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def read_file_tool(sandbox: Sandbox) -> Tool:
    """Build the ``read_file`` tool: reads a text file from ``sandbox``.

    Three optional arguments (DESIGN.md §10.2 A4) turn a whole-file read into
    a scoped one, so large files stop being read (and truncated at the
    registry's :data:`~harness.tools.registry.MAX_RESULT_BYTES` cap, §4.7/A4)
    by default:

    - ``offset``/``limit`` page through the file by 1-indexed line number.
    - ``pattern`` (a regex) greps within that window (the whole file if
      ``offset``/``limit`` are absent), returning only matching lines.

    Both scoped modes prefix every returned line with its absolute line
    number (``N:text``, mirroring ``grep -n``) so the model can hand that
    number straight to a follow-up ``offset`` or to `edit_file`. Calling with
    none of the three preserves the original behavior exactly: the raw file
    content, unmodified.
    """

    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a file in the sandbox workspace. Returns the whole file "
            "by default -- but for files that might be large, PREFER a "
            "scoped read instead: `offset`/`limit` to page through a line "
            "range, or `pattern` (a regex) to grep for matching lines "
            "(optionally within that range). Both scoped modes prefix each "
            "returned line with its 1-indexed line number, like `grep -n`, "
            "so you know what to pass to `edit_file` or to a follow-up "
            "`offset`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the sandbox workspace root.",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "1-indexed line number to start reading from "
                        "(default: 1, the start of the file)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of lines to read starting at "
                        "`offset` (default: through the end of the file). "
                        "Combine with `offset` to page through a large file."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Regex; if set, only lines matching it are "
                        "returned (searched within `offset`/`limit` if "
                        "those are also given), each prefixed with its "
                        "line number -- like `grep -n`."
                    ),
                },
            },
            "required": ["path"],
        },
    )

    async def handler(arguments: dict) -> str:
        path = _require_str("read_file", arguments, "path")
        content = await sandbox.read_file(path)

        raw_offset = arguments.get("offset")
        raw_limit = arguments.get("limit")
        pattern = arguments.get("pattern")
        if raw_offset is None and raw_limit is None and pattern is None:
            return content

        lines = content.splitlines()
        total = len(lines)

        offset = int(raw_offset) if raw_offset is not None else 1
        if offset < 1:
            raise ValueError(
                "'read_file' argument 'offset' must be >= 1 (lines are 1-indexed)"
            )

        if raw_limit is not None:
            limit = int(raw_limit)
            if limit < 1:
                raise ValueError("'read_file' argument 'limit' must be >= 1")
            end_index = min(offset - 1 + limit, total)
        else:
            end_index = total

        start_index = min(offset - 1, total)
        window = lines[start_index:end_index]
        last_line = start_index + len(window)

        if pattern is not None:
            if not isinstance(pattern, str):
                raise ValueError("'read_file' argument 'pattern' must be a string")
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"'read_file' argument 'pattern' is not a valid regex: {exc}"
                ) from None
            matched = [
                (start_index + 1 + i, line)
                for i, line in enumerate(window)
                if regex.search(line)
            ]
            if not matched:
                return (
                    f"no lines in {path} matched pattern {pattern!r} "
                    f"(searched lines {start_index + 1}-{last_line} of {total})"
                )
            body = "\n".join(f"{n}:{line}" for n, line in matched)
            return (
                f"{body}\n[{len(matched)} match(es) among lines "
                f"{start_index + 1}-{last_line} of {total} in {path}]"
            )

        if not window:
            return f"{path} has {total} line(s); offset {offset} is beyond end of file"

        body = "\n".join(
            f"{n}:{line}" for n, line in enumerate(window, start=start_index + 1)
        )
        return f"{body}\n[lines {start_index + 1}-{last_line} of {total} in {path}]"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def write_file_tool(sandbox: Sandbox) -> Tool:
    """Build the ``write_file`` tool: writes/overwrites a file in ``sandbox``."""

    spec = ToolSpec(
        name="write_file",
        description=(
            "Write content to a file in the sandbox workspace, creating it "
            "(and any missing parent directories) if needed, or overwriting "
            "it if it already exists."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the sandbox workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to write.",
                },
            },
            "required": ["path", "content"],
        },
    )

    async def handler(arguments: dict) -> str:
        path = _require_str("write_file", arguments, "path")
        content = _require_str("write_file", arguments, "content")
        await sandbox.write_file(path, content)
        size = len(content.encode("utf-8"))
        return f"wrote {size} bytes to {path}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def edit_file_tool(sandbox: Sandbox) -> Tool:
    """Build the ``edit_file`` tool: an old/new-string replacement in ``sandbox``.

    Mirrors Claude Code's tool conventions (DESIGN.md §8): ``old_string``
    must match uniquely unless ``replace_all`` is set.
    """

    spec = ToolSpec(
        name="edit_file",
        description=(
            "Replace `old_string` with `new_string` in a file in the sandbox "
            "workspace. `old_string` must match the file's current content "
            "exactly, and uniquely unless `replace_all` is set."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the sandbox workspace root.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to replace it with.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match (default false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    )

    async def handler(arguments: dict) -> str:
        path = _require_str("edit_file", arguments, "path")
        old_string = _require_str("edit_file", arguments, "old_string")
        new_string = _require_str("edit_file", arguments, "new_string")
        replace_all = bool(arguments.get("replace_all", False))
        await sandbox.edit_file(
            path, old_string, new_string, replace_all=replace_all
        )
        return f"edited {path}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


def memory_read_fact_tool(store: MemoryStore) -> Tool:
    """Build the ``memory_read_fact`` tool: reads one semantic fact by name."""

    spec = ToolSpec(
        name="memory_read_fact",
        description="Read the full body of a semantic-memory fact by its name.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The fact's kebab-case name, as listed in the memory index.",
                },
            },
            "required": ["name"],
        },
    )

    async def handler(arguments: dict) -> str:
        name = _require_str("memory_read_fact", arguments, "name")
        fact = store.read_fact(name)
        header = f"[{fact.name}] ({fact.type.value}) {fact.description}"
        if fact.sources:
            header += f"\nsources: {', '.join(fact.sources)}"
        return f"{header}\n\n{fact.body}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def memory_write_fact_tool(store: MemoryStore) -> Tool:
    """Build the ``memory_write_fact`` tool: creates or updates a fact.

    A second call with the same ``name`` overwrites the previous fact --
    the memory system's write policy (DESIGN.md §4.4.2) is "check the index
    for an existing entry to update before creating."
    """

    spec = ToolSpec(
        name="memory_write_fact",
        description=(
            "Create or update a semantic-memory fact. Calling this again "
            "with the same `name` overwrites the previous fact."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Kebab-case fact name (e.g. 'prefers-dark-mode').",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary shown in the always-in-context index.",
                },
                "type": {
                    "type": "string",
                    "enum": [t.value for t in FactType],
                    "description": "Fact category.",
                },
                "body": {
                    "type": "string",
                    "description": "The full fact content.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Episode filenames (or other provenance) this fact was derived from.",
                },
            },
            "required": ["name", "description", "type", "body"],
        },
    )

    async def handler(arguments: dict) -> str:
        name = _require_str("memory_write_fact", arguments, "name")
        description = _require_str("memory_write_fact", arguments, "description")
        fact_type = _require_str("memory_write_fact", arguments, "type")
        body = _require_str("memory_write_fact", arguments, "body")
        sources = arguments.get("sources") or []
        fact = store.write_fact(name, description, fact_type, body, sources=sources)
        return f"wrote fact {fact.name!r} ({fact.type.value})"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def memory_search_tool(store: MemoryStore) -> Tool:
    """Build the ``memory_search`` tool: full-text search over facts and episodes."""

    spec = ToolSpec(
        name="memory_search",
        description=(
            "Case-insensitive substring search over active facts and "
            "episode journals; returns matching lines with their source."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for.",
                },
            },
            "required": ["query"],
        },
    )

    async def handler(arguments: dict) -> str:
        query = _require_str("memory_search", arguments, "query")
        results = store.search(query)
        if not results:
            return f"no matches for {query!r}"
        return "\n".join(
            f"[{kind}] {name}: {line}" for kind, name, line in results
        )

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Task ledger tools
# ---------------------------------------------------------------------------


def render_task_items(items: list[TaskLedgerItem]) -> str:
    """Render task-ledger items as the model-facing listing / snapshot."""
    if not items:
        return "(task ledger is empty)"
    lines = []
    for item in items:
        line = f"- [{item.status}] {item.item_id}: {item.description}"
        if item.evidence:
            line += f" (evidence: {item.evidence})"
        lines.append(line)
    return "\n".join(lines)


def task_update_tool(
    store: RunStore, run_id: str, context: "ContextManager | None" = None
) -> Tool:
    """Build the ``task_update`` tool: upserts one task-ledger item.

    This is the diligence machinery's write path (DESIGN.md §4.9): the
    system prompt requires evidence-backed completion, so ``evidence``
    should point at something concrete in the transcript (e.g. "pytest
    output at turn 12") rather than a bare claim.

    When ``context`` is bound, every update also refreshes the context's
    task-ledger snapshot so the §4.5 trailing reminder mirrors the live
    todo list ("mirrored into context", §4.9) instead of staying empty.
    """

    spec = ToolSpec(
        name="task_update",
        description=(
            "Create or update one item in this run's task ledger. Calling "
            "this again with the same `item_id` overwrites its previous "
            "description/status/evidence."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "Stable identifier for this ledger item.",
                },
                "description": {
                    "type": "string",
                    "description": "What this item is.",
                },
                "status": {
                    "type": "string",
                    "description": "e.g. 'pending', 'in_progress', 'done', 'blocked'.",
                },
                "evidence": {
                    "type": "string",
                    "description": "Concrete evidence backing a 'done' status (e.g. test output).",
                },
            },
            "required": ["item_id", "description", "status"],
        },
    )

    async def handler(arguments: dict) -> str:
        item_id = _require_str("task_update", arguments, "item_id")
        description = _require_str("task_update", arguments, "description")
        status = _require_str("task_update", arguments, "status")
        evidence = arguments.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError("'task_update' argument 'evidence' must be a string")
        store.upsert_task_item(run_id, item_id, description, status, evidence)
        if context is not None:
            context.set_task_snapshot(
                render_task_items(store.list_task_items(run_id))
            )
        return f"updated task item {item_id!r}: {status}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def task_list_tool(store: RunStore, run_id: str) -> Tool:
    """Build the ``task_list`` tool: lists every item in this run's task ledger."""

    spec = ToolSpec(
        name="task_list",
        description="List every item currently in this run's task ledger.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )

    async def handler(arguments: dict) -> str:
        return render_task_items(store.list_task_items(run_id))

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Skill tool
# ---------------------------------------------------------------------------


def load_skill_tool(
    library: SkillLibrary, context: "ContextManager | None" = None
) -> Tool:
    """Build the ``load_skill`` tool: splices a skill's full body into context.

    Progressive disclosure (DESIGN.md §4.6): only names+descriptions sit in
    the system prompt; this tool is how the model (or a `/name` invocation)
    fetches the full instruction body on demand.

    When ``context`` is bound (as the orchestrator always does), the body is
    spliced into the system prompt via
    :meth:`~harness.context.ContextManager.add_skill_body` — exempt from
    tool-result pruning and compaction, so the skill genuinely applies "for
    the rest of the run" — and the tool result is a short acknowledgment.
    Without a context the body itself is returned (it then lives only in the
    transcript, subject to pruning; suitable for standalone/test use only).
    """

    spec = ToolSpec(
        name="load_skill",
        description=(
            "Load the full instruction body of a skill by name, splicing it "
            "into context for the rest of the run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill's name, as listed in the skills index.",
                },
            },
            "required": ["name"],
        },
    )

    async def handler(arguments: dict) -> str:
        name = _require_str("load_skill", arguments, "name")
        body = library.load(name)
        if context is None:
            return body
        context.add_skill_body(body, name)
        return (
            f"loaded skill {name!r} ({len(body)} chars); its full body is "
            "now in your system prompt for the rest of the run"
        )

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Instruction-ledger tool
# ---------------------------------------------------------------------------


def add_instruction_tool(
    store: RunStore, run_id: str, context: "ContextManager | None" = None
) -> Tool:
    """Build the ``add_instruction`` tool: records one standing constraint.

    The instruction-ledger write path (DESIGN.md §4.5): constraints recorded
    here are persisted to the run's ``instruction_ledger`` table (so they
    survive crashes and are reloaded on resume) and — when ``context`` is
    bound — join the in-context ledger that the trailing system reminder
    re-injects every few turns and immediately after every compaction.
    """

    spec = ToolSpec(
        name="add_instruction",
        description=(
            "Record a standing user constraint or instruction (e.g. 'never "
            "push to main', 'always reply in French') in the run's "
            "instruction ledger. Recorded instructions are persisted, "
            "re-shown to you periodically, and survive context compaction. "
            "Reusing an `item_id` overwrites that entry."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The constraint, stated imperatively.",
                },
                "source": {
                    "type": "string",
                    "description": "Where it came from (default 'user').",
                },
                "item_id": {
                    "type": "string",
                    "description": "Stable id for this entry (default: auto-numbered).",
                },
            },
            "required": ["instruction"],
        },
    )

    async def handler(arguments: dict) -> str:
        instruction = _require_str("add_instruction", arguments, "instruction")
        source = arguments.get("source") or "user"
        if not isinstance(source, str):
            raise ValueError("'add_instruction' argument 'source' must be a string")
        item_id = arguments.get("item_id")
        if item_id is None:
            item_id = f"instr-{len(store.list_instructions(run_id)) + 1}"
        elif not isinstance(item_id, str):
            raise ValueError("'add_instruction' argument 'item_id' must be a string")
        store.upsert_instruction(run_id, item_id, instruction, source)
        if context is not None:
            context.add_instruction(instruction, source)
        return f"recorded instruction {item_id!r}: {instruction}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Transcript-history retrieval tool
# ---------------------------------------------------------------------------

#: Characters of context shown around each search_history match.
_HISTORY_SNIPPET_RADIUS = 120

#: Default (and maximum) number of search_history matches returned.
_HISTORY_DEFAULT_LIMIT = 20


def search_history_tool(store: RunStore, run_id: str) -> Tool:
    """Build the ``search_history`` tool: grep over this run's event log.

    The §4.3 layer-4 retrieval backstop: pruned tool results and compacted
    transcript spans remain in the append-only ``transcript_events`` table,
    and this tool is how the agent gets them back — turning the failure mode
    from "forgotten" into "must think to look". Pruning stubs point here.
    """

    spec = ToolSpec(
        name="search_history",
        description=(
            "Case-insensitive substring search over this run's full "
            "persisted event log (every message, tool call, and tool "
            "result, including content that was pruned or compacted out of "
            "your context). Returns matching events with a snippet around "
            "each match. Use this to recover old tool output referenced by "
            "'[pruned: ...]' stubs or evicted by compaction summaries."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for (substring match).",
                },
                "limit": {
                    "type": "number",
                    "description": f"Maximum matches (default {_HISTORY_DEFAULT_LIMIT}).",
                },
            },
            "required": ["query"],
        },
    )

    async def handler(arguments: dict) -> str:
        query = _require_str("search_history", arguments, "query")
        limit = int(arguments.get("limit", _HISTORY_DEFAULT_LIMIT))
        if limit < 1:
            raise ValueError("'search_history' argument 'limit' must be >= 1")
        needle = query.lower()
        matches: list[str] = []
        total = 0
        for event in store.load_run_events(run_id):
            haystack = json.dumps(event.payload, ensure_ascii=False)
            index = haystack.lower().find(needle)
            if index < 0:
                continue
            total += 1
            if len(matches) >= limit:
                continue
            start = max(index - _HISTORY_SNIPPET_RADIUS, 0)
            end = min(index + len(query) + _HISTORY_SNIPPET_RADIUS, len(haystack))
            snippet = haystack[start:end]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(haystack) else ""
            matches.append(
                f"[agent {event.agent_id} seq {event.seq} {event.kind}] "
                f"{prefix}{snippet}{suffix}"
            )
        if not matches:
            return f"no matches for {query!r} in this run's event log"
        header = f"{total} match(es) for {query!r}"
        if total > len(matches):
            header += f" (showing first {len(matches)})"
        return "\n".join([header, *matches])

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


# ---------------------------------------------------------------------------
# Self-verification tool
# ---------------------------------------------------------------------------


def declare_verification_tool(
    written_data: Callable[[], WrittenData | None] | None = None,
    store: RunStore | None = None,
    agent_id: str | None = None,
) -> Tool:
    """Build the ``declare_verification`` tool (DESIGN.md §10.3 B1).

    The model declares the shell command that proves the goal is met; the
    *loop* holds it to that claim — it watches for successful calls to this
    tool (by :data:`~harness.diligence.VERIFICATION_TOOL_NAME`) and
    re-executes the declared command in the sandbox before accepting a
    final answer as ``completed`` (see :class:`~harness.loop.AgentLoop`).
    The handler itself only validates and acknowledges the declaration —
    deliberately stateless, so the mechanism's single source of truth is
    the transcript the loop already persists.

    ``written_data`` is an accessor for the loop's live
    :class:`~harness.diligence.WrittenData` map (an accessor, not the map:
    the registry is built before the loop that owns it). When given, every
    declaration is run through
    :func:`~harness.diligence.lint_verification` and any findings are
    appended to the tool result as an advisory the model reads on its next
    turn. ``store``/``agent_id``, when both given, additionally persist a
    :data:`~harness.diligence.VERIFICATION_LINT_EVENT` event.

    **The lint is warn-only.** A flagged declaration is still accepted,
    still recorded, and still the command the loop will re-run; nothing
    about control flow depends on a finding.
    """

    spec = ToolSpec(
        name=VERIFICATION_TOOL_NAME,
        description=(
            "Declare the shell command that will prove this task's goal is "
            "met (e.g. running the test suite); before your final answer "
            "is accepted, the harness re-runs it in the sandbox and treats "
            "the task as complete only if it exits 0. Declare your check "
            "early — as soon as you know what would prove success, not "
            "after the work is done. Calling this again replaces the "
            "previously declared command."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command that exits 0 exactly when the goal "
                        "is met (run in the sandbox workspace)."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "What passing this command proves.",
                },
            },
            "required": ["command", "description"],
        },
    )

    async def handler(arguments: dict) -> str:
        command = _require_str(VERIFICATION_TOOL_NAME, arguments, "command")
        description = _require_str(
            VERIFICATION_TOOL_NAME, arguments, "description"
        )
        if not command.strip():
            raise ValueError(
                f"{VERIFICATION_TOOL_NAME!r} argument 'command' must be "
                "non-empty"
            )
        acknowledgement = (
            f"verification declared: {command!r} ({description}). It will "
            "be re-run before your final answer is accepted; exit 0 means "
            "verified."
        )
        findings = lint_verification(
            command, written_data() if written_data is not None else None
        )
        if not findings:
            return acknowledgement
        if store is not None and agent_id is not None:
            store.append_event(
                agent_id,
                VERIFICATION_LINT_EVENT,
                {
                    "command": command,
                    "findings": [finding.as_payload() for finding in findings],
                    "action": "warn",
                },
            )
        return f"{acknowledgement}\n\n{format_lint_advisory(findings)}"

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)
