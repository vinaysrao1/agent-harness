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
from typing import TYPE_CHECKING

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


def bash_tool(sandbox: Sandbox) -> Tool:
    """Build the ``bash`` tool: runs a shell command in ``sandbox``.

    Result is exit code + stdout + stderr formatted as plain text -- no
    JSON wrapping, so the model reads it the way a human reads a terminal.
    """

    spec = ToolSpec(
        name="bash",
        description=(
            "Run a shell command in the sandbox workspace and return its "
            "exit code, stdout, and stderr. Times out after `timeout` "
            "seconds (default 120)."
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
        timeout = float(arguments.get("timeout", 120))
        result = await sandbox.exec(command, timeout=timeout)
        lines = [f"exit code: {result.exit_code}"]
        if result.timed_out:
            lines.append(f"(command timed out after {timeout}s)")
        lines.append("--- stdout ---")
        lines.append(result.stdout)
        lines.append("--- stderr ---")
        lines.append(result.stderr)
        return "\n".join(lines)

    return Tool(spec=spec, meta=_NOT_SIDE_EFFECTING, handler=handler)


def read_file_tool(sandbox: Sandbox) -> Tool:
    """Build the ``read_file`` tool: reads a text file from ``sandbox``."""

    spec = ToolSpec(
        name="read_file",
        description="Read the text content of a file in the sandbox workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the sandbox workspace root.",
                },
            },
            "required": ["path"],
        },
    )

    async def handler(arguments: dict) -> str:
        path = _require_str("read_file", arguments, "path")
        return await sandbox.read_file(path)

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
        return (
            f"[{fact.name}] ({fact.type.value}) {fact.description}\n"
            f"sources: {', '.join(fact.sources) or '(none)'}\n\n{fact.body}"
        )

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
