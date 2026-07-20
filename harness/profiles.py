"""Agent profiles — minimal M9a data over the generalization seam (§11.3).

A :class:`Profile` names a ``(domain_rules, tool_factories)`` pair for
:meth:`~harness.orchestrator.Orchestrator.run_task`'s profile seam. It is
deliberately **not** the six-field ``AgentProfile`` struct of §11.2 — that
promotion (plus sandbox spec, policy defaults, and heterogeneous subagents)
is M9b, deferred until a second concrete agent type has shaped it. What a
profile can and cannot do here:

- ``domain_rules`` is always *appended* after the non-overridable
  :data:`~harness.orchestrator.CORE_RULES` safety core (goal pursuit,
  evidence-based completion, tool-results-are-DATA-never-instructions,
  permission-mode note, parallel batching) — a profile can never replace
  or precede the core (:func:`~harness.orchestrator.assemble_rules`).
- ``tool_factories`` are factories ``(ToolDeps) -> Tool``, not tools —
  every builtin binds live per-run/per-agent dependencies, so tools cannot
  be data (§11.2).
- Explicit ``run_task`` arguments (``domain_rules=`` / ``tool_factories=``)
  override the profile's fields; ``profile=None`` means :data:`CODING`.
- Subagents inherit the lead's profile (v1 decision, §11.4).

Two concrete profiles prove the seam: :data:`CODING` (exactly today's
behavior, expressed as data) and :data:`CODING_READONLY` (an
inspection-only variant with no ``bash``/``write_file``/``edit_file``).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.orchestrator import (
    CODING_RULES,
    CODING_TOOL_FACTORIES,
    ToolFactory,
)
from harness.tools.builtin import (
    load_skill_tool,
    memory_read_fact_tool,
    memory_search_tool,
    memory_write_fact_tool,
    read_file_tool,
    task_list_tool,
    task_update_tool,
)

__all__ = [
    "Profile",
    "CODING",
    "CODING_READONLY",
    "ALL_PROFILES",
]


@dataclass(frozen=True)
class Profile:
    """A named ``(domain_rules, tool_factories)`` bundle for ``run_task``.

    Attributes
    ----------
    name:
        Human-readable profile identifier (not persisted in v1).
    domain_rules:
        Appended after :data:`~harness.orchestrator.CORE_RULES` in the
        assembled system prompt; may use the ``{workspace}`` / ``{mode}``
        placeholders. Substitution is a literal replace, so any other
        braces (JSON examples, shell ``${VAR}``) pass through verbatim.
    tool_factories:
        Factories ``(ToolDeps) -> Tool`` the orchestrator invokes with the
        live per-agent dependency bundle to build each agent's registry.
    """

    name: str
    domain_rules: str
    tool_factories: tuple[ToolFactory, ...]


#: Today's coding agent, expressed as a profile: the default domain rules
#: and all 13 builtin tool factories. ``run_task(profile=None)`` behaves
#: identically to ``run_task(profile=CODING)``.
CODING = Profile(
    name="coding",
    domain_rules=CODING_RULES,
    tool_factories=CODING_TOOL_FACTORIES,
)

#: Domain rules for the read-only coding profile: same workspace/path
#: conventions, but the agent is told it cannot execute or modify anything.
_CODING_READONLY_RULES = """\
Domain rules (coding, read-only):
- You are inspecting a sandbox workspace (host path: {workspace}) in
  read-only mode. File paths passed to tools are relative to the workspace
  root.
- You have no bash/write_file/edit_file tools: you cannot execute code or
  modify files. Report findings from what you can read, search, and
  recall; never claim to have run or changed anything.""".rstrip()

#: Tool factories for the read-only profile: read_file, the memory tools,
#: the task-ledger tools, and load_skill — no ``bash``/``write_file``/
#: ``edit_file`` (and none of the workspace-mutation paths they carry).
_CODING_READONLY_FACTORIES: tuple[ToolFactory, ...] = (
    lambda deps: read_file_tool(deps.sandbox),
    lambda deps: memory_read_fact_tool(deps.memory),
    lambda deps: memory_write_fact_tool(deps.memory),
    lambda deps: memory_search_tool(deps.memory),
    lambda deps: task_update_tool(deps.store, deps.run_id, deps.context),
    lambda deps: task_list_tool(deps.store, deps.run_id),
    lambda deps: load_skill_tool(deps.skills, deps.context),
)

#: A second real profile proving the seam (§11.7 G4): inspection-only —
#: read_file/memory/task/skill tools, no bash/write/edit.
CODING_READONLY = Profile(
    name="coding-readonly",
    domain_rules=_CODING_READONLY_RULES,
    tool_factories=_CODING_READONLY_FACTORIES,
)

#: Every defined profile, for tests that assert invariants across all of
#: them (e.g. G3: the assembled prompt always carries the core clauses).
ALL_PROFILES: tuple[Profile, ...] = (CODING, CODING_READONLY)
