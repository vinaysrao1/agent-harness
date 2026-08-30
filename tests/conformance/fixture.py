"""The fixed fixture the neutrality goldens are computed over (S-002).

Every golden is a hash of something derived from the ``CODING`` profile under
*these exact inputs*. They are constants, not parameters: a golden computed
over a varying fixture would drift with the fixture and pin nothing.

Nothing here may read the environment, the clock, or the filesystem outside a
caller-supplied ``tmp_path`` -- a golden that depends on where it ran is not a
golden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from harness.context import ContextManager
from harness.memory.store import MemoryStore
from harness.orchestrator import (
    Orchestrator,
    _await_agents_tool,
    _spawn_agent_tool,
)
from harness.permissions import PermissionMode
from harness.reads import FileCache
from harness.secrets import SecretRegistry
from harness.skills import SkillLibrary
from harness.types import Role, ToolSpec

#: Fixed inputs. Chosen to be boring and stable; the literal values matter only
#: in that they must never change without re-freezing the goldens.
FIXTURE_WORKSPACE = "/workspace"
FIXTURE_MODE = PermissionMode.GATED
FIXTURE_RUN_ID = "run-conformance"
FIXTURE_AGENT_ID = "agent-conformance"


#: A fixed skill, so the skills-index branch of the system prompt is pinned.
#: With an empty library that branch never renders and its header text was
#: free to drift.
FIXTURE_SKILL = """---
name: fixture-skill
description: A fixed skill so the skills index renders deterministically.
---

Body text held constant by the conformance fixture.
"""

#: A fixed memory index, so the always-in-context memory block is pinned --
#: including its data-not-instructions delimiters.
FIXTURE_MEMORY_INDEX = "- [Fixture fact](fixture.md) - held constant."

#: The goal a real run seeds into the instruction ledger as its first standing
#: instruction, which is re-rendered into the assembled system string.
FIXTURE_GOAL = "Conformance fixture goal; held constant."

#: A fixed skill body, spliced in so the loaded-skill branch renders.
FIXTURE_SKILL_BODY = "Fixed skill body held constant by the conformance fixture."


def fixture_skills(tmp_path: Path) -> SkillLibrary:
    """A skill library containing exactly :data:`FIXTURE_SKILL`."""
    root = tmp_path / "skills" / "fixture-skill"
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(FIXTURE_SKILL, encoding="utf-8")
    return SkillLibrary(tmp_path / "skills")


def coding_base_prompt(skills: SkillLibrary) -> str:
    """The base ``CODING`` rules string (the narrow half of N1).

    Kept as its own digest so a failure localizes: a change here is a rules
    edit, whereas a change in :func:`coding_assembled_system` alone is a
    context-assembly change.

    Constructed with ``__new__`` because :meth:`Orchestrator._system_prompt`
    reads only its arguments. If it ever starts reading instance state this
    raises rather than silently pinning a different prompt -- though note it
    would not catch a newly-read *class* attribute.
    """
    orchestrator = Orchestrator.__new__(Orchestrator)
    return orchestrator._system_prompt(
        FIXTURE_MODE, FIXTURE_WORKSPACE, skills, domain_rules=None
    )


def coding_assembled_system(skills: SkillLibrary) -> str:
    """The full system string a ``CODING`` turn actually sends (N1).

    This is what the spec means by ``context.assemble()[0]``: base rules plus
    skill bodies, the memory block and the instruction ledger. Hashing only
    the base rules left the entire context-assembly layer unpinned -- a new
    always-in-context section reached every turn of every run and no test
    noticed. Populated the way :meth:`Orchestrator._execute` populates it, so
    the golden tracks the real assembly path.
    """
    context = ContextManager(
        base_system_prompt=coding_base_prompt(skills),
        count_tokens=lambda messages: 0,
        max_context=200_000,
        summarize=_never_summarize,
    )
    context.add_memory_block(FIXTURE_MEMORY_INDEX)
    context.add_instruction(FIXTURE_GOAL, "user")
    # Load the body too: indexing a skill renders the index header, but the
    # "## Loaded skill:" branch only renders once a body is spliced in. Pinning
    # the index while claiming to pin bodies is the same half-covered mistake
    # an empty SkillLibrary made for the index itself.
    context.add_skill_body(FIXTURE_SKILL_BODY, name="fixture-skill")
    return context.assemble()[0]


def coding_trailing_reminder(skills: SkillLibrary) -> str:
    """The trailing system-reminder a turn also sends (N1, second half).

    ``assemble()`` returns ``(system, messages)`` and the reminder rides in
    *messages*, so hashing ``assemble()[0]`` alone left it unpinned -- yet it
    is model-facing text re-injected every ``reminder_interval`` turns and
    after every compaction. Rewriting its preamble survived the whole suite.
    """
    context = ContextManager(
        base_system_prompt=coding_base_prompt(skills),
        count_tokens=lambda messages: 0,
        max_context=200_000,
        summarize=_never_summarize,
    )
    context.add_memory_block(FIXTURE_MEMORY_INDEX)
    context.add_instruction(FIXTURE_GOAL, "user")
    # No public API forces a reminder; the flag is what a post-compaction
    # assembly sets. Setting it directly keeps the fixture deterministic --
    # driving a real compaction would pull the summarizer into the golden.
    context._reminder_due = True
    _system, messages = context.assemble()
    reminders = [m.content or "" for m in messages if m.role is Role.USER]
    if not reminders:
        raise AssertionError(
            "the fixture failed to produce a trailing reminder; the golden "
            "would pin an empty string"
        )
    return reminders[-1]


async def _never_summarize(messages: list[Any]) -> str:  # pragma: no cover
    raise AssertionError("the conformance fixture never compacts")


def coding_tool_specs(
    tmp_path: Path, store: Any, *, lead: bool
) -> list[ToolSpec]:
    """Tool specs for the ``CODING`` surface, in registration order (N2).

    Driven through the real :meth:`Orchestrator._build_registry` rather than
    by iterating :data:`CODING_TOOL_FACTORIES`. Iterating the tuple pinned the
    tuple, not the surface: a tool registered directly in ``_build_registry``
    reached the model on every run and the invariant never saw it.

    The two lead-only tools are still re-assembled here rather than driven
    through :meth:`Orchestrator._execute`, so a *third* tool registered beside
    them would not move this digest. That site is pinned separately, by an AST
    count over ``_execute`` -- see ``test_S002_n2_lead_registration_site``.

    ``lead=False`` is the subagent surface; ``lead=True`` adds the two
    lead-only tools the way :meth:`Orchestrator._execute` does, and is what a
    Terminal-Bench run presents to the model.
    """
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.store = store
    # The benchmark path has no secrets to redact, and an empty registry is
    # the identity function -- which is exactly the state N2 must pin. Set
    # explicitly rather than defaulted inside `_build_registry`, so an
    # Orchestrator built without one stays a construction bug rather than
    # becoming a silent no-op.
    orchestrator.secrets = SecretRegistry()
    # One cache per run (S-102). Set explicitly for the same reason as
    # `secrets` above: an Orchestrator built without one stays a construction
    # bug rather than becoming a silent no-op.
    orchestrator._file_cache = FileCache()
    skills = fixture_skills(tmp_path)
    registry = orchestrator._build_registry(
        sandbox=_InertSandbox(),
        memory=MemoryStore(tmp_path / "memory"),
        skills=skills,
        run_id=FIXTURE_RUN_ID,
        agent_id=FIXTURE_AGENT_ID,
        context=ContextManager(
            base_system_prompt=coding_base_prompt(skills),
            count_tokens=lambda messages: 0,
            max_context=200_000,
            summarize=_never_summarize,
        ),
    )
    if lead:

        async def _unused(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("conformance fixture never dispatches")

        registry.register(_spawn_agent_tool(_unused))
        registry.register(_await_agents_tool(_unused))
    return list(registry.specs())


class _InertSandbox:
    """A stand-in sandbox. Factories only close over it; nothing dispatches."""

    workspace = Path("/workspace")

    async def exec(self, command: str, timeout: float = 120) -> Any:  # pragma: no cover
        raise AssertionError("conformance fixture never executes")


def canonical_tool_surface(specs: list[ToolSpec]) -> str:
    """Serialize the tool surface canonically: names, order, and schemas.

    ``sort_keys`` inside each schema so a dict-ordering change is not mistaken
    for a behavioral one, but the *list* order is preserved -- tool order is
    part of what the model sees and reordering it is a real change.
    """
    payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in specs
    ]
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def sha256_text(text: str) -> str:
    """Hex digest of ``text`` as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_hash(root: Path) -> str:
    """Order-independent hash of a directory tree's paths and contents (N4).

    Files are hashed by relative path and bytes; directory mtimes and empty
    directories are ignored, because neither is something a run is forbidden
    to touch.
    """
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return sha256_text("\n".join(entries))


def coding_control_flow_digest_input() -> str:
    """What N5's golden hashes: declared gates plus the tuned budgets.

    The declared tuples alone left the *values* unpinned -- raising
    ``MAX_NUDGES`` from 2 to 5 is a real change to how the CODING path
    terminates, and it survived every test in the repository. A control-flow
    invariant that ignores the budgets governing that control flow pins its
    shape and not its behavior.
    """
    from harness.diligence import MAX_NUDGES
    from harness.loop import COMPLETION_GATES, MAX_TRUNCATION_CONTINUES, NUDGE_SOURCES

    return "\n".join(
        [
            *COMPLETION_GATES,
            "--",
            *NUDGE_SOURCES,
            "--",
            f"MAX_NUDGES={MAX_NUDGES}",
            f"MAX_TRUNCATION_CONTINUES={MAX_TRUNCATION_CONTINUES}",
        ]
    )
