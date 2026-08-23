"""S-002: the neutrality invariants (N1-N5).

The governing constraint of the M10 plan is that every change leaves the
Terminal-Bench execution path *provably* unchanged. TB2 is too slow, too
expensive and too noisy to be a CI gate, so these convert "don't regress TB2"
from an intention into tests that run in under a second.

Each invariant has a **negative test** driving it with a deliberate violation.
That is the deliverable, not a nicety: an invariant never seen to fail is not
known to work -- the lesson S-001 learned when its own T3 check turned out to
pass by not looking.

N6-N8 are defined against the replay corpus and land with S-003.

A failing invariant is not necessarily a bug. It means the change touches the
benchmark path, which makes it Lane B: declare a REFREEZE block naming the
invariant, run TB2, regenerate the golden, and record the run id in
tests/golden/CHANGELOG.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from harness.loop import COMPLETION_GATES, NUDGE_SOURCES
from harness.persistence import RunStore
from harness.skills import SkillLibrary
from harness.types import ToolSpec
from tests.conformance.fixture import (
    canonical_tool_surface,
    coding_assembled_system,
    coding_base_prompt,
    coding_control_flow_digest_input,
    coding_trailing_reminder,
    coding_tool_specs,
    fixture_skills,
    sha256_text,
)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
LOOP_SOURCE = ROOT / "harness" / "loop.py"
ORCHESTRATOR_SOURCE = ROOT / "harness" / "orchestrator.py"

#: The plan caps the CODING surface at 15 specs (13 today + 2 lead-only).
#: Tool-surface growth degrades selection quality, measurably so for
#: non-Anthropic models.
MAX_CODING_TOOLS = 15


def _golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8").strip()


def _mentions_completed(node: ast.Call) -> bool:
    """Whether a ``_finish`` call passes ``"completed"``, however spelled.

    ``status`` is positional-or-keyword, so reading ``args[0]`` alone let a
    second completion path written as ``_finish(status="completed", ...)``
    slip past -- the same positional-only blind spot S-001's T3 scan had.
    """
    for arg in node.args:
        if isinstance(arg, ast.Constant) and arg.value == "completed":
            return True
    for keyword in node.keywords:
        value = keyword.value
        if isinstance(value, ast.Constant) and value.value == "completed":
            return True
    return False


def _refreeze_hint(invariant: str, actual: str) -> str:
    return (
        f"{invariant} golden mismatch. If this change is deliberate it is Lane B: "
        f"add a REFREEZE block to the spec, run TB2, write {actual} into the "
        f"golden, and add a CHANGELOG.md entry naming the run id."
    )


@pytest.fixture
def skills(tmp_path: Path) -> SkillLibrary:
    """A library holding one fixed skill, so the index branch renders."""
    return fixture_skills(tmp_path)


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "state.db")


class TestN1PromptNeutrality:
    """The assembled CODING system prompt is byte-identical to the golden."""

    def test_S002_n1_assembled_system_matches_golden(
        self, skills: SkillLibrary
    ) -> None:
        # The string a real turn sends: base rules + skill bodies + memory
        # block + instruction ledger. Hashing only the base rules left the
        # whole context-assembly layer unpinned.
        actual = sha256_text(coding_assembled_system(skills))
        assert actual == _golden("coding_assembled_system.sha256"), _refreeze_hint(
            "N1 (assembled system)", actual
        )

    def test_S002_n1_base_prompt_matches_golden(self, skills: SkillLibrary) -> None:
        # Narrower second digest so a failure localizes to a rules edit rather
        # than to context assembly.
        actual = sha256_text(coding_base_prompt(skills))
        assert actual == _golden("coding_base_prompt.sha256"), _refreeze_hint(
            "N1 (base prompt)", actual
        )

    def test_S002_n1_detects_an_injected_prompt_line(
        self, skills: SkillLibrary
    ) -> None:
        tampered = coding_assembled_system(skills) + "\nAlways prefer ripgrep."
        assert sha256_text(tampered) != _golden("coding_assembled_system.sha256")

    def test_S002_n1_covers_the_memory_and_ledger_sections(
        self, skills: SkillLibrary
    ) -> None:
        # Guards against the fixture silently ceasing to render the layers the
        # golden is supposed to pin -- the way an empty SkillLibrary silently
        # left the skills-index header unpinned.
        assembled = coding_assembled_system(skills)
        assert "fixture-skill" in assembled, "skills index not rendered"
        assert "Fixture fact" in assembled, "memory block not rendered"
        assert "Conformance fixture goal" in assembled, "instruction ledger not rendered"

    def test_S002_n1_trailing_reminder_matches_golden(
        self, skills: SkillLibrary
    ) -> None:
        # assemble() returns (system, messages); the reminder rides in
        # messages, so hashing assemble()[0] alone left model-facing text
        # unpinned.
        actual = sha256_text(coding_trailing_reminder(skills))
        assert actual == _golden("coding_trailing_reminder.sha256"), _refreeze_hint(
            "N1 (trailing reminder)", actual
        )

    def test_S002_n1_covers_skill_bodies(self, skills: SkillLibrary) -> None:
        # Guards the loaded-skill branch, which only renders once a body is
        # spliced in. Indexing a skill is not the same as loading one.
        from tests.conformance.fixture import FIXTURE_SKILL_BODY

        assert FIXTURE_SKILL_BODY in coding_assembled_system(skills)

    def test_S002_n1_assembled_is_a_superset_of_the_base(
        self, skills: SkillLibrary
    ) -> None:
        assert len(coding_assembled_system(skills)) > len(coding_base_prompt(skills))


class TestN2ToolSurfaceNeutrality:
    """Tool names, order, and JSON schemas are byte-identical to the golden."""

    def test_S002_n2_subagent_surface_matches_golden(self, tmp_path: Path, store) -> None:
        actual = sha256_text(
            canonical_tool_surface(coding_tool_specs(tmp_path, store, lead=False))
        )
        assert actual == _golden("coding_tools_subagent.sha256"), _refreeze_hint(
            "N2 (subagent surface)", actual
        )

    def test_S002_n2_lead_surface_matches_golden(self, tmp_path: Path, store) -> None:
        actual = sha256_text(
            canonical_tool_surface(coding_tool_specs(tmp_path, store, lead=True))
        )
        assert actual == _golden("coding_tools_lead.sha256"), _refreeze_hint(
            "N2 (lead surface)", actual
        )

    def test_S002_n2_detects_an_added_tool(self, tmp_path: Path, store) -> None:
        # Negative: the reviewer's canonical mutation. An extra tool changes
        # what the model chooses between and must never pass silently.
        specs = coding_tool_specs(tmp_path, store, lead=True)
        specs.append(
            ToolSpec(
                name="ripgrep",
                description="Search the workspace.",
                input_schema={"type": "object", "properties": {}},
            )
        )
        assert sha256_text(canonical_tool_surface(specs)) != _golden(
            "coding_tools_lead.sha256"
        )

    def test_S002_n2_detects_a_description_edit(self, tmp_path: Path, store) -> None:
        # A description edit moves model behavior as surely as a new tool, and
        # is far easier to make by accident.
        specs = coding_tool_specs(tmp_path, store, lead=True)
        first = specs[0]
        specs[0] = ToolSpec(
            name=first.name,
            description=first.description + " Prefer this tool.",
            input_schema=first.input_schema,
        )
        assert sha256_text(canonical_tool_surface(specs)) != _golden(
            "coding_tools_lead.sha256"
        )

    def test_S002_n2_detects_reordering(self, tmp_path: Path, store) -> None:
        # Order is part of what the model sees; canonicalising it away would
        # make this invariant weaker than the thing it claims to pin.
        specs = coding_tool_specs(tmp_path, store, lead=True)
        swapped = [specs[1], specs[0], *specs[2:]]
        assert sha256_text(canonical_tool_surface(swapped)) != _golden(
            "coding_tools_lead.sha256"
        )

    def test_S002_n2_detects_a_schema_change(self, tmp_path: Path, store) -> None:
        specs = coding_tool_specs(tmp_path, store, lead=True)
        first = specs[0]
        schema = dict(first.input_schema)
        schema["properties"] = {**schema.get("properties", {}), "extra": {"type": "string"}}
        specs[0] = ToolSpec(
            name=first.name, description=first.description, input_schema=schema
        )
        assert sha256_text(canonical_tool_surface(specs)) != _golden(
            "coding_tools_lead.sha256"
        )

    def test_S002_n2_lead_registration_site(self) -> None:
        """Exactly two tools are registered on the lead beyond the shared set.

        The lead surface digest re-assembles spawn/await rather than driving
        ``_execute``, so a *third* tool registered beside them would not move
        it -- the tuple-versus-real-path defect, one level up. This AST count
        closes that site. If the lead genuinely gains a tool, this fails until
        the count and the golden are both updated deliberately.
        """
        tree = ast.parse(ORCHESTRATOR_SOURCE.read_text(encoding="utf-8"))
        registrations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "lead_registry"
        ]
        assert len(registrations) == 2, (
            f"_execute registers {len(registrations)} lead-only tools; the "
            "conformance fixture assumes exactly two (spawn_agent, "
            "await_agents). A third reaches the model unpinned."
        )

    def test_S002_n2_lead_registration_check_would_notice_a_third(self) -> None:
        # Negative: the count must actually move when a registration is added.
        source = ORCHESTRATOR_SOURCE.read_text(encoding="utf-8").replace(
            "lead_registry.register(_await_agents_tool(await_handler))",
            "lead_registry.register(_await_agents_tool(await_handler))\n"
            "        lead_registry.register(_await_agents_tool(await_handler))",
            1,
        )
        tree = ast.parse(source)
        registrations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "lead_registry"
        ]
        assert len(registrations) == 3

    def test_S002_n2_surface_is_within_the_tool_cap(self, tmp_path: Path, store) -> None:
        specs = coding_tool_specs(tmp_path, store, lead=True)
        assert len(specs) <= MAX_CODING_TOOLS, (
            f"CODING is capped at {MAX_CODING_TOOLS} tool specs; promoting a "
            "tool into CODING must remove or merge an existing one."
        )
        assert len({s.name for s in specs}) == len(specs), "duplicate tool name"


class TestN5ControlFlowNeutrality:
    """The CODING path has exactly the completion gates and nudge sources it
    has today.

    Two halves. The declared tuples are pinned against a golden, so a
    deliberate change forces a re-freeze. The structural counts are derived
    from loop.py's AST, so a gate added *without* declaring itself fails too --
    a declaration nobody is obliged to update is documentation, not a check.
    """

    def _loop_tree(self) -> ast.Module:
        return ast.parse(LOOP_SOURCE.read_text(encoding="utf-8"))

    def test_S002_n5_declared_gates_match_golden(self) -> None:
        actual = sha256_text(coding_control_flow_digest_input())
        assert actual == _golden("coding_control_flow.sha256"), _refreeze_hint(
            "N5", actual
        )

    def test_S002_n5_nudge_sites_match_declared_sources(self) -> None:
        # Every `nudges += 1` is one gate consuming the shared budget.
        sites = [
            node
            for node in ast.walk(self._loop_tree())
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "nudges"
        ]
        assert len(sites) == len(NUDGE_SOURCES), (
            f"loop.py has {len(sites)} nudge sites but NUDGE_SOURCES declares "
            f"{len(NUDGE_SOURCES)}. A new nudge source must declare itself."
        )

    def test_S002_n5_single_completed_return(self) -> None:
        calls = [
            node
            for node in ast.walk(self._loop_tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_finish"
            and _mentions_completed(node)
        ]
        assert len(calls) == 1, (
            f"loop.py has {len(calls)} `_finish(\"completed\")` returns; a new "
            "completion path is a change to the benchmark path."
        )

    def test_S002_n5_detects_an_added_gate(self, tmp_path: Path) -> None:
        # Negative: a third gate that nudges must trip the structural half even
        # though the declared tuples were left untouched.
        source = LOOP_SOURCE.read_text(encoding="utf-8").replace(
            "                nudges += 1", "                nudges += 1\n                nudges += 1", 1
        )
        tree = ast.parse(source)
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "nudges"
        ]
        assert len(sites) != len(NUDGE_SOURCES)

    def test_S002_n5_detects_a_keyword_completion_path(self) -> None:
        # Negative: a second completion return written with keyword arguments
        # must still be counted. `status` is positional-or-keyword.
        source = LOOP_SOURCE.read_text(encoding="utf-8").replace(
            'return self._finish("completed", final_text, total_usage, turns)',
            'return self._finish(status="completed", final_text=final_text, '
            "usage=total_usage, turns=turns)",
            1,
        )
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_finish"
            and _mentions_completed(node)
        ]
        assert len(calls) == 1, "the keyword form must still be counted"

    def test_S002_n5_gates_and_sources_are_consistent(self) -> None:
        # Today every gate is also a nudge source. If that stops being true the
        # golden must be re-frozen deliberately rather than drifting.
        assert set(NUDGE_SOURCES) <= set(COMPLETION_GATES)
