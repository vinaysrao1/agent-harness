"""Skill discovery and loading (DESIGN.md §4.6).

A skill is a directory containing a ``SKILL.md`` with frontmatter
(``name``, one-line ``description``) plus a markdown body of instructions,
and optionally scripts/resources alongside it. ``SkillLibrary`` discovers
every ``<root>/*/SKILL.md`` up front so only the name+description lines
need to sit in the system prompt (progressive disclosure); the full body is
spliced into context on demand via :meth:`SkillLibrary.load`.

Frontmatter uses the same tiny ``---``-fenced ``key: value`` format as
:mod:`harness.memory.store` (see that module's docstring for the exact
grammar) — duplicated here in miniature rather than imported, since skills
and memory are independent subsystems that happen to share a convention.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Skill", "SkillLibrary"]


@dataclass(frozen=True)
class Skill:
    """One discovered skill: its metadata plus full instruction body."""

    name: str
    description: str
    path: Path
    body: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse ``---``-fenced ``key: value`` frontmatter, without pyyaml.

    Each non-blank line between the fences must contain a colon; only the
    *first* colon splits key from value, so colons inside a description
    survive intact. Returns ``(fields, body)``, ``body`` being everything
    after the closing fence with at most one leading blank line stripped.
    Raises :class:`ValueError` if the fences are missing or a line has no
    colon.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening '---' frontmatter fence")
    fields: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        if ":" not in line:
            raise ValueError(f"malformed frontmatter line (no ':'): {line!r}")
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
        i += 1
    if i >= len(lines):
        raise ValueError("missing closing '---' frontmatter fence")
    body = "\n".join(lines[i + 1 :])
    if body.startswith("\n"):
        body = body[1:]
    return fields, body


class SkillLibrary:
    """Discovers and serves skills from ``<root>/*/SKILL.md``.

    Discovery happens once at construction and again on demand via
    :meth:`reload`. A malformed ``SKILL.md`` (bad frontmatter fences, or
    missing ``name``/``description``) is skipped with a
    :class:`UserWarning` naming the file — discovery never raises because
    of one bad skill.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        """Re-scan ``root`` for ``*/SKILL.md`` files, replacing the index."""
        skills: dict[str, Skill] = {}
        if self.root.is_dir():
            for skill_file in sorted(self.root.glob("*/SKILL.md")):
                try:
                    skill = self._parse_skill(skill_file)
                except ValueError as exc:
                    warnings.warn(
                        f"skipping malformed skill file {skill_file}: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                skills[skill.name] = skill
        self._skills = skills

    def _parse_skill(self, path: Path) -> Skill:
        fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = fields.get("name")
        if not name:
            raise ValueError("missing required frontmatter field 'name'")
        description = fields.get("description")
        if not description:
            raise ValueError("missing required frontmatter field 'description'")
        return Skill(name=name, description=description, path=path, body=body.strip("\n"))

    def skills(self) -> list[Skill]:
        """All discovered skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def index_lines(self) -> list[str]:
        """One ``- name: description`` line per skill, for the system prompt."""
        return [f"- {skill.name}: {skill.description}" for skill in self.skills()]

    def load(self, name: str) -> str:
        """Return the full instruction body of the skill named ``name``.

        Raises :class:`KeyError` listing the available skill names if
        ``name`` was not discovered.
        """
        try:
            return self._skills[name].body
        except KeyError:
            available = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(
                f"unknown skill {name!r}; available skills: {available}"
            ) from None
