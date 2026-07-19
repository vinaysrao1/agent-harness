"""Semantic + episodic memory store (DESIGN.md §4.4).

``MemoryStore`` persists two of the harness's three memory shapes as plain
markdown files under a root directory:

- **Facts** (semantic — "what's true"): one file per fact in ``facts/*.md``,
  frontmatter ``name``/``description``/``type``/``sources``. Superseded facts
  are moved to ``facts/archive/`` (§4.4.2 write policy) — never deleted.
- **Episodes** (episodic journal layer — "what happened"): one
  date-prefixed file per session in ``episodes/*.md``, frontmatter
  ``slug``/``date``/``title``/``outcome``.
- **INDEX.md**: a regenerated-on-every-mutation summary (facts + last 10
  episodes) meant to always be in context (§4.4.2).

Frontmatter is a deliberately tiny YAML-ish format — ``key: value`` lines
between ``---`` fences, comma-separated lists — parsed without a YAML
dependency (``_parse_frontmatter`` below). It is forgiving of extra
whitespace and colons inside values (only the *first* colon on a line is
the key/value separator) but requires the fences and a ``key: value`` shape
per line; malformed fact/episode files raise :class:`MemoryStoreError` with
the offending path so the failure is diagnosable.

**Concurrency:** this class assumes a single process/thread owns the memory
root at a time. Reads and writes are plain filesystem I/O with no locking,
so concurrent writers (two processes, or interleaved async tasks writing the
same fact) can race and clobber each other. Out of scope for v1 per the
module spec — callers that need concurrent access must serialize it
themselves (e.g. a single asyncio task owns the store).
"""

from __future__ import annotations

import datetime as _dt
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MemoryStoreError",
    "FactNotFoundError",
    "EpisodeNotFoundError",
    "FactType",
    "Fact",
    "FactIndexEntry",
    "Episode",
    "EpisodeIndexEntry",
    "MemoryStore",
]

#: kebab-case: lowercase letters/digits, hyphen-separated, no leading/
#: trailing/doubled hyphens.
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MemoryStoreError(Exception):
    """Base error for :class:`MemoryStore` failures (bad slugs, malformed
    frontmatter, unknown fact types, ...)."""


class FactNotFoundError(MemoryStoreError):
    """Raised when :meth:`MemoryStore.read_fact` or
    :meth:`MemoryStore.archive_fact` names an unknown fact."""


class EpisodeNotFoundError(MemoryStoreError):
    """Raised when :meth:`MemoryStore.read_episode` names an unknown
    episode filename."""


class FactType(str, Enum):
    """Semantic-fact category (DESIGN.md §4.4.2)."""

    USER = "user"
    PROJECT = "project"
    FEEDBACK = "feedback"
    REFERENCE = "reference"


class Fact(BaseModel):
    """A fully-loaded semantic fact, including its body."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    type: FactType
    sources: list[str] = Field(default_factory=list)
    body: str


class FactIndexEntry(BaseModel):
    """Lightweight fact summary as returned by :meth:`MemoryStore.list_facts`
    (no body — mirrors what actually goes in ``INDEX.md``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    type: FactType
    sources: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    """A fully-loaded episode journal entry, including its body."""

    model_config = ConfigDict(frozen=True)

    filename: str
    slug: str
    date: str
    title: str
    outcome: str
    body: str


class EpisodeIndexEntry(BaseModel):
    """Lightweight episode summary as returned by
    :meth:`MemoryStore.list_episodes`."""

    model_config = ConfigDict(frozen=True)

    filename: str
    slug: str
    date: str
    title: str
    outcome: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse ``---``-fenced ``key: value`` frontmatter, without pyyaml.

    Deliberately tiny: each non-blank line between the fences must contain
    a colon; only the *first* colon splits key from value (so colons inside
    a description survive intact). Returns ``(fields, body)`` where ``body``
    is everything after the closing fence, with at most one leading blank
    line stripped. Raises :class:`ValueError` (caller wraps with file
    context) if the fences are missing or a line has no colon.
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


def _render_frontmatter(fields: dict[str, str]) -> str:
    """Render ``fields`` back into ``---``-fenced ``key: value`` lines."""
    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in fields.items())
    lines.append("---")
    return "\n".join(lines)


def _parse_list(value: str) -> list[str]:
    """Split a comma-separated frontmatter value into a clean string list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_value(value: str, *, field: str, kind: str) -> str:
    """Validate (and normalize) a frontmatter *value* before it is written.

    The write side must never produce a file the strict read-side parser
    rejects or misreads — otherwise a bad value poisons the on-disk store
    and every subsequent read/mutation fails. So values must be non-empty,
    single-line strings; surrounding whitespace is stripped (the parser
    strips it on read anyway, guaranteeing an exact round-trip). Returns
    the normalized value; raises :class:`MemoryStoreError` otherwise.
    """
    if "\n" in value or "\r" in value:
        raise MemoryStoreError(
            f"{kind} {field} must be a single line (got a newline in {value!r})"
        )
    normalized = value.strip()
    if not normalized:
        raise MemoryStoreError(f"{kind} {field} must be non-empty")
    return normalized


def _validate_sources(sources: list[str], *, kind: str) -> list[str]:
    """Validate/normalize a ``sources`` list before writing.

    Items are frontmatter-encoded comma-separated, so each item must be a
    non-empty single-line string with no comma. Returns the normalized list.
    """
    validated = []
    for item in sources:
        item = _validate_value(item, field="sources item", kind=kind)
        if "," in item:
            raise MemoryStoreError(
                f"{kind} sources item {item!r} must not contain a comma "
                "(sources are stored comma-separated)"
            )
        validated.append(item)
    return validated


def _validate_slug(value: str, *, kind: str) -> None:
    """Raise :class:`MemoryStoreError` unless ``value`` is a kebab-case slug."""
    if not _SLUG_RE.match(value):
        raise MemoryStoreError(
            f"{kind} name {value!r} must be a kebab-case slug "
            "(lowercase letters, digits, single hyphens; no leading/"
            "trailing/double hyphens)"
        )


class MemoryStore:
    """File-backed episodic + semantic memory store rooted at ``root``.

    ``root`` is created (along with ``facts/``, ``facts/archive/``, and
    ``episodes/``) if it does not already exist. See the module docstring
    for the on-disk layout, frontmatter format, and concurrency caveats.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._facts_dir = self.root / "facts"
        self._archive_dir = self._facts_dir / "archive"
        self._episodes_dir = self.root / "episodes"
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.is_file():
            self.rebuild_index()

    @property
    def _index_path(self) -> Path:
        return self.root / "INDEX.md"

    def _fact_path(self, name: str) -> Path:
        return self._facts_dir / f"{name}.md"

    def _episode_path(self, filename: str) -> Path:
        return self._episodes_dir / filename

    # -- facts ------------------------------------------------------------

    def write_fact(
        self,
        name: str,
        description: str,
        type: FactType | str,
        body: str,
        sources: list[str] | None = None,
    ) -> Fact:
        """Write (or overwrite) the fact named ``name``.

        A second call with the same ``name`` overwrites the previous file —
        this is the update path (§4.4.2 write policy: check the index for
        an existing entry before creating a new one). Rebuilds ``INDEX.md``
        before returning.
        """
        _validate_slug(name, kind="fact")
        try:
            fact_type = FactType(type)
        except ValueError:
            valid = ", ".join(t.value for t in FactType)
            raise MemoryStoreError(
                f"invalid fact type {type!r}; must be one of: {valid}"
            ) from None
        fact = Fact(
            name=name,
            description=_validate_value(description, field="description", kind="fact"),
            type=fact_type,
            sources=_validate_sources(list(sources), kind="fact") if sources else [],
            body=body,
        )
        self._fact_path(name).write_text(self._render_fact(fact), encoding="utf-8")
        self.rebuild_index()
        return fact

    def _render_fact(self, fact: Fact) -> str:
        fields = {
            "name": fact.name,
            "description": fact.description,
            "type": fact.type.value,
            "sources": ", ".join(fact.sources),
        }
        return _render_frontmatter(fields) + "\n\n" + fact.body.strip("\n") + "\n"

    def _parse_fact_file(self, path: Path) -> Fact:
        try:
            fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MemoryStoreError(f"malformed fact file {path}: {exc}") from None
        for required in ("name", "description", "type"):
            if required not in fields or not fields[required]:
                raise MemoryStoreError(
                    f"fact file {path} is missing required field {required!r}"
                )
        try:
            fact_type = FactType(fields["type"])
        except ValueError:
            valid = ", ".join(t.value for t in FactType)
            raise MemoryStoreError(
                f"fact file {path} has invalid type {fields['type']!r}; "
                f"must be one of: {valid}"
            ) from None
        return Fact(
            name=fields["name"],
            description=fields["description"],
            type=fact_type,
            sources=_parse_list(fields.get("sources", "")),
            body=body.rstrip("\n"),
        )

    def read_fact(self, name: str) -> Fact:
        """Read the active (non-archived) fact named ``name``.

        Raises :class:`FactNotFoundError` if no such fact exists, and
        :class:`MemoryStoreError` if the file exists but its frontmatter is
        malformed or missing required fields.
        """
        path = self._fact_path(name)
        if not path.is_file():
            raise FactNotFoundError(f"no fact named {name!r} (expected {path})")
        return self._parse_fact_file(path)

    def list_facts(self) -> list[FactIndexEntry]:
        """List all active (non-archived) facts, sorted by name."""
        entries = []
        for path in sorted(self._facts_dir.glob("*.md")):
            fact = self._parse_fact_file(path)
            entries.append(
                FactIndexEntry(
                    name=fact.name,
                    description=fact.description,
                    type=fact.type,
                    sources=fact.sources,
                )
            )
        return entries

    def archive_fact(self, name: str, superseded_by: str | None = None) -> None:
        """Move the fact named ``name`` to ``facts/archive/`` — never deletes.

        If ``superseded_by`` is given, it is recorded as an extra
        ``superseded_by`` frontmatter field on the archived copy (the
        contradiction-history trail from §4.4.2). Rebuilds ``INDEX.md``
        (the archived fact drops out of it) before returning.
        """
        fact = self.read_fact(name)  # raises FactNotFoundError if missing
        fields = {
            "name": fact.name,
            "description": fact.description,
            "type": fact.type.value,
            "sources": ", ".join(fact.sources),
        }
        if superseded_by:
            fields["superseded_by"] = _validate_value(
                superseded_by, field="superseded_by", kind="fact"
            )
        archived_text = _render_frontmatter(fields) + "\n\n" + fact.body.strip("\n") + "\n"
        (self._archive_dir / f"{name}.md").write_text(archived_text, encoding="utf-8")
        self._fact_path(name).unlink()
        self.rebuild_index()

    # -- episodes -----------------------------------------------------------

    def write_episode(
        self,
        slug: str,
        title: str,
        outcome: str,
        body: str,
        date: str | None = None,
    ) -> str:
        """Write a new episode journal entry; returns its filename.

        The filename is ``YYYY-MM-DD-<slug>.md``; ``date`` defaults to
        today (UTC) but is injectable so tests can control ordering.
        Rebuilds ``INDEX.md`` before returning.
        """
        _validate_slug(slug, kind="episode")
        resolved_date = date if date is not None else _dt.date.today().isoformat()
        if not _DATE_RE.match(resolved_date):
            raise MemoryStoreError(
                f"episode date {resolved_date!r} must be YYYY-MM-DD"
            )
        filename = f"{resolved_date}-{slug}.md"
        fields = {
            "slug": slug,
            "date": resolved_date,
            "title": _validate_value(title, field="title", kind="episode"),
            "outcome": _validate_value(outcome, field="outcome", kind="episode"),
        }
        text = _render_frontmatter(fields) + "\n\n" + body.strip("\n") + "\n"
        self._episode_path(filename).write_text(text, encoding="utf-8")
        self.rebuild_index()
        return filename

    def _parse_episode_file(self, path: Path) -> Episode:
        try:
            fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MemoryStoreError(f"malformed episode file {path}: {exc}") from None
        for required in ("slug", "date", "title", "outcome"):
            if required not in fields or not fields[required]:
                raise MemoryStoreError(
                    f"episode file {path} is missing required field {required!r}"
                )
        return Episode(
            filename=path.name,
            slug=fields["slug"],
            date=fields["date"],
            title=fields["title"],
            outcome=fields["outcome"],
            body=body.rstrip("\n"),
        )

    def read_episode(self, filename: str) -> Episode:
        """Read one episode by its filename (as returned by
        :meth:`write_episode`).

        Raises :class:`EpisodeNotFoundError` if no such file exists, and
        :class:`MemoryStoreError` if ``filename`` is not a bare filename
        (episode filenames are model-facing, so path separators / ``..``
        components are rejected to keep reads inside the episodes dir).
        """
        if (
            not filename
            or filename in (".", "..")
            or filename != Path(filename).name
            or "\\" in filename
        ):
            raise MemoryStoreError(
                f"episode filename {filename!r} must be a bare filename "
                "(no path separators or '..' components)"
            )
        path = self._episode_path(filename)
        if not path.is_file():
            raise EpisodeNotFoundError(
                f"no episode file named {filename!r} (expected {path})"
            )
        return self._parse_episode_file(path)

    def list_episodes(self) -> list[EpisodeIndexEntry]:
        """List all episodes, most recent first.

        Filenames are date-prefixed, so a reverse lexicographic sort on
        filename is a reverse chronological sort (ties broken by slug).
        """
        entries = []
        for path in sorted(self._episodes_dir.glob("*.md"), reverse=True):
            episode = self._parse_episode_file(path)
            entries.append(
                EpisodeIndexEntry(
                    filename=episode.filename,
                    slug=episode.slug,
                    date=episode.date,
                    title=episode.title,
                    outcome=episode.outcome,
                )
            )
        return entries

    # -- index & search -----------------------------------------------------

    def rebuild_index(self) -> None:
        """Regenerate ``INDEX.md`` from what's currently on disk.

        Two sections: all active facts, and the 10 most recent episodes —
        one line each, ``- [name] description`` for facts and
        ``- [filename] title (outcome)`` for episodes. Called automatically
        by every mutating method; safe (if redundant) to call directly.
        """
        facts = self.list_facts()
        episodes = self.list_episodes()[:10]
        lines = [
            "# Memory Index",
            "",
            "*Auto-generated by MemoryStore.rebuild_index — do not edit by hand.*",
            "",
            "## Facts",
            "",
        ]
        if facts:
            lines.extend(f"- [{f.name}] {f.description}" for f in facts)
        else:
            lines.append("*(none yet)*")
        lines.extend(["", "## Recent episodes", ""])
        if episodes:
            lines.extend(
                f"- [{e.filename}] {e.title} ({e.outcome})" for e in episodes
            )
        else:
            lines.append("*(none yet)*")
        lines.append("")
        self._index_path.write_text("\n".join(lines), encoding="utf-8")

    def search(self, query: str) -> list[tuple[str, str, str]]:
        """Naive case-insensitive substring search over fact/episode bodies.

        Returns ``(kind, name, line)`` tuples — ``kind`` is ``"fact"`` or
        ``"episode"``; ``name`` is the fact name or episode filename; ``line``
        is the (stripped) matching line, one tuple per matching line. Facts
        are searched in name order, then episodes most-recent-first;
        archived facts are not searched. Empty/whitespace-only queries
        return no results.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return []
        results: list[tuple[str, str, str]] = []
        for entry in self.list_facts():
            fact = self.read_fact(entry.name)
            for line in fact.body.splitlines():
                if query_lower in line.lower():
                    results.append(("fact", fact.name, line.strip()))
        for entry in self.list_episodes():
            episode = self.read_episode(entry.filename)
            for line in episode.body.splitlines():
                if query_lower in line.lower():
                    results.append(("episode", episode.filename, line.strip()))
        return results
