"""What the agent has read, and whether it is still true (S-102).

Two pieces of state with **different lifetimes**, and conflating them was the
first version's central error:

**A cache, per run.** It is a belief about the *filesystem*, and the filesystem
is shared. A subagent runs in the lead's sandbox by default, so a per-agent
cache let the child rewrite a file while the parent kept serving the old bytes
until it happened to run a shell command.

**A version ledger, per agent.** It is a record of what *this agent* has seen,
so an edit can say when it is operating on a file the agent never looked at, or
looked at before something changed it. Sharing it would let one agent's read
silence another's warning.

The ledger only ever warns, and only where a profile asks for it. Rejecting a
stale edit would add a failure mode under a wall clock, and the harness's belief
about staleness is itself approximate — a `bash` command can rewrite anything
without the harness learning which paths.
"""

from __future__ import annotations

import hashlib
import posixpath
from collections import OrderedDict
from dataclasses import dataclass, field

__all__ = [
    "FileCache",
    "ReadLedger",
    "StaleRead",
    "MAX_CACHED_FILES",
    "MAX_TRACKED_VERSIONS",
    "normalise_path",
    "version_of",
]

#: Entries kept before the least recently used is dropped. Bounded because a
#: run that reads ten thousand files should not also hold them: this is a
#: latency optimisation, not a store.
MAX_CACHED_FILES = 64

#: Versions kept. Separate from the cache bound because a version is 16 bytes
#: and the content is not — but it is still bounded, because a codegen run that
#: writes thousands of files would otherwise grow a dict for the whole run.
MAX_TRACKED_VERSIONS = 512


def normalise_path(path: str) -> str:
    """One key per file, whatever the caller called it.

    ``a.py``, ``./a.py`` and ``pkg/./mod.py`` all reach the same file; keyed
    verbatim they were three cache entries, so writing through one spelling
    left the others serving content that was no longer on disk. This does not
    resolve symlinks — that needs a sandbox call, which is what the cache
    exists to avoid — so two names for one inode remain two keys.
    """
    return posixpath.normpath(path.replace("\\", "/")).lstrip("./") or "."


def version_of(content: str) -> str:
    """A short content hash. Identity of *what was read*, not when."""
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass(frozen=True)
class StaleRead:
    """Why an edit is being made on an uncertain basis."""

    path: str
    reason: str

    def advisory(self) -> str:
        """Text appended to a tool result.

        Two constraints borrowed from ``format_syntax_failure``: it names the
        harness as the author, so the model does not debug a message it never
        produced; and it carries no promise pattern and never ends in a
        question mark, so quoting it back cannot trip ``looks_unfinished`` and
        turn an advisory into a "this run is unfinished" signal.
        """
        return (
            f"Note from the harness: {self.reason} "
            f"The edit was applied. If it did not do what you expected, "
            f"re-read {self.path} before the next edit."
        )


@dataclass
class FileCache:
    """Cached file content, shared by every agent in one run.

    Carries a **generation counter**. A read suspends at an ``await`` before it
    can store what it fetched, and tool calls in a turn run concurrently — so a
    ``bash`` command's invalidation could land in that window and the read
    would then reinsert pre-command bytes that nothing would ever evict again.
    A store whose generation no longer matches is dropped instead.
    """

    _entries: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    generation: int = 0

    def get(self, path: str) -> str | None:
        key = normalise_path(path)
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, path: str, content: str, *, generation: int | None = None) -> bool:
        """Store ``content``. Returns whether it was accepted."""
        if generation is not None and generation != self.generation:
            return False
        key = normalise_path(path)
        self._entries[key] = content
        self._entries.move_to_end(key)
        while len(self._entries) > MAX_CACHED_FILES:
            self._entries.popitem(last=False)
        return True

    def drop(self, path: str) -> None:
        self._entries.pop(normalise_path(path), None)

    def drop_all(self) -> None:
        """Forget everything, and invalidate reads already in flight.

        Called after any command that could write without the harness learning
        where. Conservative on purpose: a cache surviving an unobserved write
        hands the model a file that no longer exists in that form.
        """
        self._entries.clear()
        self.generation += 1

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class ReadLedger:
    """One agent's knowledge of file contents, over a shared cache."""

    cache: FileCache = field(default_factory=FileCache)
    #: Whether a stale edit produces an advisory. **False by default**, and
    #: that default is the neutrality argument: the benchmark profile builds
    #: tools with warnings off, so its tool results are byte-identical while
    #: still getting the cache, which changes no bytes at all.
    advise: bool = False
    _versions: "OrderedDict[str, str]" = field(default_factory=OrderedDict)

    # -- cache ----------------------------------------------------------

    def cached(self, path: str) -> str | None:
        return self.cache.get(path)

    @property
    def generation(self) -> int:
        return self.cache.generation

    def record_read(self, path: str, content: str, *, generation: int | None = None) -> None:
        if self.cache.put(path, content, generation=generation):
            self._remember(path, content)

    def invalidate(self, path: str) -> None:
        self.cache.drop(path)

    def invalidate_all(self) -> None:
        self.cache.drop_all()

    # -- staleness ------------------------------------------------------

    def note_write(self, path: str, content: str) -> None:
        """Record content the harness itself produced.

        The agent did not *read* it, but it authored it, so this counts as
        knowledge. Without it, every edit after the first on one file warns —
        an advisory firing on the most ordinary sequence there is, which is how
        a warning becomes noise and then gets ignored.
        """
        self.cache.put(path, content)
        self._remember(path, content)

    def _remember(self, path: str, content: str) -> None:
        key = normalise_path(path)
        self._versions[key] = version_of(content)
        self._versions.move_to_end(key)
        while len(self._versions) > MAX_TRACKED_VERSIONS:
            self._versions.popitem(last=False)

    def check(self, path: str, current: str) -> StaleRead | None:
        """Whether an edit to ``path`` is being made on a stale basis."""
        if not self.advise:
            return None
        known = self._versions.get(normalise_path(path))
        if known is None:
            return StaleRead(
                path,
                f"{path} was edited without being read first, so the edit was "
                "based on an assumption about its contents.",
            )
        if known != version_of(current):
            return StaleRead(
                path,
                f"{path} changed since it was last read, so the version edited "
                "is not the version seen.",
            )
        return None
