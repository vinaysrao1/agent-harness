"""Permission engine (DESIGN.md §4.11).

Every tool call passes through :func:`evaluate` before execution. The engine
is a pure function of ``(tool_name, meta, policy)`` — no I/O, no hidden
state — so it can be unit tested exhaustively and called synchronously from
the dispatch hot path.

Precedence (highest to lowest):

1. ``policy.deny`` glob patterns match the tool name -> :attr:`Decision.DENY`.
2. The tool's :class:`ToolMeta.categories` intersect
   :data:`HARD_DENY_CATEGORIES` -> :attr:`Decision.DENY`, in *both* modes.
   ``auto`` mode is not zero policy (DESIGN.md §4.11): credential handling,
   permanent deletion, and payments are always blocked, and no ``allow``
   pattern can override this.
3. ``policy.allow`` glob patterns match the tool name -> :attr:`Decision.ALLOW`.
4. ``policy.mode == PermissionMode.AUTO`` -> :attr:`Decision.ALLOW`.
5. ``policy.mode == PermissionMode.GATED`` -> :attr:`Decision.ASK` if the tool
   is side-effecting, else :attr:`Decision.ALLOW`.
"""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict

from harness.config import PermissionMode

__all__ = [
    "Decision",
    "ToolMeta",
    "Policy",
    "HARD_DENY_CATEGORIES",
    "evaluate",
]


class Decision(str, Enum):
    """The outcome of evaluating a tool call against a policy."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


#: Tool categories blocked unconditionally, regardless of mode or allow
#: patterns (DESIGN.md §4.11: "Even auto mode keeps the deny-list — full
#: autonomy is not zero policy."). ``external_send`` is deliberately absent:
#: sends are gated in ``gated`` mode via ``side_effect`` but are not hard-denied
#: in ``auto`` mode.
HARD_DENY_CATEGORIES: frozenset[str] = frozenset(
    {"credential", "permanent_delete", "payment"}
)


class ToolMeta(BaseModel):
    """Static description of a tool used by the permission engine.

    ``side_effect`` marks tools that change external state (sends, writes
    outside the sandbox, deletes, ...); in ``gated`` mode these require
    approval unless otherwise allowed. ``categories`` tags the tool with zero
    or more of ``"credential"``, ``"permanent_delete"``, ``"payment"``,
    ``"external_send"`` — membership in :data:`HARD_DENY_CATEGORIES` forces a
    deny in every mode.
    """

    model_config = ConfigDict(frozen=True)

    side_effect: bool
    categories: frozenset[str] = frozenset()


class Policy(BaseModel):
    """Per-run permission policy: an autonomy mode plus pattern overrides.

    ``allow``/``deny`` are lists of :mod:`fnmatch` glob patterns matched
    against the tool name (e.g. ``"mcp.github.*"``). ``Policy`` is immutable
    (frozen); accumulate session grants via :meth:`with_grant`, which returns
    a new ``Policy`` rather than mutating this one.
    """

    model_config = ConfigDict(frozen=True)

    mode: PermissionMode
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def with_grant(self, pattern: str) -> "Policy":
        """Return a new :class:`Policy` with ``pattern`` appended to ``allow``.

        Models a user's "always for this run" response to an ASK prompt:
        the pattern is remembered for the remainder of the run without
        mutating this policy object (or any policy object already held by
        other agents/subagents sharing the run).
        """
        return self.model_copy(update={"allow": (*self.allow, pattern)})


def _matches_any(tool_name: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``tool_name`` matches any of ``patterns`` via fnmatch.

    Uses :func:`fnmatch.fnmatchcase`, which is always case-sensitive and does
    not apply :func:`os.path.normcase`. Tool names are not filesystem paths,
    so matching must not vary by platform (plain ``fnmatch.fnmatch`` would
    match case-insensitively on Windows, over-granting there).
    """
    return any(fnmatchcase(tool_name, pattern) for pattern in patterns)


def evaluate(tool_name: str, meta: ToolMeta, policy: Policy) -> Decision:
    """Decide whether a tool call is allowed, denied, or needs approval.

    Pure function; see the module docstring for the full precedence order.
    """
    if _matches_any(tool_name, policy.deny):
        return Decision.DENY

    if meta.categories & HARD_DENY_CATEGORIES:
        return Decision.DENY

    if _matches_any(tool_name, policy.allow):
        return Decision.ALLOW

    if policy.mode == PermissionMode.AUTO:
        return Decision.ALLOW

    # policy.mode == PermissionMode.GATED
    return Decision.ASK if meta.side_effect else Decision.ALLOW
