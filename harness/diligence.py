"""Diligence machinery: the deterministic stop-condition check (DESIGN.md §4.9).

Before the agent loop accepts a final answer (a model response with no tool
calls), it asks :func:`looks_unfinished`: does the last message promise
future work, end in a question the agent could answer itself, or leave task
ledger items open? If so, the loop injects :data:`CONTINUE_REMINDER` as a
user message instead of terminating — bounded to :data:`MAX_NUDGES` nudges
per run so a stubborn model cannot loop forever.

v1 is fully deterministic — no model calls. The check is a handful of
case-insensitive phrase patterns (promised-future-work phrasings like
"I will ..." / "I'll ..." / "let me know" / "next, I" / "once you ..."), a
trailing-question check, and the open-item count from the SQLite-backed task
ledger. DESIGN.md §4.9 sketches a cheap-model check; that is a possible v2
upgrade behind this same function signature.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "MAX_NUDGES",
    "CONTINUE_REMINDER",
    "looks_unfinished",
]

#: Maximum number of continue-reminders injected per run ("bounded to M
#: nudges to avoid loops", DESIGN.md §4.9). After this many, the loop
#: accepts the final answer as-is.
MAX_NUDGES: Final[int] = 2

#: Injected as a user message when :func:`looks_unfinished` flags a final
#: answer. Format with ``reason=`` (the second element of the tuple
#: :func:`looks_unfinished` returns).
CONTINUE_REMINDER: Final[str] = (
    "<system-reminder>\n"
    "Your last message looks unfinished: {reason}.\n"
    "Do not stop here. Either finish the remaining work now, or explicitly\n"
    "close out each open task-ledger item with concrete evidence (e.g. test\n"
    "output visible in the transcript) and state clearly that the task is\n"
    "complete. Do not promise future work or ask questions you can answer\n"
    "yourself — do the work, or explain precisely why it cannot be done.\n"
    "</system-reminder>"
)

#: Promised-future-work phrasings, each paired with the human-readable
#: reason reported when it matches. Patterns are matched case-insensitively
#: with word boundaries so e.g. "I willingly" does not trip "I will".
_PROMISE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\bi\s+will\b", re.IGNORECASE), "promises future work ('I will')"),
    (re.compile(r"\bi[’']ll\b", re.IGNORECASE), "promises future work (\"I'll\")"),
    (
        re.compile(r"\blet\s+me\s+know\b", re.IGNORECASE),
        "defers to the user ('let me know')",
    ),
    (
        re.compile(r"\bnext,\s*i\b", re.IGNORECASE),
        "announces a next step instead of taking it ('next, I')",
    ),
    (
        re.compile(r"\bonce\s+you\b", re.IGNORECASE),
        "waits on the user ('once you')",
    ),
)


def looks_unfinished(
    final_text: str | None, open_task_count: int
) -> tuple[bool, str]:
    """Decide whether a would-be final answer actually looks unfinished.

    Parameters
    ----------
    final_text:
        The text of the model's final (tool-call-free) message; ``None`` is
        treated as empty.
    open_task_count:
        Number of task-ledger items not yet closed out (any status other
        than done/completed/cancelled, as counted by the caller).

    Returns
    -------
    tuple[bool, str]
        ``(True, reason)`` when the message promises future work, ends in a
        question the agent could answer itself, or leaves ledger items open;
        ``(False, "")`` otherwise. ``reason`` joins every triggered signal
        with ``"; "`` so the continue-reminder can cite all of them.
    """
    text = final_text or ""
    reasons: list[str] = []

    for pattern, reason in _PROMISE_PATTERNS:
        if pattern.search(text):
            reasons.append(reason)

    if text.rstrip().endswith("?"):
        reasons.append("ends in a question the agent could answer itself")

    if open_task_count > 0:
        plural = "s" if open_task_count != 1 else ""
        reasons.append(
            f"{open_task_count} task-ledger item{plural} still open"
        )

    if reasons:
        return True, "; ".join(reasons)
    return False, ""
