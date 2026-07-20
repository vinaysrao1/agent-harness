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

Self-verification (DESIGN.md §10.3 B1) hardens this heuristic into
enforcement: the model may *declare* a shell command that proves the goal is
met (via the ``declare_verification`` tool,
:data:`VERIFICATION_TOOL_NAME`), and the loop then re-executes that command
before accepting ``completed`` — exit 0 finishes the run, anything else
injects :data:`VERIFICATION_FAILED_REMINDER` and continues, sharing the
same :data:`MAX_NUDGES` budget so a permanently-failing check cannot loop
forever. With no declaration, :func:`looks_unfinished` alone decides,
exactly as before. The constants for that mechanism live here; the
execution itself is in :mod:`harness.loop`.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "MAX_NUDGES",
    "CONTINUE_REMINDER",
    "VERIFICATION_TOOL_NAME",
    "VERIFICATION_TIMEOUT_SECONDS",
    "VERIFICATION_OUTPUT_LIMIT",
    "VERIFICATION_FAILED_REMINDER",
    "truncate_verification_output",
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

#: Name of the tool the model uses to declare its verification command
#: (DESIGN.md §10.3 B1). Shared by the tool factory
#: (:func:`harness.tools.builtin.declare_verification_tool`) and the loop,
#: which watches for successful calls to it and holds the model to the
#: declared check before accepting completion.
VERIFICATION_TOOL_NAME: Final[str] = "declare_verification"

#: Hard timeout for one execution of the declared verification command.
#: Bounded so a hung check (e.g. a test suite waiting on input) cannot
#: stall the run indefinitely; a timeout counts as a failed verification.
VERIFICATION_TIMEOUT_SECONDS: Final[float] = 300.0

#: Max characters of verification-command output persisted with
#: ``verification_passed``/``verification_failed`` events and injected into
#: :data:`VERIFICATION_FAILED_REMINDER`. The *tail* is kept — test runners
#: put the failure summary at the end.
VERIFICATION_OUTPUT_LIMIT: Final[int] = 4_000

#: Injected as a user message when the declared verification command exits
#: non-zero (or times out) on a would-be final answer. Format with
#: ``command=``, ``exit_code=``, and ``output=`` (already truncated via
#: :func:`truncate_verification_output`).
VERIFICATION_FAILED_REMINDER: Final[str] = (
    "<system-reminder>\n"
    "Your declared verification command failed (exit code {exit_code}):\n"
    "  {command}\n"
    "Output:\n"
    "{output}\n"
    "The task is not complete until this check passes. Fix the underlying\n"
    "problem and finish again — the command will be re-run before your\n"
    "answer is accepted. If the check itself is wrong, redeclare it with\n"
    "declare_verification.\n"
    "</system-reminder>"
)


def truncate_verification_output(output: str) -> str:
    """Truncate verification output to :data:`VERIFICATION_OUTPUT_LIMIT`.

    Keeps the *tail* (where test runners summarize failures) and prepends a
    marker naming how much was dropped, so the model knows it is looking at
    the end of a longer stream.
    """
    if len(output) <= VERIFICATION_OUTPUT_LIMIT:
        return output
    dropped = len(output) - VERIFICATION_OUTPUT_LIMIT
    return (
        f"[...{dropped} chars truncated...]\n"
        + output[-VERIFICATION_OUTPUT_LIMIT:]
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
