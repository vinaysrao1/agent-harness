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

Verification *quality* is a separate, **warn-only** concern (round-2
Change 4): a declared command can exit 0 without proving anything — it can
be neutralized (``pytest -q || true``), it can grep for a literal the agent
itself wrote into the very file being read (a tautology), or it can merely
assert a path exists. :func:`lint_verification` names those shapes;
:class:`WrittenData` is the bounded record of what the agent wrote where,
which the tautology detector needs. Nothing here rejects a declaration or
changes control flow — the lint appends an advisory to the tool result and
emits a ``verification_lint`` event, building the labelled corpus a later
round needs before any of this can be made to block.

Round 3 (T1) adds two more shapes to the same warn-only lint, purely to
grow that corpus: a command that never runs anything at all *and* only
inspects files this run wrote by hand (``no_execution`` — the same
read-vs-produced provenance rule detector 1 uses, so an honest check that
greps output a program produced is not flagged), and a command that runs
a checker script the agent wrote *this run*
(``self_authored_checker``) — a real tautology class the
other detectors cannot see, because the circular literal lives inside the
checker's own logic rather than in a file the command's operands name. A
verification-*strength* classifier able to act on either signal was
scoped for this round and cut: round-2's real declared commands have no
principled split between "strong" and "weak" on the tools this lint has
(``_split_segments`` can even hand back shell comparison operators like
``-eq``/``-lt`` as segment heads, not command names, when a command
substitution sits inside a ``test`` expression). Nothing reads these two
detectors' findings.
"""

from __future__ import annotations

import re
import shlex
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "MAX_NUDGES",
    "CONTINUE_REMINDER",
    "VERIFICATION_TOOL_NAME",
    "VERIFICATION_TIMEOUT_SECONDS",
    "VERIFICATION_OUTPUT_LIMIT",
    "VERIFICATION_FAILED_REMINDER",
    "VERIFICATION_LINT_EVENT",
    "WRITTEN_DATA_MAX_PATHS",
    "WRITTEN_DATA_MAX_LINES",
    "WRITTEN_DATA_MIN_LINE_LENGTH",
    "MIN_DISCRIMINATING_LITERAL",
    "WrittenData",
    "LintFinding",
    "record_written_data",
    "lint_verification",
    "format_lint_advisory",
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


# ---------------------------------------------------------------------------
# Verification-quality lint (warn-only)
# ---------------------------------------------------------------------------

#: Transcript event kind emitted for every linted declaration that produced
#: at least one finding. Payload: ``{command, findings, action}`` with
#: ``action`` pinned to ``"warn"`` — round 2 never rejects, and the event is
#: the corpus a later round needs to justify one.
VERIFICATION_LINT_EVENT: Final[str] = "verification_lint"

#: Maximum distinct paths tracked in a :class:`WrittenData` map (LRU beyond).
WRITTEN_DATA_MAX_PATHS: Final[int] = 32

#: Maximum recorded lines per tracked path.
WRITTEN_DATA_MAX_LINES: Final[int] = 200

#: Lines shorter than this are never recorded: short strings ("PASS", "ok",
#: "1") are not discriminating, and recording them is how a tautology
#: detector starts flagging every honest check.
WRITTEN_DATA_MIN_LINE_LENGTH: Final[int] = 8

#: Minimum length of a literal that can be called "discriminating". Same
#: rationale as :data:`WRITTEN_DATA_MIN_LINE_LENGTH`, applied to the
#: verification side of the comparison.
MIN_DISCRIMINATING_LITERAL: Final[int] = 8


class WrittenData:
    """Bounded record of literal text the agent wrote into specific paths.

    The tautology detector's whole precision rests on *what goes in here*:
    a key is a file the agent wrote **content into directly**
    (``write_file``/``edit_file``, or a shell ``echo``/``printf``/heredoc
    redirect), never a file some program it wrote later *produced*. That is
    the distinction between ``echo "X" > result.txt`` followed by
    ``grep -q X result.txt`` (circular — the check cannot fail) and
    ``print("PASS")`` inside ``solve.py`` followed by
    ``grep -q PASS test.log`` (honest — ``test.log`` exists only because
    the program ran and produced it).

    Bounded on purpose (it is live loop state, not a database): at most
    ``max_paths`` paths, LRU-evicted by last write, at most ``max_lines``
    lines each, and lines shorter than ``min_line_length`` are dropped.
    """

    def __init__(
        self,
        *,
        max_paths: int = WRITTEN_DATA_MAX_PATHS,
        max_lines: int = WRITTEN_DATA_MAX_LINES,
        min_line_length: int = WRITTEN_DATA_MIN_LINE_LENGTH,
    ) -> None:
        self.max_paths = max_paths
        self.max_lines = max_lines
        self.min_line_length = min_line_length
        self._paths: OrderedDict[str, set[str]] = OrderedDict()

    def record(self, path: str, text: str) -> None:
        """Record the lines of ``text`` as written into ``path``.

        Whitespace-stripped lines shorter than ``min_line_length`` are
        ignored; a path already at ``max_lines`` keeps what it has. Writing
        to a path refreshes its LRU position, and recording a new path
        beyond ``max_paths`` evicts the least recently written one.
        """
        key = _normalize_path(path)
        if not key:
            return
        lines = self._paths.get(key)
        if lines is None:
            lines = set()
            self._paths[key] = lines
            while len(self._paths) > self.max_paths:
                self._paths.popitem(last=False)
        self._paths.move_to_end(key)
        for raw in text.splitlines():
            line = raw.strip()
            if len(line) < self.min_line_length:
                continue
            if len(lines) >= self.max_lines:
                break
            lines.add(line)

    def lines_for(self, path: str) -> frozenset[str]:
        """Lines recorded for ``path``, matched by full path *or* basename.

        Both forms are accepted because the command that writes a file and
        the command that reads it routinely disagree about the working
        directory (``echo ... > /app/result.txt`` then, after ``cd /app``,
        ``grep -q ... result.txt``). The residual imprecision is a
        same-basename collision in different directories, which can only
        over-warn — acceptable while the lint is warn-only.
        """
        key = _normalize_path(path)
        if not key:
            return frozenset()
        base = key.rsplit("/", 1)[-1]
        found: set[str] = set()
        for candidate, lines in self._paths.items():
            if candidate == key or candidate.rsplit("/", 1)[-1] == base:
                found |= lines
        return frozenset(found)

    def paths(self) -> tuple[str, ...]:
        """Tracked paths, least-recently-written first."""
        return tuple(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


@dataclass(frozen=True)
class LintFinding:
    """One verification-quality warning.

    ``kind`` is the stable machine label (``"tautology"``,
    ``"neutralized_exit"``, ``"existence_only"``, ``"no_execution"``,
    ``"self_authored_checker"``), ``message`` is the advisory shown to the
    model, and ``details`` carries the structured evidence a later round
    mines from the ``verification_lint`` corpus.
    """

    kind: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        """JSON-serializable form for the ``verification_lint`` event."""
        return {
            "kind": self.kind,
            "message": self.message,
            "details": dict(self.details),
        }


def record_written_data(
    written_data: WrittenData, tool_name: str, arguments: dict
) -> None:
    """Feed one successful tool call into ``written_data``.

    Only three tools can put a literal into a file: ``write_file`` (whole
    ``content``), ``edit_file`` (the ``new_string`` it splices in), and
    ``bash`` — the latter strictly through an ``echo``/``printf`` redirect
    or a heredoc into a file (:func:`_record_bash_writes`). A ``bash``
    command that redirects a *program's* output into a file records
    nothing, which is exactly what keeps ``grep -q PASS test.log`` off the
    tautology list. Unknown tools and malformed arguments are ignored.
    """
    if not isinstance(arguments, dict):
        return
    if tool_name == "write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        if isinstance(path, str) and isinstance(content, str):
            written_data.record(path, content)
    elif tool_name == "edit_file":
        path = arguments.get("path")
        new_string = arguments.get("new_string")
        if isinstance(path, str) and isinstance(new_string, str):
            written_data.record(path, new_string)
    elif tool_name == "bash":
        command = arguments.get("command")
        if isinstance(command, str):
            _record_bash_writes(written_data, command)


def lint_verification(
    command: str, written_data: WrittenData | None = None
) -> list[LintFinding]:
    """Lint a declared verification command. **Warn-only, never rejects.**

    Five deliberately narrow detectors:

    1. **tautology** — the command greps for (or string-compares against) a
       literal that the agent itself wrote *into the file the command
       reads*. Needs ``written_data``; without it this detector is silent.
    2. **neutralized_exit** — the command contains ``|| true``, ``|| :``,
       ``; true`` or ``; exit 0``. Reported for *any* occurrence, with
       position information in ``details``, because telling a terminal
       neutralizer from a legitimate non-terminal one
       (``pkill -f server || true; pytest -q``) needs real shell
       tokenization rather than a warning heuristic.
    3. **existence_only** — the whole command is one ``test -f``/``ls``,
       which proves a path exists and nothing about its content.
    4. **no_execution** (round 3, T1) — every segment head is a known
       read-only file predicate (``test``/``cat``/``grep``/``cmp``/``ls``/
       ...) or a comparison-operator artifact :func:`_split_segments` can
       hand back from inside a command substitution, *and* at least one
       path the command reads is one this run wrote a literal into. So
       nothing in the check runs the solution and nothing it reads was
       produced by running it. An unrecognized segment head counts
       *against* the finding, never for it, and an operand of unknown
       provenance yields nothing — see :func:`_lint_no_execution`. Needs
       ``written_data``.
    5. **self_authored_checker** (round 3, T1) — the command runs
       ``<interpreter> <script>`` (``perl verify.pl``) where ``script`` is
       a path the agent wrote *this run* (present in ``written_data``). A
       checker script the agent authored can assert anything; a tautology
       inside its own logic is invisible to detector 1, which only looks
       at what the command's own operands compare against, never at what a
       script it invokes does internally. Needs ``written_data``.

    ``pytest``, ``make``, ``python3 -c ...`` and similar are never
    analyzed by any detector above. Returns ``[]`` for an empty command,
    and gives up silently (no findings from the affected detector) on
    anything it cannot confidently tokenize or that carries variables or
    command substitution in the operand it would have to reason about.
    """
    if not command or not command.strip():
        return []
    findings: list[LintFinding] = []
    tokens = _tokenize(command)
    segments = _split_segments(tokens) if tokens is not None else []
    if written_data is not None:
        findings.extend(_lint_tautology(segments, written_data))
    findings.extend(_lint_neutralized_exit(command, tokens))
    findings.extend(_lint_existence_only(segments))
    if written_data is not None:
        findings.extend(_lint_no_execution(segments, written_data))
        findings.extend(_lint_self_authored_checker(segments, written_data))
    return findings


def format_lint_advisory(findings: Sequence[LintFinding]) -> str:
    """Render ``findings`` as the advisory appended to the tool result.

    Returns ``""`` when there is nothing to say. The wording deliberately
    avoids :data:`_PROMISE_PATTERNS` phrasing and never ends in ``"?"``, so
    a model that echoes it cannot make its own final message read as
    unfinished to :func:`looks_unfinished`.
    """
    if not findings:
        return ""
    lines = [
        "Advisory (the declaration above stands; this changes nothing): "
        "this check may not prove what it appears to prove."
    ]
    lines.extend(f"- {finding.message}" for finding in findings)
    lines.append(
        "Consider redeclaring a stronger check with declare_verification "
        "if any of the above applies."
    )
    return "\n".join(lines)


# -- tokenization ------------------------------------------------------------

#: Process substitution openers, which ``punctuation_chars`` mode emits as
#: single tokens. What follows one is a *command*, not an operand: in
#: ``grep -q PASS <(python3 solve.py)`` the solution is executed and its
#: output is what grep reads. They are segment separators for exactly that
#: reason — splitting there puts the inner ``python3`` in head position, so
#: every head-based detector sees the execution, and the script name stops
#: being mistaken for a file the check merely read. (A redirect entry would
#: not do: ``<`` is followed by a path, ``<(`` by a command line.)
_PROCESS_SUBSTITUTIONS: Final[frozenset[str]] = frozenset({"<(", ">("})

#: Tokens that end one command segment. ``punctuation_chars`` mode emits
#: operator runs (``&&``, ``||``) as single tokens.
_SEGMENT_SEPARATORS: Final[frozenset[str]] = (
    frozenset({";", ";;", "&&", "||", "|", "|&", "&", "(", ")", "{", "}"})
    | _PROCESS_SUBSTITUTIONS
)

#: Shell keywords skipped at the head of a segment so ``if grep -q X f``
#: is analyzed as the grep it is.
_LEADING_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "time"}
)

#: Redirect operators, recognized when scanning a segment. Process
#: substitution is absent on purpose: it is split on instead
#: (:data:`_PROCESS_SUBSTITUTIONS`), so it never reaches a segment scan.
_REDIRECTS: Final[frozenset[str]] = frozenset({">", ">>", "<", "<<", "<<<"})


def _tokenize(command: str) -> list[str] | None:
    """Shell-tokenize ``command``; ``None`` when it cannot be tokenized.

    ``posix=True`` strips quoting (so ``'X'`` and ``"X"`` both yield ``X``)
    while leaving ``$``/backticks intact, which is what the detectors test
    for when they decide to give up. Unbalanced quotes return ``None``.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def _split_segments(tokens: Sequence[str]) -> list[list[str]]:
    """Split a token list into command segments on shell operators."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEGMENT_SEPARATORS:
            if current:
                segments.append(current)
            current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return [stripped for seg in segments if (stripped := _strip_keywords(seg))]


def _strip_keywords(segment: list[str]) -> list[str]:
    """Drop leading shell keywords and env assignments from a segment."""
    index = 0
    while index < len(segment) and (
        segment[index] in _LEADING_KEYWORDS
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[index])
    ):
        index += 1
    return segment[index:]


def _normalize_path(path: str) -> str:
    """Normalize a path operand for :class:`WrittenData` keys and lookups.

    Strips surrounding quotes and trailing separators, collapses ``./`` and
    duplicate slashes, and leaves the leading ``/`` of an absolute path
    intact so full-path and basename matching both work.
    """
    text = path.strip().strip("'\"").strip()
    if not text:
        return ""
    text = re.sub(r"/{2,}", "/", text)
    while text.startswith("./"):
        text = text[2:]
    if len(text) > 1:
        text = text.rstrip("/")
    return text


# -- detector 1: tautology ---------------------------------------------------

_GREP_COMMANDS: Final[frozenset[str]] = frozenset({"grep", "egrep"})

_TEST_COMMANDS: Final[frozenset[str]] = frozenset({"test", "[", "[["})

#: Regex metacharacters that stop a grep pattern from being read as a
#: literal. ``{``/``}`` are absent on purpose: they are literal in basic
#: regex and common inside flag payloads (``flag{...}``).
_REGEX_METACHARS: Final[frozenset[str]] = frozenset("*+?[]()|\\.")

#: ``$(cat PATH)`` / ``` `cat PATH` ``` — the only command substitution the
#: test-comparison detector understands.
_CAT_SUBSTITUTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:\$\(\s*cat\s+(?P<paren>[^\s()]+)\s*\)"
    r"|`\s*cat\s+(?P<tick>[^\s`]+)\s*`)$"
)


def _lint_tautology(
    segments: Sequence[Sequence[str]], written_data: WrittenData
) -> list[LintFinding]:
    """Flag checks whose discriminating literal was written into the file
    the check reads (see :class:`WrittenData` for why that is the rule)."""
    findings: list[LintFinding] = []
    for segment in segments:
        if not segment:
            continue
        name = segment[0].rsplit("/", 1)[-1]
        if name in _GREP_COMMANDS:
            parsed = _parse_grep(segment)
            if parsed is None:
                continue
            pattern, paths = parsed
            literal = _grep_literal(pattern)
            if literal is None:
                continue
            findings.extend(
                _tautology_findings(literal, paths, written_data, "grep")
            )
        elif name in _TEST_COMMANDS:
            parsed = _parse_test_comparison(segment)
            if parsed is None:
                continue
            literal, path = parsed
            findings.extend(
                _tautology_findings(
                    literal, [path], written_data, "string comparison"
                )
            )
    return findings


def _tautology_findings(
    literal: str,
    paths: Sequence[str],
    written_data: WrittenData,
    shape: str,
) -> list[LintFinding]:
    """Emit a finding per read path that already contains ``literal``."""
    findings: list[LintFinding] = []
    for path in paths:
        recorded = written_data.lines_for(path)
        if not recorded:
            continue
        if not any(literal == line or literal in line for line in recorded):
            continue
        findings.append(
            LintFinding(
                kind="tautology",
                message=(
                    f"the {shape} looks circular: the literal {literal!r} "
                    f"was written into {path} earlier in this run, so the "
                    "check passes whether or not the underlying task was "
                    "actually solved. A check that reads output produced by "
                    "running the solution proves more."
                ),
                details={
                    "literal": literal,
                    "path": path,
                    "shape": shape,
                },
            )
        )
    return findings


def _parse_grep(segment: Sequence[str]) -> tuple[str, list[str]] | None:
    """Split a ``grep``/``egrep`` segment into ``(pattern, file operands)``.

    Returns ``None`` — give up silently — when the pattern comes from a
    file (``-f``), when several patterns are given, or when there is no
    file operand to read (a pipeline's grep reads stdin, and stdin is not
    a path this lint can reason about).
    """
    pattern: str | None = None
    paths: list[str] = []
    args = list(segment[1:])
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _REDIRECTS:
            break
        if arg.startswith("--"):
            if arg in ("--file", "--regexp"):
                if arg == "--file":
                    return None
                index += 1
                if index >= len(args) or pattern is not None:
                    return None
                pattern = args[index]
            elif arg.startswith("--regexp="):
                if pattern is not None:
                    return None
                pattern = arg.split("=", 1)[1]
            elif arg.startswith("--file="):
                return None
            index += 1
            continue
        if arg.startswith("-") and len(arg) > 1:
            if arg == "-f":
                return None
            if arg == "-e":
                index += 1
                if index >= len(args) or pattern is not None:
                    return None
                pattern = args[index]
            elif arg in ("-m", "-A", "-B", "-C", "-d", "--"):
                index += 1
            index += 1
            continue
        if pattern is None:
            pattern = arg
        else:
            paths.append(arg)
        index += 1
    if pattern is None or not paths:
        return None
    return pattern, paths


def _grep_literal(pattern: str) -> str | None:
    """The literal string a grep pattern requires, or ``None``.

    Anchors are stripped (``^X$`` requires exactly ``X``); anything with
    regex metacharacters, variables, or command substitution is refused,
    as is anything too short or shaped like a bare path.
    """
    literal = pattern
    if literal.startswith("^"):
        literal = literal[1:]
    if literal.endswith("$") and not literal.endswith("\\$"):
        literal = literal[:-1]
    if any(char in _REGEX_METACHARS for char in literal):
        return None
    return literal if _is_discriminating(literal) else None


def _parse_test_comparison(
    segment: Sequence[str],
) -> tuple[str, str] | None:
    """Extract ``(literal, path)`` from ``test "$(cat PATH)" = 'LITERAL'``.

    Either side may hold the substitution. Returns ``None`` for any other
    shape, including ``!=`` and comparisons where the literal side is not
    discriminating.
    """
    tokens = [tok for tok in segment[1:] if tok not in ("]", "]]")]
    for index, token in enumerate(tokens):
        if token not in ("=", "=="):
            continue
        if index == 0 or index + 1 >= len(tokens):
            return None
        left, right = tokens[index - 1], tokens[index + 1]
        for source, literal in ((left, right), (right, left)):
            match = _CAT_SUBSTITUTION.match(source)
            if match is None:
                continue
            path = match.group("paren") or match.group("tick") or ""
            if not path or not _is_discriminating(literal):
                return None
            return literal, path
    return None


def _is_discriminating(literal: str) -> bool:
    """Whether ``literal`` is specific enough to call a check circular.

    Short strings, bare paths, and anything holding a variable or command
    substitution are refused — those are where the false positives live.
    """
    if len(literal) < MIN_DISCRIMINATING_LITERAL:
        return False
    if "$" in literal or "`" in literal:
        return False
    if literal.startswith(("/", "./", "../", "~/")):
        return False
    return True


# -- detector 2: neutralized exit --------------------------------------------

#: ``(operator, neutralizer)`` pairs that force exit 0.
_NEUTRALIZERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("||", ("true", ":")),
    (";", ("true",)),
)

#: Regex fallback for commands that cannot be tokenized (unbalanced
#: quotes); same four shapes, matched on the raw text.
_NEUTRALIZED_EXIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\|\|\s*(?:true\b|:(?=\s|;|$))|;\s*true\b|;\s*exit\s+0\b)"
)


def _lint_neutralized_exit(
    command: str, tokens: Sequence[str] | None
) -> list[LintFinding]:
    """Warn on every ``|| true`` / ``|| :`` / ``; true`` / ``; exit 0``.

    Warn-only and position-agnostic by design: ``pkill -f server || true;
    pytest -q`` is legitimate, and separating that from a terminal
    neutralizer needs real tokenization of shell *structure*, not of words.
    ``details["terminal"]`` records which one this occurrence looks like so
    a later round can design a precise reject rule from real data.
    """
    findings: list[LintFinding] = []
    if tokens is None:
        for match in _NEUTRALIZED_EXIT_RE.finditer(command):
            findings.append(
                _neutralized_finding(
                    match.group(0).strip(),
                    {
                        "source": "regex",
                        "char_offset": match.start(),
                        "terminal": not command[match.end() :].strip(),
                    },
                )
            )
        return findings
    for index, token in enumerate(tokens):
        text: str | None = None
        consumed = 0
        for operator, neutralizers in _NEUTRALIZERS:
            if token != operator or index + 1 >= len(tokens):
                continue
            following = tokens[index + 1]
            if following in neutralizers:
                text = f"{operator} {following}"
                consumed = 2
            elif (
                operator == ";"
                and following == "exit"
                and index + 2 < len(tokens)
                and tokens[index + 2] == "0"
            ):
                text = "; exit 0"
                consumed = 3
        if text is None:
            continue
        rest = tokens[index + consumed :]
        findings.append(
            _neutralized_finding(
                text,
                {
                    "source": "tokens",
                    "token_index": index,
                    "token_count": len(tokens),
                    "terminal": not rest,
                },
            )
        )
    return findings


def _neutralized_finding(text: str, details: dict[str, object]) -> LintFinding:
    """Build the ``neutralized_exit`` finding for one occurrence."""
    return LintFinding(
        kind="neutralized_exit",
        message=(
            f"the command contains {text!r}, which can force exit 0 even "
            "when the real check fails. Confirm that the exit status still "
            "reflects the result being proved."
        ),
        details={"text": text, **details},
    )


# -- detector 3: existence-only ----------------------------------------------

#: ``test`` operators that assert nothing about content.
_EXISTENCE_FLAGS: Final[frozenset[str]] = frozenset(
    {"-e", "-f", "-d", "-s", "-r"}
)


def _lint_existence_only(
    segments: Sequence[Sequence[str]],
) -> list[LintFinding]:
    """Warn when the entire command is one existence probe.

    Only the single-segment case: ``test -f out.txt && pytest -q`` proves
    something, so it is left alone. Structural-but-not-semantic checks in a
    real language (``python3 -c "assert set(d) == {...}"``) are deliberately
    out of scope — catching those needs semantics this lint does not have.
    """
    if len(segments) != 1:
        return []
    segment = list(segments[0])
    if not segment:
        return []
    name = segment[0].rsplit("/", 1)[-1]
    probe: str | None = None
    if name in _TEST_COMMANDS:
        args = [tok for tok in segment[1:] if tok not in ("]", "]]")]
        if len(args) == 2 and args[0] in _EXISTENCE_FLAGS:
            probe = f"{name} {args[0]}"
    elif name == "ls":
        probe = "ls"
    if probe is None:
        return []
    return [
        LintFinding(
            kind="existence_only",
            message=(
                f"the whole check is a single {probe!r}, which proves that "
                "a path exists and nothing about whether its contents are "
                "correct."
            ),
            details={"probe": probe},
        )
    ]


# -- detector 4: no_execution -------------------------------------------------

#: Segment heads recognized as pure file-reading/inspection predicates —
#: nothing in this set can run the solution under test. Deliberately small:
#: an unrecognized head (a real interpreter, a compiled binary, an
#: unfamiliar path) must count *against* the finding, never for it.
_NO_EXECUTION_SAFE_HEADS: Final[frozenset[str]] = frozenset(
    {
        "test",
        "[",
        "[[",
        "cat",
        "grep",
        "egrep",
        "fgrep",
        "head",
        "tail",
        "wc",
        "cmp",
        "diff",
        "ls",
        "stat",
        "file",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "true",
        "false",
    }
)


def _is_non_executing_artifact(head: str) -> bool:
    """Whether ``head`` cannot possibly be a real command invocation.

    ``_split_segments`` splits on ``(``/``)`` as well as ``;``/``&&``/etc.,
    so a command substitution containing a redirect — ``test $(wc -c <
    f) -lt 5000`` — gets its inner space-separated words split apart by
    the substitution's own parens, and the trailing comparison operator
    (``-lt``, ``-eq``, ``=``, ...) is handed back as if it were a fresh
    segment head. No shell can invoke a program named ``-eq`` or ``=``, so
    a head shaped like one is evidence of exactly this tokenizer artifact,
    not of an unrecognized command that might execute something.
    """
    if head in ("=", "==", "!="):
        return True
    return len(head) > 1 and head.startswith("-")


def _read_operand_paths(segments: Sequence[Sequence[str]]) -> list[str]:
    """Path-shaped operands the segments *read*, best-effort.

    Everything that is not the segment head, not a flag, not a ``test``
    bracket, and carries no variable or command substitution counts as a
    candidate path, plus the inner path of a ``$(cat PATH)`` substitution
    (:data:`_CAT_SUBSTITUTION`). A redirect *target* (``> f``) is skipped:
    that file is written, not read.

    Candidates are only ever looked up in a :class:`WrittenData` map, so a
    token that is not really a path simply matches nothing. Deliberately
    approximate in the harmless direction.
    """
    paths: list[str] = []
    for segment in segments:
        skip_next = False
        for index, token in enumerate(segment):
            if skip_next:
                skip_next = False
                continue
            if token in (">", ">>"):
                skip_next = True
                continue
            if token in _REDIRECTS:
                continue
            match = _CAT_SUBSTITUTION.match(token)
            if match is not None:
                inner = match.group("paren") or match.group("tick") or ""
                if inner:
                    paths.append(inner)
                continue
            if index == 0 or token in ("]", "]]"):
                continue
            if token.startswith("-") and len(token) > 1:
                continue
            if "$" in token or "`" in token:
                continue
            paths.append(token)
    return paths


#: A candidate operand recognizable as a filename: it contains a directory
#: separator, or its basename carries an extension. A guess, and applied
#: only to heads whose operands genuinely might not be files at all —
#: ``test``/``[``/``wc``-style segments, where a comparison literal
#: (``100``, ``flag{gc0d3_iz_ch4LLenGiNg}``) sits in the same position a
#: path would. Heads with a real grammar are parsed instead of guessed at
#: (:func:`_parse_grep`, :func:`_compare_operand_paths`), because dropping
#: a genuine operand here is not free: an unnamed file is an unrecorded
#: producer, which *causes* a false ``no_execution`` finding.
_PATH_SHAPED: Final[re.Pattern[str]] = re.compile(
    r"^(?:[^\s]*/[^\s]*|[A-Za-z0-9_@%~+-]+(?:\.[A-Za-z0-9_+-]+)+)$"
)

#: Heads whose argument list :func:`_parse_grep` can split into a pattern
#: and file operands. ``fgrep`` parses identically; only the pattern's
#: dialect differs, which does not matter for naming the files.
_GREP_LIKE_COMMANDS: Final[frozenset[str]] = _GREP_COMMANDS | {"fgrep"}

#: Heads that compare *files and nothing else*. Unlike ``test``, whose
#: operands may be literals (``test $(cat f) = ok``), every non-flag
#: operand of one of these is a filename by definition, whatever its
#: shape — which is why :func:`_compare_operand_paths` can name them
#: without the :data:`_PATH_SHAPED` guess.
_PURE_COMPARE_COMMANDS: Final[frozenset[str]] = frozenset({"cmp", "diff"})

#: Short options that consume the *following* token as their argument,
#: **keyed by command** — the two tools disagree and a head-agnostic set is
#: wrong in the dangerous direction. ``cmp -i SKIP`` and ``cmp -n LIMIT``
#: take an argument; ``diff -i`` (``--ignore-case``) and ``diff -n``
#: (``--rcs``) take none, so skipping after them swallows a real file
#: operand.
#:
#: Direction of error matters here. *Under*-skipping is safe: an argument
#: mistaken for a file lands in the unknown set, which only ever suppresses
#: a finding. *Over*-skipping is not: dropping a real operand can empty the
#: unknown set and emit a false ``no_execution`` advisory — telling the
#: model a command "reads only what you wrote" about a command that reads a
#: program's output. So each set stays conservative for its own command.
_COMPARE_FLAGS_WITH_ARG_BY_HEAD: Final[dict[str, frozenset[str]]] = {
    "cmp": frozenset({"-i", "-n"}),
    "diff": frozenset({"-I", "-D", "-S", "-W", "-x", "-X", "-F"}),
}

#: Fallback for a compare-shaped segment whose head we did not recognise:
#: the intersection, i.e. only flags that take an argument for *every*
#: known command, so an unknown head can never over-skip.
_COMPARE_FLAGS_WITH_ARG: Final[frozenset[str]] = frozenset(
    _COMPARE_FLAGS_WITH_ARG_BY_HEAD["cmp"] & _COMPARE_FLAGS_WITH_ARG_BY_HEAD["diff"]
)


def _compare_operand_paths(segment: Sequence[str]) -> list[str]:
    """Every file operand of a pure comparison (``cmp``/``diff``).

    The counterpart of :func:`_parse_grep` for the compare heads: it exists
    so :func:`_unknown_provenance_paths` can name the program-produced side
    of a golden-file check even when that side is a bare word. ``cmp
    expected.bin out`` compares two *files*; ``out`` carries no extension
    and no separator, so :data:`_PATH_SHAPED` cannot recognize it, and
    dropping it would leave the segment looking like a check whose only
    operand is hand-authored — the false ``no_execution`` finding this
    parser removes.

    Tokens carrying a variable or command substitution are still dropped,
    since their expansion is unknown, except ``$(cat PATH)``, whose inner
    path :data:`_CAT_SUBSTITUTION` recovers. A redirect *target* is skipped
    for the same reason as in :func:`_read_operand_paths`: it is written,
    not read.
    """
    paths: list[str] = []
    skip_next = False
    # Which short options consume the next token depends on WHICH compare
    # command this is: ``cmp -i`` takes a byte count, ``diff -i`` takes
    # nothing. An unrecognised head falls back to the intersection so it can
    # never over-skip and swallow a real operand.
    head = segment[0] if segment else ""
    flags_with_arg = _COMPARE_FLAGS_WITH_ARG_BY_HEAD.get(
        head.rsplit("/", 1)[-1], _COMPARE_FLAGS_WITH_ARG
    )
    for index, token in enumerate(segment):
        if skip_next:
            skip_next = False
            continue
        if token in (">", ">>"):
            skip_next = True
            continue
        if token in _REDIRECTS:
            continue
        match = _CAT_SUBSTITUTION.match(token)
        if match is not None:
            inner = match.group("paren") or match.group("tick") or ""
            if inner:
                paths.append(inner)
            continue
        if index == 0:
            continue
        if token.startswith("-") and len(token) > 1:
            if token in flags_with_arg:
                skip_next = True
            continue
        if "$" in token or "`" in token:
            continue
        paths.append(token)
    return paths


def _unknown_provenance_paths(
    segments: Sequence[Sequence[str]], written_data: WrittenData
) -> list[str]:
    """Files the segments read that this run never wrote into.

    An operand the harness never saw written has *unknown* provenance, not
    innocent provenance: a program may well have produced it. Naming those
    separately is what lets :func:`_lint_no_execution` require that every
    file a check reads be agent-authored instead of settling for one of
    them.

    Stricter than :func:`_read_operand_paths`, because here a wrong answer
    is not harmless *in either direction*. A non-path counted as an unknown
    file suppresses a true finding; an operand of a real file dropped for
    not looking like one *causes* a false finding, since
    :func:`_lint_no_execution` reads an empty unknown set as "nothing
    produced any of this". Each head therefore gets the parser its grammar
    allows:

    * grep-shaped segments are split by :func:`_parse_grep`, so the
      *pattern* is never mistaken for a file that is read —
      ``grep -q '^Qwen/Qwen3-Embedding-8B$' result.txt`` reads exactly one.
    * ``cmp``/``diff`` operands are all files by definition
      (:func:`_compare_operand_paths`), so an extensionless one — the
      ``out`` of ``cmp expected.bin out`` — is named as the file it is.
    * everything else keeps the :data:`_PATH_SHAPED` guess, because a
      ``test``/``[`` operand may be a bare literal to compare against
      rather than a file, and there the shape is all there is to go on.

    Segments whose operands cannot be named this confidently contribute
    nothing.
    """
    unknown: set[str] = set()
    for segment in segments:
        if not segment:
            continue
        head = segment[0].rsplit("/", 1)[-1]
        candidates: list[str]
        if head in _GREP_LIKE_COMMANDS:
            parsed = _parse_grep(segment)
            if parsed is None:
                # No file operand this parser is willing to name (a
                # pipeline's grep reads stdin, ``-f`` reads its pattern
                # from a file): claim nothing about this segment.
                continue
            candidates = list(parsed[1])
        elif head in _PURE_COMPARE_COMMANDS:
            candidates = _compare_operand_paths(segment)
        else:
            candidates = [
                token
                for token in _read_operand_paths([segment])
                if _PATH_SHAPED.match(token)
            ]
        unknown.update(
            path for path in candidates if not written_data.lines_for(path)
        )
    return sorted(unknown)


def _lint_no_execution(
    segments: Sequence[Sequence[str]], written_data: WrittenData
) -> list[LintFinding]:
    """Warn when a check neither runs anything nor reads anything a run
    produced.

    Two independent conditions, both required:

    **Structure.** Every segment head is in
    :data:`_NO_EXECUTION_SAFE_HEADS` or is a tokenizer artifact
    (:func:`_is_non_executing_artifact`); any other head means the command
    might run something, so nothing is reported — match conservatively and
    give up rather than guess. At least one segment must be a recognized
    read predicate, not merely an operator artifact.

    **Provenance.** *Every* file the command reads must be one this run
    wrote a literal into (:class:`WrittenData`) — at least one, and no
    operand of unknown provenance beside it
    (:func:`_unknown_provenance_paths`). One-authored-operand-is-enough
    would condemn the standard golden-file shape, ``cmp expected.bin
    build/out.bin``: its expectation is hand-written by construction, and
    the other side is the program's output, so an authored operand there is
    the check working as designed rather than evidence of circularity. An
    operand this run never wrote *is* the evidence that something ran.
    This is the same read-vs-produced distinction :func:`_lint_tautology`
    rests on, and it is what keeps the harness's own documented honest
    shape — ``python3 solve.py > test.log`` on one turn, ``grep -q PASS
    test.log`` on the next — off this list: ``test.log`` exists *only
    because the solution ran*, so a check reading it is inspecting real
    output, and telling the model otherwise would be a false diagnosis in
    text the model actually reads. A file the agent echoed or
    ``write_file``-d is
    the opposite case: nothing ran, at any point, between the literal
    being invented and the check confirming it.

    Provenance is positive evidence, never an assumption: an operand this
    run never touched (a task-supplied fixture, a file written before the
    map's LRU horizon) yields no finding, because the honest and circular
    readings are indistinguishable from here.

    A program-produced operand is by construction absent from
    ``written_data`` — :func:`_record_bash_writes` records only
    ``echo``/``printf``/heredoc redirects — so it can never satisfy the
    rule. The residual imprecision is a path first echoed and later
    overwritten by a program's output, which over-warns; that is the same
    bounded imprecision detector 1 already carries, and this lint warns
    only.
    """
    if not segments:
        return []
    matched_real = False
    for segment in segments:
        if not segment:
            return []
        head = segment[0].rsplit("/", 1)[-1]
        if head in _NO_EXECUTION_SAFE_HEADS:
            matched_real = True
            continue
        if _is_non_executing_artifact(head):
            continue
        return []
    if not matched_real:
        return []
    operands = _read_operand_paths(segments)
    authored = sorted(
        {path for path in operands if written_data.lines_for(path)}
    )
    if not authored:
        return []
    if _unknown_provenance_paths(segments, written_data):
        # A file this run never wrote sits beside the authored one — the
        # golden-file compare. Something produced that side; say nothing.
        return []
    return [
        LintFinding(
            kind="no_execution",
            message=(
                "every segment of this check only reads or compares "
                "existing files or state; nothing in it runs the solution "
                f"itself, and {', '.join(authored)} holds content this run "
                "wrote directly rather than output a program produced. A "
                "check that executes the program under test — or that "
                "reads a file running it produced — proves more."
            ),
            details={
                "segment_count": len(segments),
                "authored_paths": authored,
            },
        )
    ]


# -- detector 5: self_authored_checker -----------------------------------------

#: Interpreters whose next non-flag operand is typically a script path.
_SCRIPT_INTERPRETERS: Final[frozenset[str]] = frozenset(
    {
        "perl",
        "python",
        "python2",
        "python3",
        "ruby",
        "node",
        "nodejs",
        "php",
        "bash",
        "sh",
        "zsh",
    }
)

#: Flags that take inline code as their argument rather than a script path
#: (``python3 -c "..."``, ``perl -e '...'``) — that argument is not a path
#: this detector can look up in :class:`WrittenData`, so give up rather
#: than treat the inline body as if it were one.
_INLINE_CODE_FLAGS: Final[frozenset[str]] = frozenset({"-e", "-c", "-m"})


def _lint_self_authored_checker(
    segments: Sequence[Sequence[str]], written_data: WrittenData
) -> list[LintFinding]:
    """Warn when the command runs a script the agent wrote earlier this run.

    Matches ``<interpreter> <script>`` segments (``perl verify.pl``) where
    ``script`` names a path :class:`WrittenData` holds content for — i.e.
    the agent authored the checker itself via ``write_file``/``edit_file``
    or a bash echo/heredoc, rather than the checker shipping with the
    task. Inline code (``-e``/``-c``/``-m``) is skipped: it is not a script
    path, and guessing would be exactly the kind of unsafe assumption the
    tokenizer caveat on :func:`_lint_no_execution` warns against.
    """
    findings: list[LintFinding] = []
    for segment in segments:
        if not segment:
            continue
        head = segment[0].rsplit("/", 1)[-1]
        if head not in _SCRIPT_INTERPRETERS:
            continue
        if any(token in _INLINE_CODE_FLAGS for token in segment[1:]):
            continue
        script = next(
            (token for token in segment[1:] if not token.startswith("-")),
            None,
        )
        if script is None:
            continue
        if not written_data.lines_for(script):
            continue
        findings.append(
            LintFinding(
                kind="self_authored_checker",
                message=(
                    f"this check runs {script!r} via {head}, and "
                    f"{script!r} was written by the agent earlier in this "
                    "run rather than shipped with the task. A "
                    "self-authored checker can hide a tautology inside its "
                    "own logic where a literal-written-into-a-data-file "
                    "check cannot see it — read the script before trusting "
                    "a pass."
                ),
                details={"interpreter": head, "script": script},
            )
        )
    return findings


# -- bash write extraction ---------------------------------------------------

#: Heredoc introducer: ``<<EOF``, ``<<-'EOF'``, ``<< "EOF"``.
_HEREDOC_RE: Final[re.Pattern[str]] = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?P<word>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)

#: Commands whose arguments are literal text the agent chose, as opposed to
#: output some program computed. Only these feed :class:`WrittenData`.
_LITERAL_EMITTERS: Final[frozenset[str]] = frozenset({"echo", "printf"})


def _record_bash_writes(written_data: WrittenData, command: str) -> None:
    """Record ``echo``/``printf`` redirects and heredocs from a bash command.

    Line-oriented so heredoc bodies survive (the word tokenizer would
    scramble them). A redirect whose left-hand side is anything other than
    ``echo``/``printf`` records nothing at all — not even an empty entry —
    because a file produced by *running* a program is precisely the case
    the tautology detector must not treat as agent-written.
    """
    lines = command.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        heredoc = _HEREDOC_RE.search(line)
        if heredoc is not None:
            delimiter = heredoc.group("word")
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != delimiter:
                body.append(lines[index])
                index += 1
            index += 1
            target = _redirect_target(line)
            if target:
                written_data.record(target, "\n".join(body))
            continue
        tokens = _tokenize(line)
        if tokens is not None:
            for segment in _split_segments(tokens):
                _record_segment_write(written_data, segment)
        index += 1


def _record_segment_write(
    written_data: WrittenData, segment: Sequence[str]
) -> None:
    """Record one ``echo``/``printf`` segment that redirects into a file."""
    if not segment:
        return
    name = segment[0].rsplit("/", 1)[-1]
    if name not in _LITERAL_EMITTERS:
        return
    for index, token in enumerate(segment):
        if token not in (">", ">>") or index + 1 >= len(segment):
            continue
        target = segment[index + 1]
        values = [
            arg
            for arg in segment[1:index]
            if not (arg.startswith("-") and len(arg) > 1)
        ]
        if any("$" in value or "`" in value for value in values):
            return
        if values:
            written_data.record(target, "\n".join(values))
        return


def _redirect_target(line: str) -> str | None:
    """The file a heredoc line redirects into (``cat <<EOF > PATH``)."""
    tokens = _tokenize(line)
    if tokens is None:
        return None
    for index, token in enumerate(tokens):
        if token in (">", ">>") and index + 1 < len(tokens):
            return tokens[index + 1]
    return None
