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
    ``"neutralized_exit"``, ``"existence_only"``), ``message`` is the
    advisory shown to the model, and ``details`` carries the structured
    evidence a later round mines from the ``verification_lint`` corpus.
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

    Three deliberately narrow detectors, in decreasing severity:

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

    Everything else — ``pytest``, ``make``, ``cmp``, ``python3 -c ...`` —
    is never analyzed. Returns ``[]`` for an empty command, and gives up
    silently (no findings from the affected detector) on anything it cannot
    confidently tokenize or that carries variables or command substitution
    in the operand it would have to reason about.
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

#: Tokens that end one command segment. ``punctuation_chars`` mode emits
#: operator runs (``&&``, ``||``) as single tokens.
_SEGMENT_SEPARATORS: Final[frozenset[str]] = frozenset(
    {";", ";;", "&&", "||", "|", "|&", "&", "(", ")", "{", "}"}
)

#: Shell keywords skipped at the head of a segment so ``if grep -q X f``
#: is analyzed as the grep it is.
_LEADING_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "time"}
)

#: Redirect operators, recognized when scanning a segment.
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
