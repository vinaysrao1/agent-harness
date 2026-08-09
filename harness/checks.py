"""Harness-run syntax checks on files the agent writes (§10.3 Change A).

The harness already lets the *model* declare a verification command
(:mod:`harness.diligence`), and that mechanism has a measured 0-for-35
catch rate across two unrelated models: the model authors the checker, so
the checker converges on a tautology it cannot fail. This module is the
other half of the answer, borrowed from OpenCode's post-write hooks: **the
harness runs a check the model did not choose**, on a file the model just
wrote, and appends the result to that tool call's output.

Nothing here keys on model, provider, or adapter — the only inputs are the
file's **path** (which checker applies, by suffix and, for the JSONC
filenames, by basename) and the checker's **exit status and output**
(whether to say anything). A run with a different model behaves identically.

Two properties are load-bearing, and both are about *not* being
net-negative:

**Silence on success.** A clean file appends nothing at all. The model's
transcript only grows when the harness has something the model does not
already know.

**Fail open, always.** A missing interpreter, an unreadable file, an exec
that raised, a check that timed out, a non-zero exit with no diagnostics —
every one of those appends nothing. The only thing that ever reaches the
model is a non-zero exit that carried a real diagnostic, and even then the
text says plainly that the harness ran it. A syntax checker that is not
installed must never look like a broken file.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Final

from harness.deadline import Deadline
from harness.persistence import RunStore
from harness.sandbox.base import Sandbox

__all__ = [
    "CHECK_TIMEOUT_SECONDS",
    "CHECK_OUTPUT_LIMIT",
    "SYNTAX_CHECK_EVENT",
    "SyntaxChecker",
    "SYNTAX_CHECKERS",
    "SyntaxCheckOutcome",
    "syntax_check_command",
    "syntax_check_language",
    "truncate_check_output",
    "format_syntax_failure",
    "run_syntax_check",
]

#: Wall-clock cap on one syntax check, in seconds.
#:
#: A syntax check is a parse, not a build: every checker below is expected
#: to answer in well under a second on any file an agent writes in one tool
#: call. The cap exists for the pathological case (a checker that blocks on
#: stdin, an interpreter thrashing on a 50MB file), and it is small because
#: this work is *unrequested* — the model asked to write a file, not to
#: spend its budget on the harness's opinion of that file.
CHECK_TIMEOUT_SECONDS: Final[float] = 10.0

#: Maximum characters of checker output appended to a tool result.
#:
#: A compiler that emits a megabyte of diagnostics would otherwise blow up
#: the transcript on behalf of a mechanism the model never invoked. The
#: *head* is kept, not the tail: the first diagnostic names the first
#: syntax error, and for a parser everything after it is usually cascade.
CHECK_OUTPUT_LIMIT: Final[int] = 2000

#: Transcript event kind emitted for every write where a checker applied.
#:
#: Payload: ``{path, language, ok, exit_code, skipped_reason, tool}``.
#: ``ok`` is ``True`` (clean), ``False`` (reported to the model), or
#: ``None`` (no verdict — see ``skipped_reason``). This project has
#: repeatedly shipped mechanisms that never fired; the event exists so the
#: next round can answer "how often did a harness-run check *run*, and how
#: often did it catch something" from the log instead of from belief.
SYNTAX_CHECK_EVENT: Final[str] = "syntax_check"


@dataclass(frozen=True)
class SyntaxChecker:
    """One suffix family's syntax check: a language label and a command.

    ``template`` is a shell command with a single ``{path}`` placeholder,
    filled by :func:`syntax_check_command` with a ``shlex.quote``-escaped
    path. ``language`` is the human-readable label used in the transcript
    event and in the text appended to the tool result.
    """

    language: str
    template: str


#: The suffix -> checker registry. Deliberately data, not code: the mapping
#: is the whole policy, and :func:`syntax_check_command` is pure over it, so
#: the decision "what runs for a `.py` file" is unit-testable without a
#: sandbox, a model, or a clock.
#:
#: Two choices in here are load-bearing:
#:
#: - **Python does not use ``python3 -m py_compile``.** ``py_compile``
#:   writes ``__pycache__/<name>.cpython-NN.pyc`` next to the source, and an
#:   unexpected extra file in a deliverable directory has already cost this
#:   project one task (polyglot-c-py). ``compile()`` on the file's bytes is
#:   the same parse with no artifact and no write of any kind — and it
#:   cannot fail for want of a writable cache directory, which a
#:   ``py_compile`` invocation in a read-only tree could.
#: - **JSON redirects stdout.** ``json.tool`` pretty-prints the document it
#:   just validated; without the redirect a valid 10KB JSON file would come
#:   back as 10KB of noise on a check that passed. ``json.tool`` is also
#:   strict RFC 8259, which is why :data:`_JSONC_BASENAME_GLOBS` exists —
#:   see there.
#: - **YAML composes, it does not load.** ``yaml.safe_load`` is a *load*
#:   (parse + compose-a-single-document + construct), and both of the extra
#:   phases reject files that are valid YAML: a multi-document stream
#:   (``---``-separated Kubernetes manifests, Ansible playbooks) fails
#:   composition with "expected a single document in the stream", and a
#:   custom tag (``!Ref``/``!GetAtt`` in every CloudFormation template)
#:   fails construction with "could not determine a constructor". Those are
#:   properties of the loader, not of YAML syntax, and reporting either
#:   would be exactly the false positive on correct work this module cannot
#:   afford. ``compose_all`` parses and builds the node graph without
#:   resolving tags and without the one-document rule, which is precisely a
#:   syntax check; ``list(...)`` is what forces the lazy generator to
#:   actually parse the whole stream.
#: - **The Python-hosted checks set ``sys.tracebacklimit = 0``.** Without
#:   it, a one-line YAML error arrives behind ~1200 characters of PyYAML
#:   stack frames, and :data:`CHECK_OUTPUT_LIMIT` then truncates away the
#:   very diagnostic the check exists to deliver. Set *before* the
#:   ``import`` on purpose: a missing module then also reports as one clean
#:   ``ModuleNotFoundError`` line, which is what the "unavailable" triage
#:   below reads.
SYNTAX_CHECKERS: Final[dict[str, SyntaxChecker]] = {
    ".py": SyntaxChecker(
        language="python",
        template=(
            "python3 -c "
            "'import sys;sys.tracebacklimit=0;"
            "compile(open(sys.argv[1],\"rb\").read(),sys.argv[1],\"exec\")'"
            " {path}"
        ),
    ),
    ".sh": SyntaxChecker(language="bash", template="bash -n {path}"),
    ".bash": SyntaxChecker(language="bash", template="bash -n {path}"),
    ".json": SyntaxChecker(
        language="json", template="python3 -m json.tool {path} > /dev/null"
    ),
    ".js": SyntaxChecker(language="javascript", template="node --check {path}"),
    ".mjs": SyntaxChecker(language="javascript", template="node --check {path}"),
    ".cjs": SyntaxChecker(language="javascript", template="node --check {path}"),
    ".yaml": SyntaxChecker(
        language="yaml",
        template=(
            "python3 -c "
            "'import sys;sys.tracebacklimit=0;import yaml;"
            "list(yaml.compose_all(open(sys.argv[1])))' {path}"
        ),
    ),
    ".yml": SyntaxChecker(
        language="yaml",
        template=(
            "python3 -c "
            "'import sys;sys.tracebacklimit=0;import yaml;"
            "list(yaml.compose_all(open(sys.argv[1])))' {path}"
        ),
    ),
}

#: ``.json`` basenames (glob-matched, case-insensitively) whose consumers
#: accept JSON-with-comments, and which therefore get **no** check.
#:
#: ``python3 -m json.tool`` implements strict RFC 8259, but the most common
#: ``.json`` filenames in a real coding task are JSONC files whose actual
#: consumers — ``tsc``, VS Code, the devcontainer CLI, ESLint — accept
#: ``//`` and ``/* */``. A ``// target`` comment in a hand-written
#: ``tsconfig.json`` is correct work, and "Syntax check failed" on it is a
#: false positive. The exemption is by filename rather than by sniffing the
#: content because the decision has to stay pure (see
#: :func:`syntax_check_command`) and because the filename *is* the contract:
#: these names are read by tools that document comment support.
_JSONC_BASENAME_GLOBS: Final[tuple[str, ...]] = (
    "tsconfig*.json",
    "jsconfig*.json",
    "devcontainer.json",
    ".eslintrc.json",
)

#: Directories whose ``.json`` files are JSONC by convention, whatever they
#: are named — VS Code's own ``settings.json``/``launch.json``/``tasks.json``
#: all take comments, and so does anything the devcontainer CLI reads.
_JSONC_DIRECTORIES: Final[frozenset[str]] = frozenset({".vscode", ".devcontainer"})

#: Exit codes that mean "the shell could not run the checker at all":
#: 127 (command not found) and 126 (found but not executable). Neither says
#: anything about the file, so neither is ever reported.
UNAVAILABLE_EXIT_CODES: Final[frozenset[int]] = frozenset({126, 127})

#: Substrings (matched case-insensitively) that mark a non-zero exit as
#: "checker unavailable" rather than "file is broken". ``no module named``
#: is why PyYAML being absent is silence and not a yaml error; the rest
#: cover a missing interpreter, a missing or unreadable file, and a shell
#: that found the binary but could not execute it.
_UNAVAILABLE_MARKERS: Final[tuple[str, ...]] = (
    "not found",
    "no such file",
    "no module named",
    "cannot execute",
    "permission denied",
    "importerror",
    "modulenotfounderror",
)

#: Substrings that mark a non-zero exit as *inconclusive*: the checker ran,
#: but its complaint is about which dialect it assumed rather than about the
#: file. ``node --check`` parses a bare ``.js`` file as CommonJS, so a
#: perfectly valid ES module comes back as "Cannot use import statement
#: outside a module". Reporting that would be a false positive on a correct
#: file, which is the one outcome this whole module cannot afford.
_INCONCLUSIVE_MARKERS: Final[tuple[str, ...]] = (
    "cannot use import statement outside a module",
    "await is only valid",
    "unexpected token 'export'",
    "unexpected token export",
)

#: What a *verdict line* looks like: a bare name, dotted name, or path at
#: column 0, followed by ``": "``.
#:
#: This is the guard that keeps the markers above from being matched against
#: **the file's own contents**. Every checker here that finds a syntax error
#: echoes the offending source line back — Python renders
#:
#: .. code-block:: text
#:
#:        File "solve.py", line 3
#:          print("error: input file not found)
#:                ^
#:      SyntaxError: unterminated string literal (detected at line 3)
#:
#: and ``node --check`` does the same. Lowercasing that whole blob and
#: asking "does it contain ``not found``" hands the decision to whatever the
#: agent happened to write: a genuine syntax error in any error-handling
#: code — which agents write constantly — was silently swallowed *and*
#: logged as ``skipped_reason="unavailable"``, so the telemetry reported a
#: working interpreter as a missing one. Marker matching is therefore
#: restricted to the checker's own words (see :func:`_verdict_lines`).
#:
#: The shape is deliberately narrow. Every "checker unavailable" message is
#: an exception header (``ModuleNotFoundError: ...``,
#: ``FileNotFoundError: ...``) or a shell error (``sh: node: not found``,
#: ``/bin/sh: 1: node: Permission denied``), all of which are a plain
#: name/path plus ``": "`` at column 0.
#:
#: Shape alone is **not** sufficient, and assuming it was is what let this
#: bug survive its first fix. Echoed source is not reliably indented: CPython
#: indents its echo four spaces, but ``node --check`` prints the offending
#: line at column 0, and this pattern is a *prefix* match — a JS property
#: line such as ``error: "not found: bad input,`` matches on ``error: ``,
#: long before any of the punctuation real code carries appears. So shape is
#: only the first of two filters; the second is
#: :data:`_CARET_RULER_RE`, applied in :func:`_verdict_lines`.
_VERDICT_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_./-][A-Za-z0-9_./ -]*: "
)

#: A *caret ruler*: the ``^^^^`` / ``~~~~`` line a checker prints underneath
#: the source line it just echoed, to point at the column that broke.
#:
#: This is the indentation-independent half of the guard. Every checker here
#: that echoes source underlines it immediately afterwards — CPython prints
#: ``          ^``, ``node --check`` prints ``       ^^^^^^`` — so "the line
#: directly above a ruler came out of the file, not out of the checker" holds
#: whatever column the echo starts in. Matched against the *whole* line, so a
#: diagnostic that merely contains a caret is never mistaken for a ruler, and
#: a ruler drawn with anything else falls through to being kept — the
#: quiet-but-safe direction (see :func:`_verdict_lines`).
_CARET_RULER_RE: Final[re.Pattern[str]] = re.compile(r"^\s*[\^~]+\s*$")


def _suffix(path: str) -> str:
    """The lower-cased suffix of ``path`` (``""`` when it has none)."""
    return PurePosixPath(path).suffix.lower()


def _is_comment_tolerant_json(path: str) -> bool:
    """Whether ``path`` is a ``.json`` file that is JSONC by contract.

    See :data:`_JSONC_BASENAME_GLOBS`. Pure and case-insensitive; matched on
    the basename and on the path's directory components, never on content.
    """
    if _suffix(path) != ".json":
        return False
    parts = PurePosixPath(path).parts
    if any(part.lower() in _JSONC_DIRECTORIES for part in parts[:-1]):
        return True
    # `fnmatchcase` against an already-lowered name, so the answer does not
    # depend on the host OS's filename case rules the way `fnmatch` does.
    name = PurePosixPath(path).name.lower()
    return any(fnmatchcase(name, glob) for glob in _JSONC_BASENAME_GLOBS)


def _checker_for(path: str) -> SyntaxChecker | None:
    """The checker that applies to ``path``, or ``None`` if none does.

    The single place that answers "is this file checked at all", so the
    command and the language label can never disagree about it.
    """
    checker = SYNTAX_CHECKERS.get(_suffix(path))
    if checker is None:
        return None
    if _is_comment_tolerant_json(path):
        return None
    return checker


def syntax_check_command(path: str) -> str | None:
    """The shell command that syntax-checks ``path``, or ``None``.

    Pure: no sandbox, no clock, no I/O. ``None`` means no checker applies —
    either no checker is registered for the file's suffix (the overwhelmingly
    common case: ``.txt``, ``.md``, ``.c``, no suffix at all, and the reason
    silence has to be the default), or the path is one of the JSONC names
    exempted by :data:`_JSONC_BASENAME_GLOBS`. ``path`` is quoted with
    :func:`shlex.quote`, so spaces and shell metacharacters in a filename
    cannot alter the command.
    """
    checker = _checker_for(path)
    if checker is None:
        return None
    return checker.template.format(path=shlex.quote(path))


def syntax_check_language(path: str) -> str | None:
    """The language label for ``path``'s checker, or ``None`` if unchecked."""
    checker = _checker_for(path)
    return checker.language if checker is not None else None


def truncate_check_output(output: str) -> str:
    """Truncate checker output to :data:`CHECK_OUTPUT_LIMIT` characters.

    Keeps the *head* and appends a marker naming how much was dropped. The
    opposite of :func:`~harness.diligence.truncate_verification_output`,
    deliberately: a test runner summarizes at the end, but a parser reports
    the first syntax error first and everything after it is cascade.
    """
    if len(output) <= CHECK_OUTPUT_LIMIT:
        return output
    dropped = len(output) - CHECK_OUTPUT_LIMIT
    return output[:CHECK_OUTPUT_LIMIT] + f"\n[...{dropped} chars truncated...]"


@dataclass(frozen=True)
class SyntaxCheckOutcome:
    """What one post-write check decided, and why.

    ``ok`` is ``True`` when the checker exited 0, ``False`` when a genuine
    diagnostic was produced (the only case that reaches the model), and
    ``None`` when there is no verdict — the check was skipped, unavailable,
    or inconclusive, with ``skipped_reason`` naming which.
    """

    path: str
    language: str | None
    ok: bool | None
    exit_code: int | None
    skipped_reason: str | None
    output: str = ""

    def as_payload(self, tool: str | None = None) -> dict:
        """This outcome as a :data:`SYNTAX_CHECK_EVENT` payload dict."""
        return {
            "path": self.path,
            "language": self.language,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "skipped_reason": self.skipped_reason,
            "tool": tool,
        }


def format_syntax_failure(outcome: SyntaxCheckOutcome) -> str:
    """Render a failed check as text to append to a tool result.

    Three things this wording has to get right:

    - It **names the harness as the author**. The model did not run this
      command; text that reads like its own tool output would have it
      debugging a command it never issued.
    - It **contains no promise pattern and never ends in ``?``**, so
      quoting it back cannot trip
      :func:`~harness.diligence.looks_unfinished` (see that module's
      ``_PROMISE_PATTERNS``). That is also why the closing sentence follows
      the checker's output rather than preceding it: the appended block
      ends on the harness's own words, never on whatever punctuation a
      compiler happened to emit.
    - It **names the partial-write case**. `write_file` is documented to
      support writing a large file across several calls, and piece 1 of a
      Python file is legitimately unparseable. Left unsaid, the model could
      "fix" an incomplete file by closing it early.
    """
    language = outcome.language or "syntax"
    return (
        f"Syntax check failed (harness-run): the harness ran a {language} "
        f"syntax check on {outcome.path} after this write, and it exited "
        f"{outcome.exit_code}. This check is the harness's own, not a "
        f"command you chose.\n"
        f"--- {language} syntax check output ---\n"
        f"{outcome.output}\n"
        f"--- end of harness syntax check ---\n"
        f"If this file is still being written in pieces, ignore this and "
        f"finish the file; otherwise fix the syntax error above."
    )


def _verdict_lines(output: str) -> list[str]:
    """The lines of ``output`` that are the *checker's own* conclusion.

    Everything a checker prints falls into two classes: its own words, and
    material it copied out of the file under test. Only the first class may
    decide whether a check counts, so this drops the second — see
    :data:`_VERDICT_LINE_RE` for the shape and for what goes wrong without
    it.

    A line is the checker's own when it *looks* like a verdict and is not
    *positioned* like an echo. Both halves are needed: shape alone readmits
    ``node --check``'s column-0 echo (see :data:`_VERDICT_LINE_RE`), and
    position alone would drop the many diagnostics that carry no ruler.

    - **Position.** Any line immediately above a :data:`_CARET_RULER_RE`
      line is the file's own contents and is dropped outright, whatever it
      looks like.
    - **Shape.** Of what survives, keep every line matching
      :data:`_VERDICT_LINE_RE`, plus the last non-empty line. Every
      Python-hosted checker here sets ``sys.tracebacklimit = 0``, which makes
      its one-line exception the last thing printed, and ``bash -n`` /
      ``json.tool`` print a single line and nothing else.

    Both rules err towards keeping, because a line wrongly kept can only make
    the harness *quieter* while a line wrongly dropped can make it report a
    false failure on correct work.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    # Blank lines are already gone, so the ruler sits at exactly i-1's heels.
    echoed = {
        i - 1 for i, line in enumerate(lines) if i > 0 and _CARET_RULER_RE.match(line)
    }
    kept = [
        line
        for i, line in enumerate(lines[:-1])
        if i not in echoed and _VERDICT_LINE_RE.match(line)
    ]
    if len(lines) - 1 not in echoed:
        kept.append(lines[-1])
    return kept


def _verdict(exit_code: int, output: str) -> str | None:
    """Why a non-zero exit is *not* reportable, or ``None`` if it is.

    Fail-open triage, in one place. Returns ``"unavailable"``,
    ``"inconclusive"``, ``"no_diagnostics"``, or ``None`` when the exit is a
    genuine syntax failure worth telling the model about.
    """
    if exit_code in UNAVAILABLE_EXIT_CODES:
        return "unavailable"
    if not output.strip():
        # Non-zero with nothing to say. There is no diagnostic to hand the
        # model, so "your file is broken, no further detail" is all the
        # message could contain — worse than silence.
        return "no_diagnostics"
    # Deliberately not `output.lower()`: the markers are statements about
    # the checker, so they are matched only against what the checker said,
    # never against the source line it echoed back.
    lowered = "\n".join(_verdict_lines(output)).lower()
    if any(marker in lowered for marker in _UNAVAILABLE_MARKERS):
        return "unavailable"
    if any(marker in lowered for marker in _INCONCLUSIVE_MARKERS):
        return "inconclusive"
    return None


def _skip_for_deadline(deadline: Deadline | None) -> str | None:
    """Why the deadline forbids running a check now, or ``None``.

    Same seam the ``bash`` tool uses (see
    :func:`harness.tools.builtin.bash_tool`): the landing turn is explicit
    state, and :meth:`~harness.deadline.Deadline.affordable_exec_seconds` is
    the one place that knows what an exec can cost without eating the
    landing reserve. A harness-initiated check the model never asked for is
    the last thing that should spend that reserve, so the bar is the full
    :data:`CHECK_TIMEOUT_SECONDS` rather than whatever fraction happens to
    be left.
    """
    if deadline is None:
        return None
    if deadline.landing:
        return "landing"
    affordable = deadline.affordable_exec_seconds()
    if affordable is not None and affordable < CHECK_TIMEOUT_SECONDS:
        return "deadline"
    return None


def _emit(
    store: RunStore | None,
    agent_id: str | None,
    outcome: SyntaxCheckOutcome,
    tool: str | None,
) -> None:
    """Append a :data:`SYNTAX_CHECK_EVENT`, swallowing any failure.

    Telemetry for an unrequested check must not be able to turn a
    successful `write_file` into an error result, so a store that raises is
    absorbed here — the same fail-open rule the check itself follows.
    """
    if store is None or agent_id is None:
        return
    try:
        store.append_event(agent_id, SYNTAX_CHECK_EVENT, outcome.as_payload(tool))
    except Exception:  # pragma: no cover - defensive; store failures are rare
        return


async def run_syntax_check(
    sandbox: Sandbox,
    path: str,
    *,
    deadline: Deadline | None = None,
    store: RunStore | None = None,
    agent_id: str | None = None,
    tool: str | None = None,
    skip_reason: str | None = None,
) -> str | None:
    """Syntax-check ``path`` in ``sandbox``; return text to append, or ``None``.

    ``None`` — append nothing — is the answer for every outcome except one:
    the checker exited non-zero *and* produced a diagnostic that is not a
    "checker missing" or dialect-ambiguity artifact. Clean files, unknown
    suffixes, missing interpreters, timeouts, scarce wall-clock, and an
    ``exec`` that raised all return ``None`` silently.

    ``skip_reason``, when given, short-circuits the check before it runs
    (still emitting the event): the caller's own reason not to check, e.g.
    an ``append``-mode write, which is a partial file by construction and
    would fail a parse for reasons that are not defects.

    ``deadline``/``store``/``agent_id`` follow the ``bash`` tool's seam —
    the shared run deadline gates the check, and the pair of store and
    agent id records a :data:`SYNTAX_CHECK_EVENT` on that agent's stream.
    All are optional; with none of them the check still runs, it just goes
    unrecorded.

    This function never raises. A caller can await it directly in a tool
    handler after a successful write without a guard.
    """
    language = syntax_check_language(path)
    command = syntax_check_command(path)
    if command is None or language is None:
        # No checker for this suffix. Deliberately *no* event: `write_file`
        # is dominated by `.txt`/`.md`/no-suffix paths, and a row per one
        # would bury the rows that carry a verdict. The denominator is
        # recoverable from the `tool_call` events, which carry the path.
        return None

    reason = skip_reason or _skip_for_deadline(deadline)
    if reason is not None:
        _emit(
            store,
            agent_id,
            SyntaxCheckOutcome(
                path=path,
                language=language,
                ok=None,
                exit_code=None,
                skipped_reason=reason,
            ),
            tool,
        )
        return None

    try:
        result = await sandbox.exec(command, timeout=CHECK_TIMEOUT_SECONDS)
    except Exception:
        # Fail open. Whatever went wrong with the harness's own check, the
        # model's write succeeded and that is what the tool result says.
        _emit(
            store,
            agent_id,
            SyntaxCheckOutcome(
                path=path,
                language=language,
                ok=None,
                exit_code=None,
                skipped_reason="exec_error",
            ),
            tool,
        )
        return None

    if result.timed_out:
        _emit(
            store,
            agent_id,
            SyntaxCheckOutcome(
                path=path,
                language=language,
                ok=None,
                exit_code=None,
                skipped_reason="timeout",
            ),
            tool,
        )
        return None

    if result.exit_code == 0:
        # Silence is the success signal: nothing is appended, and the model
        # never learns the check happened.
        _emit(
            store,
            agent_id,
            SyntaxCheckOutcome(
                path=path,
                language=language,
                ok=True,
                exit_code=0,
                skipped_reason=None,
            ),
            tool,
        )
        return None

    # stderr first: every checker here reports diagnostics there, and the
    # stdout half exists only for the ones that also print to it.
    output = "\n".join(part for part in (result.stderr, result.stdout) if part)
    unreportable = _verdict(result.exit_code, output)
    if unreportable is not None:
        _emit(
            store,
            agent_id,
            SyntaxCheckOutcome(
                path=path,
                language=language,
                ok=None,
                exit_code=result.exit_code,
                skipped_reason=unreportable,
            ),
            tool,
        )
        return None

    outcome = SyntaxCheckOutcome(
        path=path,
        language=language,
        ok=False,
        exit_code=result.exit_code,
        skipped_reason=None,
        output=truncate_check_output(output.strip()),
    )
    _emit(store, agent_id, outcome, tool)
    return format_syntax_failure(outcome)
