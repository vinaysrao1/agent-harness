"""Structured search: `glob` and `grep` as tools (S-101).

Today an agent looking for a symbol runs ``bash("grep -rn ... | head -50")``.
That works, and it costs a shell round-trip, an unpredictable amount of output,
and a permission decision on a general-purpose execution tool for what is a
read. It also fails in the way shell pipelines fail: `head` closes the pipe,
`grep` dies of SIGPIPE, and the exit status the harness records describes the
last stage rather than the search.

Two tools instead, with one property that the shell version cannot have:
**the result volume is bounded before the work happens, not after.** A query
matching 500 files stops at the limit; it does not produce 500 matches and then
throw most of them away. Truncation-after-the-fact is what makes a large search
expensive in both time and context, and it is the failure the acceptance
criteria single out.

``rg`` is used when it is present and a self-contained Python fallback runs when
it is not. Nothing here ever installs anything: an attached sandbox belongs to
whoever attached it, and mutating it to make a search faster would be the
harness deciding to modify a benchmark's environment.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

__all__ = [
    "OUTPUT_MODES",
    "DEFAULT_HEAD_LIMIT",
    "MAX_HEAD_LIMIT",
    "SearchRequest",
    "ripgrep_command",
    "fallback_program",
    "render_results",
    "describe_failure",
    "MAX_LINE_CHARS",
]

#: What a search can return. ``files_with_matches`` is the default because it
#: is the cheapest useful answer: the agent almost always wants to know *where*
#: before it wants to read anything, and returning content by default spends
#: context on files it is about to discard.
OUTPUT_MODES = ("files_with_matches", "content", "count")

#: Applied when the caller names no limit. Chosen to be smaller than a
#: comfortable read: a search returning more than this is a search that should
#: have been narrower, and saying so early is more useful than paging.
DEFAULT_HEAD_LIMIT = 50

#: A caller cannot raise the limit past this. The bound exists to stop a search
#: from consuming the context window; letting the model opt out of it would
#: make it advisory, and the whole point is that it is not.
MAX_HEAD_LIMIT = 500


@dataclass(frozen=True)
class SearchRequest:
    """One structured search, validated."""

    pattern: str
    path: str | None = None
    glob: str | None = None
    type: str | None = None
    output_mode: str = "files_with_matches"
    context: int = 0
    head_limit: int = DEFAULT_HEAD_LIMIT
    case_insensitive: bool = False

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern must not be empty")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {', '.join(OUTPUT_MODES)}; "
                f"got {self.output_mode!r}"
            )
        if self.context < 0:
            raise ValueError("context must not be negative")
        if self.head_limit < 1:
            raise ValueError("head_limit must be at least 1")

    @property
    def effective_limit(self) -> int:
        return min(self.head_limit, MAX_HEAD_LIMIT)


def ripgrep_command(request: SearchRequest) -> str:
    """The ``rg`` invocation for ``request``.

    ``--max-count`` and ``-m`` bound the work *inside* ripgrep rather than
    piping through ``head``: a pipeline computes every match and discards the
    tail, and its exit status describes ``head`` rather than the search.
    """
    args = ["rg", "--no-messages", "--color=never"]

    if request.output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif request.output_mode == "count":
        args.append("--count-matches")
    else:
        args += ["--line-number", "--with-filename"]
        if request.context:
            args += ["--context", str(request.context)]

    if request.case_insensitive:
        args.append("--ignore-case")
    if request.glob:
        args += ["--glob", request.glob]
    if request.type:
        args += ["--type", request.type]

    # Bound the output at the source. `--max-count` caps matches per file so a
    # single enormous file cannot fill the whole budget on its own.
    args += ["--max-count", str(request.effective_limit)]
    args += ["--regexp", request.pattern]
    args.append(request.path or ".")

    # No `| head`. The pipeline was the point of the original complaint about
    # `bash("grep … | head")` and it reintroduced the same defect: the shell
    # reports the *last* stage's status, so ripgrep's exit code -- 1 for "no
    # matches", 2 for "bad pattern", "unknown --type", "unreadable path" --
    # was unreachable, and every one of those became "no matches" in the
    # model's context. It also bought nothing: for `files_with_matches` rg must
    # walk the whole tree regardless, so `head` capped transfer, never work.
    # The count bound now lives in `render_results`, and the *work* bound is
    # `--max-count` here and the early `break` in the fallback.
    return " ".join(shlex.quote(arg) for arg in args)


#: The fallback, as a self-contained program run through ``python3 -``.
#:
#: A single exec that does the whole walk, rather than the harness listing files
#: and reading each one back: a repository has thousands of files and a
#: round-trip each would take longer than the search is worth.
#:
#: It reads its arguments from a JSON blob on stdin rather than from ``argv``,
#: because a regex reaching the shell through ``argv`` has to survive two
#: levels of quoting, and getting that wrong turns a search into a syntax
#: error -- or worse, into a different search that silently succeeds.
_FALLBACK_SOURCE = r'''
import fnmatch, json, os, re, sys

spec = json.loads(sys.stdin.read())
try:
    pattern = re.compile(spec["pattern"], re.IGNORECASE if spec["ignore_case"] else 0)
except re.error as exc:
    # A traceback is not an actionable message. Name the cause on stderr and
    # exit 2, matching ripgrep's convention for a bad pattern, so both engines
    # report an invalid regex the same way.
    sys.stderr.write(f"invalid regular expression: {exc}\n")
    raise SystemExit(2)
root = spec["path"] or "."
limit = spec["limit"]
mode = spec["mode"]
context = spec["context"]
name_glob = spec["glob"]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", "target",
             "dist", "build", ".tox", ".next"}

def walk():
    # A `path` naming a single file yields nothing from os.walk, so the most
    # natural narrowing an agent makes after a files_with_matches search --
    # "now search just that file" -- returned "no matches" every time.
    if os.path.isfile(root):
        yield root
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            full = os.path.join(base, name)
            if name_glob and not (
                fnmatch.fnmatch(name, name_glob) or fnmatch.fnmatch(full, name_glob)
            ):
                continue
            yield full

emitted = 0
counts = {}
for path in walk():
    if emitted >= limit:
        break
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        # Binary or unreadable: skipped, exactly as ripgrep skips it.
        continue

    hits = [i for i, line in enumerate(lines) if pattern.search(line)]
    if not hits:
        continue

    if mode == "files_with_matches":
        print(path)
        emitted += 1
    elif mode == "count":
        # Matches, not matching *lines*: `rg --count-matches` counts matches,
        # and a file with three hits on one line reported 1 here and 3 there.
        counts[path] = sum(len(pattern.findall(line)) for line in lines)
        emitted += 1
    else:
        for index in hits:
            if emitted >= limit:
                break
            low = max(0, index - context)
            high = min(len(lines), index + context + 1)
            for offset in range(low, high):
                marker = ":" if offset == index else "-"
                print(f"{path}{marker}{offset + 1}{marker}{lines[offset]}")
            emitted += 1

if mode == "count":
    for path, total in counts.items():
        print(f"{path}:{total}")
'''


def fallback_program(request: SearchRequest) -> tuple[str, str]:
    """``(command, stdin)`` running the pure-Python search in the sandbox."""
    import json

    spec = json.dumps(
        {
            "pattern": request.pattern,
            "path": request.path,
            "glob": request.glob,
            "limit": request.effective_limit,
            "mode": request.output_mode,
            "context": request.context,
            "ignore_case": request.case_insensitive,
        }
    )
    program = shlex.quote(_FALLBACK_SOURCE)
    return f"printf %s {shlex.quote(spec)} | python3 -c {program}", spec


#: One result line is elided past this. A minified file is a single 280,000
#: character line, and rendering it whole produced a 100,000-character tool
#: result under a stated bound of one -- the count was bounded and the
#: *volume*, which is what actually costs context, was not.
MAX_LINE_CHARS = 400


def _elide(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    keep = MAX_LINE_CHARS // 2 - 3
    return f"{line[:keep]} … {line[-keep:]}"


def _is_match_line(line: str, mode: str) -> bool:
    """Whether ``line`` is a match rather than a context line.

    Both engines render context as ``path-N-text`` and matches as
    ``path:N:text``. Counting raw lines instead made ``context=2`` consume
    three times the budget per match: the engines stopped after ``limit``
    *matches* while the renderer cut after ``limit`` *lines*, so a stated bound
    of 10 delivered three and a third — and could end on a bare context line,
    which reads as a match on the wrong text.
    """
    if mode != "content":
        return True
    head, sep, _ = line.partition(":")
    return bool(sep) and head.rsplit("-", 1)[-1] != head or ":" in line[len(head) + 1 :]


def render_results(request: SearchRequest, raw: str, *, engine: str) -> str:
    """Format search output, saying plainly when the limit bound the result.

    A truncated result that does not announce itself is the worst outcome: the
    model concludes the symbol appears in three files when it appears in three
    hundred, and narrows its search on a false premise.
    """
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return f"no matches for {request.pattern!r} ({engine})"

    limit = request.effective_limit
    kept: list[str] = []
    matches = 0
    for line in lines:
        if _is_match_line(line, request.output_mode):
            if matches >= limit:
                break
            matches += 1
        kept.append(_elide(line))

    body = "\n".join(kept)
    if matches >= limit:
        body += (
            f"\n\n[stopped at {limit} results — this is a bound, not the total. "
            "Narrow with `path`, `glob`, or a more specific pattern.]"
        )
    return body


def describe_failure(
    request: SearchRequest, *, engine: str, exit_code: int, stderr: str, timed_out: bool
) -> str | None:
    """The error to report, or ``None`` when the search genuinely ran.

    Exists because every failure previously arrived as ``no matches``: an
    invalid regex, an unknown ``--type``, a missing ``python3``, an unreadable
    path, a catastrophic-backtracking timeout. All of them produced empty
    stdout, and empty stdout was read as "the repository does not contain
    this" — a confident wrong answer the model then acts on.

    ``rg`` exits 1 for "no matches" and 2 for an error; ``find`` and the Python
    fallback exit 0 on success. 127 is the shell's "command not found", which
    is how an image without ``python3`` presents.
    """
    if timed_out:
        return (
            f"search timed out ({engine}). The pattern may be backtracking — "
            "anchor it, or narrow with `path` or `glob`."
        )
    if exit_code == 127:
        return (
            f"the {engine} backend is not available in this sandbox "
            f"(exit 127). {stderr.strip()[:200]}"
        )
    if engine == "rg" and exit_code in (0, 1):
        return None
    if engine != "rg" and exit_code == 0:
        return None
    detail = stderr.strip()[:400] or f"exit code {exit_code}"
    return f"search failed ({engine}): {detail}"
