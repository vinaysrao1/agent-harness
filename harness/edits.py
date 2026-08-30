"""Edit diagnostics and multi-edit application (S-103).

A rejected edit costs a turn. On a wall clock that is expensive, and the
information needed to fix it is already in the file the harness just read — so
telling the model only "old_string not found in file" throws away everything
it needs and invites a blind retry.

Two things live here:

**Naming the mismatch.** The overwhelmingly common failure is not "that text
isn't there" but "that text is there with different whitespace". Indentation
width, trailing spaces, tabs against spaces, CRLF against LF: each is a
one-line fix if the model is told which one it is, and a guessing game if it is
handed a bare not-found. The classifier names the specific difference before
falling back to a character diff.

**Applying several edits atomically.** ``multi_edit`` exists so a
three-hunk change costs one turn instead of three, and so a file is never left
half-edited: every edit is applied in memory and the file is written once, or
not at all.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

__all__ = [
    "Candidate",
    "MismatchKind",
    "classify_mismatch",
    "nearest_candidates",
    "describe_mismatch",
    "apply_edits",
    "EditError",
    "DIAGNOSTIC_BUDGET",
]

#: Hard ceiling on a mismatch diagnostic, in characters. Bounded *before*
#: assembly rather than truncated after: a diagnostic cut mid-diff is worse
#: than a shorter one that ends on a whole line, and an unbounded one competes
#: with the file content for the context it is supposed to help read.
DIAGNOSTIC_BUDGET = 1200

#: How many near-misses to show. Three is the point past which the model is
#: choosing between candidates rather than reading a correction.
MAX_CANDIDATES = 3

#: Below this similarity a "candidate" is noise, and offering it would imply a
#: relationship that is not there.
_MIN_RATIO = 0.55

#: Cost ceilings. The first version scored every window of the file with a
#: full ``SequenceMatcher.ratio`` — quadratic in the block size, once per
#: line. Measured: **424 seconds** on a 5,000-line file with a 40-line
#: ``old_string``, and 200 seconds on a single 60,000-character minified line.
#: Both on the benchmark path, both synchronous on the event loop with no
#: ``await`` for the deadline to fire through, and both producing a diagnostic
#: of exactly one sentence, because every candidate exceeded the budget.
#: Maximum cost, minimum information, on a path that had already cost the
#: model a turn.
#:
#: What bounds it now is the shape of the search, not a size limit: anchor by
#: dictionary lookup, rank the few anchors with a linear ``quick_ratio``, and
#: pay the quadratic ``ratio`` only for the ``MAX_CANDIDATES`` survivors, over
#: a sampled prefix. 400,000 lines costs 0.06s.
#:
#: File-size and old_string-size caps were written here first and then
#: removed: measured with them disabled, the timings did not move, so they
#: were guards that never fired dressed as safety. The bound below is the one
#: doing the work.
MAX_ANCHORS = 40
_RATIO_SAMPLE_CHARS = 4_000

#: Rendering caps, so one enormous candidate cannot consume the whole budget
#: and leave nothing to show. Without these a minified file produced a header
#: and no content at all.
MAX_CANDIDATE_LINES = 12
MAX_LINE_CHARS = 160


class EditError(ValueError):
    """An edit could not be applied. Message is addressed to the model."""


@dataclass(frozen=True)
class Candidate:
    """A stretch of the file that nearly matches what was asked for."""

    line: int          # 1-based line where the candidate starts
    text: str
    ratio: float


class MismatchKind:
    """Named reasons an exact match failed, most actionable first."""

    INDENT_WIDTH = "the indentation width differs"
    TABS_VS_SPACES = "the file uses tabs where old_string uses spaces (or vice versa)"
    TRAILING_WHITESPACE = "there is trailing whitespace in the file"
    FILE_IS_CRLF = "the file uses CRLF line endings and old_string uses LF"
    OLD_IS_CRLF = "old_string uses CRLF line endings and the file uses LF"
    WHITESPACE_ONLY = "only whitespace differs"
    CONTENT = "the text itself differs"


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def classify_mismatch(wanted: str, found: str) -> str:
    """Name the difference between what was asked for and what is there.

    Ordered most-specific first: an indent-width difference is also a
    whitespace difference, and saying the specific thing is the entire value.

    The line-ending branch names **which side** has CRLF. An earlier version
    had a single constant asserting the file was the CRLF one, which was
    exactly backwards half the time — and a model acting on it would convert
    its ``old_string`` to the line endings it already had.
    """
    if wanted == found:
        # Reachable: `describe_mismatch` compares against a rendered window, so
        # a caller can land here with equal strings. Nothing to name.
        return MismatchKind.CONTENT
    if wanted.replace("\r\n", "\n") == found.replace("\r\n", "\n"):
        return (
            MismatchKind.FILE_IS_CRLF
            if "\r\n" in found
            else MismatchKind.OLD_IS_CRLF
        )

    wanted_lines = wanted.splitlines()
    found_lines = found.splitlines()
    if len(wanted_lines) == len(found_lines):
        if all(w.rstrip() == f.rstrip() for w, f in zip(wanted_lines, found_lines)):
            if any(f != f.rstrip() for f in found_lines):
                return MismatchKind.TRAILING_WHITESPACE
        stripped_equal = all(
            w.strip() == f.strip() for w, f in zip(wanted_lines, found_lines)
        )
        if stripped_equal:
            wl = [_leading_ws(w) for w in wanted_lines]
            fl = [_leading_ws(f) for f in found_lines]
            if any("\t" in x for x in wl) != any("\t" in x for x in fl):
                return MismatchKind.TABS_VS_SPACES
            if wl != fl:
                return MismatchKind.INDENT_WIDTH
            return MismatchKind.WHITESPACE_ONLY

    if "".join(wanted.split()) == "".join(found.split()):
        return MismatchKind.WHITESPACE_ONLY
    return MismatchKind.CONTENT


def _anchor_indices(lines: list[str], first_line: str) -> list[int]:
    """Line indices where a candidate could plausibly start.

    Anchoring on the *stripped* first line is what makes the search cheap and
    is also exactly right for the cases that matter: indent width, tabs, and
    trailing whitespace all vanish under ``strip()``, so the whitespace
    mismatches this diagnostic exists to name anchor for free by dictionary
    lookup. Only when nothing matches exactly does it pay for a fuzzy pass,
    and that pass runs over *unique* stripped lines rather than every line.
    """
    target = first_line.strip()
    if not target:
        return []

    exact: list[int] = []
    unique: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == target:
            exact.append(index)
            if len(exact) >= MAX_ANCHORS:
                return exact
        elif stripped and stripped not in unique:
            unique[stripped] = index
    if exact:
        return exact

    close = difflib.get_close_matches(target, list(unique), n=MAX_ANCHORS, cutoff=0.7)
    return sorted(unique[text] for text in close)


def nearest_candidates(
    content: str, old_string: str, limit: int = MAX_CANDIDATES
) -> list[Candidate]:
    """The closest stretches of ``content`` to ``old_string``.

    Three stages, cheapest first, because this runs on a path that has already
    cost the model a turn and must not cost it minutes of wall clock as well:
    anchoring (dictionary lookup), ranking (``quick_ratio``, linear), and
    scoring (``ratio``, quadratic) on at most ``limit`` survivors. That last
    bound is what makes the whole search cheap — 400,000 lines costs 0.06s —
    and it is why there is no file-size cap here.

    Windows keep their line terminators, so a CRLF file yields a CRLF
    candidate and :func:`classify_mismatch` can see it. Splitting on
    ``splitlines()`` and rejoining with ``"\n"`` — which the first version did
    — silently normalised every window to LF, so the line-ending branch could
    never fire on a real file and CRLF mismatches were reported as content
    differences at "100% similar".
    """
    if not content or not old_string:
        return []
    raw = content.splitlines(keepends=True)
    if not raw:
        return []

    wanted_lines = old_string.splitlines() or [old_string]
    span = len(wanted_lines)
    keeps_terminator = old_string.endswith(("\n", "\r"))

    def window_at(start: int) -> str:
        text = "".join(raw[start : start + span])
        return text if keeps_terminator else text.rstrip("\r\n")

    anchors = _anchor_indices(raw, wanted_lines[0])
    if not anchors:
        return []

    sample = old_string[:_RATIO_SAMPLE_CHARS]
    ranked: list[tuple[float, int, str]] = []
    for start in anchors:
        window = window_at(start)
        matcher = difflib.SequenceMatcher(
            None, sample, window[:_RATIO_SAMPLE_CHARS], autojunk=False
        )
        quick = matcher.quick_ratio()
        if quick >= _MIN_RATIO:
            ranked.append((quick, start, window))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    scored: list[Candidate] = []
    for _, start, window in ranked[:limit]:
        matcher = difflib.SequenceMatcher(
            None, sample, window[:_RATIO_SAMPLE_CHARS], autojunk=False
        )
        ratio = matcher.ratio()
        if ratio >= _MIN_RATIO:
            scored.append(Candidate(line=start + 1, text=window, ratio=ratio))
    scored.sort(key=lambda c: (-c.ratio, c.line))
    return scored


def describe_mismatch(content: str, old_string: str) -> str:
    """A bounded diagnostic naming why ``old_string`` did not match.

    Empty when nothing is close enough to be worth showing — silence beats
    pointing at unrelated code and implying it was meant.
    """
    candidates = nearest_candidates(content, old_string)
    if not candidates:
        return ""

    best = candidates[0]
    header = f"Nearest match is at line {best.line}, where {classify_mismatch(old_string, best.text)}."
    parts = [header]

    # -1 per part for the newline `join` will add, so the returned string
    # honours the budget rather than exceeding it by the separator count.
    budget = DIAGNOSTIC_BUDGET - len(header)
    for candidate in candidates:
        block = _render_candidate(candidate, old_string)
        if len(block) + 1 > budget:
            break
        parts.append(block)
        budget -= len(block) + 1
    return "\n".join(parts)


def _elide(text: str) -> str:
    """One line, bounded. A minified file is a single enormous line, and
    rendering it whole consumed the entire budget and left nothing to show."""
    if len(text) <= MAX_LINE_CHARS:
        return text
    keep = MAX_LINE_CHARS // 2 - 2
    return f"{text[:keep]} … {text[-keep:]}"


def _render_candidate(candidate: Candidate, wanted: str) -> str:
    """One candidate, with whitespace made visible where it is the difference."""
    kind = classify_mismatch(wanted, candidate.text)
    visible = kind in (
        MismatchKind.INDENT_WIDTH,
        MismatchKind.TABS_VS_SPACES,
        MismatchKind.TRAILING_WHITESPACE,
        MismatchKind.WHITESPACE_ONLY,
        MismatchKind.FILE_IS_CRLF,
        MismatchKind.OLD_IS_CRLF,
    )
    lines = candidate.text.splitlines()
    elided = len(lines) - MAX_CANDIDATE_LINES
    shown_lines = lines[:MAX_CANDIDATE_LINES]

    rendered = []
    for offset, text in enumerate(shown_lines):
        text = _elide(text)
        if visible:
            text = text.replace("\t", "\\t").replace(" ", "·")
        rendered.append(f"  {candidate.line + offset:>5} | {text}")
    if elided > 0:
        rendered.append(f"        | … {elided} more line(s)")

    header = f"--- line {candidate.line} ({kind}, {candidate.ratio:.0%} similar)"
    if visible:
        header += "   [· = space, \\t = tab]"
    return "\n".join([header, *rendered])


def apply_edits(content: str, edits: list[dict]) -> str:
    """Apply ``edits`` in order, each seeing the previous one's result.

    Raises :class:`EditError` naming the failing edit's index without having
    modified anything: the caller holds the original string, so a failure part
    way through leaves the file on disk untouched by construction. That is what
    makes ``multi_edit`` atomic -- there is no partially-written state to roll
    back, because nothing is written until every edit has succeeded.
    """
    if not edits:
        raise EditError("edits must not be empty")

    current = content
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise EditError(f"edit {index} is not an object")
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise EditError(
                f"edit {index} needs string 'old_string' and 'new_string'"
            )
        if old == "":
            raise EditError(f"edit {index}: old_string must be non-empty")
        replace_all = bool(edit.get("replace_all", False))

        count = current.count(old)
        if count == 0:
            detail = describe_mismatch(current, old)
            message = f"edit {index}: old_string not found in file"
            if index:
                message += (
                    f" (edits 0-{index - 1} applied cleanly to the in-memory copy; "
                    "nothing was written)"
                )
            raise EditError(f"{message}\n{detail}" if detail else message)
        if count > 1 and not replace_all:
            message = (
                f"edit {index}: old_string is not unique in file ({count} "
                "occurrences); pass replace_all=true for this edit, or include "
                "more surrounding context"
            )
            if index:
                # Without this the two remedies can both be wrong: if edit 0
                # produced the duplicate, `replace_all` would also clobber the
                # pre-existing copy, and no amount of "surrounding context"
                # disambiguates text the model did not know would exist. The
                # not-found branch already says this; the asymmetry was an
                # oversight, not a decision.
                message += (
                    f". Note edits 0-{index - 1} already applied to the "
                    "in-memory copy, so one of the occurrences may be their "
                    "output. Nothing was written."
                )
            raise EditError(message)
        current = current.replace(old, new, -1 if replace_all else 1)
    return current
