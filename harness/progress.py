"""Is this run going anywhere? (S-107)

Today the only "is this going anywhere" signal is
:func:`~harness.diligence.looks_unfinished`, which fires *after* the model
stops calling tools — by which point the budget is gone. This looks at the
turns as they happen.

Deterministic and model-free, like `diligence`. Which detectors exist was
decided by measuring candidates against 727 recorded Terminal-Bench trials
carrying the verifier's own verdict, and the measurement had to be done twice:

**Lift must be against a length-matched base rate.** Long runs fail more (32.7%
under 20 calls, 77.8% over 80), so any detector that needs a long run to fire
inherits that as apparent skill. The null "detector" `len(calls) >= 25` — no
logic at all — scores 1.31x against the global base rate. Every number below is
therefore quoted against the base rate of runs *long enough for that detector
to fire*, and the audit prints the null baselines beside them so the confound
cannot be read as signal again.

============================ ====== ========= ============ ===========
detector                      fires precision global lift  matched lift
============================ ====== ========= ============ ===========
REPEATED_CALL                    16     75.0%        1.65x        1.31x
CONSECUTIVE_FAILURE              34     64.7%        1.43x        1.22x
[null] len >= 25                230     59.6%        1.31x        1.00x
*no progress in 25 calls*       141     60.3%        1.33x        1.01x
============================ ====== ========= ============ ===========

The last row is why there is no `NO_PROGRESS` detector: at a matched base rate
it is indistinguishable from the null. It shipped once on the strength of its
global lift and was removed when the control was added.

`CONSECUTIVE_FAILURE` shipped *rejected*, on a measurement that keyed failure
off `ToolResult.is_error`. That flag means the tool itself failed; a command
exiting non-zero is a perfectly good tool result. Across the corpus:
`is_error` is set 8 times, a non-zero exit appears 299 times, and the two never
coincide. The rejection was made by an instrument that could not see the thing
it was rejecting.

Re-derive everything with ``harness progress-audit``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "StuckSignal",
    "Detector",
    "ProgressMonitor",
    "STUCK_EVENT",
    "REPEATED_CALL_THRESHOLD",
    "CONSECUTIVE_FAILURE_THRESHOLD",
    "MIN_SCOREABLE_CALLS",
    "NULL_BASELINES",
    "command_head",
    "result_failed",
    "audit",
]

#: Emitted at most once per detector per run.
STUCK_EVENT = "stuck_signal"

#: Identical `(tool, arguments)` seen this many times. Four, not three: at
#: three the precision is 54.8% against a 45.4% base rate, which is barely a
#: signal; at four it is 75.0%. Tuned on the recorded corpus, not derived.
REPEATED_CALL_THRESHOLD = 4

#: Consecutive non-zero exits from the same command. Three: at four the fire
#: count drops to 11 and the matched lift with it (1.10x), which is most of the
#: way to nothing.
CONSECUTIVE_FAILURE_THRESHOLD = 3

#: Trials below this are excluded from the audit: no detector can fire on
#: them. Disclosed in the output rather than applied silently -- they are 84%
#: failures, so dropping them moves the base rate the whole table is quoted
#: against (51.2% over everything, 45.4% over what survives).
MIN_SCOREABLE_CALLS = 5

#: Length-only "detectors" printed beside the real ones. `len(calls) >= 25`
#: scores 1.31x against the global base rate with no logic whatsoever.
NULL_BASELINES = (25, 50)


class Detector(str, Enum):
    """Which rule tripped. A `str` enum so it lands in an event payload."""

    #: The same call, arguments and all, four times over.
    REPEATED_CALL = "repeated_call"
    #: The same command failing three times running.
    CONSECUTIVE_FAILURE = "consecutive_failure"


@dataclass(frozen=True)
class StuckSignal:
    """One detector tripping, with what tripped it."""

    detector: Detector
    #: Human-readable evidence -- the repeated call, or the length of the
    #: barren stretch. Goes in the event payload so a later reader does not
    #: have to re-derive why this fired.
    evidence: str
    #: Tool calls seen when it fired, so position within the run is
    #: recoverable without joining against the transcript.
    at_call: int

    def payload(self) -> dict:
        return {
            "spec": "S-107",
            "detector": self.detector.value,
            "evidence": self.evidence,
            "at_call": self.at_call,
        }


def command_head(arguments: dict) -> str | None:
    """The command's identity for repeat-failure purposes: its first token.

    First token, not the whole line: `pytest -q tests/a.py` failing then
    `pytest -q tests/b.py` failing is the same thing going wrong twice. Keying
    on the full command was how this detector was originally measured, and it
    took the fire count from 34 to 4 -- which is what made it look refuted.
    """
    command = arguments.get("command")
    if not isinstance(command, str):
        return None
    tokens = command.split()
    return tokens[0] if tokens else None


def result_failed(content: str, is_error: bool) -> bool:
    """Whether a tool result represents a command that did not work.

    `is_error` alone is not it. That flag means the *tool* failed; a command
    exiting non-zero is a perfectly good tool result carrying bad news. Across
    727 recorded trials `is_error` is set 8 times, a non-zero exit appears 299
    times, and the two never coincide -- so a detector keyed on `is_error`
    cannot see a failing command at all.
    """
    if is_error:
        return True
    return "exit code: " in content and "exit code: 0" not in content


@dataclass
class ProgressMonitor:
    """Per-agent detector state, fed calls and their results.

    O(1) per call: a counter keyed by call, and two integers. Not a walk of
    the transcript, which would be slowest on exactly the long runs this is
    for.
    """

    #: How many times each `(tool, arguments)` has been seen.
    _seen: Counter = field(default_factory=Counter, repr=False)
    #: Total calls observed, for `at_call`.
    _calls: int = 0
    #: Consecutive failures of `_failing_head`.
    _failures: int = 0
    #: The command head those failures belong to.
    _failing_head: str | None = None
    #: Call ids still awaiting a result, so a result can be attributed to the
    #: command that produced it. Bounded by one turn's parallel calls.
    _pending: dict = field(default_factory=dict, repr=False)
    #: Detectors that have already fired. A signal that re-fires every turn
    #: once tripped makes "how often does this happen" unanswerable from the
    #: log -- one long run would swamp the count.
    _fired: set = field(default_factory=set, repr=False)

    def observe_call(
        self, call_id: str, name: str, arguments: dict
    ) -> StuckSignal | None:
        """Record one tool call; return a signal the first time one trips."""
        self._calls += 1
        if name == "bash":
            head = command_head(arguments)
            if head is not None:
                self._pending[call_id] = head

        key = _call_key(name, arguments)
        self._seen[key] += 1
        repeats = self._seen[key]
        if (
            repeats >= REPEATED_CALL_THRESHOLD
            and Detector.REPEATED_CALL not in self._fired
        ):
            self._fired.add(Detector.REPEATED_CALL)
            return StuckSignal(
                detector=Detector.REPEATED_CALL,
                evidence=f"{name} called {repeats} times with identical arguments",
                at_call=self._calls,
            )
        return None

    def observe_result(
        self, call_id: str, content: str, is_error: bool
    ) -> StuckSignal | None:
        """Record one tool result; return a signal the first time one trips."""
        head = self._pending.pop(call_id, None)
        if head is None:
            return None  # not a bash call, or a result we never saw called
        if not result_failed(content, is_error):
            self._failures, self._failing_head = 0, None
            return None
        if head == self._failing_head:
            self._failures += 1
        else:
            self._failures, self._failing_head = 1, head
        if (
            self._failures >= CONSECUTIVE_FAILURE_THRESHOLD
            and Detector.CONSECUTIVE_FAILURE not in self._fired
        ):
            self._fired.add(Detector.CONSECUTIVE_FAILURE)
            return StuckSignal(
                detector=Detector.CONSECUTIVE_FAILURE,
                evidence=f"{head!r} failed {self._failures} times running",
                at_call=self._calls,
            )
        return None


def _call_key(name: str, arguments: dict) -> str:
    """A stable identity for one call.

    `sort_keys` so two dicts differing only in insertion order are the same
    call -- providers do not promise argument order, and a detector that
    depended on it would fire or not depending on the provider.

    Unserialisable arguments fall back to `repr`: this is a detector, and
    failing to key a call is a missed signal, but raising here would take the
    run down for telemetry.
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, default=repr)
    except Exception:  # noqa: BLE001 - see docstring
        rendered = repr(arguments)
    return f"{name}\x00{rendered}"


# ---------------------------------------------------------------------------
# The audit that produced the numbers above (S-107 acceptance 3)
# ---------------------------------------------------------------------------


def audit(pattern: str) -> str:
    """Score the detectors against recorded trials that carry a verdict.

    Two things this reports that the first version did not, both of which
    changed which detectors exist:

    **A length-matched base rate.** Long runs fail more, so a detector that
    needs a long run to fire inherits that as apparent skill. Lift against the
    global base rate made `no_progress` look like a 1.33x signal when against
    runs it could fire on it was 1.01x -- indistinguishable from nothing.

    **Null baselines.** `len(calls) >= N` has no detector logic whatsoever and
    scores 1.31x globally. Printing it beside the real detectors is what makes
    the confound impossible to miss.
    """
    import glob
    import json as _json
    import os
    import sqlite3

    trials: list[tuple[bool, list]] = []
    dropped_short = 0
    dropped_short_failed = 0
    for db in sorted(glob.glob(pattern, recursive=True)):
        marker = "/agent/"
        if marker not in db:
            continue
        trial_dir = db[: db.index(marker)]
        try:
            result = _json.load(open(os.path.join(trial_dir, "result.json")))
        except Exception:
            continue
        reward = (
            (result.get("verifier_result") or {}).get("rewards") or {}
        ).get("reward")
        if reward is None:
            continue
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT kind, payload FROM transcript_events "
                "WHERE kind IN ('tool_call','tool_result') ORDER BY seq"
            ).fetchall()
        except Exception:
            continue
        results: dict = {}
        calls: list = []
        for row in rows:
            try:
                p_ = _json.loads(row["payload"])
            except Exception:
                continue
            if row["kind"] == "tool_call":
                calls.append(
                    (p_.get("id"), p_.get("name") or "", p_.get("arguments") or {})
                )
            else:
                results[p_.get("tool_call_id")] = (
                    p_.get("content") or "",
                    bool(p_.get("is_error")),
                )
        ok = reward > 0
        if len(calls) < MIN_SCOREABLE_CALLS:
            # Excluded because no detector can fire on them -- and disclosed,
            # because they are 84% failures and dropping them silently moves
            # the base rate the whole table is quoted against.
            dropped_short += 1
            dropped_short_failed += 0 if ok else 1
            continue
        trials.append((ok, [(cid, n, a, results.get(cid)) for cid, n, a in calls]))

    if not trials:
        return (
            f"no scoreable trials matched {pattern!r}. A trial needs a "
            "result.json with a verifier reward and a state.db with at least "
            f"{MIN_SCOREABLE_CALLS} tool calls; without both there is nothing "
            "to score against."
        )

    failed = sum(1 for ok, _ in trials if not ok)
    base = failed / len(trials)
    if not failed:
        # A clean sweep is the one input that has nothing to score: lift is
        # precision over a base rate of zero. Saying so beats dividing by it.
        return (
            f"trials scored : {len(trials)}\n"
            "failed        : 0\n\n"
            "Every scored trial passed, so there is no failure for a detector "
            "to predict and no base rate to compute lift against. This is not "
            "a result about the detectors."
        )
    lines = [
        f"trials scored : {len(trials)}",
        f"failed        : {failed}  ({base:.1%} base rate)",
        f"excluded      : {dropped_short} trials under "
        f"{MIN_SCOREABLE_CALLS} tool calls, of which {dropped_short_failed} "
        "failed -- no detector can fire on them, and they are dropped before "
        "the base rate above is computed",
        "",
        f"{'detector':22} {'fires':>5} {'prec':>7} {'global':>7} {'matched':>8}",
        f"{'':22} {'':>5} {'':>7} {'lift':>7} {'lift':>8}",
    ]

    def score(label: str, first_fire) -> str:
        hits, lengths = [], []
        for ok, calls in trials:
            at = first_fire(calls)
            lengths.append((ok, len(calls), at))
            if at is not None:
                hits.append(ok)
        if not hits:
            return f"{label:22} {0:>5}   never fires"
        precision = sum(1 for ok in hits if not ok) / len(hits)
        shortest = min(n for _ok, n, at in lengths if at is not None)
        comparable = [ok for ok, n, _at in lengths if n >= shortest]
        matched = sum(1 for ok in comparable if not ok) / len(comparable)
        matched_lift = (
            f"{precision / matched:>7.2f}x" if matched else f"{'n/a':>8}"
        )
        return (
            f"{label:22} {len(hits):>5} {precision:>6.1%} "
            f"{precision / base:>6.2f}x {matched_lift}"
        )

    def monitor_fire(detector: "Detector"):
        def first(calls):
            monitor = ProgressMonitor()
            for call_id, name, arguments, result in calls:
                signal = monitor.observe_call(call_id, name, arguments)
                if signal is not None and signal.detector is detector:
                    return True
                if result is not None:
                    signal = monitor.observe_result(call_id, *result)
                    if signal is not None and signal.detector is detector:
                        return True
            return None
        return first

    for detector in Detector:
        lines.append(score(detector.value, monitor_fire(detector)))

    lines.append("")
    for threshold in NULL_BASELINES:
        lines.append(
            score(
                f"[null] len >= {threshold}",
                lambda calls, t=threshold: True if len(calls) >= t else None,
            )
        )
    lines.append("")
    lines.append(
        "The [null] rows have no detector logic at all -- they fire on length "
        "alone. A real detector has to beat them on the matched column, which "
        "is by construction 1.00x for them. Global lift is mostly the "
        "length confound: long runs fail more."
    )
    return "\n".join(lines)
