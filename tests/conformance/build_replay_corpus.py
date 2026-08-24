"""Build the frozen replay corpus from real runs (S-003).

    python -m tests.conformance.build_replay_corpus jobs/ -o tests/golden/replay_corpus.jsonl

The corpus stores **shapes, not content**: per message, its role, its exact
character counts, and whether it carries a tool result. Two reasons, and the
first is the load-bearing one:

1. **It is sufficient.** Every decision N7 and N8 pin -- when pruning fires,
   which indices it sheds, when compaction fires, tokens per turn -- depends on
   message roles, ages, tool-result-ness and token counts. The default
   estimator is ``chars / 4`` (``harness/adapters/base.py``), so filler of the
   same length reproduces the counts exactly. Nothing in the decision path
   reads what the text says.
2. **It is publishable.** Real transcripts contain agent solutions to
   benchmark tasks. Committing those to a public repository risks
   contaminating the benchmark for everyone, which is why ``jobs/`` is
   gitignored. A frozen artifact has to be committed to be frozen, so it must
   not carry the content.

The limit that follows: this corpus pins *timing*, not content-dependent
behavior. A future condenser that decides what to keep by reading the text
would not be pinned by it. Stated here rather than discovered later.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from pathlib import Path

#: How many trials to keep. The plan asks for 20-30.
CORPUS_SIZE = 25


class SpecCorpusError(RuntimeError):
    """A trial could not be shaped faithfully; better to fail than to guess."""


def _trial_shapes(db: str) -> list[dict] | None:
    """Message shapes for one trial, in transcript order."""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except Exception:
        return None
    shapes: list[dict] = []
    try:
        # Both kinds, in seq order. Tool results are persisted under their own
        # `tool_result` kind, not inside `message` -- reading only `message`
        # yields a transcript with no tool-role entries at all, which silently
        # makes pruning unfireable (it only ever sheds tool results) and would
        # have left N7 pinning a mechanism that structurally could not run.
        # This mirrors Orchestrator._replay_lead_transcript, which replays both
        # for the same reason.
        for kind, payload in con.execute(
            "SELECT kind, payload FROM transcript_events "
            "WHERE kind IN ('message', 'tool_result') ORDER BY seq"
        ):
            try:
                event = json.loads(payload)
            except Exception:
                # Dropping a message would shorten every later index and
                # silently produce a wrong-shaped trial.
                raise SpecCorpusError(f"{db}: unparseable {kind} payload")
            if kind == "tool_result":
                shapes.append(
                    {
                        "role": "tool",
                        "chars": 0,
                        "calls": [],
                        "result_chars": len(event.get("content") or ""),
                        "is_error": bool(event.get("is_error")),
                    }
                )
                continue
            result = event.get("tool_result") or None
            shapes.append(
                {
                    "role": event.get("role"),
                    "chars": len(event.get("content") or ""),
                    # What harness.adapters.base.ModelAdapter.count_tokens
                    # actually counts for a call: len(name) + len(repr(args)).
                    # An earlier version stored len(json.dumps(args)), which
                    # omits the name and diverges from repr on non-ASCII -- so
                    # the "filler reproduces the counts exactly" claim the whole
                    # shape-only design rests on was not true.
                    "calls": [
                        len(call.get("name") or "") + len(repr(call.get("arguments") or {}))
                        for call in (event.get("tool_calls") or [])
                    ],
                    "result_chars": len((result or {}).get("content") or "") if result else None,
                    "is_error": bool((result or {}).get("is_error")) if result else False,
                }
            )
    except Exception:
        return None
    finally:
        con.close()
    return shapes or None


def _weight(shapes: list[dict]) -> int:
    """Total characters -- the axis the thresholds are expressed in."""
    return sum(
        s["chars"] + sum(s["calls"]) + (s["result_chars"] or 0) for s in shapes
    )


def build(jobs_root: str, size: int = CORPUS_SIZE) -> list[dict]:
    """Select ``size`` trials: the heaviest, plus a spread over the rest.

    Heaviest first because they are the ones nearest the thresholds and so the
    ones whose timing a change is most likely to move; a spread after, so the
    corpus is not exclusively pathological.
    """
    records: list[dict] = []
    pattern = os.path.join(jobs_root, "**", "agent", "harness-home", "state.db")
    for db in sorted(set(glob.glob(pattern, recursive=True))):
        shapes = _trial_shapes(db)
        if not shapes:
            continue
        parts = Path(db).parts
        trial = parts[parts.index("agent") - 1] if "agent" in parts else Path(db).stem
        records.append(
            {"trial": trial, "task": trial.rsplit("__", 1)[0], "messages": shapes}
        )

    records.sort(key=lambda r: _weight(r["messages"]), reverse=True)
    heavy = records[: size // 2]
    rest = records[size // 2 :]
    stride = max(1, len(rest) // max(1, size - len(heavy)))
    spread = rest[::stride][: size - len(heavy)]
    chosen = heavy + spread
    # Deterministic order in the file, independent of selection order.
    return sorted(chosen, key=lambda r: r["trial"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_root")
    parser.add_argument("-o", "--out", default="tests/golden/replay_corpus.jsonl")
    parser.add_argument("-n", "--size", type=int, default=CORPUS_SIZE)
    args = parser.parse_args()

    records = build(args.jobs_root, args.size)
    if not records:
        raise SystemExit(f"no trials found under {args.jobs_root}")
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    weights = sorted(_weight(r["messages"]) for r in records)
    print(f"trials     : {len(records)}")
    print(f"messages   : {sum(len(r['messages']) for r in records)}")
    print(f"chars      : min {weights[0]:,}  median {weights[len(weights)//2]:,}  max {weights[-1]:,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
