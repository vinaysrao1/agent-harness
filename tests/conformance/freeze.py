"""Regenerate the neutrality goldens (S-002).

    python -m tests.conformance.freeze

Regenerating is a **deliberate act**. If an invariant fails, the question is
whether the change to the benchmark path was intended -- not how to make the
test green. A re-freeze belongs in a Lane B spec with a REFREEZE block, a TB2
run, and a CHANGELOG.md entry naming that run id.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from harness.persistence import RunStore
from tests.conformance.fixture import (
    canonical_tool_surface,
    coding_assembled_system,
    coding_base_prompt,
    coding_control_flow_digest_input,
    coding_trailing_reminder,
    coding_tool_specs,
    fixture_skills,
    sha256_text,
)

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def compute() -> dict[str, str]:
    """Every golden, computed over the fixed fixture."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        skills = fixture_skills(tmp_path)
        store = RunStore(tmp_path / "state.db")
        return {
            "coding_base_prompt.sha256": sha256_text(coding_base_prompt(skills)),
            "coding_assembled_system.sha256": sha256_text(
                coding_assembled_system(skills)
            ),
            "coding_tools_subagent.sha256": sha256_text(
                canonical_tool_surface(
                    coding_tool_specs(tmp_path, store, lead=False)
                )
            ),
            "coding_tools_lead.sha256": sha256_text(
                canonical_tool_surface(coding_tool_specs(tmp_path, store, lead=True))
            ),
            "coding_trailing_reminder.sha256": sha256_text(
                coding_trailing_reminder(skills)
            ),
            "coding_control_flow.sha256": sha256_text(
                coding_control_flow_digest_input()
            ),
        }


def main(argv: list[str] | None = None) -> None:
    """Regenerate goldens, printing every change.

    Takes optional golden names so a Lane B re-freeze can move exactly the
    invariant its REFREEZE block names. Regenerating all five for a change
    that affects one hides unrelated drift under an approved change -- the
    ledger records a reason per invariant, so the tool should too.

    Only files whose digest actually changed are rewritten, and each change is
    printed old -> new so it is visible in the terminal as well as in the diff.
    """
    import sys

    names = argv if argv is not None else sys.argv[1:]
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    digests = compute()
    if names:
        unknown = sorted(set(names) - set(digests))
        if unknown:
            raise SystemExit(
                f"unknown golden(s): {', '.join(unknown)}\n"
                f"known: {', '.join(sorted(digests))}"
            )
        digests = {k: v for k, v in digests.items() if k in names}

    changed = 0
    for name, digest in digests.items():
        path = GOLDEN_DIR / name
        previous = path.read_text(encoding="utf-8").strip() if path.exists() else None
        if previous == digest:
            print(f"  unchanged  {name}")
            continue
        path.write_text(digest + "\n", encoding="utf-8")
        changed += 1
        print(f"  RE-FROZEN  {name}\n    {previous} -> {digest}")
    if changed:
        print(
            f"\n{changed} golden(s) moved. Add a row to tests/golden/CHANGELOG.md "
            "naming the spec, the invariant, and the qualifying TB2 run id."
        )


if __name__ == "__main__":
    main()
