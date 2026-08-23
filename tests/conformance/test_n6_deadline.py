"""S-002 / N6: deadline determinism.

The plan's N6 row reads "Already exists — promote it to the conformance suite
so it is run as a neutrality gate, not just a unit test." That is exactly what
this does: `tests/test_deadline.py` already holds the frozen `exec_capped`
corpus mined from real Terminal-Bench runs, and it already passes. What it
lacked was standing as a *gate* — something the conformance run consults
before a change to the benchmark path is considered neutral.

Deferring N6 to S-003 alongside N7/N8 would have been wrong: N7 and N8 need a
replay corpus that does not exist yet, but N6's corpus has been in the tree
since round 3. This spec can and therefore should discharge it.

Running the corpus a second time here is cheap (the whole deadline module is
well under a second) and keeps the conformance suite self-contained: `pytest
tests/conformance/` is the complete neutrality gate, not a subset of it that a
reader has to remember to supplement.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: N6 is worded as "the existing exec_capped corpus **and every deadline.py
#: decision test**", so the whole module is the faithful scope. An earlier
#: draft named three classes explicitly and two of the three names did not
#: exist -- the collection guard below is what caught it, and it stays for
#: exactly that reason.
N6_TESTS = ("tests/test_deadline.py",)


def test_S002_n6_deadline_decisions_are_unchanged() -> None:
    """Every frozen deadline decision still holds.

    Spawns a subprocess rather than importing: these are parametrized classes
    whose collection belongs to pytest, and re-implementing that here would be
    a second, driftable copy of the corpus runner.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", *N6_TESTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "N6 failed: a deadline decision changed. The exec-cap corpus is mined "
        "from real Terminal-Bench runs, so a change here is a change to how "
        "the harness spends its wall clock on the benchmark path.\n"
        f"{result.stdout[-3000:]}"
    )


def test_S002_n6_named_tests_all_exist() -> None:
    """The named node ids must actually collect.

    A renamed class would otherwise make N6 pass by running nothing -- pytest
    exits 0 on a selection that matches no tests only in some configurations,
    and relying on that distinction is exactly the kind of silent no-op this
    suite exists to prevent.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", *N6_TESTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert "tests collected" in result.stdout or "test collected" in result.stdout
    collected = [
        line for line in result.stdout.splitlines() if "::" in line
    ]
    assert len(collected) > 50, (
        f"N6 collected only {len(collected)} tests; the frozen corpus alone "
        "is dozens of rows, so this selection is not reaching it"
    )
