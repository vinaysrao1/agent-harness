"""S-103: edit ergonomics — naming the mismatch, and atomic multi-edit.

A rejected edit costs a turn, and on a wall clock that is expensive. The
information needed to fix it is already in the file the harness just read, so
answering "old_string not found in file" and nothing else throws that away and
invites a blind retry with the same whitespace.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from harness.edits import (
    DIAGNOSTIC_BUDGET,
    EditError,
    MismatchKind,
    apply_edits,
    classify_mismatch,
    describe_mismatch,
    nearest_candidates,
)
from harness.reads import ReadLedger
from harness.sandbox.base import SandboxError, apply_edit
from harness.sandbox.local import LocalSandbox
from harness.tools.builtin import multi_edit_tool


class TestTheMismatchIsNamed:
    """Acceptance (1): whitespace-only and indent-width mismatches are always
    identified as such *by name*. "Not found" sends the model looking for
    absent text when the text is right there with four spaces instead of two."""

    def test_S103_indent_width_is_named(self) -> None:
        assert classify_mismatch("    return 1", "        return 1") == (
            MismatchKind.INDENT_WIDTH
        )

    def test_S103_tabs_versus_spaces_is_named(self) -> None:
        assert classify_mismatch("    x = 1", "\tx = 1") == MismatchKind.TABS_VS_SPACES

    def test_S103_trailing_whitespace_is_named(self) -> None:
        assert classify_mismatch("x = 1", "x = 1   ") == (
            MismatchKind.TRAILING_WHITESPACE
        )

    def test_S103_line_endings_are_named_through_the_real_path(self) -> None:
        # Through `describe_mismatch`, not `classify_mismatch` directly. The
        # first version of this test called the classifier with a `found`
        # containing CRLF -- a value the real path could never produce,
        # because candidate windows were rebuilt with "\n".join(splitlines()),
        # which strips \r. So the branch was dead, CRLF files were reported as
        # "the text itself differs" at 100% similar, and this test passed
        # throughout. The archetype, inside the spec that cites it.
        crlf_file = "def f():\r\n    return 1\r\n"
        detail = describe_mismatch(crlf_file, "def f():\n    return 1")
        assert MismatchKind.FILE_IS_CRLF in detail, detail

    def test_S103_the_line_ending_direction_is_not_reversed(self) -> None:
        # A single constant asserting "the file uses CRLF" is wrong half the
        # time, and a model acting on it converts old_string to the endings it
        # already had.
        lf_file = "def f():\n    return 1\n"
        detail = describe_mismatch(lf_file, "def f():\r\n    return 1")
        assert MismatchKind.OLD_IS_CRLF in detail, detail

    def test_S103_a_crlf_candidate_can_be_copied_back(self) -> None:
        # The rendered block must be usable. When windows were LF-normalised,
        # copying the shown text into a retry failed again -- a confident
        # wrong answer, worse than a bare "not found".
        crlf_file = "alpha = 1\r\nbeta = 2\r\n"
        found = nearest_candidates(crlf_file, "alpha = 1\nbeta = 2")
        assert found and "\r\n" in found[0].text, found

    def test_S103_a_real_content_difference_is_not_blamed_on_whitespace(self) -> None:
        # The classifier must not reach for a whitespace explanation when the
        # text genuinely differs -- that would send the model to fix the wrong
        # thing, which is worse than saying nothing.
        assert classify_mismatch("return 1", "return 2") == MismatchKind.CONTENT

    def test_S103_whitespace_is_made_visible_only_when_it_is_the_difference(
        self,
    ) -> None:
        ws = describe_mismatch("def f():\n    return 1\n", "def f():\n\treturn 1\n")
        assert "·" in ws or "\\t" in ws, ws
        content = describe_mismatch("value = compute(a, b)\n", "value = compute(a, c)\n")
        assert "·" not in content, content


class TestDiagnosticsAreBounded:
    """Acceptance (4): bounded to a fixed character budget."""

    def test_S103_the_budget_actually_binds(self) -> None:
        # The candidates must be big enough that an unbounded assembly would
        # blow past the budget -- otherwise this asserts nothing. Each near
        # match here is ~600 characters, so three of them cannot fit.
        block = "\n".join(f"    value_{j} = compute(alpha, beta, gamma_{j})" for j in range(12))
        content = "\n\n".join(block.replace("compute", f"compute{k}") for k in range(4))
        wanted = block.replace("compute", "computeX")

        detail = describe_mismatch(content, wanted)
        assert detail, "precondition: there must be near matches to bound"
        assert len(detail) <= DIAGNOSTIC_BUDGET, len(detail)
        unbounded = sum(
            len(c.text) for c in nearest_candidates(content, wanted)
        )
        assert unbounded > DIAGNOSTIC_BUDGET, (
            "precondition: an unbounded diagnostic would exceed the budget, "
            "or this test cannot detect one"
        )

    def test_S103_an_oversized_candidate_still_shows_something(self) -> None:
        # Previously this degraded to a bare header: the first block exceeded
        # the budget, the loop broke, and minutes of CPU bought one sentence.
        # A candidate is now elided to fit rather than dropped.
        wide = "    payload = " + "z" * 4000
        content = f"def f():\n{wide}\n"
        detail = describe_mismatch(content, f"def f():\n\t{wide.strip()}\n")
        assert detail
        assert len(detail) <= DIAGNOSTIC_BUDGET, len(detail)
        assert "…" in detail, "an oversized candidate should elide, not vanish"
        assert detail.count("\n") >= 2, "a header alone is not a diagnostic"

    def test_S103_many_candidate_lines_are_elided_not_dropped(self) -> None:
        block = "\n".join(f"    step_{i}()" for i in range(40))
        content = f"def run():\n{block}\n"
        detail = describe_mismatch(content, f"def run():\n{block.replace('    ', '  ')}\n")
        assert "more line(s)" in detail, detail
        assert len(detail) <= DIAGNOSTIC_BUDGET

    def test_S103_the_budget_is_not_exceeded_by_the_joining_newlines(self) -> None:
        # The bound is on the returned string, separators included.
        block = "\n".join(f"    value_{j} = compute(alpha, beta)" for j in range(6))
        content = "\n\n".join(block.replace("compute", f"c{k}") for k in range(4))
        detail = describe_mismatch(content, block.replace("compute", "cX"))
        assert len(detail) <= DIAGNOSTIC_BUDGET, len(detail)

    def test_S103_a_huge_file_yields_a_bounded_diagnostic(self) -> None:
        content = "\n".join(f"line {i} of a very long file" for i in range(4000))
        detail = describe_mismatch(content, "line 12 of a very long file!!")
        assert len(detail) <= DIAGNOSTIC_BUDGET, len(detail)

    def test_S103_nothing_similar_yields_no_diagnostic(self) -> None:
        # Pointing at unrelated code and implying it was meant is worse than
        # saying nothing.
        assert describe_mismatch("alpha beta gamma\n", "zzzzzzzz qqqqqqqq") == ""

    def test_S103_candidates_are_ranked_and_capped(self) -> None:
        content = "\n".join(["x = 1", "x = 2", "x = 3", "x = 4", "x = 5"])
        found = nearest_candidates(content, "x = 9")
        assert 0 < len(found) <= 3
        assert found == sorted(found, key=lambda c: (-c.ratio, c.line))


class TestTheExistingErrorContractSurvives:
    def test_S103_the_message_still_opens_the_same_way(self) -> None:
        # A caller matching on the old text must keep matching: the diagnostic
        # is appended, never substituted.
        with pytest.raises(SandboxError, match="old_string not found in file"):
            apply_edit("def f():\n    return 1\n", "def g():\n    return 1\n", "x")

    def test_S103_the_uniqueness_error_is_unchanged(self) -> None:
        with pytest.raises(SandboxError, match=r"not unique in file \(2 occurrences\)"):
            apply_edit("a\na\n", "a", "b")

    def test_S103_a_successful_edit_is_unaffected(self) -> None:
        assert apply_edit("a\nb\n", "b", "c") == "a\nc\n"


class TestMultiEditIsAtomic:
    """Acceptance (2): a failing edit leaves the file byte-identical."""

    def test_S103_edits_apply_in_order_and_see_each_other(self) -> None:
        out = apply_edits("a = 1\n", [
            {"old_string": "a = 1", "new_string": "a = 2"},
            {"old_string": "a = 2", "new_string": "a = 3"},
        ])
        assert out == "a = 3\n"

    def test_S103_a_later_failure_discards_earlier_edits(self) -> None:
        with pytest.raises(EditError, match="edit 1"):
            apply_edits("a = 1\nb = 2\n", [
                {"old_string": "a = 1", "new_string": "a = 9"},
                {"old_string": "NOPE", "new_string": "x"},
            ])

    def test_S103_the_failure_says_nothing_was_written(self) -> None:
        with pytest.raises(EditError, match="nothing was written"):
            apply_edits("a = 1\n", [
                {"old_string": "a = 1", "new_string": "a = 9"},
                {"old_string": "NOPE", "new_string": "x"},
            ])

    def test_S103_a_non_unique_edit_names_its_index(self) -> None:
        with pytest.raises(EditError, match="edit 0: old_string is not unique"):
            apply_edits("a\na\n", [{"old_string": "a", "new_string": "b"}])

    def test_S103_replace_all_is_per_edit(self) -> None:
        assert apply_edits("a\na\n", [
            {"old_string": "a", "new_string": "b", "replace_all": True}
        ]) == "b\nb\n"

    def test_S103_an_empty_batch_is_rejected(self) -> None:
        with pytest.raises(EditError):
            apply_edits("x", [])

    async def test_S103_the_file_on_disk_is_untouched_when_a_batch_fails(
        self,
    ) -> None:
        directory = Path(tempfile.mkdtemp())
        target = directory / "m.py"
        target.write_text("a = 1\nb = 2\n")
        before = target.read_bytes()

        sandbox = LocalSandbox(directory)
        await sandbox.start()
        tool = multi_edit_tool(sandbox)
        with pytest.raises(ValueError, match="No changes were written"):
            await tool.handler({"path": "m.py", "edits": [
                {"old_string": "a = 1", "new_string": "a = 9"},
                {"old_string": "MISSING", "new_string": "x"},
            ]})
        assert target.read_bytes() == before

    async def test_S103_a_successful_batch_writes_once(self) -> None:
        directory = Path(tempfile.mkdtemp())
        (directory / "m.py").write_text("a = 1\nb = 2\nc = 3\n")
        sandbox = LocalSandbox(directory)
        await sandbox.start()
        tool = multi_edit_tool(sandbox)
        result = await tool.handler({"path": "m.py", "edits": [
            {"old_string": "a = 1", "new_string": "a = 10"},
            {"old_string": "c = 3", "new_string": "c = 30"},
        ]})
        assert "applied 2 edit(s)" in result
        assert (directory / "m.py").read_text() == "a = 10\nb = 2\nc = 30\n"


class TestOneCheckPerBatch:
    """Acceptance (3): one syntax check after the whole batch, not per edit.

    Per-edit checking would report a syntax error for every intermediate
    state -- which is expected, since a rename touching three call sites is
    broken after the first -- and would spend the deadline three times to say
    something that is not true of the result.
    """

    async def test_S103_syntax_is_checked_once_for_the_whole_batch(
        self, monkeypatch
    ) -> None:
        import harness.tools.builtin as builtin

        calls = []

        async def spy(sandbox, path, **kwargs):
            calls.append(kwargs.get("tool"))
            return None

        monkeypatch.setattr(builtin, "run_syntax_check", spy)
        directory = Path(tempfile.mkdtemp())
        (directory / "m.py").write_text("a = 1\nb = 2\nc = 3\n")
        sandbox = LocalSandbox(directory)
        await sandbox.start()
        await multi_edit_tool(sandbox).handler({"path": "m.py", "edits": [
            {"old_string": "a = 1", "new_string": "a = 10"},
            {"old_string": "b = 2", "new_string": "b = 20"},
            {"old_string": "c = 3", "new_string": "c = 30"},
        ]})
        assert calls == ["multi_edit"], calls

    async def test_S103_a_failed_batch_runs_no_check_at_all(
        self, monkeypatch
    ) -> None:
        import harness.tools.builtin as builtin

        calls = []

        async def spy(sandbox, path, **kwargs):
            calls.append(kwargs.get("tool"))
            return None

        monkeypatch.setattr(builtin, "run_syntax_check", spy)
        directory = Path(tempfile.mkdtemp())
        (directory / "m.py").write_text("a = 1\n")
        sandbox = LocalSandbox(directory)
        await sandbox.start()
        with pytest.raises(ValueError):
            await multi_edit_tool(sandbox).handler({"path": "m.py", "edits": [
                {"old_string": "MISSING", "new_string": "x"},
            ]})
        assert calls == [], "checked a file it never wrote"


class _FakeDeps:
    """The ToolDeps surface the builtin factories touch, and nothing else."""

    def __init__(self) -> None:
        import tempfile

        from harness.memory.store import MemoryStore
        from harness.persistence import RunStore
        from harness.skills import SkillLibrary

        directory = Path(tempfile.mkdtemp())
        self.sandbox = LocalSandbox(directory)
        self.deadline = None
        self.store = RunStore(directory / "state.db")
        self.agent_id = "agent"
        self.run_id = "run"
        self.memory = MemoryStore(directory / "memory")
        self.skills = SkillLibrary(directory / "skills")
        self.context = None
        self.written_data = lambda: None
        self.reads = ReadLedger()


class TestToolCountDiscipline:
    """Layer 1's binding constraint: `CODING` is capped at 15 tool specs.

    It is already at 15, so `multi_edit` ships in repo mode only. Promoting it
    would be a Lane B change requiring a TB2 run and the removal of an existing
    tool -- tool-surface growth degrades selection quality measurably on
    non-Anthropic models, and the scored model is one.
    """

    def test_S103_multi_edit_reaches_repo_mode_and_only_repo_mode(self) -> None:
        # By NAME, built through the real factories. Counting factories proved
        # nothing: `edit_file_tool` and `multi_edit_tool` have identical
        # signatures, so pointing the repo lambda at the wrong one leaves every
        # count assertion green while `multi_edit` ships to nobody.
        from harness.profiles import CODING, CODING_REPO

        def tool_names(profile):
            deps = _FakeDeps()
            return {factory(deps).spec.name for factory in profile.tool_factories}

        repo_names = tool_names(CODING_REPO)
        coding_names = tool_names(CODING)
        assert "multi_edit" in repo_names
        assert "multi_edit" not in coding_names
        assert coding_names < repo_names, "repo mode must be a superset"

    def test_S103_coding_stays_at_its_cap(self) -> None:
        # Counts the built specs, not the factory tuple. `len(factories) + 2`
        # hardcodes the lead-only count, so a third lead-only registration
        # would keep it green while the real total went to 16. The
        # authoritative guard is tests/conformance/test_neutrality.py, which
        # builds the registry; this is the S-103-local echo of it.
        from harness.profiles import CODING

        deps = _FakeDeps()
        specs = {factory(deps).spec.name for factory in CODING.tool_factories}
        assert len(specs) + 2 == 15, sorted(specs)


class TestTheDiagnosticCannotEatTheWallClock:
    """W6/D2. The first version scored every window of the file with a full
    quadratic ratio: 12s on a 2,000-line file and **424s** on a 5,000-line one,
    synchronously on the event loop, with no await point for the deadline to
    fire through — and producing a one-sentence diagnostic, because every
    candidate exceeded the budget. `edit_file` is on the benchmark path, so a
    single failed edit could consume a whole trial's wall clock.

    These are wall-clock assertions, which are ordinarily a smell. They are
    here because the defect was *only* observable as wall clock: every
    correctness test passed throughout.
    """

    def test_S103_a_realistic_failed_edit_is_fast(self) -> None:
        # One end-to-end smoke with a generous bound. The precise guards are
        # the two call-count tests below: they pin the same properties in
        # milliseconds and, unlike a timing test, they fail *fast* when the
        # property breaks instead of grinding for minutes first. That
        # distinction is not academic -- an earlier version of this class made
        # every mutation run unrunnable.
        import time

        content = "\n".join(
            f"    value_{i} = compute(alpha, beta, gamma_{i})" for i in range(5_000)
        )
        wanted = "\n".join(
            f"        value_{i} = compute(alpha, beta, gamma_{i})" for i in range(15)
        )
        started = time.monotonic()
        describe_mismatch(content, wanted)
        assert time.monotonic() - started < 2.0

    def test_S103_the_expensive_comparison_runs_a_bounded_number_of_times(
        self, monkeypatch
    ) -> None:
        # The bound that actually holds the cost down, asserted directly
        # rather than through wall clock. Size caps were tried here first and
        # deleted: with them disabled the timings did not move, so they were
        # guards that never fired.
        import difflib

        from harness.edits import MAX_CANDIDATES

        calls = {"n": 0}
        real = difflib.SequenceMatcher.ratio

        def counting(self):
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(difflib.SequenceMatcher, "ratio", counting)
        content = "\n".join(f"    do_thing(x_{i})" for i in range(5_000))
        nearest_candidates(content, "        do_thing(x_10)")
        assert calls["n"] <= MAX_CANDIDATES, (
            f"{calls['n']} quadratic comparisons for a 5,000-line file"
        )

    def test_S103_only_the_anchors_are_examined(self, monkeypatch) -> None:
        # Anchoring is what keeps the *linear* stage from running once per
        # line. Asserted on the number of windows examined rather than on wall
        # clock, so it cannot go quietly flaky on a loaded machine.
        import difflib

        # A literal for the same reason as above: a bound imported from the
        # module under test moves with it.
        sane_upper_bound = 100

        calls = {"n": 0}
        real = difflib.SequenceMatcher.quick_ratio

        def counting(self):
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(difflib.SequenceMatcher, "quick_ratio", counting)
        content = "\n".join(f"    do_thing(x_{i})" for i in range(20_000))
        nearest_candidates(content, "        do_thing(x_10)")
        assert calls["n"] <= sane_upper_bound, (
            f"{calls['n']} windows examined in a 20,000-line file; the anchor "
            "prefilter is not running"
        )

    def test_S103_a_repeated_line_does_not_defeat_the_anchor_cap(
        self, monkeypatch
    ) -> None:
        # The test above has exactly one matching line, so it passes whatever
        # the cap is -- it proves anchoring happens, not that it is bounded.
        # A file where thousands of lines match the anchor is the case the cap
        # exists for, and boilerplate that repeats is ordinary in real code.
        import difflib

        sane_upper_bound = 100
        calls = {"n": 0}
        real = difflib.SequenceMatcher.quick_ratio

        def counting(self):
            calls["n"] += 1
            return real(self)

        monkeypatch.setattr(difflib.SequenceMatcher, "quick_ratio", counting)
        content = "\n".join(["    do_thing(x)"] * 5_000)
        nearest_candidates(content, "        do_thing(x)")
        assert calls["n"] <= sane_upper_bound, (
            f"{calls['n']} windows examined; the anchor cap is not bounding a "
            "file of repeated lines"
        )

    def test_S103_the_quadratic_comparison_sees_bounded_strings(
        self, monkeypatch
    ) -> None:
        # Asserts the *input length* to the quadratic step, not the elapsed
        # time. A "does not hang" test hangs when the property breaks, which
        # makes the failure slow to observe and the suite hostage to it -- the
        # first version of this test ran past 120 seconds under a mutation and
        # had to be killed. This fails in milliseconds instead.
        import difflib

        # A literal, deliberately, NOT `_RATIO_SAMPLE_CHARS`. Importing the
        # constant makes the assertion move with the thing it is supposed to
        # pin: raising it to 10**9 left this test asserting 90,000 <= 10**9
        # and passing while the quadratic step went unbounded.
        sane_upper_bound = 10_000

        seen: list[int] = []

        def recording(self):
            # Records what it was handed and returns; it never *runs* the
            # comparison. A version that delegated to the real method still
            # had to complete the expensive call before it could assert the
            # call was expensive, so under a mutation it ground for minutes
            # instead of failing.
            seen.append(max(len(self.a), len(self.b)))
            return 1.0

        monkeypatch.setattr(difflib.SequenceMatcher, "ratio", recording)
        monkeypatch.setattr(difflib.SequenceMatcher, "quick_ratio", recording)
        body = "\n".join(f"    step_{i}(alpha, beta, gamma)" for i in range(3_000))
        content = "def run():\n" + body
        nearest_candidates(content, "def run():\n" + body.replace("    ", "        "))
        assert seen, "nothing was compared"
        assert max(seen) <= sane_upper_bound, (
            f"compared a {max(seen)}-character string; sampling is not applied "
            "and the quadratic step is unbounded"
        )

    def test_S103_the_budget_holds_across_candidate_sizes(self) -> None:
        # Sweeps the block size so the +1-per-separator accounting is actually
        # exercised at the boundary; a single fixed size can sit clear of it
        # and never notice a 3-character overshoot.
        # Step 1, not 7: the overshoot this catches is at most 3 characters
        # (one per joining newline), so a coarse sweep steps straight over the
        # boundary where it is visible.
        for width in range(20, 200):
            block = "\n".join(f"    v{j} = f({'a' * width})" for j in range(4))
            content = "\n\n".join(block.replace("f(", f"f{k}(") for k in range(4))
            detail = describe_mismatch(content, block.replace("f(", "fX("))
            assert len(detail) <= DIAGNOSTIC_BUDGET, (width, len(detail))


class TestMultiEditIsVisibleToDiligence:
    """D3. `record_written_data` branched on three tool names; `multi_edit`
    fell through every branch and recorded nothing, so the tautology,
    no-execution and self-authored-checker detectors all went silent for
    anything written through it — while its own description tells the model to
    prefer it over `edit_file`. An empty finding list looks like a clean one.
    """

    def test_S103_multi_edit_writes_are_recorded(self) -> None:
        from harness.diligence import WrittenData, record_written_data

        written = WrittenData()
        record_written_data(written, "multi_edit", {
            "path": "result.txt",
            "edits": [
                {"old_string": "a", "new_string": "EXPECTED_OUTPUT_42"},
                {"old_string": "b", "new_string": "second literal"},
            ],
        })
        assert "EXPECTED_OUTPUT_42" in written.lines_for("result.txt")
        assert "second literal" in written.lines_for("result.txt")

    def test_S103_it_records_the_same_literal_edit_file_would(self) -> None:
        from harness.diligence import WrittenData, record_written_data

        via_edit = WrittenData()
        record_written_data(via_edit, "edit_file", {
            "path": "r.txt", "old_string": "x", "new_string": "SENTINEL"})
        via_multi = WrittenData()
        record_written_data(via_multi, "multi_edit", {
            "path": "r.txt", "edits": [{"old_string": "x", "new_string": "SENTINEL"}]})
        assert via_multi.lines_for("r.txt") == via_edit.lines_for("r.txt")

    def test_S103_malformed_edits_are_ignored_not_crashed_on(self) -> None:
        from harness.diligence import WrittenData, record_written_data

        written = WrittenData()
        record_written_data(written, "multi_edit", {"path": "r.txt", "edits": "nope"})
        record_written_data(written, "multi_edit", {"path": "r.txt", "edits": [1, None]})
        assert written.lines_for("r.txt") == frozenset()
