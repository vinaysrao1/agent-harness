"""Tests for harness.diligence (DESIGN.md §4.9) — fully deterministic."""

from __future__ import annotations

import json

import pytest

from harness.diligence import (
    CONTINUE_REMINDER,
    MAX_NUDGES,
    VERIFICATION_FAILED_REMINDER,
    VERIFICATION_OUTPUT_LIMIT,
    VERIFICATION_TOOL_NAME,
    WrittenData,
    format_lint_advisory,
    lint_verification,
    looks_unfinished,
    record_written_data,
    truncate_verification_output,
)
from harness.tools.builtin import declare_verification_tool


class TestPromisedFutureWork:
    """Phrasings that promise work instead of doing it are flagged."""

    @pytest.mark.parametrize(
        "text",
        [
            "I will run the tests next.",
            "First, I will refactor the parser.",
            "I'll get to the documentation after this.",
            "I’ll wire up the CLI shortly.",  # curly apostrophe
            "Everything is drafted; let me know if you want changes.",
            "Next, I plan to add the error handling.",
            "Once you confirm, the deploy can proceed.",
        ],
    )
    def test_flags_promise_phrasings(self, text: str) -> None:
        unfinished, reason = looks_unfinished(text, 0)
        assert unfinished is True
        assert reason  # a human-readable reason is always given

    @pytest.mark.parametrize(
        "text",
        [
            "i will do it later",
            "I WILL handle that afterwards.",
            "LET ME KNOW what you think.",
        ],
    )
    def test_matching_is_case_insensitive(self, text: str) -> None:
        unfinished, _ = looks_unfinished(text, 0)
        assert unfinished is True

    def test_word_boundaries_prevent_false_positives(self) -> None:
        # "I willingly" must not trip the "I will" pattern; "concert you
        # once youths arrive" style substrings need real word boundaries.
        unfinished, reason = looks_unfinished(
            "I willingly reran the suite; all 12 tests pass.", 0
        )
        assert unfinished is False
        assert reason == ""


class TestTrailingQuestion:
    def test_trailing_question_is_flagged(self) -> None:
        unfinished, reason = looks_unfinished(
            "The fix is in. Should the tests also be run?", 0
        )
        assert unfinished is True
        assert "question" in reason

    def test_trailing_whitespace_after_question_still_flagged(self) -> None:
        unfinished, _ = looks_unfinished("Shall the branch be merged?  \n", 0)
        assert unfinished is True

    def test_question_mid_text_is_not_flagged(self) -> None:
        unfinished, _ = looks_unfinished(
            "Asked myself: does it pass? Yes — all tests green, task done.", 0
        )
        assert unfinished is False


class TestOpenLedgerItems:
    def test_open_items_flag_even_a_clean_message(self) -> None:
        unfinished, reason = looks_unfinished("All done. Tests pass.", 3)
        assert unfinished is True
        assert "3 task-ledger items still open" in reason

    def test_singular_reason_wording(self) -> None:
        _, reason = looks_unfinished("Done.", 1)
        assert "1 task-ledger item still open" in reason

    def test_none_text_with_open_items_is_unfinished(self) -> None:
        unfinished, _ = looks_unfinished(None, 1)
        assert unfinished is True


class TestFinishedAnswers:
    @pytest.mark.parametrize(
        "text",
        [
            "All done. The suite passes: 14 passed in 0.31s.",
            "Task complete. Output written to report.md.",
            None,
            "",
        ],
    )
    def test_clean_finishes_are_not_flagged(self, text: str | None) -> None:
        assert looks_unfinished(text, 0) == (False, "")

    def test_multiple_signals_join_reasons(self) -> None:
        unfinished, reason = looks_unfinished(
            "I will finish up — let me know if that works?", 2
        )
        assert unfinished is True
        parts = reason.split("; ")
        assert len(parts) >= 4  # two promises + question + open items


class TestConstants:
    def test_max_nudges_is_two(self) -> None:
        assert MAX_NUDGES == 2

    def test_reminder_formats_with_reason_and_demands_evidence(self) -> None:
        rendered = CONTINUE_REMINDER.format(reason="promises future work")
        assert "promises future work" in rendered
        assert "evidence" in rendered
        assert "{reason}" not in rendered


class TestVerificationConstants:
    """Constants for the §10.3 B1 self-verification mechanism."""

    def test_tool_name(self) -> None:
        assert VERIFICATION_TOOL_NAME == "declare_verification"

    def test_failed_reminder_formats_all_placeholders(self) -> None:
        rendered = VERIFICATION_FAILED_REMINDER.format(
            command="pytest -q",
            exit_code=1,
            output="2 failed, 3 passed",
        )
        assert "pytest -q" in rendered
        assert "exit code 1" in rendered
        assert "2 failed, 3 passed" in rendered
        assert "redeclare" in rendered
        for token in ("{command}", "{exit_code}", "{output}"):
            assert token not in rendered


class TestTruncateVerificationOutput:
    def test_short_output_is_unchanged(self) -> None:
        assert truncate_verification_output("all good") == "all good"

    def test_output_at_exact_limit_is_unchanged(self) -> None:
        text = "x" * VERIFICATION_OUTPUT_LIMIT
        assert truncate_verification_output(text) == text

    def test_long_output_keeps_the_tail_with_a_marker(self) -> None:
        # The tail carries the signal (test runners summarize at the end).
        long_text = "x" * 5000 + "FINAL SUMMARY: 1 failed"
        truncated = truncate_verification_output(long_text)
        assert truncated.endswith("FINAL SUMMARY: 1 failed")
        assert "chars truncated" in truncated
        dropped = len(long_text) - VERIFICATION_OUTPUT_LIMIT
        assert f"{dropped} chars truncated" in truncated


# ---------------------------------------------------------------------------
# Verification-quality lint (Change 4) — warn-only
# ---------------------------------------------------------------------------


def _kinds(command: str, written_data: WrittenData | None = None) -> list[str]:
    """Finding kinds for ``command``, in the order the lint reports them."""
    return [f.kind for f in lint_verification(command, written_data)]


class TestWrittenData:
    """The bounded map that decides what "the agent wrote this" means."""

    def test_write_file_records_content_lines(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "/app/solve.py", "content": "print('ALL_CHECKS_PASSED')"},
        )
        assert written.lines_for("/app/solve.py") == frozenset(
            {"print('ALL_CHECKS_PASSED')"}
        )

    def test_edit_file_records_the_new_string_only(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "edit_file",
            {
                "path": "/app/solve.py",
                "old_string": "REPLACED_TEXT_HERE",
                "new_string": "INSERTED_TEXT_HERE",
            },
        )
        lines = written.lines_for("/app/solve.py")
        assert lines == frozenset({"INSERTED_TEXT_HERE"})

    def test_echo_redirect_records_the_literal(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        assert written.lines_for("/app/result.txt") == frozenset(
            {"Qwen/Qwen3-Embedding-8B"}
        )

    def test_heredoc_records_the_body(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {
                "command": (
                    "cat > /app/out.txt <<'EOF'\n"
                    "flag{gc0d3_iz_ch4LLenGiNg}\n"
                    "EOF"
                )
            },
        )
        assert written.lines_for("/app/out.txt") == frozenset(
            {"flag{gc0d3_iz_ch4LLenGiNg}"}
        )

    def test_program_output_redirect_records_nothing(self) -> None:
        # The whole design rests on this: a file a *program* produced is
        # not agent-written data, even though the agent wrote the program.
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": "python3 /app/solve.py > test.log"}
        )
        assert written.paths() == ()
        assert written.lines_for("test.log") == frozenset()

    def test_echo_with_variables_records_nothing(self) -> None:
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": 'echo "$RESULT_VALUE" > /app/out.txt'}
        )
        assert written.lines_for("/app/out.txt") == frozenset()

    def test_short_lines_are_ignored(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "out.txt", "content": "PASS\nok\nLONG_ENOUGH_LINE"},
        )
        assert written.lines_for("out.txt") == frozenset({"LONG_ENOUGH_LINE"})

    def test_basename_and_absolute_form_both_match(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "DISCRIMINATING_VALUE" > /app/result.txt'},
        )
        assert written.lines_for("result.txt")  # after a `cd /app`
        assert written.lines_for("./result.txt")
        assert written.lines_for("/app/result.txt")
        assert not written.lines_for("other.txt")

    def test_paths_are_lru_evicted_beyond_the_bound(self) -> None:
        written = WrittenData(max_paths=32)
        for index in range(40):
            record_written_data(
                written,
                "write_file",
                {"path": f"file{index}.txt", "content": "DISCRIMINATING_LINE"},
            )
        assert len(written) == 32
        assert not written.lines_for("file0.txt")  # evicted
        assert written.lines_for("file39.txt")

    def test_lines_per_path_are_bounded(self) -> None:
        written = WrittenData(max_lines=200)
        body = "\n".join(f"UNIQUE_LINE_{index}" for index in range(500))
        record_written_data(
            written, "write_file", {"path": "big.txt", "content": body}
        )
        assert len(written.lines_for("big.txt")) == 200

    def test_unknown_tools_and_bad_arguments_are_ignored(self) -> None:
        written = WrittenData()
        record_written_data(written, "read_file", {"path": "out.txt"})
        record_written_data(written, "write_file", {"path": "out.txt"})
        record_written_data(written, "bash", {"command": None})
        assert written.paths() == ()


class TestTautologyLint:
    """A literal written into the file the check *reads* is circular."""

    def test_flags_the_mteb_pair(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        findings = lint_verification(
            "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt", written
        )
        # A bare grep of a written file is also a no_execution case (T1):
        # the command only reads, it never runs the solution.
        assert [f.kind for f in findings] == ["tautology", "no_execution"]
        assert findings[0].details["literal"] == "Qwen/Qwen3-Embedding-8B"
        assert findings[0].details["path"] == "/app/result.txt"

    def test_flags_a_heredoc_written_flag_compared_with_test(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {
                "command": (
                    "cat > /app/out.txt <<'EOF'\n"
                    "flag{gc0d3_iz_ch4LLenGiNg}\n"
                    "EOF"
                )
            },
        )
        command = (
            "test \"$(cat /app/out.txt)\" = 'flag{gc0d3_iz_ch4LLenGiNg}'"
        )
        # test + cat + a bare "=" comparison artifact is also a
        # no_execution case (T1): nothing here runs the solution.
        assert _kinds(command, written) == ["tautology", "no_execution"]

    def test_does_not_flag_the_same_check_when_a_program_produced_the_file(
        self,
    ) -> None:
        # The sharpest boundary in the design: identical verification
        # command, and the only difference is whether the flag was echoed
        # into the file or extracted into it by running a program.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {
                "path": "/app/extract.py",
                "content": "print('flag{gc0d3_iz_ch4LLenGiNg}')",
            },
        )
        record_written_data(
            written, "bash", {"command": "python3 /app/extract.py > /app/out.txt"}
        )
        command = (
            "test \"$(cat /app/out.txt)\" = 'flag{gc0d3_iz_ch4LLenGiNg}'"
        )
        # no_execution (T1) honours the same boundary: /app/out.txt is a
        # file a program produced, so the check is reading real output and
        # nothing is reported. Contrast
        # TestNoExecutionLint.test_flags_a_bare_read_only_check, which is
        # the identical command shape over an echoed file.
        assert _kinds(command, written) == []

    def test_does_not_flag_grep_of_program_output(self) -> None:
        # The X9 acceptance case: solve.py prints PASS, the run produces
        # test.log, and grepping test.log is an honest check.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "solve.py", "content": 'print("PASS")'},
        )
        record_written_data(
            written, "bash", {"command": "python3 solve.py > test.log"}
        )
        # No finding at all: not a tautology (the read-vs-produced
        # distinction) and not no_execution either, which applies the same
        # distinction — test.log exists only because the solution ran.
        assert _kinds('grep -q "PASS" test.log', written) == []

    def test_does_not_flag_program_output_even_for_a_long_literal(
        self,
    ) -> None:
        # Same case with a literal long enough to be "discriminating", so
        # the read-path rule — not the length floor — is what saves it.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {
                "path": "solve.py",
                "content": 'print("ALL_CHECKS_PASSED")',
            },
        )
        record_written_data(
            written, "bash", {"command": "python3 solve.py > test.log"}
        )
        assert _kinds('grep -q "ALL_CHECKS_PASSED" test.log', written) == []

    @pytest.mark.parametrize(
        "command",
        [
            "pytest -q",
            "make check",
            'python3 -c "import json; assert abs(d[\'G\'][\'x0\']-1580)<5"',
            "bash run_tests.sh",
        ],
    )
    def test_never_analyzes_other_command_shapes(self, command: str) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "data.txt", "content": "DISCRIMINATING_VALUE_HERE"},
        )
        assert _kinds(command, written) == []

    def test_cmp_is_never_a_tautology_and_needs_both_sides_authored(
        self,
    ) -> None:
        # cmp is untouched by the tautology/neutralized_exit/existence_only
        # detectors. T1's no_execution detector treats it as the pure
        # file-reading predicate it is, but only when *every* file it reads
        # was written by hand this run: with one side of unknown
        # provenance this is the golden-file shape (see
        # TestNoExecutionLint.test_comparing_against_a_hand_written_fixture
        # _is_not_flagged), and the unknown side is what something ran to
        # produce.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "data.txt", "content": "DISCRIMINATING_VALUE_HERE"},
        )
        assert _kinds("cmp data.txt out.txt", written) == []
        record_written_data(
            written,
            "write_file",
            {"path": "out.txt", "content": "DISCRIMINATING_VALUE_HERE"},
        )
        assert _kinds("cmp data.txt out.txt", written) == ["no_execution"]

    def test_matches_across_a_cd_using_the_basename(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        command = "cd /app && grep -q '^Qwen/Qwen3-Embedding-8B$' result.txt"
        assert _kinds(command, written) == ["tautology"]

    def test_short_literals_are_never_discriminating(self) -> None:
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": 'echo "OK_VALUE_X" > out.txt'}
        )
        assert _kinds('grep -q "OK" out.txt', written) == ["no_execution"]

    def test_bare_path_literals_are_never_discriminating(self) -> None:
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": 'echo "/app/results/final.json" > out.txt'}
        )
        assert _kinds('grep -q "/app/results/final.json" out.txt', written) == [
            "no_execution"
        ]

    @pytest.mark.parametrize(
        "command",
        [
            'grep -q "$EXPECTED_VALUE" /app/result.txt',
            "grep -q \"$(cat /app/expected.txt)\" /app/result.txt",
            "grep -qE 'Qwen.*Embedding' /app/result.txt",
            "grep -f patterns.txt /app/result.txt",
            "grep -q 'Qwen/Qwen3-Embedding-8B'",  # stdin, no file operand
        ],
    )
    def test_gives_up_silently_on_unanalyzable_patterns(
        self, command: str
    ) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        assert "tautology" not in _kinds(command, written)

    def test_silent_without_written_data(self) -> None:
        # No written_data means neither of the two provenance-based
        # detectors can fire: tautology has no literal to match, and
        # no_execution (T1) has no way to tell an echoed file from one a
        # program produced, so it gives up rather than guess.
        assert (
            _kinds("grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt")
            == []
        )

    def test_bracket_test_form_is_analyzed_too(self) -> None:
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": "echo 'flag{abcdefgh}' > /app/out.txt"}
        )
        command = "[ \"$(cat /app/out.txt)\" = 'flag{abcdefgh}' ]"
        assert _kinds(command, written) == ["tautology", "no_execution"]


class TestNeutralizedExitLint:
    """Warn on every neutralizer; record position, never reject."""

    @pytest.mark.parametrize(
        "command",
        ["pytest -q || true", "pytest -q || :", "pytest -q; true", "pytest -q; exit 0"],
    )
    def test_flags_every_neutralizer_shape(self, command: str) -> None:
        assert _kinds(command) == ["neutralized_exit"]

    def test_records_terminal_position(self) -> None:
        (finding,) = lint_verification("pytest -q || true")
        assert finding.details["terminal"] is True
        assert finding.details["text"] == "|| true"

    def test_records_non_terminal_position_without_rejecting(self) -> None:
        # A regex cannot tell this legitimate use from the terminal one —
        # so it warns, and records enough for a later round to tell them
        # apart from real data.
        (finding,) = lint_verification("pkill -f server || true; pytest -q")
        assert finding.kind == "neutralized_exit"
        assert finding.details["terminal"] is False
        assert finding.details["token_index"] == 3

    def test_clean_commands_are_not_flagged(self) -> None:
        assert _kinds("pytest -q && echo done") == []
        assert _kinds("make check") == []

    def test_falls_back_to_regex_when_tokenization_fails(self) -> None:
        # Unbalanced quote: shlex gives up, the raw-text scan does not.
        findings = lint_verification("echo \"unterminated || true")
        assert [f.kind for f in findings] == ["neutralized_exit"]
        assert findings[0].details["source"] == "regex"


class TestExistenceOnlyLint:
    def test_flags_a_bare_existence_probe(self) -> None:
        # existence_only is purely structural: it needs no written_data
        # and fires on its own. (no_execution, which also matches these
        # shapes structurally, stays silent here because it additionally
        # needs provenance — see the next test.)
        assert _kinds("test -f /app/out.txt") == ["existence_only"]
        assert _kinds("[ -e /app/out.txt ]") == ["existence_only"]
        assert _kinds("ls -la /app/results") == ["existence_only"]

    def test_an_existence_probe_of_a_self_written_file_is_also_no_execution(
        self,
    ) -> None:
        # Same three shapes, now over a file this run wrote by hand: T1's
        # no_execution joins in, and the two findings are reported in
        # detector order.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "/app/out.txt", "content": "DISCRIMINATING_VALUE_HERE"},
        )
        assert _kinds("test -f /app/out.txt", written) == [
            "existence_only",
            "no_execution",
        ]
        assert _kinds("[ -e /app/out.txt ]", written) == [
            "existence_only",
            "no_execution",
        ]

    def test_does_not_flag_an_existence_probe_guarding_real_work(
        self,
    ) -> None:
        assert _kinds("test -f /app/out.txt && pytest -q") == []

    def test_does_not_attempt_structural_assertions(self) -> None:
        # raman-fitting's keys-present check: deliberately out of scope.
        command = (
            'python3 -c "import json; d=json.load(open(\'f.json\')); '
            "assert all(k in d for k in ('G','D'))\""
        )
        assert _kinds(command) == []


class TestLintFindingAndAdvisory:
    def test_finding_payload_is_json_serializable(self) -> None:
        (finding,) = lint_verification("pytest -q || true")
        payload = finding.as_payload()
        assert set(payload) == {"kind", "message", "details"}
        assert json.loads(json.dumps(payload)) == payload

    def test_advisory_is_empty_without_findings(self) -> None:
        assert format_lint_advisory([]) == ""

    def test_advisory_cannot_make_a_message_look_unfinished(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        findings = lint_verification(
            "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt || true",
            written,
        )
        advisory = format_lint_advisory(findings)
        assert advisory
        # The model may echo this text; it must not trip the diligence
        # promise/question heuristics if it does.
        assert looks_unfinished(advisory, 0) == (False, "")

    def test_empty_command_yields_nothing(self) -> None:
        assert lint_verification("") == []
        assert lint_verification("   ") == []


class TestLintCombinations:
    def test_reports_every_applicable_finding(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "bash",
            {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt'},
        )
        kinds = _kinds(
            "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt || true",
            written,
        )
        # grep and the "|| true" neutralizer's "true" are both in the
        # no_execution whitelist, so all three detectors fire (T1).
        assert kinds == ["tautology", "neutralized_exit", "no_execution"]


# ---------------------------------------------------------------------------
# T1: two new advisory-only detectors — no_execution, self_authored_checker
# ---------------------------------------------------------------------------


def _authored(*paths: str) -> WrittenData:
    """A :class:`WrittenData` map in which every ``path`` was written by
    hand this run (``write_file``), with a discriminating-length line."""
    written = WrittenData()
    for path in paths:
        record_written_data(
            written,
            "write_file",
            {"path": path, "content": "DISCRIMINATING_VALUE_HERE"},
        )
    return written


class TestNoExecutionLint:
    """Fires only when every segment is a recognized read-only predicate
    *and* one of the files it reads is one this run wrote by hand."""

    def test_flags_a_bare_read_only_check(self) -> None:
        out = _authored("out.txt")
        assert _kinds("grep -q PASS out.txt", out) == ["no_execution"]
        assert _kinds("cat out.txt", out) == ["no_execution"]
        # combinations of whitelisted heads all fire
        assert _kinds("test -f a.txt && cat a.txt", _authored("a.txt")) == [
            "no_execution"
        ]

    def test_does_not_fire_when_any_segment_might_execute(self) -> None:
        # python3, gcc, ./a.out, make, pytest, bash, perl, a compiled path —
        # any one unrecognized head blocks the whole finding.
        # Provenance is satisfied for every one of these (each names a
        # file this run wrote), so only the structural half can be what
        # holds the finding back. `python3 solve.py` also trips
        # self_authored_checker, hence the membership form.
        written = _authored("solve.py", "a.txt", "a.c", "checker.sh")
        assert "no_execution" not in _kinds("python3 solve.py", written)
        assert "no_execution" not in _kinds(
            "test -f a.txt && python3 solve.py", written
        )
        assert _kinds("gcc -O3 a.c -o a.out && ./a.out", written) == []
        assert _kinds("./checker.sh", written) == []

    def test_comparison_operator_artifacts_do_not_block_it(self) -> None:
        # _split_segments hands back "-eq"/"-lt"/"=" as segment heads when
        # a command substitution containing a redirect sits inside a test
        # expression (the caveat this detector is built around). These
        # artifacts must not be treated as unrecognized-and-blocking.
        assert _kinds(
            "test \"$(cat /app/out.txt)\" = 'flag{gc0d3_iz_ch4LLenGiNg}'",
            _authored("/app/out.txt"),
        ) == ["no_execution"]
        # The read path here sits inside the substitution's own segment,
        # after a `<` redirect, and is still found.
        assert _kinds(
            "test $(wc -c < f.txt) -lt 100", _authored("f.txt")
        ) == ["no_execution"]

    def test_requires_a_file_this_run_wrote_by_hand(self) -> None:
        # The provenance half of the rule, in isolation: structurally
        # identical commands, and the only difference is whether the
        # harness ever saw this run put a literal into the file being
        # read. Unknown provenance is not evidence of a weak check.
        assert _kinds("grep -q PASS out.txt", WrittenData()) == []
        assert _kinds("cat out.txt", _authored("other.txt")) == []
        assert _kinds("grep -q PASS out.txt", _authored("out.txt")) == [
            "no_execution"
        ]

    def test_is_silent_when_a_program_produced_the_file_it_reads(
        self,
    ) -> None:
        # The regression this rule exists for: the harness's own
        # documented honest shape (X9). solve.py is agent-written, but the
        # check reads test.log, which exists only because solve.py *ran* —
        # so the model must not be told its correct check proves nothing.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "solve.py", "content": 'print("ALL_CHECKS_PASSED")'},
        )
        record_written_data(
            written, "bash", {"command": "python3 solve.py > test.log"}
        )
        assert _kinds('grep -q "ALL_CHECKS_PASSED" test.log', written) == []
        assert format_lint_advisory(
            lint_verification('grep -q "ALL_CHECKS_PASSED" test.log', written)
        ) == ""

    def test_process_substitution_runs_the_solution(self) -> None:
        # The same X9 shape spelled without the temp file: the solution is
        # executed inside `<(...)`, so telling the model "nothing in it runs
        # the solution itself" would be false about a command that literally
        # does. `<(` is a segment separator precisely so the inner
        # interpreter reaches head position, and solve.py — the program
        # being run — is never counted as a file the check read.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "solve.py", "content": 'print("ALL_CHECKS_PASSED")'},
        )
        command = "grep -q ALL_CHECKS_PASSED <(python3 solve.py)"
        assert "no_execution" not in _kinds(command, written)
        assert format_lint_advisory(
            [
                f
                for f in lint_verification(command, written)
                if f.kind == "no_execution"
            ]
        ) == ""
        # Both directions, under a whitelisted head, with every other
        # operand hand-authored — so the structural half is the only thing
        # that can hold the finding back.
        both = _authored("solve.py", "expected.txt")
        assert "no_execution" not in _kinds(
            "diff expected.txt <(python3 solve.py)", both
        )
        assert "no_execution" not in _kinds(
            "diff expected.txt >(python3 solve.py)", both
        )
        # Command substitution already split correctly; pinned so the two
        # spellings cannot drift apart again.
        assert "no_execution" not in _kinds(
            "test $(python3 solve.py) = ok", written
        )

    def test_comparing_against_a_hand_written_fixture_is_not_flagged(
        self,
    ) -> None:
        # The standard golden-file verification: the expectation *must* be
        # hand-authored, so "at least one operand is agent-written" would
        # condemn the shape by construction. build/out.bin is of unknown
        # provenance, which is exactly the evidence that something ran to
        # produce it.
        written = _authored("expected.bin")
        assert _kinds("cmp expected.bin build/out.bin", written) == []
        assert _kinds("diff -u expected.txt out/actual.txt", written) == []
        # Both sides hand-written: nothing ran between the literals being
        # invented and the check confirming them, so the finding stands.
        assert _kinds(
            "cmp expected.bin build/out.bin",
            _authored("expected.bin", "build/out.bin"),
        ) == ["no_execution"]

    def test_an_extensionless_compare_target_is_still_a_produced_file(
        self,
    ) -> None:
        # Same shape, output named as executables and Makefile targets
        # usually are: no extension, no directory. `cmp`/`diff` compare
        # files and nothing else, so a bare word there is a file — guessing
        # from the operand's *shape* dropped it, emptied the unknown set,
        # and reported the golden-file compare as reading only what the
        # agent wrote, which is false about a command reading a program's
        # output.
        assert _kinds("cmp expected.bin out", _authored("expected.bin")) == []
        assert (
            _kinds("diff expected.txt actual", _authored("expected.txt")) == []
        )
        assert _kinds("cmp golden.txt result", _authored("golden.txt")) == []
        # Skipped flag arguments are not files, and a flag does not swallow
        # the operand behind it either.
        assert _kinds(
            "cmp -i 512 expected.bin out", _authored("expected.bin")
        ) == []
        assert _kinds(
            "diff -I '^#' expected.txt actual", _authored("expected.txt")
        ) == []
        # The finding still stands when both extensionless sides are the
        # agent's own literals: the exemption is for a produced operand,
        # not for the spelling of the name.
        assert _kinds(
            "cmp expected out", _authored("expected", "out")
        ) == ["no_execution"]

    def test_the_shape_guess_still_guards_operands_that_may_be_literals(
        self,
    ) -> None:
        # `test`/`[` operands are not files by definition — the right-hand
        # side is a literal to compare against. Counting `ok` as a file of
        # unknown provenance would suppress a true finding, so the
        # _PATH_SHAPED guess stays where the grammar gives nothing better.
        assert "no_execution" in _kinds(
            "test $(cat result.txt) = ok", _authored("result.txt")
        )
        assert "no_execution" in _kinds(
            "[ $(wc -l < counts.txt) -eq 100 ]", _authored("counts.txt")
        )

    def test_a_grep_pattern_is_not_an_operand_of_unknown_provenance(
        self,
    ) -> None:
        # The pattern can look exactly like a path (the mteb corpus rows).
        # It is not a file the check reads, so it must not suppress the
        # finding the way a real unknown operand does.
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": 'echo "/app/results/x.json" > out.txt'}
        )
        assert "no_execution" in _kinds(
            'grep -q "/app/results/x.json" out.txt', written
        )

    def test_a_redirect_target_is_written_not_read(self) -> None:
        # `cat a.txt > b.txt` reads a.txt; b.txt is the file it creates.
        # Only the read side can carry provenance evidence.
        assert _kinds("cat a.txt > b.txt", _authored("b.txt")) == []
        assert _kinds("cat a.txt > b.txt", _authored("a.txt")) == [
            "no_execution"
        ]

    def test_artifact_only_segments_do_not_fire_alone(self) -> None:
        # Sanity: a command reduced to nothing but operator artifacts
        # (contrived, not real shell) still requires at least one real
        # whitelisted head before reporting anything.
        assert _lint_no_execution_direct(["=", "x"], _authored("x")) == []

    def test_does_not_fire_on_empty_or_unanalyzable_input(self) -> None:
        written = _authored("out.txt")
        assert _kinds("", written) == []
        # Unbalanced quote: falls back to no tokens at all.
        assert _kinds('echo "unterminated', written) == []

    def test_finding_details_record_segment_count_and_paths(self) -> None:
        (finding,) = lint_verification(
            "test -f a.txt && grep -q X a.txt && wc -l a.txt",
            _authored("a.txt"),
        )
        assert finding.kind == "no_execution"
        assert finding.details["segment_count"] == 3
        assert finding.details["authored_paths"] == ["a.txt"]

    def test_advisory_cannot_make_a_message_look_unfinished(self) -> None:
        (finding,) = lint_verification(
            "grep -q PASS out.txt", _authored("out.txt")
        )
        assert looks_unfinished(finding.message, 0) == (False, "")


def _lint_no_execution_direct(
    segments: list[list[str]], written_data: WrittenData
) -> list[str]:
    """Exercise the segment-level detector directly (bypassing the
    tokenizer) for the contrived all-artifact case."""
    from harness.diligence import _lint_no_execution

    return [f.kind for f in _lint_no_execution(segments, written_data)]


class TestSelfAuthoredCheckerLint:
    """Fires when the check runs a script the agent wrote this run."""

    def test_flags_perl_verify_pl_written_this_run(self) -> None:
        # The dna-assembly R2 pattern this detector exists for.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {
                "path": "verify.pl",
                "content": (
                    'die "assembly mismatch" unless index($out.$out,'
                    'substr($asm,0,-1))>=0;\nprint "ALL CHECKS PASSED\\n";'
                ),
            },
        )
        command = (
            'perl verify.pl | grep -q "ALL CHECKS PASSED" && '
            "test $(grep -c '>' primers.fasta) -eq 8"
        )
        assert _kinds(command, written) == ["self_authored_checker"]

    def test_matches_other_recognized_interpreters(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "checker.py", "content": "print('DISCRIMINATING_OK')"},
        )
        assert _kinds("python3 checker.py", written) == [
            "self_authored_checker"
        ]

    def test_does_not_flag_a_script_not_written_this_run(self) -> None:
        # No written_data entry for verify.pl (e.g. it shipped with the
        # task) — nothing to flag.
        written = WrittenData()
        record_written_data(
            written, "write_file", {"path": "other.txt", "content": "IRRELEVANT_X"}
        )
        assert _kinds("perl verify.pl", written) == []

    def test_silent_without_written_data(self) -> None:
        assert _kinds("perl verify.pl") == []

    def test_gives_up_on_inline_code_rather_than_guessing(self) -> None:
        # -e/-c/-m take inline code, not a script path; the raman-fitting
        # shape (`python3 -c "..."`) must never be misread as a script.
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "assert", "content": "DISCRIMINATING_CONTENT_HERE"},
        )
        command = 'python3 -c "import json; assert True"'
        assert _kinds(command, written) == []

    def test_advisory_cannot_make_a_message_look_unfinished(self) -> None:
        written = WrittenData()
        record_written_data(
            written,
            "write_file",
            {"path": "verify.pl", "content": "print 'ALL CHECKS PASSED';"},
        )
        (finding,) = lint_verification("perl verify.pl", written)
        assert looks_unfinished(finding.message, 0) == (False, "")


# ---------------------------------------------------------------------------
# T1 corpus: the 16 real verification_declared commands from round 1 + 2
# (jobs/round1-rerun, jobs/round2-rerun), verbatim, with each run's
# written_data reconstructed from its own write_file/edit_file/bash events.
# Ground truth for expected_kinds was computed by replaying every real
# tool_call for each run through record_written_data (in order) and calling
# lint_verification on the real declared command — see the T1 implementer
# notes for how it was generated. It is checked in here so a future change
# to any detector re-runs against real data, not synthetic fixtures.
# ---------------------------------------------------------------------------

_DNA_R1_DECL_0 = 'cd /app && perl -e \'\nmy%p;my$n;open F,"primers.fasta";while(<F>){chomp;die "blank" unless $_;if(/^>(\\w+)_(\\w+)$/){$n=$1}else{$p{$n}.=$_}}\nsub tm{my$s=shift;my$r=`oligotm -tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500 $s`;chomp$r;$r+0}\nmy%s;open G,"sequences.fasta";my$m;while(<G>){chomp;if(/^>(.+)/){$m=$1}else{$s{$m}.=uc($_)}}\nmy($inp,$egfp,$flag,$snap,$out)=@s{qw(input egfp flag snap output)};\nmy%oh=(input=>["AATG","TATG"],egfp=>["TATG","AAGG"],flag=>["AAGG","CAGA"],snap=>["CAGA","AATG"]);\nsub rc{my$x=shift;$x=~tr/ACGT/TGCA/;scalar reverse$x}\nmy%bodies=(input=>substr($inp,692).substr($inp,0,210),egfp=>substr($egfp,3,709),flag=>substr($flag,5,80),snap=>substr($snap,5,542));\nmy$asm=$oh{egfp}[0].$bodies{egfp}.$oh{flag}[0].$bodies{flag}.$oh{snap}[0].$bodies{snap}.$oh{input}[0].$bodies{input};\ndie "assembly mismatch" unless index($out.$out,substr($asm,0,-1))>=0 && substr($asm,-1) eq substr($asm,0,1);\nfor my$t(qw(input egfp flag snap)){\n my($lo,$ro)=@{$oh{$t}};\n my$f=$p{$t};my$r=rc(substr($p{$t},1));\n my$fp=substr($f,0,length($f));my$rp=substr($r,0,length($r));\n my$tailf=substr($fp,0,10);my$tailr=substr($rp,0,10);\n die "site $t" unless $tailf eq "tttGGTCTCa" && $tailr eq "tttGGTCTCa";\n die "oh $t" unless substr($fp,10,4) eq $lo && substr($rp,10,4) eq rc($ro);\n my$af=substr($fp,14);my$ar=substr($rp,14);\n die "len $t" if length($af)<15||length($af)>45||length($ar)<15||length($ar)>45;\n die "anneal $t" unless $af eq substr($bodies{$t},0,length($af)) && $ar eq rc(substr($bodies{$t},-length($ar)));\n my$tf=tm($af);my$tr=tm($ar);\n die "tm $t" if $tf<58||$tf>72||$tr<58||$tr>72||abs($tf-$tr)>5;\n print "$t fwd Tm=$tf rev Tm=$tr OK\\n";\n}\nprint "ALL CHECKS PASSED\\n";\''

_DNA_R1_DECL_1 = 'cd /app && perl -e \'\nmy%p;open F,"primers.fasta";my$h;\nwhile(<F>){chomp;die "blank line" unless length;if(/^>(\\w+)_(fwd|rev)$/){$h="$1_$2";die "bad header" unless grep {$_ eq $1} qw(input egfp flag snap);}else{die "dup $h" if $p{$h};$p{$h}=$_}}\ndie "need 8 primers" unless scalar(keys %p)==8;\nsub tm{my$s=shift;my$r=`oligotm -tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500 $s`;chomp$r;$r+0}\nmy%s;open G,"sequences.fasta";my$m;while(<G>){chomp;if(/^>(.+)/){$m=$1}else{$s{$m}.=uc($_)}}\nmy($inp,$egfp,$flag,$snap,$out)=@s{qw(input egfp flag snap output)};\nmy%oh=(input=>["AATG","TATG"],egfp=>["TATG","AAGG"],flag=>["AAGG","CAGA"],snap=>["CAGA","AATG"]);\nsub rc{my$x=shift;$x=~tr/ACGT/TGCA/;scalar reverse$x}\nmy%bodies=(input=>substr($inp,692).substr($inp,0,210),egfp=>substr($egfp,3,709),flag=>substr($flag,5,80),snap=>substr($snap,5,542));\nmy$asm="TATG".$bodies{egfp}."AAGG".$bodies{flag}."CAGA".$bodies{snap}."AATG".$bodies{input};\ndie "assembly mismatch" unless index($out.$out,substr($asm,0,-1))>=0 && substr($asm,-1) eq substr($asm,0,1);\nfor my$t(qw(input egfp flag snap)){\n my($lo,$ro)=@{$oh{$t}};\n my$fp=$p{"${t}_fwd"};my$rp=$p{"${t}_rev"};\n die "tail $t" unless substr($fp,0,10) eq "tttGGTCTCa" && substr($rp,0,10) eq "tttGGTCTCa";\n die "oh $t" unless substr($fp,10,4) eq $lo && substr($rp,10,4) eq rc($ro);\n my$af=substr($fp,14);my$ar=substr($rp,14);\n die "len $t" if length($af)<15||length($af)>45||length($ar)<15||length($ar)>45;\n die "anneal $t" unless $af eq substr($bodies{$t},0,length($af)) && $ar eq rc(substr($bodies{$t},-length($ar)));\n my$tf=tm($af);my$tr=tm($ar);\n die "tm $t" if $tf<58||$tf>72||$tr<58||$tr>72||abs($tf-$tr)>5;\n printf "%s fwd Tm=%.2f rev Tm=%.2f OK\\n",$t,$tf,$tr;\n}\nprint "ALL CHECKS PASSED\\n";\''

_GCODE_R1 = 'test "$(cat /app/out.txt)" = \'flag{gc0d3_iz_ch4LLenGiNg}\''

_MCMC_R1 = 'Rscript -e \'stopifnot(packageVersion("rstan")=="2.32.7")\' && test -f /app/hierarchical_model.stan && test -f /app/analysis.R && test -f /app/posterior_alpha_mean.txt && test -f /app/posterior_beta_mean.txt && Rscript -e \'a<-as.numeric(readLines("/app/posterior_alpha_mean.txt")); b<-as.numeric(readLines("/app/posterior_beta_mean.txt")); stopifnot(is.finite(a), is.finite(b), a>0, b>0)\''

_MTEB_R1 = "grep -q '^Qwen/Qwen3-Embedding-8B$' /app/result.txt"

_QEMU_SSH_R1 = "sshpass -p password123 ssh -p 2222 -o StrictHostKeyChecking=no root@localhost 'echo SSH_OK && whoami' | grep -q 'SSH_OK'"

_RAMAN_R1 = 'python3 -c "import json; d=json.load(open(\'/app/results.json\')); assert all(k in d for k in (\'G\',\'2D\')) and all(f in d[k] for k in d for f in (\'x0\',\'gamma\',\'amplitude\',\'offset\'))"'

_TORCH_R1 = 'test -f /app/pipeline_parallel.py && grep -q "def train_step_pipeline_afab" /app/pipeline_parallel.py'

_COMPCERT_R2 = "test -x /tmp/CompCert/ccomp && printf 'int main(){return 0;}\\n' > /tmp/v.c && /tmp/CompCert/ccomp /tmp/v.c -o /tmp/v && /tmp/v"

_DNA_R2 = 'perl verify.pl | grep -q "ALL CHECKS PASSED" && test $(grep -c \'>\' primers.fasta) -eq 8 && test $(grep -c \'^$\' primers.fasta) -eq 0'

_EXTRACT_R2 = "test -s /app/solution.txt && grep -q '^kill man$' /app/solution.txt && grep -q '^sw$' /app/solution.txt && wc -l /app/solution.txt"

_GPT2_R2 = 'cd /app && test $(wc -c < gpt2.c) -lt 5000 && gcc -O3 -lm gpt2.c -o a.out && ./a.out gpt2-124M.ckpt vocab.bpe "Alan Turing theorized that computers would one day become" | grep -q "the most powerful machines on the planet"'

_MCMC_R2 = 'test -s /app/posterior_alpha_mean.txt && test -s /app/posterior_beta_mean.txt && test -s /app/hierarchical_model.stan && test -s /app/analysis.R && Rscript -e \'cat(as.character(packageVersion("rstan")))\' | grep -q 2.32.7 && awk \'NF==1 && $1+0>0 {ok=1} END{exit !ok}\' /app/posterior_alpha_mean.txt /app/posterior_beta_mean.txt'

_MTEB_R2 = "grep -qx 'Salesforce/SFR-Embedding-2_R' /app/result.txt"

_QEMU_STARTUP_R2 = "kill -0 $(cat /tmp/qemu.pid) && grep -q 'login:' /tmp/boot.log && (exec 3<>/dev/tcp/127.0.0.1/6665 && echo port-open)"

_RAMAN_R2 = 'python3 -c "\nimport json\nr=json.load(open(\'/app/results.json\'))\nassert set(r)=={\'G\',\'2D\'}\nfor k,v in r.items(): assert all(isinstance(v[f],(int,float)) for f in (\'x0\',\'gamma\',\'amplitude\',\'offset\'))\nprint(\'ok\')"'


def _written_gcode_r1() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "bash",
        {"command": "cd /app && echo 'flag{gc0d3_iz_ch4LLenGiNg}' > out.txt && cat out.txt"},
    )
    return written


def _written_mteb_r1() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "bash",
        {"command": 'echo "Qwen/Qwen3-Embedding-8B" > /app/result.txt && cat /app/result.txt'},
    )
    return written


def _written_mteb_r2() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "bash",
        {"command": "printf 'Salesforce/SFR-Embedding-2_R' > /app/result.txt && cat /app/result.txt"},
    )
    return written


def _written_torch_r1() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "write_file",
        {
            "path": "/app/pipeline_parallel.py",
            "content": (
                "import torch\n\n"
                "def train_step_pipeline_afab(model, inputs, targets, device, dtype):\n"
                "    pass\n"
            ),
        },
    )
    return written


def _written_extract_r2() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "write_file",
        {
            "path": "/app/solution.txt",
            "content": (
                "get sack\nw\nput all\ncase\nget sack\nopen sack\n"
                "give egg\nkill man\ng\ng\ng\n"
                "get head,jade,cup,egg,golden\ntemple\ns\npray\n"
            ),
        },
    )
    return written


def _written_dna_r2() -> WrittenData:
    written = WrittenData()
    record_written_data(
        written,
        "write_file",
        {
            "path": "verify.pl",
            "content": (
                'my%p;my$n;open F,"primers.fasta";\n'
                "sub rc{my$x=shift;$x=~tr/ACGTacgt/TGCAtgca/;$x}\n"
                'die "assembly mismatch" unless index($out.$out,$asm)>=0;\n'
                'print "ALL CHECKS PASSED\\n";\n'
            ),
        },
    )
    return written


# (label, command, written_data factory or None, expected finding kinds)
_REAL_ROUND2_CORPUS = [
    ("dna-assembly R1 decl#1", _DNA_R1_DECL_0, None, []),
    ("dna-assembly R1 decl#2", _DNA_R1_DECL_1, None, []),
    ("gcode-to-text R1", _GCODE_R1, _written_gcode_r1, ["tautology", "no_execution"]),
    ("mcmc-sampling-stan R1", _MCMC_R1, None, []),
    ("mteb-leaderboard R1", _MTEB_R1, _written_mteb_r1, ["tautology", "no_execution"]),
    ("qemu-alpine-ssh R1", _QEMU_SSH_R1, None, []),
    ("raman-fitting R1", _RAMAN_R1, None, []),
    ("torch-pipeline-parallelism R1", _TORCH_R1, _written_torch_r1, ["tautology", "no_execution"]),
    ("compile-compcert R2", _COMPCERT_R2, None, []),
    ("dna-assembly R2", _DNA_R2, _written_dna_r2, ["self_authored_checker"]),
    ("extract-moves-from-video R2", _EXTRACT_R2, _written_extract_r2, ["tautology", "no_execution"]),
    ("gpt2-codegolf R2", _GPT2_R2, None, []),
    ("mcmc-sampling-stan R2", _MCMC_R2, None, []),
    ("mteb-leaderboard R2", _MTEB_R2, _written_mteb_r2, ["tautology", "no_execution"]),
    ("qemu-startup R2", _QEMU_STARTUP_R2, None, []),
    ("raman-fitting R2", _RAMAN_R2, None, []),
]


class TestRealRound2CorpusTable:
    """The 16 real declared verification commands from the round 1 + 2
    Terminal-Bench runs (jobs/round1-rerun, jobs/round2-rerun), verbatim.
    """

    def test_all_16_commands_are_present(self) -> None:
        assert len(_REAL_ROUND2_CORPUS) == 16

    @pytest.mark.parametrize(
        "label,command,written_factory,expected_kinds",
        _REAL_ROUND2_CORPUS,
        ids=[row[0] for row in _REAL_ROUND2_CORPUS],
    )
    def test_expected_finding_kinds(
        self, label, command, written_factory, expected_kinds
    ) -> None:
        written = written_factory() if written_factory is not None else None
        assert _kinds(command, written) == expected_kinds

    @pytest.mark.parametrize(
        "label,command,written_factory,expected_kinds",
        _REAL_ROUND2_CORPUS,
        ids=[row[0] for row in _REAL_ROUND2_CORPUS],
    )
    async def test_declare_verification_still_acknowledges_every_one(
        self, label, command, written_factory, expected_kinds
    ) -> None:
        # T1 is advisory-only: however many findings a declaration turns
        # up, declare_verification must still accept and acknowledge it —
        # nothing here ever rejects.
        written = written_factory() if written_factory is not None else None
        tool = declare_verification_tool(written_data=lambda: written)
        result = await tool.handler(
            {"command": command, "description": "T1 corpus regression"}
        )
        assert result.startswith("verification declared:")
        # The acknowledgement embeds the command via repr(); T1's job is
        # just to confirm nothing about the lint ever rejects.
        assert repr(command) in result


class TestCompareFlagsAreHeadKeyed:
    """`-i`/`-n` take an argument for cmp but not for diff (round-3 review).

    A head-agnostic skip set swallowed the file operand after ``diff -i``,
    emptying the unknown-provenance set and emitting a false ``no_execution``
    advisory about a command that does read program output.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "diff -i out expected.txt",
            "diff -n out expected.txt",
            "diff -i actual golden.txt",
        ],
    )
    def test_diff_no_arg_flags_do_not_swallow_the_produced_operand(self, command):
        written = WrittenData()
        written.record("expected.txt", "expected\n")
        written.record("golden.txt", "golden\n")
        findings = lint_verification(command, written)
        assert "no_execution" not in [f.kind for f in findings]

    def test_cmp_arg_flags_still_skip_their_argument(self):
        written = WrittenData()
        written.record("expected.txt", "expected\n")
        # 512 is a byte count, not a path: it must not be read as an operand.
        findings = lint_verification("cmp -i 512 out expected.txt", written)
        assert "no_execution" not in [f.kind for f in findings]
