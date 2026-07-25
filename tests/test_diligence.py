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
        assert [f.kind for f in findings] == ["tautology"]
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
        assert _kinds(command, written) == ["tautology"]

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
            "cmp data.txt out.txt",
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
        assert _kinds('grep -q "OK" out.txt', written) == []

    def test_bare_path_literals_are_never_discriminating(self) -> None:
        written = WrittenData()
        record_written_data(
            written, "bash", {"command": 'echo "/app/results/final.json" > out.txt'}
        )
        assert _kinds('grep -q "/app/results/final.json" out.txt', written) == []

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
        assert _kinds(command, written) == ["tautology"]


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
        assert _kinds("test -f /app/out.txt") == ["existence_only"]
        assert _kinds("[ -e /app/out.txt ]") == ["existence_only"]
        assert _kinds("ls -la /app/results") == ["existence_only"]

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
        assert kinds == ["tautology", "neutralized_exit"]
