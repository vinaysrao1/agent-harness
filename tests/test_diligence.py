"""Tests for harness.diligence (DESIGN.md §4.9) — fully deterministic."""

from __future__ import annotations

import pytest

from harness.diligence import CONTINUE_REMINDER, MAX_NUDGES, looks_unfinished


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
