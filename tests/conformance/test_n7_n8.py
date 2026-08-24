"""S-003 / N7 + N8: context-timing and cost neutrality.

Both are defined over the frozen replay corpus, because neither can be
unit-tested and neither is observable on the benchmark: compaction has fired
**zero** times across 645 real trials, so a Terminal-Bench run cannot tell you
whether a change to the condenser broke it.

N7  pruning and compaction fire at the same turn indices.
N8  tokens-per-turn does not increase.

These are the invariants that make the three riskiest changes in the plan
safe to attempt -- S-105 (condenser seam), S-109 (real token counting) and
S-102 (file cache) all move context timing, and all three would otherwise be
benchmark gambles. S-109 in particular replaces the ``chars / 4`` estimator
that every threshold in ``harness/context.py`` was tuned against; N7 is what
turns "the constants were tuned to a biased estimator" from a footnote into a
failing test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conformance.replay import (
    REPLAY_WINDOWS,
    load_corpus,
    replay_corpus,
    replay_trial,
)

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "replay_trace.json"

#: The production window, where the replay must agree with observed reality.
PRODUCTION_WINDOW = 128_000


@pytest.fixture(scope="module")
def trace() -> dict:
    return replay_corpus()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


class TestN7ContextTiming:
    def test_S003_n7_prune_and_compaction_turns_match_golden(
        self, trace: dict, golden: dict
    ) -> None:
        assert set(trace) == set(golden), (
            "the replay trace covers different keys than the golden; "
            "re-freeze with `python -m tests.conformance.freeze`"
        )
        drift = {
            key: {
                "golden_prune": golden[key]["prune_turns"],
                "actual_prune": trace[key]["prune_turns"],
                "golden_shed": golden[key].get("prune_shed"),
                "actual_shed": trace[key]["prune_shed"],
                "golden_compact": golden[key]["compact_turns"],
                "actual_compact": trace[key]["compact_turns"],
            }
            for key in sorted(trace)
            if trace[key]["prune_turns"] != golden[key]["prune_turns"]
            or trace[key]["prune_shed"] != golden[key].get("prune_shed")
            or trace[key]["compact_turns"] != golden[key]["compact_turns"]
        }
        assert not drift, (
            "N7: context timing moved. Pruning or compaction now fires at "
            "different turns for:\n"
            + "\n".join(f"  {k}: {v}" for k, v in list(drift.items())[:5])
            + "\nIf deliberate this is Lane B: REFREEZE, run TB2, re-freeze."
        )

    def test_S003_n7_replay_is_deterministic(self) -> None:
        # A golden over a non-deterministic trace pins nothing. Replay one
        # trial twice and require identical output.
        # Use the heaviest trial at the smallest window, so both mechanisms
        # actually fire. The previous version used corpus[0] at 128K, whose
        # trace is three empty lists and a constant -- comparing nothing to
        # nothing, twice.
        record = max(load_corpus(), key=lambda r: len(r["messages"]))
        first = replay_trial(record["messages"], 16_000)
        second = replay_trial(record["messages"], 16_000)
        assert first == second
        assert first["prune_turns"] and first["compact_turns"], (
            "the determinism check must run on a trial where both mechanisms "
            "fire, or it compares empty lists"
        )

    def test_S003_n7_corpus_actually_exercises_both_mechanisms(
        self, trace: dict
    ) -> None:
        """The corpus must fire what it claims to pin.

        This is the invariant's own non-vacuity check. A replay corpus on which
        nothing ever fires would freeze an empty trace and stay green through
        any change to the machinery -- which is precisely what a naive corpus
        built only from `message` events did: it contained no tool-role
        entries at all, so pruning (which only sheds tool results) could not
        fire at any window.
        """
        pruning = sum(1 for v in trace.values() if v["prune_turns"])
        compacting = sum(1 for v in trace.values() if v["compact_turns"])
        assert pruning >= 20, f"only {pruning} traces prune; corpus is too weak"
        assert compacting >= 10, f"only {compacting} traces compact; corpus is too weak"

    def test_S003_n7_production_window_reproduces_production(
        self, trace: dict
    ) -> None:
        """Fidelity check: at 128K the replay must match what really happens.

        Across 645 real trials there is not one `compaction` event, while
        pruning is common. If the replay contradicted that, the corpus would
        be measuring something other than the harness.
        """
        keys = [k for k in trace if k.endswith(f"@{PRODUCTION_WINDOW}")]
        assert keys, "no traces at the production window"
        compacted = [k for k in keys if trace[k]["compact_turns"]]
        assert not compacted, (
            f"replay compacts at the production window ({compacted[:3]}) but "
            "no real run ever has; the corpus no longer reflects the harness"
        )
        pruned = [k for k in keys if trace[k]["prune_turns"]]
        assert pruned, "replay never prunes at the production window, but real runs do"

    def test_S003_n7_detects_a_shed_volume_change(self) -> None:
        # Negative: PRUNE_TARGET_FRACTION changes how much is shed while
        # leaving the firing turns untouched. Pinning only the turns let this
        # through, and shedding more context is a regression even though it
        # reduces tokens.
        import harness.context as context_module

        record = max(load_corpus(), key=lambda r: len(r["messages"]))
        before = replay_trial(record["messages"], PRODUCTION_WINDOW)
        original = context_module.PRUNE_TARGET_FRACTION
        try:
            context_module.PRUNE_TARGET_FRACTION = 0.20
            after = replay_trial(record["messages"], PRODUCTION_WINDOW)
        finally:
            context_module.PRUNE_TARGET_FRACTION = original
        assert after["prune_shed"] != before["prune_shed"], (
            "changing the prune target fraction did not change the shed "
            "volume; N7 would not catch a more destructive prune"
        )

    def test_S003_n7_detects_a_threshold_change(self) -> None:
        # Negative: moving the pressure threshold must move the trace. This is
        # the S-109 trap in miniature -- retuning a constant without noticing
        # it retimed every run.
        import harness.context as context_module

        record = max(load_corpus(), key=lambda r: len(r["messages"]))
        before = replay_trial(record["messages"], PRODUCTION_WINDOW)
        original = context_module.PRUNE_PRESSURE_THRESHOLD
        try:
            context_module.PRUNE_PRESSURE_THRESHOLD = 0.05
            after = replay_trial(record["messages"], PRODUCTION_WINDOW)
        finally:
            context_module.PRUNE_PRESSURE_THRESHOLD = original
        assert after["prune_turns"] != before["prune_turns"], (
            "lowering the prune pressure threshold did not change when "
            "pruning fires; N7 would not catch a retuning"
        )


class TestN8CostNeutrality:
    def test_S003_n8_tokens_per_turn_did_not_increase(
        self, trace: dict, golden: dict
    ) -> None:
        """N8 is one-sided: cheaper is fine, dearer is a regression.

        Equality would fail on any improvement, which would make the invariant
        an obstacle to the very work it is meant to make safe.
        """
        worse = {
            key: (golden[key]["total_tokens"], trace[key]["total_tokens"])
            for key in sorted(trace)
            if trace[key]["total_tokens"] > golden[key]["total_tokens"]
        }
        assert not worse, (
            "N8: tokens-per-turn increased on the replay corpus "
            "(golden -> actual):\n"
            + "\n".join(f"  {k}: {v[0]:,} -> {v[1]:,}" for k, v in list(worse.items())[:5])
        )

    def test_S003_n8_peak_tokens_did_not_increase(
        self, trace: dict, golden: dict
    ) -> None:
        worse = [
            key
            for key in sorted(trace)
            if trace[key]["peak_tokens"] > golden[key]["peak_tokens"]
        ]
        assert not worse, f"N8: peak assembly grew for {worse[:5]}"

    def test_S003_n8_rejects_a_real_cost_increase(
        self, golden: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative: drive the real path with a genuinely dearer harness.

        The previous version asserted ``x + 1 > x`` on a dict it had just
        built. It never called replay_trial, never touched the assertion under
        test, and would have passed with both N8 checks deleted -- while being
        N8's only negative test.
        """
        import tests.conformance.replay as replay_module

        record = max(load_corpus(), key=lambda r: len(r["messages"]))
        key = f"{record['trial']}@{PRODUCTION_WINDOW}"
        original = replay_module._estimate

        def dearer(messages: list) -> int:
            # Every message costs 10 tokens more: a uniform prompt-side
            # regression of exactly the kind N8 exists to catch.
            return original(messages) + 10 * len(messages)

        monkeypatch.setattr(replay_module, "_estimate", dearer)
        after = replay_module.replay_trial(record["messages"], PRODUCTION_WINDOW)
        assert after["total_tokens"] > golden[key]["total_tokens"], (
            "a harness that spends more per turn did not raise total_tokens; "
            "N8 cannot see cost regressions"
        )

    def test_S003_n8_measures_the_real_system_prompt(self) -> None:
        # N8's stated job is to guard silent prompt growth. It cannot do that
        # against a placeholder: the replay must carry the real assembled
        # system string, ~2.4k chars charged on every turn.
        from tests.conformance.replay import production_system_prompt

        assert len(production_system_prompt()) > 1500


class TestCorpusIntegrity:
    def test_S003_corpus_carries_no_task_content(self) -> None:
        """The corpus must be shapes only.

        Real transcripts contain agent solutions to benchmark tasks;
        committing those to a public repository risks contaminating the
        benchmark, which is why `jobs/` is gitignored. A frozen artifact has
        to be committed, so it must not carry the content.
        """
        import re

        allowed = {"trial", "task", "messages"}
        message_keys = {"role", "chars", "calls", "result_chars", "is_error"}
        # trial/task come straight from a run directory name. The previous
        # version never checked them, so arbitrary prose -- a patch, a
        # solution -- could sit in either field and this test still passed.
        identifier = re.compile(r"^[A-Za-z0-9._-]+$")
        for record in load_corpus():
            assert set(record) == allowed, f"unexpected keys: {set(record) - allowed}"
            assert identifier.match(record["trial"]), (
                f"trial is not an identifier: {record['trial']!r}"
            )
            assert identifier.match(record["task"]), (
                f"task is not an identifier: {record['task']!r}"
            )
            for shape in record["messages"]:
                assert set(shape) == message_keys
                assert shape["role"] in {"user", "assistant", "tool", "system"}
                assert isinstance(shape["chars"], int)
                assert isinstance(shape["is_error"], bool)
                assert shape["result_chars"] is None or isinstance(
                    shape["result_chars"], int
                )
                assert isinstance(shape["calls"], list)
                # A list of strings would slip past a naive isinstance check on
                # the value itself.
                assert all(isinstance(size, int) for size in shape["calls"]), (
                    f"calls must be integer lengths, got {shape['calls']!r}"
                )

    def test_S003_corpus_integrity_rejects_planted_text(self) -> None:
        # Negative: prove the check would reject prose in trial/task.
        import re

        identifier = re.compile(r"^[A-Za-z0-9._-]+$")
        assert not identifier.match("apply this patch: sed -i 's/foo/bar/' x.c")
        assert identifier.match("build-cython-ext__C2S4aUu")

    def test_S003_corpus_is_the_documented_size(self) -> None:
        corpus = load_corpus()
        assert 20 <= len(corpus) <= 30, (
            f"corpus holds {len(corpus)} trials; the plan asks for 20-30"
        )
        assert len({r["trial"] for r in corpus}) == len(corpus), "duplicate trial"

    def test_S003_every_window_is_traced(self, trace: dict) -> None:
        corpus = load_corpus()
        assert len(trace) == len(corpus) * len(REPLAY_WINDOWS)
