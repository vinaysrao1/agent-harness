"""S-106: which model serves which kind of call.

The plan called this a one-line change and said the spec existed to make it
*measurable*. Measuring it was the whole job: `RunStore.record_usage` had
exactly one caller — the loop's main model call — so the compaction
summarizer's tokens were recorded **nowhere**, and every cost figure the
harness produced excluded them.

That error was zero only because compaction never fires on this workload
(S-404). Routing the summarizer to a cheap model without counting it would
have shown up as a saving with no line item, because the expensive calls it
replaced were uncounted too.
"""

from __future__ import annotations

import pytest

from harness.config import HarnessConfig, ModelConfig
from harness.persistence import RunStore
from harness.routing import (
    CallPurpose,
    Router,
    UnroutableModelError,
    build_router,
)
from harness.types import Message, Role, ToolCall, Usage


class TestTheDefaultRoutesNowhere:
    def test_S106_an_unrouted_purpose_gets_the_runs_model(self) -> None:
        router = build_router("opus", {}, {"opus"})
        assert router.model_for(CallPurpose.MAIN) == "opus"
        assert router.model_for(CallPurpose.SUMMARIZE) == "opus"

    def test_S106_unrouted_means_reuse_the_object_not_an_equal_one(
        self,
    ) -> None:
        # `is_routed` is what the orchestrator asks before building a second
        # adapter. Returning True for an unrouted purpose would construct an
        # equivalent adapter instead of reusing the run's own, and "the
        # default path is unchanged" would become a claim about two objects
        # rather than a fact about one.
        router = build_router("opus", {}, {"opus"})
        assert not router.is_routed(CallPurpose.SUMMARIZE)

    def test_S106_routing_to_the_same_model_is_still_not_routed(self) -> None:
        router = build_router("opus", {"summarize": "opus"}, {"opus"})
        assert router.model_for(CallPurpose.SUMMARIZE) == "opus"
        assert not router.is_routed(CallPurpose.SUMMARIZE)

    def test_S106_the_config_default_is_empty(self) -> None:
        assert HarnessConfig().routing == {}


class TestRoutingIsAConfigChange:
    """Acceptance 2."""

    def test_S106_a_routed_purpose_resolves_elsewhere(self) -> None:
        router = build_router(
            "opus", {"summarize": "glm-flash"}, {"opus", "glm-flash"}
        )
        assert router.model_for(CallPurpose.MAIN) == "opus"
        assert router.model_for(CallPurpose.SUMMARIZE) == "glm-flash"
        assert router.is_routed(CallPurpose.SUMMARIZE)


class TestATypoRaisesRatherThanFallingBack:
    """Acceptance 4. A silent fallback is the failure where a run reports the
    cheap model in its config and bills the expensive one — and the only place
    that surfaces is a bill, weeks later, with nothing naming the purpose."""

    def test_S106_an_unknown_model_raises_and_names_the_purpose(self) -> None:
        with pytest.raises(UnroutableModelError, match="summarize") as caught:
            build_router("opus", {"summarize": "gpt-nope"}, {"opus"})
        assert "gpt-nope" in str(caught.value)
        assert "opus" in str(caught.value)  # lists what *is* available

    def test_S106_an_unknown_purpose_raises(self) -> None:
        # `[routing] summarise = "..."` is a plausible typo, and an ignored
        # key looks exactly like a working configuration.
        with pytest.raises(UnroutableModelError, match="summarise"):
            build_router("opus", {"summarise": "opus"}, {"opus"})

    def test_S106_the_error_names_the_known_purposes(self) -> None:
        with pytest.raises(UnroutableModelError) as caught:
            build_router("opus", {"lint": "opus"}, {"opus"})
        assert "summarize" in str(caught.value)


class TestOnlyPurposesWithCallSitesExist:
    """The plan names `main`, `summarize`, `classify`, `lint`. Only the first
    two are model calls here — `diligence.lint_verification` is a pure
    function over a command string. A purpose nothing can route is a config
    surface implying a mechanism that does not exist."""

    def test_S106_there_are_exactly_two_purposes(self) -> None:
        assert {p.value for p in CallPurpose} == {"main", "summarize"}

    def test_S106_lint_is_not_a_model_call(self) -> None:
        import inspect

        from harness.diligence import lint_verification

        assert not inspect.iscoroutinefunction(lint_verification)
        assert lint_verification("pytest -q") is not None

    def test_S106_a_purpose_is_its_own_string(self) -> None:
        # It lands in a SQLite column and a JSON payload; a plain `Enum`
        # would store `CallPurpose.SUMMARIZE` and every reader would have to
        # know that.
        assert CallPurpose.SUMMARIZE.value == "summarize"
        assert f"{CallPurpose.SUMMARIZE.value}" == "summarize"


class TestUsageCarriesThePurpose:
    """Acceptance 3."""

    def test_S106_a_recorded_row_defaults_to_main(self, tmp_path) -> None:
        with RunStore(tmp_path / "state.db") as store:
            run_id = store.create_run("goal", "m", "auto")
            agent_id = store.create_agent(run_id, "goal")
            store.record_usage(run_id, agent_id, "m", Usage(input_tokens=5))
            rows = store._conn.execute("SELECT purpose FROM usage").fetchall()
        assert [r["purpose"] for r in rows] == ["main"]

    def test_S106_a_summarizer_row_says_so(self, tmp_path) -> None:
        with RunStore(tmp_path / "state.db") as store:
            run_id = store.create_run("goal", "m", "auto")
            agent_id = store.create_agent(run_id, "goal")
            store.record_usage(
                run_id, agent_id, "cheap", Usage(input_tokens=5),
                purpose=CallPurpose.SUMMARIZE.value,
            )
            rows = store._conn.execute(
                "SELECT purpose, model FROM usage"
            ).fetchall()
        assert [(r["purpose"], r["model"]) for r in rows] == [
            ("summarize", "cheap")
        ]

    def test_S106_an_older_database_migrates(self, tmp_path) -> None:
        # `usage` predates the column, and every row already in one *is* a
        # main-model call, because `record_usage` had one caller.
        import sqlite3

        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE runs (id TEXT PRIMARY KEY);"
            "CREATE TABLE agents (id TEXT PRIMARY KEY);"
            "CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " run_id TEXT NOT NULL REFERENCES runs(id),"
            " agent_id TEXT REFERENCES agents(id), model TEXT NOT NULL,"
            " input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,"
            " cache_read_tokens INTEGER NOT NULL,"
            " cache_write_tokens INTEGER NOT NULL, created_at TEXT NOT NULL);"
            "INSERT INTO runs (id) VALUES ('r1');"
            "INSERT INTO usage (run_id, agent_id, model, input_tokens,"
            " output_tokens, cache_read_tokens, cache_write_tokens,"
            " created_at) VALUES ('r1', NULL, 'm', 1, 2, 0, 0, 'x');"
        )
        conn.commit()
        conn.close()

        with RunStore(path) as store:
            rows = store._conn.execute("SELECT purpose FROM usage").fetchall()
        assert [r["purpose"] for r in rows] == ["main"]


class TestTheSummarizerIsCountedAtAll:
    """The finding this spec exists for. Before S-106 the summarizer produced
    **zero** usage rows: `record_usage` had one caller and the summarizer was
    not it. Asserting the purpose without asserting the count would have
    tested a column on rows that never appear."""

    async def _run(self, tmp_path, compactions_wanted: int, routing=None):
        from harness.adapters.fake import FakeAdapter
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.types import Capabilities, ModelResponse, StopReason

        store = RunStore(tmp_path / "state.db")
        # A real script: a `fake` adapter with a bare model name gets an
        # *empty* script, so its first `complete` raises and the summarizer
        # takes the run down with it. The routed model has to be able to
        # actually answer.
        script_path = tmp_path / "cheap.jsonl"
        script_path.write_text(
            '{"content": "a summary"}\n' * 60, encoding="utf-8"
        )
        config = HarnessConfig(
            home=tmp_path / "home",
            models={
                "cheap": ModelConfig(adapter="fake", model=str(script_path)),
            },
            routing=routing or {},
        )
        orchestrator = Orchestrator(config, store)

        class _Narrow(FakeAdapter):
            @property
            def capabilities(self) -> Capabilities:
                return Capabilities(
                    max_context=2_000, supports_cache_control=False
                )

        script = [
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=f"step {i} " + "y" * 800,
                    tool_calls=[
                        ToolCall(
                            id=f"c{i}", name="note", arguments={"text": "ok"}
                        )
                    ],
                ),
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason=StopReason.TOOL_USE,
            )
            for i in range(30)
        ]
        from tests.test_loop import simple_tool

        run_id, _ = await orchestrator.run_task(
            "do the thing",
            "fake-model",
            adapter_override=_Narrow(script),
            budgets=Budgets(max_turns=compactions_wanted),
            tool_factories=[lambda deps: simple_tool("note")],
        )
        return store, run_id

    async def test_S106_the_summarizer_produces_usage_rows(
        self, tmp_path
    ) -> None:
        store, run_id = await self._run(tmp_path, 20)
        rows = store._conn.execute(
            "SELECT purpose, COUNT(*) AS n FROM usage WHERE run_id = ? "
            "GROUP BY purpose",
            (run_id,),
        ).fetchall()
        counts = {r["purpose"]: r["n"] for r in rows}
        assert counts.get("main", 0) > 0
        assert counts.get("summarize", 0) > 0, (
            "the summarizer's tokens are still recorded nowhere"
        )

    async def test_S106_an_unrouted_run_bills_the_summarizer_to_its_own_model(
        self, tmp_path
    ) -> None:
        store, run_id = await self._run(tmp_path, 20)
        models = {
            r["model"]
            for r in store._conn.execute(
                "SELECT DISTINCT model FROM usage WHERE purpose = 'summarize'"
            )
        }
        assert models == {"fake-model"}

    async def test_S106_a_routed_run_bills_the_summarizer_elsewhere(
        self, tmp_path
    ) -> None:
        # Acceptance 2, end to end: config only, no code change.
        store, run_id = await self._run(
            tmp_path, 20, routing={"summarize": "cheap"}
        )
        models = {
            r["model"]
            for r in store._conn.execute(
                "SELECT DISTINCT model FROM usage WHERE purpose = 'summarize'"
            )
        }
        assert models == {"cheap"}

        # And the routed adapter did the work. Asserting only the model
        # column left the orchestrator free to ignore routing entirely and
        # still label the row "cheap" -- the exact failure this spec says it
        # prevents, a run reporting the cheap model while billing the
        # expensive one. The routed script answers "a summary"; the run's own
        # adapter answers "step N ...".
        summaries = [
            e.payload["summary"]
            for e in store.load_events(store.list_agents(run_id)[0].id)
            if e.kind == "compaction"
        ]
        assert summaries, "the run never compacted"
        assert all("a summary" in (text or "") for text in summaries), summaries

    async def test_S106_an_unrouted_run_uses_its_own_adapter_for_summaries(
        self, tmp_path
    ) -> None:
        # The control: without routing, the summary comes from the run's own
        # adapter, so the routed assertion above is about routing and not
        # about the script happening to say that.
        store, run_id = await self._run(tmp_path, 20)
        summaries = [
            e.payload["summary"]
            for e in store.load_events(store.list_agents(run_id)[0].id)
            if e.kind == "compaction"
        ]
        assert summaries
        assert not any("a summary" in (text or "") for text in summaries)

    async def test_S106_a_typo_in_routing_stops_the_run(self, tmp_path) -> None:
        with pytest.raises(UnroutableModelError):
            await self._run(tmp_path, 20, routing={"summarize": "nope"})

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )


class TestEveryConfigFieldIsActuallyReadable:
    """`load_config` builds `HarnessConfig` from an explicit keyword list, so a
    field added to the model and not added there is silently unreachable from
    `config.toml` — the setting exists, validates, documents itself, and does
    nothing.

    `routing` (S-106) and `condenser` (S-105) both shipped that way. For
    `routing` it inverted this spec's own acceptance 4: `[routing] summarize =
    "gpt-typo"` was discarded before `build_router` saw it, so the run silently
    used the main model instead of raising — verbatim the failure the raise
    exists to prevent.
    """

    #: Not settable from TOML by design: it is the directory the config file
    #: was found in, resolved from `$HARNESS_HOME`.
    _NOT_FROM_TOML = {"home"}

    def test_S106_routing_survives_a_round_trip_through_toml(
        self, tmp_path
    ) -> None:
        from harness.config import load_config

        (tmp_path / "c.toml").write_text(
            'condenser = "summarize-halve"\n\n'
            '[models.cheap]\nadapter = "fake"\nmodel = "m2"\n\n'
            '[routing]\nsummarize = "cheap"\n',
            encoding="utf-8",
        )
        config = load_config(tmp_path / "c.toml")
        assert config.routing == {"summarize": "cheap"}
        assert config.condenser == "summarize-halve"

    def test_S106_a_routing_typo_in_a_real_config_file_raises(
        self, tmp_path
    ) -> None:
        # Acceptance 4 through the path an operator actually uses. Before the
        # field was read, this passed silently and billed the main model.
        from harness.config import load_config

        (tmp_path / "c.toml").write_text(
            '[models.cheap]\nadapter = "fake"\nmodel = "m2"\n\n'
            '[routing]\nsummarize = "gpt-typo"\n',
            encoding="utf-8",
        )
        config = load_config(tmp_path / "c.toml")
        assert config.routing == {"summarize": "gpt-typo"}
        with pytest.raises(UnroutableModelError):
            build_router("cheap", config.routing, set(config.models))

    def test_S106_every_declared_field_is_read_from_toml(self) -> None:
        # The guard. Reflects over the model rather than naming fields, so it
        # fails on the *next* one dropped rather than on these two.
        import inspect

        from harness.config import load_config

        source = inspect.getsource(load_config)
        missing = [
            name
            for name in HarnessConfig.model_fields
            if name not in self._NOT_FROM_TOML and f"{name}=" not in source
        ]
        assert not missing, (
            f"{missing} are declared on HarnessConfig but never read by "
            "load_config, so they cannot be set from config.toml"
        )


class TestTheHeadlineIsTokensNotRows:
    """`COUNT(*) > 0` passes when the tokens are recorded as zero. The failure
    message said "the summarizer's tokens are still recorded nowhere", which
    is exactly what a row of zeroes means."""

    async def test_S106_the_summarizer_rows_carry_real_tokens(
        self, tmp_path
    ) -> None:
        store, run_id = await TestTheSummarizerIsCountedAtAll()._run(
            tmp_path, 20
        )
        total = store._conn.execute(
            "SELECT SUM(input_tokens + output_tokens) AS t FROM usage "
            "WHERE run_id = ? AND purpose = 'summarize'",
            (run_id,),
        ).fetchone()["t"]
        assert total and total > 0, "summarizer rows exist but cost nothing"

    async def test_S106_the_summarizer_rows_carry_a_sane_duration(
        self, tmp_path
    ) -> None:
        # Reading the clock *after* the await, or forgetting to read it
        # before, reports a duration of monotonic-since-boot -- ~18 days --
        # straight into the §10.2 A5 figure the CLI prints.
        store, run_id = await TestTheSummarizerIsCountedAtAll()._run(
            tmp_path, 20
        )
        durations = [
            r["duration_ms"]
            for r in store._conn.execute(
                "SELECT duration_ms FROM usage WHERE run_id = ? "
                "AND purpose = 'summarize'",
                (run_id,),
            )
        ]
        assert durations
        assert all(0 <= d < 60_000 for d in durations), durations

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )


class TestAttributionIsPerAgent:
    """`build_context` takes the agent id so a subagent's summarizer bills
    that subagent. Shipped correct and asserted by nobody: both
    `agent_id=None` and `build_context(child_adapter, lead_agent_id)` passed
    the whole suite."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def test_S106_a_subagents_summaries_bill_the_subagent(
        self, tmp_path
    ) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.loop import Budgets
        from harness.orchestrator import Orchestrator
        from harness.types import (
            Capabilities,
            ModelResponse,
            StopReason,
        )

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(
            HarnessConfig(home=tmp_path / "home"), store
        )

        def turn(text, calls=None, stop=StopReason.TOOL_USE):
            return ModelResponse(
                message=Message(
                    role=Role.ASSISTANT, content=text, tool_calls=calls or []
                ),
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason=stop,
            )

        lead = [
            turn(
                "delegating",
                [ToolCall(id="s1", name="spawn_agent",
                          arguments={"prompt": "do the sub-task"})],
            ),
            # `spawn_agent` returns the id immediately; without an
            # `await_agents` the lead finishes and the child never gets far
            # enough to compact.
            turn(
                "waiting",
                [ToolCall(id="w1", name="await_agents", arguments={})],
            ),
            turn("Task complete. The subagent did it.", stop=StopReason.END_TURN),
        ]
        child = [
            turn(
                f"child step {i} " + "y" * 800,
                [ToolCall(id=f"c{i}", name="note", arguments={"text": "ok"})],
            )
            for i in range(30)
        ]

        class _Narrow(FakeAdapter):
            @property
            def capabilities(self) -> Capabilities:
                return Capabilities(
                    max_context=2_000, supports_cache_control=False
                )

        scripts = iter([FakeAdapter(lead), _Narrow(child)])
        from tests.test_loop import simple_tool

        run_id, _ = await orchestrator.run_task(
            "delegate it",
            "fake-model",
            adapter_override=lambda: next(scripts),
            budgets=Budgets(max_turns=25),
            tool_factories=[lambda deps: simple_tool("note")],
        )

        agents = {a.id: a for a in store.list_agents(run_id)}
        lead_id = next(a.id for a in agents.values() if a.parent_agent_id is None)
        billed = {
            r["agent_id"]
            for r in store._conn.execute(
                "SELECT DISTINCT agent_id FROM usage WHERE run_id = ? "
                "AND purpose = 'summarize'",
                (run_id,),
            )
        }
        assert billed, "the subagent never compacted; the test proves nothing"
        assert lead_id not in billed, (
            "the subagent's summaries were billed to the lead"
        )
        assert billed <= set(agents) - {lead_id}


class TestATypoDoesNotLeaveAnOrphanRun:
    """`UnknownModelError` is documented as raising before any row exists.
    Routing validated inside `_execute` instead, so a typo left a run stuck in
    `status='running'`, an agent row, and a workspace directory — for a
    mistake knowable from config alone."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def test_S106_no_run_row_survives_a_routing_typo(
        self, tmp_path
    ) -> None:
        from harness.adapters.fake import FakeAdapter
        from harness.orchestrator import Orchestrator

        store = RunStore(tmp_path / "state.db")
        orchestrator = Orchestrator(
            HarnessConfig(home=tmp_path / "home",
                          routing={"summarize": "nope"}),
            store,
        )
        with pytest.raises(UnroutableModelError):
            await orchestrator.run_task(
                "goal", "fake-model", adapter_override=FakeAdapter([])
            )
        assert store.list_runs() == []
