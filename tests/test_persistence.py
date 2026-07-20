"""Unit tests for harness.persistence."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest import mock

import pytest

from harness import persistence
from harness.persistence import (
    Agent,
    ApprovalRecord,
    InstructionLedgerItem,
    Run,
    RunStore,
    TaskLedgerItem,
    TranscriptEvent,
    UsageRecord,
)
from harness.types import Usage


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def store(db_path: Path) -> RunStore:
    s = RunStore(db_path)
    yield s
    s.close()


# -- runs ---------------------------------------------------------------------


def test_create_and_get_run(store: RunStore) -> None:
    run_id = store.create_run("ship the widget", "opus", "gated")
    run = store.get_run(run_id)
    assert isinstance(run, Run)
    assert run.id == run_id
    assert run.goal == "ship the widget"
    assert run.model == "opus"
    assert run.permission_mode == "gated"
    assert run.status == "running"
    # ISO-8601 timestamp, parseable, with a UTC offset.
    assert "T" in run.created_at
    assert run.created_at.endswith("+00:00")


def test_get_run_missing_returns_none(store: RunStore) -> None:
    assert store.get_run("does-not-exist") is None


def test_create_run_ids_are_unique_uuid4_hex(store: RunStore) -> None:
    ids = {store.create_run("g", "m", "gated") for _ in range(20)}
    assert len(ids) == 20
    for run_id in ids:
        assert len(run_id) == 32
        int(run_id, 16)  # valid hex


def test_list_runs_ordered(store: RunStore) -> None:
    r1 = store.create_run("first", "m", "gated")
    r2 = store.create_run("second", "m", "gated")
    runs = store.list_runs()
    assert [r.id for r in runs] == [r1, r2]


def test_list_runs_ordered_survives_identical_timestamps(store: RunStore) -> None:
    # created_at ties (same-microsecond writes) used to be broken by the
    # random uuid4 id, which carries no insertion-order information. Only
    # ordering by rowid (insertion order) upholds the "oldest first"
    # contract in that case.
    fixed_ts = "2024-01-01T00:00:00+00:00"
    with mock.patch.object(persistence, "_utc_now_iso", return_value=fixed_ts):
        r1 = store.create_run("first", "m", "gated")
        r2 = store.create_run("second", "m", "gated")
        r3 = store.create_run("third", "m", "gated")
    runs = store.list_runs()
    assert all(r.created_at == fixed_ts for r in runs)
    assert [r.id for r in runs] == [r1, r2, r3]


def test_set_run_status(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    store.set_run_status(run_id, "completed")
    assert store.get_run(run_id).status == "completed"


def test_set_run_status_unknown_raises(store: RunStore) -> None:
    with pytest.raises(KeyError):
        store.set_run_status("nope", "completed")


# -- agents ---------------------------------------------------------------------


def test_create_and_get_agent(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "do the thing")
    agent = store.get_agent(agent_id)
    assert isinstance(agent, Agent)
    assert agent.run_id == run_id
    assert agent.prompt == "do the thing"
    assert agent.parent_agent_id is None
    assert agent.status == "running"


def test_create_agent_with_parent(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    parent_id = store.create_agent(run_id, "orchestrate")
    child_id = store.create_agent(run_id, "subtask", parent_agent_id=parent_id)
    child = store.get_agent(child_id)
    assert child.parent_agent_id == parent_id


def test_list_agents_ordered_survives_identical_timestamps(store: RunStore) -> None:
    # Same rationale as test_list_runs_ordered_survives_identical_timestamps:
    # tied created_at values must not fall back to random-uuid ordering.
    run_id = store.create_run("g", "m", "gated")
    fixed_ts = "2024-01-01T00:00:00+00:00"
    with mock.patch.object(persistence, "_utc_now_iso", return_value=fixed_ts):
        a1 = store.create_agent(run_id, "one")
        a2 = store.create_agent(run_id, "two")
        a3 = store.create_agent(run_id, "three")
    agents = store.list_agents(run_id)
    assert all(a.created_at == fixed_ts for a in agents)
    assert [a.id for a in agents] == [a1, a2, a3]


def test_create_agent_unknown_run_raises_foreign_key_error(store: RunStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.create_agent("no-such-run", "prompt")


def test_set_agent_status(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    store.set_agent_status(agent_id, "done")
    assert store.get_agent(agent_id).status == "done"


def test_set_agent_status_unknown_raises(store: RunStore) -> None:
    with pytest.raises(KeyError):
        store.set_agent_status("nope", "done")


def test_list_agents(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    a1 = store.create_agent(run_id, "one")
    a2 = store.create_agent(run_id, "two")
    agents = store.list_agents(run_id)
    assert [a.id for a in agents] == [a1, a2]


# -- transcript events ------------------------------------------------------------


def test_append_event_seq_starts_at_one_and_increments(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    seq1 = store.append_event(agent_id, "message", {"role": "user", "text": "hi"})
    seq2 = store.append_event(agent_id, "message", {"role": "assistant", "text": "yo"})
    seq3 = store.append_event(agent_id, "tool_call", {"name": "bash"})
    assert (seq1, seq2, seq3) == (1, 2, 3)


def test_append_event_unknown_agent_raises(store: RunStore) -> None:
    with pytest.raises(KeyError):
        store.append_event("no-such-agent", "message", {})


def test_load_events_ordered_by_seq(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    for i in range(5):
        store.append_event(agent_id, "message", {"i": i})
    events = store.load_events(agent_id)
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert [e.payload["i"] for e in events] == [0, 1, 2, 3, 4]
    assert all(isinstance(e, TranscriptEvent) for e in events)
    assert all(e.kind == "message" for e in events)


def test_two_agents_interleaved_have_independent_seq(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_a = store.create_agent(run_id, "a")
    agent_b = store.create_agent(run_id, "b")

    # Interleave appends across the two agents.
    a1 = store.append_event(agent_a, "message", {"who": "a", "n": 1})
    b1 = store.append_event(agent_b, "message", {"who": "b", "n": 1})
    a2 = store.append_event(agent_a, "message", {"who": "a", "n": 2})
    b2 = store.append_event(agent_b, "message", {"who": "b", "n": 2})
    a3 = store.append_event(agent_a, "message", {"who": "a", "n": 3})

    assert (a1, a2, a3) == (1, 2, 3)
    assert (b1, b2) == (1, 2)

    events_a = store.load_events(agent_a)
    events_b = store.load_events(agent_b)
    assert [e.seq for e in events_a] == [1, 2, 3]
    assert [e.seq for e in events_b] == [1, 2]
    assert all(e.payload["who"] == "a" for e in events_a)
    assert all(e.payload["who"] == "b" for e in events_b)


def test_load_run_events_covers_all_agents_in_insertion_order(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_a = store.create_agent(run_id, "a")
    agent_b = store.create_agent(run_id, "b")

    store.append_event(agent_a, "message", {"tag": "a1"})
    store.append_event(agent_b, "message", {"tag": "b1"})
    store.append_event(agent_a, "message", {"tag": "a2"})

    events = store.load_run_events(run_id)
    assert [e.payload["tag"] for e in events] == ["a1", "b1", "a2"]
    assert {e.agent_id for e in events} == {agent_a, agent_b}


def test_events_from_other_runs_excluded(store: RunStore) -> None:
    run1 = store.create_run("g1", "m", "gated")
    run2 = store.create_run("g2", "m", "gated")
    agent1 = store.create_agent(run1, "a")
    agent2 = store.create_agent(run2, "b")
    store.append_event(agent1, "message", {"tag": "run1"})
    store.append_event(agent2, "message", {"tag": "run2"})

    events1 = store.load_run_events(run1)
    assert [e.payload["tag"] for e in events1] == ["run1"]


def test_duplicate_seq_rejected_at_db_level(store: RunStore) -> None:
    # UNIQUE(agent_id, seq) is the backstop: even a direct low-level insert
    # bypassing append_event() cannot create two rows with the same seq for
    # one agent.
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    store.append_event(agent_id, "message", {})
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO transcript_events (run_id, agent_id, seq, kind, "
            "payload, created_at) VALUES (?, ?, 1, 'message', '{}', 'x')",
            (run_id, agent_id),
        )


def test_transcript_events_has_no_public_mutation_api(store: RunStore) -> None:
    # Append-only enforcement: there must be no update/delete entry point for
    # transcript events, only append_event() and the two loaders.
    public_names = {name for name in dir(store) if not name.startswith("_")}
    forbidden_substrings = ("update_event", "delete_event", "remove_event", "edit_event")
    for name in public_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in name.lower()


def test_no_public_method_ever_issues_update_or_delete_on_transcript_events(
    store: RunStore,
) -> None:
    # Behavioral version of the append-only guarantee: rather than trusting
    # method *names*, install an SQLite authorizer that vetoes any UPDATE or
    # DELETE against transcript_events, then drive every public RunStore
    # method that could plausibly touch that table through a representative
    # workflow. If anyone ever adds a mutation path under a different name
    # (set_event, rewrite_event, purge_events, ...), this fails loudly.
    def authorizer(action, arg1, arg2, db_name, trigger_name):
        if action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            if arg1 == "transcript_events":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    run_id = store.create_run("g", "m", "gated")
    agent_a = store.create_agent(run_id, "a")
    agent_b = store.create_agent(run_id, "b")

    store._conn.set_authorizer(authorizer)
    try:
        store.append_event(agent_a, "message", {"n": 1})
        store.append_event(agent_b, "message", {"n": 1})
        store.append_event(agent_a, "message", {"n": 2})
        store.load_events(agent_a)
        store.load_run_events(run_id)
        store.set_run_status(run_id, "completed")
        store.set_agent_status(agent_a, "done")
        store.upsert_task_item(run_id, "t1", "desc", "done")
        store.upsert_instruction(run_id, "i1", "instr", "user")
        store.record_approval(run_id, agent_a, "bash", {}, "allow", "policy")
        store.record_usage(run_id, agent_a, "opus", Usage(input_tokens=1, output_tokens=1))
        store.total_usage(run_id)
    finally:
        store._conn.set_authorizer(None)

    # Snapshot corroborates the authorizer: the rows written are still
    # exactly the three appended, byte-for-byte.
    rows = store._conn.execute(
        "SELECT agent_id, seq, kind, payload FROM transcript_events ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (agent_a, 1, "message", '{"n": 1}'),
        (agent_b, 1, "message", '{"n": 1}'),
        (agent_a, 2, "message", '{"n": 2}'),
    ]


def test_append_event_is_atomic_across_concurrent_connections(db_path: Path) -> None:
    # Regression test for the seq-assignment race: two separate RunStore
    # connections (simulating separate threads/processes on the same
    # db_path, as the module docstring says is supported) appending for the
    # *same* agent concurrently must never collide on seq or raise
    # IntegrityError — the read-then-write must be atomic across
    # connections, not just within one.
    setup = RunStore(db_path)
    run_id = setup.create_run("g", "m", "gated")
    agent_id = setup.create_agent(run_id, "prompt")
    setup.close()

    n_threads = 4
    n_per_thread = 15
    errors: list[BaseException] = []

    def worker() -> None:
        s = RunStore(db_path)
        try:
            for i in range(n_per_thread):
                s.append_event(agent_id, "message", {"i": i})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []

    verify = RunStore(db_path)
    try:
        events = verify.load_events(agent_id)
        seqs = sorted(e.seq for e in events)
        assert len(events) == n_threads * n_per_thread
        assert seqs == list(range(1, n_threads * n_per_thread + 1))
    finally:
        verify.close()


# -- task ledger ---------------------------------------------------------------


def test_task_ledger_upsert_creates_then_updates(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    store.upsert_task_item(run_id, "task-1", "write tests", "in_progress")
    items = store.list_task_items(run_id)
    assert len(items) == 1
    assert isinstance(items[0], TaskLedgerItem)
    assert items[0].status == "in_progress"
    assert items[0].evidence is None

    store.upsert_task_item(
        run_id, "task-1", "write tests", "done", evidence="pytest output: 12 passed"
    )
    items = store.list_task_items(run_id)
    assert len(items) == 1  # still one row, not a duplicate
    assert items[0].status == "done"
    assert items[0].evidence == "pytest output: 12 passed"


def test_task_ledger_multiple_items_ordered(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    store.upsert_task_item(run_id, "b", "second", "open")
    store.upsert_task_item(run_id, "a", "first", "open")
    items = store.list_task_items(run_id)
    assert [i.item_id for i in items] == ["a", "b"]


# -- instruction ledger ----------------------------------------------------------


def test_instruction_ledger_upsert_creates_then_updates(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    store.upsert_instruction(run_id, "inst-1", "never push to main", "user")
    items = store.list_instructions(run_id)
    assert len(items) == 1
    assert isinstance(items[0], InstructionLedgerItem)
    assert items[0].instruction == "never push to main"
    assert items[0].source == "user"

    store.upsert_instruction(run_id, "inst-1", "never force-push to main", "user")
    items = store.list_instructions(run_id)
    assert len(items) == 1
    assert items[0].instruction == "never force-push to main"


# -- approvals -------------------------------------------------------------------


def test_record_and_list_approvals(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    store.record_approval(
        run_id, agent_id, "bash", {"cmd": "rm -rf /"}, "deny", "user"
    )
    store.record_approval(
        run_id, agent_id, "read_file", {"path": "a.txt"}, "allow", "policy"
    )
    approvals = store.list_approvals(run_id)
    assert len(approvals) == 2
    assert isinstance(approvals[0], ApprovalRecord)
    assert approvals[0].tool_name == "bash"
    assert approvals[0].arguments == {"cmd": "rm -rf /"}
    assert approvals[0].decision == "deny"
    assert approvals[1].decision == "allow"
    assert approvals[1].decided_by == "policy"


# -- usage -------------------------------------------------------------------------


def test_record_usage_and_list(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    store.record_usage(
        run_id, agent_id, "opus", Usage(input_tokens=100, output_tokens=50)
    )
    records = store.list_usage(run_id)
    assert len(records) == 1
    assert isinstance(records[0], UsageRecord)
    assert records[0].usage.input_tokens == 100
    assert records[0].usage.output_tokens == 50


def test_total_usage_aggregates_across_agents_and_models(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    agent_a = store.create_agent(run_id, "a")
    agent_b = store.create_agent(run_id, "b")

    store.record_usage(
        run_id,
        agent_a,
        "opus",
        Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=5,
            cache_write_tokens=1,
        ),
    )
    store.record_usage(
        run_id,
        agent_b,
        "kimi",
        Usage(
            input_tokens=30,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
    )

    total = store.total_usage(run_id)
    assert total == {
        "input_tokens": 130,
        "output_tokens": 30,
        "cache_read_tokens": 5,
        "cache_write_tokens": 1,
        "duration_ms": 0,
    }


def test_total_usage_empty_run_is_all_zero(store: RunStore) -> None:
    run_id = store.create_run("g", "m", "gated")
    assert store.total_usage(run_id) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "duration_ms": 0,
    }


def test_record_usage_with_duration_and_aggregate(store: RunStore) -> None:
    """A5: per-call duration_ms is stored, listed, and summed by
    total_usage; omitting it defaults to 0."""
    run_id = store.create_run("g", "m", "gated")
    agent_id = store.create_agent(run_id, "prompt")
    store.record_usage(
        run_id,
        agent_id,
        "opus",
        Usage(input_tokens=1, output_tokens=1),
        duration_ms=1500,
    )
    store.record_usage(
        run_id,
        agent_id,
        "opus",
        Usage(input_tokens=2, output_tokens=2),
        duration_ms=250,
    )
    store.record_usage(
        run_id, agent_id, "opus", Usage(input_tokens=3, output_tokens=3)
    )

    records = store.list_usage(run_id)
    assert [record.duration_ms for record in records] == [1500, 250, 0]
    assert store.total_usage(run_id)["duration_ms"] == 1750


def test_open_pre_duration_schema_db_migrates_cleanly(db_path: Path) -> None:
    """A5 migration: a database created before the ``duration_ms`` column
    existed opens without crashing; old rows read back as duration 0 and
    new rows record durations normally."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            id               TEXT PRIMARY KEY,
            created_at       TEXT NOT NULL,
            goal             TEXT NOT NULL,
            model            TEXT NOT NULL,
            permission_mode  TEXT NOT NULL,
            status           TEXT NOT NULL
        );
        CREATE TABLE agents (
            id               TEXT PRIMARY KEY,
            run_id           TEXT NOT NULL REFERENCES runs(id),
            parent_agent_id  TEXT REFERENCES agents(id),
            prompt           TEXT NOT NULL,
            status           TEXT NOT NULL,
            created_at       TEXT NOT NULL
        );
        CREATE TABLE usage (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL REFERENCES runs(id),
            agent_id            TEXT REFERENCES agents(id),
            model               TEXT NOT NULL,
            input_tokens        INTEGER NOT NULL,
            output_tokens       INTEGER NOT NULL,
            cache_read_tokens   INTEGER NOT NULL,
            cache_write_tokens  INTEGER NOT NULL,
            created_at          TEXT NOT NULL
        );
        INSERT INTO runs VALUES ('r1', 't0', 'old goal', 'opus', 'gated', 'completed');
        INSERT INTO agents VALUES ('a1', 'r1', NULL, 'old prompt', 'completed', 't0');
        INSERT INTO usage (run_id, agent_id, model, input_tokens,
            output_tokens, cache_read_tokens, cache_write_tokens, created_at)
        VALUES ('r1', 'a1', 'opus', 10, 5, 0, 0, 't0');
        """
    )
    conn.commit()
    conn.close()

    store = RunStore(db_path)  # must not crash on the old schema
    try:
        records = store.list_usage("r1")
        assert len(records) == 1
        assert records[0].duration_ms == 0
        assert records[0].usage.input_tokens == 10

        store.record_usage(
            "r1", "a1", "opus", Usage(input_tokens=1), duration_ms=42
        )
        assert store.total_usage("r1")["duration_ms"] == 42

        # Reopening again (already-migrated schema) is a no-op, not an error.
        store.close()
        store = RunStore(db_path)
        assert [r.duration_ms for r in store.list_usage("r1")] == [0, 42]
    finally:
        store.close()


def test_total_usage_excludes_other_runs(store: RunStore) -> None:
    run1 = store.create_run("g1", "m", "gated")
    run2 = store.create_run("g2", "m", "gated")
    agent1 = store.create_agent(run1, "a")
    agent2 = store.create_agent(run2, "b")
    store.record_usage(run1, agent1, "opus", Usage(input_tokens=10, output_tokens=1))
    store.record_usage(run2, agent2, "opus", Usage(input_tokens=999, output_tokens=1))
    assert store.total_usage(run1)["input_tokens"] == 10


# -- WAL mode ------------------------------------------------------------------


def test_wal_mode_is_enabled(store: RunStore) -> None:
    row = store._conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"


def test_foreign_keys_enabled(store: RunStore) -> None:
    row = store._conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


# -- resume flow -----------------------------------------------------------------


def test_resume_flow_reopen_reloads_identical_state(db_path: Path) -> None:
    store1 = RunStore(db_path)
    run_id = store1.create_run("long task", "opus", "gated")
    agent_id = store1.create_agent(run_id, "do work")
    store1.append_event(agent_id, "message", {"role": "user", "text": "start"})
    store1.append_event(agent_id, "tool_call", {"name": "bash", "args": {"cmd": "ls"}})
    store1.upsert_task_item(run_id, "t1", "list files", "done", evidence="ls output")
    store1.upsert_instruction(run_id, "i1", "never delete files", "user")
    store1.record_usage(run_id, agent_id, "opus", Usage(input_tokens=42, output_tokens=7))
    store1.set_run_status(run_id, "paused")
    store1.close()

    # Simulate a crash + `harness resume <run_id>`: a brand-new RunStore on
    # the same db_path must see exactly what was written.
    store2 = RunStore(db_path)
    try:
        run = store2.get_run(run_id)
        assert run is not None
        assert run.status == "paused"
        assert run.goal == "long task"

        events = store2.load_events(agent_id)
        assert [e.kind for e in events] == ["message", "tool_call"]
        assert events[0].payload == {"role": "user", "text": "start"}
        assert events[1].payload == {"name": "bash", "args": {"cmd": "ls"}}

        run_events = store2.load_run_events(run_id)
        assert len(run_events) == 2

        task_items = store2.list_task_items(run_id)
        assert task_items[0].status == "done"
        assert task_items[0].evidence == "ls output"

        instructions = store2.list_instructions(run_id)
        assert instructions[0].instruction == "never delete files"

        assert store2.total_usage(run_id) == {
            "input_tokens": 42,
            "output_tokens": 7,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "duration_ms": 0,
        }

        # Resumed store can keep appending right after the last seq.
        next_seq = store2.append_event(agent_id, "message", {"role": "assistant"})
        assert next_seq == 3
    finally:
        store2.close()


def test_resume_preserves_multi_agent_seq_independently(db_path: Path) -> None:
    store1 = RunStore(db_path)
    run_id = store1.create_run("g", "m", "gated")
    agent_a = store1.create_agent(run_id, "a")
    agent_b = store1.create_agent(run_id, "b")
    store1.append_event(agent_a, "message", {"n": 1})
    store1.append_event(agent_a, "message", {"n": 2})
    store1.append_event(agent_b, "message", {"n": 1})
    store1.close()

    store2 = RunStore(db_path)
    try:
        assert store2.append_event(agent_a, "message", {"n": 3}) == 3
        assert store2.append_event(agent_b, "message", {"n": 2}) == 2
    finally:
        store2.close()


# -- context manager -----------------------------------------------------------


def test_context_manager_closes(db_path: Path) -> None:
    with RunStore(db_path) as s:
        run_id = s.create_run("g", "m", "gated")
        assert s.get_run(run_id) is not None
    with pytest.raises(sqlite3.ProgrammingError):
        s._conn.execute("SELECT 1")
