"""SQLite persistence and crash recovery (DESIGN.md §4.10).

:class:`RunStore` is the single durability layer for the harness: every run,
agent, transcript event, ledger entry, approval decision, and usage record
lands here. The design goals straight from DESIGN.md §4.10 are:

- **WAL mode** so readers (e.g. a `harness cost`/`harness resume` CLI
  invocation) never block a concurrently-running agent loop's writer.
- **Foreign keys on** so orphaned agents/events can't silently accumulate.
- **`transcript_events` is append-only.** There is deliberately no public API
  that updates or deletes a row in that table — resuming a run replays it
  from the beginning, never mutates history.
- **Resume-ability.** Reopening a :class:`RunStore` against the same
  ``db_path`` after a crash reconstructs identical state; nothing lives only
  in process memory.

Threading model
----------------
:class:`RunStore` wraps one :mod:`sqlite3` connection opened with the default
``check_same_thread=True``. It is **not** a connection pool and does no
internal locking. It is safe to call from `async def` code as plain
(blocking) sync calls — sqlite operations here are small and fast enough that
running them inline on the event loop thread is the intended usage — but
**all calls for a given instance must happen on the one thread/event loop
that constructed it**. Handing the instance to `asyncio.to_thread` or a
thread pool, or sharing it across event loops, will raise a
``sqlite3.ProgrammingError`` (by design: that's the guardrail catching a
violation of this contract rather than silently corrupting state). Separate
processes/threads may each open their own :class:`RunStore` on the same
``db_path`` — WAL mode is precisely what makes that safe for readers running
concurrently with a writer. For writers, SQLite's single-writer-at-a-time
rule is what serializes them: :meth:`RunStore.append_event` wraps its
read-then-write ``seq`` computation in an explicit ``BEGIN IMMEDIATE``
transaction so two connections appending for the same agent at the same
time queue up rather than racing on ``MAX(seq)``; every other mutation here
is a single atomic statement, which SQLite already serializes on its own.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict

from harness.types import Usage

__all__ = [
    "Run",
    "Agent",
    "TranscriptEvent",
    "TaskLedgerItem",
    "InstructionLedgerItem",
    "ApprovalRecord",
    "UsageRecord",
    "RunStore",
]


def _utc_now_iso() -> str:
    """Return the current time as an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Return a fresh id: a uuid4 in bare hex form (no dashes)."""
    return uuid.uuid4().hex


class Run(BaseModel):
    """A row of the `runs` table (DESIGN.md §4.10)."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: str
    goal: str
    model: str
    permission_mode: str
    status: str


class Agent(BaseModel):
    """A row of the `agents` table.

    ``parent_agent_id`` is ``None`` for the run's lead/orchestrator agent and
    set for spawned subagents (DESIGN.md §4.12).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    run_id: str
    parent_agent_id: str | None
    prompt: str
    status: str
    created_at: str


class TranscriptEvent(BaseModel):
    """A row of the append-only `transcript_events` table.

    ``seq`` is a per-``agent_id`` monotonically increasing counter assigned
    by :meth:`RunStore.append_event`; ``id`` is the table-wide autoincrement
    primary key, which also happens to reflect global insertion order across
    every agent in a run.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    run_id: str
    agent_id: str
    seq: int
    kind: str
    payload: dict
    created_at: str


class TaskLedgerItem(BaseModel):
    """A row of the `task_ledger` table (DESIGN.md §4.9)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    item_id: str
    description: str
    status: str
    evidence: str | None
    updated_at: str


class InstructionLedgerItem(BaseModel):
    """A row of the `instruction_ledger` table (DESIGN.md §4.5)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    item_id: str
    instruction: str
    source: str
    updated_at: str


class ApprovalRecord(BaseModel):
    """A row of the `approvals` table (DESIGN.md §4.11).

    ``decision`` and ``decided_by`` are stored as plain strings rather than
    tied to the permission engine's ``Decision`` enum (owned by another
    module) so this module has no dependency on it; callers pass the enum's
    ``.value`` or any string label they like (e.g. ``"user"``/``"policy"``
    for ``decided_by``).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    run_id: str
    agent_id: str
    tool_name: str
    arguments: dict
    decision: str
    decided_by: str
    created_at: str


class UsageRecord(BaseModel):
    """A row of the `usage` table."""

    model_config = ConfigDict(frozen=True)

    id: int
    run_id: str
    agent_id: str | None
    model: str
    usage: Usage
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    goal             TEXT NOT NULL,
    model            TEXT NOT NULL,
    permission_mode  TEXT NOT NULL,
    status           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(id),
    parent_agent_id  TEXT REFERENCES agents(id),
    prompt           TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    agent_id    TEXT NOT NULL REFERENCES agents(id),
    seq         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (agent_id, seq)
);

CREATE TABLE IF NOT EXISTS task_ledger (
    run_id       TEXT NOT NULL REFERENCES runs(id),
    item_id      TEXT NOT NULL,
    description  TEXT NOT NULL,
    status       TEXT NOT NULL,
    evidence     TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS instruction_ledger (
    run_id       TEXT NOT NULL REFERENCES runs(id),
    item_id      TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    source       TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    agent_id        TEXT NOT NULL REFERENCES agents(id),
    tool_name       TEXT NOT NULL,
    arguments_json  TEXT NOT NULL,
    decision        TEXT NOT NULL,
    decided_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
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
"""


class RunStore:
    """SQLite-backed persistence for runs, agents, transcripts, and ledgers.

    See the module docstring for the threading contract. All timestamps are
    ISO-8601 UTC strings; all ids are uuid4 hex strings minted by the store
    itself (callers never supply their own).
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (creating if necessary) the SQLite database at ``db_path``.

        Applies the schema (idempotent — ``CREATE TABLE IF NOT EXISTS``),
        turns on WAL journaling and foreign-key enforcement.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "RunStore":
        """Support ``with RunStore(path) as store: ...``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection when leaving a ``with`` block."""
        self.close()

    # -- runs -----------------------------------------------------------

    def create_run(
        self,
        goal: str,
        model: str,
        permission_mode: str,
        *,
        status: str = "running",
    ) -> str:
        """Insert a new run row and return its freshly minted id."""
        run_id = _new_id()
        created_at = _utc_now_iso()
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, created_at, goal, model, "
                "permission_mode, status) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, created_at, goal, model, permission_mode, status),
            )
        return run_id

    def get_run(self, run_id: str) -> Run | None:
        """Fetch one run by id, or ``None`` if it doesn't exist."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return Run(**dict(row)) if row is not None else None

    def list_runs(self) -> list[Run]:
        """List every run, oldest first.

        Ordered by SQLite's implicit ``rowid`` rather than ``created_at``:
        ``created_at`` has only microsecond resolution and ties would
        otherwise be broken by the (random, insertion-order-free) uuid4
        ``id``, so ``rowid`` — which is assigned in strict insertion order —
        is the only column that actually guarantees "oldest first".
        """
        rows = self._conn.execute("SELECT * FROM runs ORDER BY rowid").fetchall()
        return [Run(**dict(row)) for row in rows]

    def set_run_status(self, run_id: str, status: str) -> None:
        """Update a run's status. Raises :class:`KeyError` if unknown."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?", (status, run_id)
            )
            if cur.rowcount == 0:
                raise KeyError(f"no such run: {run_id!r}")

    # -- agents -----------------------------------------------------------

    def create_agent(
        self,
        run_id: str,
        prompt: str,
        *,
        parent_agent_id: str | None = None,
        status: str = "running",
    ) -> str:
        """Insert a new agent row under ``run_id`` and return its id.

        ``parent_agent_id`` links a spawned subagent to its orchestrator
        (DESIGN.md §4.12); leave it ``None`` for a run's lead agent.
        """
        agent_id = _new_id()
        created_at = _utc_now_iso()
        with self._conn:
            self._conn.execute(
                "INSERT INTO agents (id, run_id, parent_agent_id, prompt, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, run_id, parent_agent_id, prompt, status, created_at),
            )
        return agent_id

    def get_agent(self, agent_id: str) -> Agent | None:
        """Fetch one agent by id, or ``None`` if it doesn't exist."""
        row = self._conn.execute(
            "SELECT * FROM agents WHERE id = ?", (agent_id,)
        ).fetchone()
        return Agent(**dict(row)) if row is not None else None

    def list_agents(self, run_id: str) -> list[Agent]:
        """List every agent belonging to ``run_id``, in creation order.

        Ordered by SQLite's implicit ``rowid`` for the same reason as
        :meth:`list_runs` — it reflects true insertion order, unlike
        ``created_at`` ties broken by a random uuid4 ``id``.
        """
        rows = self._conn.execute(
            "SELECT * FROM agents WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        return [Agent(**dict(row)) for row in rows]

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Update an agent's status. Raises :class:`KeyError` if unknown."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE agents SET status = ? WHERE id = ?", (status, agent_id)
            )
            if cur.rowcount == 0:
                raise KeyError(f"no such agent: {agent_id!r}")

    # -- transcript events (append-only) -----------------------------------

    def append_event(self, agent_id: str, kind: str, payload: dict) -> int:
        """Append one transcript event for ``agent_id`` and return its ``seq``.

        ``seq`` is a per-agent counter starting at 1 that increases by
        exactly 1 on each call. The read-then-write that computes it is
        wrapped in an explicit ``BEGIN IMMEDIATE`` transaction, so the seq
        read and the insert are atomic with respect to *any* other
        connection on the same database file (not just other callers on
        this instance) — a second connection's ``BEGIN IMMEDIATE`` blocks
        until the first commits rather than racing to read the same
        ``MAX(seq)``. The ``UNIQUE(agent_id, seq)`` constraint remains as a
        backstop that would turn any remaining violation into a loud
        :class:`sqlite3.IntegrityError` instead of silent corruption, but it
        should never actually fire in practice now.

        This is the *only* write path onto `transcript_events`; there is
        deliberately no update or delete method — the log is append-only by
        construction, per DESIGN.md §4.10.
        """
        payload_json = json.dumps(payload)
        created_at = _utc_now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            agent_row = self._conn.execute(
                "SELECT run_id FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if agent_row is None:
                raise KeyError(f"no such agent: {agent_id!r}")
            run_id = agent_row["run_id"]
            (next_seq,) = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM transcript_events "
                "WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO transcript_events (run_id, agent_id, seq, kind, "
                "payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, agent_id, next_seq, kind, payload_json, created_at),
            )
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()
        return next_seq

    def load_events(self, agent_id: str) -> list[TranscriptEvent]:
        """Load all events for one agent, ordered by ``seq``."""
        rows = self._conn.execute(
            "SELECT * FROM transcript_events WHERE agent_id = ? ORDER BY seq",
            (agent_id,),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def load_run_events(self, run_id: str) -> list[TranscriptEvent]:
        """Load all events for every agent in a run, in global insertion order.

        Ordering is by the autoincrement primary key, which reflects the
        actual order events were appended across all of the run's agents —
        the order needed to reconstruct a run on `harness resume`.
        """
        rows = self._conn.execute(
            "SELECT * FROM transcript_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    # -- task ledger --------------------------------------------------------

    def upsert_task_item(
        self,
        run_id: str,
        item_id: str,
        description: str,
        status: str,
        evidence: str | None = None,
    ) -> None:
        """Create or update one task-ledger item (DESIGN.md §4.9).

        Keyed on ``(run_id, item_id)``: calling this again with the same
        ``item_id`` overwrites the previous description/status/evidence
        rather than adding a duplicate row.
        """
        updated_at = _utc_now_iso()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO task_ledger
                    (run_id, item_id, description, status, evidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (run_id, item_id) DO UPDATE SET
                    description = excluded.description,
                    status = excluded.status,
                    evidence = excluded.evidence,
                    updated_at = excluded.updated_at
                """,
                (run_id, item_id, description, status, evidence, updated_at),
            )

    def list_task_items(self, run_id: str) -> list[TaskLedgerItem]:
        """List every task-ledger item for a run, ordered by ``item_id``."""
        rows = self._conn.execute(
            "SELECT * FROM task_ledger WHERE run_id = ? ORDER BY item_id",
            (run_id,),
        ).fetchall()
        return [TaskLedgerItem(**dict(row)) for row in rows]

    # -- instruction ledger ---------------------------------------------------

    def upsert_instruction(
        self,
        run_id: str,
        item_id: str,
        instruction: str,
        source: str,
    ) -> None:
        """Create or update one instruction-ledger item (DESIGN.md §4.5).

        Keyed on ``(run_id, item_id)`` like :meth:`upsert_task_item`.
        """
        updated_at = _utc_now_iso()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO instruction_ledger
                    (run_id, item_id, instruction, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (run_id, item_id) DO UPDATE SET
                    instruction = excluded.instruction,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (run_id, item_id, instruction, source, updated_at),
            )

    def list_instructions(self, run_id: str) -> list[InstructionLedgerItem]:
        """List every instruction-ledger item for a run, ordered by ``item_id``."""
        rows = self._conn.execute(
            "SELECT * FROM instruction_ledger WHERE run_id = ? ORDER BY item_id",
            (run_id,),
        ).fetchall()
        return [InstructionLedgerItem(**dict(row)) for row in rows]

    # -- approvals ------------------------------------------------------------

    def record_approval(
        self,
        run_id: str,
        agent_id: str,
        tool_name: str,
        arguments: dict,
        decision: str,
        decided_by: str,
    ) -> int:
        """Log one permission-engine decision (DESIGN.md §4.11) and return its id.

        Every decision is logged here, including auto-allows — ``decision``
        and ``decided_by`` are free-form strings (see :class:`ApprovalRecord`)
        so this module stays independent of the permission engine's enums.
        """
        created_at = _utc_now_iso()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO approvals (run_id, agent_id, tool_name, "
                "arguments_json, decision, decided_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    agent_id,
                    tool_name,
                    json.dumps(arguments),
                    decision,
                    decided_by,
                    created_at,
                ),
            )
        return int(cur.lastrowid)

    def list_approvals(self, run_id: str) -> list[ApprovalRecord]:
        """List every approval decision for a run, in the order recorded."""
        rows = self._conn.execute(
            "SELECT * FROM approvals WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            ApprovalRecord(
                id=row["id"],
                run_id=row["run_id"],
                agent_id=row["agent_id"],
                tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]),
                decision=row["decision"],
                decided_by=row["decided_by"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- usage ------------------------------------------------------------------

    def record_usage(
        self,
        run_id: str,
        agent_id: str | None,
        model: str,
        usage: Usage,
    ) -> int:
        """Log token usage for one model call and return the new row's id."""
        created_at = _utc_now_iso()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO usage (run_id, agent_id, model, input_tokens, "
                "output_tokens, cache_read_tokens, cache_write_tokens, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    agent_id,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_write_tokens,
                    created_at,
                ),
            )
        return int(cur.lastrowid)

    def list_usage(self, run_id: str) -> list[UsageRecord]:
        """List every usage record for a run, in the order recorded."""
        rows = self._conn.execute(
            "SELECT * FROM usage WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            UsageRecord(
                id=row["id"],
                run_id=row["run_id"],
                agent_id=row["agent_id"],
                model=row["model"],
                usage=Usage(
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    cache_read_tokens=row["cache_read_tokens"],
                    cache_write_tokens=row["cache_write_tokens"],
                ),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def total_usage(self, run_id: str) -> dict[str, int]:
        """Aggregate token usage across every recorded call in a run.

        Returns a dict with the same field names as :class:`~harness.types.Usage`
        (``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens``), summed across every agent and model. Missing
        usage rows sum to ``0``, never ``NULL``.
        """
        row = self._conn.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0)       AS input_tokens,
                COALESCE(SUM(output_tokens), 0)      AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0)  AS cache_read_tokens,
                COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens
            FROM usage WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cache_read_tokens": row["cache_read_tokens"],
            "cache_write_tokens": row["cache_write_tokens"],
        }


def _row_to_event(row: sqlite3.Row) -> TranscriptEvent:
    """Build a :class:`TranscriptEvent` from a `transcript_events` row."""
    return TranscriptEvent(
        id=row["id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        seq=row["seq"],
        kind=row["kind"],
        payload=json.loads(row["payload"]),
        created_at=row["created_at"],
    )
