"""SQLite connection handling and schema.

One database file holds two schemas side by side:

* the tables below -- run lifecycle, execution trace, side-effect ledger. This
  is *our* state model, and it is what the API and CLI report on.
* LangGraph's own checkpoint tables, created by ``SqliteSaver.setup()``. Those
  hold *where execution is*: graph state and pending interrupts.

Keeping both in one file means a single artefact to back up, copy or delete,
and means a restart reloads business state and execution position from the same
place. They stay logically separate: nothing here reads LangGraph's tables and
nothing in the graph writes ours except through the repositories.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "./offboarding.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    thread_id        TEXT NOT NULL UNIQUE,
    employee_id      TEXT NOT NULL,
    status           TEXT NOT NULL,
    pause_reason     TEXT,
    failure_reason   TEXT,
    failure_detail   TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    step_count       INTEGER NOT NULL DEFAULT 0,
    tool_call_count  INTEGER NOT NULL DEFAULT 0,
    max_steps        INTEGER NOT NULL DEFAULT 20,
    max_tool_calls   INTEGER NOT NULL DEFAULT 30,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Append-only execution trace. One row per attempt at a step; a retry adds a
-- row rather than updating one, so retry history survives in the trace.
CREATE TABLE IF NOT EXISTS step_trace (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_name     TEXT NOT NULL,
    attempt       INTEGER NOT NULL,
    status        TEXT NOT NULL,
    tool_invoked  TEXT,
    pause_reason  TEXT,
    error_type    TEXT,
    error_message TEXT,
    detail        TEXT NOT NULL DEFAULT '{}',
    started_at    TEXT NOT NULL,
    ended_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_step_trace_run
    ON step_trace(run_id, id);

-- Idempotency ledger. The UNIQUE key is what actually prevents a side effect
-- from firing twice: a second reservation for the same key cannot be inserted.
CREATE TABLE IF NOT EXISTS side_effects (
    idempotency_key TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_name       TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    state           TEXT NOT NULL,
    result          TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_side_effects_run
    ON side_effects(run_id);

-- Stands in for the downstream systems' own storage (the IAM provider, the
-- document service). It belongs to the *tools*, not to the orchestrator:
-- nothing outside offboarding/tools/ may read or write it. Persisting it means
-- a mocked provider can still answer "did this effect land?" after a restart,
-- which is what makes the reconciliation path demonstrable rather than
-- theoretical.
CREATE TABLE IF NOT EXISTS provider_effects (
    provider        TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (provider, idempotency_key)
);
"""


def resolve_db_path(path: str | os.PathLike[str] | None = None) -> str:
    """Return the database path, preferring an explicit argument over the env."""
    if path is not None:
        return str(path)
    return os.environ.get("OFFBOARDING_DB_PATH", DEFAULT_DB_PATH)


def connect(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas this application depends on.

    ``foreign_keys`` is off by default in SQLite and must be set per connection.
    WAL keeps a reader (the API inspecting a run) from blocking the writer (the
    graph advancing it), which matters once the CLI and API are both live.
    """
    resolved = resolve_db_path(path)
    if resolved != ":memory:":
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)

    # isolation_level=None disables the driver's implicit transactions, so the
    # only transactions that exist are the BEGIN IMMEDIATE blocks we open
    # ourselves. Without this, the driver holds a transaction open across an
    # arbitrary stretch of a node and our explicit BEGIN fails.
    conn = sqlite3.connect(resolved, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if resolved != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    """Create our tables if they do not exist. Safe to call on every startup."""
    conn.executescript(SCHEMA)
    conn.commit()
