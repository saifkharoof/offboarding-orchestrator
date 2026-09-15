"""Repositories: the only code that reads or writes our tables.

Nodes, tools and the API all go through these. Two consequences worth naming:

* Every run status change flows through :meth:`RunRepository.set_status`, which
  validates the move against the state machine inside a single transaction. A
  concurrent approve and cancel cannot interleave into an illegal state.
* The trace is written by the orchestrator around every step attempt, not by
  the individual nodes, so no step can quietly skip being traced.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from offboarding.domain.errors import RunNotFound
from offboarding.domain.models import Run, SideEffectRecord, StepAttempt
from offboarding.domain.run import (
    FailureReason,
    PauseReason,
    RunStatus,
    StepStatus,
    transition_run,
    transition_step,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@contextmanager
def _immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a read-modify-write under a write lock taken up front.

    ``BEGIN IMMEDIATE`` acquires the write lock before we read, so two callers
    cannot both read status ``paused`` and then both transition away from it.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


class RunRepository:
    """Reads and writes the ``runs`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_run(row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            employee_id=row["employee_id"],
            status=RunStatus(row["status"]),
            pause_reason=(
                PauseReason(row["pause_reason"]) if row["pause_reason"] else None
            ),
            failure_reason=(
                FailureReason(row["failure_reason"])
                if row["failure_reason"]
                else None
            ),
            failure_detail=row["failure_detail"],
            cancel_requested=bool(row["cancel_requested"]),
            step_count=row["step_count"],
            tool_call_count=row["tool_call_count"],
            max_steps=row["max_steps"],
            max_tool_calls=row["max_tool_calls"],
            created_at=_parse_ts(row["created_at"]),  # type: ignore[arg-type]
            updated_at=_parse_ts(row["updated_at"]),  # type: ignore[arg-type]
        )

    def create(
        self,
        employee_id: str,
        *,
        run_id: str | None = None,
        max_steps: int = 20,
        max_tool_calls: int = 30,
    ) -> Run:
        """Insert a new run in :attr:`RunStatus.PENDING`.

        The LangGraph thread id is derived from the run id so that one run maps
        to exactly one checkpoint thread, and so a resume cannot be pointed at
        the wrong thread by a caller.
        """
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        now = _now()
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, thread_id, employee_id, status,
                max_steps, max_tool_calls, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                f"thread_{rid}",
                employee_id,
                RunStatus.PENDING.value,
                max_steps,
                max_tool_calls,
                now,
                now,
            ),
        )
        self._conn.commit()
        return self.get(rid)

    def get(self, run_id: str) -> Run:
        """Return a run, raising :class:`RunNotFound` if it does not exist."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFound(f"no run with id {run_id!r}")
        return self._to_run(row)

    def list_runs(self, limit: int = 50) -> list[Run]:
        """Return runs newest first."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._to_run(r) for r in rows]

    def set_status(
        self,
        run_id: str,
        target: RunStatus,
        *,
        pause_reason: PauseReason | None = None,
        failure_reason: FailureReason | None = None,
        failure_detail: str | None = None,
    ) -> Run:
        """Transition a run, validating the move against the state machine.

        Raises:
            RunNotFound: if the run does not exist.
            InvalidStateTransition: if the move is illegal.
        """
        with _immediate(self._conn):
            row = self._conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFound(f"no run with id {run_id!r}")

            current = RunStatus(row["status"])
            transition_run(current, target)  # raises if illegal

            self._conn.execute(
                """
                UPDATE runs
                   SET status = ?,
                       pause_reason = ?,
                       failure_reason = COALESCE(?, failure_reason),
                       failure_detail = COALESCE(?, failure_detail),
                       updated_at = ?
                 WHERE run_id = ?
                """,
                (
                    target.value,
                    pause_reason.value if pause_reason else None,
                    failure_reason.value if failure_reason else None,
                    failure_detail,
                    _now(),
                    run_id,
                ),
            )
        return self.get(run_id)

    def request_cancel(self, run_id: str) -> Run:
        """Flag a run for cancellation without changing its status.

        The guard that runs before each step observes the flag and performs the
        actual transition. Setting a flag rather than transitioning directly is
        what lets us cancel a run that is mid-step without tearing execution out
        from under a side effect.
        """
        with _immediate(self._conn):
            row = self._conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFound(f"no run with id {run_id!r}")
            self._conn.execute(
                "UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?",
                (_now(), run_id),
            )
        return self.get(run_id)

    def bump_counters(
        self, run_id: str, *, steps: int = 0, tool_calls: int = 0
    ) -> Run:
        """Increment the bounded-execution counters."""
        self._conn.execute(
            """
            UPDATE runs
               SET step_count = step_count + ?,
                   tool_call_count = tool_call_count + ?,
                   updated_at = ?
             WHERE run_id = ?
            """,
            (steps, tool_calls, _now(), run_id),
        )
        self._conn.commit()
        return self.get(run_id)


# --------------------------------------------------------------------------
# Execution trace
# --------------------------------------------------------------------------


class TraceRepository:
    """Reads and writes the ``step_trace`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_attempt(row: sqlite3.Row) -> StepAttempt:
        return StepAttempt(
            id=row["id"],
            run_id=row["run_id"],
            step_name=row["step_name"],
            attempt=row["attempt"],
            status=StepStatus(row["status"]),
            tool_invoked=row["tool_invoked"],
            pause_reason=(
                PauseReason(row["pause_reason"]) if row["pause_reason"] else None
            ),
            error_type=row["error_type"],
            error_message=row["error_message"],
            detail=json.loads(row["detail"]),
            started_at=_parse_ts(row["started_at"]),  # type: ignore[arg-type]
            ended_at=_parse_ts(row["ended_at"]),
        )

    def start_attempt(
        self, run_id: str, step_name: str, *, tool_invoked: str | None = None
    ) -> StepAttempt:
        """Open a new attempt row for a step and return it.

        The attempt number is derived from existing rows, so a resume that
        re-enters a node produces attempt N+1 rather than overwriting N.
        """
        with _immediate(self._conn):
            row = self._conn.execute(
                """
                SELECT attempt, status FROM step_trace
                 WHERE run_id = ? AND step_name = ?
                 ORDER BY attempt DESC LIMIT 1
                """,
                (run_id, step_name),
            ).fetchone()

            if row is None:
                attempt = 1
            else:
                transition_step(StepStatus(row["status"]), StepStatus.RUNNING)
                attempt = row["attempt"] + 1

            cursor = self._conn.execute(
                """
                INSERT INTO step_trace (
                    run_id, step_name, attempt, status, tool_invoked, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_name,
                    attempt,
                    StepStatus.RUNNING.value,
                    tool_invoked,
                    _now(),
                ),
            )
            attempt_id = cursor.lastrowid

        return self.get(int(attempt_id))  # type: ignore[arg-type]

    def _close_attempt(
        self,
        attempt_id: int,
        target: StepStatus,
        *,
        pause_reason: PauseReason | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        detail: dict[str, Any] | None = None,
        tool_invoked: str | None = None,
    ) -> StepAttempt:
        with _immediate(self._conn):
            row = self._conn.execute(
                "SELECT status FROM step_trace WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no step attempt with id {attempt_id}")
            transition_step(StepStatus(row["status"]), target)

            self._conn.execute(
                """
                UPDATE step_trace
                   SET status = ?,
                       pause_reason = ?,
                       error_type = ?,
                       error_message = ?,
                       detail = COALESCE(?, detail),
                       tool_invoked = COALESCE(?, tool_invoked),
                       ended_at = ?
                 WHERE id = ?
                """,
                (
                    target.value,
                    pause_reason.value if pause_reason else None,
                    error_type,
                    error_message,
                    json.dumps(detail) if detail is not None else None,
                    tool_invoked,
                    _now(),
                    attempt_id,
                ),
            )
        return self.get(attempt_id)

    def complete_attempt(
        self,
        attempt_id: int,
        *,
        detail: dict[str, Any] | None = None,
        tool_invoked: str | None = None,
    ) -> StepAttempt:
        """Mark an attempt completed."""
        return self._close_attempt(
            attempt_id,
            StepStatus.COMPLETED,
            detail=detail,
            tool_invoked=tool_invoked,
        )

    def fail_attempt(
        self,
        attempt_id: int,
        *,
        error_type: str,
        error_message: str,
        detail: dict[str, Any] | None = None,
    ) -> StepAttempt:
        """Mark an attempt failed, recording the error class and message.

        ``error_type`` is the exception class name, which is what distinguishes
        a retryable failure from a permanent one when reading the trace back.
        """
        return self._close_attempt(
            attempt_id,
            StepStatus.FAILED,
            error_type=error_type,
            error_message=error_message,
            detail=detail,
        )

    def pause_attempt(
        self,
        attempt_id: int,
        *,
        pause_reason: PauseReason,
        detail: dict[str, Any] | None = None,
    ) -> StepAttempt:
        """Mark an attempt paused, recording why."""
        return self._close_attempt(
            attempt_id,
            StepStatus.PAUSED,
            pause_reason=pause_reason,
            detail=detail,
        )

    def get(self, attempt_id: int) -> StepAttempt:
        """Return a single attempt row."""
        row = self._conn.execute(
            "SELECT * FROM step_trace WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no step attempt with id {attempt_id}")
        return self._to_attempt(row)

    def list_for_run(self, run_id: str) -> list[StepAttempt]:
        """Return the full trace for a run in execution order."""
        rows = self._conn.execute(
            "SELECT * FROM step_trace WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [self._to_attempt(r) for r in rows]

    def latest_for_step(self, run_id: str, step_name: str) -> StepAttempt | None:
        """Return the most recent attempt at a step, or ``None``."""
        row = self._conn.execute(
            """
            SELECT * FROM step_trace
             WHERE run_id = ? AND step_name = ?
             ORDER BY attempt DESC LIMIT 1
            """,
            (run_id, step_name),
        ).fetchone()
        return self._to_attempt(row) if row else None


# --------------------------------------------------------------------------
# Side-effect ledger
# --------------------------------------------------------------------------


class SideEffectRepository:
    """Storage for the idempotency ledger.

    Policy (when to call, what an existing row means) lives in
    ``offboarding.tools.ledger``; this class only persists rows.
    """

    RESERVED = "reserved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_record(row: sqlite3.Row) -> SideEffectRecord:
        return SideEffectRecord(
            idempotency_key=row["idempotency_key"],
            run_id=row["run_id"],
            step_name=row["step_name"],
            tool_name=row["tool_name"],
            state=row["state"],
            result=json.loads(row["result"]) if row["result"] else None,
            created_at=_parse_ts(row["created_at"]),  # type: ignore[arg-type]
            completed_at=_parse_ts(row["completed_at"]),
        )

    def try_reserve(
        self, key: str, *, run_id: str, step_name: str, tool_name: str
    ) -> SideEffectRecord | None:
        """Insert a reservation, or return ``None`` if the key already exists.

        The ``None`` return is the whole point: the UNIQUE constraint on
        ``idempotency_key`` is what makes duplicate prevention a database
        guarantee rather than a check that could race.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO side_effects (
                    idempotency_key, run_id, step_name, tool_name, state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, run_id, step_name, tool_name, self.RESERVED, _now()),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            return None
        return self.get(key)

    def get(self, key: str) -> SideEffectRecord | None:
        """Return a ledger row, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM side_effects WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return self._to_record(row) if row else None

    def _set_state(
        self, key: str, state: str, *, result: dict[str, Any] | None = None
    ) -> SideEffectRecord:
        completed = state in (self.COMPLETED, self.FAILED)
        self._conn.execute(
            """
            UPDATE side_effects
               SET state = ?,
                   result = COALESCE(?, result),
                   completed_at = ?
             WHERE idempotency_key = ?
            """,
            (
                state,
                json.dumps(result) if result is not None else None,
                _now() if completed else None,
                key,
            ),
        )
        self._conn.commit()
        record = self.get(key)
        assert record is not None
        return record

    def mark_in_progress(self, key: str) -> SideEffectRecord:
        """Record that the tool call is about to be made.

        Written and committed *before* the call, so a process that dies mid-call
        leaves evidence that an attempt was made.
        """
        return self._set_state(key, self.IN_PROGRESS)

    def mark_completed(
        self, key: str, result: dict[str, Any]
    ) -> SideEffectRecord:
        """Record the tool call succeeded, caching its result for replay."""
        return self._set_state(key, self.COMPLETED, result=result)

    def mark_failed(self, key: str) -> SideEffectRecord:
        """Record the tool call failed in a way that permits a fresh attempt."""
        return self._set_state(key, self.FAILED)

    def list_for_run(self, run_id: str) -> list[SideEffectRecord]:
        """Return every ledger row for a run."""
        rows = self._conn.execute(
            "SELECT * FROM side_effects WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [self._to_record(r) for r in rows]
