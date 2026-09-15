"""The ``step_trace`` table: an append-only record of every step attempt.

The trace is written by the orchestrator around every step attempt, not by the
individual nodes, so no step can quietly skip being traced.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from offboarding.domain.models import StepAttempt
from offboarding.domain.run import PauseReason, StepStatus, transition_step
from offboarding.persistence.base import BaseRepository, now, parse_ts


class TraceRepository(BaseRepository):
    """Reads and writes the ``step_trace`` table."""

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
            started_at=parse_ts(row["started_at"]),  # type: ignore[arg-type]
            ended_at=parse_ts(row["ended_at"]),
        )

    def start_attempt(
        self, run_id: str, step_name: str, *, tool_invoked: str | None = None
    ) -> StepAttempt:
        """Open a new attempt row for a step and return it.

        The attempt number is derived from existing rows, so a resume that
        re-enters a node produces attempt N+1 rather than overwriting N.
        """
        with self._immediate():
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
                    now(),
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
        with self._immediate():
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
                    now(),
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
