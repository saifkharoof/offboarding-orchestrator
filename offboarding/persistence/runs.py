"""The ``runs`` table: one row per agent run, and its lifecycle.

Every run status change flows through :meth:`RunRepository.set_status`, which
validates the move against the state machine inside a single transaction. A
concurrent approve and cancel cannot interleave into an illegal state.
"""

from __future__ import annotations

import sqlite3
import uuid

from offboarding.domain.errors import RunNotFound
from offboarding.domain.models import Run
from offboarding.domain.run import (
    FailureReason,
    PauseReason,
    RunStatus,
    transition_run,
)
from offboarding.persistence.base import BaseRepository, now, parse_ts


class RunRepository(BaseRepository):
    """Reads and writes the ``runs`` table."""

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
            created_at=parse_ts(row["created_at"]),  # type: ignore[arg-type]
            updated_at=parse_ts(row["updated_at"]),  # type: ignore[arg-type]
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
        ts = now()
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
                ts,
                ts,
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
        with self._immediate():
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
                    now(),
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
        with self._immediate():
            row = self._conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFound(f"no run with id {run_id!r}")
            self._conn.execute(
                "UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?",
                (now(), run_id),
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
            (steps, tool_calls, now(), run_id),
        )
        self._conn.commit()
        return self.get(run_id)
