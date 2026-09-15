"""The ``side_effects`` table: the idempotency ledger's storage.

Policy (when to call, what an existing row means) lives in
``offboarding.tools.ledger``; this class only persists rows.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from offboarding.domain.models import SideEffectRecord
from offboarding.persistence.base import BaseRepository, now, parse_ts


class SideEffectRepository(BaseRepository):
    """Storage for the idempotency ledger."""

    RESERVED = "reserved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

    @staticmethod
    def _to_record(row: sqlite3.Row) -> SideEffectRecord:
        return SideEffectRecord(
            idempotency_key=row["idempotency_key"],
            run_id=row["run_id"],
            step_name=row["step_name"],
            tool_name=row["tool_name"],
            state=row["state"],
            result=json.loads(row["result"]) if row["result"] else None,
            created_at=parse_ts(row["created_at"]),  # type: ignore[arg-type]
            completed_at=parse_ts(row["completed_at"]),
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
                (key, run_id, step_name, tool_name, self.RESERVED, now()),
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
                now() if completed else None,
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
