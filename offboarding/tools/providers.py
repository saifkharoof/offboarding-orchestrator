"""Durable storage for the mocked downstream systems.

Stands in for whatever the real IAM or document vendor keeps on their side. It
lives in the same SQLite file only for convenience; conceptually it is across a
network boundary, so only code in ``offboarding/tools/`` touches it.

Persisting it is what lets a mocked provider answer ``lookup()`` after the
orchestrator process has been killed and restarted -- the case the ledger needs
in order to reconcile an interrupted call instead of guessing.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


class ProviderStore:
    """Append-only record of effects a mocked provider has applied."""

    def __init__(self, conn: sqlite3.Connection, provider: str) -> None:
        self._conn = conn
        self._provider = provider

    def record(self, idempotency_key: str, payload: dict[str, Any]) -> None:
        """Persist an effect. Re-recording the same key is a no-op.

        Providers that honour idempotency keys behave this way, so a duplicate
        call that somehow reaches the provider cannot double-apply.
        """
        self._conn.execute(
            """
            INSERT OR IGNORE INTO provider_effects (
                provider, idempotency_key, payload, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                self._provider,
                idempotency_key,
                json.dumps(payload),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return the effect recorded for a key, or ``None`` if there is none."""
        row = self._conn.execute(
            """
            SELECT payload FROM provider_effects
             WHERE provider = ? AND idempotency_key = ?
            """,
            (self._provider, idempotency_key),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def count(self) -> int:
        """Number of effects this provider has applied. Used by tests."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM provider_effects WHERE provider = ?",
            (self._provider,),
        ).fetchone()
        return int(row["n"])
