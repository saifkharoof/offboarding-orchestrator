"""Side-effecting tool: the document service.

Two operations, both idempotent at the provider: sending the exit paperwork and
archiving the finished offboarding record.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from offboarding.domain.errors import TransientToolError
from offboarding.tools.base import ToolSpec
from offboarding.tools.providers import ProviderStore


class SendExitPaperworkTool:
    """Send exit paperwork to the departing employee for signature."""

    spec = ToolSpec(
        name="documents.send_exit_paperwork",
        description="Send the exit paperwork packet to an employee for signing.",
        side_effecting=True,
        requires_approval=False,
    )

    def __init__(self, conn: sqlite3.Connection, *, fail_times: int = 0) -> None:
        self._store = ProviderStore(conn, provider="documents")
        self._remaining_failures = fail_times

    def call(
        self,
        *,
        employee_id: str,
        email: str,
        checklist: list[dict[str, Any]],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Send the packet and return the document receipt."""
        existing = self._store.lookup(idempotency_key)
        if existing is not None:
            return existing

        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise TransientToolError("document service timed out")

        receipt = {
            "document_id": f"doc_{idempotency_key[-12:]}",
            "employee_id": employee_id,
            "sent_to": email,
            "item_count": len(checklist),
            "status": "awaiting_signature",
        }
        self._store.record(idempotency_key, receipt)
        return receipt

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Ask the provider whether this packet was sent."""
        return self._store.lookup(idempotency_key)


class ArchiveRecordTool:
    """Write the completed offboarding record to the document archive."""

    spec = ToolSpec(
        name="documents.archive_record",
        description="Archive the completed offboarding record.",
        side_effecting=True,
        requires_approval=False,
    )

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._store = ProviderStore(conn, provider="archive")

    def call(
        self,
        *,
        employee_id: str,
        summary: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Archive the record and return its archive receipt."""
        existing = self._store.lookup(idempotency_key)
        if existing is not None:
            return existing

        receipt = {
            "archive_id": f"arc_{idempotency_key[-12:]}",
            "employee_id": employee_id,
            "summary": summary,
        }
        self._store.record(idempotency_key, receipt)
        return receipt

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Ask the archive whether this record was written."""
        return self._store.lookup(idempotency_key)
