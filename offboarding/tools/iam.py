"""Side-effecting tool: identity and access management.

This is the high-risk action of the workflow -- it is what the HR approval gate
exists to protect. It carries the fault injection used to demonstrate retry
behaviour and the crash-mid-call recovery path.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from offboarding.domain.errors import PermanentToolError, TransientToolError
from offboarding.tools.base import ToolSpec
from offboarding.tools.providers import ProviderStore


class SimulatedProcessDeath(BaseException):
    """Raised to simulate the process dying mid-call, after the effect landed.

    Deliberately derives from :class:`BaseException`, not ``Exception``: the
    retry executor must not catch it, because a real ``SIGKILL`` would not be
    catchable either. Only tests raise this.
    """


class IamTool:
    """Revoke an employee's access across a set of systems.

    Fault injection (all default to off, and exist so the retry and recovery
    behaviour can be triggered on demand rather than described):

    * ``fail_times`` -- raise a transient error this many times before the call
      is allowed to succeed. Set ``OFFBOARDING_IAM_FAIL_TIMES=2`` to reproduce
      "fail twice, then succeed" from the CLI without editing code.
    * ``fail_permanently`` -- raise a non-retryable error instead.
    * ``crash_after_apply`` -- record the effect with the provider and then die
      before returning, which is the stretch scenario: the side effect landed
      but the orchestrator never learned about it.
    """

    spec = ToolSpec(
        name="iam.revoke_access",
        description="Revoke an employee's access across the listed systems.",
        side_effecting=True,
        requires_approval=True,
    )

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        fail_times: int = 0,
        fail_permanently: bool = False,
        crash_after_apply: bool = False,
    ) -> None:
        self._store = ProviderStore(conn, provider="iam")
        self._remaining_failures = fail_times
        self._fail_permanently = fail_permanently
        self._crash_after_apply = crash_after_apply

    def call(
        self,
        *,
        employee_id: str,
        systems: list[str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Revoke access, returning the revocation receipt.

        The provider honours the idempotency key: a call whose key has already
        been applied returns the original receipt without revoking twice. That
        is the provider-side half of duplicate protection; the ledger is the
        orchestrator-side half, and we want both.
        """
        existing = self._store.lookup(idempotency_key)
        if existing is not None:
            return existing

        if self._fail_permanently:
            raise PermanentToolError(
                f"iam rejected the revocation for {employee_id!r}: "
                "account is under legal hold"
            )

        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise TransientToolError(
                "iam gateway returned 503 (service unavailable)"
            )

        receipt = {
            "employee_id": employee_id,
            "revoked_systems": list(systems),
            "revocation_id": f"rev_{idempotency_key[-12:]}",
        }
        self._store.record(idempotency_key, receipt)

        if self._crash_after_apply:
            raise SimulatedProcessDeath(
                "process died after IAM applied the revocation"
            )

        return receipt

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Ask the provider whether this revocation was applied."""
        return self._store.lookup(idempotency_key)
