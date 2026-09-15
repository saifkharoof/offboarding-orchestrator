"""The idempotency ledger: every side effect in the system passes through here.

The contract is that :meth:`SideEffectLedger.run_once` performs a side-effecting
tool call *at most once per idempotency key*, no matter how many times it is
called. Resumes, retries and restarts all funnel through it, so duplicate
protection is one code path rather than a rule each node has to remember.

The key deliberately excludes the attempt number::

    {run_id}:{step_name}:{operation}

Attempt 1 and attempt 4 of ``revoke_access`` therefore share a key, which is
exactly what makes a retry safe.

Ordering, and why
-----------------

The ledger row is written and committed *before* the tool is called
(write-ahead). That costs one extra write, and buys the only thing that makes
crash recovery possible: after an unexpected death we can always tell whether a
call was in flight. A ledger written after the call would leave us unable to
distinguish "never attempted" from "applied but unrecorded" -- and those two
require opposite responses.

Recovery states
---------------

On re-entry a ledger row can be in four states, each with one correct action:

``completed``    The effect landed and we have its result. Replay the cached
                 result; do not call the tool.
``reserved``     We died between reserving and calling. The call never
                 happened, so it is safe to proceed.
``in_progress``  The ambiguous case -- we died during the call. Ask the
                 provider (``lookup``) whether the effect landed. If it did,
                 adopt it; if it definitively did not, retry; if the provider
                 cannot answer, fail closed with
                 :class:`SideEffectReconciliationRequired` rather than risk
                 firing the effect twice.
``failed``       A previous attempt failed with nothing applied. Retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from offboarding.domain.errors import (
    PermanentToolError,
    SideEffectReconciliationRequired,
)
from offboarding.persistence.side_effects import SideEffectRepository
from offboarding.tools.base import SideEffectingTool, Tool


def build_key(run_id: str, step_name: str, operation: str) -> str:
    """Build the idempotency key for one logical side effect.

    Excludes the attempt number by design: every attempt at the same logical
    operation must collide on the same key.
    """
    return f"{run_id}:{step_name}:{operation}"


@dataclass(frozen=True, slots=True)
class LedgerOutcome:
    """What happened when a side effect was requested."""

    result: dict[str, Any]
    #: True when the tool was not called because the effect had already landed.
    replayed: bool = False
    #: True when the result was recovered from the provider after an
    #: interrupted call rather than from our own ledger.
    reconciled: bool = False

    @property
    def performed(self) -> bool:
        """True only when this call actually changed the world."""
        return not self.replayed


class SideEffectLedger:
    """Executes side-effecting tool calls at most once per idempotency key."""

    def __init__(self, store: SideEffectRepository) -> None:
        self._store = store

    def run_once(
        self,
        *,
        key: str,
        run_id: str,
        step_name: str,
        tool: Tool,
        kwargs: dict[str, Any],
    ) -> LedgerOutcome:
        """Perform ``tool.call(**kwargs)`` at most once for ``key``.

        Returns:
            The outcome, whose ``replayed`` flag says whether the tool was
            actually invoked.

        Raises:
            SideEffectReconciliationRequired: if a previous attempt was in
                flight and the provider cannot say whether it landed.
            ToolError: whatever the tool raised, after the ledger has recorded
                the attempt.
        """
        existing = self._store.get(key)

        if existing is not None:
            outcome = self._resume_existing(existing, key=key, tool=tool)
            if outcome is not None:
                return outcome
        else:
            reserved = self._store.try_reserve(
                key,
                run_id=run_id,
                step_name=step_name,
                tool_name=tool.spec.name,
            )
            if reserved is None:
                # Someone reserved the key between our read and our insert.
                # Re-enter rather than race: the second pass sees their row.
                return self.run_once(
                    key=key,
                    run_id=run_id,
                    step_name=step_name,
                    tool=tool,
                    kwargs=kwargs,
                )

        # Write-ahead: committed before the call, so a death during the call is
        # distinguishable from a death before it.
        self._store.mark_in_progress(key)

        try:
            # The key is injected rather than threaded through every caller: the
            # ledger owns it, and the provider needs it to honour idempotency on
            # its own side. Callers pass only the business arguments.
            result = tool.call(**kwargs, idempotency_key=key)
        except Exception as exc:
            landed = self._ask_provider(tool, key)
            if landed is not None:
                # The effect applied; the failure was in reporting it back. Adopt
                # the provider's result rather than retrying an applied change.
                self._store.mark_completed(key, landed)
                return LedgerOutcome(result=landed, reconciled=True)

            if isinstance(exc, PermanentToolError):
                # Nothing applied and retrying cannot help: close the row so the
                # ledger does not read as an attempt still in flight.
                self._store.mark_failed(key)
            # Transient failures keep the row in_progress on purpose: the next
            # attempt re-enters via _resume_existing and reconciles first.
            raise

        self._store.mark_completed(key, result)
        return LedgerOutcome(result=result)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _resume_existing(
        self, existing, *, key: str, tool: Tool
    ) -> LedgerOutcome | None:
        """Decide what an existing ledger row means.

        Returns an outcome when the call must not be made, or ``None`` when it
        is safe to proceed to the tool.
        """
        if existing.state == SideEffectRepository.COMPLETED:
            # The single most important line in the project: a completed effect
            # is replayed from the ledger and the tool is never touched.
            return LedgerOutcome(result=existing.result or {}, replayed=True)

        if existing.state == SideEffectRepository.IN_PROGRESS:
            landed = self._ask_provider(tool, key)
            if landed is not None:
                self._store.mark_completed(key, landed)
                return LedgerOutcome(
                    result=landed, replayed=True, reconciled=True
                )
            if not isinstance(tool, SideEffectingTool):
                raise SideEffectReconciliationRequired(
                    f"side effect {key!r} was in flight when the process "
                    f"stopped and {tool.spec.name!r} cannot be queried; "
                    "refusing to retry a possibly-applied effect"
                )
            # The provider is authoritative and says it never landed.
            return None

        # reserved | failed -- nothing was applied, so proceed.
        return None

    @staticmethod
    def _ask_provider(tool: Tool, key: str) -> dict[str, Any] | None:
        """Ask a provider whether an effect landed, tolerating tools that cannot.

        A provider that raises while being queried is treated as unable to
        answer, which keeps us on the fail-closed path.
        """
        if not isinstance(tool, SideEffectingTool):
            return None
        try:
            return tool.lookup(key)
        except Exception:
            return None
