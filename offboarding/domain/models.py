"""Plain data records exchanged between the orchestrator, the store and the API.

These are read models: repositories build them, callers read them. Mutating one
does not change anything persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from offboarding.domain.run import (
    FailureReason,
    PauseReason,
    RunStatus,
    StepStatus,
)


@dataclass(frozen=True, slots=True)
class Run:
    """The lifecycle record for one agent run."""

    run_id: str
    thread_id: str
    employee_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    pause_reason: PauseReason | None = None
    failure_reason: FailureReason | None = None
    failure_detail: str | None = None
    cancel_requested: bool = False
    step_count: int = 0
    tool_call_count: int = 0
    max_steps: int = 20
    max_tool_calls: int = 30

    @property
    def is_terminal(self) -> bool:
        from offboarding.domain.run import is_terminal

        return is_terminal(self.status)


@dataclass(frozen=True, slots=True)
class StepAttempt:
    """One attempt at one step: a single append-only row of the execution trace.

    A retry produces a new ``StepAttempt`` with an incremented ``attempt``
    rather than mutating the previous one, so the trace shows retry history
    instead of hiding it behind a final status.
    """

    id: int
    run_id: str
    step_name: str
    attempt: int
    status: StepStatus
    started_at: datetime
    ended_at: datetime | None = None
    tool_invoked: str | None = None
    pause_reason: PauseReason | None = None
    error_type: str | None = None
    error_message: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SideEffectRecord:
    """A row of the idempotency ledger guarding one side-effecting tool call."""

    idempotency_key: str
    run_id: str
    step_name: str
    tool_name: str
    state: str  # reserved | in_progress | completed | failed
    created_at: datetime
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
