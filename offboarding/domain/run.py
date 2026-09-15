"""The run/step state machine.

This module is the authority on what an agent run *is* and which lifecycle
transitions are legal. It deliberately knows nothing about LangGraph, SQLite or
HTTP: the orchestrator drives these states, it does not define them.

Two separate state machines live here:

* ``RunStatus``  -- the lifecycle of a whole run (what an operator asks about).
* ``StepStatus`` -- the lifecycle of a single attempt at a single step.

Every transition goes through :func:`transition_run` / :func:`transition_step`,
which raise :class:`InvalidStateTransition` rather than silently accepting an
impossible move. A corrupt run state is a bug we want to see immediately, not a
row we discover is wrong three resumes later.
"""

from __future__ import annotations

from enum import StrEnum

from offboarding.domain.errors import InvalidStateTransition


class RunStatus(StrEnum):
    """Lifecycle of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    """Lifecycle of one attempt at one step."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepName(StrEnum):
    """The seven steps of the offboarding workflow, in execution order."""

    FETCH_EMPLOYEE = "fetch_employee"
    PLAN_DEPROVISIONING = "plan_deprovisioning"
    AWAIT_HR_APPROVAL = "await_hr_approval"
    REVOKE_ACCESS = "revoke_access"
    SEND_EXIT_PAPERWORK = "send_exit_paperwork"
    AWAIT_SIGNED_DOCUMENT = "await_signed_document"
    FINALIZE = "finalize"


class PauseReason(StrEnum):
    """Why a run is sitting in :attr:`RunStatus.PAUSED`.

    Adding a new pause reason is intended to be a one-line change here plus a
    node that raises it -- the orchestrator itself does not enumerate these.
    """

    HR_APPROVAL = "hr_approval"
    SIGNED_DOCUMENT = "signed_document"


class FailureReason(StrEnum):
    """Why a run is sitting in :attr:`RunStatus.FAILED`.

    ``NEEDS_RECONCILIATION`` is the deliberate fail-closed outcome when we
    cannot determine whether a side effect landed. See ``tools/ledger.py``.
    """

    TOOL_FAILED = "tool_failed"
    APPROVAL_REJECTED = "approval_rejected"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_PLAN = "invalid_plan"
    NEEDS_RECONCILIATION = "needs_reconciliation"


# --------------------------------------------------------------------------
# Transition tables
# --------------------------------------------------------------------------

#: Legal run transitions. A status absent from a value set cannot be reached
#: from that key, and terminal statuses map to the empty set.
ALLOWED_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

#: Legal step transitions, applied between consecutive trace rows for the same
#: step. Retries do NOT transition a step: each attempt is its own trace row, so
#: attempt 2 of ``revoke_access`` starts again at ``RUNNING``.
ALLOWED_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset(
        {StepStatus.RUNNING, StepStatus.SKIPPED}
    ),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.PAUSED,
            StepStatus.RUNNING,  # next retry attempt
        }
    ),
    StepStatus.PAUSED: frozenset(
        {StepStatus.RUNNING, StepStatus.COMPLETED, StepStatus.FAILED}
    ),
    StepStatus.FAILED: frozenset({StepStatus.RUNNING}),  # retried attempt
    StepStatus.COMPLETED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
}

#: Run statuses from which no further work happens.
TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)

#: Run statuses a resume/approve request may act on.
RESUMABLE_RUN_STATUSES: frozenset[RunStatus] = frozenset({RunStatus.PAUSED})


def is_terminal(status: RunStatus) -> bool:
    """Return whether ``status`` admits no further transitions."""
    return status in TERMINAL_RUN_STATUSES


def transition_run(current: RunStatus, target: RunStatus) -> RunStatus:
    """Validate a run transition and return ``target``.

    Raises:
        InvalidStateTransition: if the move is not in
            :data:`ALLOWED_RUN_TRANSITIONS`.
    """
    if target not in ALLOWED_RUN_TRANSITIONS[current]:
        raise InvalidStateTransition(
            f"illegal run transition {current.value} -> {target.value}"
        )
    return target


def transition_step(current: StepStatus, target: StepStatus) -> StepStatus:
    """Validate a step transition and return ``target``.

    Raises:
        InvalidStateTransition: if the move is not in
            :data:`ALLOWED_STEP_TRANSITIONS`.
    """
    if target not in ALLOWED_STEP_TRANSITIONS[current]:
        raise InvalidStateTransition(
            f"illegal step transition {current.value} -> {target.value}"
        )
    return target
