"""The seven steps of the offboarding workflow.

Each node does one thing and records it. Three rules hold across all of them,
and they are the reason the orchestration stays readable:

1. **Nodes never set run status.** The run service owns the run lifecycle: it
   moves the run to RUNNING before invoking the graph and decides PAUSED,
   COMPLETED or FAILED from the outcome. A node that also wrote run status
   would give us two writers for one field and no way to reason about resumes.
2. **Nodes never call a tool directly.** Every call goes through the executor,
   which applies the allowlist, the budget, the ledger and the trace.
3. **Every node starts with the guard**, so a cancellation or an exhausted
   budget stops the run *before* the step does any work.

Step order matters: the approval gate sits ahead of ``revoke_access``, because
revoking access is the high-risk action the gate exists to protect.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from offboarding.domain.errors import (
    ApprovalRejected,
    BudgetExceeded,
    RunCancelled,
)
from offboarding.domain.run import PauseReason, StepName, StepStatus
from offboarding.orchestration.services import Services
from offboarding.orchestration.state import OffboardingState
from offboarding.orchestration.tracing import step_span

# --------------------------------------------------------------------------
# Per-step tool allowlists
#
# The guardrail the brief asks for. A step can reach exactly the tools named
# here and nothing else -- note that the two pause gates and the planning step
# hold no tools at all, so a bug in the LLM step cannot revoke anything.
# --------------------------------------------------------------------------

ALLOWLISTS: dict[str, frozenset[str]] = {
    StepName.FETCH_EMPLOYEE: frozenset({"hr_directory.get_employee"}),
    StepName.PLAN_DEPROVISIONING: frozenset(),
    StepName.AWAIT_HR_APPROVAL: frozenset(),
    StepName.REVOKE_ACCESS: frozenset({"iam.revoke_access"}),
    StepName.SEND_EXIT_PAPERWORK: frozenset({"documents.send_exit_paperwork"}),
    StepName.AWAIT_SIGNED_DOCUMENT: frozenset(),
    StepName.FINALIZE: frozenset({"documents.archive_record"}),
}


def guard(services: Services, state: OffboardingState, step: str) -> None:
    """Refuse to start a step that must not run.

    Runs before any work, which is what makes cancellation safe: a cancelled
    run never stops halfway through a side effect, it stops between steps.

    Raises:
        RunCancelled: a cancellation has been requested.
        BudgetExceeded: the run has used its step allowance.
    """
    run = services.runs.get(state["run_id"])

    if run.cancel_requested:
        raise RunCancelled(f"run {run.run_id} was cancelled before {step}")

    if run.step_count >= run.max_steps:
        raise BudgetExceeded(
            f"run {run.run_id} has used its {run.max_steps} step allowance"
        )

    services.runs.bump_counters(run.run_id, steps=1)


def _already_paused_here(services: Services, run_id: str, step: str) -> bool:
    """Whether this step's latest trace row is already a pause.

    Resuming re-runs a node from the top, so everything before ``interrupt()``
    executes again. Checking before writing keeps one pause from producing a
    fresh trace row on every resume attempt -- the check-before-create half of
    keeping pre-interrupt code idempotent.
    """
    latest = services.trace.latest_for_step(run_id, step)
    return latest is not None and latest.status is StepStatus.PAUSED


# --------------------------------------------------------------------------
# 1. Fetch the employee record  (read tool)
# --------------------------------------------------------------------------


def fetch_employee(state: OffboardingState, services: Services) -> dict[str, Any]:
    """Load the HR record the rest of the workflow depends on."""
    guard(services, state, StepName.FETCH_EMPLOYEE)

    outcome = services.executor.execute(
        run_id=state["run_id"],
        step_name=StepName.FETCH_EMPLOYEE,
        tool_name="hr_directory.get_employee",
        allowlist=ALLOWLISTS[StepName.FETCH_EMPLOYEE],
        kwargs={"employee_id": state["employee_id"]},
    )
    employee = outcome.result
    return {
        "employee": employee,
        "events": [
            f"Fetched HR record for {employee['name']} "
            f"({len(employee['systems'])} systems, last day "
            f"{employee['last_day']})."
        ],
    }


# --------------------------------------------------------------------------
# 2. Generate the deprovisioning checklist  (LLM)
# --------------------------------------------------------------------------


def plan_deprovisioning(
    state: OffboardingState, services: Services
) -> dict[str, Any]:
    """Ask the model for a checklist, and validate it before trusting it.

    The plan is validated against the HR record inside the provider, so a model
    that invents a system or drops one fails the run here rather than reaching
    the IAM tool with a plan nobody checked.
    """
    guard(services, state, StepName.PLAN_DEPROVISIONING)

    with step_span(
        services.trace,
        state["run_id"],
        StepName.PLAN_DEPROVISIONING,
        tool_invoked=f"llm:{services.llm.name}",
    ) as span:
        plan = services.llm.generate_plan(state["employee"])
        checklist = [item.model_dump() for item in plan.items]
        span.set_detail(
            provider=services.llm.name,
            item_count=len(checklist),
            systems=[item["system"] for item in checklist],
        )

    return {
        "checklist": checklist,
        "plan_notes": plan.notes,
        "events": [f"Generated a {len(checklist)}-item deprovisioning plan."],
    }


# --------------------------------------------------------------------------
# 3. Pause for HR approval  (human-in-the-loop)
# --------------------------------------------------------------------------


def await_hr_approval(
    state: OffboardingState, services: Services
) -> dict[str, Any]:
    """Pause until a human approves or rejects the plan.

    Everything before ``interrupt()`` re-runs on resume, so the only work here
    is the check-before-create trace write. Nothing side-effecting happens on
    this side of the pause.

    Raises:
        ApprovalRejected: the reviewer said no. The run service records the run
            as failed with ``APPROVAL_REJECTED`` and the IAM tool is never
            reached.
    """
    guard(services, state, StepName.AWAIT_HR_APPROVAL)

    run_id = state["run_id"]
    if not _already_paused_here(services, run_id, StepName.AWAIT_HR_APPROVAL):
        row = services.trace.start_attempt(run_id, StepName.AWAIT_HR_APPROVAL)
        services.trace.pause_attempt(
            row.id,
            pause_reason=PauseReason.HR_APPROVAL,
            detail={"item_count": len(state.get("checklist", []))},
        )

    employee = state["employee"]
    decision = interrupt(
        {
            # The run service reads this to record *why* the run is paused.
            "pause_reason": PauseReason.HR_APPROVAL.value,
            "run_id": run_id,
            "employee": {
                "employee_id": employee["employee_id"],
                "name": employee["name"],
                "department": employee["department"],
                "last_day": employee["last_day"],
            },
            "checklist": state.get("checklist", []),
            "notes": state.get("plan_notes", ""),
            "prompt": (
                "Approve revoking access to these systems? "
                "This action cannot be undone from here."
            ),
        }
    )

    # --- resumed from here on ---
    with step_span(
        services.trace, run_id, StepName.AWAIT_HR_APPROVAL
    ) as span:
        approved = bool(decision.get("approved"))
        span.set_detail(
            approved=approved,
            approver=decision.get("approver"),
            note=decision.get("note"),
        )
        if not approved:
            raise ApprovalRejected(
                decision.get("note") or "HR rejected the deprovisioning plan"
            )

    return {
        "approval": dict(decision),
        "events": [
            f"Approved by {decision.get('approver', 'unknown')}."
        ],
    }


# --------------------------------------------------------------------------
# 4. Revoke access  (side-effecting, high risk)
# --------------------------------------------------------------------------


def revoke_access(state: OffboardingState, services: Services) -> dict[str, Any]:
    """Revoke access across every system in the approved checklist.

    The first irreversible action in the workflow, and the one the approval gate
    above protects. The executor routes it through the ledger, so a resume or a
    retry replays the receipt instead of revoking twice.
    """
    guard(services, state, StepName.REVOKE_ACCESS)

    systems = [item["system"] for item in state["checklist"]]
    outcome = services.executor.execute(
        run_id=state["run_id"],
        step_name=StepName.REVOKE_ACCESS,
        tool_name="iam.revoke_access",
        allowlist=ALLOWLISTS[StepName.REVOKE_ACCESS],
        operation="revoke_all_systems",
        kwargs={"employee_id": state["employee_id"], "systems": systems},
    )

    verb = "Replayed" if outcome.replayed else "Revoked"
    return {
        "revocation": outcome.result,
        "events": [
            f"{verb} access to {len(systems)} systems "
            f"(attempts: {outcome.attempts}, "
            f"revocation {outcome.result['revocation_id']})."
        ],
    }


# --------------------------------------------------------------------------
# 5. Send the exit paperwork  (side-effecting)
# --------------------------------------------------------------------------


def send_exit_paperwork(
    state: OffboardingState, services: Services
) -> dict[str, Any]:
    """Send the exit packet for signature.

    Separated from :func:`revoke_access` rather than bundled with it so that a
    failure sending paperwork cannot cause access revocation to be retried.
    One node, one side effect, one ledger key.
    """
    guard(services, state, StepName.SEND_EXIT_PAPERWORK)

    outcome = services.executor.execute(
        run_id=state["run_id"],
        step_name=StepName.SEND_EXIT_PAPERWORK,
        tool_name="documents.send_exit_paperwork",
        allowlist=ALLOWLISTS[StepName.SEND_EXIT_PAPERWORK],
        operation="send_packet",
        kwargs={
            "employee_id": state["employee_id"],
            "email": state["employee"]["email"],
            "checklist": state["checklist"],
        },
    )

    verb = "Replayed sending" if outcome.replayed else "Sent"
    return {
        "paperwork": outcome.result,
        "events": [
            f"{verb} exit paperwork to {state['employee']['email']} "
            f"({outcome.result['document_id']})."
        ],
    }


# --------------------------------------------------------------------------
# 6. Wait for the signed document  (external event)
# --------------------------------------------------------------------------


def await_signed_document(
    state: OffboardingState, services: Services
) -> dict[str, Any]:
    """Pause until the signed paperwork comes back.

    The second pause, and a different kind from the first: nobody is being asked
    to decide anything, the run is waiting on the outside world. It uses the
    same mechanism, which is the point -- adding a third pause reason needs a
    node and an enum member, not a change to the orchestrator.
    """
    guard(services, state, StepName.AWAIT_SIGNED_DOCUMENT)

    run_id = state["run_id"]
    if not _already_paused_here(
        services, run_id, StepName.AWAIT_SIGNED_DOCUMENT
    ):
        row = services.trace.start_attempt(
            run_id, StepName.AWAIT_SIGNED_DOCUMENT
        )
        services.trace.pause_attempt(
            row.id,
            pause_reason=PauseReason.SIGNED_DOCUMENT,
            detail={"document_id": state["paperwork"]["document_id"]},
        )

    event = interrupt(
        {
            "pause_reason": PauseReason.SIGNED_DOCUMENT.value,
            "run_id": run_id,
            "document_id": state["paperwork"]["document_id"],
            "prompt": (
                "Waiting for the signed exit paperwork. Resume this run when "
                "the signed document is received."
            ),
        }
    )

    # --- resumed from here on ---
    with step_span(
        services.trace, run_id, StepName.AWAIT_SIGNED_DOCUMENT
    ) as span:
        span.set_detail(
            document_id=event.get("document_id"),
            signed_at=event.get("signed_at"),
        )

    return {
        "signed_document": dict(event),
        "events": [
            f"Received signed document "
            f"{event.get('document_id', 'unknown')}."
        ],
    }


# --------------------------------------------------------------------------
# 7. Archive the record  (side-effecting)
# --------------------------------------------------------------------------


def finalize(state: OffboardingState, services: Services) -> dict[str, Any]:
    """Archive the completed offboarding record."""
    guard(services, state, StepName.FINALIZE)

    summary = {
        "employee_id": state["employee_id"],
        "systems_revoked": [item["system"] for item in state["checklist"]],
        "revocation_id": state["revocation"]["revocation_id"],
        "document_id": state["paperwork"]["document_id"],
        "approved_by": state.get("approval", {}).get("approver"),
    }
    outcome = services.executor.execute(
        run_id=state["run_id"],
        step_name=StepName.FINALIZE,
        tool_name="documents.archive_record",
        allowlist=ALLOWLISTS[StepName.FINALIZE],
        operation="archive",
        kwargs={"employee_id": state["employee_id"], "summary": summary},
    )

    return {
        "archive": outcome.result,
        "events": [
            f"Offboarding complete; archived as "
            f"{outcome.result['archive_id']}."
        ],
    }
