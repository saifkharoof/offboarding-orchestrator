"""``RunService``: the single public surface over one offboarding run.

Both the CLI and (eventually) the API call only this. It is the layer that
owns the run *lifecycle* -- nodes never set run status (see
``orchestration/nodes.py``), so this module is where PENDING -> RUNNING ->
{PAUSED, COMPLETED, FAILED, CANCELLED} actually gets written, in exactly one
place.

Two things worth knowing before reading the methods below:

Restart recovery is a constructor property, not a method.
    A ``RunService`` holds nothing about any particular run in memory -- no
    cache, no per-run state. Everything it needs comes from the database file
    behind its ``Services`` and from the ``thread_id`` stored on the ``Run``
    row. That is what makes "stop the process, start a new one, resume the
    same run" work: the new process builds a brand new ``RunService`` from the
    same file, and it has everything it needs.

Every graph invocation is funneled through :meth:`_advance`.
    ``graph.invoke`` can end four ways: a normal return (the run is done), an
    interrupt (the run is paused, and *why* is in the interrupt payload), one
    of our own control-flow exceptions (cancelled, rejected), or one of our
    tool exceptions escaping a step that could not recover. ``_advance`` is
    the one place all four are turned into a run status, so "what does this
    outcome mean for the run" is answered once rather than at every call site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from offboarding.domain.errors import (
    ApprovalRejected,
    BudgetExceeded,
    InvalidPlan,
    InvalidRunOperation,
    RunCancelled,
    SideEffectReconciliationRequired,
    ToolError,
    ToolNotAllowed,
)
from offboarding.domain.models import Run, SideEffectRecord, StepAttempt
from offboarding.domain.run import FailureReason, PauseReason, RunStatus
from offboarding.orchestration.graph import build_graph
from offboarding.orchestration.services import Services


class RunService:
    """Start, inspect, approve, resume and cancel offboarding runs."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._runs = services.runs
        self._trace = services.trace
        self._side_effects = services.side_effects
        self._graph = build_graph(services)

    # ------------------------------------------------------------------
    # Starting and inspecting
    # ------------------------------------------------------------------

    def start_run(
        self,
        employee_id: str,
        *,
        max_steps: int = 20,
        max_tool_calls: int = 30,
    ) -> Run:
        """Create a run and drive it to its first pause, failure, or completion."""
        run = self._runs.create(
            employee_id, max_steps=max_steps, max_tool_calls=max_tool_calls
        )
        run = self._runs.set_status(run.run_id, RunStatus.RUNNING)
        self._advance(
            run,
            {"run_id": run.run_id, "employee_id": employee_id, "events": []},
        )
        return self._runs.get(run.run_id)

    def get_run(self, run_id: str) -> Run:
        """Return a run's current lifecycle state.

        Raises:
            RunNotFound: no such run.
        """
        return self._runs.get(run_id)

    def get_trace(self, run_id: str) -> list[StepAttempt]:
        """Return the full step-level execution trace for a run, in order.

        Raises:
            RunNotFound: no such run -- checked explicitly so the error names
                the run rather than surfacing as an empty trace.
        """
        self._runs.get(run_id)
        return self._trace.list_for_run(run_id)

    def get_side_effects(self, run_id: str) -> list[SideEffectRecord]:
        """Return the idempotency ledger rows recorded for a run."""
        self._runs.get(run_id)
        return self._side_effects.list_for_run(run_id)

    def list_runs(self, limit: int = 50) -> list[Run]:
        """Return recent runs, newest first."""
        return self._runs.list_runs(limit)

    # ------------------------------------------------------------------
    # Approval gate
    # ------------------------------------------------------------------

    def approve(
        self, run_id: str, *, approver: str, note: str | None = None
    ) -> Run:
        """Approve a run paused for HR approval, resuming it.

        Raises:
            RunNotFound: no such run.
            InvalidRunOperation: the run is not currently paused for HR
                approval -- including a second call after the first already
                moved the run on, which is how a duplicate approval request
                is refused before it ever reaches the graph.
        """
        return self._resolve_approval(
            run_id, approved=True, approver=approver, note=note
        )

    def reject(
        self, run_id: str, *, approver: str, note: str | None = None
    ) -> Run:
        """Reject a run paused for HR approval, failing it.

        Raises:
            RunNotFound: no such run.
            InvalidRunOperation: the run is not currently paused for HR
                approval.
        """
        return self._resolve_approval(
            run_id, approved=False, approver=approver, note=note
        )

    def _resolve_approval(
        self, run_id: str, *, approved: bool, approver: str, note: str | None
    ) -> Run:
        run = self._runs.get(run_id)
        self._require_paused_for(run, PauseReason.HR_APPROVAL)
        run = self._runs.set_status(run.run_id, RunStatus.RUNNING)
        self._advance(
            run,
            Command(
                resume={"approved": approved, "approver": approver, "note": note}
            ),
        )
        return self._runs.get(run.run_id)

    # ------------------------------------------------------------------
    # External event
    # ------------------------------------------------------------------

    def submit_signed_document(
        self, run_id: str, *, document_id: str, signed_at: str | None = None
    ) -> Run:
        """Resume a run paused waiting for the signed exit paperwork.

        Raises:
            RunNotFound: no such run.
            InvalidRunOperation: the run is not currently paused for the
                signed document -- including a second submission after the
                first already resumed the run.
        """
        run = self._runs.get(run_id)
        self._require_paused_for(run, PauseReason.SIGNED_DOCUMENT)
        run = self._runs.set_status(run.run_id, RunStatus.RUNNING)
        event = {
            "document_id": document_id,
            "signed_at": signed_at or datetime.now(UTC).isoformat(),
        }
        self._advance(run, Command(resume=event))
        return self._runs.get(run.run_id)

    # ------------------------------------------------------------------
    # Recovering from a crash mid-step
    # ------------------------------------------------------------------

    def resume_run(self, run_id: str) -> Run:
        """Continue a run that stopped mid-step rather than at a designed pause.

        This is different from :meth:`approve` and :meth:`submit_signed_document`,
        which feed a value into a *pending* ``interrupt()`` via
        ``Command(resume=...)``. A process that dies mid-step -- the stretch
        scenario, where a side-effecting tool call lands but the process ends
        before the node returns -- leaves nothing waiting on ``interrupt()``, so
        there is no resume value to feed. The mechanism for that case is
        invoking the graph with no input at all: it reads the last checkpoint
        for this run's thread and continues from there, re-running the
        interrupted node from its own top.

        That re-run is exactly as safe as any other: the guard checks the
        budget and cancellation flag first, and any side-effecting call the
        step already reserved or completed goes through the ledger, which
        replays or reconciles it rather than repeating it. See
        ``tools/ledger.py``.

        Raises:
            RunNotFound: no such run.
            InvalidRunOperation: the run is not ``RUNNING`` -- a ``PENDING``
                run was never started, a ``PAUSED`` one should go through
                :meth:`approve` or :meth:`submit_signed_document` instead, and
                a terminal run has nothing left to continue.
        """
        run = self._runs.get(run_id)
        if run.status is not RunStatus.RUNNING:
            raise InvalidRunOperation(
                f"run {run.run_id} is {run.status.value}, not running -- a "
                "paused run resumes via approve() or submit_signed_document(); "
                "a pending, completed, failed or cancelled run has nothing to "
                "continue"
            )
        self._advance(run, None)
        return self._runs.get(run.run_id)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_run(self, run_id: str) -> Run:
        """Cancel a run.

        A run sitting at ``PENDING`` or ``PAUSED`` has nothing executing, so it
        is cancelled immediately. A ``RUNNING`` run only exists while some
        process's call to this service is actually inside ``graph.invoke`` --
        cancelling it sets a flag that the guard in every node checks *before*
        starting its step, so a side effect already in flight is never torn
        out from under itself. Whichever call is driving that step will see
        :class:`RunCancelled` and record ``CANCELLED`` itself, via
        :meth:`_advance`.

        Raises:
            RunNotFound: no such run.
            InvalidRunOperation: the run has already reached a terminal
                status.
        """
        run = self._runs.get(run_id)
        if run.is_terminal:
            raise InvalidRunOperation(
                f"run {run.run_id} is already {run.status.value}"
            )
        if run.status in (RunStatus.PENDING, RunStatus.PAUSED):
            return self._runs.set_status(run.run_id, RunStatus.CANCELLED)
        return self._runs.request_cancel(run.run_id)

    # ------------------------------------------------------------------
    # Driving the graph
    # ------------------------------------------------------------------

    def _advance(self, run: Run, graph_input: Any) -> None:
        """Invoke the graph once and reconcile the run's status with the outcome.

        The four outcomes graph.invoke can produce, and what each means for the
        run, are handled here and nowhere else:

        * a normal return with no pending interrupt -> the run is COMPLETED.
        * a normal return carrying ``__interrupt__`` -> the run is PAUSED, with
          the pause reason read straight out of the interrupt payload the
          pausing node built.
        * :class:`RunCancelled` -> CANCELLED. Not a failure: the operator asked
          for this.
        * anything else our own code raises -> FAILED, with a
          :class:`FailureReason` chosen by exception type.
        """
        config = self._config_for(run)

        try:
            result = self._graph.invoke(graph_input, config)
        except RunCancelled:
            self._runs.set_status(run.run_id, RunStatus.CANCELLED)
            return
        except ApprovalRejected as exc:
            self._fail(run, FailureReason.APPROVAL_REJECTED, exc)
            return
        except BudgetExceeded as exc:
            self._fail(run, FailureReason.BUDGET_EXCEEDED, exc)
            return
        except ToolNotAllowed as exc:
            self._fail(run, FailureReason.TOOL_NOT_ALLOWED, exc)
            return
        except InvalidPlan as exc:
            self._fail(run, FailureReason.INVALID_PLAN, exc)
            return
        except SideEffectReconciliationRequired as exc:
            self._fail(run, FailureReason.NEEDS_RECONCILIATION, exc)
            return
        except ToolError as exc:
            # Catches what nothing more specific above already caught: a
            # TransientToolError that exhausted its retries, or a
            # PermanentToolError with no dedicated FailureReason of its own.
            self._fail(run, FailureReason.TOOL_FAILED, exc)
            return

        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            self._runs.set_status(
                run.run_id,
                RunStatus.PAUSED,
                pause_reason=PauseReason(payload["pause_reason"]),
            )
        else:
            self._runs.set_status(run.run_id, RunStatus.COMPLETED)

    def _fail(self, run: Run, reason: FailureReason, exc: Exception) -> None:
        self._runs.set_status(
            run.run_id,
            RunStatus.FAILED,
            failure_reason=reason,
            failure_detail=str(exc),
        )

    def _require_paused_for(self, run: Run, reason: PauseReason) -> None:
        """Refuse an operation unless the run is paused for exactly ``reason``.

        This is the first line of duplicate-request defence, ahead of the
        ledger: a second ``approve()`` after the run has already moved past the
        approval gate sees a status other than ``PAUSED`` (or the wrong pause
        reason) and is refused here, before a second ``Command(resume=...)``
        ever reaches the graph.

        Raises:
            InvalidRunOperation: the run is not paused for ``reason``.
        """
        if run.status is not RunStatus.PAUSED or run.pause_reason is not reason:
            current = run.status.value
            if run.pause_reason:
                current += f" ({run.pause_reason.value})"
            raise InvalidRunOperation(
                f"run {run.run_id} is {current}, not paused for {reason.value}"
            )

    @staticmethod
    def _config_for(run: Run) -> dict[str, Any]:
        return {"configurable": {"thread_id": run.thread_id}}
