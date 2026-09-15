"""FastAPI over :class:`RunService` -- the same operations as the CLI, over HTTP.

A thin layer on purpose: every endpoint calls exactly one ``RunService``
method and returns its result. No orchestration logic lives here -- that
would defeat the point of having a single public surface both clients share.

One ``RunService`` is built lazily on first request and reused for the life of
the server process, which is the correct analogue of the CLI's "fresh per
invocation" pattern: a CLI invocation *is* a process, so it always starts
fresh; an API server *is* one long-running process, so it builds once. Either
way, restarting the process is what rebuilds everything from the database file
-- that guarantee doesn't depend on which client is driving it.

Run with:

    uvicorn offboarding.api.main:app --reload
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from offboarding.domain.errors import InvalidRunOperation, RunNotFound
from offboarding.domain.models import Run, SideEffectRecord, StepAttempt
from offboarding.orchestration.run_service import RunService
from offboarding.orchestration.services import build_services

load_dotenv()

app = FastAPI(
    title="Offboarding Orchestrator",
    description="Durable agent orchestration for employee offboarding.",
    version="0.1.0",
)

_service: RunService | None = None


def get_service() -> RunService:
    """Return the process-lifetime RunService, building it on first use."""
    global _service
    if _service is None:
        _service = RunService(build_services())
    return _service


@app.exception_handler(RunNotFound)
def _handle_not_found(request: Request, exc: RunNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidRunOperation)
def _handle_invalid_operation(
    request: Request, exc: InvalidRunOperation
) -> JSONResponse:
    # 409 Conflict: the request is well-formed but the run's current state
    # doesn't admit it -- including a duplicate approval or signature.
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------


class StartRunRequest(BaseModel):
    employee_id: str
    max_steps: int = 20
    max_tool_calls: int = 30


class ApprovalRequest(BaseModel):
    approver: str
    note: str | None = None


class SignRequest(BaseModel):
    document_id: str
    signed_at: str | None = None


# --------------------------------------------------------------------------
# Start a new agent run.
# --------------------------------------------------------------------------


@app.post("/runs", response_model=Run, status_code=201)
def start_run(
    body: StartRunRequest, service: RunService = Depends(get_service)
) -> Run:
    """Start a new offboarding run."""
    return service.start_run(
        body.employee_id,
        max_steps=body.max_steps,
        max_tool_calls=body.max_tool_calls,
    )


# --------------------------------------------------------------------------
# Inspect the current run and step state.
# --------------------------------------------------------------------------


@app.get("/runs", response_model=list[Run])
def list_runs(limit: int = 50, service: RunService = Depends(get_service)) -> list[Run]:
    """List recent runs, newest first."""
    return service.list_runs(limit)


@app.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str, service: RunService = Depends(get_service)) -> Run:
    """Get one run's current lifecycle status."""
    return service.get_run(run_id)


# --------------------------------------------------------------------------
# Approve or reject a paused approval step.
# --------------------------------------------------------------------------


@app.post("/runs/{run_id}/approve", response_model=Run)
def approve(
    run_id: str, body: ApprovalRequest, service: RunService = Depends(get_service)
) -> Run:
    """Approve a run paused for HR approval."""
    return service.approve(run_id, approver=body.approver, note=body.note)


@app.post("/runs/{run_id}/reject", response_model=Run)
def reject(
    run_id: str, body: ApprovalRequest, service: RunService = Depends(get_service)
) -> Run:
    """Reject a run paused for HR approval."""
    return service.reject(run_id, approver=body.approver, note=body.note)


# --------------------------------------------------------------------------
# Resume a run after an external wait condition is satisfied.
# --------------------------------------------------------------------------


@app.post("/runs/{run_id}/sign", response_model=Run)
def sign(
    run_id: str, body: SignRequest, service: RunService = Depends(get_service)
) -> Run:
    """Submit the signed exit paperwork, resuming a run paused for it."""
    return service.submit_signed_document(
        run_id, document_id=body.document_id, signed_at=body.signed_at
    )


@app.post("/runs/{run_id}/resume", response_model=Run)
def resume(run_id: str, service: RunService = Depends(get_service)) -> Run:
    """Continue a run that stopped mid-step (a crash, not a designed pause).

    See :meth:`RunService.resume_run` -- this is not for a run paused for
    approval or a signature; use `approve`/`reject`/`sign` for those.
    """
    return service.resume_run(run_id)


# --------------------------------------------------------------------------
# Cancel a run.
# --------------------------------------------------------------------------


@app.post("/runs/{run_id}/cancel", response_model=Run)
def cancel(run_id: str, service: RunService = Depends(get_service)) -> Run:
    """Cancel a run."""
    return service.cancel_run(run_id)


# --------------------------------------------------------------------------
# Inspect the execution trace for a run.
# --------------------------------------------------------------------------


@app.get("/runs/{run_id}/trace", response_model=list[StepAttempt])
def get_trace(
    run_id: str, service: RunService = Depends(get_service)
) -> list[StepAttempt]:
    """Get the step-level execution trace for a run, in order."""
    return service.get_trace(run_id)


@app.get("/runs/{run_id}/side-effects", response_model=list[SideEffectRecord])
def get_side_effects(
    run_id: str, service: RunService = Depends(get_service)
) -> list[SideEffectRecord]:
    """Get the idempotency ledger rows recorded for a run."""
    return service.get_side_effects(run_id)
