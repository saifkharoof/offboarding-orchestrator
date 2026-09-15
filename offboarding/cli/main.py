"""CLI over :class:`RunService` -- every operation the brief's interface asks for.

Every command builds a fresh :class:`RunService` from the configured database
file and does nothing else stateful. That is deliberate: it is what makes
"start a run, kill the process, run the CLI again to resume it" true by
construction rather than by care taken at each call site -- there is no
process-lifetime state to lose.

Configuration comes from the environment (``OFFBOARDING_DB_PATH``,
``OFFBOARDING_LLM_PROVIDER``, ``OFFBOARDING_IAM_FAIL_TIMES``, ...), loaded from
a ``.env`` file if one is present. See ``.env.example``.
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from offboarding.domain.errors import InvalidRunOperation, RunNotFound
from offboarding.domain.models import Run, SideEffectRecord, StepAttempt
from offboarding.orchestration.run_service import RunService
from offboarding.orchestration.services import build_services

app = typer.Typer(
    add_completion=False,
    help="Employee offboarding agent orchestrator.",
    no_args_is_help=True,
)


def _service() -> RunService:
    load_dotenv()
    return RunService(build_services())


def _fail(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(1)


def _print_run(run: Run) -> None:
    typer.echo(f"run_id       {run.run_id}")
    typer.echo(f"employee_id  {run.employee_id}")
    typer.echo(f"status       {run.status.value}")
    if run.pause_reason:
        typer.echo(f"paused_for   {run.pause_reason.value}")
    if run.failure_reason:
        typer.echo(f"failed_with  {run.failure_reason.value}: {run.failure_detail}")
    typer.echo(f"steps        {run.step_count}/{run.max_steps}")
    typer.echo(f"tool_calls   {run.tool_call_count}/{run.max_tool_calls}")


def _print_run_line(run: Run) -> None:
    status = run.status.value
    if run.pause_reason:
        status += f"[{run.pause_reason.value}]"
    typer.echo(f"{run.run_id:<20} {status:<28} emp={run.employee_id}")


def _print_trace_line(attempt: StepAttempt) -> None:
    parts = [f"#{attempt.attempt:<2} {attempt.step_name:<24} {attempt.status.value}"]
    if attempt.tool_invoked:
        parts.append(f"tool={attempt.tool_invoked}")
    if attempt.pause_reason:
        parts.append(f"pause={attempt.pause_reason.value}")
    if attempt.error_type:
        parts.append(f"error={attempt.error_type}: {attempt.error_message}")
    if attempt.detail.get("replayed"):
        parts.append("[replayed: idempotent no-op]")
    if attempt.detail.get("reconciled"):
        parts.append("[reconciled: recovered from provider after a crash]")
    typer.echo("  ".join(parts))


def _print_side_effect_line(record: SideEffectRecord) -> None:
    typer.echo(
        f"{record.step_name:<24} {record.state:<12} key={record.idempotency_key}"
    )


# --------------------------------------------------------------------------
# Start a new agent run.
# --------------------------------------------------------------------------


@app.command()
def start(
    employee_id: str,
    max_steps: int = typer.Option(20, help="Bounded-execution step limit."),
    max_tool_calls: int = typer.Option(
        30, help="Bounded-execution tool-call limit."
    ),
) -> None:
    """Start a new offboarding run for EMPLOYEE_ID."""
    try:
        run = _service().start_run(
            employee_id, max_steps=max_steps, max_tool_calls=max_tool_calls
        )
    except RunNotFound as exc:  # pragma: no cover - defensive
        _fail(exc)
        return
    _print_run(run)


# --------------------------------------------------------------------------
# Inspect the current run and step state.
# --------------------------------------------------------------------------


@app.command()
def show(run_id: str) -> None:
    """Show RUN_ID's current lifecycle status."""
    try:
        run = _service().get_run(run_id)
    except RunNotFound as exc:
        _fail(exc)
        return
    _print_run(run)


@app.command(name="list")
def list_runs(limit: int = typer.Option(20, help="Maximum runs to show.")) -> None:
    """List recent runs, newest first."""
    for run in _service().list_runs(limit):
        _print_run_line(run)


# --------------------------------------------------------------------------
# Approve or reject a paused approval step.
# --------------------------------------------------------------------------


@app.command()
def approve(
    run_id: str,
    approver: str = typer.Option(..., help="Who is approving."),
    note: str = typer.Option(None, help="Optional note recorded on the trace."),
) -> None:
    """Approve RUN_ID, currently paused for HR approval."""
    try:
        run = _service().approve(run_id, approver=approver, note=note)
    except (RunNotFound, InvalidRunOperation) as exc:
        _fail(exc)
        return
    _print_run(run)


@app.command()
def reject(
    run_id: str,
    approver: str = typer.Option(..., help="Who is rejecting."),
    note: str = typer.Option(None, help="Why. Recorded on the trace."),
) -> None:
    """Reject RUN_ID, currently paused for HR approval."""
    try:
        run = _service().reject(run_id, approver=approver, note=note)
    except (RunNotFound, InvalidRunOperation) as exc:
        _fail(exc)
        return
    _print_run(run)


# --------------------------------------------------------------------------
# Resume a run after an external wait condition is satisfied.
# --------------------------------------------------------------------------


@app.command()
def sign(
    run_id: str,
    document_id: str = typer.Option(..., help="The signed document's id."),
    signed_at: str = typer.Option(
        None, help="ISO timestamp; defaults to now if omitted."
    ),
) -> None:
    """Submit the signed exit paperwork for RUN_ID, resuming it."""
    try:
        run = _service().submit_signed_document(
            run_id, document_id=document_id, signed_at=signed_at
        )
    except (RunNotFound, InvalidRunOperation) as exc:
        _fail(exc)
        return
    _print_run(run)


@app.command()
def resume(run_id: str) -> None:
    """Continue RUN_ID after a crash mid-step (not a designed pause).

    Use this only when `show` reports a run stuck RUNNING with no pause
    reason -- the process ended partway through a step. A run paused for
    approval or a signature should be resumed with `approve`/`reject`/`sign`
    instead.
    """
    try:
        run = _service().resume_run(run_id)
    except (RunNotFound, InvalidRunOperation) as exc:
        _fail(exc)
        return
    _print_run(run)


# --------------------------------------------------------------------------
# Cancel a run.
# --------------------------------------------------------------------------


@app.command()
def cancel(run_id: str) -> None:
    """Cancel RUN_ID."""
    try:
        run = _service().cancel_run(run_id)
    except (RunNotFound, InvalidRunOperation) as exc:
        _fail(exc)
        return
    _print_run(run)


# --------------------------------------------------------------------------
# Inspect the execution trace for a run.
# --------------------------------------------------------------------------


@app.command()
def trace(run_id: str) -> None:
    """Show RUN_ID's step-level execution trace, in order."""
    try:
        attempts = _service().get_trace(run_id)
    except RunNotFound as exc:
        _fail(exc)
        return
    for attempt in attempts:
        _print_trace_line(attempt)


@app.command(name="side-effects")
def side_effects(run_id: str) -> None:
    """Show RUN_ID's idempotency ledger -- one row per protected side effect."""
    try:
        records = _service().get_side_effects(run_id)
    except RunNotFound as exc:
        _fail(exc)
        return
    for record in records:
        _print_side_effect_line(record)


if __name__ == "__main__":
    app()
