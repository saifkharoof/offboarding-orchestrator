"""RunService: the public surface, lifecycle correctness, and restart recovery.

The two tests under TestRestartRecovery are the ones that matter most against
the brief: every service instance here is built fresh from a db_path, never
reused across a "process boundary" in the same test, so a passing test is
actually evidence the durability claim holds -- not just that objects still
in memory remember what happened.
"""

from __future__ import annotations

import pytest

from offboarding.domain.errors import InvalidRunOperation, RunNotFound
from offboarding.domain.run import FailureReason, PauseReason, RunStatus, StepName
from offboarding.orchestration.run_service import RunService
from offboarding.orchestration.services import build_services
from offboarding.tools.iam import SimulatedProcessDeath
from offboarding.tools.providers import ProviderStore


def approve(service: RunService, run_id: str):
    return service.approve(run_id, approver="priya.raman", note="Confirmed.")


def sign(service: RunService, run_id: str, document_id: str = "doc_e2e"):
    return service.submit_signed_document(run_id, document_id=document_id)


@pytest.fixture
def service(db_path) -> RunService:
    return RunService(build_services(db_path=db_path))


class TestStartRun:
    def test_start_pauses_for_hr_approval(self, service):
        run = service.start_run("emp-001")
        assert run.status is RunStatus.PAUSED
        assert run.pause_reason is PauseReason.HR_APPROVAL

    def test_unknown_employee_fails_the_run(self, service):
        run = service.start_run("emp-does-not-exist")
        assert run.status is RunStatus.FAILED
        assert run.failure_reason is FailureReason.TOOL_FAILED


class TestApprovalGate:
    def test_approve_advances_to_the_second_pause(self, service):
        run = service.start_run("emp-001")
        approved = approve(service, run.run_id)
        assert approved.status is RunStatus.PAUSED
        assert approved.pause_reason is PauseReason.SIGNED_DOCUMENT

    def test_reject_fails_the_run_with_the_right_reason(self, service):
        run = service.start_run("emp-001")
        rejected = service.reject(run.run_id, approver="priya.raman", note="No.")
        assert rejected.status is RunStatus.FAILED
        assert rejected.failure_reason is FailureReason.APPROVAL_REJECTED

    def test_rejection_never_reaches_the_high_risk_tool(self, service, db_path):
        run = service.start_run("emp-001")
        service.reject(run.run_id, approver="priya.raman")
        assert ProviderStore(service._services.conn, "iam").count() == 0

    def test_duplicate_approve_is_refused_not_re_executed(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)
        with pytest.raises(InvalidRunOperation):
            approve(service, run.run_id)
        # Still exactly one revocation: the second approve never touched the
        # graph at all.
        assert ProviderStore(service._services.conn, "iam").count() == 1

    def test_approving_a_run_paused_for_the_wrong_reason_is_refused(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)  # now paused for SIGNED_DOCUMENT
        with pytest.raises(InvalidRunOperation, match="not paused for"):
            approve(service, run.run_id)

    def test_approving_a_missing_run_raises_run_not_found(self, service):
        with pytest.raises(RunNotFound):
            approve(service, "run_does_not_exist")


class TestExternalEvent:
    def test_signing_completes_the_run(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)
        completed = service.submit_signed_document(
            run.run_id, document_id="doc_123"
        )
        assert completed.status is RunStatus.COMPLETED

    def test_signing_before_approval_is_refused(self, service):
        run = service.start_run("emp-001")
        with pytest.raises(InvalidRunOperation, match="not paused for"):
            sign(service, run.run_id)

    def test_duplicate_signature_is_refused_not_re_executed(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)
        sign(service, run.run_id)
        with pytest.raises(InvalidRunOperation):
            sign(service, run.run_id)


class TestResumeRun:
    def test_resuming_a_paused_run_is_refused(self, service):
        run = service.start_run("emp-001")  # paused, not mid-step
        with pytest.raises(InvalidRunOperation, match="not running"):
            service.resume_run(run.run_id)

    def test_resuming_a_pending_run_is_refused(self, service):
        run = service._runs.create("emp-001")  # never started
        with pytest.raises(InvalidRunOperation, match="not running"):
            service.resume_run(run.run_id)

    def test_resuming_a_completed_run_is_refused(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)
        sign(service, run.run_id)
        with pytest.raises(InvalidRunOperation, match="not running"):
            service.resume_run(run.run_id)


class TestCancellation:
    def test_cancelling_a_paused_run_is_immediate(self, service):
        run = service.start_run("emp-001")
        cancelled = service.cancel_run(run.run_id)
        assert cancelled.status is RunStatus.CANCELLED

    def test_cancelling_a_terminal_run_is_refused(self, service):
        run = service.start_run("emp-001")
        service.reject(run.run_id, approver="priya.raman")
        with pytest.raises(InvalidRunOperation):
            service.cancel_run(run.run_id)

    def test_cancel_flag_stops_the_next_step_of_a_running_call(self, service):
        # Simulates a cancel request arriving from another process while this
        # one is mid-run: flag it, then the in-flight advance sees the guard's
        # RunCancelled and records CANCELLED itself.
        run = service.start_run("emp-001")
        service._runs.set_status(run.run_id, RunStatus.RUNNING)
        service._runs.request_cancel(run.run_id)
        service._advance(service.get_run(run.run_id), None)
        assert service.get_run(run.run_id).status is RunStatus.CANCELLED


class TestInspection:
    def test_trace_covers_every_step_reached_so_far(self, service):
        run = service.start_run("emp-001")
        traced = {r.step_name for r in service.get_trace(run.run_id)}
        assert traced == {
            StepName.FETCH_EMPLOYEE,
            StepName.PLAN_DEPROVISIONING,
            StepName.AWAIT_HR_APPROVAL,
        }

    def test_trace_of_a_missing_run_raises(self, service):
        with pytest.raises(RunNotFound):
            service.get_trace("run_does_not_exist")

    def test_list_runs_returns_newest_first(self, service):
        first = service.start_run("emp-001")
        second = service.start_run("emp-002")
        ids = [r.run_id for r in service.list_runs()]
        assert ids.index(second.run_id) < ids.index(first.run_id)

    def test_side_effects_are_visible_once_revoked(self, service):
        run = service.start_run("emp-001")
        approve(service, run.run_id)
        records = service.get_side_effects(run.run_id)
        assert {r.step_name for r in records} == {
            StepName.REVOKE_ACCESS,
            StepName.SEND_EXIT_PAPERWORK,
        }
        assert all(r.state == "completed" for r in records)


class TestRestartRecovery:
    """Stop the process, start a new one, resume the same run."""

    def test_completes_a_run_paused_across_a_process_boundary(self, db_path):
        """The brief's literal scenario.

        Steps 1-2 complete, step 3 (and step 5) perform side effects, the run
        later pauses. The process is stopped -- every connection this service
        holds is closed -- and a completely independent RunService is built
        from nothing but the database file. It resumes the same run to
        completion, and neither side effect fired twice.
        """
        service_a = RunService(build_services(db_path=db_path))
        run = service_a.start_run("emp-001")
        approve(service_a, run.run_id)  # revokes access, sends paperwork, pauses
        assert run.run_id  # sanity: we have something to resume

        paused = service_a.get_run(run.run_id)
        assert paused.status is RunStatus.PAUSED
        assert paused.pause_reason is PauseReason.SIGNED_DOCUMENT

        # "Stop the process": drop every reference and close the connection.
        service_a._services.conn.close()
        del service_a

        # "Start a new one": nothing here is shared with service_a.
        service_b = RunService(build_services(db_path=db_path))
        completed = service_b.submit_signed_document(
            run.run_id, document_id="doc_restart"
        )

        assert completed.status is RunStatus.COMPLETED
        conn_b = service_b._services.conn
        assert ProviderStore(conn_b, "iam").count() == 1
        assert ProviderStore(conn_b, "documents").count() == 1
        assert ProviderStore(conn_b, "archive").count() == 1

        trace = service_b.get_trace(run.run_id)
        completed_steps = [r.step_name for r in trace if r.status.value == "completed"]
        # Every step completes exactly once: nothing the first process did was
        # replayed or repeated by the second one just resuming past a pause.
        assert completed_steps.count(StepName.REVOKE_ACCESS) == 1
        assert completed_steps.count(StepName.SEND_EXIT_PAPERWORK) == 1

    def test_reconciles_a_side_effect_that_landed_before_the_crash(self, db_path):
        """The stretch scenario: the side effect lands, then the process dies
        before recording it -- driven through the full RunService and graph,
        not just the ledger in isolation (see TestCrashRecovery in
        test_tools.py for that).
        """
        dying_services = build_services(db_path=db_path, iam_fail_times=0)
        # Swap in a crashing IAM tool after construction, same registry
        # instance the graph and executor already hold.
        from offboarding.tools.iam import IamTool

        dying_services.executor._registry._tools["iam.revoke_access"] = IamTool(
            dying_services.conn, crash_after_apply=True
        )
        service_a = RunService(dying_services)
        run = service_a.start_run("emp-001")

        with pytest.raises(SimulatedProcessDeath):
            approve(service_a, run.run_id)

        # Ground truth: IAM actually applied the revocation before dying.
        assert ProviderStore(dying_services.conn, "iam").count() == 1
        # But nothing recorded the run as having moved past the crash: the
        # exception escaped _advance entirely, so run status is stuck RUNNING,
        # exactly as it would be if the process had really been killed here.
        assert service_a.get_run(run.run_id).status is RunStatus.RUNNING

        dying_services.conn.close()
        del service_a, dying_services

        # A fresh process resumes the same run. No interrupt was pending when
        # it died -- it crashed mid-step, not mid-pause -- so this is
        # resume_run(), not approve(): there is no resume value to feed,
        # only a checkpoint to continue from.
        service_b = RunService(build_services(db_path=db_path))
        recovered = service_b.resume_run(run.run_id)

        assert recovered.status is RunStatus.PAUSED
        assert recovered.pause_reason is PauseReason.SIGNED_DOCUMENT
        # The reconciled attempt is visible in the trace.
        revoke_rows = [
            r
            for r in service_b.get_trace(run.run_id)
            if r.step_name == StepName.REVOKE_ACCESS
        ]
        assert revoke_rows[-1].detail.get("reconciled") is True
        # Still exactly one revocation ever applied.
        assert ProviderStore(service_b._services.conn, "iam").count() == 1
