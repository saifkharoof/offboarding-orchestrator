"""Persistence behaviour: transitions, trace shape, ledger uniqueness, reload."""

from __future__ import annotations

import pytest

from offboarding.domain.errors import InvalidStateTransition, RunNotFound
from offboarding.domain.run import (
    FailureReason,
    PauseReason,
    RunStatus,
    StepName,
    StepStatus,
)
from offboarding.persistence.db import connect, initialize
from offboarding.persistence.repositories import (
    RunRepository,
    SideEffectRepository,
    TraceRepository,
)


class TestRunRepository:
    def test_new_run_starts_pending(self, runs):
        run = runs.create("emp-001")
        assert run.status is RunStatus.PENDING
        assert run.cancel_requested is False
        assert run.step_count == 0

    def test_thread_id_is_derived_from_run_id(self, runs):
        run = runs.create("emp-001")
        assert run.thread_id == f"thread_{run.run_id}"

    def test_missing_run_raises(self, runs):
        with pytest.raises(RunNotFound):
            runs.get("run_nope")

    def test_illegal_transition_is_rejected_at_the_store(self, runs):
        run = runs.create("emp-001")
        runs.set_status(run.run_id, RunStatus.RUNNING)
        runs.set_status(run.run_id, RunStatus.COMPLETED)
        with pytest.raises(InvalidStateTransition):
            runs.set_status(run.run_id, RunStatus.RUNNING)

    def test_pause_records_its_reason(self, runs):
        run = runs.create("emp-001")
        runs.set_status(run.run_id, RunStatus.RUNNING)
        paused = runs.set_status(
            run.run_id, RunStatus.PAUSED, pause_reason=PauseReason.HR_APPROVAL
        )
        assert paused.status is RunStatus.PAUSED
        assert paused.pause_reason is PauseReason.HR_APPROVAL

    def test_resuming_clears_the_pause_reason(self, runs):
        run = runs.create("emp-001")
        runs.set_status(run.run_id, RunStatus.RUNNING)
        runs.set_status(
            run.run_id, RunStatus.PAUSED, pause_reason=PauseReason.HR_APPROVAL
        )
        resumed = runs.set_status(run.run_id, RunStatus.RUNNING)
        assert resumed.pause_reason is None

    def test_failure_reason_is_retained(self, runs):
        run = runs.create("emp-001")
        runs.set_status(run.run_id, RunStatus.RUNNING)
        failed = runs.set_status(
            run.run_id,
            RunStatus.FAILED,
            failure_reason=FailureReason.TOOL_FAILED,
            failure_detail="iam unreachable",
        )
        assert failed.failure_reason is FailureReason.TOOL_FAILED
        assert failed.failure_detail == "iam unreachable"

    def test_cancel_sets_a_flag_without_moving_the_run(self, runs):
        run = runs.create("emp-001")
        runs.set_status(run.run_id, RunStatus.RUNNING)
        flagged = runs.request_cancel(run.run_id)
        assert flagged.cancel_requested is True
        assert flagged.status is RunStatus.RUNNING

    def test_counters_accumulate(self, runs):
        run = runs.create("emp-001")
        runs.bump_counters(run.run_id, steps=1)
        updated = runs.bump_counters(run.run_id, steps=1, tool_calls=2)
        assert (updated.step_count, updated.tool_call_count) == (2, 2)


class TestTraceRepository:
    def test_first_attempt_is_numbered_one_and_running(self, runs, trace):
        run = runs.create("emp-001")
        attempt = trace.start_attempt(run.run_id, StepName.FETCH_EMPLOYEE)
        assert attempt.attempt == 1
        assert attempt.status is StepStatus.RUNNING
        assert attempt.ended_at is None

    def test_retry_adds_a_row_rather_than_overwriting(self, runs, trace):
        run = runs.create("emp-001")
        first = trace.start_attempt(run.run_id, StepName.REVOKE_ACCESS)
        trace.fail_attempt(
            first.id, error_type="TransientToolError", error_message="503"
        )
        second = trace.start_attempt(run.run_id, StepName.REVOKE_ACCESS)
        trace.complete_attempt(second.id)

        rows = trace.list_for_run(run.run_id)
        assert [r.attempt for r in rows] == [1, 2]
        assert [r.status for r in rows] == [StepStatus.FAILED, StepStatus.COMPLETED]
        # The failure is still visible after the retry succeeds.
        assert rows[0].error_type == "TransientToolError"

    def test_completed_step_cannot_start_another_attempt(self, runs, trace):
        run = runs.create("emp-001")
        attempt = trace.start_attempt(run.run_id, StepName.REVOKE_ACCESS)
        trace.complete_attempt(attempt.id)
        with pytest.raises(InvalidStateTransition):
            trace.start_attempt(run.run_id, StepName.REVOKE_ACCESS)

    def test_pause_records_reason_and_allows_a_resume_attempt(self, runs, trace):
        run = runs.create("emp-001")
        first = trace.start_attempt(run.run_id, StepName.AWAIT_HR_APPROVAL)
        paused = trace.pause_attempt(
            first.id, pause_reason=PauseReason.HR_APPROVAL
        )
        assert paused.status is StepStatus.PAUSED
        assert paused.pause_reason is PauseReason.HR_APPROVAL

        resumed = trace.start_attempt(run.run_id, StepName.AWAIT_HR_APPROVAL)
        assert resumed.attempt == 2

    def test_trace_is_ordered_across_steps(self, runs, trace):
        run = runs.create("emp-001")
        for step in (StepName.FETCH_EMPLOYEE, StepName.PLAN_DEPROVISIONING):
            attempt = trace.start_attempt(run.run_id, step)
            trace.complete_attempt(attempt.id)
        assert [r.step_name for r in trace.list_for_run(run.run_id)] == [
            StepName.FETCH_EMPLOYEE,
            StepName.PLAN_DEPROVISIONING,
        ]

    def test_detail_round_trips_as_json(self, runs, trace):
        run = runs.create("emp-001")
        attempt = trace.start_attempt(run.run_id, StepName.FETCH_EMPLOYEE)
        stored = trace.complete_attempt(
            attempt.id, detail={"employee": "emp-001", "systems": 3}
        )
        assert stored.detail == {"employee": "emp-001", "systems": 3}


class TestSideEffectRepository:
    def test_second_reservation_of_a_key_is_refused(self, runs, ledger_store):
        run = runs.create("emp-001")
        kwargs = {
            "run_id": run.run_id,
            "step_name": StepName.REVOKE_ACCESS,
            "tool_name": "iam.revoke",
        }
        first = ledger_store.try_reserve("k1", **kwargs)
        second = ledger_store.try_reserve("k1", **kwargs)
        assert first is not None
        # The UNIQUE constraint, not an application check, is what refuses this.
        assert second is None

    def test_completed_record_caches_its_result(self, runs, ledger_store):
        run = runs.create("emp-001")
        ledger_store.try_reserve(
            "k1",
            run_id=run.run_id,
            step_name=StepName.SEND_EXIT_PAPERWORK,
            tool_name="documents.send",
        )
        ledger_store.mark_in_progress("k1")
        record = ledger_store.mark_completed("k1", {"document_id": "doc_7"})
        assert record.state == SideEffectRepository.COMPLETED
        assert record.result == {"document_id": "doc_7"}
        assert record.completed_at is not None

    def test_release_permits_a_fresh_reservation(self, runs, ledger_store):
        run = runs.create("emp-001")
        kwargs = {
            "run_id": run.run_id,
            "step_name": StepName.REVOKE_ACCESS,
            "tool_name": "iam.revoke",
        }
        ledger_store.try_reserve("k1", **kwargs)
        ledger_store.release("k1")
        assert ledger_store.try_reserve("k1", **kwargs) is not None


class TestDurability:
    def test_state_survives_closing_every_connection(self, db_path):
        """The restart guarantee at the storage layer.

        Nothing here is held in memory: a second process opening the same file
        sees the same run, trace and ledger.
        """
        first = connect(db_path)
        initialize(first)
        run = RunRepository(first).create("emp-001")
        RunRepository(first).set_status(run.run_id, RunStatus.RUNNING)
        attempt = TraceRepository(first).start_attempt(
            run.run_id, StepName.FETCH_EMPLOYEE
        )
        TraceRepository(first).complete_attempt(attempt.id)
        SideEffectRepository(first).try_reserve(
            "k1",
            run_id=run.run_id,
            step_name=StepName.REVOKE_ACCESS,
            tool_name="iam.revoke",
        )
        first.close()

        second = connect(db_path)
        initialize(second)  # idempotent on an existing database
        reloaded = RunRepository(second).get(run.run_id)
        assert reloaded.status is RunStatus.RUNNING
        assert len(TraceRepository(second).list_for_run(run.run_id)) == 1
        assert SideEffectRepository(second).get("k1") is not None
        second.close()
