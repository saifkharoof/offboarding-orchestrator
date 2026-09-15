"""The lifecycle rules themselves, independent of storage."""

from __future__ import annotations

import pytest

from offboarding.domain.errors import InvalidStateTransition
from offboarding.domain.run import (
    ALLOWED_RUN_TRANSITIONS,
    RunStatus,
    StepStatus,
    is_terminal,
    transition_run,
    transition_step,
)


class TestRunTransitions:
    def test_happy_path_sequence_is_legal(self):
        status = RunStatus.PENDING
        for target in (
            RunStatus.RUNNING,
            RunStatus.PAUSED,
            RunStatus.RUNNING,
            RunStatus.COMPLETED,
        ):
            status = transition_run(status, target)
        assert status is RunStatus.COMPLETED

    @pytest.mark.parametrize(
        "terminal", [s for s in RunStatus if is_terminal(s)]
    )
    def test_terminal_statuses_admit_no_transitions(self, terminal):
        assert ALLOWED_RUN_TRANSITIONS[terminal] == frozenset()

    def test_completed_run_cannot_be_resumed(self):
        with pytest.raises(InvalidStateTransition):
            transition_run(RunStatus.COMPLETED, RunStatus.RUNNING)

    def test_cancelled_run_cannot_be_completed(self):
        with pytest.raises(InvalidStateTransition):
            transition_run(RunStatus.CANCELLED, RunStatus.COMPLETED)

    def test_pending_run_cannot_jump_straight_to_paused(self):
        # A run only pauses from RUNNING; pausing before it has started would
        # mean a pause with no step attached to it.
        with pytest.raises(InvalidStateTransition):
            transition_run(RunStatus.PENDING, RunStatus.PAUSED)

    def test_paused_run_can_be_cancelled(self):
        assert transition_run(RunStatus.PAUSED, RunStatus.CANCELLED) is (
            RunStatus.CANCELLED
        )

    def test_every_status_has_a_transition_entry(self):
        assert set(ALLOWED_RUN_TRANSITIONS) == set(RunStatus)


class TestStepTransitions:
    def test_retry_is_running_to_running(self):
        assert transition_step(StepStatus.RUNNING, StepStatus.RUNNING) is (
            StepStatus.RUNNING
        )

    def test_failed_step_may_be_retried(self):
        assert transition_step(StepStatus.FAILED, StepStatus.RUNNING) is (
            StepStatus.RUNNING
        )

    def test_paused_step_resumes_to_running(self):
        assert transition_step(StepStatus.PAUSED, StepStatus.RUNNING) is (
            StepStatus.RUNNING
        )

    def test_completed_step_may_re_enter_on_resume(self):
        """LangGraph re-runs a node from the top when resuming after interrupt().

        A step that completed a side effect before pausing therefore re-enters
        legitimately, and the trace must be able to record that. Duplicate
        protection lives in the ledger, not here -- see TestDuplicateSideEffects.
        """
        assert transition_step(StepStatus.COMPLETED, StepStatus.RUNNING) is (
            StepStatus.RUNNING
        )

    def test_skipped_step_is_terminal(self):
        with pytest.raises(InvalidStateTransition):
            transition_step(StepStatus.SKIPPED, StepStatus.RUNNING)
