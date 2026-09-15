"""Guardrails, retries and duplicate side-effect protection."""

from __future__ import annotations

import pytest

from offboarding.domain.errors import (
    BudgetExceeded,
    PermanentToolError,
    ToolNotAllowed,
    TransientToolError,
)
from offboarding.domain.redaction import REDACTED, redact
from offboarding.domain.run import StepName, StepStatus
from offboarding.persistence.repositories import SideEffectRepository
from offboarding.tools.executor import RetryPolicy
from offboarding.tools.iam import SimulatedProcessDeath
from offboarding.tools.providers import ProviderStore

READ_ONLY = frozenset({"hr_directory.get_employee"})
REVOKE_ONLY = frozenset({"iam.revoke_access"})


def revoke(executor, run_id, **kwargs):
    """Invoke the revocation step the way a node would."""
    return executor.execute(
        run_id=run_id,
        step_name=StepName.REVOKE_ACCESS,
        tool_name="iam.revoke_access",
        allowlist=REVOKE_ONLY,
        operation="revoke_all_systems",
        kwargs={
            "employee_id": "emp-001",
            "systems": ["github", "aws"],
            **kwargs,
        },
    )


class TestAllowlist:
    def test_tool_outside_the_allowlist_is_refused(
        self, build_executor, started_run
    ):
        executor = build_executor()
        with pytest.raises(ToolNotAllowed):
            executor.execute(
                run_id=started_run.run_id,
                step_name=StepName.FETCH_EMPLOYEE,
                tool_name="iam.revoke_access",
                allowlist=READ_ONLY,  # does not include the IAM tool
                operation="revoke_all_systems",
                kwargs={},
            )

    def test_a_refusal_is_visible_in_the_trace(
        self, build_executor, started_run, trace
    ):
        executor = build_executor()
        with pytest.raises(ToolNotAllowed):
            executor.execute(
                run_id=started_run.run_id,
                step_name=StepName.FETCH_EMPLOYEE,
                tool_name="iam.revoke_access",
                allowlist=READ_ONLY,
                operation="revoke_all_systems",
                kwargs={},
            )
        rows = trace.list_for_run(started_run.run_id)
        assert rows[-1].status is StepStatus.FAILED
        assert rows[-1].error_type == "ToolNotAllowed"
        assert rows[-1].detail["guardrail"] is True

    def test_unregistered_tool_is_refused_even_if_allowlisted(
        self, build_executor, started_run
    ):
        executor = build_executor()
        with pytest.raises(ToolNotAllowed):
            executor.execute(
                run_id=started_run.run_id,
                step_name=StepName.FETCH_EMPLOYEE,
                tool_name="iam.delete_everything",
                allowlist=frozenset({"iam.delete_everything"}),
                operation="boom",
                kwargs={},
            )


class TestReadTool:
    def test_read_tool_succeeds_on_the_first_attempt(
        self, build_executor, started_run, trace
    ):
        executor = build_executor()
        outcome = executor.execute(
            run_id=started_run.run_id,
            step_name=StepName.FETCH_EMPLOYEE,
            tool_name="hr_directory.get_employee",
            allowlist=READ_ONLY,
            kwargs={"employee_id": "emp-001"},
        )
        assert outcome.attempts == 1
        assert outcome.result["name"] == "Dana Okafor"
        assert trace.list_for_run(started_run.run_id)[0].status is (
            StepStatus.COMPLETED
        )

    def test_unknown_employee_is_permanent_not_transient(
        self, build_executor, started_run, trace
    ):
        executor = build_executor()
        with pytest.raises(PermanentToolError):
            executor.execute(
                run_id=started_run.run_id,
                step_name=StepName.FETCH_EMPLOYEE,
                tool_name="hr_directory.get_employee",
                allowlist=READ_ONLY,
                kwargs={"employee_id": "emp-nope"},
            )
        # One attempt only: a permanent error must not be retried.
        assert len(trace.list_for_run(started_run.run_id)) == 1


class TestRetries:
    def test_two_transient_failures_then_success(
        self, build_executor, started_run, trace
    ):
        """The 'make a tool fail twice before succeeding' scenario."""
        executor = build_executor(fail_times=2)
        outcome = revoke(executor, started_run.run_id)

        assert outcome.attempts == 3
        assert outcome.result["revoked_systems"] == ["github", "aws"]

        rows = trace.list_for_run(started_run.run_id)
        assert [r.status for r in rows] == [
            StepStatus.FAILED,
            StepStatus.FAILED,
            StepStatus.COMPLETED,
        ]
        # The failures stay legible after the success.
        assert all(r.error_type == "TransientToolError" for r in rows[:2])
        assert all(r.detail["retryable"] is True for r in rows[:2])
        assert [r.attempt for r in rows] == [1, 2, 3]

    def test_backoff_grows_between_attempts(self, build_executor, started_run):
        executor = build_executor(
            fail_times=2,
            policy=RetryPolicy(
                max_attempts=3, initial_backoff=0.01, multiplier=2.0
            ),
        )
        revoke(executor, started_run.run_id)
        assert build_executor.slept == [0.01, 0.02]

    def test_exhausting_retries_raises_the_transient_error(
        self, build_executor, started_run, trace
    ):
        executor = build_executor(fail_times=99)
        with pytest.raises(TransientToolError):
            revoke(executor, started_run.run_id)

        rows = trace.list_for_run(started_run.run_id)
        assert len(rows) == 3
        assert all(r.status is StepStatus.FAILED for r in rows)

    def test_permanent_failure_is_not_retried(
        self, build_executor, started_run, trace
    ):
        executor = build_executor(fail_permanently=True)
        with pytest.raises(PermanentToolError):
            revoke(executor, started_run.run_id)

        rows = trace.list_for_run(started_run.run_id)
        assert len(rows) == 1
        assert rows[0].detail["retryable"] is False


class TestDuplicateSideEffects:
    def test_repeating_a_completed_effect_replays_it(
        self, build_executor, started_run, conn
    ):
        """Calling the same operation twice must revoke access only once."""
        executor = build_executor()
        first = revoke(executor, started_run.run_id)
        assert first.replayed is False

        # A fresh executor, as a restarted process would have.
        second = revoke(build_executor(), started_run.run_id)
        assert second.replayed is True
        assert second.result == first.result
        assert ProviderStore(conn, "iam").count() == 1

    def test_a_retry_reuses_one_idempotency_key(
        self, build_executor, started_run, ledger_store
    ):
        executor = build_executor(fail_times=2)
        revoke(executor, started_run.run_id)
        # Three attempts, one ledger row: the key excludes the attempt number.
        assert len(ledger_store.list_for_run(started_run.run_id)) == 1

    def test_distinct_operations_get_distinct_keys(
        self, build_executor, started_run, ledger_store
    ):
        executor = build_executor()
        revoke(executor, started_run.run_id)
        executor.execute(
            run_id=started_run.run_id,
            step_name=StepName.SEND_EXIT_PAPERWORK,
            tool_name="documents.send_exit_paperwork",
            allowlist=frozenset({"documents.send_exit_paperwork"}),
            operation="send_packet",
            kwargs={
                "employee_id": "emp-001",
                "email": "dana.okafor@example.com",
                "checklist": [],
            },
        )
        assert len(ledger_store.list_for_run(started_run.run_id)) == 2


class TestCrashRecovery:
    """The stretch scenario: the effect landed, the orchestrator never knew."""

    def test_effect_applied_then_process_dies_is_reconciled(
        self, build_executor, started_run, conn, ledger_store
    ):
        dying = build_executor(crash_after_apply=True)
        with pytest.raises(SimulatedProcessDeath):
            revoke(dying, started_run.run_id)

        # The provider applied it; our ledger only knows a call was in flight.
        assert ProviderStore(conn, "iam").count() == 1
        row = ledger_store.list_for_run(started_run.run_id)[0]
        assert row.state == SideEffectRepository.IN_PROGRESS

        # A restarted process asks the provider instead of guessing.
        recovered = revoke(build_executor(), started_run.run_id)
        assert recovered.reconciled is True
        assert recovered.replayed is True
        assert recovered.result["revocation_id"].startswith("rev_")
        # Still exactly one revocation: it was adopted, not repeated.
        assert ProviderStore(conn, "iam").count() == 1

    def test_death_before_the_call_permits_a_clean_retry(
        self, build_executor, started_run, ledger_store, conn
    ):
        """A reserved-but-never-called row means the effect never happened."""
        from offboarding.tools.ledger import build_key

        key = build_key(
            started_run.run_id, StepName.REVOKE_ACCESS, "revoke_all_systems"
        )
        ledger_store.try_reserve(
            key,
            run_id=started_run.run_id,
            step_name=StepName.REVOKE_ACCESS,
            tool_name="iam.revoke_access",
        )

        outcome = revoke(build_executor(), started_run.run_id)
        assert outcome.replayed is False
        assert ProviderStore(conn, "iam").count() == 1


class TestBudget:
    def test_exhausting_the_tool_budget_fails_the_call(
        self, build_executor, runs, trace
    ):
        run = runs.create("emp-001", max_tool_calls=2)
        from offboarding.domain.run import RunStatus

        runs.set_status(run.run_id, RunStatus.RUNNING)
        executor = build_executor()

        for _ in range(2):
            executor.execute(
                run_id=run.run_id,
                step_name=StepName.FETCH_EMPLOYEE,
                tool_name="hr_directory.get_employee",
                allowlist=READ_ONLY,
                kwargs={"employee_id": "emp-001"},
            )

        with pytest.raises(BudgetExceeded):
            executor.execute(
                run_id=run.run_id,
                step_name=StepName.PLAN_DEPROVISIONING,
                tool_name="hr_directory.get_employee",
                allowlist=READ_ONLY,
                kwargs={"employee_id": "emp-001"},
            )
        assert trace.list_for_run(run.run_id)[-1].error_type == "BudgetExceeded"

    def test_retries_count_against_the_budget(self, build_executor, runs):
        run = runs.create("emp-001", max_tool_calls=2)
        from offboarding.domain.run import RunStatus

        runs.set_status(run.run_id, RunStatus.RUNNING)
        executor = build_executor(fail_times=99)

        # Attempts 1 and 2 spend the budget; the third is refused by it.
        with pytest.raises(BudgetExceeded):
            revoke(executor, run.run_id)


class TestRedaction:
    def test_sensitive_keys_are_masked_recursively(self):
        payload = {
            "employee_id": "emp-001",
            "api_key": "sk-live-123",
            "nested": {"refresh_token": "abc", "safe": 1},
            "items": [{"password": "hunter2"}],
        }
        assert redact(payload) == {
            "employee_id": "emp-001",
            "api_key": REDACTED,
            "nested": {"refresh_token": REDACTED, "safe": 1},
            "items": [{"password": REDACTED}],
        }

    def test_traced_results_pass_through_redaction(
        self, build_executor, started_run, trace, monkeypatch
    ):
        from offboarding.tools import hr_directory

        monkeypatch.setitem(
            hr_directory.EMPLOYEE_DIRECTORY,
            "emp-001",
            {**hr_directory.EMPLOYEE_DIRECTORY["emp-001"], "vpn_token": "s3cr3t"},
        )
        executor = build_executor()
        executor.execute(
            run_id=started_run.run_id,
            step_name=StepName.FETCH_EMPLOYEE,
            tool_name="hr_directory.get_employee",
            allowlist=READ_ONLY,
            kwargs={"employee_id": "emp-001"},
        )
        detail = trace.list_for_run(started_run.run_id)[0].detail
        assert detail["result"]["vpn_token"] == REDACTED
