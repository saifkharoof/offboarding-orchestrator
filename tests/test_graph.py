"""The graph itself: linear execution, two pauses, two resumes.

These drive the compiled graph directly rather than through the run service,
which does not exist yet. They establish that the orchestration mechanics work
before a service layer is wrapped around them.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from offboarding.domain.errors import ApprovalRejected, RunCancelled
from offboarding.domain.run import PauseReason, RunStatus, StepName, StepStatus
from offboarding.orchestration.graph import build_graph
from offboarding.orchestration.services import build_services
from offboarding.tools.providers import ProviderStore

APPROVAL = {"approved": True, "approver": "priya.raman", "note": "Confirmed."}
SIGNATURE = {"document_id": "doc_abc", "signed_at": "2026-09-20T10:00:00Z"}


@pytest.fixture
def services(db_path):
    return build_services(db_path=db_path)


@pytest.fixture
def graph(services):
    return build_graph(services)


@pytest.fixture
def pending_run(services):
    run = services.runs.create("emp-001")
    return services.runs.set_status(run.run_id, RunStatus.RUNNING)


def config_for(run):
    return {"configurable": {"thread_id": run.thread_id}}


def start(graph, run):
    return graph.invoke(
        {"run_id": run.run_id, "employee_id": run.employee_id, "events": []},
        config_for(run),
    )


def interrupt_payload(result):
    return result["__interrupt__"][0].value


class TestPauses:
    def test_run_pauses_at_the_approval_gate(self, graph, pending_run):
        result = start(graph, pending_run)
        payload = interrupt_payload(result)

        assert payload["pause_reason"] == PauseReason.HR_APPROVAL.value
        assert payload["employee"]["name"] == "Dana Okafor"
        assert len(payload["checklist"]) == 5

    def test_nothing_is_revoked_before_approval(
        self, graph, pending_run, services
    ):
        """The gate sits ahead of the high-risk action, not behind it."""
        start(graph, pending_run)
        assert ProviderStore(services.conn, "iam").count() == 0
        assert services.side_effects.list_for_run(pending_run.run_id) == []

    def test_run_pauses_again_for_the_external_event(
        self, graph, pending_run
    ):
        start(graph, pending_run)
        result = graph.invoke(
            Command(resume=APPROVAL), config_for(pending_run)
        )
        payload = interrupt_payload(result)

        assert payload["pause_reason"] == PauseReason.SIGNED_DOCUMENT.value
        assert payload["document_id"].startswith("doc_")

    def test_side_effects_land_between_the_two_pauses(
        self, graph, pending_run, services
    ):
        start(graph, pending_run)
        graph.invoke(Command(resume=APPROVAL), config_for(pending_run))

        assert ProviderStore(services.conn, "iam").count() == 1
        assert ProviderStore(services.conn, "documents").count() == 1
        # Archiving has not happened yet: it is behind the second pause.
        assert ProviderStore(services.conn, "archive").count() == 0


class TestCompletion:
    def test_full_run_completes(self, graph, pending_run, services):
        start(graph, pending_run)
        graph.invoke(Command(resume=APPROVAL), config_for(pending_run))
        final = graph.invoke(Command(resume=SIGNATURE), config_for(pending_run))

        assert "__interrupt__" not in final
        assert final["archive"]["archive_id"].startswith("arc_")
        assert final["revocation"]["revoked_systems"] == [
            "github",
            "aws",
            "slack",
            "jira",
            "vpn",
        ]
        assert final["signed_document"] == SIGNATURE

    def test_every_step_appears_in_the_trace(
        self, graph, pending_run, services
    ):
        start(graph, pending_run)
        graph.invoke(Command(resume=APPROVAL), config_for(pending_run))
        graph.invoke(Command(resume=SIGNATURE), config_for(pending_run))

        traced = {r.step_name for r in services.trace.list_for_run(
            pending_run.run_id
        )}
        assert traced == {s.value for s in StepName}

    def test_each_side_effect_has_exactly_one_ledger_row(
        self, graph, pending_run, services
    ):
        start(graph, pending_run)
        graph.invoke(Command(resume=APPROVAL), config_for(pending_run))
        graph.invoke(Command(resume=SIGNATURE), config_for(pending_run))

        rows = services.side_effects.list_for_run(pending_run.run_id)
        assert {r.step_name for r in rows} == {
            StepName.REVOKE_ACCESS,
            StepName.SEND_EXIT_PAPERWORK,
            StepName.FINALIZE,
        }
        assert all(r.state == "completed" for r in rows)


class TestRejection:
    def test_rejection_stops_before_the_high_risk_tool(
        self, graph, pending_run, services
    ):
        start(graph, pending_run)
        with pytest.raises(ApprovalRejected):
            graph.invoke(
                Command(
                    resume={
                        "approved": False,
                        "approver": "priya.raman",
                        "note": "Last day moved.",
                    }
                ),
                config_for(pending_run),
            )
        assert ProviderStore(services.conn, "iam").count() == 0


class TestGuard:
    def test_cancellation_stops_the_run_between_steps(
        self, graph, pending_run, services
    ):
        start(graph, pending_run)
        services.runs.request_cancel(pending_run.run_id)

        with pytest.raises(RunCancelled):
            graph.invoke(Command(resume=APPROVAL), config_for(pending_run))

        # The guard runs before the step, so nothing was revoked on the way out.
        assert ProviderStore(services.conn, "iam").count() == 0


class TestRetryInsideTheGraph:
    def test_transient_iam_failures_are_retried_and_traced(
        self, db_path
    ):
        services = build_services(db_path=db_path, iam_fail_times=2)
        graph = build_graph(services)
        run = services.runs.create("emp-001")
        services.runs.set_status(run.run_id, RunStatus.RUNNING)

        start(graph, run)
        graph.invoke(Command(resume=APPROVAL), config_for(run))
        final = graph.invoke(Command(resume=SIGNATURE), config_for(run))

        assert final["archive"]["archive_id"].startswith("arc_")
        revoke_rows = [
            r
            for r in services.trace.list_for_run(run.run_id)
            if r.step_name == StepName.REVOKE_ACCESS
        ]
        assert [r.status for r in revoke_rows] == [
            StepStatus.FAILED,
            StepStatus.FAILED,
            StepStatus.COMPLETED,
        ]
        assert ProviderStore(services.conn, "iam").count() == 1
