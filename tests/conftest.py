"""Shared fixtures.

``db_path`` is a real file rather than ``:memory:`` on purpose: the restart
recovery tests need to close every connection and reopen the database from
disk, which an in-memory database cannot do.
"""

from __future__ import annotations

import pytest

from offboarding.persistence.db import connect, initialize
from offboarding.persistence.runs import RunRepository
from offboarding.persistence.side_effects import SideEffectRepository
from offboarding.persistence.trace import TraceRepository


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def conn(db_path):
    connection = connect(db_path)
    initialize(connection)
    yield connection
    connection.close()


@pytest.fixture
def runs(conn):
    return RunRepository(conn)


@pytest.fixture
def trace(conn):
    return TraceRepository(conn)


@pytest.fixture
def ledger_store(conn):
    return SideEffectRepository(conn)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@pytest.fixture
def ledger(ledger_store):
    from offboarding.tools.ledger import SideEffectLedger

    return SideEffectLedger(ledger_store)


@pytest.fixture
def build_tools(conn):
    """Build a registry with configurable fault injection on the IAM tool."""
    from offboarding.tools.base import ToolRegistry
    from offboarding.tools.documents import (
        ArchiveRecordTool,
        SendExitPaperworkTool,
    )
    from offboarding.tools.hr_directory import HrDirectoryTool
    from offboarding.tools.iam import IamTool

    def _build(**iam_kwargs):
        return ToolRegistry(
            [
                HrDirectoryTool(),
                IamTool(conn, **iam_kwargs),
                SendExitPaperworkTool(conn),
                ArchiveRecordTool(conn),
            ]
        )

    return _build


@pytest.fixture
def build_executor(runs, trace, ledger, build_tools):
    """Build an executor whose backoff does not actually sleep."""
    from offboarding.tools.executor import RetryPolicy, ToolExecutor

    slept: list[float] = []

    def _build(*, policy=None, **iam_kwargs):
        return ToolExecutor(
            registry=build_tools(**iam_kwargs),
            ledger=ledger,
            runs=runs,
            trace=trace,
            policy=policy or RetryPolicy(max_attempts=3, initial_backoff=0.01),
            sleep=slept.append,
        )

    _build.slept = slept  # type: ignore[attr-defined]
    return _build


@pytest.fixture
def started_run(runs):
    """A run already moved into RUNNING, ready for steps to execute."""
    from offboarding.domain.run import RunStatus

    run = runs.create("emp-001")
    return runs.set_status(run.run_id, RunStatus.RUNNING)
