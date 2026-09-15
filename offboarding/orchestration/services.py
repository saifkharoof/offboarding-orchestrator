"""Dependency container for the graph.

Nodes take their collaborators from a :class:`Services` instance rather than
reaching for module-level singletons. That is what lets a test build a graph
against a temporary database with a two-failure IAM tool, and what lets the CLI
and the API share one wiring function.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from offboarding.llm.factory import build_llm
from offboarding.llm.provider import LLMProvider
from offboarding.persistence.db import connect, initialize, resolve_db_path
from offboarding.persistence.runs import RunRepository
from offboarding.persistence.side_effects import SideEffectRepository
from offboarding.persistence.trace import TraceRepository
from offboarding.tools.base import ToolRegistry
from offboarding.tools.documents import ArchiveRecordTool, SendExitPaperworkTool
from offboarding.tools.executor import RetryPolicy, ToolExecutor
from offboarding.tools.hr_directory import HrDirectoryTool
from offboarding.tools.iam import IamTool
from offboarding.tools.ledger import SideEffectLedger


@dataclass(frozen=True)
class Services:
    """Everything a node may use."""

    conn: sqlite3.Connection
    #: Where the database lives, so the checkpointer can open its own
    #: connection to the same file. See graph.build_checkpointer.
    db_path: str
    runs: RunRepository
    trace: TraceRepository
    side_effects: SideEffectRepository
    executor: ToolExecutor
    llm: LLMProvider


def build_services(
    *,
    db_path: str | None = None,
    conn: sqlite3.Connection | None = None,
    llm: LLMProvider | None = None,
    policy: RetryPolicy | None = None,
    iam_fail_times: int = 0,
    iam_fail_permanently: bool = False,
) -> Services:
    """Wire the application together against one database.

    Args:
        iam_fail_times: Fault injection for the demo and the interview's
            "make a tool fail twice" request. Also settable without touching
            code via ``OFFBOARDING_IAM_FAIL_TIMES``.
    """
    import os

    resolved_path = resolve_db_path(db_path)
    connection = conn or connect(resolved_path)
    initialize(connection)

    fail_times = iam_fail_times or int(
        os.environ.get("OFFBOARDING_IAM_FAIL_TIMES", "0")
    )

    runs = RunRepository(connection)
    trace = TraceRepository(connection)
    side_effects = SideEffectRepository(connection)

    registry = ToolRegistry(
        [
            HrDirectoryTool(),
            IamTool(
                connection,
                fail_times=fail_times,
                fail_permanently=iam_fail_permanently,
            ),
            SendExitPaperworkTool(connection),
            ArchiveRecordTool(connection),
        ]
    )

    return Services(
        conn=connection,
        db_path=resolved_path,
        runs=runs,
        trace=trace,
        side_effects=side_effects,
        executor=ToolExecutor(
            registry=registry,
            ledger=SideEffectLedger(side_effects),
            runs=runs,
            trace=trace,
            policy=policy,
        ),
        llm=llm or build_llm(),
    )
