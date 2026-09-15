"""Graph wiring.

The workflow is a straight line, and that is a deliberate choice rather than a
limitation. Branching lives in exactly two places -- the approval gate, which
raises rather than routing, and the guard, which stops a run before a step --
so the execution order you read here is the execution order you get.

What LangGraph provides, and what it does not
---------------------------------------------

LangGraph is used for two things: durable execution (the checkpointer records
where execution is, so a resume continues from the right node) and ``interrupt``
/ ``Command(resume=...)`` for pausing. That is all.

It is *not* the state model, the lifecycle, the trace, the retry policy or the
idempotency guard. Those are ours, in ``domain/``, ``persistence/`` and
``tools/``, because they are the parts an operator needs to reason about at
2am -- and because the checkpointer knows where execution stopped but has no
opinion about whether a side effect already landed.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from offboarding.domain.run import StepName
from offboarding.persistence.db import connect
from offboarding.orchestration import nodes
from offboarding.orchestration.services import Services
from offboarding.orchestration.state import OffboardingState

#: The workflow, in order. Adding a step is an entry here plus a node function;
#: the wiring below does not need to change shape.
WORKFLOW: tuple[tuple[str, object], ...] = (
    (StepName.FETCH_EMPLOYEE, nodes.fetch_employee),
    (StepName.PLAN_DEPROVISIONING, nodes.plan_deprovisioning),
    (StepName.AWAIT_HR_APPROVAL, nodes.await_hr_approval),
    (StepName.REVOKE_ACCESS, nodes.revoke_access),
    (StepName.SEND_EXIT_PAPERWORK, nodes.send_exit_paperwork),
    (StepName.AWAIT_SIGNED_DOCUMENT, nodes.await_signed_document),
    (StepName.FINALIZE, nodes.finalize),
)


def build_checkpointer(db_path: str) -> SqliteSaver:
    """Create the checkpointer over its *own* connection to the same file.

    Same file, separate connection, and both parts matter.

    Same file, because one file should be the whole durable state of the
    system: copy it and you have moved every in-flight run, its business state
    and its execution position together.

    Separate connection, because the checkpointer manages its own transactions
    around node execution. Sharing one connection would leave its transaction
    open while a node tried to open ours, and SQLite does not nest. WAL mode
    plus a busy timeout is what lets the two connections write to one file.
    """
    saver = SqliteSaver(connect(db_path))
    saver.setup()
    return saver


def build_graph(services: Services):
    """Compile the offboarding workflow.

    Nodes are bound to ``services`` here rather than reading globals, so a test
    can compile the same graph against a temporary database and a fault-injected
    IAM tool.
    """
    builder = StateGraph(OffboardingState)

    for step_name, fn in WORKFLOW:
        builder.add_node(step_name, partial(fn, services=services))

    previous = START
    for step_name, _ in WORKFLOW:
        builder.add_edge(previous, step_name)
        previous = step_name
    builder.add_edge(previous, END)

    return builder.compile(checkpointer=build_checkpointer(services.db_path))
