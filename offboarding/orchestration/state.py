"""The graph's state schema.

Kept deliberately small. State holds raw results the later steps need, not
formatted text and not anything the ``runs`` table already owns: run status,
pause reason and counters live in SQLite, because an operator asking "what is
this run doing?" must be answerable without loading a graph.

Only ``events`` has a reducer. Every other field is written by exactly one node,
so last-write-wins is the correct semantics and an accidental overwrite would be
a bug we want to surface rather than merge away.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class OffboardingState(TypedDict, total=False):
    """Shared state for one offboarding run."""

    #: Identifies the run in our tables. Set once, at start.
    run_id: str
    employee_id: str

    #: Step 1 -- the HR record.
    employee: dict[str, Any]

    #: Step 2 -- the validated deprovisioning plan.
    checklist: list[dict[str, Any]]
    plan_notes: str

    #: Step 3 -- the approval decision recorded at the gate.
    approval: dict[str, Any]

    #: Steps 4, 5, 7 -- receipts from the side-effecting tools. Present means
    #: the effect is known to have landed.
    revocation: dict[str, Any]
    paperwork: dict[str, Any]
    archive: dict[str, Any]

    #: Step 6 -- the external event that released the second pause.
    signed_document: dict[str, Any]

    #: Human-readable progress notes, appended by each node. The authoritative
    #: record is the step_trace table; this is a convenience for streaming and
    #: for reading final state in one place.
    events: Annotated[list[str], operator.add]
