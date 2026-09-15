"""Trace helper for steps that do not go through the tool executor.

The executor traces its own attempts. The LLM step and the two pause gates do
not call registry tools, so they use :func:`step_span` to get the same
one-row-per-attempt treatment. Without it those steps would be invisible in the
trace, and "which step is this run on?" would have a gap in the middle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from offboarding.domain.redaction import redact
from offboarding.persistence.repositories import TraceRepository


class Span:
    """Handle for an open step attempt."""

    def __init__(self, attempt_id: int) -> None:
        self.attempt_id = attempt_id
        self._detail: dict[str, Any] = {}

    def set_detail(self, **fields: Any) -> None:
        """Attach fields to the trace row. Redacted before they are written."""
        self._detail.update(fields)

    @property
    def detail(self) -> dict[str, Any]:
        return self._detail


@contextmanager
def step_span(
    trace: TraceRepository,
    run_id: str,
    step_name: str,
    *,
    tool_invoked: str | None = None,
) -> Iterator[Span]:
    """Open a step attempt, closing it as completed or failed on exit.

    An exception closes the row with the exception's class name, so a failure
    is recorded even though the exception continues to propagate.
    """
    row = trace.start_attempt(run_id, step_name, tool_invoked=tool_invoked)
    span = Span(row.id)
    try:
        yield span
    except Exception as exc:
        trace.fail_attempt(
            row.id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            detail=redact(span.detail) or None,
        )
        raise
    trace.complete_attempt(row.id, detail=redact(span.detail) or None)
