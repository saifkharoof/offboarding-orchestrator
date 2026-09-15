"""Error taxonomy.

The split that matters is :class:`TransientToolError` vs
:class:`PermanentToolError`. The tool executor retries the first with backoff
and refuses to retry the second; everything downstream (trace rows, run
failure reason, CLI exit codes) keys off that distinction rather than off
error messages.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for every error this package raises deliberately."""


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------


class InvalidStateTransition(OrchestratorError):
    """A run or step was asked to move to a status it cannot reach."""


class RunNotFound(OrchestratorError):
    """No run exists with the given id."""


class InvalidRunOperation(OrchestratorError):
    """The requested operation is not valid for the run's current status.

    Raised for, among others, approving a run that is not awaiting approval and
    resuming a run that has already reached a terminal status. Duplicate
    approval and duplicate resume requests both land here, which is what makes
    them safe to retry from a client.
    """


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class ToolError(OrchestratorError):
    """Base class for failures raised by a tool call."""


class TransientToolError(ToolError):
    """A failure the executor should retry with backoff.

    Network blips, rate limits, 5xx responses from a downstream system.
    """


class PermanentToolError(ToolError):
    """A failure retrying cannot fix. Fails the step and the run immediately.

    Bad input, a 4xx response, a validation failure on an LLM plan.
    """


class ToolNotAllowed(PermanentToolError):
    """A step attempted a tool outside its declared allowlist.

    A guardrail breach rather than an ordinary failure, so it is permanent by
    construction: retrying would breach the guardrail again.
    """


class BudgetExceeded(PermanentToolError):
    """The run hit its maximum step or tool-call budget."""


class InvalidPlan(PermanentToolError):
    """The LLM's checklist failed schema or allowlist validation.

    Covers an unparseable response, an empty plan, or one that names a system
    the employee does not have or omits one they do. The run service maps this
    to ``FailureReason.INVALID_PLAN`` specifically, rather than the generic
    ``TOOL_FAILED``, since the fix here is a prompt or model change, not a
    retry.
    """


class SideEffectReconciliationRequired(OrchestratorError):
    """A side effect may or may not have landed, and we cannot determine which.

    Raised when the ledger finds an ``in_progress`` reservation from a previous
    process and the provider cannot confirm the outcome. The run fails closed
    with ``FailureReason.NEEDS_RECONCILIATION`` rather than risk firing the
    side effect a second time.
    """


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------


class RunCancelled(OrchestratorError):
    """A cancellation was requested and the guard stopped the run.

    Control flow rather than failure: the run service catches it and records
    ``RunStatus.CANCELLED``. Raised by the guard *before* a step does any work,
    so cancelling never interrupts a side effect mid-flight.
    """


class ApprovalRejected(OrchestratorError):
    """A human rejected the action at the approval gate.

    Also control flow: the run service records ``RunStatus.FAILED`` with
    ``FailureReason.APPROVAL_REJECTED``. The high-risk tool is never reached.
    """
